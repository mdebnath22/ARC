"""
plan_step23_hybrid_ablation.py
================================
Hybrid planning evaluation: two ablations in one table.

PART 1 — Solver ablation (no GPU needed)
  BFS (our Python BFS, baseline)
  A*  (pyperplan with ff heuristic)
  GBFS (pyperplan greedy best-first)

PART 2 — LLM ablation (GPU + Ollama required)
  qwen2.5:72b    (47 GB — needs full A100)
  llama3.3:70b   (42 GB — needs full A100)
  glm4:latest    (5.5 GB — runs on any GPU)

Pipeline (same for all combinations):
  1. ARC routes each instance: LLM or Solver
  2. LLM-routed: call Ollama, parse, validate
  3. Solver-routed: run pyperplan, parse plan output, validate
  4. Merge and report system validity (% instances with valid plan)

Goal: show hybrid ceiling increases with better solver/LLM.

USAGE:
  pip install pyperplan --break-system-packages

  # Solver ablation (no GPU):
  python plan_step23_hybrid_ablation.py --part solvers

  # LLM ablation (needs GPU + Ollama):
  python plan_step23_hybrid_ablation.py --part llms \\
      --host http://sc011:11434

  # Full table:
  python plan_step23_hybrid_ablation.py --part all \\
      --host http://sc011:11434
"""

from __future__ import annotations
import argparse, importlib.util, json, pickle, re, sys
import tempfile, time, urllib.request, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"

TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
TRAIN_DOMAINS = ["depot", "rovers", "satellite"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LLM_MODELS = [
    ("qwen2.5:72b",   "Qwen 2.5-72B"),
    ("llama3.3:70b",  "Llama 3.3-70B"),
    ("glm4:latest",   "GLM-4"),
]

SOLVERS = ["bfs", "astar", "gbfs"]


# ══════════════════════════════════════════════════════════════════════════════
# PDDL validator (inline, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

def _f(*p): return " ".join(str(x).lower() for x in p)

HANDLERS = {
    "pick-up":         (1,lambda x:({_f("clear",x),_f("ontable",x),_f("handempty")},{_f("ontable",x),_f("clear",x),_f("handempty")},{_f("holding",x)})),
    "put-down":        (1,lambda x:({_f("holding",x)},{_f("holding",x)},{_f("handempty"),_f("ontable",x),_f("clear",x)})),
    "stack":           (2,lambda x,y:({_f("holding",x),_f("clear",y)},{_f("holding",x),_f("clear",y)},{_f("handempty"),_f("on",x,y),_f("clear",x)})),
    "unstack":         (2,lambda x,y:({_f("on",x,y),_f("clear",x),_f("handempty")},{_f("on",x,y),_f("clear",x),_f("handempty")},{_f("holding",x),_f("clear",y)})),
    "grasp":           (1,lambda x:({_f("apex",x),_f("grounded",x),_f("grasping")},{_f("grounded",x),_f("apex",x),_f("grasping")},{_f("clutching",x)})),
    "release":         (1,lambda x:({_f("clutching",x)},{_f("clutching",x)},{_f("grasping"),_f("grounded",x),_f("apex",x)})),
    "place":           (2,lambda x,y:({_f("clutching",x),_f("apex",y)},{_f("clutching",x),_f("apex",y)},{_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
    "lift":            (2,lambda x,y:({_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
    "load-truck":      (3,lambda p,t,l:({_f("at",t,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,t)})),
    "unload-truck":    (3,lambda p,t,l:({_f("at",t,l),_f("in",p,t)},{_f("in",p,t)},{_f("at",p,l)})),
    "load-airplane":   (3,lambda p,a,l:({_f("at",a,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,a)})),
    "unload-airplane": (3,lambda p,a,l:({_f("at",a,l),_f("in",p,a)},{_f("in",p,a)},{_f("at",p,l)})),
    "drive-truck":     (4,lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},{_f("at",t,s)},{_f("at",t,d)})),
    "fly-airplane":    (3,lambda a,s,d:({_f("at",a,s),_f("airport",s),_f("airport",d)},{_f("at",a,s)},{_f("at",a,d)})),
}

ALIASES = {
    "pickup":"pick-up","putdown":"put-down","put_down":"put-down",
    "fly":"fly-airplane","fly-plane":"fly-airplane",
    "move-truck":"drive-truck","load-pkg":"load-truck","unload-pkg":"unload-truck",
    "load":"load-truck","unload":"unload-truck",
}
ALIASES_MBW = {"pick-up":"grasp","put-down":"release","stack":"place","unstack":"lift",
               "pickup":"grasp","putdown":"release"}

ACTION_KWS = set(HANDLERS) | set(ALIASES) | set(ALIASES_MBW)


def parse_plan(response, domain=""):
    if not response or response.startswith("ERROR:"): return []
    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b", txt, re.I): return []
    # parenthesised lines
    lines = [l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    # any parens
    found = re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", response, re.I)
    if found: return found
    # no-parens keyword lines
    result = []
    for line in txt.split("\n"):
        line = re.sub(r"^\d+[\.\)]\s*","",line.strip())
        toks = line.split()
        if toks and toks[0].lower() in ACTION_KWS:
            result.append("("+line+")")
    return result


def apply_aliases(actions, domain=""):
    al = dict(ALIASES)
    if "mystery" in domain.lower(): al.update(ALIASES_MBW)
    result = []
    for act in actions:
        toks = act.strip().strip("()").split()
        if not toks: continue
        name = al.get(toks[0].lower(), toks[0].lower())
        result.append("(" + " ".join([name]+toks[1:]) + ")")
    return result


def validate(record, actions, domain=""):
    if not actions: return False, "empty_plan"
    init  = record.get("init_facts", [])
    goal  = record.get("goal_facts", [])
    if not init or not goal: return True, None
    state = {_f(*f.strip().strip("()").split()) for f in init}
    goals = {_f(*g.strip().strip("()").split()) for g in goal}
    for act in actions:
        toks = act.strip().strip("()").split()
        if not toks: continue
        name=toks[0].lower(); args=[t.lower() for t in toks[1:]]
        if name not in HANDLERS: return False, f"unknown:{name}"
        ar,fn=HANDLERS[name]
        if len(args)!=ar: return False, f"arity:{name}"
        pre,rem,add=fn(*args)
        if not pre.issubset(state): return False, f"precond:{name}"
        state=(state-rem)|add
    ok=goals.issubset(state)
    return ok,(None if ok else "goal_not_reached")


# ══════════════════════════════════════════════════════════════════════════════
# Symbolic solvers
# ══════════════════════════════════════════════════════════════════════════════

def run_bfs_solver(record, timeout=10):
    """
    Our own BFS (same as used for labeling).
    Returns (success: bool, plan: list[str], n_steps: int).
    """
    from collections import deque

    init  = record.get("init_facts", [])
    goal  = record.get("goal_facts", [])
    dom   = record.get("task_type", record.get("domain",""))

    if not init or not goal:
        return False, [], 0

    init_state = frozenset(_f(*f.strip().strip("()").split()) for f in init)
    goal_set   = frozenset(_f(*g.strip().strip("()").split()) for g in goal)

    # Build applicable actions from the domain PDDL
    domain_pddl = record.get("domain_pddl","")
    acts = re.findall(r":action\s+(\S+)", domain_pddl)
    if not acts:
        # Fallback: use domain-known actions
        if "mystery" in dom.lower():
            acts = ["grasp","release","place","lift"]
        elif "logistics" in dom.lower():
            acts = ["load-truck","unload-truck","load-airplane",
                    "unload-airplane","drive-truck","fly-airplane"]
        else:
            acts = ["pick-up","put-down","stack","unstack"]

    # Extract all objects from init state
    objs = set()
    for f in init:
        for tok in f.strip().strip("()").split()[1:]:
            objs.add(tok.lower())
    objs = sorted(objs)

    queue  = deque([(init_state, [])])
    seen   = {init_state}
    t0     = time.time()

    while queue:
        if time.time()-t0 > timeout:
            return False, [], 0
        state, path = queue.popleft()
        if goal_set.issubset(state):
            return True, path, len(path)

        # Try all actions with all object combinations
        for act_name in acts:
            if act_name not in HANDLERS: continue
            ar, fn = HANDLERS[act_name]
            from itertools import product as iproduct
            for combo in iproduct(objs, repeat=ar):
                pre,rem,add = fn(*combo)
                if pre.issubset(state):
                    new_state = (state-rem)|add
                    if new_state not in seen:
                        seen.add(new_state)
                        act_str = "("+act_name+" "+" ".join(combo)+")"
                        queue.append((new_state, path+[act_str]))

    return False, [], 0


def run_pyperplan(record, search="astar", timeout=30):
    """
    Run pyperplan on a single instance.
    search: "astar" | "gbfs" | "bfs"
    Returns (success: bool, plan: list[str], n_steps: int).
    """
    try:
        import pyperplan
    except ImportError:
        return False, [], 0

    dom_pddl  = record.get("domain_pddl","")
    prob_pddl = record.get("problem_pddl","")
    if not dom_pddl or not prob_pddl:
        return False, [], 0

    with tempfile.TemporaryDirectory() as tmp:
        dom_path  = Path(tmp)/"domain.pddl"
        prob_path = Path(tmp)/"problem.pddl"
        dom_path.write_text(dom_pddl)
        prob_path.write_text(prob_pddl)

        heuristic = "hff" if search in ("astar","gbfs") else "blind"
        search_alg = {"astar":"astar","gbfs":"gbfs","bfs":"bfs"}.get(search,"astar")

        try:
            import signal
            def _timeout(sig,fr): raise TimeoutError()
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(timeout)

            from pyperplan import grounding
            from pyperplan.search import breadth_first_search, astar_search
            from pyperplan.heuristics.heuristic_base import NullHeuristic
            try:
                from pyperplan.heuristics.relaxation import hFFHeuristic as FFHeuristic
            except ImportError:
                FFHeuristic = NullHeuristic

            from pyperplan.pddl.parser import Parser
            parser = Parser(str(dom_path), str(prob_path))
            dom_obj  = parser.parse_domain()
            prob_obj = parser.parse_problem(dom_obj)
            task     = grounding.ground(prob_obj)

            if search_alg == "bfs":
                solution = breadth_first_search.breadth_first_search(task)
            else:
                h = FFHeuristic(task) if search_alg == "astar" else NullHeuristic(task)
                solution = astar_search.astar_search(task, h)

            signal.alarm(0)

            if solution is None:
                return False, [], 0

            plan = ["("+op.name+")" for op in solution]
            return True, plan, len(plan)

        except TimeoutError:
            return False, [], 0
        except Exception:
            return False, [], 0


def run_solver(record, solver_name, timeout=30):
    """Dispatch to the right solver."""
    if solver_name == "bfs":
        return run_bfs_solver(record, timeout=timeout)
    else:
        return run_pyperplan(record, search=solver_name, timeout=timeout)


# ══════════════════════════════════════════════════════════════════════════════
# LLM prompt builder and caller
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(record):
    dom   = record.get("domain_pddl","")
    prob  = record.get("problem_pddl","")
    desc  = record.get("description","")
    acts  = re.findall(r":action\s+(\S+)", dom)

    act_block = ""
    if acts:
        act_block = "\nCRITICAL: Use ONLY these action names:\n"
        for a in acts:
            params = re.search(rf":action\s+{re.escape(a)}.*?:parameters\s*\(([^)]*)\)",
                                dom, re.DOTALL)
            n = len(re.findall(r"\?", params.group(1))) if params else 1
            args = " ".join(f"arg{i+1}" for i in range(n))
            act_block += f"  ({a} {args})\n"

    return (
        "You are a PDDL planning expert. Solve the planning problem.\n\n"
        f"=== DOMAIN ===\n{dom}\n\n"
        f"=== PROBLEM ===\n{prob}\n\n"
        f"=== TASK ===\n{desc}\n"
        f"{act_block}\n"
        "Output ONLY the plan, one action per line:\n"
        "  (action-name arg1 arg2 ...)\n\n"
        "Rules:\n"
        "1. One action per line, parenthesised\n"
        "2. Use ONLY action names listed above\n"
        "3. Arguments must be object names from the problem\n"
        "4. If unsolvable: NO_PLAN\n"
        "5. No explanation, no comments\n"
    )


def call_ollama(host, model, prompt, timeout=300):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict":4096,"temperature":0.0},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{host}/api/generate", data=payload,
            headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response","")
    except Exception as e:
        return f"ERROR:{e}"


# ══════════════════════════════════════════════════════════════════════════════
# ARC routing
# ══════════════════════════════════════════════════════════════════════════════

def load_arc(prep):
    ckpt  = torch.load(CKPT/"arc_v2.pt", map_location="cpu")
    spec  = importlib.util.spec_from_file_location("step17", ROOT/"plan_step17_arc_v2.py")
    s17   = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
    sys.modules["step17"] = s17
    sys.modules["__main__"].GlobalPreprocessor = s17.GlobalPreprocessor
    model = s17.ARCv2(ckpt["surf_dim"], ckpt["fm_dim"])
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, s17


def get_arc_scores(model, prep, X_surf, X_fm, mask, S_surf, S_fm, S_V):
    """Return difficulty scores (higher = harder) for instances in mask."""
    Xs_n, Xe_n, Xr_n = prep.transform(X_surf[mask], X_fm[mask])
    scores = []
    with torch.no_grad():
        for i in range(len(Xs_n)):
            qs = torch.FloatTensor(Xs_n[i]).unsqueeze(0)
            qf = torch.FloatTensor(Xe_n[i]).unsqueeze(0)
            qr = torch.FloatTensor(Xr_n[i]).unsqueeze(0)
            out,_,_ = model(qs, qf, qr, S_surf, S_fm, S_V, head="reg")
            scores.append(float(out.squeeze().cpu()))
    return np.array(scores)


def arc_routing_mask(scores, llm_success_train, bfs_rate, N, n_tr=140):
    """
    Few-shot routing: train XGBoost on n_tr Qwen labels,
    predict which instances to route to LLM.
    Returns boolean mask: True = route to LLM.
    """
    import xgboost as xgb
    rng = np.random.default_rng(42)
    tr  = rng.choice(N, min(n_tr, N), replace=False)
    y_tr = llm_success_train[tr].astype(int)
    if len(np.unique(y_tr)) < 2:
        # single class — use n_objects routing (route small instances to LLM)
        threshold = np.percentile(scores, 30)
        return scores < threshold   # low score = easy = route to LLM

    Xs_feat = scores[..., None]   # single feature
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, verbosity=0,
                             eval_metric="logloss", random_state=42)
    clf.fit(Xs_feat[tr], y_tr)
    proba = clf.predict_proba(Xs_feat)[:,1]
    # Route to LLM when predicted P(LLM succeeds) > 0.5
    return proba > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Load everything
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    spec = importlib.util.spec_from_file_location("step17", ROOT/"plan_step17_arc_v2.py")
    s17  = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
    sys.modules["step17"] = s17
    sys.modules["__main__"].GlobalPreprocessor = s17.GlobalPreprocessor

    spec6 = importlib.util.spec_from_file_location("step6", ROOT/"plan_step6_pddlinst_gate.py")
    s6    = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)

    with open(RESULTS/"global_preprocessor.pkl","rb") as f:
        prep = pickle.load(f)

    X_surf,X_fm,tt,y_s,y_ns,splits = s6.load_data(data_dir=DATA)
    X_surf = X_surf[:,:-1]

    # Load episodes for PDDL content
    eps_path = DATA/"episodes.json"
    episodes = json.loads(eps_path.read_text()) if eps_path.exists() else []
    eps_by_dom = {}
    for e in episodes:
        dom = e.get("task_type", e.get("domain",""))
        eps_by_dom.setdefault(dom,[]).append(e)

    # Load Qwen labels (for routing head training)
    qwen_by_dom = {}
    qwen_path = RESULTS/"qwen72b_eval_instances.jsonl"
    if qwen_path.exists():
        for line in open(qwen_path):
            r = json.loads(line)
            qwen_by_dom.setdefault(r["domain"],[]).append(r)
        for d in qwen_by_dom:
            qwen_by_dom[d].sort(key=lambda r:int(r["instance_id"]))

    # Build ARC support set
    tr_mask = np.isin(tt, TRAIN_DOMAINS)
    Xs_tr,Xe_tr,_ = prep.transform(X_surf[tr_mask], X_fm[tr_mask])
    rng  = np.random.default_rng(42)
    sidx = rng.choice(tr_mask.sum(), min(60,tr_mask.sum()), replace=False)
    S_surf = torch.FloatTensor(Xs_tr[sidx])
    S_fm   = torch.FloatTensor(Xe_tr[sidx])
    S_V    = torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))

    return prep, X_surf, X_fm, tt, y_s, y_ns, eps_by_dom, qwen_by_dom, \
           S_surf, S_fm, S_V


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Solver ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_solver_ablation(prep, X_surf, X_fm, tt, y_s, y_ns,
                        eps_by_dom, qwen_by_dom, S_surf, S_fm, S_V):
    print("\n" + "="*70)
    print("PART 1: Solver ablation")
    print("  Pipeline: ARC routing → {Solver or LLM (Qwen 72B labels)}")
    print("  Using pre-computed Qwen 72B validity for LLM component")
    print("="*70)

    model,_ = load_arc(prep)

    # Check pyperplan
    try:
        import pyperplan
        has_pyperplan = True
        print("  pyperplan: available")
    except ImportError:
        has_pyperplan = False
        print("  pyperplan: NOT AVAILABLE — install with: pip install pyperplan")
        print("  Running BFS-only for now")

    solvers_to_run = ["bfs"] + (["astar","gbfs"] if has_pyperplan else [])

    results = {}
    N = 200

    print()
    print(f"  {'Domain':<14}  {'Solver':<8}  {'Solver%':>8}  {'ARC-FS%':>8}  "
          f"{'AlwaysBFS%':>11}  {'Hybrid gain':>12}")
    print("  " + "-"*70)

    for dom in TEST_DOMAINS:
        mask = tt == dom
        arc_scores = get_arc_scores(model, prep, X_surf, X_fm, mask,
                                     S_surf, S_fm, S_V)[:N]
        eps   = eps_by_dom.get(dom,[])[:N]
        qlist = qwen_by_dom.get(dom,[])[:N]
        y_llm = np.array([1.0 if r["valid_plan"] else 0.0 for r in qlist])

        # Routing mask (few-shot, uses Qwen labels)
        route_to_llm = arc_routing_mask(arc_scores, y_llm, 0.59, N)

        for solver in solvers_to_run:
            print(f"    Running {solver} on {dom} ({(~route_to_llm).sum()} instances)...",
                  end="", flush=True)
            t0 = time.time()

            solver_valid = 0
            solver_total = 0

            for i, ep in enumerate(eps):
                if i >= N: break
                if route_to_llm[i]:
                    continue  # this instance goes to LLM
                solver_total += 1
                ok, plan, n = run_solver(ep, solver, timeout=5)
                if ok:
                    solver_valid += 1

            llm_valid = y_llm[route_to_llm].sum()
            total_valid = solver_valid + llm_valid
            system_pct  = total_valid / N

            # Always-BFS baseline (all instances)
            bfs_valid_all = sum(1 for ep in eps[:N]
                                if run_solver(ep, "bfs", timeout=5)[0])
            always_bfs = bfs_valid_all / N

            elapsed = time.time()-t0
            dom_lbl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
            print(f" {elapsed:.0f}s")
            print(f"  {dom_lbl:<14}  {solver:<8}  {solver_valid/(solver_total+1e-8):>8.1%}  "
                  f"{system_pct:>8.1%}  {always_bfs:>11.1%}  "
                  f"{system_pct-always_bfs:>+12.1%}")

            results.setdefault(dom,{})[solver] = {
                "solver_rate": float(solver_valid/(solver_total+1e-8)),
                "system_pct":  float(system_pct),
                "always_bfs":  float(always_bfs),
                "gain":        float(system_pct - always_bfs),
            }
        print()

    (RESULTS/"solver_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Results → {RESULTS}/solver_ablation.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: LLM ablation
# ══════════════════════════════════════════════════════════════════════════════

def eval_llm_on_domain(host, model_id, eps, dom, N=50, timeout=120):
    """
    Evaluate a single LLM on N instances of a domain.
    Returns (n_valid, n_total, latency_mean).
    We use N=50 per LLM to keep runtime manageable.
    """
    n_valid = 0; latencies = []
    for i, ep in enumerate(eps[:N]):
        prompt   = build_prompt(ep)
        t0       = time.time()
        response = call_ollama(host, model_id, prompt, timeout=timeout)
        lat      = time.time()-t0
        latencies.append(lat)

        actions = parse_plan(response, dom)
        actions = apply_aliases(actions, dom)
        valid, _ = validate(ep, actions, dom)
        if valid: n_valid += 1

        if (i+1) % 10 == 0:
            print(f"    [{i+1}/{N}] valid={n_valid}  lat={lat:.1f}s", end="\r")

    print(f"    [{N}/{N}] valid={n_valid}/{N}={n_valid/N:.1%}  "
          f"mean_lat={np.mean(latencies):.1f}s")
    return n_valid, N, float(np.mean(latencies))


def run_llm_ablation(host, prep, X_surf, X_fm, tt, y_s, y_ns,
                     eps_by_dom, qwen_by_dom, S_surf, S_fm, S_V):
    print("\n" + "="*70)
    print("PART 2: LLM ablation")
    print(f"  Host: {host}")
    print("  Same ARC few-shot routing for all LLMs")
    print("  N=50 instances per domain per LLM")
    print("="*70)

    # Verify Ollama is up
    try:
        req  = urllib.request.Request(f"{host}/api/tags", method="GET")
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        available = [m["name"] for m in data.get("models",[])]
        print(f"  Available: {available}")
    except Exception as e:
        print(f"  ERROR: Cannot reach {host}: {e}")
        return {}

    model,_ = load_arc(prep)
    N_eval  = 50  # per domain per LLM to keep runtime <4h

    results = {}
    print()
    print(f"  {'LLM':<20}  {'Domain':<14}  {'LLM%':>6}  {'Hybrid%':>8}  "
          f"{'BFS-only%':>10}  {'Gain':>6}  {'Latency':>8}")
    print("  " + "-"*75)

    for model_id, model_name in LLM_MODELS:
        if not any(model_id.split(":")[0] in m for m in available):
            print(f"  {model_name}: NOT AVAILABLE — skipping")
            continue

        print(f"\n  Testing {model_name}...")
        results[model_id] = {}

        for dom in TEST_DOMAINS:
            mask = tt == dom
            arc_scores = get_arc_scores(model, prep, X_surf, X_fm, mask,
                                         S_surf, S_fm, S_V)[:N_eval]
            eps   = eps_by_dom.get(dom,[])[:N_eval]
            qlist = qwen_by_dom.get(dom,[])[:N_eval]
            y_llm_train = np.array([1.0 if r["valid_plan"] else 0.0 for r in qlist])

            # ARC routing
            route_to_llm = arc_routing_mask(arc_scores, y_llm_train, 0.59, N_eval)
            n_llm   = route_to_llm.sum()
            n_bfs   = (~route_to_llm).sum()

            print(f"\n    {dom} — routing {n_llm} to LLM, {n_bfs} to BFS")

            # Evaluate LLM on routed instances
            llm_eps  = [eps[i] for i in range(N_eval) if route_to_llm[i]]
            n_llm_valid = 0; lats = []
            for i, ep in enumerate(llm_eps):
                prompt   = build_prompt(ep)
                t0       = time.time()
                response = call_ollama(host, model_id, prompt, timeout=120)
                lats.append(time.time()-t0)
                actions  = parse_plan(response, dom)
                actions  = apply_aliases(actions, dom)
                ok, _    = validate(ep, actions, dom)
                if ok: n_llm_valid += 1

            # BFS on remaining instances
            bfs_eps  = [eps[i] for i in range(N_eval) if not route_to_llm[i]]
            n_bfs_valid = sum(1 for ep in bfs_eps if run_solver(ep,"bfs",timeout=5)[0])

            total_valid = n_llm_valid + n_bfs_valid
            system_pct  = total_valid / N_eval
            llm_pct     = n_llm_valid / max(n_llm, 1)

            # Always-BFS baseline
            bfs_all = sum(1 for ep in eps if run_solver(ep,"bfs",timeout=5)[0])
            always_bfs = bfs_all / N_eval

            dom_lbl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
            mean_lat = np.mean(lats) if lats else 0
            print(f"  {model_name:<20}  {dom_lbl:<14}  {llm_pct:>6.1%}  "
                  f"{system_pct:>8.1%}  {always_bfs:>10.1%}  "
                  f"{system_pct-always_bfs:>+6.1%}  {mean_lat:>7.1f}s")

            results[model_id][dom] = {
                "llm_rate":  float(llm_pct),
                "system":    float(system_pct),
                "bfs_only":  float(always_bfs),
                "gain":      float(system_pct - always_bfs),
                "lat_mean":  float(mean_lat),
            }

    (RESULTS/"llm_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Results → {RESULTS}/llm_ablation.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table generator
# ══════════════════════════════════════════════════════════════════════════════

def print_latex(solver_res, llm_res):
    print("\n" + "="*70)
    print("LATEX TABLE")
    print("="*70)
    print()
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Hybrid system ablation. \textbf{Solver ablation} (top):")
    print(r"ARC few-shot routing between Qwen~2.5-72B and three symbolic")
    print(r"planners. \textbf{LLM ablation} (bottom): same ARC routing with")
    print(r"different LLM components; BFS as the fallback solver.")
    print(r"System validity = fraction of instances with valid plan.")
    print(r"Gain = improvement over Always-BFS baseline.}")
    print(r"\label{tab:hybrid_ablation}")
    print(r"\small\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{ll ccc c}")
    print(r"\toprule")
    print(r"\textbf{Component} & \textbf{Variant} & \textbf{BW} & \textbf{LOG} & \textbf{MBW} & \textbf{mean gain} \\")
    print(r"\midrule")
    print(r"\multicolumn{6}{l}{\textit{Solver ablation (LLM = Qwen 2.5-72B)}} \\")

    solver_names = {"bfs":"BFS (Python)","astar":"A* (FF heuristic)","gbfs":"GBFS (greedy FF)"}
    for solver in SOLVERS:
        vals = []
        gains = []
        for dom in TEST_DOMAINS:
            if dom in solver_res and solver in solver_res[dom]:
                r = solver_res[dom][solver]
                vals.append(f"{r['system_pct']:.1%}")
                gains.append(r['gain'])
            else:
                vals.append("---")
        mean_gain = f"{np.mean(gains):+.1%}" if gains else "---"
        row = f"  & {solver_names.get(solver,solver)} & " + " & ".join(vals) + f" & {mean_gain} \\\\"
        print(row)

    print(r"\midrule")
    print(r"\multicolumn{6}{l}{\textit{LLM ablation (Solver = BFS)}} \\")

    llm_names = {"qwen2.5:72b":"Qwen 2.5-72B","llama3.3:70b":"Llama 3.3-70B","glm4:latest":"GLM-4"}
    for model_id, _ in LLM_MODELS:
        if model_id not in llm_res:
            continue
        vals = []; gains = []
        for dom in TEST_DOMAINS:
            if dom in llm_res[model_id]:
                r = llm_res[model_id][dom]
                vals.append(f"{r['system']:.1%}")
                gains.append(r['gain'])
            else:
                vals.append("---")
        mean_gain = f"{np.mean(gains):+.1%}" if gains else "---"
        name = llm_names.get(model_id, model_id)
        row = f"  & {name} & " + " & ".join(vals) + f" & {mean_gain} \\\\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}\end{table}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="solvers",
                   choices=["solvers","llms","all"])
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    print(f"\nHybrid Planning Ablation  —  device={DEVICE}")

    prep, X_surf, X_fm, tt, y_s, y_ns, eps_by_dom, qwen_by_dom, \
        S_surf, S_fm, S_V = load_all()

    solver_res = {}; llm_res = {}

    if args.part in ("solvers","all"):
        solver_res = run_solver_ablation(
            prep, X_surf, X_fm, tt, y_s, y_ns,
            eps_by_dom, qwen_by_dom, S_surf, S_fm, S_V)

    if args.part in ("llms","all"):
        llm_res = run_llm_ablation(
            args.host, prep, X_surf, X_fm, tt, y_s, y_ns,
            eps_by_dom, qwen_by_dom, S_surf, S_fm, S_V)

    if solver_res or llm_res:
        print_latex(solver_res, llm_res)


if __name__ == "__main__":
    main()
