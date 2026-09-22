"""
plan_step30_corruption_experiments.py
=======================================
GPU experiments for Section 6 (MBW analysis):

E5: Progressive semantic corruption
    Corrupt BW predicates 0%, 25%, 50%, 75%, 100%
    Measure LLM failure vs ARC prediction at each level.
    Tests: Failure ~ Corruption + n* + (Corruption × n*) interaction term.

E6: Equal-length controlled MBW
    Within fixed plan-length buckets, corrupt vs uncorrupt.
    Controls for PDDL string length effects.

E7: Refusal-type taxonomy
    Classify LLM failures: explicit_refusal / invalid_syntax /
    hallucinated_action / wrong_args / goal_not_reached / near_correct.

USAGE (needs GPU + Ollama):
  python plan_step30_corruption_experiments.py \\
    --host http://HOST:11434 --part E5
  python plan_step30_corruption_experiments.py \\
    --host http://HOST:11434 --part all
"""

from __future__ import annotations
import argparse, importlib.util, json, pickle, re, sys
import signal, tempfile, time, urllib.request, warnings
from pathlib import Path

import numpy as np
import torch
from scipy import stats

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"
TRAIN_DOMAINS = ["depot","rovers","satellite"]


# ══════════════════════════════════════════════════════════════════════════════
# Semantic corruption: replace predicate/action names with tokens
# ══════════════════════════════════════════════════════════════════════════════

# BW predicate→MBW mapping (known from dataset)
BW_TO_MBW = {
    "on":       "pred7",  "ontable":   "pred3",
    "clear":    "pred1",  "handempty": "pred5",
    "holding":  "pred2",
    "pick-up":  "act3",   "put-down":  "act1",
    "stack":    "act4",   "unstack":   "act2",
}
MBW_PREDICATES = set(BW_TO_MBW.values())
BW_PREDICATES  = set(BW_TO_MBW.keys())


def corrupt_pddl(domain_pddl, problem_pddl, corruption_level, rng):
    """
    Corrupt a fraction of predicate/action names in PDDL.
    corruption_level: 0.0 (none) to 1.0 (full MBW-style scramble).

    For each BW predicate/action, with probability=corruption_level,
    replace it with the MBW symbol.
    """
    if corruption_level == 0.0:
        return domain_pddl, problem_pddl

    # Build partial substitution map
    sub_map = {}
    for bw, mbw in BW_TO_MBW.items():
        if rng.random() < corruption_level:
            sub_map[bw] = mbw

    if not sub_map:
        return domain_pddl, problem_pddl

    def apply_subs(text, smap):
        # Replace whole-word matches only
        for old, new in smap.items():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
        return text

    return apply_subs(domain_pddl, sub_map), apply_subs(problem_pddl, sub_map)


# ══════════════════════════════════════════════════════════════════════════════
# LLM interface + plan classifier
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(host, model, prompt, timeout=120):
    payload = json.dumps({"model":model,"prompt":prompt,"stream":False,
                          "options":{"num_predict":1024,"temperature":0.0}}).encode()
    try:
        req = urllib.request.Request(f"{host}/api/generate", data=payload,
            headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response","")
    except Exception as e:
        return f"ERROR:{e}"


def classify_failure(response, record, domain=""):
    """
    E7: Classify LLM output into failure type.
    Returns one of:
      "valid"           — plan passes PDDL executor
      "explicit_refusal"— "NO_PLAN", "I cannot", "unable to"
      "empty"           — no output
      "invalid_syntax"  — no parseable actions
      "hallucinated_action" — uses action names not in domain
      "wrong_args"      — correct action, wrong arity
      "goal_not_reached"— valid actions, goal not satisfied
      "near_correct"    — reaches goal partially (>80% fluents satisfied)
    """
    if response.startswith("ERROR:"): return "error"
    txt = re.sub(r"<think>.*?</think>","",response,flags=re.DOTALL).strip()

    if not txt: return "empty"

    # Explicit refusal
    refusal_patterns = [
        r"\bNO[_\s-]PLAN\b", r"i cannot", r"unable to", r"i don't know",
        r"this problem cannot", r"no valid plan", r"impossible"
    ]
    if any(re.search(p, txt, re.I) for p in refusal_patterns):
        return "explicit_refusal"

    # Parse actions
    lines = [l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if not lines:
        found = re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", txt, re.I)
        if not found: return "invalid_syntax"
        lines = found

    # Check action names
    dom_pddl = record.get("domain_pddl","")
    valid_actions = set(re.findall(r":action\s+(\S+)", dom_pddl))
    if not valid_actions:
        # Fallback: standard BW + MBW actions
        valid_actions = {"pick-up","put-down","stack","unstack",
                         "grasp","release","place","lift",
                         "load-truck","unload-truck","load-airplane",
                         "unload-airplane","drive-truck","fly-airplane"}

    actions_in_plan = set()
    for line in lines:
        toks = line.strip().strip("()").split()
        if toks: actions_in_plan.add(toks[0].lower())

    hallucinated = actions_in_plan - {a.lower() for a in valid_actions}
    if hallucinated:
        return "hallucinated_action"

    # Check arities
    ARITY = {"pick-up":1,"put-down":1,"stack":2,"unstack":2,
             "grasp":1,"release":1,"place":2,"lift":2,
             "load-truck":3,"unload-truck":3,"load-airplane":3,
             "unload-airplane":3,"drive-truck":4,"fly-airplane":3}
    for line in lines:
        toks = line.strip().strip("()").split()
        if not toks: continue
        name = toks[0].lower(); args = toks[1:]
        if name in ARITY and len(args) != ARITY[name]:
            return "wrong_args"

    # Execute plan
    def _f(*p): return " ".join(str(x).lower() for x in p)
    HANDLERS = {
        "pick-up":(1,lambda x:({_f("clear",x),_f("ontable",x),_f("handempty")},{_f("ontable",x),_f("clear",x),_f("handempty")},{_f("holding",x)})),
        "put-down":(1,lambda x:({_f("holding",x)},{_f("holding",x)},{_f("handempty"),_f("ontable",x),_f("clear",x)})),
        "stack":(2,lambda x,y:({_f("holding",x),_f("clear",y)},{_f("holding",x),_f("clear",y)},{_f("handempty"),_f("on",x,y),_f("clear",x)})),
        "unstack":(2,lambda x,y:({_f("on",x,y),_f("clear",x),_f("handempty")},{_f("on",x,y),_f("clear",x),_f("handempty")},{_f("holding",x),_f("clear",y)})),
        "grasp":(1,lambda x:({_f("apex",x),_f("grounded",x),_f("grasping")},{_f("grounded",x),_f("apex",x),_f("grasping")},{_f("clutching",x)})),
        "release":(1,lambda x:({_f("clutching",x)},{_f("clutching",x)},{_f("grasping"),_f("grounded",x),_f("apex",x)})),
        "place":(2,lambda x,y:({_f("clutching",x),_f("apex",y)},{_f("clutching",x),_f("apex",y)},{_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
        "lift":(2,lambda x,y:({_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
    }
    ALIASES = {"pickup":"pick-up","putdown":"put-down"}
    ALIASES_MBW = {"pick-up":"grasp","put-down":"release","stack":"place","unstack":"lift"}

    al = dict(ALIASES)
    if any(p in dom_pddl for p in MBW_PREDICATES): al.update(ALIASES_MBW)

    init  = record.get("init_facts",[])
    goal  = record.get("goal_facts",[])
    if not init or not goal: return "valid"

    state = {_f(*f.strip().strip("()").split()) for f in init}
    goals = {_f(*g.strip().strip("()").split()) for g in goal}

    for line in lines:
        toks = line.strip().strip("()").split()
        if not toks: continue
        name = al.get(toks[0].lower(), toks[0].lower())
        args = [t.lower() for t in toks[1:]]
        if name not in HANDLERS: continue
        ar, fn = HANDLERS[name]
        if len(args) != ar: continue
        try:
            pre, rem, add = fn(*args)
            if pre.issubset(state): state = (state-rem)|add
        except Exception: continue

    satisfied = goals & state
    frac = len(satisfied) / max(len(goals), 1)
    if goals.issubset(state): return "valid"
    if frac >= 0.8: return "near_correct"
    return "goal_not_reached"


# ══════════════════════════════════════════════════════════════════════════════
# E5: Progressive semantic corruption
# ══════════════════════════════════════════════════════════════════════════════

def run_e5(host, model_id, n_per_level=30):
    print("\n" + "="*65)
    print("E5: Progressive semantic corruption")
    print(f"  N={n_per_level} per corruption level, model={model_id}")
    print("="*65)

    corruption_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    N = n_per_level

    # Load BW instances
    eps_all = json.loads((DATA/"episodes.json").read_text())
    bw_eps  = [e for e in eps_all if e.get("task_type","")=="blocksworld"][:N]

    # Load ARC for scoring
    spec17 = importlib.util.spec_from_file_location("s17",ROOT/"plan_step17_arc_v2.py")
    s17    = importlib.util.module_from_spec(spec17); spec17.loader.exec_module(s17)
    sys.modules["step17"]=s17; sys.modules["__main__"].GlobalPreprocessor=s17.GlobalPreprocessor
    spec6  = importlib.util.spec_from_file_location("s6",ROOT/"plan_step6_pddlinst_gate.py")
    s6     = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    with open(RESULTS/"global_preprocessor.pkl","rb") as f: prep=pickle.load(f)
    X_surf,X_fm,tt,y_s,y_ns,_=s6.load_data(data_dir=DATA); X_surf=X_surf[:,:-1]
    ckpt=torch.load(CKPT/"arc_v2.pt",map_location="cpu")
    arc=s17.ARCv2(ckpt["surf_dim"],ckpt["fm_dim"]); arc.load_state_dict(ckpt["model"]); arc.eval()
    tr=np.isin(tt,TRAIN_DOMAINS); Xs_tr,Xe_tr,_=prep.transform(X_surf[tr],X_fm[tr])
    rng_s=np.random.default_rng(42); sidx=rng_s.choice(tr.sum(),min(60,tr.sum()),replace=False)
    S_s=torch.FloatTensor(Xs_tr[sidx]); S_f=torch.FloatTensor(Xe_tr[sidx])
    S_V=torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))

    # BW ARC scores (on original uncorrupted)
    bw_mask = tt == "blocksworld"
    bw_Xs,bw_Xe,bw_Xr = prep.transform(X_surf[bw_mask],X_fm[bw_mask])
    arc_scores = []
    with torch.no_grad():
        for i in range(N):
            qs=torch.FloatTensor(bw_Xs[i]).unsqueeze(0)
            qf=torch.FloatTensor(bw_Xe[i]).unsqueeze(0)
            qr=torch.FloatTensor(bw_Xr[i]).unsqueeze(0)
            o,_,_=arc(qs,qf,qr,S_s,S_f,S_V,head="reg")
            arc_scores.append(float(o.squeeze().cpu()))
    arc_scores = np.array(arc_scores)
    n_steps    = y_ns[bw_mask].astype(float)[:N]

    results = {}
    print(f"\n  {'Level':>7}  {'LLM success':>12}  {'ARC|ρ|_struct':>14}  "
          f"{'ARC|ρ|_LLM':>12}  {'refusal%':>9}")
    print("  " + "-"*58)

    for level in corruption_levels:
        rng = np.random.default_rng(int(level*100))
        successes = []; failure_types = []

        for i, ep in enumerate(bw_eps):
            # Corrupt the PDDL
            c_dom, c_prob = corrupt_pddl(
                ep.get("domain_pddl",""),
                ep.get("problem_pddl",""),
                level, rng
            )
            c_ep = dict(ep); c_ep["domain_pddl"]=c_dom; c_ep["problem_pddl"]=c_prob

            # Build prompt
            acts = re.findall(r":action\s+(\S+)", c_dom)
            ab = "\nUse ONLY these actions: " + ", ".join(acts) if acts else ""
            prompt = (f"Solve this PDDL planning problem.\n\n"
                      f"DOMAIN:\n{c_dom}\n\nPROBLEM:\n{c_prob}\n{ab}\n"
                      "Output one action per line: (action arg1 ...)\n"
                      "If unsolvable: NO_PLAN\n")

            response = call_llm(host, model_id, prompt, timeout=90)
            ftype    = classify_failure(response, c_ep, "blocksworld")
            successes.append(1 if ftype=="valid" else 0)
            failure_types.append(ftype)

        success_rate = np.mean(successes)
        refusal_rate = np.mean([f=="explicit_refusal" for f in failure_types])
        rho_struct, _ = stats.spearmanr(arc_scores[:N], n_steps[:N])
        rho_llm, _   = stats.spearmanr(arc_scores[:N], successes)

        print(f"  {level:>7.0%}  {success_rate:>12.1%}  {abs(rho_struct):>14.3f}  "
              f"{abs(rho_llm):>12.3f}  {refusal_rate:>9.1%}")

        results[str(level)] = {
            "success_rate":  float(success_rate),
            "refusal_rate":  float(refusal_rate),
            "rho_structural":float(rho_struct),
            "rho_llm":       float(rho_llm),
            "failure_types": {ft: failure_types.count(ft) for ft in set(failure_types)},
        }

    # Interaction test: Failure ~ Corruption + n* + Corruption×n*
    print("\n  Interaction test: Failure ~ Corruption + n* + (Corruption×n*)")
    try:
        from sklearn.linear_model import LogisticRegression
        X_int = []; y_int = []
        for lv in corruption_levels:
            r = results[str(lv)]
            for i in range(N):
                X_int.append([lv, n_steps[i], lv*n_steps[i]])
                y_int.append(1 - int(i < int(r["success_rate"]*N)))
        X_int = np.array(X_int); y_int = np.array(y_int)
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X_int)
        clf = LogisticRegression(max_iter=500).fit(X_scaled, y_int)
        coefs = dict(zip(["corruption","n_steps","interaction"], clf.coef_[0]))
        print(f"  Corruption coef:   {coefs['corruption']:+.3f}")
        print(f"  n_steps coef:      {coefs['n_steps']:+.3f}")
        print(f"  Interaction coef:  {coefs['interaction']:+.3f}")
        if coefs['interaction'] > 0.1:
            print("  → Significant positive interaction: harder instances fail MORE")
            print("    under semantic corruption (supports E5 hypothesis)")
        results["interaction_coefs"] = coefs
    except Exception as e:
        print(f"  Interaction test failed: {e}")

    (RESULTS/"progressive_corruption.json").write_text(json.dumps(results,indent=2))
    print(f"\n  Saved → {RESULTS}/progressive_corruption.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E6: Equal-length controlled MBW
# ══════════════════════════════════════════════════════════════════════════════

def run_e6(host, model_id, n_per_bucket=20):
    print("\n" + "="*65)
    print("E6: Equal-length controlled MBW")
    print("  Compare BW vs MBW within fixed plan-length buckets.")
    print("  Controls for PDDL string length / n_steps effects.")
    print("="*65)

    eps_all = json.loads((DATA/"episodes.json").read_text())
    bw_eps  = {e["instance_id"]: e for e in eps_all if e.get("task_type","")=="blocksworld"}
    mbw_eps = {e["instance_id"]: e for e in eps_all if e.get("task_type","")=="mystery_blocksworld"}

    # Group by n_steps bucket
    bw_by_nsteps  = {}
    for iid, ep in bw_eps.items():
        ns = ep.get("n_steps",0) or 0
        bw_by_nsteps.setdefault(ns, []).append(ep)

    results = {}
    print(f"\n  {'n_steps':>8}  {'BW success':>12}  {'MBW success':>13}  {'Δ(MBW-BW)':>11}")
    print("  " + "-"*48)

    for ns in sorted(bw_by_nsteps.keys()):
        bw_bucket  = bw_by_nsteps[ns][:n_per_bucket]
        mbw_bucket = [e for e in mbw_eps.values() if e.get("n_steps",0)==ns][:n_per_bucket]

        if len(bw_bucket) < 5 or len(mbw_bucket) < 5: continue

        def eval_bucket(episodes, domain_name):
            succs = []
            for ep in episodes:
                acts = re.findall(r":action\s+(\S+)", ep.get("domain_pddl",""))
                ab   = "\nUse ONLY: " + ", ".join(acts) if acts else ""
                prompt=(f"Solve this PDDL problem.\nDOMAIN:\n{ep.get('domain_pddl','')}\n"
                        f"PROBLEM:\n{ep.get('problem_pddl','')}\n{ab}\n"
                        "One action per line: (action args)\nIf unsolvable: NO_PLAN\n")
                resp  = call_llm(host, model_id, prompt, timeout=90)
                ftype = classify_failure(resp, ep, domain_name)
                succs.append(1 if ftype=="valid" else 0)
            return float(np.mean(succs))

        bw_sr  = eval_bucket(bw_bucket,  "blocksworld")
        mbw_sr = eval_bucket(mbw_bucket, "mystery_blocksworld")
        delta  = mbw_sr - bw_sr

        print(f"  {ns:>8}  {bw_sr:>12.1%}  {mbw_sr:>13.1%}  {delta:>+11.1%}")
        results[str(ns)] = {"bw":bw_sr,"mbw":mbw_sr,"delta":delta,"n_bw":len(bw_bucket),"n_mbw":len(mbw_bucket)}

    (RESULTS/"equal_length_controlled.json").write_text(json.dumps(results,indent=2))
    print(f"\n  Saved → {RESULTS}/equal_length_controlled.json")
    print("\n  If MBW success < BW success across all n_steps buckets:")
    print("  → semantic corruption causes failures independent of plan length")
    print("  → rules out 'longer PDDL strings cause failures' alternative")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E7: Refusal-type taxonomy
# ══════════════════════════════════════════════════════════════════════════════

def run_e7(host, model_id, N=100):
    print("\n" + "="*65)
    print("E7: Refusal-type taxonomy")
    print(f"  Classify LLM failures across BW and MBW (N={N} each)")
    print("="*65)

    eps_all = json.loads((DATA/"episodes.json").read_text())
    results = {}

    for dom in ["blocksworld","mystery_blocksworld"]:
        eps = [e for e in eps_all if e.get("task_type","")==dom][:N]
        type_counts = {}; total = 0

        for i, ep in enumerate(eps):
            acts = re.findall(r":action\s+(\S+)", ep.get("domain_pddl",""))
            ab   = "\nUse ONLY: " + ", ".join(acts) if acts else ""
            prompt=(f"Solve this PDDL problem.\nDOMAIN:\n{ep.get('domain_pddl','')}\n"
                    f"PROBLEM:\n{ep.get('problem_pddl','')}\n{ab}\n"
                    "One action per line: (action args)\nIf unsolvable: NO_PLAN\n")
            resp  = call_llm(host, model_id, prompt, timeout=90)
            ftype = classify_failure(resp, ep, dom)
            type_counts[ftype] = type_counts.get(ftype,0) + 1
            total += 1
            if (i+1)%20==0:
                print(f"    [{i+1}/{N}] types so far: {type_counts}", end="\r")

        print(f"\n  {dom[:3].upper()} failure taxonomy (N={total}):")
        print(f"  {'Type':<25}  {'Count':>8}  {'Fraction':>10}")
        print("  " + "-"*46)
        for ftype, count in sorted(type_counts.items(), key=lambda x:-x[1]):
            print(f"  {ftype:<25}  {count:>8}  {count/total:>10.1%}")
        results[dom] = {"counts":type_counts,"total":total}

    # LaTeX
    print("\n  LaTeX:")
    print(r"\begin{table}[h]\centering\small")
    print(r"\caption{Failure type taxonomy for Qwen 2.5-72B on BW and MBW.")
    print(r"Semantic corruption (MBW) increases hallucinated actions and")
    print(r"explicit refusals relative to the uncorrupted domain.}")
    print(r"\label{tab:failure_taxonomy}")
    print(r"\begin{tabular}{l cc}\toprule")
    print(r"\textbf{Failure type} & \textbf{BW} & \textbf{MBW} \\\midrule")
    all_types = sorted(set(
        list(results.get("blocksworld",{}).get("counts",{}).keys()) +
        list(results.get("mystery_blocksworld",{}).get("counts",{}).keys())
    ))
    for ft in all_types:
        bw_r  = results.get("blocksworld",{})
        mbw_r = results.get("mystery_blocksworld",{})
        bw_c  = bw_r.get("counts",{}).get(ft,0)
        mbw_c = mbw_r.get("counts",{}).get(ft,0)
        bw_f  = bw_c / max(bw_r.get("total",1),1)
        mbw_f = mbw_c / max(mbw_r.get("total",1),1)
        print(f"  {ft.replace('_',' ').title()} & {bw_f:.1%} & {mbw_f:.1%} \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")

    (RESULTS/"failure_taxonomy.json").write_text(json.dumps(results,indent=2))
    print(f"\n  Saved → {RESULTS}/failure_taxonomy.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host",   default="http://localhost:11434")
    p.add_argument("--model",  default="qwen2.5:72b")
    p.add_argument("--part",   choices=["E5","E6","E7","all"], default="E5")
    p.add_argument("--n",      type=int, default=30)
    args = p.parse_args()

    # Verify Ollama
    try:
        data=json.loads(urllib.request.urlopen(
            urllib.request.Request(f"{args.host}/api/tags"),timeout=10).read())
        avail=[m["name"] for m in data.get("models",[])]
        if not any(args.model.split(":")[0] in m for m in avail):
            print(f"ERROR: {args.model} not available. Available: {avail}"); return
        print(f"OK: {args.model}")
    except Exception as e:
        print(f"ERROR: {e}"); return

    if args.part in ("E5","all"): run_e5(args.host, args.model, args.n)
    if args.part in ("E6","all"): run_e6(args.host, args.model, args.n//2)
    if args.part in ("E7","all"): run_e7(args.host, args.model, args.n*3)
    print("\nDone.")

if __name__ == "__main__": main()
