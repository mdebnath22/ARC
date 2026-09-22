"""
plan_step23_v2.py  —  Complete hybrid planning ablation
=========================================================
Datasets: our generated instances + PlanBench (if cloned)
Solvers:  all pyperplan search x heuristic combinations
Stats:    solved%, nodes_expanded, h_init, h_mean, wall_time
LLMs:     qwen2.5:72b, llama3.3:70b, glm4:latest

SETUP:
  pip install pyperplan --break-system-packages
  cd data && git clone https://github.com/karthikv792/LLMs-Planning planbench

USAGE:
  python plan_step23_v2.py --part solvers          # no GPU
  python plan_step23_v2.py --part llms --host URL  # needs GPU
  python plan_step23_v2.py --part all  --host URL
"""
from __future__ import annotations
import argparse, importlib.util, json, pickle, re, sys, signal, time
import tempfile, urllib.request, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
PLANBENCH = ROOT / "data" / "planbench"
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"
TEST_DOMAINS  = ["blocksworld","logistics","mystery_blocksworld"]
TRAIN_DOMAINS = ["depot","rovers","satellite"]

LLM_MODELS = [
    ("qwen2.5:72b",  "Qwen 2.5-72B"),
    ("llama3.3:70b", "Llama 3.3-70B"),
    ("glm4:latest",  "GLM-4"),
]

SOLVER_CONFIGS = [
    # (search,    heuristic,   label)
    ("bfs",       "blind",     "BFS"),
    ("astar",     "hFF",       "A*(hFF)"),
    ("astar",     "hAdd",      "A*(hAdd)"),
    ("astar",     "hMax",      "A*(hMax)"),
    ("astar",     "lm_cut",    "A*(LM-Cut)"),
    ("astar",     "landmarks", "A*(Landmarks)"),
    ("gbfs",      "hFF",       "GBFS(hFF)"),
    ("gbfs",      "hAdd",      "GBFS(hAdd)"),
    ("wastar_3",  "hFF",       "WA*(3,hFF)"),
    ("ehc",       "hFF",       "EHC(hFF)"),
]

# ── PDDL validator ────────────────────────────────────────────────────────────
def _f(*p): return " ".join(str(x).lower() for x in p)
HANDLERS = {
    "pick-up":   (1,lambda x:({_f("clear",x),_f("ontable",x),_f("handempty")},{_f("ontable",x),_f("clear",x),_f("handempty")},{_f("holding",x)})),
    "put-down":  (1,lambda x:({_f("holding",x)},{_f("holding",x)},{_f("handempty"),_f("ontable",x),_f("clear",x)})),
    "stack":     (2,lambda x,y:({_f("holding",x),_f("clear",y)},{_f("holding",x),_f("clear",y)},{_f("handempty"),_f("on",x,y),_f("clear",x)})),
    "unstack":   (2,lambda x,y:({_f("on",x,y),_f("clear",x),_f("handempty")},{_f("on",x,y),_f("clear",x),_f("handempty")},{_f("holding",x),_f("clear",y)})),
    "grasp":     (1,lambda x:({_f("apex",x),_f("grounded",x),_f("grasping")},{_f("grounded",x),_f("apex",x),_f("grasping")},{_f("clutching",x)})),
    "release":   (1,lambda x:({_f("clutching",x)},{_f("clutching",x)},{_f("grasping"),_f("grounded",x),_f("apex",x)})),
    "place":     (2,lambda x,y:({_f("clutching",x),_f("apex",y)},{_f("clutching",x),_f("apex",y)},{_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
    "lift":      (2,lambda x,y:({_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
    "load-truck":(3,lambda p,t,l:({_f("at",t,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,t)})),
    "unload-truck":(3,lambda p,t,l:({_f("at",t,l),_f("in",p,t)},{_f("in",p,t)},{_f("at",p,l)})),
    "load-airplane":(3,lambda p,a,l:({_f("at",a,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,a)})),
    "unload-airplane":(3,lambda p,a,l:({_f("at",a,l),_f("in",p,a)},{_f("in",p,a)},{_f("at",p,l)})),
    "drive-truck":(4,lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},{_f("at",t,s)},{_f("at",t,d)})),
    "fly-airplane":(3,lambda a,s,d:({_f("at",a,s),_f("airport",s),_f("airport",d)},{_f("at",a,s)},{_f("at",a,d)})),
}
ALIASES = {"pickup":"pick-up","putdown":"put-down","put_down":"put-down",
           "fly":"fly-airplane","fly-plane":"fly-airplane","move-truck":"drive-truck",
           "load-pkg":"load-truck","unload-pkg":"unload-truck",
           "load":"load-truck","unload":"unload-truck"}
ALIASES_MBW = {"pick-up":"grasp","put-down":"release","stack":"place",
               "unstack":"lift","pickup":"grasp","putdown":"release"}
ACTION_KWS = set(HANDLERS)|set(ALIASES)|set(ALIASES_MBW)

def parse_plan(response, domain=""):
    if not response or response.startswith("ERROR:"): return []
    txt = re.sub(r"<think>.*?</think>","",response,flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b",txt,re.I): return []
    lines=[l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    found=re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)",response,re.I)
    if found: return found
    result=[]
    for line in txt.split("\n"):
        line=re.sub(r"^\d+[\.\)]\s*","",line.strip()); toks=line.split()
        if toks and toks[0].lower() in ACTION_KWS: result.append("("+line+")")
    return result

def apply_aliases(actions, domain=""):
    al=dict(ALIASES)
    if "mystery" in domain.lower(): al.update(ALIASES_MBW)
    result=[]
    for act in actions:
        toks=act.strip().strip("()").split()
        if not toks: continue
        result.append("("+" ".join([al.get(toks[0].lower(),toks[0].lower())]+toks[1:])+")")
    return result

def validate(record, actions):
    if not actions: return False,"empty_plan"
    init=record.get("init_facts",[]); goal=record.get("goal_facts",[])
    if not init or not goal: return True,None
    state={_f(*f.strip().strip("()").split()) for f in init}
    goals={_f(*g.strip().strip("()").split()) for g in goal}
    for act in actions:
        toks=act.strip().strip("()").split()
        if not toks: continue
        name=toks[0].lower(); args=[t.lower() for t in toks[1:]]
        if name not in HANDLERS: return False,f"unknown:{name}"
        ar,fn=HANDLERS[name]
        if len(args)!=ar: return False,f"arity:{name}"
        pre,rem,add=fn(*args)
        if not pre.issubset(state): return False,f"precond:{name}"
        state=(state-rem)|add
    ok=goals.issubset(state)
    return ok,(None if ok else "goal_not_reached")

# ── Instrumented pyperplan runner ─────────────────────────────────────────────
def _timeout_handler(sig,frame): raise TimeoutError()

def run_solver(record, search_name="astar", heuristic_name="hFF", timeout=30):
    """
    Run pyperplan and return a stats dict:
      solved, plan_length, nodes_expanded, h_init, h_mean, h_min, wall_time, timeout
    """
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding
    from pyperplan.search.a_star import astar_search, greedy_best_first_search, weighted_astar_search
    from pyperplan.search.breadth_first_search import breadth_first_search
    from pyperplan.search.enforced_hillclimbing_search import enforced_hillclimbing_search
    from pyperplan.search.iterative_deepening_search import iterative_deepening_search
    from pyperplan.heuristics.relaxation import hFFHeuristic,hAddHeuristic,hMaxHeuristic,hSAHeuristic
    from pyperplan.heuristics.landmarks import LandmarkHeuristic
    from pyperplan.heuristics.lm_cut import LmCutHeuristic
    from pyperplan.heuristics.blind import BlindHeuristic

    HEURISTICS={"hFF":hFFHeuristic,"hAdd":hAddHeuristic,"hMax":hMaxHeuristic,
                "hSA":hSAHeuristic,"landmarks":LandmarkHeuristic,
                "lm_cut":LmCutHeuristic,"blind":BlindHeuristic}

    empty={"solved":False,"plan_length":0,"nodes_expanded":0,
           "h_init":None,"h_mean":None,"h_min":None,"wall_time":0.0,"timeout":False}

    dom_pddl=record.get("domain_pddl",""); prob_pddl=record.get("problem_pddl","")
    if not dom_pddl or not prob_pddl: return empty

    with tempfile.TemporaryDirectory() as tmp:
        dp=Path(tmp)/"domain.pddl"; pp=Path(tmp)/"problem.pddl"
        # Strip :typing for pyperplan compatibility
        import re as _re
        def _strip_typing(pddl):
            pddl = _re.sub(r'\s*:typing', '', pddl)
            pddl = _re.sub(r'\(:types[^)]*\)', '', pddl, flags=_re.DOTALL)
            pddl = _re.sub(r'(\?\w+)\s+-\s+\(either[^)]+\)', r'\1', pddl)
            pddl = _re.sub(r'(\?\w+)\s+-\s+\w+', r'\1', pddl)
            pddl = _re.sub(r'(\s+[\w\d_]+)\s+-\s+\w+', r'\1', pddl)
            return pddl
        _dom = _strip_typing(dom_pddl)
        prob_pddl2 = _strip_typing(prob_pddl)
        dp.write_text(_dom)
        pp.write_text(prob_pddl2)
        signal.signal(signal.SIGALRM,_timeout_handler)
        signal.alarm(timeout); t0=time.perf_counter()
        try:
            parser=Parser(str(dp),str(pp))
            task=grounding.ground(parser.parse_problem(parser.parse_domain()))
            HClass=HEURISTICS.get(heuristic_name,hFFHeuristic)
            base_h=HClass(task)
            h_vals=[]; h_calls=[0]
            def h(node):
                v=float(base_h(node)); h_vals.append(v)
                if h_calls[0]==0: h_calls[0]+=1; return v
                h_calls[0]+=1; return v

            if search_name=="bfs":         sol=breadth_first_search(task)
            elif search_name=="gbfs":      sol=greedy_best_first_search(task,h)
            elif search_name.startswith("wastar"):
                w=float(search_name.split("_")[1]) if "_" in search_name else 3.0
                sol=weighted_astar_search(task,h,weight=w)
            elif search_name=="ehc":       sol=enforced_hillclimbing_search(task,h)
            elif search_name=="ids":       sol=iterative_deepening_search(task,h)
            else:                          sol=astar_search(task,h)

            signal.alarm(0); wt=time.perf_counter()-t0
            if sol is None: return {**empty,"wall_time":wt,"nodes_expanded":len(h_vals)}
            # initial h: call heuristic on root
            from pyperplan.search.searchspace import make_root_node
            root=make_root_node(task.initial_state)
            h_init_val=float(base_h(root))
            return {"solved":True,"plan_length":len(sol),"nodes_expanded":len(h_vals),
                    "h_init":h_init_val,
                    "h_mean":float(np.mean(h_vals)) if h_vals else None,
                    "h_min":float(np.min(h_vals)) if h_vals else None,
                    "wall_time":wt,"timeout":False}
        except TimeoutError:
            signal.alarm(0)
            return {**empty,"timeout":True,"wall_time":time.perf_counter()-t0}
        except Exception:
            signal.alarm(0)
            return {**empty,"wall_time":time.perf_counter()-t0}

# ── PlanBench loader ──────────────────────────────────────────────────────────
def load_planbench():
    records=defaultdict(list)
    # Search for PlanBench location
    pb_root=None
    for candidate in [
    PLANBENCH/"llm_planning_analysis"/"instances",
    PLANBENCH/"LLMs-Planning"/"instances",
                       PLANBENCH/"instances", PLANBENCH]:
        if candidate.exists() and any(candidate.glob("blocksworld*")):
            pb_root=candidate; break
    if pb_root is None:
        print(f"  PlanBench not found. Clone with:")
        print(f"    cd {ROOT}/data && git clone https://github.com/karthikv792/LLMs-Planning planbench")
        return {}
    print(f"  PlanBench root: {pb_root}")
    dom_map={"blocksworld":"blocksworld","mystery_blocksworld":"mystery_blocksworld"}
    for pb_dom,our_dom in dom_map.items():
        dom_dir=pb_root/pb_dom
        if not dom_dir.exists(): continue
        dom_files=sorted(dom_dir.rglob("*domain*.pddl"))
        prob_files=sorted(dom_dir.rglob("instance-*.pddl"))
        if not dom_files: continue
        dom_pddl=dom_files[0].read_text()
        for i,pf in enumerate(prob_files[:200]):
            prob_pddl=pf.read_text()
            def extract_facts(pddl, keyword):
                m=re.search(rf"\(:{keyword}(.*?)\)(?=\s*\(:|$)",pddl,re.DOTALL|re.I)
                return re.findall(r"\([^()]+\)",m.group(1)) if m else []
            init=extract_facts(prob_pddl,"init")
            goal_m=re.search(r"\(:goal\s*\(and(.*?)\)\s*\)",prob_pddl,re.DOTALL|re.I)
            goal=re.findall(r"\([^()]+\)",goal_m.group(1)) if goal_m else []
            obj_m=re.search(r"\(:objects(.*?)\)",prob_pddl,re.DOTALL|re.I)
            n_obj=len(obj_m.group(1).split()) if obj_m else 0
            records[our_dom].append({"instance_id":i,"task_type":our_dom,"domain":our_dom,
                "domain_pddl":dom_pddl,"problem_pddl":prob_pddl,
                "init_facts":init,"goal_facts":goal,"n_objects":n_obj,
                "source":"planbench","filename":pf.name})
        print(f"  PlanBench {pb_dom}: {len(records[our_dom])} instances")
    return dict(records)

# ── Data loader ───────────────────────────────────────────────────────────────
def load_all():
    import torch
    spec=importlib.util.spec_from_file_location("step17",ROOT/"plan_step17_arc_v2.py")
    s17=importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
    sys.modules["step17"]=s17; sys.modules["__main__"].GlobalPreprocessor=s17.GlobalPreprocessor
    spec6=importlib.util.spec_from_file_location("step6",ROOT/"plan_step6_pddlinst_gate.py")
    s6=importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    with open(RESULTS/"global_preprocessor.pkl","rb") as f: prep=pickle.load(f)
    X_surf,X_fm,tt,y_s,y_ns,splits=s6.load_data(data_dir=DATA); X_surf=X_surf[:,:-1]
    ckpt=torch.load(CKPT/"arc_v2.pt",map_location="cpu")
    model=s17.ARCv2(ckpt["surf_dim"],ckpt["fm_dim"]); model.load_state_dict(ckpt["model"]); model.eval()
    tr=np.isin(tt,TRAIN_DOMAINS); Xs_tr,Xe_tr,_=prep.transform(X_surf[tr],X_fm[tr])
    rng=np.random.default_rng(42); sidx=rng.choice(tr.sum(),min(60,tr.sum()),replace=False)
    S_surf=torch.FloatTensor(Xs_tr[sidx]); S_fm=torch.FloatTensor(Xe_tr[sidx])
    S_V=torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))
    eps_by_dom=defaultdict(list)
    eps_path=DATA/"episodes.json"
    if eps_path.exists():
        for e in json.loads(eps_path.read_text()):
            eps_by_dom[e.get("task_type",e.get("domain",""))].append(e)
    qwen_by_dom=defaultdict(list)
    qp=RESULTS/"qwen72b_eval_instances.jsonl"
    if qp.exists():
        for line in open(qp):
            r=json.loads(line); qwen_by_dom[r["domain"]].append(r)
        for d in qwen_by_dom: qwen_by_dom[d].sort(key=lambda r:int(r["instance_id"]))
    # ── Inject domain_pddl into episodes (missing from episodes.json) ──
    for dom, eps_list in eps_by_dom.items():
        dom_pddl_path = DATA / dom / "domain.pddl"
        dom_pddl_text = dom_pddl_path.read_text() if dom_pddl_path.exists() else ""
        for ep in eps_list:
            ep.setdefault("domain_pddl", dom_pddl_text)
    return prep,X_surf,X_fm,tt,y_s,y_ns,model,s17,S_surf,S_fm,S_V,dict(eps_by_dom),dict(qwen_by_dom)

def get_arc_scores(model,prep,X_surf,X_fm,mask,S_surf,S_fm,S_V):
    import torch
    Xs_n,Xe_n,Xr_n=prep.transform(X_surf[mask],X_fm[mask])
    scores=[]
    with torch.no_grad():
        for i in range(len(Xs_n)):
            qs=torch.FloatTensor(Xs_n[i]).unsqueeze(0); qf=torch.FloatTensor(Xe_n[i]).unsqueeze(0)
            qr=torch.FloatTensor(Xr_n[i]).unsqueeze(0)
            out,_,_=model(qs,qf,qr,S_surf,S_fm,S_V,head="reg"); scores.append(float(out.squeeze().cpu()))
    return np.array(scores)

def arc_routing_mask(scores,y_llm,N,n_tr=140):
    import xgboost as xgb
    rng=np.random.default_rng(42); tr=rng.choice(N,min(n_tr,N),replace=False)
    y_tr=y_llm[tr].astype(int)
    if len(np.unique(y_tr))<2: return scores<np.percentile(scores,30)
    clf=xgb.XGBClassifier(n_estimators=50,max_depth=3,verbosity=0,eval_metric="logloss",random_state=42)
    clf.fit(scores[tr,None],y_tr); return clf.predict_proba(scores[:,None])[:,1]>0.5

# ── PART 1: Solver benchmarking ───────────────────────────────────────────────
def run_solver_ablation(eps_by_dom,planbench,N=50,timeout=10):
    print("\n"+"="*70)
    print(f"PART 1: Solver benchmarking  N={N}/domain timeout={timeout}s")
    print("="*70)
    all_results={}
    for ds_name,dataset in [("ours",eps_by_dom),("planbench",planbench)]:
        if not dataset: print(f"\n  {ds_name}: skipping"); continue
        print(f"\n  Dataset: {ds_name}")
        all_results[ds_name]={}
        for dom in [d for d in TEST_DOMAINS if d in dataset]:
            import random as _random
            _all = dataset[dom]
            _rng = _random.Random(42)
            eps = _rng.sample(_all, min(N, len(_all)))  # random sample
            print(f"\n  {dom} (N={len(eps)}):")
            print(f"  {'Solver':<18}  {'Solved%':>8}  {'Nodes(med)':>11}  {'h_init(med)':>12}  {'h_mean(med)':>12}  {'Time(s)':>8}")
            print("  "+"-"*75)
            dom_res={}
            for search,heuristic,label in SOLVER_CONFIGS:
                stats_list=[run_solver(ep,search,heuristic,timeout) for ep in eps]
                solved=[s for s in stats_list if s["solved"]]
                h_inits=[s["h_init"] for s in stats_list if s["h_init"] is not None]
                h_means=[s["h_mean"] for s in solved if s["h_mean"] is not None]
                rate=len(solved)/len(stats_list)
                med_n=np.median([s["nodes_expanded"] for s in stats_list])
                med_hi=float(np.median(h_inits)) if h_inits else float("nan")
                med_hm=float(np.median(h_means)) if h_means else float("nan")
                med_t=np.median([s["wall_time"] for s in stats_list])
                print(f"  {label:<18}  {rate:>8.1%}  {med_n:>11.0f}  {med_hi:>12.1f}  {med_hm:>12.1f}  {med_t:>8.3f}")
                dom_res[label]={"solve_rate":rate,"med_nodes":float(med_n),
                                "med_h_init":med_hi,"med_h_mean":med_hm,"med_time":float(med_t)}
            all_results[ds_name][dom]=dom_res
    out=RESULTS/"solver_ablation_v2.json"
    out.write_text(json.dumps(all_results,indent=2))
    print(f"\n  Saved → {out}")
    return all_results

# ── PART 2: LLM ablation ──────────────────────────────────────────────────────
def build_prompt(record):
    dom=record.get("domain_pddl",""); prob=record.get("problem_pddl","")
    acts=re.findall(r":action\s+(\S+)",dom)
    ab=""
    if acts:
        ab="\nCRITICAL — use ONLY these actions:\n"
        for a in acts:
            m=re.search(rf":action\s+{re.escape(a)}.*?:parameters\s*\(([^)]*)\)",dom,re.DOTALL)
            n=len(re.findall(r"\?",m.group(1))) if m else 1
            ab+=f"  ({a} {' '.join(f'arg{i+1}' for i in range(n))})\n"
    return (f"You are a PDDL planning expert.\n\n=== DOMAIN ===\n{dom}\n\n"
            f"=== PROBLEM ===\n{prob}\n\n{ab}\n"
            "Output ONLY the plan, one action per line: (action arg1 ...)\n"
            "If unsolvable: NO_PLAN\n")

def call_ollama(host,model,prompt,timeout=300):
    payload=json.dumps({"model":model,"prompt":prompt,"stream":False,
                        "options":{"num_predict":4096,"temperature":0.0}}).encode()
    try:
        req=urllib.request.Request(f"{host}/api/generate",data=payload,
            headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return json.loads(r.read()).get("response","")
    except Exception as e: return f"ERROR:{e}"

def run_llm_ablation(host,eps_by_dom,model,prep,X_surf,X_fm,tt,
                      qwen_by_dom,S_surf,S_fm,S_V,N=50,timeout=120):
    print("\n"+"="*70+"\nPART 2: LLM ablation\n"+"="*70)
    try:
        data=json.loads(urllib.request.urlopen(
            urllib.request.Request(f"{host}/api/tags"),timeout=10).read())
        avail=[m["name"] for m in data.get("models",[])]
        print(f"  Available: {avail}\n")
    except Exception as e:
        print(f"  ERROR: {e}"); return {}
    results={}
    print(f"  {'LLM':<20}  {'Dom':<8}  {'LLM%':>6}  {'Hybrid%':>8}  {'BFS%':>6}  {'Gain':>6}  {'Lat':>6}")
    print("  "+"-"*64)
    for model_id,model_name in LLM_MODELS:
        if not any(model_id.split(":")[0] in m for m in avail):
            print(f"  {model_name}: NOT AVAILABLE"); continue
        results[model_id]={}
        for dom in TEST_DOMAINS:
            eps=eps_by_dom.get(dom,[])[:N]; qlist=qwen_by_dom.get(dom,[])[:N]
            if not eps: continue
            mask=tt==dom
            scores=get_arc_scores(model,prep,X_surf,X_fm,mask,S_surf,S_fm,S_V)[:N]
            y_q=np.array([1.0 if r["valid_plan"] else 0.0 for r in qlist])
            route=arc_routing_mask(scores,y_q,N)
            llm_valid=0; lats=[]
            for i,ep in enumerate(eps):
                if not route[i]: continue
                t0=time.time(); resp=call_ollama(host,model_id,build_prompt(ep),timeout)
                lats.append(time.time()-t0)
                acts=apply_aliases(parse_plan(resp,dom),dom)
                ok,_=validate(ep,acts)
                if ok: llm_valid+=1
            bfs_valid=sum(1 for i,ep in enumerate(eps)
                          if not route[i] and run_solver(ep,"bfs","blind",5)["solved"])
            always_bfs=sum(1 for ep in eps if run_solver(ep,"bfs","blind",5)["solved"])/N
            system=(llm_valid+bfs_valid)/N; llm_r=llm_valid/max(route.sum(),1)
            gain=system-always_bfs; lat=float(np.mean(lats)) if lats else 0
            dl=dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
            print(f"  {model_name:<20}  {dl:<8}  {llm_r:>6.1%}  {system:>8.1%}  {always_bfs:>6.1%}  {gain:>+6.1%}  {lat:>6.1f}")
            results[model_id][dom]={"llm_rate":float(llm_r),"system":float(system),
                                     "bfs_only":float(always_bfs),"gain":float(gain),"lat":lat}
    (RESULTS/"llm_ablation_v2.json").write_text(json.dumps(results,indent=2))
    print(f"\n  Saved → {RESULTS}/llm_ablation_v2.json")
    return results

# ── LaTeX ─────────────────────────────────────────────────────────────────────
def make_latex(solver_res,llm_res):
    print("\n"+"="*70+"\nLaTeX TABLE\n"+"="*70)
    lines=[
        r"\begin{table}[t]\centering",
        r"\caption{Hybrid planning ablation. \textbf{Solver ablation} (top):",
        r"  10 solver configurations on our generated instances (N=50/domain).",
        r"  \emph{Nodes}: median node expansions. $h_0$: initial heuristic value",
        r"  (difficulty proxy). \textbf{LLM ablation} (bottom): ARC few-shot",
        r"  routing + BFS fallback, N=50/domain. Gain = system validity $-$ BFS.}",
        r"\label{tab:hybrid_ablation}",
        r"\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ll cccccc}",r"\toprule",
        r"\textbf{Component} & \textbf{Config} & \textbf{BW} & \textbf{LOG}",
        r"  & \textbf{MBW} & \textbf{Nodes} & $h_0$ & \textbf{Time(s)} \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Solver ablation}} \\",
    ]
    if "ours" in solver_res:
        for _,_,label in SOLVER_CONFIGS:
            row_vals=[]
            nodes_vals=[]; h0_vals=[]; t_vals=[]
            for dom in TEST_DOMAINS:
                v=solver_res["ours"].get(dom,{}).get(label,{})
                row_vals.append(f"{v.get('solve_rate',0):.0%}" if v else "---")
                if v:
                    nodes_vals.append(v.get("med_nodes",0))
                    if not np.isnan(v.get("med_h_init",float("nan"))): h0_vals.append(v["med_h_init"])
                    t_vals.append(v.get("med_time",0))
            n_s=f"{np.mean(nodes_vals):.0f}" if nodes_vals else "---"
            h0_s=f"{np.mean(h0_vals):.1f}" if h0_vals else "---"
            t_s=f"{np.mean(t_vals):.3f}" if t_vals else "---"
            lines.append(f"  & {label} & "+" & ".join(row_vals)+f" & {n_s} & {h0_s} & {t_s} \\\\")
    lines+=[r"\midrule",r"\multicolumn{8}{l}{\textit{LLM ablation (BFS fallback)}} \\"]
    for model_id,name in LLM_MODELS:
        if model_id not in llm_res: continue
        gains=[llm_res[model_id][d]["gain"] for d in TEST_DOMAINS if d in llm_res[model_id]]
        systems=[llm_res[model_id][d]["system"] for d in TEST_DOMAINS if d in llm_res[model_id]]
        s_strs=[f"{llm_res[model_id].get(d,{}).get('system',0):.0%}" for d in TEST_DOMAINS]
        mg=f"{np.mean(gains):+.1%}" if gains else "---"
        lines.append(f"  & {name} & "+" & ".join(s_strs)+f" & --- & --- & {mg} \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    tex="\n".join(lines)
    (RESULTS/"hybrid_ablation_table.tex").write_text(tex)
    print(tex)
    print(f"\n  Saved → {RESULTS}/hybrid_ablation_table.tex")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--part",default="solvers",choices=["solvers","llms","all"])
    p.add_argument("--host",default="http://localhost:11434")
    p.add_argument("--timeout",type=int,default=10)
    p.add_argument("--n",type=int,default=50)
    args=p.parse_args()

    print(f"\nHybrid Ablation v2")
    prep,X_surf,X_fm,tt,y_s,y_ns,model,s17,S_surf,S_fm,S_V,eps_by_dom,qwen_by_dom=load_all()
    print("\nLoading PlanBench...")
    planbench=load_planbench()

    solver_res={}; llm_res={}

    if args.part in ("solvers","all"):
        solver_res=run_solver_ablation(eps_by_dom,planbench,N=args.n,timeout=args.timeout)

    if args.part in ("llms","all"):
        llm_res=run_llm_ablation(args.host,eps_by_dom,model,prep,X_surf,X_fm,tt,
                                  qwen_by_dom,S_surf,S_fm,S_V,N=args.n,timeout=120)

    if solver_res or llm_res:
        make_latex(solver_res,llm_res)

if __name__=="__main__": main()
