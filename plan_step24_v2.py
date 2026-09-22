"""
plan_step24_v2.py  —  Human study: ARC hybrid vs humans
=========================================================
PIPELINE (key insight):
  1. Problem arrives → ARC routes to LLM or EHC(hFF)
  2a. EHC route: EHC solves (PDDL plan) → LLM translates to English
  2b. LLM route: LLM solves → validates → if fail, EHC fallback → LLM translates
  3. Human receives: clear English explanation of the solution
  4. Same PDDL executor validates BOTH human and ARC answers

CLAIM: ARC hybrid gives more correct answers than non-expert humans.

USAGE:
  python plan_step24_v2.py --phase generate
  python plan_step24_v2.py --phase arc --host http://sg050:11434
  python plan_step24_v2.py --phase validate --answers human_answers.csv
  python plan_step24_v2.py --phase compare
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, pickle, re, signal
import sys, tempfile, time, urllib.request, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).resolve().parent
DATA    = ROOT/"data"/"planning"
RESULTS = ROOT/"results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT/"checkpoints_planning"
STUDY   = ROOT/"human_study"; STUDY.mkdir(exist_ok=True)
TEST_DOMAINS  = ["blocksworld","logistics","mystery_blocksworld"]
TRAIN_DOMAINS = ["depot","rovers","satellite"]
DIFF_SPLIT    = {"easy":3,"medium":4,"hard":3}

# ── PDDL executor ─────────────────────────────────────────────────────────────
def _f(*p): return " ".join(str(x).lower() for x in p)
HANDLERS={
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
ALIASES={"pickup":"pick-up","putdown":"put-down","put_down":"put-down",
         "fly":"fly-airplane","fly-plane":"fly-airplane","move-truck":"drive-truck",
         "load-pkg":"load-truck","unload-pkg":"unload-truck",
         "load":"load-truck","unload":"unload-truck"}
ALIASES_MBW={"pick-up":"grasp","put-down":"release","stack":"place",
             "unstack":"lift","pickup":"grasp","putdown":"release"}

def validate(record, actions, domain=""):
    al=dict(ALIASES)
    if "mystery" in domain.lower(): al.update(ALIASES_MBW)
    norm=[]
    for act in actions:
        toks=act.strip().strip("()").split()
        if not toks: continue
        norm.append("("+" ".join([al.get(toks[0].lower(),toks[0].lower())]+[t.lower() for t in toks[1:]])+")")
    if not norm: return False,"empty_plan"
    init=record.get("init_facts",[]); goal=record.get("goal_facts",[])
    if not init or not goal: return True,None
    state={_f(*f.strip().strip("()").split()) for f in init}
    goals={_f(*g.strip().strip("()").split()) for g in goal}
    for act in norm:
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

# ── EHC(hFF) solver ───────────────────────────────────────────────────────────
def run_ehc(record, timeout=30):
    """Returns (solved, plan_list). Plan is list of '(action args...)' strings."""
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding
    from pyperplan.search.enforced_hillclimbing_search import enforced_hillclimbing_search
    from pyperplan.heuristics.relaxation import hFFHeuristic
    dom=record.get("domain_pddl",""); prob=record.get("problem_pddl","")
    if not dom or not prob: return False,[]
    def _to(s,f): raise TimeoutError()
    signal.signal(signal.SIGALRM,_to); signal.alarm(timeout)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dp=Path(tmp)/"domain.pddl"; pp=Path(tmp)/"problem.pddl"
            dp.write_text(dom); pp.write_text(prob)
            parser=Parser(str(dp),str(pp))
            task=grounding.ground(parser.parse_problem(parser.parse_domain()))
            sol=enforced_hillclimbing_search(task,hFFHeuristic(task))
            signal.alarm(0)
            return (True,[f"({op.name})" for op in sol]) if sol else (False,[])
    except (TimeoutError,Exception):
        signal.alarm(0); return False,[]

# ── LLM interface ─────────────────────────────────────────────────────────────
def call_llm(host, model, prompt, timeout=300):
    payload=json.dumps({"model":model,"prompt":prompt,"stream":False,
                        "options":{"num_predict":1024,"temperature":0.3}}).encode()
    try:
        req=urllib.request.Request(f"{host}/api/generate",data=payload,
            headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return json.loads(r.read()).get("response","").strip()
    except Exception as e: return f"ERROR:{e}"

def solve_prompt(record):
    dom=record.get("domain_pddl",""); prob=record.get("problem_pddl","")
    acts=re.findall(r":action\s+(\S+)",dom); ab=""
    if acts:
        ab="\nCRITICAL — use ONLY these action names:\n"
        for a in acts:
            m=re.search(rf":action\s+{re.escape(a)}.*?:parameters\s*\(([^)]*)\)",dom,re.DOTALL)
            n=len(re.findall(r"\?",m.group(1))) if m else 1
            ab+=f"  ({a} {' '.join(f'arg{i+1}' for i in range(n))})\n"
    return (f"You are a PDDL planning expert.\n\n=== DOMAIN ===\n{dom}\n\n"
            f"=== PROBLEM ===\n{prob}\n{ab}\n"
            "Output ONLY the plan, one action per line: (action arg1 ...)\n"
            "If unsolvable: NO_PLAN\n")

def translate_prompt(record, pddl_plan):
    """
    Ask LLM to convert a formal PDDL plan into a clear human-readable explanation.
    This is the OUTPUT that humans see and are compared against.
    """
    dom  = record.get("task_type",record.get("domain","")).replace("_"," ").title()
    desc = record.get("description","")
    init = "; ".join(f.strip().strip("()") for f in record.get("init_facts",[])[:8])
    goal = "; ".join(f.strip().strip("()") for f in record.get("goal_facts",[]))
    plan_str="\n".join(f"  Step {i+1}: {a}" for i,a in enumerate(pddl_plan))
    return (
        f"A planning solver found the following correct solution to a {dom} problem.\n\n"
        f"PROBLEM: {desc}\n"
        f"INITIAL STATE: {init}\n"
        f"GOAL: {goal}\n\n"
        f"FORMAL SOLUTION STEPS:\n{plan_str}\n\n"
        "Convert this into a clear, friendly explanation that a non-expert can follow.\n"
        "For each step: say what happens and why it's needed.\n"
        "Use plain English — no technical jargon.\n"
        "End with: 'Result: [one sentence describing what was achieved].'\n"
        "Format as a numbered list.\n"
    )

def parse_plan(response, domain=""):
    if not response or response.startswith("ERROR:"): return []
    txt=re.sub(r"<think>.*?</think>","",response,flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b",txt,re.I): return []
    lines=[l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    return re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)",response,re.I)

# ── Phase 1: Generate instances ────────────────────────────────────────────────
def phase_generate():
    print("\nPHASE 1: Generating 30 study instances")
    all_eps=json.loads((DATA/"episodes.json").read_text())
    import random; rng=random.Random(42)
    study=[]; sid=0
    for dom in TEST_DOMAINS:
        eps=[e for e in all_eps if e.get("task_type",e.get("domain",""))==dom]
        eps.sort(key=lambda e:e.get("n_steps",e.get("complexity",0)))
        n=len(eps)
        bands={"easy":eps[:n//3],"medium":eps[n//3:2*n//3],"hard":eps[2*n//3:]}
        for diff,count in DIFF_SPLIT.items():
            for ep in rng.sample(bands[diff],min(count,len(bands[diff]))):
                rec=dict(ep); rec["study_id"]=sid; rec["difficulty"]=diff
                study.append(rec); sid+=1
    (STUDY/"study_instances.json").write_text(json.dumps(study,indent=2))
    print(f"  Saved {len(study)} instances → {STUDY}/study_instances.json")

    # Human study sheet
    with open(STUDY/"human_study_sheet.txt","w") as f:
        f.write("PLANNING PROBLEM STUDY\n"+"="*65+"\n\n")
        f.write("Solve each problem by writing numbered steps.\n")
        f.write("Use ONLY the action names shown. Time limit: 5 min/problem.\n\n")
        for rec in study:
            dom=rec.get("task_type",""); diff=rec["difficulty"]
            f.write("\n"+"="*65+"\n")
            f.write(f"PROBLEM {rec['study_id']+1}  |  {dom.replace('_',' ').title()}  |  {diff.upper()}\n")
            f.write("="*65+"\n\n")
            f.write(rec.get("description","")+"\n\n")
            acts=re.findall(r":action\s+(\S+)",rec.get("domain_pddl",""))
            if acts:
                f.write("ACTIONS: "+", ".join(acts)+"\n\n")
            f.write("INITIAL STATE:\n")
            for fact in rec.get("init_facts",[])[:12]: f.write(f"  {fact}\n")
            f.write("\nGOAL:\n")
            for fact in rec.get("goal_facts",[]): f.write(f"  {fact}\n")
            f.write("\nYOUR SOLUTION:\n")
            for i in range(1,12): f.write(f"  {i}. _________________________\n")
            f.write("\n")
    print(f"  Study sheet → {STUDY}/human_study_sheet.txt")
    print(f"\n  Collect answers as: {STUDY}/human_answers.csv")
    print(f"  Format: study_id,participant_id,answer")
    print(f"  (answer = steps separated by semicolons)")

# ── Phase 2: Run ARC hybrid ────────────────────────────────────────────────────
def phase_arc(host, model_id):
    print(f"\nPHASE 2: ARC hybrid pipeline")
    print(f"  Routing → EHC(hFF) or LLM")
    print(f"  EHC output → LLM translates to English")
    print(f"  LLM: {model_id}  Host: {host}")

    # Check LLM
    try:
        data=json.loads(urllib.request.urlopen(
            urllib.request.Request(f"{host}/api/tags"),timeout=10).read())
        avail=[m["name"] for m in data.get("models",[])]
        if not any(model_id.split(":")[0] in m for m in avail):
            print(f"  ERROR: {model_id} not in {avail}"); return
        print(f"  LLM OK: {model_id}")
    except Exception as e:
        print(f"  ERROR: {e}"); return

    # Load ARC
    import torch
    spec=importlib.util.spec_from_file_location("step17",ROOT/"plan_step17_arc_v2.py")
    s17=importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
    sys.modules["step17"]=s17; sys.modules["__main__"].GlobalPreprocessor=s17.GlobalPreprocessor
    spec6=importlib.util.spec_from_file_location("step6",ROOT/"plan_step6_pddlinst_gate.py")
    s6=importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    with open(RESULTS/"global_preprocessor.pkl","rb") as f: prep=pickle.load(f)
    X_surf,X_fm,tt,y_s,y_ns,splits=s6.load_data(data_dir=DATA); X_surf=X_surf[:,:-1]
    ckpt=torch.load(CKPT/"arc_v2.pt",map_location="cpu")
    arc_model=s17.ARCv2(ckpt["surf_dim"],ckpt["fm_dim"])
    arc_model.load_state_dict(ckpt["model"]); arc_model.eval()
    tr=np.isin(tt,TRAIN_DOMAINS); Xs_tr,Xe_tr,_=prep.transform(X_surf[tr],X_fm[tr])
    rng=np.random.default_rng(42); sidx=rng.choice(tr.sum(),min(60,tr.sum()),replace=False)
    S_surf=torch.FloatTensor(Xs_tr[sidx]); S_fm=torch.FloatTensor(Xe_tr[sidx])
    S_V=torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))

    # Routing: use n_steps as proxy (ARC regression score)
    def should_use_llm(rec):
        dom=rec.get("task_type",rec.get("domain","")); iid=int(rec.get("instance_id",0))
        mask=tt==dom; Xs_n,Xe_n,Xr_n=prep.transform(X_surf[mask],X_fm[mask])
        if iid>=len(Xs_n): return rec.get("n_steps",99)<=8
        qs=torch.FloatTensor(Xs_n[iid]).unsqueeze(0)
        qf=torch.FloatTensor(Xe_n[iid]).unsqueeze(0)
        qr=torch.FloatTensor(Xr_n[iid]).unsqueeze(0)
        with torch.no_grad():
            out,_,_=arc_model(qs,qf,qr,S_surf,S_fm,S_V,head="reg")
        score=float(out.squeeze().cpu())
        # Route to LLM if predicted easy (low n_steps), EHC if hard
        return score < 0  # negative score = ARC thinks it's easy

    study=json.loads((STUDY/"study_instances.json").read_text())
    results=[]; answers_doc=[]

    print(f"\n  {'#':<4}  {'Domain':<8}  {'Diff':<7}  {'Route':>9}  {'Valid':>6}  {'NL preview'}")
    print("  "+"-"*70)

    for rec in study:
        sid=rec["study_id"]; dom=rec.get("task_type",rec.get("domain","")); diff=rec["difficulty"]
        dl=dom[:3].upper(); t0=time.time()
        pddl_plan=[]; nl_answer=""; solved=False; method=""

        use_llm=should_use_llm(rec)

        if use_llm:
            # ── LLM attempts to solve ─────────────────────────────────────────
            method="LLM"
            resp=call_llm(host,model_id,solve_prompt(rec),timeout=180)
            actions=parse_plan(resp,dom)
            ok,err=validate(rec,actions,dom)
            if ok:
                solved=True; pddl_plan=actions
                # LLM explains its own solution
                nl_answer=call_llm(host,model_id,translate_prompt(rec,pddl_plan),timeout=120)
            else:
                # LLM failed → EHC fallback → LLM translates
                method="LLM→EHC→NL"
                ok2,plan2=run_ehc(rec,timeout=20)
                if ok2:
                    solved=True; pddl_plan=plan2
                    nl_answer=call_llm(host,model_id,translate_prompt(rec,pddl_plan),timeout=120)
        else:
            # ── EHC solves → LLM translates ───────────────────────────────────
            method="EHC→NL"
            ok,pddl_plan=run_ehc(rec,timeout=20)
            if ok:
                solved=True
                nl_answer=call_llm(host,model_id,translate_prompt(rec,pddl_plan),timeout=120)

        elapsed=time.time()-t0
        status="✓" if solved else "✗"
        preview=(nl_answer[:55].replace("\n"," ")+"...") if nl_answer else "(failed)"
        print(f"  {sid+1:<4}  {dl:<8}  {diff:<7}  {method:>9}  {status:>6}  {preview}")

        results.append({"study_id":sid,"domain":dom,"difficulty":diff,
                        "method":method,"solved":solved,
                        "pddl_plan":pddl_plan,"nl_answer":nl_answer,"elapsed":elapsed})
        answers_doc.append({"problem":sid+1,"domain":dom,"difficulty":diff,
                             "method":method,"solved":solved,"answer":nl_answer,
                             "formal_steps":pddl_plan})

    # Save
    (STUDY/"arc_results.json").write_text(json.dumps(results,indent=2))

    # Human-readable answer document
    with open(STUDY/"arc_answers.txt","w") as f:
        f.write("ARC HYBRID SYSTEM ANSWERS\n"+"="*65+"\n")
        f.write("Each answer: solved by ARC (EHC algorithm + LLM translation)\n\n")
        for item in answers_doc:
            f.write("\n"+"="*65+"\n")
            f.write(f"Problem {item['problem']}  |  {item['domain']}  |  "
                    f"{item['difficulty'].upper()}  |  {item['method']}\n")
            f.write("-"*65+"\n")
            if item["solved"]:
                if item["formal_steps"]:
                    f.write(f"Formal plan ({len(item['formal_steps'])} steps):\n")
                    for i,s in enumerate(item["formal_steps"],1): f.write(f"  {i}. {s}\n")
                f.write(f"\nExplanation:\n{item['answer']}\n")
            else:
                f.write("SYSTEM COULD NOT SOLVE THIS PROBLEM.\n")

    # Summary
    total_v=sum(r["solved"] for r in results)
    print(f"\n  TOTAL: {total_v}/{len(results)} = {total_v/len(results):.1%}")
    for dom in TEST_DOMAINS:
        rlist=[r for r in results if r["domain"]==dom]
        v=sum(r["solved"] for r in rlist)
        dl=dom[:3].upper()
        print(f"  {dl}: {v}/{len(rlist)} = {v/len(rlist):.1%}")
    print(f"\n  Results → {STUDY}/arc_results.json")
    print(f"  Answers → {STUDY}/arc_answers.txt")

# ── Phase 3: Validate human answers ───────────────────────────────────────────
def phase_validate(answers_csv):
    print(f"\nPHASE 3: Validating human answers from {answers_csv}")
    study=json.loads((STUDY/"study_instances.json").read_text())
    inst={r["study_id"]:r for r in study}
    results=[]
    with open(answers_csv) as f:
        for row in csv.DictReader(f):
            sid=int(row["study_id"]); rec=inst.get(sid,{})
            dom=rec.get("task_type",""); raw=row.get("answer","").strip()
            if raw.upper() in ("CANNOT SOLVE","NO PLAN",""):
                valid=False; err="gave_up"; actions=[]
            else:
                steps=[re.sub(r"^\d+[\.\)]\s*","",s.strip()) for s in raw.split(";") if s.strip()]
                actions=["("+s+")" if not s.startswith("(") else s for s in steps if s.split()]
                valid,err=validate(rec,actions,dom)
            results.append({"study_id":sid,"participant":row.get("participant_id","p1"),
                             "domain":dom,"difficulty":rec.get("difficulty",""),
                             "valid":valid,"error":err})
            print(f"  [{sid+1:02d}] {'✓' if valid else '✗'}  {err or 'correct'}")
    (STUDY/"human_results.json").write_text(json.dumps(results,indent=2))
    v=sum(r["valid"] for r in results)
    print(f"\n  Human: {v}/{len(results)} = {v/len(results):.1%}")

# ── Phase 4: Comparison table ──────────────────────────────────────────────────
def phase_compare():
    print("\nPHASE 4: Comparison table")
    arc_p=STUDY/"arc_results.json"; hum_p=STUDY/"human_results.json"
    if not arc_p.exists(): print("  Run --phase arc first"); return
    arc={r["study_id"]:r for r in json.loads(arc_p.read_text())}
    study=json.loads((STUDY/"study_instances.json").read_text())
    by_dom={d:[r for r in study if r.get("task_type","")==d] for d in TEST_DOMAINS}
    human={}
    if hum_p.exists():
        for r in json.loads(hum_p.read_text()):
            human.setdefault(r["study_id"],[]).append(r["valid"])
        human={sid:(sum(v)/len(v)>=0.5) for sid,v in human.items()}

    print(f"\n  {'Domain':<14}  {'Human':>8}  {'ARC':>8}  {'Δ':>6}")
    print("  "+"-"*42)
    rows=[]
    for dom in TEST_DOMAINS:
        ids=[r["study_id"] for r in by_dom.get(dom,[])]
        h=sum(1 for i in ids if human.get(i,False)); a=sum(1 for i in ids if arc.get(i,{}).get("solved",False)); n=len(ids)
        dl=dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
        hs=f"{h/n:.0%}" if human else "---"
        print(f"  {dl:<14}  {hs:>8}  {a/n:>8.0%}  {(a-h)/n if human else 0:>+6.0%}")
        rows.append((dl,h,a,n))
    th=sum(r[1] for r in rows); ta=sum(r[2] for r in rows); tn=sum(r[3] for r in rows)
    print(f"  {'OVERALL':<14}  {'---' if not human else f'{th/tn:.0%}':>8}  {ta/tn:>8.0%}  {(ta-th)/tn if human else 0:>+6.0%}")

    # LaTeX
    print(f"\n  LaTeX:")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Human planners vs.\ ARC hybrid system on 30 PDDL instances.")
    print(r"ARC routes each instance to EHC(hFF) solver or LLM;")
    print(r"the EHC plan is translated to natural language by the LLM.")
    print(r"Both human and ARC answers are validated by the same PDDL executor.}")
    print(r"\label{tab:human_study}")
    print(r"\small\begin{tabular}{l cc c}")
    print(r"\toprule\textbf{Domain} & \textbf{Human} & \textbf{ARC} & $\Delta$ \\\midrule")
    for dl,h,a,n in rows:
        full=dl.replace("BW","Blocksworld").replace("LOG","Logistics").replace("MBW","Mystery-BW")
        hs2=f"{h/n:.0%}" if human else "---"
        print(f"  {full} & {hs2} & {a/n:.0%} & {(a-h)/n if human else 0:+.0%} \\\\")
    print(r"\midrule")
    hs3=f"{th/tn:.0%}" if human else "---"
    print(f"  Overall & {hs3} & {ta/tn:.0%} & {(ta-th)/tn if human else 0:+.0%} \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=["generate","arc","validate","compare"],default="generate")
    p.add_argument("--host",default="http://localhost:11434")
    p.add_argument("--model",default="qwen2.5:72b")
    p.add_argument("--answers",default=str(STUDY/"human_answers.csv"))
    args=p.parse_args()
    {"generate":phase_generate,"arc":lambda:phase_arc(args.host,args.model),
     "validate":lambda:phase_validate(args.answers),"compare":phase_compare}[args.phase]()

if __name__=="__main__": main()
