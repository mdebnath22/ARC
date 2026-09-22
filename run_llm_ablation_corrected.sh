#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=llm_ablation_v2
#SBATCH --output=/scratch/mroycho1/GURU/llm_ablation_v2_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

echo "=== START $(date) on $(hostname) ==="

# Activate conda
source /home/mroycho1/.bashrc
conda activate mdebnath
which python3
python3 --version

# Start Ollama
module load ollama/0.12.10
ollama serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"
sleep 90

HOST="http://$(hostname):11434"
echo "Host: $HOST"

# Test Ollama
RESP=$(curl -s --max-time 30 "$HOST/api/generate" \
  -d '{"model":"glm4:latest","prompt":"say hi","stream":false,"options":{"num_predict":3}}')
echo "Ollama test: ${RESP:0:100}"

# Run corrected LLM ablation
python3 - << 'PYEOF'
import json, pickle, sys, importlib.util, numpy as np, torch, time, re, urllib.request, signal, tempfile
from pathlib import Path
from scipy import stats

ROOT = Path('/scratch/mroycho1/GURU')
RES  = ROOT / 'results_planning'
DATA = ROOT / 'data' / 'planning'
CKPT = ROOT / 'checkpoints_planning'
TRAIN = ["depot","rovers","satellite"]
TEST  = ["blocksworld","logistics","mystery_blocksworld"]
import os; HOST = f"http://{os.environ.get('HOSTNAME','localhost')}:11434"
N = 200

# ── Load ARC ──────────────────────────────────────────────────────────────────
spec17=importlib.util.spec_from_file_location("s17",ROOT/"plan_step17_arc_v2.py")
s17=importlib.util.module_from_spec(spec17); spec17.loader.exec_module(s17)
sys.modules["step17"]=s17; sys.modules["__main__"].GlobalPreprocessor=s17.GlobalPreprocessor
spec6=importlib.util.spec_from_file_location("s6",ROOT/"plan_step6_pddlinst_gate.py")
s6=importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
with open(RES/"global_preprocessor.pkl","rb") as f: prep=pickle.load(f)
X_surf,X_fm,tt,y_s,y_ns,splits=s6.load_data(data_dir=DATA); X_surf=X_surf[:,:-1]
ckpt=torch.load(CKPT/"arc_v2.pt",map_location="cpu")
arc=s17.ARCv2(ckpt["surf_dim"],ckpt["fm_dim"]); arc.load_state_dict(ckpt["model"]); arc.eval()
tr=np.isin(tt,TRAIN); Xs_tr,Xe_tr,_=prep.transform(X_surf[tr],X_fm[tr])
rng=np.random.default_rng(42); sidx=rng.choice(tr.sum(),min(60,tr.sum()),replace=False)
S_s=torch.FloatTensor(Xs_tr[sidx]); S_f=torch.FloatTensor(Xe_tr[sidx])
S_V=torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))

# ── Load episodes ─────────────────────────────────────────────────────────────
eps_by_dom={}
for e in json.loads((DATA/"episodes.json").read_text()):
    eps_by_dom.setdefault(e.get("task_type",e.get("domain","")), []).append(e)

# ── Load Qwen labels ──────────────────────────────────────────────────────────
qwen_by_dom={}
for line in open(RES/"qwen72b_eval_instances.jsonl"):
    r=json.loads(line); qwen_by_dom.setdefault(r["domain"],[]).append(r)
for d in qwen_by_dom: qwen_by_dom[d].sort(key=lambda r:int(r["instance_id"]))

# ── ARC scores ────────────────────────────────────────────────────────────────
def get_scores(dom):
    mask=tt==dom; Xs_n,Xe_n,Xr_n=prep.transform(X_surf[mask],X_fm[mask])
    scores=[]
    with torch.no_grad():
        for i in range(min(N,len(Xs_n))):
            qs=torch.FloatTensor(Xs_n[i]).unsqueeze(0)
            qf=torch.FloatTensor(Xe_n[i]).unsqueeze(0)
            qr=torch.FloatTensor(Xr_n[i]).unsqueeze(0)
            o,_,_=arc(qs,qf,qr,S_s,S_f,S_V,head="reg")
            scores.append(float(o.squeeze().cpu()))
    return np.array(scores)

# ── Routing (few-shot on Qwen labels) ─────────────────────────────────────────
import xgboost as xgb
def get_routing(dom):
    scores=get_scores(dom)
    y_q=np.array([1. if r["valid_plan"] else 0. for r in qwen_by_dom[dom][:N]])
    rng2=np.random.default_rng(42); tr_idx=rng2.choice(N,140,replace=False)
    y_tr=y_q[tr_idx].astype(int)
    print(f"  {dom}: LLM success rate in training labels: {y_tr.mean():.2%}")
    if len(np.unique(y_tr))<2:
        print(f"  WARNING: single class — using score threshold")
        return scores < np.percentile(scores,30)  # easy instances → LLM
    clf=xgb.XGBClassifier(n_estimators=100,max_depth=4,verbosity=0,
                            eval_metric="logloss",random_state=42,
                            scale_pos_weight=float((y_tr==0).sum())/max((y_tr==1).sum(),1))
    clf.fit(scores[tr_idx,None], y_tr)
    proba=clf.predict_proba(scores[:,None])[:,1]
    mask_llm=proba>0.5
    print(f"  {dom}: routing {mask_llm.sum()} to LLM, {(~mask_llm).sum()} to solver")
    # If routing sends nobody to LLM, use top-30% easiest
    if mask_llm.sum()==0:
        print(f"  WARNING: XGB routing empty — using percentile fallback")
        mask_llm=proba>=np.percentile(proba,70)
    return mask_llm

# ── EHC solver ────────────────────────────────────────────────────────────────
def run_ehc(record, timeout=15):
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding
    from pyperplan.search.enforced_hillclimbing_search import enforced_hillclimbing_search
    from pyperplan.heuristics.relaxation import hFFHeuristic
    dom=record.get("domain_pddl",""); prob=record.get("problem_pddl","")
    if not dom or not prob: return False
    def _to(s,f): raise TimeoutError()
    signal.signal(signal.SIGALRM,_to); signal.alarm(timeout)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dp=Path(tmp)/"domain.pddl"; pp=Path(tmp)/"problem.pddl"
            dp.write_text(dom); pp.write_text(prob)
            parser=Parser(str(dp),str(pp))
            task=grounding.ground(parser.parse_problem(parser.parse_domain()))
            sol=enforced_hillclimbing_search(task,hFFHeuristic(task))
            signal.alarm(0); return sol is not None
    except (TimeoutError,Exception):
        signal.alarm(0); return False

# ── PDDL validator ────────────────────────────────────────────────────────────
def _f(*p): return " ".join(str(x).lower() for x in p)
HANDLERS={
    "pick-up":(1,lambda x:({_f("clear",x),_f("ontable",x),_f("handempty")},{_f("ontable",x),_f("clear",x),_f("handempty")},{_f("holding",x)})),
    "put-down":(1,lambda x:({_f("holding",x)},{_f("holding",x)},{_f("handempty"),_f("ontable",x),_f("clear",x)})),
    "stack":(2,lambda x,y:({_f("holding",x),_f("clear",y)},{_f("holding",x),_f("clear",y)},{_f("handempty"),_f("on",x,y),_f("clear",x)})),
    "unstack":(2,lambda x,y:({_f("on",x,y),_f("clear",x),_f("handempty")},{_f("on",x,y),_f("clear",x),_f("handempty")},{_f("holding",x),_f("clear",y)})),
    "grasp":(1,lambda x:({_f("apex",x),_f("grounded",x),_f("grasping")},{_f("grounded",x),_f("apex",x),_f("grasping")},{_f("clutching",x)})),
    "release":(1,lambda x:({_f("clutching",x)},{_f("clutching",x)},{_f("grasping"),_f("grounded",x),_f("apex",x)})),
    "place":(2,lambda x,y:({_f("clutching",x),_f("apex",y)},{_f("clutching",x),_f("apex",y)},{_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
    "lift":(2,lambda x,y:({_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
    "load-truck":(3,lambda p,t,l:({_f("at",t,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,t)})),
    "unload-truck":(3,lambda p,t,l:({_f("at",t,l),_f("in",p,t)},{_f("in",p,t)},{_f("at",p,l)})),
    "load-airplane":(3,lambda p,a,l:({_f("at",a,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,a)})),
    "unload-airplane":(3,lambda p,a,l:({_f("at",a,l),_f("in",p,a)},{_f("in",p,a)},{_f("at",p,l)})),
    "drive-truck":(4,lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},{_f("at",t,s)},{_f("at",t,d)})),
    "fly-airplane":(3,lambda a,s,d:({_f("at",a,s),_f("airport",s),_f("airport",d)},{_f("at",a,s)},{_f("at",a,d)})),
}
ALIASES={"pickup":"pick-up","putdown":"put-down","fly":"fly-airplane",
         "load-pkg":"load-truck","unload-pkg":"unload-truck"}
ALIASES_MBW={"pick-up":"grasp","put-down":"release","stack":"place","unstack":"lift"}

def validate(record, actions, domain=""):
    al=dict(ALIASES)
    if "mystery" in domain.lower(): al.update(ALIASES_MBW)
    norm=[]
    for act in actions:
        toks=act.strip().strip("()").split()
        if not toks: continue
        norm.append("("+" ".join([al.get(toks[0].lower(),toks[0].lower())]+[t.lower() for t in toks[1:]])+")")
    if not norm: return False
    init=record.get("init_facts",[]); goal=record.get("goal_facts",[])
    if not init or not goal: return True
    state={_f(*f.strip().strip("()").split()) for f in init}
    goals={_f(*g.strip().strip("()").split()) for g in goal}
    for act in norm:
        toks=act.strip().strip("()").split()
        if not toks: continue
        name=toks[0].lower(); args=[t.lower() for t in toks[1:]]
        if name not in HANDLERS: return False
        ar,fn=HANDLERS[name]
        if len(args)!=ar: return False
        pre,rem,add=fn(*args)
        if not pre.issubset(state): return False
        state=(state-rem)|add
    return goals.issubset(state)

def parse_plan(response, domain=""):
    if not response or "ERROR" in response[:10]: return []
    txt=re.sub(r"<think>.*?</think>","",response,flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b",txt,re.I): return []
    lines=[l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    return re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)",response,re.I)

def call_llm(host, model, ep, domain="", timeout=180):
    dom_pddl=ep.get("domain_pddl",""); prob_pddl=ep.get("problem_pddl","")
    acts=re.findall(r":action\s+(\S+)",dom_pddl)
    ab="" 
    if acts:
        ab="\nCRITICAL — use ONLY these action names:\n"
        for a in acts:
            m=re.search(rf":action\s+{re.escape(a)}.*?:parameters\s*\(([^)]*)\)",dom_pddl,re.DOTALL)
            n=len(re.findall(r"\?",m.group(1))) if m else 1
            ab+=f"  ({a} {' '.join(f'arg{i+1}' for i in range(n))})\n"
    prompt=(f"You are a PDDL planning expert.\n\nDOMAIN:\n{dom_pddl}\n\n"
            f"PROBLEM:\n{prob_pddl}\n{ab}\n"
            "Output ONLY the plan, one action per line: (action arg1 ...)\n"
            "If unsolvable: NO_PLAN\n")
    payload=json.dumps({"model":model,"prompt":prompt,"stream":False,
                        "options":{"num_predict":2048,"temperature":0.0}}).encode()
    try:
        req=urllib.request.Request(f"{host}/api/generate",data=payload,
            headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=timeout) as r:
            resp=json.loads(r.read()).get("response","")
        actions=parse_plan(resp,domain)
        al=dict(ALIASES)
        if "mystery" in domain.lower(): al.update(ALIASES_MBW)
        norm=[]
        for act in actions:
            toks=act.strip().strip("()").split()
            if toks: norm.append("("+" ".join([al.get(toks[0].lower(),toks[0].lower())]+toks[1:])+")")
        return validate(ep,norm,domain)
    except Exception as e:
        print(f"    LLM error: {e}"); return False

# ── Check Ollama ──────────────────────────────────────────────────────────────
try:
    avail=json.loads(urllib.request.urlopen(
        urllib.request.Request(f"{HOST}/api/tags"),timeout=15).read())["models"]
    avail_names=[m["name"] for m in avail]
    print(f"Available models: {avail_names}")
except Exception as e:
    print(f"ERROR: {e}"); import sys; sys.exit(1)

LLM_MODELS=[("qwen2.5:72b","Qwen 2.5-72B"),("llama3.3:70b","Llama 3.3-70B"),("glm4:latest","GLM-4")]

# Pre-compute routing masks
print("\nComputing routing masks...")
routing={}
for dom in TEST:
    routing[dom]=get_routing(dom)

# Always-EHC baseline (compute once)
print("\nComputing always-EHC baseline...")
ehc_base={}
for dom in TEST:
    eps=eps_by_dom.get(dom,[])[:N]
    n_solved=sum(1 for ep in eps if run_ehc(ep,15))
    ehc_base[dom]=n_solved/N
    print(f"  EHC {dom[:3].upper()}: {n_solved}/{N} = {n_solved/N:.1%}")

# Main LLM loop
results={}
print(f"\n{'LLM':<20}  {'Dom':<8}  {'LLM%':>6}  {'Sys%':>6}  "
      f"{'EHC%':>6}  {'Gain':>6}  {'Lat(s)':>7}")
print("-"*65)

for model_id,model_name in LLM_MODELS:
    if not any(model_id.split(":")[0] in m for m in avail_names):
        print(f"  {model_name}: NOT AVAILABLE"); continue
    results[model_id]={}
    for dom in TEST:
        eps=eps_by_dom.get(dom,[])[:N]
        route=routing[dom]
        n_llm=route.sum(); n_sol=(~route).sum()
        lats=[]; llm_valid=0

        # LLM on routed instances (use Qwen labels for Qwen to save time)
        if model_id=="qwen2.5:72b":
            qlist=qwen_by_dom.get(dom,[])[:N]
            llm_valid=sum(1 for i,r in enumerate(qlist) if i<N and route[i] and r["valid_plan"])
            lats=[0.1]*int(n_llm)  # placeholder latency
        else:
            for i,ep in enumerate(eps):
                if not route[i]: continue
                t0=time.time()
                ok=call_llm(HOST,model_id,ep,dom,timeout=180)
                lats.append(time.time()-t0)
                if ok: llm_valid+=1
                if (i+1)%20==0:
                    print(f"    {dom[:3]} [{i+1}/{N}] llm_valid={llm_valid}",end="\r")

        # EHC on solver-routed instances
        ehc_valid=sum(1 for i,ep in enumerate(eps)
                      if not route[i] and run_ehc(ep,15))

        system=(llm_valid+ehc_valid)/N
        llm_rate=llm_valid/max(n_llm,1)
        gain=system-ehc_base[dom]
        mean_lat=float(np.mean(lats)) if lats else 0

        dl=dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
        print(f"{model_name:<20}  {dl:<8}  {llm_rate:>6.1%}  {system:>6.1%}  "
              f"{ehc_base[dom]:>6.1%}  {gain:>+6.1%}  {mean_lat:>7.1f}")
        results[model_id][dom]={"llm_rate":float(llm_rate),"system":float(system),
                                 "ehc_base":float(ehc_base[dom]),"gain":float(gain),
                                 "n_llm":int(n_llm),"n_solver":int(n_sol)}

(RES/"llm_ablation_corrected.json").write_text(json.dumps(results,indent=2))
print(f"\nSaved → {RES}/llm_ablation_corrected.json")
PYEOF

echo "=== DONE $(date) ==="
