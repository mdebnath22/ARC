#!/bin/bash
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --job-name=dissociation
#SBATCH --output=/scratch/mroycho1/GURU/dissociation_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

source /home/mroycho1/.bashrc
micromamba activate mdebnath 2>/dev/null || conda activate mdebnath 2>/dev/null

export OLLAMA_MODELS=/scratch/mroycho1/.ollama/models
export OLLAMA_HOME=/scratch/mroycho1/.ollama
module load ollama/0.12.10
ollama serve > /tmp/ollama_dissoc.log 2>&1 &
sleep 90

HOST="http://$(hostname):11434"
echo "Host: $HOST"

# Verify 72B loads
RESP=$(curl -s --max-time 180 "$HOST/api/generate" \
  -d '{"model":"qwen2.5:72b","prompt":"say hi","stream":false,"options":{"num_predict":5}}')
echo "Test: $RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK:', d.get('response','FAIL')[:20])" 2>/dev/null

python3 plan_step31_with_perinst.py \
    --part C \
    --host "$HOST" \
    --model qwen2.5:72b \
    --n 40

# Compute dissociation table
python3 - << 'DISSOC'
import json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path; from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import importlib.util, unittest.mock

ROOT = Path('/scratch/mroycho1/GURU')
spec6=importlib.util.spec_from_file_location('s6',ROOT/'plan_step6_pddlinst_gate.py')
s6=importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
spec3=importlib.util.spec_from_file_location('s3',ROOT/'plan_step3_guru.py')
s3=importlib.util.module_from_spec(spec3)
with unittest.mock.patch('sys.argv',['x']):
    try: spec3.loader.exec_module(s3)
    except: pass

X_surf,X_fm,tt,_,y_ns,_=s6.load_data(data_dir=ROOT/'data'/'planning')
TRAIN=['depot','rovers','satellite']
tr=np.isin(tt,TRAIN)
sc_s=StandardScaler().fit(X_surf[tr]); sc_f=StandardScaler().fit(X_fm[tr])
pp=Pipeline([('pca',PCA(20)),('r',Ridge(1.0))])
pp.fit(sc_s.transform(X_surf[tr]),sc_f.transform(X_fm[tr]))
def tfm(Xs,Xe):
    Xs_n=sc_s.transform(Xs);Xe_n=sc_f.transform(Xe);return Xs_n,Xe_n,Xe_n-pp.predict(Xs_n)
rng=np.random.default_rng(42); sidx=rng.choice(tr.sum(),min(60,tr.sum()),replace=False)
Xs_tr,Xe_tr,_=tfm(X_surf[tr],X_fm[tr])
S_s=torch.FloatTensor(Xs_tr[sidx]); S_f=torch.FloatTensor(Xe_tr[sidx])
S_V=torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))
ck=torch.load(ROOT/'checkpoints_planning'/'guru_success.pt',map_location='cpu')
model=s3.PlanningGURU(30,X_fm.shape[1]); model.load_state_dict(ck['model']); model.eval()

bw_idx = np.where(tt=='blocksworld')[0][:40]
Xs_n,Xe_n,Xr_n = tfm(X_surf[bw_idx], X_fm[bw_idx])
arc_scores = []
with torch.no_grad():
    for i in range(40):
        qs=torch.FloatTensor(Xs_n[i]).unsqueeze(0)
        qf=torch.FloatTensor(Xe_n[i]).unsqueeze(0)
        qr=torch.FloatTensor(Xr_n[i]).unsqueeze(0)
        o,_,_=model(qs,qf,qr,S_s,S_f,S_V,head='cls')
        arc_scores.append(float(F.softmax(o[0],-1)[1].cpu()))
arc_scores = np.array(arc_scores)
ns_40 = y_ns[bw_idx].astype(float)
n_obj_40 = X_surf[bw_idx, 0]
valid = ns_40 > 0
rho_struct,_ = stats.spearmanr(arc_scores[valid], ns_40[valid])

d = json.loads((ROOT/'results_planning/progressive_corruption.json').read_text())

print("\nSTRUCTURAL-SEMANTIC DISSOCIATION TABLE")
print("="*75)
print(f"{'Level':<8} {'LLM%':>8} {'Refusal%':>10} {'ARC|rho|_struct':>15} {'rho(ARC,LLM)':>13} {'rho(|O|,LLM)':>13}")
print("-"*75)

for k_str, v in sorted(d.items(),
                        key=lambda x: float(x[0]) if x[0]!='stratified' else 99):
    if k_str == 'stratified': continue
    k = float(k_str)
    per_inst = v.get('per_instance', [])
    if per_inst:
        llm_arr = np.array([p['valid'] for p in per_inst[:40]])
        rho_arc_llm,_ = stats.spearmanr(arc_scores[:len(llm_arr)], llm_arr)
        rho_obj_llm,_ = stats.spearmanr(n_obj_40[:len(llm_arr)], llm_arr)
        ra = f"{rho_arc_llm:>13.3f}"; ro = f"{rho_obj_llm:>13.3f}"
    else:
        ra = f"{'N/A':>13}"; ro = f"{'N/A':>13}"
    label = f"{int(k*100)}%"
    print(f"  {label:<6} {v['success']*100:>8.1f}% {v['refusal']*100:>10.1f}% "
          f"{abs(rho_struct):>15.3f} {ra} {ro}")

print(f"\nARC structural |rho| (constant): {abs(rho_struct):.3f}")
DISSOC
