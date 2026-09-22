#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --job-name=arc_retrain
#SBATCH --output=/scratch/mroycho1/GURU/retrain_%j.log
#SBATCH --chdir=/scratch/mroycho1/GURU

source /home/mroycho1/.bashrc && conda activate mdebnath

# Train with contrastive loss, 5000 episodes, surf_dim=30
# Save to new filename (DO NOT overwrite guru_contrastive.pt)
python plan_step3_guru.py \
    --n_episodes 5000 \
    --contrastive \
    --seed 42 \
    --save_path checkpoints_planning/guru_contrastive_v2.pt

# Evaluate immediately after
python3 - << 'EVAL'
import torch, numpy as np, importlib.util, sys
from pathlib import Path; from scipy import stats
ROOT = Path('/scratch/mroycho1/GURU')
spec6 = importlib.util.spec_from_file_location("s6", ROOT/"plan_step6_pddlinst_gate.py")
s6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
spec3 = importlib.util.spec_from_file_location("s3", ROOT/"plan_step3_guru.py")
s3 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(s3)
X_surf,X_fm,tt,y_s,y_ns,_=s6.load_data(data_dir=ROOT/"data"/"planning")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
TRAIN=["depot","rovers","satellite"]
tr=np.isin(tt,TRAIN)
sc_s=StandardScaler().fit(X_surf[tr]); sc_f=StandardScaler().fit(X_fm[tr])
pp=Pipeline([("pca",PCA(20)),("r",Ridge(1.0))]); pp.fit(sc_s.transform(X_surf[tr]),sc_f.transform(X_fm[tr]))
def tfm(Xs,Xe):
    Xs_n=sc_s.transform(Xs); Xe_n=sc_f.transform(Xe)
    return Xs_n,Xe_n,Xe_n-pp.predict(Xs_n)
ck=torch.load(ROOT/"checkpoints_planning/guru_contrastive_v2.pt",map_location="cpu")
model=s3.PlanningGURU(30,X_fm.shape[1]); model.load_state_dict(ck["model"]); model.eval()
rng=np.random.default_rng(42); sidx=rng.choice(tr.sum(),min(60,tr.sum()),replace=False)
Xs_tr,Xe_tr,_=tfm(X_surf[tr],X_fm[tr])
S_s=torch.FloatTensor(Xs_tr[sidx]); S_f=torch.FloatTensor(Xe_tr[sidx])
S_V=torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))
print("\n=== RETRAINED MODEL EVALUATION ===")
for dom in ["blocksworld","logistics","mystery_blocksworld"]:
    mask=tt==dom; Xs_n,Xe_n,Xr_n=tfm(X_surf[mask],X_fm[mask])
    sc=[]
    with torch.no_grad():
        for i in range(200):
            qs=torch.FloatTensor(Xs_n[i]).unsqueeze(0); qf=torch.FloatTensor(Xe_n[i]).unsqueeze(0)
            qr=torch.FloatTensor(Xr_n[i]).unsqueeze(0)
            import torch.nn.functional as F
            o,_,_=model(qs,qf,qr,S_s,S_f,S_V,head="cls")
            sc.append(float(F.softmax(o.squeeze(0),-1)[1].cpu()))
    rho,_=stats.spearmanr(sc,y_ns[mask][:200])
    print(f"  {dom[:3].upper()}: rho={rho:.3f}")
print("Target: BW≈0.72  LOG≈0.76  MBW≈0.73")
EVAL
