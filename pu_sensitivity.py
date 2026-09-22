import importlib.util, pickle, sys, warnings, json
from pathlib import Path
import numpy as np, torch
from scipy import stats
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results_planning"
TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
TRAIN_DOMAINS = ["depot", "rovers", "satellite"]
DEVICE = "cpu"

def load_all():
    spec = importlib.util.spec_from_file_location("step17", ROOT/"plan_step17_arc_v2.py")
    s17  = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
    sys.modules["step17"] = s17
    sys.modules["__main__"].GlobalPreprocessor = s17.GlobalPreprocessor
    spec6 = importlib.util.spec_from_file_location("step6", ROOT/"plan_step6_pddlinst_gate.py")
    s6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    with open(RESULTS/"global_preprocessor.pkl","rb") as f:
        prep = pickle.load(f)
    X_surf,X_fm,tt,y_s,y_ns,splits = s6.load_data(data_dir=ROOT/"data"/"planning")
    X_surf = X_surf[:,:-1]
    return s17, prep, X_surf, X_fm, tt, y_s, y_ns

def train_eval(s17, prep, X_surf, X_fm, tt, y_s, y_ns, pi, n_ep=1000):
    surf_dim = X_surf.shape[1]; fm_dim = X_fm.shape[1]
    model = s17.ARCv2(surf_dim, fm_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    pu    = s17.NNPULoss(prior=pi)
    rng   = np.random.default_rng(42)
    tr    = np.isin(tt, TRAIN_DOMAINS)
    Xs_tr, Xe_tr, Xr_tr = prep.transform(X_surf[tr], X_fm[tr])
    sidx  = rng.choice(tr.sum(), min(60, tr.sum()), replace=False)
    S_surf = torch.FloatTensor(Xs_tr[sidx])
    S_fm   = torch.FloatTensor(Xe_tr[sidx])
    S_V    = torch.FloatTensor(np.hstack([Xs_tr[sidx], Xe_tr[sidx]]))
    spec3  = importlib.util.spec_from_file_location("step3", ROOT/"plan_step3_guru.py")
    s3     = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(s3)
    sampler = s3.PlanningMetaSampler(TRAIN_DOMAINS, X_surf, X_fm, y_s, y_ns, tt, DEVICE)
    model.train()
    for _ in range(n_ep):
        ep = sampler.sample_episode(label="success")
        if ep is None: continue
        out,_,_,c = model(ep["Q_surf"],ep["Q_fm"],ep["Q_resid"],
                          ep["S_surf"],ep["S_fm"],ep["S_V"],head="bfs",return_gate=True)
        loss = pu(out, ep["Y_cls"], ep["Y_reg"]>12)
        if torch.isnan(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    out_r = {}
    for dom in TEST_DOMAINS:
        mask = tt==dom
        Xs_n,Xe_n,Xr_n = prep.transform(X_surf[mask], X_fm[mask])
        ns_q = y_ns[mask].astype(float)
        scores = []
        with torch.no_grad():
            for i in range(len(Xs_n)):
                qs=torch.FloatTensor(Xs_n[i]).unsqueeze(0)
                qf=torch.FloatTensor(Xe_n[i]).unsqueeze(0)
                qr=torch.FloatTensor(Xr_n[i]).unsqueeze(0)
                o,_,_=model(qs,qf,qr,S_surf,S_fm,S_V,head="reg")
                scores.append(float(o.squeeze().cpu()))
        rho,_=stats.spearmanr(scores,ns_q)
        out_r[dom]=abs(float(rho))
    return out_r

if __name__=="__main__":
    s17,prep,X_surf,X_fm,tt,y_s,y_ns = load_all()
    print(f"{'pi':<6}  {'BW':>7}  {'LOG':>7}  {'MBW':>7}  {'mean':>7}")
    print("-"*36)
    all_r={}
    for pi in [0.25, 0.33, 0.40]:
        print(f"  pi={pi:.2f}...", flush=True)
        r = train_eval(s17,prep,X_surf,X_fm,tt,y_s,y_ns,pi,n_ep=1000)
        m = np.mean(list(r.values()))
        print(f"  {pi:<6.2f}  {r['blocksworld']:>7.3f}  {r['logistics']:>7.3f}  {r['mystery_blocksworld']:>7.3f}  {m:>7.3f}")
        all_r[pi]=r
    spread = max(np.mean(list(r.values())) for r in all_r.values()) - min(np.mean(list(r.values())) for r in all_r.values())
    print(f"\nSpread: {spread:.4f}  Stable (< 0.02): {spread < 0.02}")
    (RESULTS/"pu_sensitivity.json").write_text(json.dumps({str(k):v for k,v in all_r.items()},indent=2))
    print("Saved → results_planning/pu_sensitivity.json")