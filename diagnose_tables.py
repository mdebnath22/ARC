"""
diagnose_tables.py  (fixed)
============================
Run on sg027:  python diagnose_tables.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
import importlib.util

ROOT = Path(__file__).resolve().parent

# Load step3 (no data_dir argument — hardcoded internally)
spec = importlib.util.spec_from_file_location(
    "step3", ROOT / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step3)

# Load step6 for get_guru_scores_per_instance
spec6 = importlib.util.spec_from_file_location(
    "step6", ROOT / "plan_step6_pddlinst_gate.py")
step6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(step6)

print("Loading data...")
X_surf, X_fm, y_success, y_nsteps, task_types, registry = step3.load_all_data()

splits     = registry["splits"]
train_doms = splits["meta_train"]["domains"]
train_mask = np.isin(task_types, train_doms)
X_surf_s   = X_surf[train_mask]
X_fm_s     = X_fm[train_mask]
y_success_s = y_success[train_mask]

import torch
ckpt  = torch.load(ROOT / "checkpoints_planning" / "guru_success.pt",
                   map_location="cpu")
state = ckpt["model"]
model = step3.PlanningGURU(state["key_enc.net.0.weight"].shape[1],
                            state["query_enc.net.0.weight"].shape[1])
model.load_state_dict(state)
model.eval()

rows = [json.loads(l) for l in open(ROOT / "gpt4o_eval_instances.jsonl") if l.strip()]
by_dom = {}
for r in rows:
    by_dom.setdefault(r["domain"], []).append(r)
for dom in by_dom:
    by_dom[dom].sort(key=lambda r: int(r["instance_id"]))
print(f"Loaded {len(rows)} GPT-4o results\n")

TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]
LABELS = {"blocksworld":"Blocksworld","logistics":"Logistics",
          "mystery_blocksworld":"Mystery-BW"}

for dom in TEST_DOMAINS:
    mask   = task_types == dom
    Xs_q   = X_surf[mask];  Xf_q = X_fm[mask]
    y_q    = y_success[mask].astype(float)
    ns_q   = y_nsteps[mask]
    n_obj  = Xs_q[:, 0].astype(int)
    N      = mask.sum()
    y_gpt  = np.array([1.0 if r["valid_plan"] else 0.0
                       for r in by_dom.get(dom, [])[:N]])

    print(f"{'='*65}")
    print(f"{LABELS[dom].upper()}  (N={N})")
    print(f"{'='*65}")

    n_cap = (ns_q > 12).sum()
    print(f"  GPT-4o validity:  {y_gpt.mean():.1%}  ({int(y_gpt.sum())}/{N})")
    print(f"  Within-budget:    {(ns_q<=12).sum()}  Cap-exceeded: {n_cap}")
    print(f"  y_success=1:      {int(y_q.sum())}  y_success=0: {int((1-y_q).sum())}")
    print(f"  n_steps range:    [{ns_q.min():.0f},{ns_q.max():.0f}]  "
          f"median={np.median(ns_q):.1f}")

    # AUC with different predictors
    auc_nobj = roc_auc_score(y_q, -n_obj)
    cap_flag = (ns_q <= 12).astype(float)
    auc_cap  = roc_auc_score(y_q, cap_flag) if len(np.unique(cap_flag))>1 else float("nan")
    print(f"\n  AUC (BFS y_success label):")
    print(f"    n_objects alone:  {auc_nobj:.4f}  ← single feature")
    print(f"    cap_flag alone:   {auc_cap:.4f}  ← pure boundary")

    print(f"    Computing ARC scores...", end=" ", flush=True)
    arc = step6.get_guru_scores_per_instance(
        model, Xs_q, Xf_q, y_q, X_surf_s, X_fm_s, n_runs=10, rng_seed=42)
    print("done")
    auc_arc = roc_auc_score(y_q, arc)
    print(f"    ARC:              {auc_arc:.4f}")

    # Quintile breakdown
    sidx = np.argsort(arc)[::-1]
    qsz  = N // 5
    qb   = [0, qsz, 2*qsz, 3*qsz, 4*qsz, N]
    print(f"\n  Quintiles (Q1=easiest by ARC, equal size={qsz}):")
    print(f"    {'Q':<3} {'n':>4}  {'ARC score':>14}  "
          f"{'BFS y=1':>8}  {'GPT-4o%':>8}  {'n_obj':>8}  {'n_steps':>10}")
    print("    " + "-"*65)
    q_gpt = []
    for qi in range(5):
        idx = sidx[qb[qi]:qb[qi+1]]
        q_gpt.append(y_gpt[idx].mean())
        print(f"    Q{qi+1}  {len(idx):>4}  "
              f"[{arc[idx].min():.3f},{arc[idx].max():.3f}]  "
              f"  {y_q[idx].mean():>8.1%}  {y_gpt[idx].mean():>8.1%}"
              f"  {n_obj[idx].min():>3}-{n_obj[idx].max():<3}"
              f"  {ns_q[idx].min():.0f}-{ns_q[idx].max():.0f}")

    rho_q, p_q = stats.spearmanr([1,2,3,4,5], q_gpt)
    total_q = sum(r * qsz for r in q_gpt)
    print(f"    Spearman ρ (GPT-4o): {rho_q:+.3f}  p={p_q:.4f}")
    print(f"    Sum valid from table: {total_q:.1f}  actual: {y_gpt.sum():.0f}"
          + ("  ← MATCH" if abs(total_q-y_gpt.sum())<2 else "  ← MISMATCH"))

    # Within-budget only
    wb = ns_q <= 12
    print(f"\n  Within-budget only (n={wb.sum()}):")
    if wb.sum() > 5:
        rho1, p1 = stats.spearmanr(arc[wb], ns_q[wb])
        rho2, p2 = stats.spearmanr(n_obj[wb], ns_q[wb])
        print(f"    ρ(ARC, n_steps):       {rho1:+.3f}  p={p1:.4f}")
        print(f"    ρ(n_objects, n_steps): {rho2:+.3f}  p={p2:.4f}")
        ul = np.unique(y_q[wb])
        if len(ul) > 1:
            print(f"    AUC (within only): {roc_auc_score(y_q[wb], arc[wb]):.4f}")
        else:
            print(f"    AUC: UNDEFINED — all within-budget labels = {ul[0]:.0f}")
            print(f"    ← CONFIRMS: AUC=1.000 is purely cap-boundary detection")
    print()

print("CONCLUSION:")
print("  AUC=1.000 = ARC detects whether problem is within BFS budget.")
print("  This is learnable from n_objects alone (check auc_nobj above).")
print("  The genuine transfer metric is |ρ| on continuous n_steps,")
print("  which is NOT inflated by the cap boundary.")