"""
plan_step11_domain_id_ablation.py
===================================
Experiment 5: Ablate the domain_id feature from X_surf.

Runs three conditions:
  A. Full ARC (with domain_id at feature index N)
  B. ARC with domain_id zeroed out
  C. ARC with domain_id replaced by random noise

If performance is stable across A/B/C → domain_id is irrelevant (clean).
If B/C drop significantly vs A → potential leakage, must be investigated.

Usage:
  python plan_step11_domain_id_ablation.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import importlib.util
from sklearn.metrics import roc_auc_score, r2_score

warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/planning")
RESULTS_DIR = Path("results_planning"); RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR    = Path("checkpoints_planning")

spec = importlib.util.spec_from_file_location("step3", Path(__file__).parent / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(step3)
PlanningGURU    = step3.PlanningGURU
PlanningMetaSampler = step3.PlanningMetaSampler
train_guru      = step3.train_guru
evaluate_on_domain = step3.evaluate_on_domain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Feature index for domain_id in X_surf (see plan_step1_data_prep.py)
# features[i, 1] = domain_id hash (integer, range 0-7)
# Change this if your feature ordering is different!
DOMAIN_ID_IDX = 1


def make_ablated_surf(X_surf, mode="zero", rng_seed=42):
    """Return a copy of X_surf with domain_id modified per mode."""
    X = X_surf.copy()
    if mode == "zero":
        X[:, DOMAIN_ID_IDX] = 0.0
    elif mode == "random":
        rng = np.random.default_rng(rng_seed)
        X[:, DOMAIN_ID_IDX] = rng.uniform(0, 7, size=len(X))
    # mode == "full": return unchanged
    return X


def train_condition(X_surf_mod, X_fm, task_types, y_success, y_nsteps,
                    registry, label, n_episodes, tag):
    train_domains = registry["splits"]["meta_train"]["tasks"]
    val_domains   = registry["splits"]["meta_val"]["tasks"]
    surf_dim, fm_dim = X_surf_mod.shape[1], X_fm.shape[1]

    train_samp = PlanningMetaSampler(
        train_domains, X_surf_mod, X_fm,
        y_success.astype(int), y_nsteps, task_types, DEVICE
    )
    val_samp = PlanningMetaSampler(
        val_domains, X_surf_mod, X_fm,
        y_success.astype(int), y_nsteps, task_types, DEVICE
    )
    model = PlanningGURU(surf_dim, fm_dim).to(DEVICE)
    print(f"\n  Training condition: {tag}  (label={label})")
    train_guru(model, train_samp, n_episodes, label=label,
               device=DEVICE, val_sampler=val_samp, val_every=300)
    ckpt = CKPT_DIR / f"guru_{label}_domid_{tag}.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    print(f"  Saved → {ckpt}")
    return model


def eval_condition(model, X_surf_mod, X_fm, task_types, y, registry, label):
    test_domains  = registry["splits"]["meta_test"]["tasks"]
    train_domains = registry["splits"]["meta_train"]["tasks"]
    train_mask    = np.isin(task_types, train_domains)
    scores = {}
    for dom in test_domains:
        mask = task_types == dom
        res = evaluate_on_domain(
            model, dom,
            X_surf_mod[mask], X_fm[mask], y[mask],
            X_surf_mod[train_mask], X_fm[train_mask], y[train_mask],
            label=label, device=DEVICE, cross_domain_support=True
        )
        scores[dom] = res.get("guru_cross", {}).get("mean", float("nan")) if res else float("nan")
    return scores


def run(n_episodes=2000, label="success"):
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y_success  = np.load(DATA_DIR / "y_success.npy")
    y_nsteps   = np.load(DATA_DIR / "y_nsteps.npy")
    y = y_success if label == "success" else y_nsteps
    metric = "AUC" if label == "success" else "R²"

    test_domains = registry["splits"]["meta_test"]["tasks"]
    conditions = ["full", "zero", "random"]
    all_scores = {c: {} for c in conditions}

    for cond in conditions:
        X_mod = make_ablated_surf(X_surf, mode=cond)
        model = train_condition(X_mod, X_fm, task_types, y_success, y_nsteps,
                                registry, label, n_episodes, tag=cond)
        all_scores[cond] = eval_condition(model, X_mod, X_fm, task_types, y, registry, label)

    print(f"\n{'Domain':<25} {'Full':>8} {'Zero':>8} {'Random':>8}  Δ(full-zero)  ({metric})")
    print("-" * 70)
    results = {}
    for dom in test_domains:
        f = all_scores["full"].get(dom, float("nan"))
        z = all_scores["zero"].get(dom, float("nan"))
        r = all_scores["random"].get(dom, float("nan"))
        delta = f - z if not (np.isnan(f) or np.isnan(z)) else float("nan")
        flag = "⚠️  LEAKAGE" if abs(delta) > 0.05 else "✓  stable"
        print(f"  {dom:<23} {f:>8.4f} {z:>8.4f} {r:>8.4f}  {delta:>+.4f}  {flag}")
        results[dom] = {"full": f, "zero": z, "random": r, "delta": delta}

    out = RESULTS_DIR / f"e5_domain_id_ablation_{label}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")
    return results


if __name__ == "__main__":
    for lbl in ["success", "n_steps"]:
        run(n_episodes=2000, label=lbl)