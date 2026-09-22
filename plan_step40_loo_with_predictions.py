"""
plan_step40_loo_with_predictions.py  (v2 - corrected)
========================================================
Reproduces the paper's stated leave-one-domain-out methodology:
  - All 7 IPC domains (Depot, Rovers, Satellite, Gripper,
    Blocksworld, Logistics, Mystery-Blocksworld)
  - For each held-out domain, train on the OTHER 6
  - Report Spearman |rho| directly

CORRECTED APPROACH (v2): run_config() cannot be reused for LOO because
it internally hardcodes train_doms/val_doms from splits["meta_train"]/
["meta_val"], ignoring any train_mask passed in. evaluate_model() also
cannot be reused because it hardcodes `for dom in TEST_DOMAINS`, only
ever scoring the fixed 3-domain test split.

This script instead calls the verified lower-level functions directly:
  - PlanningMetaSampler(domain_list, ...) -- confirmed to accept any
    domain list (plan_step3_guru.py:190-220)
  - train_improved(model, sampler, n_episodes, ...) -- confirmed to
    return (model, history) (plan_step16_arc_improvements.py:376)
  - model(Q_surf, Q_fm, Q_resid, S_surf, S_fm, S_V, head=head) --
    confirmed single-instance call signature, returns (out, feats, alpha)

Saves per-instance (ARC score, object count, true n_steps) for every
held-out domain, enabling a paired bootstrap on
Delta_rho = rho_ARC - rho_|O| after the fact.

USAGE:
  python plan_step40_loo_with_predictions.py --n_episodes 5000 --seed 42

Recommended: run a smoke test with --n_episodes 200 first (see bottom
of this file for the smoke-test invocation) before the full 5000-episode
x 7-fold run.
"""
import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results_planning"; RES.mkdir(exist_ok=True)

ALL_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld",
               "depot", "rovers", "satellite", "gripper"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_module(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_support_transform(X_surf_train, X_fm_train, n_sup, seed):
    """
    Fit StandardScaler + PCA-Ridge residualizer on a random support
    subset of the TRAINING domains, matching evaluate_model()'s
    approach exactly (same n_sup default, same PCA/Ridge config).
    """
    rng = np.random.default_rng(seed)
    n_sup = min(n_sup, len(X_surf_train))
    sidx = rng.choice(len(X_surf_train), n_sup, replace=False)

    Xs_sup = X_surf_train[sidx]
    Xe_sup = X_fm_train[sidx]
    sc_p = StandardScaler().fit(Xs_sup)
    sc_e = StandardScaler().fit(Xe_sup)
    Xs_n = sc_p.transform(Xs_sup)
    Xe_n = sc_e.transform(Xe_sup)
    n_comp = max(2, min(20, n_sup // 10, Xs_n.shape[1]))
    rp = Pipeline([("pca", PCA(n_components=n_comp)), ("ridge", Ridge(alpha=1.0))])
    rp.fit(Xs_n, Xe_n)
    Xr_n = Xe_n - rp.predict(Xs_n)

    S_surf = torch.FloatTensor(Xs_n).to(DEVICE)
    S_fm   = torch.FloatTensor(Xe_n).to(DEVICE)
    S_V    = torch.FloatTensor(np.hstack([Xs_n, Xr_n])).to(DEVICE)

    return sc_p, sc_e, rp, S_surf, S_fm, S_V


def score_domain(model, sc_p, sc_e, rp, S_surf, S_fm, S_V,
                  X_surf, X_fm, y_ns, tt, target_domain, label="success"):
    """Score every instance in target_domain. Returns per-instance arrays."""
    mask = tt == target_domain
    idx = np.where(mask)[0][:200]
    Xs_q = X_surf[idx]
    Xf_q = X_fm[idx]
    ns_q = y_ns[idx].astype(float)
    n_obj = X_surf[idx, 0]

    Xs_n_q = sc_p.transform(Xs_q)
    Xe_n_q = sc_e.transform(Xf_q)
    Xr_n_q = Xe_n_q - rp.predict(Xs_n_q)

    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(len(Xs_n_q)):
            out, _, _ = model(
                torch.FloatTensor(Xs_n_q[i]).to(DEVICE),
                torch.FloatTensor(Xe_n_q[i]).to(DEVICE),
                torch.FloatTensor(Xr_n_q[i]).to(DEVICE),
                S_surf, S_fm, S_V,
                head="cls" if label == "success" else "reg",
            )
            if label == "success":
                s = float(torch.softmax(out.unsqueeze(0), -1)[0, 1].cpu())
            else:
                s = float(out.cpu())
            scores.append(s)

    return {
        "instance_id": [int(i) for i in idx],
        "arc_score": [float(s) for s in scores],
        "n_objects": [float(o) for o in n_obj],
        "n_steps":   [float(n) for n in ns_q],
    }


def run_one_fold(step3, X_surf, X_fm, y_success, y_ns, tt,
                  held_out_domain, train_domains, n_episodes, seed,
                  lambda_ent=0.05, lambda_contrast=0.1, n_sup=60):
    """Train ARC on train_domains, evaluate on held_out_domain."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    surf_dim = X_surf.shape[1]
    fm_dim = X_fm.shape[1]

    model = step3.PlanningGURU(surf_dim, fm_dim).to(DEVICE)

    sampler = step3.PlanningMetaSampler(
        train_domains, X_surf, X_fm, y_success, y_ns, tt, DEVICE)

    if len(sampler.valid_tasks) < 2:
        raise RuntimeError(
            f"Only {len(sampler.valid_tasks)} valid training domains "
            f"for held-out={held_out_domain}; need >=2")

    model, history = load_module(
        "plan_step16_arc_improvements.py", "s16"
    ).train_improved(
        model, sampler, n_episodes=n_episodes,
        label="success", lr=3e-4,
        lambda_ent=lambda_ent, lambda_contrast=lambda_contrast,
        use_contrastive=True,
        val_sampler=None, val_every=200, device=DEVICE,
    )

    train_mask = np.isin(tt, train_domains)
    sc_p, sc_e, rp, S_surf, S_fm, S_V = build_support_transform(
        X_surf[train_mask], X_fm[train_mask], n_sup=n_sup, seed=seed)

    per_instance = score_domain(
        model, sc_p, sc_e, rp, S_surf, S_fm, S_V,
        X_surf, X_fm, y_ns, tt, held_out_domain, label="success")

    return per_instance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_support", type=int, default=60)
    p.add_argument("--domains", type=str, default=None,
                   help="Comma-separated subset of domains to run "
                        "(default: all 7). Use for smoke tests, "
                        "e.g. --domains blocksworld")
    args = p.parse_args()

    step3 = load_module("plan_step3_guru.py", "step3")
    step6 = load_module("plan_step6_pddlinst_gate.py", "step6")

    X_surf, X_fm, tt, y_success, y_ns, _ = step6.load_data(
        data_dir=ROOT / "data" / "planning")

    available = sorted(set(tt.tolist()) & set(ALL_DOMAINS))
    print(f"Available domains: {available}")

    targets = available
    if args.domains:
        requested = args.domains.split(",")
        targets = [d for d in requested if d in available]
        print(f"Running smoke-test subset: {targets}")

    all_results = {}

    for held_out in targets:
        train_domains = [d for d in available if d != held_out]
        print(f"\n=== Held out: {held_out} "
              f"(training on {len(train_domains)} domains: {train_domains}) ===")

        try:
            per_instance = run_one_fold(
                step3, X_surf, X_fm, y_success, y_ns, tt,
                held_out, train_domains,
                n_episodes=args.n_episodes, seed=args.seed,
                n_sup=args.n_support)
        except Exception as e:
            import traceback
            print(f"  FOLD FAILED: {e}")
            traceback.print_exc()
            continue

        arc_scores = np.array(per_instance["arc_score"])
        n_objects  = np.array(per_instance["n_objects"])
        n_steps    = np.array(per_instance["n_steps"])
        valid = n_steps > 0

        if valid.sum() < 5:
            print(f"  Only {valid.sum()} labeled instances -- skipping correlation")
            continue

        rho_arc, _ = stats.spearmanr(arc_scores[valid], n_steps[valid])
        rho_obj, _ = stats.spearmanr(n_objects[valid], n_steps[valid])

        print(f"  N={valid.sum()}  ARC |rho|={abs(rho_arc):.3f}  "
              f"|O| |rho|={abs(rho_obj):.3f}")

        all_results[held_out] = {
            "train_domains": train_domains,
            "n_instances": int(valid.sum()),
            "rho_arc": float(rho_arc),
            "rho_obj": float(rho_obj),
            "per_instance": per_instance,
        }

    if all_results:
        rhos_arc = [abs(v["rho_arc"]) for v in all_results.values()]
        rhos_obj = [abs(v["rho_obj"]) for v in all_results.values()]
        print(f"\n=== SUMMARY ({len(all_results)} domains) ===")
        print(f"ARC: mean={np.mean(rhos_arc):.3f} std={np.std(rhos_arc):.3f}")
        print(f"|O|: mean={np.mean(rhos_obj):.3f} std={np.std(rhos_obj):.3f}")

        all_results["_summary"] = {
            "n_domains": len(rhos_arc),
            "arc_mean": float(np.mean(rhos_arc)), "arc_std": float(np.std(rhos_arc)),
            "obj_mean": float(np.mean(rhos_obj)), "obj_std": float(np.std(rhos_obj)),
        }

    out_path = RES / "loo_with_predictions.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved (with per-instance predictions) -> {out_path}")


if __name__ == "__main__":
    main()

# SMOKE TEST (run this first, ~1-2 min on CPU with 200 episodes):
#   python plan_step40_loo_with_predictions.py --n_episodes 200 --domains blocksworld
# If that produces a sane rho_arc (not NaN, not exactly 0), scale up:
#   python plan_step40_loo_with_predictions.py --n_episodes 5000
# Expect ~7x single-model training time (each fold trains from scratch).
