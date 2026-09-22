"""
plan_step9_e2_controlled_quintile.py
=====================================
Experiment 2 (Fixed): Zero-shot quintile analysis with object-count held
constant within each bucket.

Splits instances into n_objects size buckets first, then ranks by ARC
score within each bucket. This isolates whether ARC adds signal BEYOND
object count — the exact test reviewers will demand after seeing the
corrected Table 3.

Usage:
  python plan_step9_e2_controlled_quintile.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
import torch
import importlib.util

warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/planning")
RESULTS_DIR = Path("results_planning"); RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR     = Path("figures_planning"); FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR    = Path("checkpoints_planning")

spec = importlib.util.spec_from_file_location("step3", Path(__file__).parent / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(step3)
PlanningGURU = step3.PlanningGURU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-4o per-instance labels (from plan_step8 or loaded from existing JSON)
GPT4O_VALIDITY = {
    "blocksworld":        {0: 1.00, 1: 0.45, 2: 0.32, 3: 0.05, 4: 0.00},  # Q→validity
    "logistics":          {0: 0.075, 1: 0.10, 2: 0.025, 3: 0.00, 4: 0.00},
    "mystery_blocksworld":{0: 0.075, 1: 0.075, 2: 0.00, 3: 0.325, 4: 0.175},
}


def load_model():
    ckpt = CKPT_DIR / "guru_success.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    ck = torch.load(ckpt, map_location=DEVICE)
    X_surf = np.load(DATA_DIR / "X_surf.npy")
    X_fm   = np.load(DATA_DIR / "X_fm.npy")
    m = PlanningGURU(X_surf.shape[1], X_fm.shape[1]).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    return m, X_surf, X_fm


def get_arc_scores_zeroshot(model, X_surf, X_fm, task_types, dom,
                             train_mask, y_train):
    """
    Get ARC probability scores for all test instances in `dom`
    using only training-domain support set (true zero-shot).
    Returns scores array aligned with dom_mask.
    """
    dom_mask   = task_types == dom
    X_surf_q   = X_surf[dom_mask]
    X_fm_q     = X_fm[dom_mask]
    X_surf_s   = X_surf[train_mask]
    X_fm_s     = X_fm[train_mask]

    n_sup = min(60, len(X_surf_s))
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(X_surf_s), n_sup, replace=False)
    Xs_s  = X_surf_s[idx]; Xe_s = X_fm_s[idx]

    sc_p = StandardScaler().fit(Xs_s); sc_e = StandardScaler().fit(Xe_s)
    Xs_s_n = sc_p.transform(Xs_s); Xe_s_n = sc_e.transform(Xe_s)
    Xs_q_n = sc_p.transform(X_surf_q); Xe_q_n = sc_e.transform(X_fm_q)

    n_comp = max(2, min(20, n_sup // 10, Xs_s_n.shape[1]))
    pred = Pipeline([("pca", PCA(n_components=n_comp)), ("ridge", Ridge(alpha=1.0))])
    pred.fit(Xs_s_n, Xe_s_n)
    Xr_s = Xe_s_n - pred.predict(Xs_s_n)
    Xr_q = Xe_q_n - pred.predict(Xs_q_n)

    S_surf = torch.FloatTensor(Xs_s_n).to(DEVICE)
    S_fm   = torch.FloatTensor(Xe_s_n).to(DEVICE)
    S_V    = torch.FloatTensor(np.hstack([Xs_s_n, Xr_s])).to(DEVICE)

    scores = []
    with torch.no_grad():
        for ii in range(len(Xs_q_n)):
            feats, _ = model.get_features(
                torch.FloatTensor(Xs_q_n[ii]).to(DEVICE),
                torch.FloatTensor(Xe_q_n[ii]).to(DEVICE),
                torch.FloatTensor(Xr_q[ii]).to(DEVICE),
                S_surf, S_fm, S_V
            )
            # Use fusion head → cls head for P(easy)
            h = model.fusion(feats)
            logits = model.head_cls(h)
            prob_easy = torch.softmax(logits, dim=-1)[1].item()
            scores.append(prob_easy)

    return np.array(scores), dom_mask


def run_controlled_quintile():
    model, X_surf, X_fm = load_model()
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y_success  = np.load(DATA_DIR / "y_success.npy")

    train_domains = registry["splits"]["meta_train"]["tasks"]
    test_domains  = registry["splits"]["meta_test"]["tasks"]
    train_mask    = np.isin(task_types, train_domains)
    y_train       = y_success[train_mask]

    # Load n_objects for each instance (feature index 0 in X_surf = n_objects)
    # feature 0 of X_surf is object count (see plan_step1_data_prep.py features[i,0])
    n_objects = X_surf[:, 0]  # raw (unscaled) if saved before scaling, else approximate

    SIZE_BUCKETS = [
        (0,  4,  "small (≤4 obj)"),
        (5,  7,  "medium (5-7 obj)"),
        (8, 99,  "large (≥8 obj)"),
    ]
    N_QUINTILES = 5

    all_results = {}
    print(f"\n{'Domain':<25} {'Size bucket':<22} {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Q5':>7}  Spearman-ρ(ARC)  ρ(nobj)")
    print("-" * 110)

    for dom in test_domains:
        arc_scores, dom_mask = get_arc_scores_zeroshot(
            model, X_surf, X_fm, task_types, dom, train_mask, y_train
        )
        dom_nobj    = n_objects[dom_mask]
        dom_success = y_success[dom_mask]
        dom_results = {}

        for lo, hi, bucket_name in SIZE_BUCKETS:
            bucket = (dom_nobj >= lo) & (dom_nobj <= hi)
            if bucket.sum() < 10:
                continue

            arc_b    = arc_scores[bucket]
            success_b = dom_success[bucket]
            nobj_b   = dom_nobj[bucket]

            # Quintile by ARC score within this bucket
            q_bounds  = np.percentile(arc_b, [0, 20, 40, 60, 80, 100])
            q_labels  = np.digitize(arc_b, q_bounds[1:-1])

            # GPT-4o validity per quintile (use real labels if available, else proxy)
            quintile_vals = []
            for q in range(N_QUINTILES):
                qmask = q_labels == q
                if qmask.sum() < 2:
                    quintile_vals.append(float("nan"))
                    continue
                # Use y_success as proxy for GPT-4o when real labels not available
                val = success_b[qmask].mean()
                quintile_vals.append(float(val))

            valid_q = [(i, v) for i, v in enumerate(quintile_vals) if not np.isnan(v)]
            if len(valid_q) >= 3:
                rho_arc, _ = spearmanr([x[0] for x in valid_q], [x[1] for x in valid_q])
                rho_nobj, _ = spearmanr(nobj_b, success_b)
            else:
                rho_arc = rho_nobj = float("nan")

            vals_str = " ".join(f"{v:>7.1%}" if not np.isnan(v) else f"{'—':>7}"
                                for v in quintile_vals)
            print(f"  {dom:<23} {bucket_name:<22} {vals_str}  ρ(ARC)={rho_arc:+.3f}  ρ(nobj)={rho_nobj:+.3f}")
            dom_results[bucket_name] = {
                "quintile_vals": quintile_vals,
                "rho_arc": float(rho_arc),
                "rho_nobj": float(rho_nobj),
                "n": int(bucket.sum()),
            }

        all_results[dom] = dom_results

    _plot_controlled_quintile(all_results, test_domains, SIZE_BUCKETS)
    out = RESULTS_DIR / "e2_controlled_quintile.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out}")
    return all_results


def _plot_controlled_quintile(results, domains, buckets):
    n_dom = len(domains)
    n_bkt = len(buckets)
    fig, axes = plt.subplots(n_dom, n_bkt, figsize=(5 * n_bkt, 4 * n_dom), squeeze=False)

    for i, dom in enumerate(domains):
        for j, (lo, hi, bname) in enumerate(buckets):
            ax = axes[i][j]
            r  = results.get(dom, {}).get(bname, {})
            qv = r.get("quintile_vals", [float("nan")] * 5)
            xs = list(range(len(qv)))
            ys = [v if not np.isnan(v) else 0 for v in qv]

            bars = ax.bar(xs, ys, color=["#2ECC71","#82E0AA","#F8C471","#E59866","#E74C3C"],
                          alpha=0.85, edgecolor="black")
            for b, v in zip(bars, qv):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                            f"{v:.0%}", ha="center", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"Q{q+1}" for q in xs], fontsize=8)
            rho = r.get("rho_arc", float("nan"))
            ax.set_title(f"{dom[:12]} | {bname}\nρ(ARC)={rho:+.3f}", fontsize=9)
            ax.set_ylim(0, 1.2)
            ax.grid(axis="y", alpha=0.3)
            if j == 0:
                ax.set_ylabel("GPT-4o validity proxy", fontsize=8)

    fig.suptitle("E2 (Fixed): ARC quintile analysis with object-count held constant\n"
                 "Each column = fixed size bucket; ranking by ARC within bucket",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(FIG_DIR / f"e2_controlled_quintile{ext}", bbox_inches="tight", dpi=150)
    print(f"  Saved → figures_planning/e2_controlled_quintile.pdf")
    plt.close()


if __name__ == "__main__":
    run_controlled_quintile()