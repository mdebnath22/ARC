"""
plan_step12_calibration.py
===========================
Experiment 7: Reliability diagrams and Expected Calibration Error (ECE).

Plots predicted P(easy) vs empirical GPT-4o success rate per domain.
A well-calibrated ARC supports cost-aware routing decisions.

Usage:
  python plan_step12_calibration.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
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

# Real GPT-4o per-instance success labels
# If plan_step8_llm_labels.py has been run, load from file; else use BFS labels as proxy
def load_eval_labels():
    llm_path = DATA_DIR / "y_llm.npy"
    if llm_path.exists():
        print("  Using real GPT-4o labels (y_llm.npy)")
        return np.load(llm_path)
    print("  WARNING: No y_llm.npy found — using BFS labels as proxy")
    return np.load(DATA_DIR / "y_success.npy")


def get_arc_probs(model, X_surf, X_fm, task_types, dom, train_mask):
    """Returns (probs, y_true) for all instances in dom."""
    dom_mask = task_types == dom
    Xs_q = X_surf[dom_mask]; Xe_q = X_fm[dom_mask]
    Xs_s = X_surf[train_mask][:60]; Xe_s = X_fm[train_mask][:60]

    sc_p = StandardScaler().fit(Xs_s); sc_e = StandardScaler().fit(Xe_s)
    Xs_s_n = sc_p.transform(Xs_s); Xe_s_n = sc_e.transform(Xe_s)
    Xs_q_n = sc_p.transform(Xs_q); Xe_q_n = sc_e.transform(Xe_q)

    n_comp = max(2, min(20, 60 // 10, Xs_s_n.shape[1]))
    pred = Pipeline([("pca", PCA(n_comp)), ("ridge", Ridge(alpha=1.0))])
    pred.fit(Xs_s_n, Xe_s_n)
    Xr_s = Xe_s_n - pred.predict(Xs_s_n)
    Xr_q = Xe_q_n - pred.predict(Xs_q_n)

    S_surf = torch.FloatTensor(Xs_s_n).to(DEVICE)
    S_fm   = torch.FloatTensor(Xe_s_n).to(DEVICE)
    S_V    = torch.FloatTensor(np.hstack([Xs_s_n, Xr_s])).to(DEVICE)

    probs = []
    with torch.no_grad():
        for ii in range(len(Xs_q_n)):
            feats, _ = model.get_features(
                torch.FloatTensor(Xs_q_n[ii]).to(DEVICE),
                torch.FloatTensor(Xe_q_n[ii]).to(DEVICE),
                torch.FloatTensor(Xr_q[ii]).to(DEVICE),
                S_surf, S_fm, S_V
            )
            h = model.fusion(feats)
            logit = model.head_cls(h)
            p = torch.softmax(logit, dim=-1)[1].item()
            probs.append(p)

    return np.array(probs), dom_mask


def ece_score(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(acc - conf)
    return float(ece)


def run_calibration():
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y_eval     = load_eval_labels()

    test_domains  = registry["splits"]["meta_test"]["tasks"]
    train_domains = registry["splits"]["meta_train"]["tasks"]
    train_mask    = np.isin(task_types, train_domains)

    ckpt = CKPT_DIR / "guru_success.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    ck = torch.load(ckpt, map_location=DEVICE)
    model = PlanningGURU(X_surf.shape[1], X_fm.shape[1]).to(DEVICE)
    model.load_state_dict(ck["model"]); model.eval()

    n_bins = 10
    fig, axes = plt.subplots(1, len(test_domains), figsize=(5 * len(test_domains), 5))
    if len(test_domains) == 1:
        axes = [axes]

    results = {}
    print(f"\n{'Domain':<25} {'ECE':>8}  Calibration quality")
    print("-" * 55)

    for ax, dom in zip(axes, test_domains):
        probs, dom_mask = get_arc_probs(model, X_surf, X_fm, task_types, dom, train_mask)
        y_true = y_eval[dom_mask].astype(int)

        # Calibration curve
        try:
            frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
        except ValueError:
            frac_pos = mean_pred = np.array([float("nan")])

        ece = ece_score(probs, y_true, n_bins=n_bins)
        quality = "well-calibrated" if ece < 0.10 else "moderate" if ece < 0.20 else "poorly calibrated"
        print(f"  {dom:<23} {ece:>8.4f}  {quality}")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", color="#E74C3C", lw=2,
                label=f"ARC  ECE={ece:.3f}")

        # Histogram of predicted probabilities
        ax2 = ax.twinx()
        ax2.hist(probs, bins=10, alpha=0.2, color="#3498DB")
        ax2.set_ylabel("Count", fontsize=8, color="#3498DB")

        ax.set_xlabel("Mean predicted P(easy)", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.set_title(f"{dom}\nECE = {ece:.4f}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        results[dom] = {"ece": ece, "frac_pos": frac_pos.tolist(),
                        "mean_pred": mean_pred.tolist()}

    fig.suptitle("E7: ARC Reliability Diagrams\n"
                 "A well-calibrated model supports cost-aware routing (e.g., route when P > 0.7)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(FIG_DIR / f"e7_calibration{ext}", bbox_inches="tight", dpi=150)
    print(f"\n  Saved → figures_planning/e7_calibration.pdf")
    plt.close()

    (RESULTS_DIR / "e7_calibration.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_calibration()