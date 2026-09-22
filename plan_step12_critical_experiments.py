"""
plan_step12_critical_experiments.py
=====================================
Three experiments that directly address the top reviewer attacks.

EXP A — Fixed-object-count subset  (addresses: "ARC just learns size")
  Create subsets where all instances have the SAME n_objects.
  Object-count routing is constant → useless.
  ARC must use structural features beyond size.
  Show ARC still ranks difficulty; object-count collapses.

EXP B — MLP non-episodic baseline  (addresses: "meta-learning not necessary")
  Train a deep MLP on [x_s | x_r] with the same capacity as ARC's
  fusion network, but NO episodic training / support-set mechanism.
  Compare cross-domain R² and AUC vs ARC.
  Isolates the episodic meta-learning as the key contribution.

EXP C — Attention visualization  (addresses: "attention is not interpretable")
  For each test domain, take 5 query instances (easy/medium/hard).
  Show: which support instances get highest attention weight.
  Measure: correlation between attention weight and structural similarity.
  Show: top-attended instances match the query's difficulty level.

USAGE:
  python plan_step12_critical_experiments.py --exp A
  python plan_step12_critical_experiments.py --exp B
  python plan_step12_critical_experiments.py --exp C
  python plan_step12_critical_experiments.py --exp all
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import torch.nn as nn

warnings.filterwarnings("ignore")

ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data" / "planning"
RESULTS_DIR = ROOT_DIR / "results_planning"; RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR     = ROOT_DIR / "figures_planning";  FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR    = ROOT_DIR / "checkpoints_planning"

DOMAIN_LABELS = {
    "blocksworld":         "Blocksworld",
    "logistics":           "Logistics",
    "mystery_blocksworld": "Mystery-BW",
}
COLORS = {
    "blocksworld":         "#2980B9",
    "logistics":           "#27AE60",
    "mystery_blocksworld": "#8E44AD",
}
TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Shared data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_step6():
    spec = importlib.util.spec_from_file_location(
        "step6", ROOT_DIR / "plan_step6_pddlinst_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_step3():
    spec = importlib.util.spec_from_file_location(
        "step3", ROOT_DIR / "plan_step3_guru.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_arc_model(label="success"):
    ckpt = CKPT_DIR / f"guru_{label}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing: {ckpt}")
    step3 = load_step3()
    PlanningGURU = step3.PlanningGURU
    ck    = torch.load(ckpt, map_location=DEVICE)
    state = ck["model"]
    surf_dim = state["key_enc.net.0.weight"].shape[1]
    fm_dim   = state["query_enc.net.0.weight"].shape[1]
    model    = PlanningGURU(surf_dim, fm_dim).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EXP A — Fixed-object-count subset
# ══════════════════════════════════════════════════════════════════════════════

def exp_a_fixed_nobj(args):
    """
    Critical experiment: "ARC is not just learning object count."

    Protocol:
      For each test domain, find the most common n_objects value
      (or median) and select ONLY instances with that n_objects.
      Within this subset:
        - Object-count routing → CONSTANT (all instances same size)
        - ARC routing → still varies if ARC learned structural features

      Measure:
        - Spearman ρ(ARC_score, n_steps) within fixed-size subset
        - Spearman ρ(n_objects, n_steps) within fixed-size subset = 0
        - AUC(ARC) within fixed-size subset vs chance (0.5)

      The key result: ARC achieves ρ > 0 (statistically significant)
      even when object count is constant, proving it learned structural
      features beyond size.
    """
    print("\n" + "=" * 65)
    print("EXP A — Fixed-object-count subset")
    print("  Object-count routing = constant → ARC must use structure")
    print("=" * 65)

    step6 = load_step6()
    X_surf, X_fm, task_types, y_success, y_steps, splits = step6.load_data(
        data_dir=DATA_DIR)
    train_domains = splits["meta_train"]["domains"]
    train_mask    = np.isin(task_types, train_domains)
    X_surf_s      = X_surf[train_mask]
    X_fm_s        = X_fm[train_mask]

    model = load_arc_model("success")

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "EXP A: ARC on Fixed-Object-Count Subsets\n"
        "Object-count routing is constant; ARC must use structural features",
        fontsize=11, fontweight="bold")

    print(f"\n  {'Domain':<22}  {'Target n_obj':>12}  {'n instances':>12}  "
          f"{'ρ(ARC,steps)':>13}  {'p-value':>9}  {'AUC(ARC)':>9}")
    print("  " + "-" * 82)

    for ax, dom in zip(axes, TEST_DOMAINS):
        mask   = task_types == dom
        Xs_q   = X_surf[mask]
        Xf_q   = X_fm[mask]
        y_q    = y_success[mask].astype(float)
        ns_q   = y_steps[mask]
        n_obj  = Xs_q[:, 0].astype(int)   # feature 0 = n_objects

        # Find the most common n_objects value with enough instances
        unique_vals, counts = np.unique(n_obj, return_counts=True)
        # Pick value with most instances (need at least 20 for stats)
        viable = [(v, c) for v, c in zip(unique_vals, counts) if c >= 20]
        if not viable:
            viable = [(unique_vals[np.argmax(counts)], counts.max())]
        target_nobj = max(viable, key=lambda x: x[1])[0]

        subset_mask = n_obj == target_nobj
        n_subset    = subset_mask.sum()

        Xs_sub = Xs_q[subset_mask]
        Xf_sub = Xf_q[subset_mask]
        y_sub  = y_q[subset_mask]
        ns_sub = ns_q[subset_mask]

        # Get ARC scores for the FULL domain first (for support set consistency)
        # then filter to subset
        arc_scores_full = step6.get_guru_scores_per_instance(
            model, Xs_q, Xf_q, y_q,
            X_surf_s, X_fm_s,
            n_runs=args.n_runs, rng_seed=args.seed)
        arc_scores_sub = arc_scores_full[subset_mask]

        # n_objects within subset = constant = target_nobj → ρ=0 by construction
        # ARC score within subset
        if len(np.unique(ns_sub)) < 2 or n_subset < 5:
            print(f"  {dom:<22}  n_obj={target_nobj}  "
                  f"n={n_subset}  (insufficient variation — skip)")
            continue

        rho_arc, p_arc = stats.spearmanr(arc_scores_sub, ns_sub)
        rho_nobj_full, _ = stats.spearmanr(n_obj, ns_q)   # full domain baseline

        # AUC within fixed-size subset
        if len(np.unique(y_sub)) > 1:
            auc_arc = float(roc_auc_score(y_sub, arc_scores_sub))
            auc_chance = 0.5
        else:
            auc_arc = float("nan")

        sig_flag = "***" if p_arc < 0.001 else "**" if p_arc < 0.01 else "*" if p_arc < 0.05 else ""
        print(f"  {dom:<22}  {target_nobj:>12}  {n_subset:>12}  "
              f"{rho_arc:>+13.3f}{sig_flag}  {p_arc:>9.4f}  {auc_arc:>9.3f}")

        # Plot: ARC score vs n_steps within fixed-size subset
        ax.scatter(arc_scores_sub, ns_sub,
                   c=[COLORS[dom]] * n_subset,
                   alpha=0.6, s=40, edgecolors="white", linewidths=0.5)

        # Add trend line
        if n_subset >= 5:
            z_fit = np.polyfit(arc_scores_sub, ns_sub, 1)
            p_fit = np.poly1d(z_fit)
            x_line = np.linspace(arc_scores_sub.min(), arc_scores_sub.max(), 50)
            ax.plot(x_line, p_fit(x_line), "-", color=COLORS[dom],
                    lw=2, alpha=0.8)

        ax.set_title(f"{DOMAIN_LABELS[dom]}\n"
                     f"n_objects fixed = {target_nobj} (n={n_subset})\n"
                     f"ρ(ARC, n_steps) = {rho_arc:+.3f} {sig_flag}",
                     fontsize=9)
        ax.set_xlabel("ARC P(easy) score", fontsize=9)
        ax.set_ylabel("BFS solution length", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate: n_obj routing is constant
        ax.axhline(ns_sub.mean(), color="gray", ls="--", lw=1, alpha=0.5,
                   label=f"n_obj routing\n(constant at {target_nobj})")
        ax.legend(fontsize=7)

        results[dom] = {
            "target_nobj":   int(target_nobj),
            "n_subset":      int(n_subset),
            "rho_arc":       float(rho_arc),
            "p_arc":         float(p_arc),
            "auc_arc":       float(auc_arc) if not np.isnan(auc_arc) else None,
            "rho_nobj_full": float(rho_nobj_full),
            "significant":   bool(p_arc < 0.05),
            "interpretation": (
                f"ARC achieves ρ={rho_arc:+.3f} (p={p_arc:.4f}) within instances "
                f"of fixed size n_obj={target_nobj}, where object-count routing "
                f"is constant. ARC discriminates difficulty using structural features "
                f"beyond problem size."
                if p_arc < 0.05 else
                f"ARC does not significantly outperform chance within fixed-size "
                f"subset (n_obj={target_nobj}, n={n_subset}). "
                f"This domain's difficulty may be sufficiently captured by size alone."
            ),
        }

    plt.tight_layout()
    plt.savefig(FIG_DIR / "expA_fixed_nobj.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(FIG_DIR / "expA_fixed_nobj.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Figure → {FIG_DIR}/expA_fixed_nobj.pdf")

    print("\n  INTERPRETATION:")
    for dom, r in results.items():
        print(f"    {DOMAIN_LABELS[dom]}: {r['interpretation'][:90]}")

    out = RESULTS_DIR / "expA_fixed_nobj.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {out}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXP B — MLP non-episodic deep baseline
# ══════════════════════════════════════════════════════════════════════════════

class DeepMLP(nn.Module):
    """
    Deep MLP baseline with same input features as ARC's fusion head
    but NO episodic training / support-set mechanism.

    Input: [x_s | x_r] = [30 | 768] = 798 dims
    Architecture: mirrors ARC's fusion head capacity.
    Trained directly on meta-train instances, evaluated on meta-test.

    This baseline isolates the episodic meta-learning contribution:
    if DeepMLP fails cross-domain, it's because episodic training is
    necessary, not because ARC has more capacity or features.
    """
    def __init__(self, in_dim=798, head="cls"):
        super().__init__()
        self.head_type = head
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),   nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05),
        )
        if head == "cls":
            self.out = nn.Linear(128, 2)
        else:
            self.out = nn.Linear(128, 1)

    def forward(self, x):
        return self.out(self.net(x))


def compute_residuals(X_surf_tr, X_fm_tr, X_surf_te, X_fm_te):
    """Compute FM residuals using the same Ridge pipeline as ARC."""
    n_comp = max(2, min(20, X_surf_tr.shape[0] // 10, X_surf_tr.shape[1]))
    sc_s   = StandardScaler().fit(X_surf_tr)
    sc_e   = StandardScaler().fit(X_fm_tr)
    Xs_tr_n = sc_s.transform(X_surf_tr)
    Xe_tr_n = sc_e.transform(X_fm_tr)
    Xs_te_n = sc_s.transform(X_surf_te)
    Xe_te_n = sc_e.transform(X_fm_te)
    pipe = Pipeline([("pca", PCA(n_components=n_comp)),
                     ("ridge", Ridge(alpha=1.0))])
    pipe.fit(Xs_tr_n, Xe_tr_n)
    Xr_tr = Xe_tr_n - pipe.predict(Xs_tr_n)
    Xr_te = Xe_te_n - pipe.predict(Xs_te_n)
    return (Xs_tr_n, Xr_tr), (Xs_te_n, Xr_te)


def train_mlp(X_tr, y_tr, head="cls", n_epochs=200, lr=3e-4, seed=42):
    """Train DeepMLP on training-domain instances."""
    torch.manual_seed(seed)
    in_dim = X_tr.shape[1]
    model  = DeepMLP(in_dim=in_dim, head=head).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=1e-5)
    X_t    = torch.FloatTensor(X_tr).to(DEVICE)

    if head == "cls":
        y_t  = torch.LongTensor(y_tr.astype(int)).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_t  = torch.FloatTensor(y_tr.astype(float)).to(DEVICE)
        loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        out  = model(X_t)
        loss = loss_fn(out.squeeze(), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    return model


def eval_mlp(model, X_te, y_te, head="cls"):
    """Evaluate MLP on test instances."""
    model.eval()
    X_t  = torch.FloatTensor(X_te).to(DEVICE)
    with torch.no_grad():
        out = model(X_t)
    if head == "cls":
        proba  = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
        try:
            auc = float(roc_auc_score(y_te, proba))
        except Exception:
            auc = float("nan")
        return proba, auc
    else:
        preds = out.squeeze().cpu().numpy()
        rho, _ = stats.spearmanr(preds, y_te)
        return preds, float(rho)


def exp_b_mlp_baseline(args):
    """
    EXP B: Deep MLP non-episodic baseline.

    Key design: MLP has SAME input features as ARC ([x_s | x_r])
    and SAME capacity (512→256→128). Only difference: no support set,
    no episodic training. Trained once on all meta-train instances.

    If MLP fails cross-domain → episodic training is necessary.
    If MLP succeeds → ARC's meta-learning is not the key factor.
    """
    print("\n" + "=" * 65)
    print("EXP B — Deep MLP Non-Episodic Baseline")
    print("  Same features [x_s|x_r], same capacity, no episodic training")
    print("=" * 65)

    step6 = load_step6()
    X_surf, X_fm, task_types, y_success, y_steps, splits = step6.load_data(
        data_dir=DATA_DIR)

    train_domains = splits["meta_train"]["domains"]
    test_domains  = splits["meta_test"]["domains"]
    train_mask    = np.isin(task_types, train_domains)

    X_surf_tr = X_surf[train_mask]
    X_fm_tr   = X_fm[train_mask]
    y_cls_tr  = y_success[train_mask]
    y_reg_tr  = y_steps[train_mask]

    # Also get ARC scores for comparison on same test instances
    model_arc = load_arc_model("success")
    X_surf_s  = X_surf_tr
    X_fm_s    = X_fm_tr

    results = {}

    print(f"\n  {'Domain':<22}  {'Method':<20}  {'AUC':>7}  {'|ρ| n_steps':>12}  {'Note':>20}")
    print("  " + "-" * 82)

    for dom in test_domains:
        mask   = task_types == dom
        Xs_te  = X_surf[mask]
        Xf_te  = X_fm[mask]
        y_cls_te = y_success[mask].astype(float)
        y_reg_te = y_steps[mask].astype(float)

        # Compute residuals (same pipeline as ARC)
        (Xs_tr_n, Xr_tr), (Xs_te_n, Xr_te) = compute_residuals(
            X_surf_tr, X_fm_tr, Xs_te, Xf_te)

        X_mlp_tr = np.hstack([Xs_tr_n, Xr_tr])
        X_mlp_te = np.hstack([Xs_te_n, Xr_te])

        # Train MLP for classification
        print(f"  {dom:<22}  Training MLP...", end=" ", flush=True)
        mlp_cls = train_mlp(X_mlp_tr, y_cls_tr, head="cls",
                             n_epochs=300, seed=args.seed)
        proba_mlp, auc_mlp = eval_mlp(mlp_cls, X_mlp_te, y_cls_te, head="cls")

        # Train MLP for regression
        mlp_reg = train_mlp(X_mlp_tr, y_reg_tr, head="reg",
                             n_epochs=300, seed=args.seed)
        preds_mlp, rho_mlp = eval_mlp(mlp_reg, X_mlp_te, y_reg_te, head="reg")
        print(f"done")

        # ARC scores on same instances
        arc_scores = step6.get_guru_scores_per_instance(
            model_arc, Xs_te, Xf_te, y_cls_te,
            X_surf_s, X_fm_s,
            n_runs=args.n_runs, rng_seed=args.seed)

        try:
            auc_arc = float(roc_auc_score(y_cls_te, arc_scores))
        except Exception:
            auc_arc = float("nan")
        rho_arc, _ = stats.spearmanr(arc_scores, y_reg_te)

        # Object-count baseline
        n_obj = Xs_te[:, 0]
        try:
            auc_nobj = float(roc_auc_score(y_cls_te, -n_obj))
        except Exception:
            auc_nobj = float("nan")
        rho_nobj, _ = stats.spearmanr(n_obj, y_reg_te)

        for method, auc, rho, note in [
            ("ARC (episodic)",     auc_arc,  abs(rho_arc),  "← target"),
            ("DeepMLP (no epis.)", auc_mlp,  abs(rho_mlp),  "← baseline"),
            ("n_objects",          auc_nobj, abs(rho_nobj), "← heuristic"),
        ]:
            print(f"  {dom if method=='ARC (episodic)' else '':<22}  "
                  f"{method:<20}  {auc:>7.3f}  {rho:>12.3f}  {note:>20}")

        gap_auc = auc_arc - auc_mlp
        gap_rho = abs(rho_arc) - abs(rho_mlp)

        results[dom] = {
            "arc":  {"auc": float(auc_arc),  "rho_nsteps": float(rho_arc)},
            "mlp":  {"auc": float(auc_mlp),  "rho_nsteps": float(rho_mlp)},
            "nobj": {"auc": float(auc_nobj), "rho_nsteps": float(rho_nobj)},
            "gap_auc_arc_minus_mlp": float(gap_auc),
            "gap_rho_arc_minus_mlp": float(gap_rho),
            "interpretation": (
                f"ARC (episodic) outperforms DeepMLP by ΔAUC={gap_auc:+.3f}, "
                f"Δρ={gap_rho:+.3f}. Episodic meta-learning is necessary."
                if gap_auc > 0.01 or gap_rho > 0.05 else
                f"DeepMLP matches ARC (ΔAUC={gap_auc:+.3f}). "
                f"Episodic training advantage is small on this domain."
            ),
        }
        print()

    # Summary
    print("  SUMMARY:")
    print(f"  {'Domain':<22}  {'ΔAUC (ARC-MLP)':>15}  {'Δρ (ARC-MLP)':>13}  "
          f"{'Verdict':>25}")
    print("  " + "-" * 78)
    for dom, r in results.items():
        dA  = r["gap_auc_arc_minus_mlp"]
        drho = r["gap_rho_arc_minus_mlp"]
        verdict = "Episodic ✓" if dA > 0.01 or drho > 0.05 else "Inconclusive"
        print(f"  {dom:<22}  {dA:>+15.3f}  {drho:>+13.3f}  {verdict:>25}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("EXP B: ARC vs Deep MLP (non-episodic) Baseline\n"
                 "Same features [x_s|x_r], same capacity — only training differs",
                 fontsize=11, fontweight="bold")

    dom_lbls = [DOMAIN_LABELS[d] for d in test_domains if d in results]
    x = np.arange(len(dom_lbls))
    w = 0.25
    colors3 = ["#2980B9", "#E74C3C", "#95A5A6"]

    for ax, metric, metric_name in [
        (axes[0], "auc",        "AUC (plan success)"),
        (axes[1], "rho_nsteps", "|ρ| (solution length)"),
    ]:
        for j, (method_key, lbl, col) in enumerate([
            ("arc",  "ARC (episodic)", "#2980B9"),
            ("mlp",  "DeepMLP",        "#E74C3C"),
            ("nobj", "n_objects",      "#95A5A6"),
        ]):
            vals = [abs(results[d][method_key][metric])
                    for d in test_domains if d in results]
            bars = ax.bar(x + (j-1)*w, vals, w, label=lbl,
                          color=col, edgecolor="white", alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

        ax.set_xticks(x); ax.set_xticklabels(dom_lbls, fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "expB_mlp_baseline.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(FIG_DIR / "expB_mlp_baseline.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Figure → {FIG_DIR}/expB_mlp_baseline.pdf")

    out = RESULTS_DIR / "expB_mlp_baseline.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {out}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXP C — Attention visualization
# ══════════════════════════════════════════════════════════════════════════════

def exp_c_attention_viz(args):
    """
    EXP C: Attention visualization.

    For each test domain, select 6 query instances (2 easy, 2 medium, 2 hard
    by BFS n_steps). For each query, show:
      1. The top-5 attended support instances (highest α_i)
      2. Their domain, n_objects, n_steps
      3. Spearman ρ between attention weights and structural similarity

    Structural similarity between query and support instance:
      sim(q, s) = 1 - |n_steps(q) - n_steps(s)| / max_n_steps
      (difficulty-matched similarity — harder is more similar to harder)

    Key claim to validate:
      Hard query instances → attend to hard support instances
      Easy query instances → attend to easy support instances
      ρ(attention_weight, similarity) > 0 and significant
    """
    print("\n" + "=" * 65)
    print("EXP C — Attention Visualization")
    print("  Shows ARC attends to structurally similar support instances")
    print("=" * 65)

    step6 = load_step6()
    X_surf, X_fm, task_types, y_success, y_steps, splits = step6.load_data(
        data_dir=DATA_DIR)

    train_domains = splits["meta_train"]["domains"]
    test_domains  = splits["meta_test"]["domains"]
    train_mask    = np.isin(task_types, train_domains)

    X_surf_s = X_surf[train_mask]
    X_fm_s   = X_fm[train_mask]
    ns_s     = y_steps[train_mask]
    tt_s     = task_types[train_mask]

    model = load_arc_model("success")

    # Prepare fixed support set (same across all queries for consistency)
    rng     = np.random.default_rng(args.seed)
    n_sup   = min(60, len(X_surf_s))
    sup_idx = rng.choice(len(X_surf_s), n_sup, replace=False)

    sc_p_sup = StandardScaler().fit(X_surf_s[sup_idx])
    sc_e_sup = StandardScaler().fit(X_fm_s[sup_idx])
    Xs_sup_n = sc_p_sup.transform(X_surf_s[sup_idx])
    Xe_sup_n = sc_e_sup.transform(X_fm_s[sup_idx])
    n_comp   = max(2, min(20, n_sup // 10, Xs_sup_n.shape[1]))
    res_pipe = Pipeline([("pca", PCA(n_components=n_comp)),
                          ("ridge", Ridge(alpha=1.0))])
    res_pipe.fit(Xs_sup_n, Xe_sup_n)
    Xr_sup_n = Xe_sup_n - res_pipe.predict(Xs_sup_n)

    S_surf = torch.FloatTensor(Xs_sup_n).to(DEVICE)
    S_fm   = torch.FloatTensor(Xe_sup_n).to(DEVICE)
    S_V    = torch.FloatTensor(np.hstack([Xs_sup_n, Xr_sup_n])).to(DEVICE)

    sup_domains = tt_s[sup_idx]
    sup_nsteps  = ns_s[sup_idx]
    sup_nobj    = X_surf_s[sup_idx, 0].astype(int)

    results = {}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        "EXP C: ARC Attention Visualization\n"
        "Rows: easy / medium / hard query instances per domain\n"
        "Bars: attention weights on 60 support instances (colored by domain)",
        fontsize=10, fontweight="bold")

    TRAIN_COLORS = {
        "depot":     "#E74C3C", "satellite": "#F39C12",
        "rovers":    "#1ABC9C", "ferry":     "#95A5A6",
        "gripper":   "#E67E22",
    }

    dom_rhos = {}

    for col_idx, dom in enumerate(TEST_DOMAINS):
        mask  = task_types == dom
        Xs_q  = X_surf[mask]
        Xf_q  = X_fm[mask]
        ns_q  = y_steps[mask]

        # Select 2 easy, 2 medium, 2 hard query instances
        q33, q66 = np.percentile(ns_q, [33, 66])
        easy_idx  = np.where(ns_q <= q33)[0]
        hard_idx  = np.where(ns_q >= q66)[0]
        med_idx   = np.where((ns_q > q33) & (ns_q < q66))[0]

        selected = []
        for group, label in [(easy_idx, "easy"), (med_idx, "medium"),
                              (hard_idx, "hard")]:
            if len(group) > 0:
                chosen = rng.choice(group)
                selected.append((chosen, ns_q[chosen], label))

        # Compute attention weights for ALL instances (for correlation analysis)
        all_alphas   = []
        all_nsteps_q = []

        for i in range(len(Xs_q)):
            sc_p_q = StandardScaler().fit(Xs_q[[i]])   # trivial but consistent
            Xs_qi_n = sc_p_sup.transform(Xs_q[[i]])    # use support scaler
            Xf_qi_n = sc_e_sup.transform(Xf_q[[i]])
            Xr_qi_n = Xf_qi_n - res_pipe.predict(Xs_qi_n)

            q_surf_t  = torch.FloatTensor(Xs_qi_n[0]).to(DEVICE)
            q_fm_t    = torch.FloatTensor(Xf_qi_n[0]).to(DEVICE)
            q_resid_t = torch.FloatTensor(Xr_qi_n[0]).to(DEVICE)

            with torch.no_grad():
                _, alpha = model.get_features(
                    q_surf_t, q_fm_t, q_resid_t, S_surf, S_fm, S_V)
            all_alphas.append(alpha.cpu().numpy())
            all_nsteps_q.append(int(ns_q[i]))

        all_alphas   = np.array(all_alphas)   # shape (N_q, n_sup)
        all_nsteps_q = np.array(all_nsteps_q)

        # Correlation: for each query, compute weighted mean support n_steps
        # (what difficulty does ARC "retrieve"?)
        weighted_sup_nsteps = all_alphas @ sup_nsteps   # (N_q,)
        rho_corr, p_corr = stats.spearmanr(all_nsteps_q, weighted_sup_nsteps)
        dom_rhos[dom] = {"rho": float(rho_corr), "p": float(p_corr)}

        print(f"\n  {dom}:")
        print(f"    ρ(query_difficulty, attended_difficulty) = "
              f"{rho_corr:+.3f}  (p={p_corr:.4f})")

        # Plot attention for 3 selected queries (easy/medium/hard)
        for row_idx, (q_i, q_nsteps, difficulty) in enumerate(selected[:3]):
            ax = axes[row_idx, col_idx]
            alpha = all_alphas[q_i]

            # Color bars by training domain
            bar_colors = [TRAIN_COLORS.get(d, "#888") for d in sup_domains]
            ax.bar(range(n_sup), alpha, color=bar_colors, alpha=0.8,
                   edgecolor="none")

            # Mark top-3 attended
            top3 = np.argsort(alpha)[::-1][:3]
            for t in top3:
                ax.bar(t, alpha[t], color=bar_colors[t],
                       edgecolor="black", linewidth=1.5, alpha=1.0)
                ax.text(t, alpha[t] + 0.005,
                        f"d={sup_nsteps[t]}", ha="center", va="bottom",
                        fontsize=6, rotation=90)

            ax.set_title(
                f"{DOMAIN_LABELS[dom]} | query: {difficulty} (steps={q_nsteps})\n"
                f"top-3 support: steps={sorted(sup_nsteps[top3])}",
                fontsize=7.5)
            ax.set_xlabel("Support instance index", fontsize=7)
            ax.set_ylabel("Attention weight", fontsize=7)
            ax.set_xlim(-1, n_sup)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.2)

        results[dom] = {
            "rho_query_vs_attended": float(rho_corr),
            "p_value":               float(p_corr),
            "n_queries":             len(all_nsteps_q),
            "selected_examples": [
                {"difficulty": diff, "query_nsteps": int(ns),
                 "top3_support_nsteps": sorted(
                     sup_nsteps[np.argsort(all_alphas[qi])[::-1][:3]].tolist())}
                for qi, ns, diff in selected[:3]
            ],
        }

    # Print attention correlation summary
    print("\n  ATTENTION CORRELATION SUMMARY:")
    print(f"  ρ(query difficulty, attended difficulty) — "
          f"positive = ARC attends to similar difficulty")
    print(f"  {'Domain':<22}  {'ρ':>8}  {'p-value':>10}  {'Verdict':>25}")
    print("  " + "-" * 68)
    for dom, r in dom_rhos.items():
        v = "Meaningful ✓" if r["rho"] > 0.2 and r["p"] < 0.05 else \
            "Weak" if r["rho"] > 0 else "Random"
        print(f"  {dom:<22}  {r['rho']:>+8.3f}  {r['p']:>10.4f}  {v:>25}")

    # Add domain legend to last plot
    from matplotlib.patches import Patch
    legend_els = [Patch(color=c, label=d)
                  for d, c in TRAIN_COLORS.items()]
    fig.legend(handles=legend_els, loc="lower center",
               ncol=len(TRAIN_COLORS), fontsize=8,
               title="Training domain colors", title_fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(FIG_DIR / "expC_attention_viz.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(FIG_DIR / "expC_attention_viz.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Figure → {FIG_DIR}/expC_attention_viz.pdf")

    out = RESULTS_DIR / "expC_attention_viz.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {out}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A","B","C","all"], default="all")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    run_all = args.exp == "all"

    if run_all or args.exp == "A":
        print("\n" + "="*65)
        print("Running EXP A: Fixed-object-count subset")
        exp_a_fixed_nobj(args)

    if run_all or args.exp == "B":
        print("\n" + "="*65)
        print("Running EXP B: Deep MLP non-episodic baseline")
        exp_b_mlp_baseline(args)

    if run_all or args.exp == "C":
        print("\n" + "="*65)
        print("Running EXP C: Attention visualization")
        exp_c_attention_viz(args)

    print("\nAll experiments complete.")
    print(f"Figures → {FIG_DIR}/expA_*.pdf, expB_*.pdf, expC_*.pdf")
    print(f"Results → {RESULTS_DIR}/expA_*.json, expB_*.json, expC_*.json")


if __name__ == "__main__":
    main()
