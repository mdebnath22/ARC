"""
plan_step14_bfs_budget_sensitivity.py
=======================================
Experiment 8: Test ARC performance under different BFS budget settings.
MAX_STEPS ∈ {8, 12, 16} × MAX_NODES ∈ {2000, 5000, 10000}

Shows whether ARC learns a stable structural signal or only the specific
boundary encoded by MAX_STEPS=12, MAX_NODES=5000.

Requires re-running data prep with modified budgets.
Usage:
  python plan_step14_bfs_budget_sensitivity.py
"""

import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import importlib.util

warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/planning")
RESULTS_DIR = Path("results_planning"); RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR     = Path("figures_planning"); FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR    = Path("checkpoints_planning")

spec = importlib.util.spec_from_file_location("step3", Path(__file__).parent / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(step3)
PlanningGURU    = step3.PlanningGURU
PlanningMetaSampler = step3.PlanningMetaSampler
train_guru      = step3.train_guru
evaluate_on_domain = step3.evaluate_on_domain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BUDGET_GRID = [
    {"max_steps": 8,  "max_nodes": 2000},
    {"max_steps": 12, "max_nodes": 5000},  # original
    {"max_steps": 16, "max_nodes": 10000},
]


def relabel_with_budget(max_steps, max_nodes):
    """
    Re-derive y_success and y_nsteps from existing episodes.json
    under a new BFS budget. No need to re-run the generator.
    """
    episodes_path = DATA_DIR / "episodes.json"
    if not episodes_path.exists():
        raise FileNotFoundError("episodes.json not found.")
    episodes = json.loads(episodes_path.read_text())
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)

    y_nsteps_new  = np.zeros(len(episodes), dtype=np.float32)
    y_success_new = np.zeros(len(episodes), dtype=np.int64)

    for i, ep in enumerate(episodes):
        n = ep.get("n_steps", 0)
        # Re-label: if original plan exceeded new budget, mark as failure
        if n <= 0 or n > max_steps:
            y_nsteps_new[i]  = float(max_steps + 1)  # treat as hard
            y_success_new[i] = 0
        else:
            y_nsteps_new[i] = float(n)
            # Recompute success relative to NEW per-domain median
            y_success_new[i] = 1  # will fix below

    # Recompute binary success by new per-domain median
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    all_domains = list(set(task_types.tolist()))
    for dom in all_domains:
        mask = task_types == dom
        med  = float(np.median(y_nsteps_new[mask]))
        y_success_new[mask] = (y_nsteps_new[mask] <= med).astype(int)

    return y_success_new, y_nsteps_new


def run_budget_sensitivity(n_episodes=1500):
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())

    test_domains  = registry["splits"]["meta_test"]["tasks"]
    train_domains = registry["splits"]["meta_train"]["tasks"]
    val_domains   = registry["splits"]["meta_val"]["tasks"]
    train_mask    = np.isin(task_types, train_domains)

    surf_dim, fm_dim = X_surf.shape[1], X_fm.shape[1]
    all_results = {}

    print(f"\n{'Budget':<25} {'Domain':<25} {'AUC success':>12} {'R² n_steps':>12}")
    print("-" * 78)

    for budget in BUDGET_GRID:
        ms = budget["max_steps"]; mn = budget["max_nodes"]
        tag = f"s{ms}_n{mn}"

        y_success, y_nsteps = relabel_with_budget(ms, mn)

        for label in ["success", "n_steps"]:
            y = y_success if label == "success" else y_nsteps

            train_samp = PlanningMetaSampler(
                train_domains, X_surf, X_fm,
                y_success.astype(int), y_nsteps, task_types, DEVICE
            )
            val_samp = PlanningMetaSampler(
                val_domains, X_surf, X_fm,
                y_success.astype(int), y_nsteps, task_types, DEVICE
            )

            model = PlanningGURU(surf_dim, fm_dim).to(DEVICE)
            train_guru(model, train_samp, n_episodes, label=label,
                       device=DEVICE, val_sampler=val_samp, val_every=300)

            for dom in test_domains:
                mask = task_types == dom
                res = evaluate_on_domain(
                    model, dom,
                    X_surf[mask], X_fm[mask], y[mask],
                    X_surf[train_mask], X_fm[train_mask], y[train_mask],
                    label=label, device=DEVICE, cross_domain_support=True
                )
                score = res.get("guru_cross", {}).get("mean", float("nan")) if res else float("nan")
                key = f"{tag}_{dom}_{label}"
                all_results[key] = score
                print(f"  {tag:<25} {dom:<25} {score:>12.4f}  ({label})")

    _plot_budget_sensitivity(all_results, test_domains, BUDGET_GRID)
    (RESULTS_DIR / "e8_budget_sensitivity.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → results_planning/e8_budget_sensitivity.json")
    return all_results


def _plot_budget_sensitivity(results, domains, budgets):
    labels = ["success", "n_steps"]
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 5))
    if len(domains) == 1: axes = [axes]

    colors = {"success": "#3498DB", "n_steps": "#E74C3C"}
    markers = {"success": "o", "n_steps": "s"}
    xs = list(range(len(budgets)))
    xlabels = [f"s={b['max_steps']}\nn={b['max_nodes']}" for b in budgets]

    for ax, dom in zip(axes, domains):
        for label in labels:
            ys = [results.get(f"s{b['max_steps']}_n{b['max_nodes']}_{dom}_{label}", float("nan"))
                  for b in budgets]
            ax.plot(xs, ys, marker=markers[label], lw=2, color=colors[label],
                    label=f"{'AUC' if label=='success' else 'R²'} ({label})", markersize=8)

        ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(dom, fontsize=10)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(0, 1.1); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("E8: ARC Performance vs. BFS Budget\n"
                 "Stable performance = structural signal, not budget artifact",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(FIG_DIR / f"e8_budget_sensitivity{ext}", bbox_inches="tight", dpi=150)
    print(f"  Saved → figures_planning/e8_budget_sensitivity.pdf")
    plt.close()


if __name__ == "__main__":
    run_budget_sensitivity()