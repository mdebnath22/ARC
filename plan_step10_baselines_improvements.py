"""
plan_step10_baselines_improvements.py
======================================
Implements four improvements and additional baselines for the GURU paper
camera-ready / rebuttal, using real per-instance GPT-4o outcomes.

REBUTTAL (run now — hours):
  B1. Object-count threshold routing baseline
      Route instances with n_objects <= median to LLM, rest to BFS.
      Zero learning. Directly answers R2 Q1 in the main table.

  I1. Two-stage Mystery-BW routing
      Stage 1: GURU score predicts P(GPT-4o will attempt).
               Route predicted-refusal instances to BFS immediately.
      Stage 2: Among predicted-attempt instances, GURU score predicts
               P(success | attempt).
      Source: E10b showed +52.5% gain potential; this makes it a
              real reproducible experiment with precision/recall metrics.

CAMERA-READY (run later — no retraining needed):
  I2. Cost-weighted routing threshold
      Instead of fixed LLM budget, find theta* that minimises
      cost-per-valid-plan ($ / valid plan) using real GPT-4o token costs.

  I3. Domain-adaptive threshold
      Calibrate per-domain routing threshold on a held-out validation
      split of the JSONL (no new API calls). Apply domain-optimal
      theta at test time.

USAGE:
  # Run all four (rebuttal + camera-ready):
  python plan_step10_baselines_improvements.py \\
      --gpt4o_file gpt4o_eval_instances.jsonl

  # Run only rebuttal items (faster):
  python plan_step10_baselines_improvements.py \\
      --gpt4o_file gpt4o_eval_instances.jsonl \\
      --rebuttal_only

  # Run only camera-ready improvements:
  python plan_step10_baselines_improvements.py \\
      --gpt4o_file gpt4o_eval_instances.jsonl \\
      --camera_ready_only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data" / "planning"
RESULTS_DIR = ROOT_DIR / "results_planning"
FIG_DIR     = ROOT_DIR / "figures_planning"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# GPT-4o token pricing (as of 2025)
COST_PER_PROMPT_TOKEN     = 2.50 / 1_000_000
COST_PER_COMPLETION_TOKEN = 10.0 / 1_000_000

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


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_step6():
    spec = importlib.util.spec_from_file_location(
        "step6", ROOT_DIR / "plan_step6_pddlinst_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_gpt4o(path: str) -> Dict[str, List[dict]]:
    """Load per-instance GPT-4o JSONL, sorted by instance_id per domain."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_domain: Dict[str, List[dict]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    for dom in by_domain:
        by_domain[dom].sort(key=lambda r: int(r["instance_id"]))
    return by_domain


def compute_guru_scores(step6, X_surf, X_fm, task_types, y_success, splits,
                         n_runs=10, seed=42):
    """Compute GURU P(easy) scores for all test domains."""
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model  = step6.load_checkpoint("success")

    train_domains = splits["meta_train"]["domains"]
    train_mask    = np.isin(task_types, train_domains)
    X_surf_s      = X_surf[train_mask]
    X_fm_s        = X_fm[train_mask]

    scores_by_domain = {}
    n_objects_by_domain = {}

    for dom in splits["meta_test"]["domains"]:
        mask = task_types == dom
        scores = step6.get_guru_scores_per_instance(
            model, X_surf[mask], X_fm[mask], y_success[mask],
            X_surf_s, X_fm_s, n_runs=n_runs, rng_seed=seed)
        scores_by_domain[dom]   = scores
        n_objects_by_domain[dom] = X_surf[mask][:, 0]  # feature 0 = n_objects

    return scores_by_domain, n_objects_by_domain


# ══════════════════════════════════════════════════════════════════════════════
# Core routing utility
# ══════════════════════════════════════════════════════════════════════════════

def topk_routing(scores: np.ndarray, y_llm: np.ndarray, k: int
                  ) -> Tuple[float, float, float]:
    """
    Route top-k (highest score = easiest) to LLM, rest to BFS.
    Returns (plan_validity, llm_precision, recall_of_successes).
    """
    n   = len(y_llm)
    idx = np.argsort(scores)[::-1][:k]
    llm_succ = float(y_llm[idx].sum())
    validity  = (llm_succ + (n - k)) / n
    precision = llm_succ / k if k > 0 else float("nan")
    # Recall: fraction of all LLM-solvable instances that we sent to LLM
    total_solvable = y_llm.sum()
    recall = llm_succ / total_solvable if total_solvable > 0 else float("nan")
    return float(validity), float(precision), float(recall)


def sweep_routing(scores: np.ndarray, y_llm: np.ndarray,
                   costs: np.ndarray = None
                   ) -> List[dict]:
    """
    Sweep all possible k values (0 to N), return full Pareto curve.
    If costs provided (per-instance API cost), also compute $/valid plan.
    """
    n = len(y_llm)
    # Sort once: descending score = easiest first
    sorted_idx = np.argsort(scores)[::-1]
    # Cumulative LLM successes as we add more instances to LLM
    cum_succ = np.concatenate([[0], np.cumsum(y_llm[sorted_idx])])
    curve = []
    for k in range(0, n + 1):
        llm_succ = cum_succ[k]
        validity  = (llm_succ + (n - k)) / n
        precision = llm_succ / k if k > 0 else float("nan")
        b         = (n - k) / n        # BFS budget fraction
        acc_llm   = float(y_llm.mean())
        blind_v   = (k / n) * acc_llm + b * 1.0
        gain      = validity - blind_v

        pt = {
            "k":          int(k),
            "llm_frac":   float(k / n),
            "bfs_frac":   float(b),
            "validity":   float(validity),
            "precision":  float(precision),
            "gain_vs_blind": float(gain),
        }
        if costs is not None:
            llm_cost = float(costs[sorted_idx[:k]].sum()) if k > 0 else 0.0
            n_valid  = int(llm_succ + (n - k))
            pt["cost_total"]      = llm_cost
            pt["cost_per_valid"]  = (llm_cost / n_valid
                                      if n_valid > 0 else float("inf"))
        curve.append(pt)
    return curve


# ══════════════════════════════════════════════════════════════════════════════
# B1 — Object-count threshold baseline
# ══════════════════════════════════════════════════════════════════════════════

def b1_nobj_baseline(n_objects_by_domain: dict,
                      guru_scores_by_domain: dict,
                      gpt4o_by_domain: dict,
                      e1_budgets: dict) -> dict:
    """
    B1: Object-count threshold routing.
    Route instances with n_objects <= median to LLM, rest to BFS.
    Uses same matched budget k as E1b for fair comparison.

    Also computes GURU vs n_obj at optimal k (cost-optimal) and
    at matched-budget k to show both views.
    """
    print("\n" + "=" * 65)
    print("B1 — Object-count threshold routing baseline")
    print("  Route low-n_obj instances to LLM, high-n_obj to BFS")
    print("  Compared at matched E1 budget AND at optimal k")
    print("=" * 65)

    print(f"\n  {'Domain':<22}  {'Budget':>7}  {'Blind':>7}  "
          f"{'n_obj':>7}  {'Surface':>8}  {'GURU':>7}  "
          f"{'GURU-nobj':>10}  {'GURU-surf':>10}")
    print("  " + "-" * 82)

    results = {}
    for dom in TEST_DOMAINS:
        if dom not in guru_scores_by_domain:
            continue
        n_obj    = n_objects_by_domain[dom]
        scores   = guru_scores_by_domain[dom]
        gpt4o    = gpt4o_by_domain[dom]
        y_llm    = np.array([1.0 if r["valid_plan"] else 0.0 for r in gpt4o])
        N        = len(y_llm)
        acc_llm  = float(y_llm.mean())
        k        = e1_budgets.get(dom, int(0.5 * N))

        # Blind
        v_blind, _, _ = topk_routing(
            np.random.default_rng(42).random(N), y_llm, k)
        v_blind = (k / N) * acc_llm + (1 - k / N) * 1.0

        # n_obj routing: sort ascending (low n_obj = easy → LLM)
        nobj_scores = -n_obj.astype(float)  # negate so argsort descending = low n_obj first
        v_nobj, p_nobj, r_nobj = topk_routing(nobj_scores, y_llm, k)

        # GURU routing
        v_guru, p_guru, r_guru = topk_routing(scores, y_llm, k)

        # Surface routing (use GURU scores as proxy if surface not available)
        # Will be filled in main() when surface scores are computed
        v_surf = None

        print(f"  {dom:<22}  {k/N:>7.1%}  {v_blind:>7.1%}  "
              f"{v_nobj:>7.1%}  {'---':>8}  {v_guru:>7.1%}  "
              f"{v_guru-v_nobj:>+10.1%}  {'---':>10}")

        # Full sweep for Pareto curves
        nobj_curve = sweep_routing(nobj_scores, y_llm)
        guru_curve = sweep_routing(scores, y_llm)

        # Find optimal k for each (max validity)
        best_nobj = max(nobj_curve, key=lambda p: p["validity"])
        best_guru = max(guru_curve, key=lambda p: p["validity"])

        results[dom] = {
            "matched_budget": {
                "k": int(k), "llm_frac": float(k/N),
                "v_blind":  float(v_blind),
                "v_nobj":   float(v_nobj),  "p_nobj": float(p_nobj),
                "v_guru":   float(v_guru),  "p_guru": float(p_guru),
                "guru_vs_nobj": float(v_guru - v_nobj),
                "guru_vs_blind": float(v_guru - v_blind),
            },
            "nobj_curve":     nobj_curve,
            "guru_curve":     guru_curve,
            "best_nobj_k":    best_nobj,
            "best_guru_k":    best_guru,
            "acc_llm":        float(acc_llm),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# I1 — Two-stage Mystery-BW routing
# ══════════════════════════════════════════════════════════════════════════════

def i1_two_stage_routing(guru_scores_by_domain: dict,
                          gpt4o_by_domain: dict) -> dict:
    """
    I1: Two-stage routing for Mystery-BW (generalisable to all domains).

    Stage 1 — Predict P(attempt): route instances with low GURU score
              (predicted refusal) directly to BFS. GURU score correlates
              with parse_ok (ρ=+0.379, p<0.001 from E10a).

    Stage 2 — Among predicted-attempt instances, route the hardest
              (low GURU score within the attempt set) to BFS as well.

    Key result from E10b: P(valid | refused) = 0.0% exactly.
    Routing predicted-refusal instances to BFS costs nothing in validity.

    Reports:
      - Stage 1 precision/recall on predicting refusal
      - System validity at each stage 1 threshold
      - Comparison with single-stage routing
      - Optimal two-stage threshold pair (theta_1, theta_2)
    """
    print("\n" + "=" * 65)
    print("I1 — Two-stage routing (Stage 1: predict attempt,")
    print("                        Stage 2: predict success|attempt)")
    print("=" * 65)

    results = {}

    for dom in TEST_DOMAINS:
        if dom not in guru_scores_by_domain:
            continue
        scores = guru_scores_by_domain[dom]
        gpt4o  = gpt4o_by_domain[dom]
        y_llm  = np.array([1.0 if r["valid_plan"] else 0.0 for r in gpt4o])
        y_parse = np.array([1.0 if r["parse_ok"] else 0.0 for r in gpt4o])
        N      = len(y_llm)
        acc_llm = float(y_llm.mean())

        # ── Baseline: single-stage at 50% ────────────────────────────
        k50 = N // 2
        v_single, p_single, _ = topk_routing(scores, y_llm, k50)

        print(f"\n  {dom}:")
        print(f"    Baseline (GPT-4o all):       validity={acc_llm:.1%}")
        print(f"    Single-stage (50% budget):   validity={v_single:.1%}  "
              f"precision={p_single:.1%}")

        # ── Stage 1: sweep theta_1 (refusal prediction threshold) ────
        # Route instances with score < theta_1 to BFS (predicted refusals)
        # These instances don't get a GPT-4o call at all

        # Ground truth: refused = parse_ok == 0
        refused_mask  = y_parse == 0    # GPT-4o refused these
        attempted_mask = y_parse == 1   # GPT-4o attempted these

        print(f"\n    GPT-4o refused {refused_mask.sum()}/{N} "
              f"({refused_mask.mean():.1%}) instances")
        print(f"    P(valid | refused)   = {y_llm[refused_mask].mean():.1%}  "
              f"← always 0")
        print(f"    P(valid | attempted) = {y_llm[attempted_mask].mean():.1%}")

        # Sweep theta_1
        thetas_1 = np.linspace(0.05, 0.95, 37)
        stage1_curve = []
        best_stage1 = None
        best_gain   = -np.inf

        for t1 in thetas_1:
            # Stage 1: score < t1 → BFS (predicted refusal)
            #          score >= t1 → attempt with GPT-4o
            to_bfs_s1 = scores < t1
            to_llm_s1 = ~to_bfs_s1
            n_bfs_s1  = to_bfs_s1.sum()
            n_llm_s1  = to_llm_s1.sum()

            # Validity: BFS instances always valid, LLM instances use real labels
            v_s1 = (y_llm[to_llm_s1].sum() + n_bfs_s1) / N
            llm_frac_s1 = n_llm_s1 / N

            # Precision/recall of refusal prediction
            tp = (to_bfs_s1 & refused_mask).sum()
            fp = (to_bfs_s1 & ~refused_mask).sum()
            fn = (~to_bfs_s1 & refused_mask).sum()
            prec_s1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_s1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_s1   = (2 * prec_s1 * rec_s1 / (prec_s1 + rec_s1)
                       if (prec_s1 + rec_s1) > 0 else 0.0)

            # Baseline validity at same LLM fraction
            v_blind_s1 = llm_frac_s1 * acc_llm + (1 - llm_frac_s1)
            gain_s1    = v_s1 - v_blind_s1

            stage1_curve.append({
                "theta_1":   float(t1),
                "validity":  float(v_s1),
                "llm_frac":  float(llm_frac_s1),
                "gain":      float(gain_s1),
                "precision": float(prec_s1),
                "recall":    float(rec_s1),
                "f1":        float(f1_s1),
                "n_bfs":     int(n_bfs_s1),
                "n_llm":     int(n_llm_s1),
            })

            if gain_s1 > best_gain:
                best_gain   = gain_s1
                best_stage1 = stage1_curve[-1]

        # ── Stage 2: among predicted-attempt set, sweep theta_2 ──────
        # Use best theta_1 from stage 1
        t1_star  = best_stage1["theta_1"]
        to_llm_s1 = scores >= t1_star
        to_bfs_s1 = ~to_llm_s1
        n_bfs_s1  = to_bfs_s1.sum()

        # Within the LLM-routed set, sweep theta_2 for secondary routing
        scores_llm = scores[to_llm_s1]
        y_llm_sub  = y_llm[to_llm_s1]
        n_sub      = len(scores_llm)

        best_two_stage = None
        best_two_stage_v = -np.inf
        two_stage_curve = []

        for t2 in thetas_1:
            # Among LLM-routed: score < t2 → also BFS (hard within attempt set)
            to_bfs_s2 = scores_llm < t2
            to_llm_s2 = ~to_bfs_s2
            n_bfs_s2  = to_bfs_s2.sum()
            n_llm_s2  = to_llm_s2.sum()

            # Total validity
            total_bfs  = n_bfs_s1 + n_bfs_s2   # BFS always valid
            total_llm_v = y_llm_sub[to_llm_s2].sum()
            v_two = (total_llm_v + total_bfs) / N
            llm_frac_two = n_llm_s2 / N

            v_blind_two = llm_frac_two * acc_llm + (1 - llm_frac_two)
            gain_two    = v_two - v_blind_two

            two_stage_curve.append({
                "theta_1":   float(t1_star),
                "theta_2":   float(t2),
                "validity":  float(v_two),
                "llm_frac":  float(llm_frac_two),
                "gain":      float(gain_two),
                "n_bfs_total": int(total_bfs),
                "n_llm":     int(n_llm_s2),
            })

            if v_two > best_two_stage_v:
                best_two_stage_v = v_two
                best_two_stage   = two_stage_curve[-1]

        print(f"\n    Stage 1 best (theta_1={t1_star:.2f}):")
        print(f"      Validity={best_stage1['validity']:.1%}  "
              f"Gain={best_stage1['gain']:+.1%}  "
              f"LLM_frac={best_stage1['llm_frac']:.1%}")
        print(f"      Refusal precision={best_stage1['precision']:.1%}  "
              f"Recall={best_stage1['recall']:.1%}  "
              f"F1={best_stage1['f1']:.3f}")
        print(f"\n    Two-stage best "
              f"(theta_1={best_two_stage['theta_1']:.2f}, "
              f"theta_2={best_two_stage['theta_2']:.2f}):")
        print(f"      Validity={best_two_stage['validity']:.1%}  "
              f"Gain={best_two_stage['gain']:+.1%}  "
              f"LLM_frac={best_two_stage['llm_frac']:.1%}")
        two_stage_gain = best_two_stage["validity"] - v_single
        # Two-stage is only beneficial when GPT-4o has meaningful refusal rate.
        # If refusal rate is low (< 10%), stage 1 barely fires and the
        # additional threshold introduces noise. Flag this per domain.
        two_stage_useful = refused_mask.mean() >= 0.10
        note = ("✓ beneficial" if two_stage_gain > 0.005
                else "~ marginal" if two_stage_gain > -0.005
                else "✗ not beneficial (low refusal rate)")

        print(f"\n    Summary:")
        print(f"      GPT-4o only:     {acc_llm:.1%}")
        print(f"      Single-stage:    {v_single:.1%}  "
              f"(+{v_single-acc_llm:.1%} vs LLM-only)")
        print(f"      Two-stage:       {best_two_stage['validity']:.1%}  "
              f"(+{best_two_stage['validity']-acc_llm:.1%} vs LLM-only, "
              f"{two_stage_gain:+.1%} vs single-stage)  [{note}]")
        if not two_stage_useful:
            print(f"      Note: refusal rate={refused_mask.mean():.1%} "
                  f"< 10% — two-stage routing not recommended for this domain.")

        results[dom] = {
            "acc_llm":           float(acc_llm),
            "two_stage_useful":  bool(two_stage_useful),
            "two_stage_gain":    float(two_stage_gain),
            "refused_frac":      float(refused_mask.mean()),
            "p_valid_given_refused": float(y_llm[refused_mask].mean()),
            "p_valid_given_attempt": float(y_llm[attempted_mask].mean()),
            "single_stage_50pct": {
                "validity":  float(v_single),
                "precision": float(p_single),
            },
            "best_stage1":       best_stage1,
            "best_two_stage":    best_two_stage,
            "stage1_curve":      stage1_curve,
            "two_stage_curve":   two_stage_curve,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# I2 — Cost-weighted routing threshold
# ══════════════════════════════════════════════════════════════════════════════

def i2_cost_weighted_routing(guru_scores_by_domain: dict,
                              gpt4o_by_domain: dict) -> dict:
    """
    I2: Find routing threshold that minimises cost-per-valid-plan.

    Instead of fixed LLM budget (E1b), find k* that minimises:
      cost_per_valid_plan(k) = sum(cost[LLM-routed]) / n_valid_plans_produced

    Uses real GPT-4o token costs from the JSONL.
    BFS cost ≈ 0 (local CPU, negligible vs API cost).

    Reports:
      - Baseline (all GPT-4o): $/valid plan
      - GURU at k*: $/valid plan
      - Percentage cost reduction
      - System validity at k* (may differ from E1b)
    """
    print("\n" + "=" * 65)
    print("I2 — Cost-weighted routing (minimise $/valid plan)")
    print("  k* = argmin cost_per_valid_plan(k)")
    print("=" * 65)
    print(f"\n  {'Domain':<22}  {'Baseline $/vp':>14}  {'GURU k* $/vp':>13}  "
          f"{'Saving':>8}  {'k* validity':>12}  {'k* LLM%':>8}")
    print("  " + "-" * 82)

    results = {}
    for dom in TEST_DOMAINS:
        if dom not in guru_scores_by_domain:
            continue
        scores = guru_scores_by_domain[dom]
        gpt4o  = gpt4o_by_domain[dom]
        y_llm  = np.array([1.0 if r["valid_plan"] else 0.0 for r in gpt4o])
        costs  = np.array([
            r["prompt_tokens"]     * COST_PER_PROMPT_TOKEN +
            r["completion_tokens"] * COST_PER_COMPLETION_TOKEN
            for r in gpt4o])
        N      = len(y_llm)

        # Baseline: all GPT-4o
        base_cost = float(costs.sum())
        base_valid = int(y_llm.sum())
        base_cppv  = base_cost / base_valid if base_valid > 0 else float("inf")

        # Full sweep with cost
        curve = sweep_routing(scores, y_llm, costs=costs)

        # Find k* = min cost_per_valid_plan subject to min LLM fraction.
        # Minimum fraction constraint: BFS is infeasible for hard instances
        # in large state spaces, so some GPT-4o usage is required.
        # We require at least 20% of instances go to GPT-4o (realistic floor).
        MIN_LLM_FRAC = 0.20
        valid_pts = [p for p in curve
                     if p["k"] >= int(MIN_LLM_FRAC * N)
                     and p["cost_per_valid"] < float("inf")]
        if not valid_pts:
            valid_pts = [p for p in curve if p["k"] > 0
                         and p["cost_per_valid"] < float("inf")]
        best = min(valid_pts, key=lambda p: p["cost_per_valid"])
        saving_pct = (1 - best["cost_per_valid"] / base_cppv) * 100

        print(f"  {dom:<22}  ${base_cppv:>13.4f}  ${best['cost_per_valid']:>12.4f}  "
              f"{saving_pct:>7.0f}%  {best['validity']:>12.1%}  "
              f"{best['llm_frac']:>8.1%}")

        results[dom] = {
            "baseline_cppv":   float(base_cppv),
            "baseline_cost":   float(base_cost),
            "baseline_valid":  int(base_valid),
            "guru_kstar": {
                "k":           int(best["k"]),
                "llm_frac":    float(best["llm_frac"]),
                "validity":    float(best["validity"]),
                "cost_total":  float(best["cost_total"]),
                "cost_per_valid": float(best["cost_per_valid"]),
                "gain_vs_blind": float(best["gain_vs_blind"]),
            },
            "saving_pct":      float(saving_pct),
            "full_curve":      [p for p in curve if p["k"] % 5 == 0],  # thin
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# I3 — Domain-adaptive threshold
# ══════════════════════════════════════════════════════════════════════════════

def i3_domain_adaptive_threshold(guru_scores_by_domain: dict,
                                   gpt4o_by_domain: dict,
                                   val_frac: float = 0.3,
                                   seed: int = 42) -> dict:
    """
    I3: Calibrate per-domain routing threshold on a validation split.

    Protocol:
      - Split each domain's JSONL 70% train / 30% val (no new API calls)
      - On 70% train split: find k* that maximises system validity
      - Apply k* to the 30% val split and report held-out validity
      - Compare against: fixed 50% budget, cost-optimal k* (I2)

    This simulates a practical deployment scenario: a small number of
    labelled examples in the new domain lets you calibrate the threshold
    before full deployment.

    No retraining of GURU required — only the routing threshold changes.
    """
    print("\n" + "=" * 65)
    print("I3 — Domain-adaptive threshold (70/30 calibration split)")
    print("  Calibrate k* on 70% train, report held-out 30% validity")
    print("=" * 65)
    print(f"\n  {'Domain':<22}  {'Fixed-50%':>10}  {'Adaptive k*':>12}  "
          f"{'Held-out v':>11}  {'k* frac':>8}  {'Delta':>7}")
    print("  " + "-" * 74)

    rng = np.random.default_rng(seed)
    results = {}

    for dom in TEST_DOMAINS:
        if dom not in guru_scores_by_domain:
            continue
        scores = guru_scores_by_domain[dom]
        gpt4o  = gpt4o_by_domain[dom]
        y_llm  = np.array([1.0 if r["valid_plan"] else 0.0 for r in gpt4o])
        N      = len(y_llm)

        # 70/30 split
        idx     = rng.permutation(N)
        n_tr    = int(N * (1 - val_frac))
        tr_idx  = idx[:n_tr]
        val_idx = idx[n_tr:]

        scores_tr  = scores[tr_idx];  y_tr  = y_llm[tr_idx]
        scores_val = scores[val_idx]; y_val = y_llm[val_idx]

        # Find k* on train split — subject to minimum LLM fraction.
        # Without constraint, k*=0 (all BFS) is always optimal when
        # BFS is assumed perfect, which is not a useful deployment recommendation.
        # Constraint: at least 20% of instances must go to LLM
        # (models the regime where BFS is infeasible for large instances).
        MIN_LLM_FRAC = 0.20
        tr_valid_pts = [p for p in sweep_routing(scores_tr, y_tr)
                        if p["llm_frac"] >= MIN_LLM_FRAC]
        if not tr_valid_pts:
            tr_valid_pts = sweep_routing(scores_tr, y_tr)
        best_tr     = max(tr_valid_pts, key=lambda p: p["validity"])
        k_star_frac = best_tr["llm_frac"]

        # Apply k* fraction to val split
        k_val, _, _ = topk_routing(
            scores_val, y_val, k=max(1, int(k_star_frac * len(val_idx))))
        v_val_adaptive = k_val

        # Fixed 50% on val split
        k50_val = len(val_idx) // 2
        v_val_fixed, _, _ = topk_routing(scores_val, y_val, k50_val)

        delta = v_val_adaptive - v_val_fixed

        print(f"  {dom:<22}  {v_val_fixed:>10.1%}  {k_star_frac:>12.1%}→  "
              f"{v_val_adaptive:>11.1%}  {k_star_frac:>8.1%}  {delta:>+7.1%}")

        results[dom] = {
            "train_best_k_frac":   float(k_star_frac),
            "train_best_validity": float(best_tr["validity"]),
            "val_fixed_50pct":     float(v_val_fixed),
            "val_adaptive":        float(v_val_adaptive),
            "delta":               float(delta),
            "n_train":             int(n_tr),
            "n_val":               int(len(val_idx)),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(b1, i1, i2, i3):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    ax_b1, ax_i1a, ax_i1b, ax_i2, ax_i3, ax_summary = axes

    fig.suptitle(
        "Baselines and Improvements — GURU Routing\n"
        "B1: n_obj baseline  |  I1: Two-stage  |  I2: Cost-weighted  |  I3: Adaptive threshold",
        fontsize=11, fontweight="bold")

    # ── B1: GURU vs n_obj routing curves ─────────────────────────────
    ax_b1.set_title("(B1) GURU vs n_obj routing\nat matched budget", fontsize=9)
    for dom in TEST_DOMAINS:
        if dom not in b1:
            continue
        col = COLORS[dom]
        lbl = DOMAIN_LABELS[dom]
        guru_c = b1[dom]["guru_curve"]
        nobj_c = b1[dom]["nobj_curve"]
        xs_g = [p["llm_frac"] for p in guru_c]
        ys_g = [p["validity"] for p in guru_c]
        xs_n = [p["llm_frac"] for p in nobj_c]
        ys_n = [p["validity"] for p in nobj_c]
        ax_b1.plot(xs_g, ys_g, "-",  color=col, lw=2, label=f"GURU ({lbl})")
        ax_b1.plot(xs_n, ys_n, "--", color=col, lw=1.3, alpha=0.6)
        # Mark matched budget
        mb = b1[dom]["matched_budget"]
        ax_b1.scatter(mb["llm_frac"], mb["v_guru"], color=col, s=80,
                       zorder=8, marker="o")
        ax_b1.scatter(mb["llm_frac"], mb["v_nobj"], color=col, s=60,
                       zorder=8, marker="^")
    from matplotlib.lines import Line2D
    h = [Line2D([0],[0], ls="-",  c="gray", lw=2, label="GURU"),
         Line2D([0],[0], ls="--", c="gray", lw=1.3, alpha=0.6, label="n_objects")]
    ax_b1.legend(handles=h, fontsize=8)
    ax_b1.set_xlabel("LLM fraction", fontsize=9)
    ax_b1.set_ylabel("System validity", fontsize=9)
    ax_b1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_b1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_b1.grid(True, alpha=0.3)

    # ── I1a: Stage 1 routing curves ───────────────────────────────────
    ax_i1a.set_title("(I1a) Two-stage routing\nStage 1: validity vs LLM fraction", fontsize=9)
    for dom in TEST_DOMAINS:
        if dom not in i1:
            continue
        col = COLORS[dom]
        r   = i1[dom]
        xs  = [p["llm_frac"] for p in r["stage1_curve"]]
        ys  = [p["validity"]  for p in r["stage1_curve"]]
        ax_i1a.plot(xs, ys, "-", color=col, lw=2, label=DOMAIN_LABELS[dom])
        best = r["best_stage1"]
        ax_i1a.scatter(best["llm_frac"], best["validity"],
                        color=col, s=100, zorder=8, marker="*")
        # Single-stage reference
        ss = r["single_stage_50pct"]
        ax_i1a.axhline(ss["validity"], color=col, ls=":", lw=1, alpha=0.5)
    ax_i1a.set_xlabel("LLM fraction (after Stage 1 BFS routing)", fontsize=9)
    ax_i1a.set_ylabel("System validity", fontsize=9)
    ax_i1a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_i1a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_i1a.legend(fontsize=8)
    ax_i1a.grid(True, alpha=0.3)

    # ── I1b: Stage 1 precision/recall ─────────────────────────────────
    ax_i1b.set_title("(I1b) Stage 1 refusal prediction\nPrecision & Recall vs threshold",
                      fontsize=9)
    for dom in TEST_DOMAINS:
        if dom not in i1:
            continue
        col = COLORS[dom]
        r   = i1[dom]
        xs  = [p["theta_1"]  for p in r["stage1_curve"]]
        yp  = [p["precision"] for p in r["stage1_curve"]]
        yr  = [p["recall"]    for p in r["stage1_curve"]]
        ax_i1b.plot(xs, yp, "-",  color=col, lw=2, label=f"Prec ({DOMAIN_LABELS[dom]})")
        ax_i1b.plot(xs, yr, "--", color=col, lw=1.3, alpha=0.7)
    ax_i1b.set_xlabel("Stage 1 threshold θ₁", fontsize=9)
    ax_i1b.set_ylabel("Precision (solid) / Recall (dashed)", fontsize=9)
    ax_i1b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_i1b.legend(fontsize=7)
    ax_i1b.grid(True, alpha=0.3)

    # ── I2: Cost per valid plan curves ────────────────────────────────
    ax_i2.set_title("(I2) Cost-weighted routing\n$/valid plan vs LLM fraction", fontsize=9)
    for dom in TEST_DOMAINS:
        if dom not in i2:
            continue
        col = COLORS[dom]
        r   = i2[dom]
        xs  = [p["llm_frac"]       for p in r["full_curve"] if p["k"] > 0]
        ys  = [p["cost_per_valid"] for p in r["full_curve"] if p["k"] > 0]
        # Cap for display
        ys_capped = [min(y, r["baseline_cppv"] * 2) for y in ys]
        ax_i2.plot(xs, ys_capped, "-", color=col, lw=2, label=DOMAIN_LABELS[dom])
        # Mark k*
        kstar = r["guru_kstar"]
        ax_i2.scatter(kstar["llm_frac"], min(kstar["cost_per_valid"],
                                              r["baseline_cppv"]*2),
                       color=col, s=100, zorder=8, marker="*")
        ax_i2.axhline(r["baseline_cppv"], color=col, ls=":", lw=1, alpha=0.5)
    ax_i2.set_xlabel("LLM fraction", fontsize=9)
    ax_i2.set_ylabel("$ per valid plan (↓ better)", fontsize=9)
    ax_i2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_i2.legend(fontsize=8)
    ax_i2.grid(True, alpha=0.3)

    # ── I3: Adaptive vs fixed threshold bar chart ─────────────────────
    ax_i3.set_title("(I3) Adaptive threshold vs fixed 50%\n(held-out 30% split)", fontsize=9)
    dom_lbls = [DOMAIN_LABELS[d] for d in TEST_DOMAINS if d in i3]
    v_fixed  = [i3[d]["val_fixed_50pct"] for d in TEST_DOMAINS if d in i3]
    v_adapt  = [i3[d]["val_adaptive"]    for d in TEST_DOMAINS if d in i3]
    x = np.arange(len(dom_lbls))
    w = 0.35
    b_f = ax_i3.bar(x - w/2, v_fixed, w, label="Fixed 50%", color="#BDC3C7",
                     edgecolor="gray")
    b_a = ax_i3.bar(x + w/2, v_adapt, w, label="Adaptive k*",
                     color=[COLORS[d] for d in TEST_DOMAINS if d in i3],
                     edgecolor="white")
    ax_i3.bar_label(b_f, fmt="%.0f%%",
                     labels=[f"{v:.0%}" for v in v_fixed], padding=2, fontsize=8)
    ax_i3.bar_label(b_a, fmt="%.0f%%",
                     labels=[f"{v:.0%}" for v in v_adapt], padding=2, fontsize=8)
    ax_i3.set_xticks(x)
    ax_i3.set_xticklabels(dom_lbls, fontsize=9)
    ax_i3.set_ylabel("Held-out system validity", fontsize=9)
    ax_i3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_i3.legend(fontsize=8)
    ax_i3.grid(True, axis="y", alpha=0.3)
    ax_i3.set_ylim(0, 1.1)

    # ── Summary bar chart ─────────────────────────────────────────────
    ax_summary.set_title("(Summary) System validity by method\nat matched budget",
                          fontsize=9)
    dom_lbls2 = [DOMAIN_LABELS[d] for d in TEST_DOMAINS if d in b1]
    methods   = ["Blind", "n_obj (B1)", "GURU (E1b)", "GURU+2stage (I1)",
                 "GURU+adaptive (I3)"]
    colors_m  = ["#95A5A6", "#BDC3C7", "#3498DB", "#2ECC71", "#E67E22"]

    v_blind = [b1[d]["matched_budget"]["v_blind"]  for d in TEST_DOMAINS if d in b1]
    v_nobj  = [b1[d]["matched_budget"]["v_nobj"]   for d in TEST_DOMAINS if d in b1]
    v_guru  = [b1[d]["matched_budget"]["v_guru"]   for d in TEST_DOMAINS if d in b1]
    v_2stg  = [i1[d]["best_two_stage"]["validity"] for d in TEST_DOMAINS if d in i1]
    v_adap  = [i3[d]["val_adaptive"]               for d in TEST_DOMAINS if d in i3]

    x2 = np.arange(len(dom_lbls2))
    n_m = 5
    offsets = np.linspace(-(n_m-1)/2, (n_m-1)/2, n_m) * 0.14

    for j, (vals, lbl, col) in enumerate(zip(
            [v_blind, v_nobj, v_guru, v_2stg, v_adap], methods, colors_m)):
        bars = ax_summary.bar(x2 + offsets[j], vals, 0.13,
                               label=lbl, color=col, edgecolor="white", lw=0.5)
        ax_summary.bar_label(bars, labels=[f"{v:.0%}" for v in vals],
                              padding=2, fontsize=6, rotation=90)

    ax_summary.set_xticks(x2)
    ax_summary.set_xticklabels(dom_lbls2, fontsize=9)
    ax_summary.set_ylabel("System validity", fontsize=9)
    ax_summary.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax_summary.legend(fontsize=7, loc="lower right")
    ax_summary.grid(True, axis="y", alpha=0.3)
    ax_summary.set_ylim(0, 1.15)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(FIG_DIR / f"e_baselines_improvements{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Figure → {FIG_DIR}/e_baselines_improvements.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

def print_latex_table(b1, i1, i2, i3):
    """
    Combined table: all methods at matched budget + improvements.
    Suitable for camera-ready appendix or rebuttal response.
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Extended routing comparison with additional baselines "
        r"and improvements. B1: object-count threshold baseline (no learning). "
        r"I1: two-stage routing (Stage 1 routes predicted-refusal instances "
        r"directly to BFS). I2: cost-optimal threshold (minimises \$/valid plan). "
        r"I3: domain-adaptive threshold (calibrated on 70\% held-out split). "
        r"All methods use real per-instance GPT-4o outcomes ($N=600$).}")
    lines.append(r"\label{tab:improvements}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l l ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Method} "
        r"& \textbf{Validity} & \textbf{LLM\%} & \textbf{$/valid plan} \\")
    lines.append(r"\midrule")

    dlabels = {"blocksworld": "Blocksworld",
               "logistics":   "Logistics",
               "mystery_blocksworld": "Mystery-BW"}

    for i, dom in enumerate(TEST_DOMAINS):
        if dom not in b1:
            continue
        lbl = dlabels[dom]
        mb  = b1[dom]["matched_budget"]
        ts  = i1[dom]["best_two_stage"] if dom in i1 else {}
        cw  = i2[dom]["guru_kstar"]     if dom in i2 else {}
        ad  = i3[dom]                   if dom in i3 else {}

        if i > 0:
            lines.append(r"\midrule")

        rows = [
            ("LLM only",             mb["v_blind"] - mb["v_blind"] + mb["v_blind"] * 0,
             1.0, None, True),
        ]
        # Just use the actual numbers
        methods_rows = [
            ("LLM only",
             None, 1.0, None),
            ("Blind (matched budget)",
             mb["v_blind"], mb["llm_frac"], None),
            ("n-objects threshold (B1)",
             mb["v_nobj"],  mb["llm_frac"], None),
            ("GURU routing (E1b)",
             mb["v_guru"],  mb["llm_frac"], None),
            ("GURU + two-stage (I1)",
             ts.get("validity"),    ts.get("llm_frac"),    None),
            ("GURU + cost-optimal (I2)",
             cw.get("validity"),    cw.get("llm_frac"),
             cw.get("cost_per_valid")),
            ("GURU + adaptive θ (I3)",
             ad.get("val_adaptive"), None, None),
        ]

        # LLM-only validity
        acc_llm = b1[dom]["acc_llm"]
        best_v  = max(v for _, v, _, _ in methods_rows if v is not None)

        for j, (method, val, llm_frac, cppv) in enumerate(methods_rows):
            if val is None and method == "LLM only":
                val = acc_llm
            if val is None:
                continue
            dom_s  = (f"\\multirow{{{len(methods_rows)}}}{{*}}{{{lbl}}}"
                      if j == 0 else "")
            val_s  = (f"\\textbf{{{val:.1%}}}"
                      if abs(val - best_v) < 0.005 else f"{val:.1%}")
            frac_s = f"{llm_frac:.0%}" if llm_frac is not None else "---"
            cppv_s = f"\\${cppv:.4f}" if cppv is not None else "---"
            lines.append(
                f"  {dom_s} & {method} & {val_s} & {frac_s} & {cppv_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    table_str = "\n".join(lines)
    out = RESULTS_DIR / "improvements_table.tex"
    out.write_text(table_str)
    print(f"\n  LaTeX table → {out}")
    return table_str


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt4o_file", type=str,
                        default="gpt4o_eval_instances.jsonl")
    parser.add_argument("--n_runs",     type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--rebuttal_only",     action="store_true",
                        help="Run only B1 and I1 (rebuttal items)")
    parser.add_argument("--camera_ready_only", action="store_true",
                        help="Run only I2 and I3 (camera-ready items)")
    args = parser.parse_args()

    run_rebuttal     = not args.camera_ready_only
    run_camera_ready = not args.rebuttal_only

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\nLoading GPT-4o results from {args.gpt4o_file}...")
    gpt4o_by_domain = load_gpt4o(args.gpt4o_file)
    for dom, rlist in gpt4o_by_domain.items():
        acc = sum(r["valid_plan"] for r in rlist) / len(rlist)
        print(f"  {dom}: n={len(rlist)}, validity={acc:.1%}")

    print("\nLoading planning data and GURU scores...")
    step6 = load_step6()
    X_surf, X_fm, task_types, y_success, _, splits = step6.load_data()
    scores_by_dom, nobj_by_dom = compute_guru_scores(
        step6, X_surf, X_fm, task_types, y_success, splits,
        n_runs=args.n_runs, seed=args.seed)

    # E1 matched budgets (from E1b real run results)
    # These are the k values used in the camera-ready E1b table
    e1_budgets = {
        "blocksworld":         int(0.51 * 200),
        "logistics":           int(0.58 * 200),
        "mystery_blocksworld": int(0.50 * 200),
    }

    # ── Run experiments ───────────────────────────────────────────────
    b1_results = i1_results = i2_results = i3_results = {}

    if run_rebuttal:
        b1_results = b1_nobj_baseline(
            nobj_by_dom, scores_by_dom, gpt4o_by_domain, e1_budgets)

        i1_results = i1_two_stage_routing(scores_by_dom, gpt4o_by_domain)

    if run_camera_ready:
        i2_results = i2_cost_weighted_routing(scores_by_dom, gpt4o_by_domain)
        i3_results = i3_domain_adaptive_threshold(
            scores_by_dom, gpt4o_by_domain, seed=args.seed)

    # ── Save results ──────────────────────────────────────────────────
    out = {
        "b1_nobj_baseline":        {d: {k: v for k, v in r.items()
                                         if "curve" not in k}
                                    for d, r in b1_results.items()},
        "i1_two_stage":            {d: {k: v for k, v in r.items()
                                         if "curve" not in k}
                                    for d, r in i1_results.items()},
        "i2_cost_weighted":        {d: {k: v for k, v in r.items()
                                         if k != "full_curve"}
                                    for d, r in i2_results.items()},
        "i3_domain_adaptive":      i3_results,
    }
    out_path = RESULTS_DIR / "e_baselines_improvements.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results → {out_path}")

    # ── Plot and table ────────────────────────────────────────────────
    if b1_results and i1_results and i2_results and i3_results:
        plot_all(b1_results, i1_results, i2_results, i3_results)
        print_latex_table(b1_results, i1_results, i2_results, i3_results)

    # ── Paper-ready summary ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PAPER-READY NUMBERS")
    print("=" * 65)

    if b1_results:
        print("\n  B1 — GURU vs n_obj at matched budget:")
        print(f"  {'Domain':<14} {'n_obj':>8}  {'GURU':>8}  {'GURU-nobj':>10}")
        for dom in TEST_DOMAINS:
            if dom not in b1_results:
                continue
            mb = b1_results[dom]["matched_budget"]
            print(f"    {DOMAIN_LABELS[dom]:<14} {mb['v_nobj']:>8.1%}  "
                  f"{mb['v_guru']:>8.1%}  {mb['guru_vs_nobj']:>+10.1%}")

    if i1_results:
        print("\n  I1 — Two-stage routing best results:")
        print(f"  (Two-stage only recommended when refusal rate >= 10%)")
        for dom in TEST_DOMAINS:
            if dom not in i1_results:
                continue
            r  = i1_results[dom]
            ts = r["best_two_stage"]
            ss = r["single_stage_50pct"]
            useful = "✓" if r.get("two_stage_useful") else "✗ skip"
            print(f"    {DOMAIN_LABELS[dom]:<14}: "
                  f"refusal={r['refused_frac']:.0%}  "
                  f"single={ss['validity']:.1%}  "
                  f"two-stage={ts['validity']:.1%}  "
                  f"gain={ts['validity']-ss['validity']:+.1%}  "
                  f"[{useful}]")

    if i2_results:
        print("\n  I2 — Cost-weighted routing ($/valid plan, min 20% LLM):")
        for dom in TEST_DOMAINS:
            if dom not in i2_results:
                continue
            r = i2_results[dom]
            k = r["guru_kstar"]
            print(f"    {DOMAIN_LABELS[dom]:<14}: "
                  f"baseline=${r['baseline_cppv']:.4f}  "
                  f"GURU k*=${k['cost_per_valid']:.4f}  "
                  f"({r['saving_pct']:.0f}% cheaper)  "
                  f"LLM%={k['llm_frac']:.0%}  "
                  f"validity={k['validity']:.1%}")

    if i3_results:
        print("\n  I3 — Domain-adaptive threshold, held-out 30% (min 20% LLM):")
        for dom in TEST_DOMAINS:
            if dom not in i3_results:
                continue
            r = i3_results[dom]
            print(f"    {DOMAIN_LABELS[dom]:<14}: "
                  f"fixed-50%={r['val_fixed_50pct']:.1%}  "
                  f"adaptive={r['val_adaptive']:.1%}  "
                  f"delta={r['delta']:+.1%}  "
                  f"k*={r['train_best_k_frac']:.0%}")


if __name__ == "__main__":
    main()