"""
plan_step41_loo_bootstrap.py
==============================
Computes paired bootstrap confidence intervals on
Delta_rho = |rho_ARC| - |rho_|O|| for every leave-one-domain-out fold,
using the per-instance predictions saved by plan_step40_loo_with_predictions.py.

For each held-out domain:
  1. Load the saved per-instance arrays (arc_score, n_objects, n_steps).
  2. Resample instances WITH replacement, 1000 times.
  3. On each resample, recompute rho_ARC and rho_|O| using the SAME
     resampled index set for both estimators (this is what makes it a
     PAIRED bootstrap -- it isolates the sampling variability of the
     comparison, not of each estimator independently).
  4. Report the empirical 95% interval on Delta_rho = |rho_ARC| - |rho_|O||
     per domain, plus a pooled interval across all domains.

USAGE:
  python plan_step41_loo_bootstrap.py \
      --input results_planning/loo_with_predictions.json \
      --n_boot 1000 --seed 42
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_delta_rho(arc_scores, n_objects, n_steps, n_boot=1000, seed=42):
    """
    Paired bootstrap on Delta_rho = |rho_ARC| - |rho_obj| for one domain.
    Returns a dict with point estimate, 95% CI, and raw bootstrap array.
    """
    rng = np.random.default_rng(seed)
    valid = n_steps > 0
    arc_v = arc_scores[valid]
    obj_v = n_objects[valid]
    ns_v  = n_steps[valid]
    n_valid = valid.sum()

    if n_valid < 10:
        return None

    # Point estimate (matches the paper's reported per-domain values)
    rho_arc_pt, _ = stats.spearmanr(arc_v, ns_v)
    rho_obj_pt, _ = stats.spearmanr(obj_v, ns_v)
    delta_pt = abs(rho_arc_pt) - abs(rho_obj_pt)

    boot_deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_valid, n_valid)  # resample WITH replacement
        try:
            rho_arc_b, _ = stats.spearmanr(arc_v[idx], ns_v[idx])
            rho_obj_b, _ = stats.spearmanr(obj_v[idx], ns_v[idx])
            if np.isnan(rho_arc_b) or np.isnan(rho_obj_b):
                boot_deltas[b] = np.nan
            else:
                boot_deltas[b] = abs(rho_arc_b) - abs(rho_obj_b)
        except Exception:
            boot_deltas[b] = np.nan

    boot_deltas_clean = boot_deltas[~np.isnan(boot_deltas)]
    n_dropped = n_boot - len(boot_deltas_clean)

    ci_low  = float(np.percentile(boot_deltas_clean, 2.5))
    ci_high = float(np.percentile(boot_deltas_clean, 97.5))

    return {
        "point_estimate": float(delta_pt),
        "rho_arc_point": float(abs(rho_arc_pt)),
        "rho_obj_point": float(abs(rho_obj_pt)),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "n_valid": int(n_valid),
        "n_boot_used": int(len(boot_deltas_clean)),
        "n_boot_dropped": int(n_dropped),
        "boot_array": boot_deltas_clean.tolist(),
    }


def pooled_bootstrap(all_domain_results, n_boot=1000, seed=42):
    """
    Pooled interval across domains: for each of n_boot draws, resample
    domains (with replacement) and, within each resampled domain, resample
    one of its own bootstrap draws; average across the resampled domains.
    This propagates both within-domain (instance-level) and
    between-domain uncertainty into a single pooled interval.
    """
    rng = np.random.default_rng(seed)
    domains = [d for d, r in all_domain_results.items() if r is not None]
    boot_matrix = np.array([all_domain_results[d]["boot_array"][:n_boot]
                             for d in domains])

    pooled_boot = np.empty(n_boot)
    for b in range(n_boot):
        domain_idx = rng.integers(0, boot_matrix.shape[0], boot_matrix.shape[0])
        col_idx = rng.integers(0, boot_matrix.shape[1], boot_matrix.shape[0])
        pooled_boot[b] = boot_matrix[domain_idx, col_idx].mean()

    pooled_point = float(np.mean([all_domain_results[d]["point_estimate"]
                                   for d in domains]))

    return {
        "pooled_point_estimate": pooled_point,
        "pooled_ci_95_low": float(np.percentile(pooled_boot, 2.5)),
        "pooled_ci_95_high": float(np.percentile(pooled_boot, 97.5)),
        "n_domains": int(boot_matrix.shape[0]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str,
                   default="results_planning/loo_with_predictions.json")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str,
                   default="results_planning/loo_bootstrap_ci.json")
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text())

    domain_results = {}
    print(f"{'Domain':<22} {'Delta_rho':>10} {'95% CI':>20} {'Excludes 0?':>12} {'N':>6}")
    print("-" * 76)

    for domain, fold_data in data.items():
        if domain == "_summary":
            continue
        per_inst = fold_data.get("per_instance")
        if per_inst is None:
            print(f"  {domain:<20} SKIPPED (no per-instance data)")
            continue

        arc_scores = np.array(per_inst["arc_score"])
        n_objects  = np.array(per_inst["n_objects"])
        n_steps    = np.array(per_inst["n_steps"])

        result = bootstrap_delta_rho(arc_scores, n_objects, n_steps,
                                     n_boot=args.n_boot, seed=args.seed)
        if result is None:
            print(f"  {domain:<20} SKIPPED (insufficient valid instances)")
            continue

        domain_results[domain] = result
        excl = "YES" if result["excludes_zero"] else "no"
        ci_str = f"[{result['ci_95_low']:+.3f}, {result['ci_95_high']:+.3f}]"
        print(f"  {domain:<20} {result['point_estimate']:>+9.3f} {ci_str:>20} "
              f"{excl:>12} {result['n_valid']:>6}")
        if result["n_boot_dropped"] > 0:
            print(f"    (note: {result['n_boot_dropped']}/{args.n_boot} "
                  f"bootstrap resamples dropped due to degenerate rho)")

    print()
    pooled = pooled_bootstrap(domain_results, n_boot=args.n_boot, seed=args.seed)
    print(f"POOLED (across {pooled['n_domains']} domains):")
    print(f"  Mean Delta_rho = {pooled['pooled_point_estimate']:+.3f}")
    print(f"  95% CI = [{pooled['pooled_ci_95_low']:+.3f}, "
          f"{pooled['pooled_ci_95_high']:+.3f}]")

    output_data = {
        "per_domain": {
            d: {k: v for k, v in r.items() if k != "boot_array"}
            for d, r in domain_results.items()
        },
        "pooled": pooled,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nSaved -> {out_path}")

    print("\n=== LaTeX table rows ===")
    for domain, r in domain_results.items():
        dl = (domain.replace("mystery_blocksworld", "Mystery-Blocksworld")
                     .replace("blocksworld", "Blocksworld")
                     .replace("logistics", "Logistics")
                     .replace("depot", "Depot")
                     .replace("gripper", "Gripper")
                     .replace("rovers", "Rovers")
                     .replace("satellite", "Satellite"))
        sign = "+" if r["point_estimate"] >= 0 else ""
        print(f"{dl} & ${sign}{r['point_estimate']:.3f}$ & "
              f"$[{r['ci_95_low']:+.3f}, {r['ci_95_high']:+.3f}]$ \\\\")
    print(f"\\midrule")
    sign = "+" if pooled["pooled_point_estimate"] >= 0 else ""
    print(f"Pooled & ${sign}{pooled['pooled_point_estimate']:.3f}$ & "
          f"$[{pooled['pooled_ci_95_low']:+.3f}, {pooled['pooled_ci_95_high']:+.3f}]$ \\\\")


if __name__ == "__main__":
    main()
