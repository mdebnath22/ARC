"""
TODO #4a — Scaling the Search-Loop Benchmark to Significance
================================================================

Drop this file into your EXISTING repo's experiments/ folder (the one with
gmm/task_fitness.py, ptg/body_grammar.py etc. from earlier sessions) --
it imports evolutionary_search / random_search from your existing
experiments/m_search_loop.py and does NOT redefine the grammar, physics,
or fitness code. This only adds: (a) more seeds, (b) multiprocessing so
more seeds doesn't mean more wall-clock time, (c) a proper effect-size
report alongside the p-value.

Run (from your repo root, i.e. where `python3 -m experiments.m_search_loop`
already works):

    python3 -m experiments.todo4a_scaled_benchmark --n_seeds 24 --budget 60

Requires: your existing repo's dependencies (numpy, scipy, mujoco) --
no new dependencies.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from typing import Dict, Tuple

import numpy as np

# NOTE: this import assumes you run `python3 -m experiments.todo4a_scaled_benchmark`
# from your existing repo root, where experiments/m_search_loop.py already exists.
from experiments.m_search_loop import evolutionary_search, random_search


def _one_seed(args: Tuple[int, int]) -> Tuple[float, float]:
    seed, budget = args
    evo = evolutionary_search(budget, seed=seed, verbose=False)
    rnd = random_search(budget, seed=seed + 10_000, verbose=False)
    return evo["best_fitness"], rnd["best_fitness"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d on the differences (a - b)."""
    diff = a - b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def run(n_seeds: int = 24, budget: int = 60, n_workers: int = 4,
       verbose: bool = True) -> Dict:
    tasks = [(s, budget) for s in range(n_seeds)]
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_one_seed, tasks)
    else:
        results = [_one_seed(t) for t in tasks]

    evo_fits = np.array([r[0] for r in results])
    rnd_fits = np.array([r[1] for r in results])
    wins = int((evo_fits > rnd_fits).sum())

    from scipy import stats
    t_stat, p_val = stats.ttest_rel(evo_fits, rnd_fits)
    d = cohens_d(evo_fits, rnd_fits)
    # bootstrap CI on the mean difference (per the paper's own statistical-
    # honesty standard: a p-value alone is not enough for a marginal effect)
    rng = np.random.default_rng(0)
    diffs = evo_fits - rnd_fits
    boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean()
                 for _ in range(5000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    if verbose:
        print(f"n_seeds={n_seeds}, budget={budget}")
        print(f"evolutionary: {evo_fits.mean():.3f} +/- {evo_fits.std():.3f}")
        print(f"random      : {rnd_fits.mean():.3f} +/- {rnd_fits.std():.3f}")
        print(f"wins: {wins}/{n_seeds}")
        print(f"paired t-test: t={t_stat:.3f}  p={p_val:.4f}")
        print(f"Cohen's d (paired): {d:.3f}")
        print(f"bootstrap 95% CI on mean difference: [{ci_lo:.3f}, {ci_hi:.3f}]")
        sig = "YES" if p_val < 0.05 else "NOT YET"
        print(f"significant at p<0.05: {sig}")
    return {"evo_fits": evo_fits.tolist(), "rnd_fits": rnd_fits.tolist(),
           "wins": wins, "t_stat": float(t_stat), "p_val": float(p_val),
           "cohens_d": d, "ci": (float(ci_lo), float(ci_hi))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=24)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--n_workers", type=int, default=4)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, budget=args.budget, n_workers=args.n_workers)
