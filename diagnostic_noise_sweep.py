"""
Diagnostic — Is Fitness Noise Masking (or Reversing) the Search Advantage?
============================================================================

At n_seeds=8 grammar-guided search appeared to beat random search (7/8,
p=0.082). At n_seeds=24 the effect did not hold (both seed-offset
conventions give evolutionary <= random, p>0.05, point estimate favoring
random). This is a real, reportable correction -- but before concluding
"no effect," we test one specific, principled hypothesis: the per-genome
fitness estimate (averaged over only n_gait_seeds=3 CPG rollouts by
default) may be noisy enough that evolutionary search's selection pressure
locks onto genomes that got a LUCKY evaluation rather than a genuinely
good one -- a well-documented failure mode in noisy black-box optimization
("selecting for luck"). Random search doesn't suffer this bias because it
never re-selects based on a noisy estimate.

Testable prediction: if this is the mechanism, increasing n_gait_seeds
(tighter per-genome fitness estimate, at the cost of trying fewer distinct
genomes for the same total physics-call budget) should let evolutionary
search's advantage re-emerge, or at least stop reversing.

This holds the TOTAL number of physics evaluations roughly fixed across
conditions (n_gait_seeds x n_genomes_tried ~= constant) so the comparison
is fair: more averaging per genome necessarily means fewer genomes tried.

Run (from your existing repo root):
    python3 -m experiments.diagnostic_noise_sweep --n_seeds 16
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from experiments.m_search_loop import (Genome, random_genome, mutate,
                                       crossover, evaluate_genome)


def _search_fixed_physics_budget(total_physics_calls: int, n_gait_seeds: int,
                                 seed: int, use_gate: bool = True,
                                 pop_size: int = 8) -> float:
    """Evolutionary search where each genome evaluation costs n_gait_seeds
    physics calls; the number of GENOMES tried is total_physics_calls //
    n_gait_seeds, so total physics cost is held constant across different
    n_gait_seeds settings."""
    n_genomes_budget = max(1, total_physics_calls // n_gait_seeds)
    rng = np.random.default_rng(seed)
    population: List = []

    def evaluate(genome: Genome) -> float:
        fit, _ = evaluate_genome(genome, use_gate=use_gate,
                                 n_gait_seeds=n_gait_seeds)
        return fit

    tried = 0
    while len(population) < pop_size and tried < n_genomes_budget:
        g = random_genome(rng)
        population.append((evaluate(g), g))
        tried += 1

    while tried < n_genomes_budget:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            child = crossover(survivors[i], survivors[j], rng)
        else:
            child = mutate(survivors[rng.integers(len(survivors))], rng)
        population.append((evaluate(child), child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        tried += 1

    return max(f for f, _ in population)


def _random_fixed_physics_budget(total_physics_calls: int, n_gait_seeds: int,
                                 seed: int, use_gate: bool = True) -> float:
    n_genomes_budget = max(1, total_physics_calls // n_gait_seeds)
    rng = np.random.default_rng(seed)
    best = -1e9
    for _ in range(n_genomes_budget):
        g = random_genome(rng)
        fit, _ = evaluate_genome(g, use_gate=use_gate, n_gait_seeds=n_gait_seeds)
        best = max(best, fit)
    return best


def run(n_seeds: int = 16, total_physics_calls: int = 180,
       gait_seed_settings: List[int] = (1, 3, 6, 10),
       verbose: bool = True) -> Dict:
    """total_physics_calls is held fixed across settings; n_gait_seeds=1
    reproduces the original (noisiest) setting at ~3x more genomes tried
    than n_gait_seeds=3's original budget=60 config."""
    results = {}
    for ngs in gait_seed_settings:
        evo_fits, rnd_fits = [], []
        for s in range(n_seeds):
            evo_fits.append(_search_fixed_physics_budget(
                total_physics_calls, ngs, seed=s))
            rnd_fits.append(_random_fixed_physics_budget(
                total_physics_calls, ngs, seed=s + 1000))
        evo_arr, rnd_arr = np.array(evo_fits), np.array(rnd_fits)
        wins = int((evo_arr > rnd_arr).sum())
        try:
            from scipy import stats
            t_stat, p_val = stats.ttest_rel(evo_arr, rnd_arr)
        except Exception:
            t_stat, p_val = float("nan"), float("nan")
        results[ngs] = {"evo_mean": float(evo_arr.mean()),
                        "evo_std": float(evo_arr.std()),
                        "rnd_mean": float(rnd_arr.mean()),
                        "rnd_std": float(rnd_arr.std()),
                        "wins": wins, "t_stat": float(t_stat),
                        "p_val": float(p_val)}
        if verbose:
            n_genomes = total_physics_calls // ngs
            print(f"n_gait_seeds={ngs:2d}  (~{n_genomes} genomes/search)  "
                 f"evo={evo_arr.mean():.3f}+/-{evo_arr.std():.3f}  "
                 f"rnd={rnd_arr.mean():.3f}+/-{rnd_arr.std():.3f}  "
                 f"wins={wins}/{n_seeds}  p={p_val:.3f}")

    if verbose:
        print("\nPrediction check: if fitness noise is the cause, the "
              "evo-vs-rnd gap should INCREASE (or at least stop favoring "
              "random) as n_gait_seeds increases left-to-right above.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=16)
    ap.add_argument("--total_physics_calls", type=int, default=180)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, total_physics_calls=args.total_physics_calls)
