"""
TODO #3 — Extending Box Embeddings Past Two Data Points
=========================================================

The paper's box-embedding result (Section 4.4) has exactly ONE override/
general rule pair per grammar, so "volume(override) < volume(general)" is a
two-point comparison, not a distribution. This script adds several more
override rules to a small synthetic rule catalogue (mirroring the paper's
manipulation grammar's structure) and reports the volume-ordering claim as
a Spearman rank correlation between specificity and (negative) log-volume
over N>5 pairs, which is the statistic the paper's TODO calls for.

Design: each rule is defined by (required_features, specificity). A higher
specificity rule requires a strict superset of a lower-specificity rule's
features -- exactly the paper's override relation. We add 6 override pairs
across 3 "families" (APPROACH, GRASP, PLACE), each with a 2-level or
3-level specificity chain, to get 6 (specific, general) pairs total.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .box_embed_np import BoxBankNP

# Synthetic extended catalogue: (rule_id, required_feature_indices, specificity)
# Feature space: 8 binary features (roles/markers), rules require subsets.
RULES: List[Tuple[str, List[int], int]] = [
    # APPROACH family: generic -> tool -> tool+fragile (3-level chain)
    ("APPROACH_GENERIC", [0, 1], 1),
    ("APPROACH_TOOL", [0, 1, 2], 2),
    ("APPROACH_TOOL_FRAGILE", [0, 1, 2, 3], 3),
    # GRASP family: generic -> two-finger -> suction (parallel overrides)
    ("GRASP_GENERIC", [0, 4], 1),
    ("GRASP_TWOFINGER", [0, 4, 5], 2),
    ("GRASP_SUCTION", [0, 4, 6], 2),
    # PLACE family: generic -> precision (2-level)
    ("PLACE_GENERIC", [0, 7], 1),
    ("PLACE_PRECISION", [0, 7, 2], 2),
]
FEATURE_DIM = 8

# (specific, general) pairs implied by the chains above
OVERRIDE_PAIRS = [
    ("APPROACH_TOOL", "APPROACH_GENERIC"),
    ("APPROACH_TOOL_FRAGILE", "APPROACH_TOOL"),
    ("APPROACH_TOOL_FRAGILE", "APPROACH_GENERIC"),  # transitive pair too
    ("GRASP_TWOFINGER", "GRASP_GENERIC"),
    ("GRASP_SUCTION", "GRASP_GENERIC"),
    ("PLACE_PRECISION", "PLACE_GENERIC"),
]


def rule_feature_vector(feature_idx: List[int]) -> np.ndarray:
    v = np.zeros(FEATURE_DIM)
    v[feature_idx] = 1.0
    return v


def sample_states(rng: np.random.Generator, n: int) -> List[Tuple[str, np.ndarray]]:
    """Sample states that satisfy at least one rule's requirements, and
    record which rule SHOULD fire (highest specificity among satisfied)."""
    fires = []
    for _ in range(n):
        active = (rng.random(FEATURE_DIM) < 0.4).astype(float)
        satisfied = [(rid, spec) for rid, feats, spec in RULES
                    if np.all(active[feats] == 1.0)]
        if not satisfied:
            continue
        # highest-specificity rule wins (paper's arbitration rule)
        winner = max(satisfied, key=lambda x: x[1])[0]
        fires.append((winner, active.copy()))
    return fires


def run(n_states: int = 500, epochs: int = 300, seed: int = 0,
       verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    rule_ids = [r[0] for r in RULES]
    bank = BoxBankNP(rule_ids, dim=FEATURE_DIM, seed=seed)
    # initialize centers at each rule's own feature prototype
    for rid, feats, _ in RULES:
        bank.center[bank.index[rid]] = rule_feature_vector(feats)

    fires = sample_states(rng, n_states)
    if verbose:
        print(f"training on {len(fires)} fires across {len(RULES)} rules, "
              f"{len(OVERRIDE_PAIRS)} override pairs")

    for ep in range(epochs):
        loss = bank.train_step(fires, OVERRIDE_PAIRS, lr=0.05)
        if verbose and (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:4d}  loss={loss:.4f}")

    volumes = bank.all_log_volumes()
    specificities = {rid: spec for rid, _, spec in RULES}

    # rank correlation: higher specificity should mean LOWER log-volume
    ids = list(volumes.keys())
    vol_vals = np.array([volumes[i] for i in ids])
    spec_vals = np.array([specificities[i] for i in ids])
    # Spearman via rank + Pearson on ranks (no scipy dependency)
    def rankdata(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
    r_vol = rankdata(vol_vals)
    r_spec = rankdata(-spec_vals)  # negate: higher spec should rank as LOWER volume
    spearman = float(np.corrcoef(r_vol, r_spec)[0, 1])

    # per-pair containment check
    contain_results = []
    for spec_id, gen_id in OVERRIDE_PAIRS:
        lo_s, hi_s = bank.bounds(spec_id)
        lo_g, hi_g = bank.bounds(gen_id)
        holds = bool(np.all(lo_s >= lo_g - 1e-4) and np.all(hi_s <= hi_g + 1e-4))
        vol_ok = volumes[spec_id] < volumes[gen_id]
        contain_results.append({"pair": f"{spec_id} < {gen_id}",
                               "containment_holds": holds, "vol_ok": vol_ok})

    if verbose:
        print(f"\nVolume-ordering rank correlation (Spearman, "
              f"specificity vs -log-volume): {spearman:.3f}  (n={len(RULES)} rules, "
              f"{len(OVERRIDE_PAIRS)} pairs)")
        print(f"\n{'pair':<45}{'containment':>13}{'vol order':>11}")
        for r in contain_results:
            print(f"{r['pair']:<45}{'OK' if r['containment_holds'] else 'FAIL':>13}"
                  f"{'OK' if r['vol_ok'] else 'FAIL':>11}")
        n_ok = sum(r["containment_holds"] and r["vol_ok"] for r in contain_results)
        print(f"\n{n_ok}/{len(contain_results)} pairs satisfy both checks")
    return {"spearman": spearman, "pairs": contain_results, "volumes": volumes}


if __name__ == "__main__":
    run()
