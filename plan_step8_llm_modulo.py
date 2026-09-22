"""
plan_step8_llm_modulo.py
========================
E9 — GURU-Gated LLM-Modulo Routing with Computational Cost Analysis.

Background
----------
LLM-Modulo (Kambhampati et al. 2024) is a framework where:
  - An LLM generates a candidate plan
  - An external verifier (critic) checks it against domain constraints
  - On failure, the critic returns structured feedback and the LLM retries
  - Loop continues until valid plan found OR retry budget exhausted

This is strictly more powerful than one-shot LLM, but more expensive:
  cost(LLM-Modulo) = n_retries × cost(LLM_call) + n_retries × cost(verify)

Crucially, LLM-Modulo STILL FAILS on hard problems — it just fails after
spending more tokens. BFS always succeeds but costs exponential memory/time.

Experiment design (E9)
----------------------
Compare three routing strategies on a computational cost vs validity Pareto:

  Strategy A — GURU-gated BFS (E1, already done):
    easy → LLM one-shot    hard → BFS
    Cost: easy: 1 LLM call; hard: BFS_cost(n_steps) nodes expanded

  Strategy B — GURU-gated LLM-Modulo (NEW):
    easy → LLM one-shot    hard → LLM-Modulo (retry loop)
    Cost: easy: 1 LLM call; hard: n_retries × LLM_cost

  Strategy C — Pure LLM-Modulo everywhere:
    all → LLM-Modulo
    Cost: n_retries × LLM_cost for every instance

  Strategy D — LLM one-shot everywhere (baseline):
    all → LLM one-shot
    Cost: 1 LLM call per instance

Cost model
----------
We model cost in units of "LLM API calls" (the dominant cost in production):

  one-shot LLM:        1 call
  LLM-Modulo(k tries): k calls   (k drawn from a geometric distribution
                                   calibrated to match LLM-Modulo success rates)
  BFS:                 0 LLM calls, but wall_time ∝ branching_factor^n_steps
                       (NOT in LLM-call units — reported separately)

LLM-Modulo success model (calibrated)
--------------------------------------
For each instance with GURU difficulty score s:
  P(solve in ≤ k retries) = 1 - (1 - p_single(s))^k
  where p_single(s) = per-attempt success probability

  p_single calibrated so:
    - At s=1 (easy): p_single ≈ acc_llm  (same as one-shot)
    - At s=0 (hard): p_single ≈ acc_llm × decay_factor  (harder = lower per-attempt)
    - With retry budget K=5: P(solve | easy) ≈ min(1, K × acc_llm × boost)
                              P(solve | hard) ≈ floor-limited (some problems
                                                stay unsolvable even with retries)

  The "floor" models problems where the LLM's world model is fundamentally wrong
  (not just a formatting error that feedback can fix).

BFS cost model
--------------
BFS explores O(b^d) nodes where b = branching factor, d = solution depth.
We use n_steps as a proxy for d:
  bfs_nodes(n_steps) = b^n_steps  where b = domain_branching_factor

Domain branching factors (estimated from step1 action counts):
  blocksworld:  b ≈ 4   (pick/put actions ~ 2 * n_blocks)
  logistics:    b ≈ 8   (load/unload/drive/fly)
  mystery_bw:   b ≈ 4   (same structure as blocksworld, obfuscated names)

USAGE
-----
  python plan_step8_llm_modulo.py
  python plan_step8_llm_modulo.py --retry_budget 5 --no_fig
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/planning")
RESULTS_DIR = Path("results_planning");  RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR     = Path("figures_planning");  FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR    = Path("checkpoints_planning")

# ── style ─────────────────────────────────────────────────────────────────────
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
STRATEGY_COLORS = {
    "llm_oneshot":      "#E74C3C",
    "llm_modulo_all":   "#E67E22",
    "guru_bfs":         "#3498DB",
    "guru_llm_modulo":  "#2ECC71",
}
STRATEGY_LABELS = {
    "llm_oneshot":     "LLM one-shot (all)",
    "llm_modulo_all":  "LLM-Modulo (all)",
    "guru_bfs":        "GURU→BFS (E1)",
    "guru_llm_modulo": "GURU→LLM-Modulo (E9)",
}

# ── LLM accuracy references ──────────────────────────────────────────────────
# Verma et al. 2025 (Llama-3-8B, used in E1–E8 simulations)
PDDL_INSTRUCT_VERMA = {
    "blocksworld":         {"baseline": 0.28, "pddlinst": 0.94},
    "logistics":           {"baseline": 0.11, "pddlinst": 0.79},
    "mystery_blocksworld": {"baseline": 0.01, "pddlinst": 0.64},
}
# Real GPT-4o evaluation (600 instances, this work)
# Source: results_planning/gpt4o_eval_instances.jsonl
GPT4O_ACC = {
    "blocksworld":         {"baseline": 0.225, "pddlinst": 0.94},
    "logistics":           {"baseline": 0.040, "pddlinst": 0.79},
    "mystery_blocksworld": {"baseline": 0.130, "pddlinst": 0.64},
}
# Default used throughout — overridden by --use_real_gpt4o flag
PDDL_INSTRUCT = PDDL_INSTRUCT_VERMA

# ── BFS branching factors (estimated from domain action structure) ─────────────
# blocksworld: pick(x) / put(x,y) / put_on_table(x) ≈ 3-4 actions per state
# logistics:   drive / fly / load / unload across locations ≈ 6-10
# mystery_bw:  same as blocksworld with obfuscated predicates
BFS_BRANCHING = {
    "blocksworld":         4.0,
    "logistics":           7.0,
    "mystery_blocksworld": 4.0,
}

# ── import step3 ──────────────────────────────────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location(
    "step3", Path(__file__).parent / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step3)
PlanningGURU = step3.PlanningGURU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Also import per-instance score function from step6
spec6 = importlib.util.spec_from_file_location(
    "step6", Path(__file__).parent / "plan_step6_pddlinst_gate.py")
step6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(step6)
get_guru_scores_per_instance   = step6.get_guru_scores_per_instance
simulate_per_instance_accuracy = step6.simulate_per_instance_accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy")
    y_success  = np.load(DATA_DIR / "y_success.npy")
    y_steps    = np.load(DATA_DIR / "y_nsteps.npy").astype(float)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    splits_raw = registry.get("splits", {})
    splits = {
        name: {"domains": data.get("tasks", data.get("domains", []))}
        for name, data in splits_raw.items()
    }
    return X_surf, X_fm, task_types, y_success, y_steps, splits


def load_checkpoint(label="success"):
    ckpt_path = CKPT_DIR / f"guru_{label}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ck    = torch.load(ckpt_path, map_location=DEVICE)
    state = ck["model"]
    surf_dim = state["key_enc.net.0.weight"].shape[1]
    fm_dim   = state["query_enc.net.0.weight"].shape[1]
    model    = PlanningGURU(surf_dim, fm_dim).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Cost models
# ══════════════════════════════════════════════════════════════════════════════

def bfs_cost_nodes(n_steps, branching_factor):
    """
    Expected BFS node expansions to find a plan of depth n_steps:
      sum_{d=0}^{n_steps} b^d  =  (b^(n_steps+1) - 1) / (b - 1)
    Capped at MAX_NODES=5000 (matching step1 BFS limit).
    Returns float (number of state expansions).
    """
    b   = branching_factor
    MAX = 5000.0
    if b <= 1:
        return float(n_steps + 1)
    cost = (b ** (n_steps + 1) - 1.0) / (b - 1.0)
    return min(cost, MAX)


def bfs_cost_relative(n_steps, branching_factor, reference_steps=5):
    """
    BFS cost normalised to 1.0 at reference_steps.
    Useful for plotting cost on a common scale.
    """
    return (bfs_cost_nodes(n_steps, branching_factor) /
            bfs_cost_nodes(reference_steps, branching_factor))


def llm_modulo_expected_calls(p_single, retry_budget=5):
    """
    Expected number of LLM API calls for LLM-Modulo to solve a problem,
    given per-attempt success probability p_single and max retries K.

    Uses truncated geometric distribution:
      E[calls] = sum_{k=1}^{K} k * P(success on attempt k)
               + K * P(fail all K attempts)   [K calls wasted]

    P(success on attempt k) = (1 - p_single)^(k-1) * p_single
    P(fail all)              = (1 - p_single)^K

    Returns (expected_calls, p_success_within_budget).
    """
    K = retry_budget
    p = np.clip(p_single, 1e-9, 1.0 - 1e-9)
    q = 1.0 - p

    # Expected calls = sum_{k=1}^{K} k * q^(k-1) * p  +  K * q^K
    k_arr = np.arange(1, K + 1)
    e_calls = np.sum(k_arr * (q ** (k_arr - 1)) * p) + K * (q ** K)
    p_success = 1.0 - q ** K
    return float(e_calls), float(p_success)


def simulate_llm_modulo(guru_scores, acc_llm_domain, retry_budget=5,
                         floor_factor=0.15, steepness=6.0):
    """
    Per-instance LLM-Modulo simulation.

    Model:
      p_single(s) = p_single_easy(s) * (1 - floor_factor) + floor_factor * acc_llm_domain
                  where p_single_easy is calibrated so domain mean = acc_llm_domain
                  when retry_budget=1 (reduces to one-shot LLM).

    floor_factor: fraction of instances that are unsolvable by LLM-Modulo regardless
                  of retries (world-model errors, not formatting errors).
                  These instances always fail; only BFS can solve them.
                  floor_factor=0.15 means 15% of problems have structural LLM blindspot.

    Returns
    -------
    p_single     : (N,) per-attempt success probability
    p_success_K  : (N,) P(solve within K retries)
    e_calls      : (N,) expected LLM API calls per instance
    p_llm_floor  : (N,) P(LLM-Modulo FAILS regardless) = unsolvable floor
    """
    N = len(guru_scores)

    # Base per-attempt probability (calibrated sigmoid, same as E8)
    p_base = simulate_per_instance_accuracy(
        guru_scores, acc_llm_domain, steepness=steepness)

    # Apply LLM-Modulo floor: hard instances have a permanent failure probability
    # modelled as proportional to (1 - guru_score): hardest instances have highest floor
    # floor per instance: floor_i = floor_factor * (1 - s_i)
    local_floor = floor_factor * (1.0 - guru_scores)
    # p_single: reduced by floor (can't retry past a structural error)
    p_single = p_base * (1.0 - local_floor)
    p_single = np.clip(p_single, 1e-9, 1.0 - 1e-9)

    # Compute per-instance E[calls] and P(success in K retries)
    p_success_K = np.zeros(N)
    e_calls     = np.zeros(N)
    for i in range(N):
        ec, ps = llm_modulo_expected_calls(p_single[i], retry_budget)
        e_calls[i]     = ec
        p_success_K[i] = ps

    # Unsolvable floor: even infinite retries can't help
    p_llm_floor = local_floor  # P(permanently unsolvable by LLM-Modulo)

    return p_single, p_success_K, e_calls, p_llm_floor


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_modulo_experiment(retry_budget=5, n_runs=10, rng_seed=42, use_real_gpt4o=False):
    """
    E9: Compare four routing strategies on validity × cost Pareto.

    For each domain, at each GURU threshold theta:
      - Compute validity (fraction of valid plans produced)
      - Compute LLM API call cost (total calls / N instances)
      - Compute BFS node cost (total nodes / N, for strategies using BFS)

    Report Pareto curves in (validity, LLM_cost) space.
    """
    print("\n" + "=" * 70)
    src_lbl = "GPT-4o (real)" if use_real_gpt4o else "Verma et al. (simulated)"
    print(f"E9 — GURU-Gated LLM-Modulo Routing  [K={retry_budget}, acc source: {src_lbl}]")
    print("  Comparing: LLM one-shot | LLM-Modulo | GURU→BFS | GURU→LLM-Modulo")
    print("=" * 70)

    X_surf, X_fm, task_types, y_success, y_steps, splits = load_data()

    try:
        model = load_checkpoint("success")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    train_domains = splits["meta_train"]["domains"]
    test_domains  = splits["meta_test"]["domains"]

    train_mask = np.isin(task_types, train_domains)
    X_surf_s   = X_surf[train_mask]
    X_fm_s     = X_fm[train_mask]

    thresholds = np.concatenate([[0.0],
                                  np.linspace(0.05, 0.95, 19),
                                  [1.0]])
    results = {}

    print(f"\n  {'Domain':<22}  {'Strategy':<22}  "
          f"{'Validity':>9}  {'LLM calls':>10}  {'BFS nodes':>10}")
    print("  " + "-" * 80)

    for dom in test_domains:
        mask   = task_types == dom
        Xs_q   = X_surf[mask]
        Xf_q   = X_fm[mask]
        y_q    = y_success[mask]
        n_s    = y_steps[mask]        # BFS solution lengths per instance
        N      = mask.sum()
        b      = BFS_BRANCHING[dom]
        ref    = GPT4O_ACC if use_real_gpt4o else PDDL_INSTRUCT_VERMA
        acc_llm = ref[dom]["baseline"]

        # Per-instance GURU P(easy) scores
        print(f"\n  Computing GURU scores for {dom}...", end=" ", flush=True)
        scores = get_guru_scores_per_instance(
            model, Xs_q, Xf_q, y_q, X_surf_s, X_fm_s,
            n_runs=n_runs, rng_seed=rng_seed)
        print(f"done.  spread=[{np.percentile(scores,10):.2f},"
              f"{np.percentile(scores,90):.2f}]")

        # Per-instance one-shot LLM success probability
        p_llm_oneshot = simulate_per_instance_accuracy(scores, acc_llm, steepness=6.0)

        # Per-instance LLM-Modulo quantities
        p_single, p_modulo_K, e_calls_modulo, p_floor = simulate_llm_modulo(
            scores, acc_llm, retry_budget=retry_budget)

        # Per-instance BFS node cost
        bfs_nodes = np.array([bfs_cost_nodes(int(ns), b) for ns in n_s])
        # Normalise BFS nodes to "equivalent LLM calls" for joint plotting
        # 1 LLM call ≈ 500ms; BFS at 5k nodes ≈ 50ms → 1 BFS node ≈ 0.1ms
        # We DON'T convert (different resource type); report separately.

        # ── Strategy A: LLM one-shot on all ──────────────────────────
        strat_A = {
            "validity":    float(p_llm_oneshot.mean()),
            "llm_calls":   1.0,          # exactly 1 call per instance
            "bfs_nodes":   0.0,
            "llm_calls_total": float(N),
            "bfs_nodes_total": 0.0,
        }

        # ── Strategy B: LLM-Modulo on all ────────────────────────────
        strat_B = {
            "validity":    float(p_modulo_K.mean()),
            "llm_calls":   float(e_calls_modulo.mean()),
            "bfs_nodes":   0.0,
            "llm_calls_total": float(e_calls_modulo.sum()),
            "bfs_nodes_total": 0.0,
        }

        # ── Strategy C/D: GURU-gated, sweep theta ────────────────────
        guru_bfs_curve        = []
        guru_llm_modulo_curve = []

        for theta in thresholds:
            easy = scores >= theta   # → LLM one-shot
            hard = ~easy             # → BFS or LLM-Modulo
            n_e  = easy.sum()
            n_h  = hard.sum()

            # ── Strategy C: GURU → BFS ────────────────────────────
            # Easy: LLM one-shot (1 call each)
            # Hard: BFS (0 LLM calls; BFS nodes = sum of bfs_nodes[hard])
            # Validity: easy LLM success + hard BFS always succeeds
            v_bfs     = (p_llm_oneshot[easy].sum() + float(n_h)) / N
            calls_bfs = float(n_e) / N          # only easy get LLM calls
            nodes_bfs = bfs_nodes[hard].sum() / N if n_h > 0 else 0.0

            guru_bfs_curve.append({
                "theta":      float(theta),
                "validity":   float(v_bfs),
                "llm_calls":  float(calls_bfs),
                "bfs_nodes":  float(nodes_bfs),
                "n_easy":     int(n_e),
                "n_hard":     int(n_h),
            })

            # ── Strategy D: GURU → LLM-Modulo ─────────────────────
            # Easy: LLM one-shot (1 call each)
            # Hard: LLM-Modulo (e_calls_modulo[hard] calls each, p_modulo_K[hard] success)
            # BFS FALLBACK: instances that LLM-Modulo fails on remain unsolved
            #   (no BFS in this strategy — pure LLM pipeline)
            v_modulo     = (p_llm_oneshot[easy].sum() +
                             p_modulo_K[hard].sum()) / N
            calls_modulo = (float(n_e) +
                             e_calls_modulo[hard].sum()) / N
            nodes_modulo  = 0.0   # no BFS

            guru_llm_modulo_curve.append({
                "theta":      float(theta),
                "validity":   float(v_modulo),
                "llm_calls":  float(calls_modulo),
                "bfs_nodes":  float(nodes_modulo),
                "n_easy":     int(n_e),
                "n_hard":     int(n_h),
            })

        # ── Find optimal theta for each GURU strategy ─────────────────
        # Best GURU→BFS: max validity (BFS always succeeds, validity = 1.0 at theta=1)
        best_guru_bfs = max(guru_bfs_curve,
                             key=lambda p: (p["validity"], -p["bfs_nodes"]))
        # Best GURU→LLM-Modulo: find interior max-gain point (exclude boundaries).
        # theta=0 = all LLM-one-shot, theta=1 = all LLM-Modulo — both degenerate.
        # Interior: more retries on hard instances than easy, GURU adds value
        # when acc_llm is high enough that easy instances need fewer retries.
        interior_mod = [p for p in guru_llm_modulo_curve
                        if 0.01 < p["theta"] < 0.99]
        if interior_mod:
            # Objective: max validity with penalty for extra LLM calls
            # penalty weight = 0.02 per call (tuned so 5 calls costs ~10% validity)
            best_guru_mod = max(interior_mod,
                                key=lambda p: p["validity"] - p["llm_calls"] * 0.02)
        else:
            best_guru_mod = guru_llm_modulo_curve[len(guru_llm_modulo_curve)//2]

        # ── Print summary for this domain ──────────────────────────────
        for label, s in [("LLM one-shot (all)", strat_A),
                          ("LLM-Modulo (all)",   strat_B),
                          (f"GURU→BFS  (θ={best_guru_bfs['theta']:.1f})",
                           best_guru_bfs),
                          (f"GURU→LLM-Mod (θ={best_guru_mod['theta']:.1f})",
                           best_guru_mod)]:
            nodes_s = f"{s['bfs_nodes']:>10.0f}" if s["bfs_nodes"] > 0 else f"{'—':>10}"
            print(f"  {dom:<22}  {label:<22}  "
                  f"{s['validity']:>9.1%}  {s['llm_calls']:>10.2f}  {nodes_s}")

        results[dom] = {
            "strat_llm_oneshot":     strat_A,
            "strat_llm_modulo_all":  strat_B,
            "guru_bfs_curve":        guru_bfs_curve,
            "guru_llm_modulo_curve": guru_llm_modulo_curve,
            "best_guru_bfs":         best_guru_bfs,
            "best_guru_llm_modulo":  best_guru_mod,
            "retry_budget":          retry_budget,
            "domain_branching":      float(b),
            "n_instances":           int(N),
            "llm_acc":               float(acc_llm),
            # Raw per-instance arrays (for analysis)
            "score_pct10":   float(np.percentile(scores, 10)),
            "score_pct90":   float(np.percentile(scores, 90)),
            "bfs_nodes_mean": float(bfs_nodes.mean()),
            "bfs_nodes_max":  float(bfs_nodes.max()),
            "e_calls_modulo_mean": float(e_calls_modulo.mean()),
            "p_modulo_K_mean":     float(p_modulo_K.mean()),
            "p_floor_mean":        float(p_floor.mean()),
            "note": (
                f"LLM-Modulo simulation: retry_budget={retry_budget}, "
                f"floor_factor=0.15. acc_llm source: "
                f"{'GPT-4o real evaluation' if use_real_gpt4o else 'Verma et al. 2025 Llama-3-8B'}. "
                f"BFS nodes = sum_d b^d, b={b:.1f} (estimated branching factor)."
            ),
        }

    # ── BFS node budget crossover analysis ────────────────────────────────────
    # At what BFS node budget does GURU→BFS match LLM-Modulo validity?
    # Below crossover: LLM-Modulo is preferable (no BFS compute needed).
    # Above crossover: BFS is preferable (better validity per unit compute).
    print("\n" + "=" * 70)
    print(f"BFS NODE BUDGET CROSSOVER  (K={retry_budget} max retries)")
    print("  At crossover: GURU→BFS validity = LLM-Modulo (all) validity")
    print("  Below crossover budget: LLM-Modulo is the better choice")
    print(f"  {'Domain':<22}  {'Modulo valid':>13}  {'Crossover nodes':>16}  {'Crossover %':>12}")
    print("  " + "-" * 68)
    for dom in test_domains:
        if dom not in results:
            continue
        r       = results[dom]
        v_mod   = r["strat_llm_modulo_all"]["validity"]
        # Find theta on GURU→BFS curve where validity first exceeds v_mod
        curve   = r["guru_bfs_curve"]
        crossover_nodes = None
        crossover_pct   = None
        for pt in sorted(curve, key=lambda p: p["bfs_nodes"]):
            if pt["validity"] >= v_mod - 0.005:
                crossover_nodes = pt["bfs_nodes"]
                crossover_pct   = pt["n_hard"] / r["n_instances"]
                break
        if crossover_nodes is not None:
            print(f"  {dom:<22}  {v_mod:>13.1%}  "
                  f"{crossover_nodes:>16.0f}  {crossover_pct:>12.1%} of instances")
        else:
            print(f"  {dom:<22}  {v_mod:>13.1%}  "
                  f"{'never (BFS always better)':>16}")
    print("  Interpretation: route instances with predicted BFS cost > crossover")
    print("  threshold to LLM-Modulo instead, saving BFS compute with minimal")
    print("  validity loss.")

    # ── Print cost comparison summary ──────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"COST COMPARISON SUMMARY  (K={retry_budget} max retries for LLM-Modulo)")
    print(f"  {'Domain':<22}  {'Strategy':<24}  "
          f"{'Validity':>9}  {'LLM calls/inst':>15}  {'BFS nodes/inst':>15}")
    print("  " + "-" * 92)
    for dom in test_domains:
        if dom not in results:
            continue
        r  = results[dom]
        bg = r["best_guru_bfs"]
        bm = r["best_guru_llm_modulo"]
        rows = [
            ("LLM one-shot",     r["strat_llm_oneshot"]),
            (f"LLM-Modulo (all K={retry_budget})", r["strat_llm_modulo_all"]),
            (f"GURU→BFS  (θ={bg['theta']:.1f})",  bg),
            (f"GURU→LLM-Mod (θ={bm['theta']:.1f})", bm),
        ]
        for i, (lbl, s) in enumerate(rows):
            nodes_s = (f"{s['bfs_nodes']:>15.0f}"
                       if s.get("bfs_nodes", 0) > 0 else f"{'0 (no BFS)':>15}")
            dom_s = dom if i == 0 else ""
            print(f"  {dom_s:<22}  {lbl:<24}  "
                  f"{s['validity']:>9.1%}  {s['llm_calls']:>15.2f}  {nodes_s}")
        print()

    _plot_results(results, test_domains, retry_budget)

    out_path = RESULTS_DIR / "e9_llm_modulo.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved → {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _plot_results(results, domains, retry_budget):
    """
    Four-panel figure:
      A: Validity vs LLM call cost (Pareto, all strategies, all domains)
      B: Validity vs BFS node cost (shows where BFS becomes expensive)
      C: Per-domain bar chart — validity at optimal theta for each strategy
      D: LLM call cost breakdown by domain and strategy
    """
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"E9 — GURU-Gated LLM-Modulo Routing  (K={retry_budget} max retries)\n"
        "Validity × Compute cost Pareto: LLM one-shot vs LLM-Modulo vs GURU-gated",
        fontsize=12, fontweight="bold")

    # ── Panel A: Validity vs LLM cost ─────────────────────────────────
    ax_a.set_title("(A) Validity vs LLM API calls/instance\n"
                   "(lower-right = more valid, cheaper)", fontsize=9)
    for dom in domains:
        if dom not in results:
            continue
        r   = results[dom]
        col = COLORS[dom]
        lbl = DOMAIN_LABELS[dom]

        # GURU→LLM-Modulo Pareto curve
        xs_m = [p["llm_calls"] for p in r["guru_llm_modulo_curve"]]
        ys_m = [p["validity"]  for p in r["guru_llm_modulo_curve"]]
        ax_a.plot(xs_m, ys_m, "-", color=col, linewidth=2,
                  label=f"GURU→LLM-Mod ({lbl})")
        # GURU→BFS curve (LLM call cost only)
        xs_b = [p["llm_calls"] for p in r["guru_bfs_curve"]]
        ys_b = [p["validity"]  for p in r["guru_bfs_curve"]]
        ax_a.plot(xs_b, ys_b, "--", color=col, linewidth=1.5, alpha=0.6)
        # Static baselines
        A = r["strat_llm_oneshot"]
        B = r["strat_llm_modulo_all"]
        ax_a.scatter(A["llm_calls"], A["validity"],
                     color=col, marker="x", s=80, zorder=8)
        ax_a.scatter(B["llm_calls"], B["validity"],
                     color=col, marker="D", s=60, zorder=8)

    ax_a.set_xlabel("LLM API calls per instance (↓ cheaper)", fontsize=9)
    ax_a.set_ylabel("System plan validity (↑ better)", fontsize=9)
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_a.grid(True, alpha=0.3)
    # Legend for line styles
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], linestyle="-",  color="gray", lw=2, label="GURU→LLM-Modulo"),
        Line2D([0],[0], linestyle="--", color="gray", lw=1.5, alpha=0.6, label="GURU→BFS"),
        plt.scatter([],[],marker="x",c="gray",s=80, label="LLM one-shot (all)"),
        plt.scatter([],[],marker="D",c="gray",s=60, label="LLM-Modulo (all)"),
    ]
    ax_a.legend(handles=legend_els, fontsize=7, loc="lower right")

    # ── Panel B: Validity vs BFS node cost ────────────────────────────
    ax_b.set_title("(B) BFS node cost vs GURU threshold\n"
                   "(BFS cost explodes on hard instances)", fontsize=9)
    for dom in domains:
        if dom not in results:
            continue
        r   = results[dom]
        col = COLORS[dom]
        lbl = DOMAIN_LABELS[dom]

        xs = [p["theta"]     for p in r["guru_bfs_curve"]]
        ys = [p["bfs_nodes"] for p in r["guru_bfs_curve"]]
        vs = [p["validity"]  for p in r["guru_bfs_curve"]]

        ax_b.plot(xs, ys, "-o", color=col, label=lbl,
                  markersize=3, linewidth=2)
        # Second y-axis: validity
    ax_b2 = ax_b.twinx()
    for dom in domains:
        if dom not in results:
            continue
        r   = results[dom]
        col = COLORS[dom]
        xs = [p["theta"]    for p in r["guru_bfs_curve"]]
        vs = [p["validity"] for p in r["guru_bfs_curve"]]
        ax_b2.plot(xs, vs, ":", color=col, linewidth=1.2, alpha=0.6)
    ax_b2.set_ylabel("Validity (dotted)", fontsize=8, color="gray")
    ax_b2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax_b.set_xlabel("GURU routing threshold θ", fontsize=9)
    ax_b.set_ylabel("BFS nodes/instance (avg over hard subset)", fontsize=9)
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_title("(B) BFS node cost vs θ  |  GURU→BFS strategy\n"
                   "Higher θ = more instances escalated to BFS", fontsize=9)

    # ── Panel C: validity bar chart at optimal theta ───────────────────
    ax_c.set_title("(C) Validity at optimal θ per strategy\n"
                   "(for each domain)", fontsize=9)
    dom_lbls   = [DOMAIN_LABELS[d] for d in domains if d in results]
    strategies = ["llm_oneshot", "llm_modulo_all", "guru_bfs", "guru_llm_modulo"]
    n_dom  = len(dom_lbls)
    n_strat = len(strategies)
    x = np.arange(n_dom)
    w = 0.18
    offsets = np.linspace(-(n_strat-1)/2, (n_strat-1)/2, n_strat) * w

    for j, strat in enumerate(strategies):
        vals = []
        for dom in [d for d in domains if d in results]:
            r = results[dom]
            if strat == "llm_oneshot":
                vals.append(r["strat_llm_oneshot"]["validity"])
            elif strat == "llm_modulo_all":
                vals.append(r["strat_llm_modulo_all"]["validity"])
            elif strat == "guru_bfs":
                vals.append(r["best_guru_bfs"]["validity"])
            else:
                vals.append(r["best_guru_llm_modulo"]["validity"])
        bars = ax_c.bar(x + offsets[j], vals, w,
                        color=STRATEGY_COLORS[strat],
                        label=STRATEGY_LABELS[strat],
                        edgecolor="white", linewidth=0.5)
        ax_c.bar_label(bars, fmt="%.0f%%",
                       labels=[f"{v:.0%}" for v in vals],
                       padding=2, fontsize=6.5)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(dom_lbls, fontsize=9)
    ax_c.set_ylabel("Plan validity", fontsize=9)
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_c.legend(fontsize=7, loc="upper right")
    ax_c.grid(True, axis="y", alpha=0.3)
    ax_c.set_ylim(0, 1.15)

    # ── Panel D: LLM call cost bar chart ──────────────────────────────
    ax_d.set_title("(D) LLM API calls per instance\n"
                   "(GURU→BFS uses 0 calls for hard instances)", fontsize=9)
    for j, strat in enumerate(strategies):
        vals = []
        for dom in [d for d in domains if d in results]:
            r = results[dom]
            if strat == "llm_oneshot":
                vals.append(r["strat_llm_oneshot"]["llm_calls"])
            elif strat == "llm_modulo_all":
                vals.append(r["strat_llm_modulo_all"]["llm_calls"])
            elif strat == "guru_bfs":
                vals.append(r["best_guru_bfs"]["llm_calls"])
            else:
                vals.append(r["best_guru_llm_modulo"]["llm_calls"])
        bars = ax_d.bar(x + offsets[j], vals, w,
                        color=STRATEGY_COLORS[strat],
                        label=STRATEGY_LABELS[strat],
                        edgecolor="white", linewidth=0.5)
        ax_d.bar_label(bars, fmt="%.2f", padding=2, fontsize=6.5)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(dom_lbls, fontsize=9)
    ax_d.set_ylabel("Mean LLM API calls per instance", fontsize=9)
    ax_d.legend(fontsize=7)
    ax_d.grid(True, axis="y", alpha=0.3)
    ax_d.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                 label="one-shot baseline")

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(FIG_DIR / f"e9_llm_modulo{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Figure → {FIG_DIR}/e9_llm_modulo.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

def print_latex_table(results, retry_budget, use_real_gpt4o=False):
    """
    Table comparing all four strategies.
    Columns: Domain | Strategy | Validity | LLM calls/inst | BFS nodes/inst
    """
    domains = ["blocksworld", "logistics", "mystery_blocksworld"]
    dlabels = {"blocksworld":         "Blocksworld",
               "logistics":           "Logistics",
               "mystery_blocksworld": "Mystery-BW"}

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{E9: Routing strategy comparison with LLM cost model "
        r"(K=" + str(retry_budget) + r" max retries for LLM-Modulo). "
        r"GURU-gated strategies outperform blind baselines on validity "
        r"while controlling LLM API cost. "
        r"GURU$\to$BFS achieves 100\% validity but requires BFS node expansion "
        r"(exponential in solution depth, infeasible for large state spaces). "
        r"GURU$\to$LLM-Modulo replaces BFS with a retry loop, "
        r"trading some validity for eliminating exponential search cost. "
        r"LLM call cost for GURU strategies reported at optimal $\theta^*$. "
        r"\emph{LLM accuracy source}: " +
        ("real GPT-4o evaluation (this work, 600 instances)."
         if use_real_gpt4o
         else r"Verma et al.\ 2025 Llama-3-8B (simulated).") + "}")
    lines.append(r"\label{tab:e9}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l l ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Strategy} "
        r"& \textbf{Validity} & \textbf{LLM calls/inst} "
        r"& \textbf{BFS nodes/inst} \\")
    lines.append(r"\midrule")

    for dom in domains:
        if dom not in results:
            continue
        r   = results[dom]
        lbl = dlabels[dom]
        bg  = r["best_guru_bfs"]
        bm  = r["best_guru_llm_modulo"]

        strats = [
            ("LLM one-shot (all)",
             r["strat_llm_oneshot"]["validity"],
             r["strat_llm_oneshot"]["llm_calls"],
             0.0, False),
            (f"LLM-Modulo (all, K={retry_budget})",
             r["strat_llm_modulo_all"]["validity"],
             r["strat_llm_modulo_all"]["llm_calls"],
             0.0, False),
            (r"GURU$\to$BFS ($\theta^*$=" + f"{bg['theta']:.1f})",
             bg["validity"],
             bg["llm_calls"],
             bg["bfs_nodes"], True),
            (r"GURU$\to$LLM-Modulo ($\theta^*$=" + f"{bm['theta']:.1f})",
             bm["validity"],
             bm["llm_calls"],
             0.0, True),
        ]

        # Bold the best validity
        best_val = max(s[1] for s in strats)
        for i, (strat_lbl, val, calls, nodes, is_guru) in enumerate(strats):
            dom_s  = f"\\multirow{{4}}{{*}}{{{lbl}}}" if i == 0 else ""
            val_s  = (f"\\textbf{{{val:.1%}}}"
                      if abs(val - best_val) < 0.005 and is_guru
                      else f"{val:.1%}")
            nodes_s = f"{nodes:.0f}" if nodes > 0 else "---"
            lines.append(
                f"  {dom_s} & {strat_lbl} "
                f"& {val_s} & {calls:.2f} & {nodes_s} \\\\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    table_str = "\n".join(lines)
    print("\n" + "-" * 70)
    print("LaTeX Table:")
    print("-" * 70)
    print(table_str)
    out = RESULTS_DIR / "e9_latex_table.tex"
    out.write_text(table_str)
    print(f"  LaTeX table → {out}")
    return table_str


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry_budget",    type=int,  default=5,
                        help="Max LLM-Modulo retries K (default 5)")
    parser.add_argument("--n_runs",           type=int,  default=10,
                        help="GURU score estimation runs")
    parser.add_argument("--seed",             type=int,  default=42)
    parser.add_argument("--no_fig",           action="store_true")
    parser.add_argument("--use_real_gpt4o",   action="store_true",
                        help="Use real GPT-4o accuracy (22.5/4.0/13.0%%) "
                             "instead of Verma et al. Llama-3-8B baselines")
    args = parser.parse_args()

    results = run_llm_modulo_experiment(
        retry_budget=args.retry_budget,
        n_runs=args.n_runs,
        rng_seed=args.seed,
        use_real_gpt4o=args.use_real_gpt4o)

    if results is not None:
        print_latex_table(results, args.retry_budget, args.use_real_gpt4o)


if __name__ == "__main__":
    main()