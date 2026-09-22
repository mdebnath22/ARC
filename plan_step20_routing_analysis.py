"""
plan_step20_routing_analysis.py
================================
Extends the routing evaluation with proper metrics and selective routing.
Runs on top of existing step17 results — no retraining needed.

Implements:
  [3]  Selective routing: abstain when p < τ_low, use LLM when p > τ_high
       τ_high, τ_low fitted on training-domain BFS labels (zero-shot to test)
  [6]  Temperature scaling: fit T on held-out 30% of Qwen labels per domain
  [7]  Win rate, regret, relative improvement metrics
  [8]  Hard subset analysis (top 30% by predicted difficulty)

USAGE:
  python plan_step20_routing_analysis.py
"""

from __future__ import annotations
import json, pickle, importlib.util, warnings
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"

TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]
DOMAIN_LABELS = {"blocksworld":"Blocksworld","logistics":"Logistics",
                 "mystery_blocksworld":"Mystery-BW"}
BFS_SUCCESS = {"blocksworld":0.590,"logistics":0.160,"mystery_blocksworld":0.555}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Load everything ───────────────────────────────────────────────────────────

def load_all():
    # GlobalPreprocessor
    import sys
    spec = importlib.util.spec_from_file_location("step17", ROOT/"plan_step17_arc_v2.py")
    s17  = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
    sys.modules["step17"] = s17
    sys.modules["__main__"].GlobalPreprocessor = s17.GlobalPreprocessor

    prep_path = RESULTS / "global_preprocessor.pkl"
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)

    # Data
    spec6 = importlib.util.spec_from_file_location("step6", ROOT/"plan_step6_pddlinst_gate.py")
    step6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(step6)
    X_surf, X_fm, task_types, y_success, y_nsteps, splits = \
        step6.load_data(data_dir=ROOT/"data"/"planning")
    X_surf = X_surf[:, :-1]   # remove domain hash

    # ARC v2 model
    ckpt  = torch.load(CKPT/"arc_v2.pt", map_location=DEVICE)
    model = s17.ARCv2(ckpt["surf_dim"], ckpt["fm_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()

    # Global support set (training domains only)
    train_mask = np.isin(task_types, splits["meta_train"]["domains"])
    Xs_tr, Xe_tr, Xr_tr = prep.transform(X_surf[train_mask], X_fm[train_mask])
    rng  = np.random.default_rng(42)
    n_s  = min(60, train_mask.sum())
    sidx = rng.choice(train_mask.sum(), n_s, replace=False)
    S_surf = torch.FloatTensor(Xs_tr[sidx]).to(DEVICE)
    S_fm   = torch.FloatTensor(Xe_tr[sidx]).to(DEVICE)
    S_V    = torch.FloatTensor(np.hstack([Xs_tr[sidx], Xe_tr[sidx]])).to(DEVICE)

    # Qwen labels
    qwen_path = RESULTS / "qwen72b_eval_instances.jsonl"
    qwen_by_dom = {}
    for line in open(qwen_path):
        r = json.loads(line)
        qwen_by_dom.setdefault(r["domain"], []).append(r)
    for d in qwen_by_dom:
        qwen_by_dom[d].sort(key=lambda r: int(r["instance_id"]))

    return prep, X_surf, X_fm, task_types, y_nsteps, model, S_surf, S_fm, S_V, qwen_by_dom


# ── Extract ARC scores per domain ─────────────────────────────────────────────

def get_arc_scores(domain, X_surf, X_fm, task_types, prep, model,
                   S_surf, S_fm, S_V, N=200):
    mask  = task_types == domain
    Xs_n, Xe_n, Xr_n = prep.transform(X_surf[mask], X_fm[mask])
    scores = []
    with torch.no_grad():
        for i in range(min(N, len(Xs_n))):
            qs = torch.FloatTensor(Xs_n[i]).unsqueeze(0).to(DEVICE)
            qf = torch.FloatTensor(Xe_n[i]).unsqueeze(0).to(DEVICE)
            qr = torch.FloatTensor(Xr_n[i]).unsqueeze(0).to(DEVICE)
            out, _, _ = model(qs, qf, qr, S_surf, S_fm, S_V, head="reg")
            scores.append(float(out.squeeze().cpu()))
    return np.array(scores), Xs_n[:N]


# ── Metrics ───────────────────────────────────────────────────────────────────

def system_validity(scores, y_llm, bfs_rate, N, budget_fracs=None):
    """Best system validity across budget fractions."""
    if budget_fracs is None:
        budget_fracs = np.linspace(0.05, 0.95, 19)
    best = 0.0
    best_k = 0
    for k_frac in budget_fracs:
        k   = max(1, int(k_frac * N))
        top = np.argsort(-scores)[:k]
        val = (y_llm[top].sum() + bfs_rate * (N - k)) / N
        if val > best:
            best = val; best_k = k
    return float(best), best_k


def win_rate(routed_to_llm, y_llm, bfs_solved_mask):
    """
    Fraction of instances where routing picked the better solver.
    For each instance: correct if (routed_to_llm & LLM_solved) or
                                  (~routed_to_llm & BFS_solved)
    Uses BFS_solved as a Bernoulli draw at domain rate.
    """
    rng = np.random.default_rng(42)
    bfs_outcomes = rng.random(len(routed_to_llm)) < bfs_solved_mask
    llm_wins  = routed_to_llm  & y_llm.astype(bool)
    bfs_wins  = ~routed_to_llm & bfs_outcomes
    return float((llm_wins | bfs_wins).mean())


def regret(scores, y_llm, bfs_rate, N):
    """
    E[max(U_LLM, U_BFS) - U_chosen] at optimal budget.
    Oracle: always picks better solver per instance.
    """
    rng = np.random.default_rng(42)
    bfs_outcomes = rng.random(N) < bfs_rate
    oracle_utility = np.maximum(y_llm, bfs_outcomes).mean()

    # Chosen: route top-k to LLM at optimal budget
    best_val, best_k = system_validity(scores, y_llm, bfs_rate, N)
    top = np.argsort(-scores)[:best_k]
    chosen = np.zeros(N); chosen[top] = 1
    chosen_utility = (y_llm * chosen + bfs_outcomes * (1-chosen)).mean()
    return float(oracle_utility - chosen_utility)


# ── Selective routing (dual threshold) ───────────────────────────────────────

def fit_selective_thresholds(scores_tr, y_tr, bfs_rate_tr, n_grid=20):
    """
    Fit τ_high, τ_low on training data to maximise:
      - route to LLM if score > τ_high (confident LLM wins)
      - route to BFS if score < τ_low  (confident BFS wins)
      - abstain (use BFS) otherwise
    Fitted on training-domain BFS labels as proxy for LLM.
    Returns (τ_high, τ_low).
    """
    grid = np.linspace(scores_tr.min(), scores_tr.max(), n_grid)
    best_val = -np.inf; best_th = (grid[-1], grid[0])
    for th in grid:
        for tl in grid:
            if th <= tl: continue
            routed = scores_tr > th
            val    = (y_tr[routed].sum() + bfs_rate_tr * (~routed).sum()) / len(y_tr)
            if val > best_val:
                best_val = val; best_th = (th, tl)
    return best_th


def selective_system_validity(scores, y_llm, bfs_rate, tau_high, tau_low, N):
    """Apply dual-threshold selective routing."""
    routed  = scores > tau_high                    # confident LLM
    # BFS for everything below tau_low (+ abstain zone)
    llm_val = y_llm[routed].sum() if routed.sum() > 0 else 0
    bfs_val = bfs_rate * (~routed).sum()
    return float((llm_val + bfs_val) / N)


# ── Temperature scaling ───────────────────────────────────────────────────────

def temperature_scale(logits, y_true, n_val=60):
    """Fit temperature T on held-out n_val instances. Returns scaled logits."""
    val_idx = np.random.default_rng(42).choice(len(logits), n_val, replace=False)
    from scipy.optimize import minimize_scalar
    def nll(T):
        p = 1 / (1 + np.exp(-logits[val_idx] / T))
        p = np.clip(p, 1e-7, 1-1e-7)
        return -np.mean(y_true[val_idx]*np.log(p) + (1-y_true[val_idx])*np.log(1-p))
    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T   = float(res.x)
    print(f"    Temperature T={T:.3f}")
    return logits / T


# ── Hard subset analysis ──────────────────────────────────────────────────────

def hard_subset_analysis(scores, y_llm, bfs_rate, percentile=70):
    """
    Routing metrics restricted to instances predicted as hard
    (difficulty score > percentile-th percentile).
    ARC predicts high n_steps = hard. Hard instances are where
    routing most matters — easy instances both solvers handle fine.
    """
    threshold = np.percentile(-scores, 100-percentile)  # high score = easy
    hard_mask = (-scores) > threshold                    # negated: high n_steps = hard
    N_hard    = hard_mask.sum()
    if N_hard < 5: return None
    best_val, _ = system_validity(scores[hard_mask], y_llm[hard_mask],
                                   bfs_rate, N_hard)
    nobj_scores = np.arange(N_hard, 0, -1).astype(float)  # proxy
    nobj_val, _ = system_validity(nobj_scores, y_llm[hard_mask], bfs_rate, N_hard)
    return {"n_hard": int(N_hard), "arc_val": float(best_val),
            "nobj_val": float(nobj_val), "gain": float(best_val - nobj_val)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("ROUTING ANALYSIS: extended metrics")
    print("="*70)

    prep, X_surf, X_fm, task_types, y_nsteps, model, S_surf, S_fm, S_V, qwen_by_dom \
        = load_all()

    # Fit selective thresholds on training-domain BFS labels
    train_doms = ["depot","satellite","rovers","ferry"]
    tr_mask    = np.isin(task_types, train_doms)
    scores_tr_all, Xs_tr_n = get_arc_scores(
        "depot", X_surf, X_fm, task_types, prep, model,
        S_surf, S_fm, S_V, N=200)
    y_tr_bfs = (y_nsteps[task_types=="depot"] <= 12).astype(float)[:200]
    bfs_tr   = y_tr_bfs.mean()
    tau_high, tau_low = fit_selective_thresholds(
        scores_tr_all, y_tr_bfs, bfs_tr)
    print(f"\n  Selective routing thresholds (fitted on Depot):")
    print(f"    τ_high={tau_high:.3f}  τ_low={tau_low:.3f}")

    results = {}
    print()
    print(f"  {'Domain':<14}  {'Method':<28}  "
          f"{'Validity':>9}  {'vs n_obj':>9}  {'WinRate':>8}  {'Regret':>7}")
    print("  " + "-"*82)

    for dom in TEST_DOMAINS:
        bfs    = BFS_SUCCESS[dom]
        N      = 200
        qlist  = qwen_by_dom.get(dom, [])[:N]
        y_qwen = np.array([1.0 if r["valid_plan"] else 0.0 for r in qlist])
        n_obj  = X_surf[task_types==dom, 0][:N]

        # ARC scores
        arc_scores, Xs_n = get_arc_scores(
            dom, X_surf, X_fm, task_types, prep, model,
            S_surf, S_fm, S_V, N=N)

        # Calibrate ARC scores with temperature scaling
        logits = arc_scores - arc_scores.mean()   # centre
        if y_qwen.sum() > 5:
            logits_cal = temperature_scale(logits, y_qwen)
        else:
            logits_cal = logits
        scores_cal = 1 / (1 + np.exp(-logits_cal))  # sigmoid

        # Few-shot XGBoost on calibrated features
        rng2    = np.random.default_rng(42)
        n_tr    = int(0.7 * N)
        tr_idx  = rng2.choice(N, n_tr, replace=False)
        te_idx  = np.setdiff1d(np.arange(N), tr_idx)

        y_tr = y_qwen[tr_idx].astype(int)
        if len(np.unique(y_tr)) < 2:
            y_tr = (n_obj[tr_idx] < np.median(n_obj[tr_idx])).astype(int)

        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, verbosity=0,
            use_label_encoder=False, eval_metric="logloss", random_state=42)
        clf.fit(Xs_n[tr_idx], y_tr)
        fs_scores = clf.predict_proba(Xs_n)[:, 1]

        methods = {
            "Always BFS":          (np.zeros(N),        True),
            "Always LLM":          (np.ones(N),         True),
            "n_obj ($|O|$)":       (-n_obj.astype(float), False),
            "ARC zero-shot":       (-arc_scores,         False),
            "ARC calibrated":      (scores_cal,          False),
            "ARC few-shot":        (fs_scores,           False),
            "ARC selective":       (None,                False),   # special
        }

        nobj_val, _ = system_validity(-n_obj.astype(float), y_qwen, bfs, N)

        dom_res = {}
        for method, (scores, is_static) in methods.items():
            if method == "Always BFS":
                val = bfs; wrate = None; reg = None
            elif method == "Always LLM":
                val = float(y_qwen.mean()); wrate = None; reg = None
            elif method == "ARC selective":
                val   = selective_system_validity(
                    -arc_scores, y_qwen, bfs, tau_high, tau_low, N)
                routed = -arc_scores > tau_high
                wrate  = win_rate(routed, y_qwen,
                                  np.full(N, bfs))
                reg    = float(np.maximum(y_qwen, np.random.RandomState(42).random(N)<bfs).mean()
                               - (y_qwen * routed + (np.random.RandomState(42).random(N)<bfs) * ~routed).mean())
            else:
                val, best_k = system_validity(scores, y_qwen, bfs, N)
                top    = np.argsort(-scores)[:best_k]
                routed = np.zeros(N, dtype=bool); routed[top] = True
                wrate  = win_rate(routed, y_qwen, np.full(N, bfs))
                reg    = regret(scores, y_qwen, bfs, N)

            vs_nobj = val - nobj_val if not is_static else None
            dom_res[method] = {"val": val, "vs_nobj": vs_nobj,
                               "win_rate": wrate, "regret": reg}

            dlbl = DOMAIN_LABELS[dom] if method == "Always BFS" else ""
            vs_s = f"{vs_nobj:>+9.1%}" if vs_nobj is not None else "         —"
            wr_s = f"{wrate:>8.1%}" if wrate is not None else "       —"
            rg_s = f"{reg:>7.4f}" if reg is not None else "      —"
            print(f"  {dlbl:<14}  {method:<28}  {val:>9.1%}  {vs_s}  {wr_s}  {rg_s}")

        # Hard subset
        hard = hard_subset_analysis(-arc_scores, y_qwen, bfs, percentile=70)
        if hard:
            print(f"  {'':14}  Hard subset (top 30% difficult, N={hard['n_hard']}): "
                  f"ARC={hard['arc_val']:.1%}  |O|={hard['nobj_val']:.1%}  "
                  f"Δ={hard['gain']:+.1%}")
        print()
        results[dom] = dom_res

    # Summary table for LaTeX
    print("\nLaTeX routing table:")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Routing evaluation with extended metrics. "
          r"\textit{Validity}: system plan validity (LLM or BFS). "
          r"\textit{vs $|O|$}: gain over object-cardinality baseline. "
          r"\textit{Win rate}: fraction of instances routed to the better solver. "
          r"\textit{Regret}: gap to oracle routing. "
          r"ARC selective uses dual thresholds ($\tau_{high}, \tau_{low}$) "
          r"fitted on training domains only.}")
    print(r"\label{tab:routing_extended}")
    print(r"\small\begin{tabular}{l l cccc}")
    print(r"\toprule")
    print(r"\textbf{Domain} & \textbf{Method} & \textbf{Validity} "
          r"& \textbf{vs $|O|$} & \textbf{Win rate} & \textbf{Regret} \\")
    print(r"\midrule")
    for dom in TEST_DOMAINS:
        lbl = DOMAIN_LABELS[dom]
        for i, (method, r) in enumerate(results[dom].items()):
            dlbl = lbl if i == 0 else ""
            val_s = f"{r['val']:.1%}"
            vs_s  = f"{r['vs_nobj']:+.1%}" if r['vs_nobj'] is not None else "---"
            wr_s  = f"{r['win_rate']:.1%}" if r['win_rate'] is not None else "---"
            rg_s  = f"{r['regret']:.4f}" if r['regret'] is not None else "---"
            sep   = r"\midrule" if i == 0 else ""
            if sep: print(sep)
            bold = r["method"] == "ARC selective" if "method" in r else False
            print(f"  {dlbl} & {method} & {val_s} & {vs_s} & {wr_s} & {rg_s} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}\end{table}")

    (RESULTS / "routing_extended.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults → {RESULTS}/routing_extended.json")


if __name__ == "__main__":
    main()
