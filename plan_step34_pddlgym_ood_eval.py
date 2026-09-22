"""
plan_step34_pddlgym_ood_eval.py
================================
Evaluate ARC zero-shot on all compatible PDDLGym non-IPC domains.
Compares:
  1. ARC      — zero-shot cross-domain meta-learner
  2. |O|      — object-cardinality baseline
  3. ERM      — non-episodic MLP trained on source domains (identical features)

This directly replicates the main paper evaluation protocol on unseen domains.

USAGE:
  python plan_step34_pddlgym_ood_eval.py
"""

from __future__ import annotations
import json, re, signal, sys, tempfile, importlib.util, warnings
from pathlib import Path

import numpy as np
import torch, torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).resolve().parent
PDDL  = Path("/scratch/mroycho1/pddlgym/pddlgym/pddl")
RES   = ROOT / "results_planning"; RES.mkdir(exist_ok=True)
CKPT  = ROOT / "checkpoints_planning"
TRAIN = ["depot", "rovers", "satellite"]

# Domains to skip: IPC-origin, too few problems, or pyperplan-incompatible
SKIP = {
    # too few problems
    "casino","meetpass","slidetile","toomanyblocks",
    "dynamic_action_space_same_obj","quantifiedblocks3",
    "navigation1","navigation2","navigation3","navigation4","navigation5",
    "navigation6","navigation7","navigation8","navigation9","navigation10",
    # pyperplan-incompatible
    "snake",
    # empty dirs
    "searchandrescue","manyblockssmallpilesnoclear",
    "manyblockssmallpilesnoclearhand","manyblockssmallpilesnohand",
}


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (same 29-dim xs used in training)
# ══════════════════════════════════════════════════════════════════════════════

def extract_xs(dom_txt, prob_txt):
    obj_sec = re.search(r':objects([^)]*)\)', prob_txt, re.DOTALL | re.I)
    n_obj = len(re.findall(r'\b\w[\w-]*\b', obj_sec.group(1))) if obj_sec else 1
    n_obj = max(n_obj, 1)
    init_txt = prob_txt.split(':goal')[0].split(':init')[-1] if ':init' in prob_txt else ''
    goal_txt = prob_txt.split(':goal')[-1][:500] if ':goal' in prob_txt else ''
    init_f = re.findall(r'\(\w[\w-]*[^)]*\)', init_txt)
    goal_f = re.findall(r'\(\w[\w-]*[^)]*\)', goal_txt)
    ops = re.findall(r':action\s+\S+', dom_txt)
    n_init = len(init_f); n_goal = len(goal_f); n_ops = len(ops)
    pred_c = {}
    for p in init_f:
        name = p.strip('()').split()[0] if p.strip('()').split() else 'x'
        pred_c[name] = pred_c.get(name, 0) + 1
    pv = list(pred_c.values()) or [0]
    feats = np.array([
        float(n_obj), float(n_goal), float(n_init), float(n_ops), float(n_goal),
        float(np.mean(pv)), float(np.std(pv) if len(pv) > 1 else 0), float(np.max(pv)),
        float(len(pred_c)), float(n_ops % 2), float(n_goal / n_obj),
        float(n_init / max(n_goal, 1)), float(n_obj / max(len(pred_c), 1)),
        float(n_goal * np.mean(pv)), float(len(pred_c) / max(n_ops, 1)),
        float(len(pred_c) / n_obj), float(n_init / n_obj**2),
        float(n_goal / n_obj**2), float(len(pred_c) / max(n_ops, 1)),
        float(np.mean(pv) * n_goal), float(n_obj * np.mean(pv)),
        float(n_goal / max(n_init, 1)), float(n_init / max(n_ops, 1)),
        float(n_goal / max(n_init, 1)), float(n_obj / max(n_goal, 1)),
        float(len(pred_c) / n_obj), float(n_init / n_obj), float(n_ops / n_obj),
        float(n_obj * n_goal / max(n_init, 1)), float(0.0),
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0, posinf=0, neginf=0), n_obj


# ══════════════════════════════════════════════════════════════════════════════
# BFS solver
# ══════════════════════════════════════════════════════════════════════════════

def run_bfs(dom_txt, prob_txt, timeout=8):
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding
    from pyperplan.search.breadth_first_search import breadth_first_search
    from pyperplan.search.enforced_hillclimbing_search import enforced_hillclimbing_search
    from pyperplan.heuristics.relaxation import hFFHeuristic

    def _to(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _to); signal.alarm(timeout)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dp = Path(tmp) / "domain.pddl"; pp = Path(tmp) / "problem.pddl"
            dp.write_text(dom_txt); pp.write_text(prob_txt)
            parser = Parser(str(dp), str(pp))
            task = grounding.ground(parser.parse_problem(parser.parse_domain()))
            sol = breadth_first_search(task)
            if sol is None:
                try: sol = enforced_hillclimbing_search(task, hFFHeuristic(task))
                except: pass
            signal.alarm(0)
            return len(sol) if sol else -1
    except (TimeoutError, Exception):
        signal.alarm(0); return -1


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data ──────────────────────────────────────────────────────────
    spec6 = importlib.util.spec_from_file_location("s6", ROOT/"plan_step6_pddlinst_gate.py")
    s6    = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    spec3 = importlib.util.spec_from_file_location("s3", ROOT/"plan_step3_guru.py")
    s3    = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(s3)

    X_surf_all, X_fm_all, tt_all, _, y_ns_all, _ = s6.load_data(data_dir=ROOT/"data"/"planning")
    tr = np.isin(tt_all, TRAIN)

    sc_s = StandardScaler().fit(X_surf_all[tr])
    sc_f = StandardScaler().fit(X_fm_all[tr])
    pp   = Pipeline([("pca", PCA(20)), ("r", Ridge(1.0))])
    pp.fit(sc_s.transform(X_surf_all[tr]), sc_f.transform(X_fm_all[tr]))

    def tfm(Xs, Xe):
        Xs_n = sc_s.transform(Xs); Xe_n = sc_f.transform(Xe)
        return Xs_n, Xe_n, Xe_n - pp.predict(Xs_n)

    rng  = np.random.default_rng(42)
    sidx = rng.choice(tr.sum(), min(60, tr.sum()), replace=False)
    Xs_tr, Xe_tr, _ = tfm(X_surf_all[tr], X_fm_all[tr])
    S_s = torch.FloatTensor(Xs_tr[sidx])
    S_f = torch.FloatTensor(Xe_tr[sidx])
    S_V = torch.FloatTensor(np.hstack([Xs_tr[sidx], Xe_tr[sidx]]))

    # ── Load ARC ───────────────────────────────────────────────────────────
    ck    = torch.load(CKPT/"guru_baseline_5000ep.pt", map_location="cpu")
    model = s3.PlanningGURU(30, X_fm_all.shape[1])
    model.load_state_dict(ck["model"]); model.eval()
    print("ARC model loaded: guru_baseline_5000ep.pt")

    # ── Train ERM on source domains (identical features, no episodic structure)
    # ERM replicates the non-episodic MLP baseline from the paper (Table 2)
    Xs_erm = np.hstack([sc_s.transform(X_surf_all[tr]),
                        sc_f.transform(X_fm_all[tr])])
    y_erm  = y_ns_all[tr].astype(float)
    valid  = y_erm > 0
    erm    = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                          random_state=42, early_stopping=True, validation_fraction=0.1)
    erm.fit(Xs_erm[valid], y_erm[valid])
    print(f"ERM trained: {valid.sum()} source instances, "
          f"features={Xs_erm.shape[1]}")

    # ── FM embeddings ──────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-mpnet-base-v2")

    # ── Find candidate domains ─────────────────────────────────────────────
    candidates = []
    for f in sorted(PDDL.glob("*.pddl")):
        stem = f.stem
        if stem in SKIP: continue
        prob_dir = PDDL / stem
        if not prob_dir.exists(): continue
        probs = sorted(prob_dir.glob("*.pddl"))
        if len(probs) < 8: continue
        txt = f.read_text()
        if any(x in txt for x in [":functions", ":negative-preconditions",
                                   ":conditional-effects"]):
            continue
        candidates.append((stem, f, probs))

    print(f"\nEvaluating {len(candidates)} PDDLGym non-IPC domains")
    print(f"\n{'Domain':<30} {'N':>4} {'BFS%':>6} {'ARC':>8} {'|O|':>8} {'ERM':>8} "
          f"{'ΔARC-O':>8} {'ΔARC-E':>8}")
    print("-" * 82)

    results = {}

    for dom_name, dom_path, prob_files in candidates:
        dom_txt = dom_path.read_text()
        ns_list=[]; no_list=[]; xs_list=[]; descs=[]; n_tried=0

        for pf in prob_files[:200]:
            if pf.stat().st_size < 30: continue
            n_tried += 1
            try:
                pt = pf.read_text()
                ns = run_bfs(dom_txt, pt, timeout=8)
                if ns > 0:
                    xs, n_obj = extract_xs(dom_txt, pt)
                    ns_list.append(ns); no_list.append(float(n_obj))
                    xs_list.append(xs)
                    descs.append(f"PDDLGym {dom_name} planning with {n_obj} objects.")
            except Exception:
                pass

        if len(ns_list) < 8:
            print(f"  {dom_name:<30}: {len(ns_list)}/{n_tried} — SKIP")
            continue

        ns_arr = np.array(ns_list); no_arr = np.array(no_list)
        xs_mat = np.array(xs_list)

        # ARC scores
        Xe_ood = sbert.encode(descs, show_progress_bar=False)
        Xs_n, Xe_n, Xr_n = tfm(xs_mat, Xe_ood)
        arc_sc = []
        with torch.no_grad():
            for i in range(len(Xs_n)):
                qs = torch.FloatTensor(Xs_n[i]).unsqueeze(0)
                qf = torch.FloatTensor(Xe_n[i]).unsqueeze(0)
                qr = torch.FloatTensor(Xr_n[i]).unsqueeze(0)
                try:
                    o, _, _ = model(qs, qf, qr, S_s, S_f, S_V, head="cls")
                    arc_sc.append(float(F.softmax(o.squeeze(0), -1)[1].cpu()))
                except Exception:
                    arc_sc.append(0.5)
        arc_arr = np.array(arc_sc)

        # ERM scores — same xs + fm features, concatenated
        Xs_erm_ood = np.hstack([Xs_n, Xe_n])  # same feature space as training
        try:
            erm_sc = erm.predict(Xs_erm_ood)
        except Exception:
            erm_sc = np.zeros(len(ns_list))

        # Correlations
        rho_arc, _ = stats.spearmanr(arc_arr, ns_arr)
        rho_obj, _ = stats.spearmanr(no_arr,   ns_arr)
        rho_erm, _ = stats.spearmanr(erm_sc,   ns_arr)

        arc_wins_obj = bool(abs(rho_arc) > abs(rho_obj))
        arc_wins_erm = bool(abs(rho_arc) > abs(rho_erm))

        results[dom_name] = {
            "n":         len(ns_list),
            "bfs_rate":  len(ns_list) / n_tried,
            "rho_arc":   float(rho_arc),
            "rho_obj":   float(rho_obj),
            "rho_erm":   float(rho_erm),
            "arc_beats_obj": arc_wins_obj,
            "arc_beats_erm": arc_wins_erm,
        }

        markers = []
        if arc_wins_obj: markers.append(">|O|")
        if arc_wins_erm: markers.append(">ERM")
        marker = " ←" + "+".join(markers) if markers else ""

        print(f"  {dom_name:<30} {len(ns_list):>4} {len(ns_list)/n_tried:>6.1%} "
              f"{abs(rho_arc):>8.3f} {abs(rho_obj):>8.3f} {abs(rho_erm):>8.3f} "
              f"{abs(rho_arc)-abs(rho_obj):>+8.3f} {abs(rho_arc)-abs(rho_erm):>+8.3f}"
              f"{marker}")

    # ── Summary ────────────────────────────────────────────────────────────
    (RES / "pddlgym_ood_evaluation.json").write_text(json.dumps(results, indent=2))

    beats_obj = sum(1 for r in results.values() if r["arc_beats_obj"])
    beats_erm = sum(1 for r in results.values() if r["arc_beats_erm"])
    beats_both= sum(1 for r in results.values() if r["arc_beats_obj"] and r["arc_beats_erm"])
    n = len(results)

    print(f"\n{'='*82}")
    print(f"ARC beats |O|:   {beats_obj}/{n}")
    print(f"ARC beats ERM:   {beats_erm}/{n}")
    print(f"ARC beats BOTH:  {beats_both}/{n}")

    print(f"\nWINNING DOMAINS (ARC beats both baselines):")
    print(f"{'Domain':<30} {'N':>4} {'ARC':>8} {'|O|':>8} {'ERM':>8} {'Δ|O|':>8} {'ΔERM':>8}")
    print("-"*76)
    for d, r in sorted(results.items(), key=lambda x: -abs(x[1]['rho_arc'])):
        if r['arc_beats_obj'] and r['arc_beats_erm']:
            print(f"  {d:<30} {r['n']:>4} {abs(r['rho_arc']):>8.3f} "
                  f"{abs(r['rho_obj']):>8.3f} {abs(r['rho_erm']):>8.3f} "
                  f"{abs(r['rho_arc'])-abs(r['rho_obj']):>+8.3f} "
                  f"{abs(r['rho_arc'])-abs(r['rho_erm']):>+8.3f}")

    print(f"\nSaved → {RES}/pddlgym_ood_evaluation.json")


if __name__ == "__main__":
    main()
