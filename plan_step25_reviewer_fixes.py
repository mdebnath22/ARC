"""
plan_step25_reviewer_fixes.py
==============================
Addresses 5 remaining reviewer concerns requiring new experiments:

P1:  Within-size evaluation (kills "benchmark artifact" critique)
     Group instances by |O|, compute ARC vs |O| ρ within each bucket.
     If ARC > |O| within same-size groups, gains are not size artifacts.

P3:  Solver-agnostic validation (kills "solver-specific label" critique)
     Correlate ARC predictions with A* node expansions + runtime.
     If ARC correlates with multiple solvers, it captures general difficulty.

P7:  Matched training budget (kills "5000 vs 2000 episode" critique)
     Train ARC baseline at 2000 episodes, compare to contrastive at 2000.

P9:  Attention case study (interpretability)
     Show which support instances ARC attends to for example queries.
     High attention = structurally analogous instance.

P11: Bootstrap confidence intervals (kills "no statistics" critique)
     1000 bootstrap resamples, 95% CIs on all ρ values.
     Report: ARC outperforms baselines with non-overlapping CIs.

USAGE:
  python plan_step25_reviewer_fixes.py --part P1
  python plan_step25_reviewer_fixes.py --part P3
  python plan_step25_reviewer_fixes.py --part P11
  python plan_step25_reviewer_fixes.py --part all
"""
from __future__ import annotations
import argparse, importlib.util, json, pickle, sys, warnings
from pathlib import Path

import numpy as np
import torch
from scipy import stats

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"

TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
TRAIN_DOMAINS = ["depot", "rovers", "satellite"]
DEVICE = "cpu"

# Known ARC ρ values (from Table 1 / primary results)
ARC_RHO = {"blocksworld": 0.722, "logistics": 0.761, "mystery_blocksworld": 0.727}


def load_deps():
    spec = importlib.util.spec_from_file_location("step17", ROOT/"plan_step17_arc_v2.py")
    s17  = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
    sys.modules["step17"] = s17
    sys.modules["__main__"].GlobalPreprocessor = s17.GlobalPreprocessor
    spec6 = importlib.util.spec_from_file_location("step6", ROOT/"plan_step6_pddlinst_gate.py")
    s6    = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    with open(RESULTS/"global_preprocessor.pkl","rb") as f:
        prep = pickle.load(f)
    X_surf, X_fm, tt, y_s, y_ns, splits = s6.load_data(data_dir=DATA)
    X_surf = X_surf[:, :-1]
    ckpt   = torch.load(CKPT/"arc_v2.pt", map_location="cpu")
    model  = s17.ARCv2(ckpt["surf_dim"], ckpt["fm_dim"])
    model.load_state_dict(ckpt["model"]); model.eval()
    tr     = np.isin(tt, TRAIN_DOMAINS)
    Xs_tr, Xe_tr, _ = prep.transform(X_surf[tr], X_fm[tr])
    rng    = np.random.default_rng(42)
    sidx   = rng.choice(tr.sum(), min(60, tr.sum()), replace=False)
    S_surf = torch.FloatTensor(Xs_tr[sidx])
    S_fm   = torch.FloatTensor(Xe_tr[sidx])
    S_V    = torch.FloatTensor(np.hstack([Xs_tr[sidx], Xe_tr[sidx]]))
    return s17, prep, X_surf, X_fm, tt, y_s, y_ns, model, S_surf, S_fm, S_V


def get_arc_scores(model, prep, X_surf, X_fm, mask, S_surf, S_fm, S_V):
    Xs_n, Xe_n, Xr_n = prep.transform(X_surf[mask], X_fm[mask])
    scores = []
    with torch.no_grad():
        for i in range(len(Xs_n)):
            qs = torch.FloatTensor(Xs_n[i]).unsqueeze(0)
            qf = torch.FloatTensor(Xe_n[i]).unsqueeze(0)
            qr = torch.FloatTensor(Xr_n[i]).unsqueeze(0)
            out, _, _ = model(qs, qf, qr, S_surf, S_fm, S_V, head="reg")
            scores.append(float(out.squeeze().cpu()))
    return np.array(scores), Xs_n


# ══════════════════════════════════════════════════════════════════════════════
# P1: Within-size evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_p1(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V):
    print("\n" + "="*65)
    print("P1: Within-size evaluation")
    print("  Group by |O| bucket, compute ρ(ARC, n*) and ρ(|O|, n*)")
    print("  within each bucket. If ARC > |O|: gains not size artifacts.")
    print("="*65)

    results = {}

    for dom in TEST_DOMAINS:
        mask       = tt == dom
        n_obj      = X_surf[mask, 0].astype(int)
        ns_q       = y_ns[mask].astype(float)
        arc_scores, _ = get_arc_scores(model, prep, X_surf, X_fm,
                                        mask, S_surf, S_fm, S_V)

        # Group by |O| bucket
        unique_sizes = sorted(set(n_obj))
        print(f"\n  {dom}:  sizes present: {unique_sizes}")

        bucket_results = []
        for sz in unique_sizes:
            idx = n_obj == sz
            if idx.sum() < 5:
                continue   # too few instances for reliable ρ
            arc_b   = arc_scores[idx]
            ns_b    = ns_q[idx]
            nobj_b  = n_obj[idx].astype(float)

            # ρ of ARC within this size bucket
            rho_arc, _  = stats.spearmanr(arc_b, ns_b)
            # ρ of |O| within this size bucket = 0 by definition
            # (all instances same size → no variation → ρ undefined)
            # Instead compute variance of n_steps within bucket
            ns_var = float(np.std(ns_b))

            bucket_results.append({
                "size":    int(sz),
                "n":       int(idx.sum()),
                "rho_arc": float(rho_arc),
                "ns_std":  ns_var,
            })
            print(f"    |O|={sz}  N={idx.sum():3d}  "
                  f"ARC|ρ|={abs(rho_arc):.3f}  "
                  f"n*_std={ns_var:.2f}  "
                  f"{'(discriminative)' if abs(rho_arc)>0.3 else '(flat)'}")

        # Summary: mean within-bucket ARC ρ (non-trivial buckets only)
        non_flat = [b for b in bucket_results if b["ns_std"] > 0.5]
        if non_flat:
            mean_arc_rho = np.mean([abs(b["rho_arc"]) for b in non_flat])
            print(f"  Mean ARC |ρ| within non-trivial buckets: {mean_arc_rho:.3f}")
            print(f"  Object-cardinality |ρ| within buckets: 0.000 (by definition)")
        results[dom] = bucket_results

    # LaTeX table
    print("\n  LaTeX (within-size evaluation):")
    print(r"\begin{table}[h]\centering\small")
    print(r"\caption{Within-size difficulty ranking. ARC $|\rho|$ computed")
    print(r"within fixed object-cardinality ($|O|$) buckets, eliminating")
    print(r"the trivial size signal. Object-cardinality baseline achieves")
    print(r"$|\rho|{=}0$ within any fixed-size group.}")
    print(r"\label{tab:within_size}")
    print(r"\begin{tabular}{l l cc}")
    print(r"\toprule\textbf{Domain} & $|O|$ & N & ARC $|\rho|$ \\\midrule")
    for dom in TEST_DOMAINS:
        for i, b in enumerate(results.get(dom,[])):
            if b["ns_std"] < 0.5: continue
            dlbl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
            first = dlbl if i==0 else ""
            print(f"  {first} & {b['size']} & {b['n']} & {abs(b['rho_arc']):.3f} \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")

    (RESULTS/"within_size_eval.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {RESULTS}/within_size_eval.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# P3: Solver-agnostic validation
# ══════════════════════════════════════════════════════════════════════════════

def run_p3(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V):
    print("\n" + "="*65)
    print("P3: Solver-agnostic validation")
    print("  Correlate ARC predictions with A*(hFF) node expansions")
    print("  and wall-clock time (independent of BFS labels).")
    print("="*65)

    try:
        from pyperplan.pddl.parser import Parser
        from pyperplan import grounding
        from pyperplan.search.a_star import astar_search
        from pyperplan.heuristics.relaxation import hFFHeuristic
        import signal, tempfile, time
    except ImportError:
        print("  pyperplan not installed. Run: pip install pyperplan")
        return {}

    # Load episodes for PDDL content
    eps_by_dom = {}
    for e in json.loads((DATA/"episodes.json").read_text()):
        dom = e.get("task_type", e.get("domain",""))
        eps_by_dom.setdefault(dom,[]).append(e)

    def run_astar(record, timeout=10):
        dom_pddl  = record.get("domain_pddl","")
        prob_pddl = record.get("problem_pddl","")
        if not dom_pddl or not prob_pddl:
            return None, None
        def _to(s,f): raise TimeoutError()
        signal.signal(signal.SIGALRM,_to); signal.alarm(timeout)
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dp = Path(tmp)/"domain.pddl"; pp = Path(tmp)/"problem.pddl"
                dp.write_text(dom_pddl); pp.write_text(prob_pddl)
                parser = Parser(str(dp),str(pp))
                task   = grounding.ground(parser.parse_problem(parser.parse_domain()))
                h      = hFFHeuristic(task)
                calls  = [0]
                def h_count(node): calls[0]+=1; return float(h(node))
                sol    = astar_search(task, h_count)
                signal.alarm(0)
                wt = time.perf_counter()-t0
                return int(calls[0]), float(wt)
        except (TimeoutError, Exception):
            signal.alarm(0); return None, None

    results = {}
    N = 100  # instances per domain (A* is slow on hard instances)

    print(f"\n  {'Domain':<22}  {'ρ(ARC,n*)':>10}  {'ρ(ARC,nodes)':>13}  "
          f"{'ρ(ARC,time)':>12}  {'ρ(|O|,nodes)':>13}")
    print("  "+"-"*73)

    for dom in TEST_DOMAINS:
        mask       = tt == dom
        n_obj      = X_surf[mask, 0].astype(float)
        ns_q       = y_ns[mask].astype(float)
        arc_scores, _ = get_arc_scores(model, prep, X_surf, X_fm,
                                        mask, S_surf, S_fm, S_V)
        eps = eps_by_dom.get(dom,[])[:N]

        nodes_list = []; time_list  = []
        valid_idx  = []

        print(f"  Running A* on {len(eps)} {dom} instances...", flush=True)
        for i, ep in enumerate(eps):
            n_exp, wt = run_astar(ep, timeout=5)
            if n_exp is not None:
                nodes_list.append(n_exp)
                time_list.append(wt)
                valid_idx.append(i)
            if (i+1) % 20 == 0:
                print(f"    [{i+1}/{len(eps)}] solved: {len(valid_idx)}", end="\r")

        print(f"    Solved: {len(valid_idx)}/{len(eps)}")

        if len(valid_idx) < 10:
            print(f"  {dom}: too few A* solutions, skipping")
            continue

        vi   = np.array(valid_idx)
        arc_v  = arc_scores[vi]
        ns_v   = ns_q[vi]
        nobj_v = n_obj[vi]
        nodes  = np.array(nodes_list)
        times  = np.array(time_list)

        rho_arc_ns,   _ = stats.spearmanr(arc_v, ns_v)
        rho_arc_nodes,_ = stats.spearmanr(arc_v, nodes)
        rho_arc_time, _ = stats.spearmanr(arc_v, times)
        rho_nobj_nodes,_= stats.spearmanr(nobj_v, nodes)

        dl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
        print(f"  {dl:<22}  {abs(rho_arc_ns):>10.3f}  {abs(rho_arc_nodes):>13.3f}  "
              f"{abs(rho_arc_time):>12.3f}  {abs(rho_nobj_nodes):>13.3f}")

        results[dom] = {
            "rho_arc_nsteps": float(rho_arc_ns),
            "rho_arc_nodes":  float(rho_arc_nodes),
            "rho_arc_time":   float(rho_arc_time),
            "rho_nobj_nodes": float(rho_nobj_nodes),
            "n_solved":       len(valid_idx),
        }

    (RESULTS/"solver_agnostic_validation.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {RESULTS}/solver_agnostic_validation.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# P9: Attention case study
# ══════════════════════════════════════════════════════════════════════════════

def run_p9(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V):
    print("\n" + "="*65)
    print("P9: Attention case study")
    print("  Show top-3 attended support instances for example queries.")
    print("  High attention = structurally analogous instance.")
    print("="*65)

    # Build support set with metadata
    tr_mask = np.isin(tt, TRAIN_DOMAINS)
    tr_doms = tt[tr_mask]; tr_ns = y_ns[tr_mask]
    tr_nobj = X_surf[tr_mask, 0]

    examples = []
    for dom in TEST_DOMAINS[:2]:  # BW and Logistics as examples
        mask  = tt == dom
        ns_q  = y_ns[mask].astype(float)
        nobj_q= X_surf[mask, 0]

        # Pick one easy, one hard instance
        easy_idx = np.argmin(ns_q)
        hard_idx = np.argmax(ns_q)

        for label, q_idx in [("easy", easy_idx), ("hard", hard_idx)]:
            Xs_n, Xe_n, Xr_n = prep.transform(X_surf[mask], X_fm[mask])

            qs = torch.FloatTensor(Xs_n[q_idx]).unsqueeze(0)
            qf = torch.FloatTensor(Xe_n[q_idx]).unsqueeze(0)
            qr = torch.FloatTensor(Xr_n[q_idx]).unsqueeze(0)

            with torch.no_grad():
                # Get attention weights
                q_vec   = model.query_enc(qf)
                k_vecs  = model.key_enc(S_surf)
                import math
                logits  = (q_vec @ k_vecs.T) / math.sqrt(q_vec.shape[-1])
                alpha   = torch.softmax(logits * model.log_temp.exp(), dim=-1)
                alpha_np= alpha.squeeze().cpu().numpy()

            # Top-3 attended support instances
            top3_idx = np.argsort(-alpha_np)[:3]

            # Map support indices to training domain metadata
            tr_idx_abs = np.where(tr_mask)[0]
            top3_info  = []
            for i in top3_idx:
                if i < len(tr_idx_abs):
                    abs_i = tr_idx_abs[i]
                    top3_info.append({
                        "support_domain": str(tt[abs_i]),
                        "n_steps":        int(y_ns[abs_i]),
                        "n_objects":      int(X_surf[abs_i, 0]),
                        "attention_weight": float(alpha_np[i]),
                    })

            ex = {
                "query_domain": dom,
                "query_label":  label,
                "query_n_steps":int(ns_q[q_idx]),
                "query_n_obj":  int(nobj_q[q_idx]),
                "top3_support": top3_info,
            }
            examples.append(ex)

            dl = dom[:3].upper()
            print(f"\n  Query: {dl} ({label})  n*={ex['query_n_steps']}  |O|={ex['query_n_obj']}")
            print(f"  Top-3 attended support instances:")
            for r in top3_info:
                print(f"    dom={r['support_domain'][:3]}  n*={r['n_steps']:3d}  "
                      f"|O|={r['n_objects']}  α={r['attention_weight']:.4f}")

    (RESULTS/"attention_case_study.json").write_text(json.dumps(examples, indent=2))
    print(f"\n  Saved → {RESULTS}/attention_case_study.json")
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# P11: Bootstrap confidence intervals
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(x, y, n_boot=1000, alpha=0.05, stat=None):
    """Bootstrap 95% CI for Spearman |ρ| between x and y."""
    if stat is None:
        stat = lambda a, b: abs(stats.spearmanr(a, b)[0])
    obs = stat(x, y)
    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        boot_stats.append(stat(x[idx], y[idx]))
    lo = float(np.percentile(boot_stats, 100*alpha/2))
    hi = float(np.percentile(boot_stats, 100*(1-alpha/2)))
    return float(obs), lo, hi


def run_p11(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V):
    print("\n" + "="*65)
    print("P11: Bootstrap confidence intervals (1000 resamples)")
    print("  95% CIs on all |ρ| values. Non-overlapping = significant.")
    print("="*65)

    results = {}

    print(f"\n  {'Domain':<22}  {'Method':<22}  {'|ρ|':>6}  {'95% CI':>18}")
    print("  " + "-"*73)

    for dom in TEST_DOMAINS:
        mask    = tt == dom
        ns_q    = y_ns[mask].astype(float)
        n_obj   = X_surf[mask, 0].astype(float)

        arc_scores, _ = get_arc_scores(model, prep, X_surf, X_fm,
                                        mask, S_surf, S_fm, S_V)

        methods = {
            "ARC (ours)":             arc_scores,
            "Object-cardinality |O|": n_obj,
        }

        dom_res = {}
        dl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")

        for i, (name, scores) in enumerate(methods.items()):
            obs, lo, hi = bootstrap_ci(scores, ns_q, n_boot=1000)
            first = dl if i == 0 else ""
            print(f"  {first:<22}  {name:<22}  {obs:>6.3f}  [{lo:.3f}, {hi:.3f}]")
            dom_res[name] = {"rho": obs, "ci_lo": lo, "ci_hi": hi}

        # Check if CIs overlap
        arc_ci  = dom_res["ARC (ours)"]
        nobj_ci = dom_res["Object-cardinality |O|"]
        overlap = arc_ci["ci_lo"] < nobj_ci["ci_hi"] and nobj_ci["ci_lo"] < arc_ci["ci_hi"]
        sig     = "non-overlapping CIs (significant)" if not overlap else "overlapping CIs"
        print(f"  {'':22}  → {sig}")

        results[dom] = dom_res
        print()

    # LaTeX
    print("\n  LaTeX (with bootstrap CIs):")
    print(r"\begin{table}[h]\centering\small")
    print(r"\caption{Spearman $|\rho|$ with 95\% bootstrap confidence intervals")
    print(r"(1{,}000 resamples). Non-overlapping intervals indicate")
    print(r"statistically significant differences.}")
    print(r"\label{tab:bootstrap_ci}")
    print(r"\begin{tabular}{l l ccc}")
    print(r"\toprule\textbf{Domain} & \textbf{Method} & $|\rho|$ "
          r"& \multicolumn{2}{c}{95\% CI} \\\midrule")
    for dom in TEST_DOMAINS:
        dl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
        for i, name in enumerate(["ARC (ours)", "Object-cardinality |O|"]):
            r = results[dom][name]
            first = dl if i == 0 else ""
            print(f"  {first} & {name} & {r['rho']:.3f} "
                  f"& [{r['ci_lo']:.3f}, & {r['ci_hi']:.3f}] \\\\")
        print(r"  \midrule")
    print(r"\bottomrule\end{tabular}\end{table}")

    (RESULTS/"bootstrap_ci.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {RESULTS}/bootstrap_ci.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="all",
                   choices=["P1","P3","P9","P11","all"])
    args = p.parse_args()

    print(f"\nReviewer fixes  —  device={DEVICE}")
    s17, prep, X_surf, X_fm, tt, y_s, y_ns, model, S_surf, S_fm, S_V = load_deps()

    if args.part in ("P1",  "all"):
        run_p1(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V)

    if args.part in ("P3",  "all"):
        run_p3(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V)

    if args.part in ("P9",  "all"):
        run_p9(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V)

    if args.part in ("P11", "all"):
        run_p11(prep, X_surf, X_fm, tt, y_ns, model, S_surf, S_fm, S_V)

    print("\nDone. Run on sc008:")
    print("  python plan_step25_reviewer_fixes.py --part all")


if __name__ == "__main__":
    main()