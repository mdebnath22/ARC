"""
plan_step9_empirical_hybrid_real_llm.py
=======================================
Build E1b empirical hybrid-routing metrics using real per-instance LLM outcomes
from step8 JSONL and matched LLM budgets from E1 oracle outputs.

Outputs (default):
  - results_planning/e1b_empirical_gpt4o.json
  - results_planning/e1b_empirical_gpt4o.tex

Usage:
  python plan_step9_empirical_hybrid_real_llm.py
  python plan_step9_empirical_hybrid_real_llm.py --self-test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "planning"
RESULTS_DIR = ROOT_DIR / "results_planning"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_step6_module():
    spec = importlib.util.spec_from_file_location(
        "step6_mod", ROOT_DIR / "plan_step6_pddlinst_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def topk_hybrid_metrics(scores: np.ndarray, y_llm: np.ndarray, k: int) -> Tuple[float, float]:
    """
    Route top-k (highest scores) to LLM, the rest to BFS (assumed perfect).
    Returns:
      (plan_validity, llm_precision)
    """
    n = len(y_llm)
    idx = np.argsort(scores)[::-1][:k]
    llm_success = float(y_llm[idx].sum())
    plan_validity = (llm_success + (n - k)) / n
    llm_precision = llm_success / k if k > 0 else float("nan")
    return float(plan_validity), float(llm_precision)


def build_latex_table(domain_results: Dict[str, dict]) -> str:
    dom_order = ["blocksworld", "logistics", "mystery_blocksworld"]
    labels = {
        "blocksworld": "Blocksworld",
        "logistics": "Logistics",
        "mystery_blocksworld": "Mystery-BW",
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{E1b: Empirical hybrid routing with real GPT-4o outcomes. "
        r"LLM outcomes are per-instance valid-plan labels from step8; BFS fallback is "
        r"assumed perfect. GURU and surface-only route top-$k$ instances at matched "
        r"LLM budgets from E1 oracle (52\%, 56\%, 54.7\%).}"
    )
    lines.append(r"\label{tab:e1b_empirical_gpt4o}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Method} & \textbf{Plan validity} & "
        r"\textbf{LLM usage} & \textbf{Precision} \\"
    )
    lines.append(r"\midrule")

    for i, dom in enumerate(dom_order):
        r = domain_results[dom]
        m = r["methods"]
        if i > 0:
            lines.append(r"\midrule")

        lines.append(
            f"\\multirow{{5}}{{*}}{{{labels[dom]}}}"
            f" & LLM only & {m['llm_only']['plan_validity']:.1%} & 100\\% & {m['llm_only']['precision']:.1%} \\\\"
        )
        lines.append(
            f" & Blind (matched budget) & {m['blind_matched']['plan_validity']:.1%} "
            f"& {m['blind_matched']['llm_usage']:.1%} & --- \\\\"
        )
        lines.append(
            f" & Surface-only routing & {m['surface_routing']['plan_validity']:.1%} "
            f"& {m['surface_routing']['llm_usage']:.1%} & {m['surface_routing']['precision']:.1%} \\\\"
        )
        lines.append(
            f" & \\textbf{{GURU routing}} & \\textbf{{{m['guru_routing']['plan_validity']:.1%}}} "
            f"& \\textbf{{{m['guru_routing']['llm_usage']:.1%}}} & \\textbf{{{m['guru_routing']['precision']:.1%}}} \\\\"
        )
        lines.append(
            f" & BFS only & {m['bfs_only']['plan_validity']:.1%} & 0\\% & --- \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def run_self_test() -> None:
    # E1b matched-budget formula test
    y = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    v, p = topk_hybrid_metrics(scores, y, k=2)
    assert abs(v - 0.75) < 1e-9
    assert abs(p - 0.50) < 1e-9

    # E8 gain/capture formula sanity check
    y_llm = np.array([1.0, 1.0, 0.0, 0.0])
    acc_llm = float(y_llm.mean())  # 0.5
    acc_pddl = 0.8
    b = 0.5
    # Route the two successes to LLM, two failures to PDDL
    v_guru = (2.0 + 0.8 * 2.0) / 4.0  # 0.9
    v_blind = (1.0 - b) * acc_llm + b * acc_pddl  # 0.65
    gain = v_guru - v_blind  # 0.25
    v_oracle = v_guru
    oracle_gain = v_oracle - v_blind  # 0.25
    capture = gain / oracle_gain
    assert abs(gain - 0.25) < 1e-9
    assert abs(capture - 1.0) < 1e-9

    print("Self-test passed: E1b matched-budget math + E8 gain/capture math")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-outcomes-jsonl",
        default=str(RESULTS_DIR / "gpt4o_eval_instances.jsonl"),
        help="Step8 JSONL path with per-instance valid_plan outcomes.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Dataset directory containing arrays/registry/episodes.",
    )
    parser.add_argument(
        "--e1-json",
        default=str(RESULTS_DIR / "e1_hybrid_planner.json"),
        help="E1 oracle JSON containing matched llm_usage targets per domain.",
    )
    parser.add_argument("--output_tag", default="gpt4o")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    step6 = _load_step6_module()
    step6.assert_success_labels_supported(data_dir=args.data_dir)

    e1_path = Path(args.e1_json)
    if not e1_path.exists():
        raise FileNotFoundError(f"Missing E1 oracle file: {e1_path}")
    e1 = json.loads(e1_path.read_text())

    X_surf, X_fm, task_types, y_success, _y_steps, splits = step6.load_data(
        data_dir=args.data_dir
    )
    train_domains = splits["meta_train"]["domains"]
    test_domains = splits["meta_test"]["domains"]

    train_mask = np.isin(task_types, train_domains)
    X_surf_s = X_surf[train_mask]
    X_fm_s = X_fm[train_mask]
    y_train = y_success[train_mask]

    model = step6.load_checkpoint("success")
    empirical_llm = step6.load_empirical_llm_outcomes(
        args.llm_outcomes_jsonl,
        test_domains,
        data_dir=args.data_dir,
    )

    results = {}
    for dom in test_domains:
        if dom not in e1:
            raise KeyError(f"E1 JSON missing domain: {dom}")

        mask = task_types == dom
        Xs_q = X_surf[mask]
        Xf_q = X_fm[mask]
        y_proxy = y_success[mask]
        y_llm = empirical_llm[dom]
        n = len(y_llm)
        if len(y_proxy) != n:
            raise ValueError(f"Domain {dom}: proxy label length mismatch ({len(y_proxy)} vs {n})")

        target_usage = float(e1[dom]["best"]["llm_usage"])
        k = max(1, int(round(target_usage * n)))
        llm_usage = k / n

        guru_scores = step6.get_guru_scores_per_instance(
            model,
            Xs_q,
            Xf_q,
            y_proxy,
            X_surf_s,
            X_fm_s,
            n_runs=args.n_runs,
            rng_seed=args.seed,
        )
        surface_scores = step6.get_surface_scores_per_instance(
            Xs_q,
            y_proxy,
            X_surf_s,
            y_train,
            n_runs=args.n_runs,
            rng_seed=args.seed,
        )

        guru_validity, guru_precision = topk_hybrid_metrics(guru_scores, y_llm, k)
        surf_validity, surf_precision = topk_hybrid_metrics(surface_scores, y_llm, k)
        llm_only = float(y_llm.mean())
        blind = (1.0 - llm_usage) * 1.0 + llm_usage * llm_only

        results[dom] = {
            "n_instances": int(n),
            "target_llm_usage": float(target_usage),
            "k_routed_to_llm": int(k),
            "methods": {
                "llm_only": {
                    "plan_validity": llm_only,
                    "llm_usage": 1.0,
                    "precision": llm_only,
                },
                "blind_matched": {
                    "plan_validity": float(blind),
                    "llm_usage": float(llm_usage),
                    "precision": None,
                },
                "surface_routing": {
                    "plan_validity": float(surf_validity),
                    "llm_usage": float(llm_usage),
                    "precision": float(surf_precision),
                },
                "guru_routing": {
                    "plan_validity": float(guru_validity),
                    "llm_usage": float(llm_usage),
                    "precision": float(guru_precision),
                },
                "bfs_only": {
                    "plan_validity": 1.0,
                    "llm_usage": 0.0,
                    "precision": None,
                },
            },
        }

    tag = args.output_tag.strip() or "gpt4o"
    json_out = RESULTS_DIR / f"e1b_empirical_{tag}.json"
    tex_out = RESULTS_DIR / f"e1b_empirical_{tag}.tex"

    payload = {
        "meta": {
            "llm_outcomes_jsonl": str(Path(args.llm_outcomes_jsonl)),
            "e1_json": str(e1_path),
            "data_dir": str(Path(args.data_dir)),
            "output_tag": tag,
            "n_runs": int(args.n_runs),
            "seed": int(args.seed),
            "note": "Empirical LLM outcomes from step8 valid_plan labels; BFS fallback assumed perfect.",
        },
        "domains": results,
    }
    json_out.write_text(json.dumps(payload, indent=2))

    tex = build_latex_table(results)
    tex_out.write_text(tex)

    print("=" * 72)
    print("Step 9 - E1b empirical hybrid table from real LLM outcomes")
    for dom in ["blocksworld", "logistics", "mystery_blocksworld"]:
        m = results[dom]["methods"]
        print(
            f"  {dom:<22} LLM-only={m['llm_only']['plan_validity']:.1%} "
            f"Surface={m['surface_routing']['plan_validity']:.1%} "
            f"GURU={m['guru_routing']['plan_validity']:.1%} "
            f"(usage={m['guru_routing']['llm_usage']:.1%})"
        )
    print(f"  JSON -> {json_out}")
    print(f"  TEX  -> {tex_out}")


if __name__ == "__main__":
    main()
