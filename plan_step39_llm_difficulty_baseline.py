"""
plan_step39_llm_difficulty_baseline.py
========================================
Addresses the TMLR reviewer's core missing-baseline objection:
"A purely LLM-based baseline that predicts problem difficulty using
the same conditioning information as ARC (training domains, labels
for training domains, structural features) should be provided."

Design: give an LLM the support set (labeled source-domain instances:
PDDL problem + BFS-optimal plan length) as in-context examples, then
ask it to predict a difficulty SCORE (not a plan) for each query
instance in the held-out domain. No ARC architecture, no attention,
no training — pure in-context learning with the identical information
ARC's support set provides.

Two variants:
  --mode rank     : ask for a 0-100 difficulty score directly
  --mode pairwise : ask the LLM to compare two instances and say
                    which is harder (more reliable for LLMs, then
                    convert pairwise comparisons to a ranking)

USAGE:
  python plan_step39_llm_difficulty_baseline.py \
      --host http://sg049:11434 --model qwen2.5:72b \
      --mode rank --n_support 20 --n_query 200

Output: results_planning/llm_difficulty_baseline_{mode}_{model}.json
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results_planning"; RES.mkdir(exist_ok=True)
TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]


def call_llm(host, model, prompt, timeout=120, keep_alive="30m"):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": 300, "temperature": 0.0}
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=payload,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
            return resp.get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def build_support_block(support_eps):
    """Render labeled support examples as in-context demonstrations."""
    lines = []
    for i, ep in enumerate(support_eps):
        prob = ep.get("problem_pddl", "")[:400]  # truncate for context budget
        n_steps = ep.get("n_steps", -1)
        lines.append(
            f"--- Example {i+1} ---\n"
            f"Problem:\n{prob}\n"
            f"Optimal plan length: {n_steps}\n"
        )
    return "\n".join(lines)


def build_rank_prompt(support_block, query_prob, domain_pddl):
    return (
        "You are an expert at analyzing PDDL planning problems. "
        "Below are labeled examples from RELATED planning domains, "
        "each showing a problem and its optimal (shortest) plan length. "
        "The query problem is from a DIFFERENT domain with different "
        "predicates and actions, but similar structural principles "
        "(object counts, goal complexity, initial state structure) "
        "may transfer.\n\n"
        f"=== LABELED EXAMPLES (from other domains) ===\n{support_block}\n\n"
        "=== QUERY DOMAIN ===\n"
        f"{domain_pddl[:800]}\n\n"
        "=== QUERY PROBLEM (predict difficulty for this one) ===\n"
        f"{query_prob[:600]}\n\n"
        "Based on the pattern in the labeled examples (how problem "
        "structure relates to plan length), estimate a DIFFICULTY SCORE "
        "for the query problem, from 0 (trivially easy, very short plan) "
        "to 100 (very hard, very long plan).\n"
        "Output ONLY a single integer between 0 and 100. No explanation.\n"
        "Score:"
    )


def build_pairwise_prompt(support_block, prob_a, prob_b, domain_pddl):
    return (
        "You are an expert at analyzing PDDL planning problems. "
        "Below are labeled examples from RELATED planning domains, "
        "each showing a problem and its optimal (shortest) plan length.\n\n"
        f"=== LABELED EXAMPLES (from other domains) ===\n{support_block}\n\n"
        "=== QUERY DOMAIN ===\n"
        f"{domain_pddl[:800]}\n\n"
        "=== PROBLEM A ===\n"
        f"{prob_a[:500]}\n\n"
        "=== PROBLEM B ===\n"
        f"{prob_b[:500]}\n\n"
        "Which problem requires a LONGER optimal plan (is harder)? "
        "Output ONLY 'A' or 'B'. No explanation.\n"
        "Answer:"
    )


def parse_score(resp):
    if resp.startswith("ERROR:"):
        return None
    m = re.search(r"-?\d+", resp)
    if m:
        v = int(m.group())
        return max(0, min(100, v))
    return None


def run_rank_mode(host, model, eps, tt, y_ns, n_support, n_query):
    """Direct scoring: LLM assigns a 0-100 difficulty score per instance."""
    rng = np.random.default_rng(42)
    train_domains = ["depot", "rovers", "satellite"]
    train_idx = np.where(np.isin(tt, train_domains))[0]
    support_idx = rng.choice(train_idx, min(n_support, len(train_idx)), replace=False)
    support_eps = [eps[i] for i in support_idx]
    support_block = build_support_block(support_eps)

    results = {}
    for dom in TEST_DOMAINS:
        dom_idx = np.where(tt == dom)[0][:n_query]
        dom_eps = [eps[i] for i in dom_idx]
        dom_pddl = dom_eps[0].get("domain_pddl", "") if dom_eps else ""

        scores = []
        n_errors = 0
        print(f"\n  Domain: {dom} (N={len(dom_eps)})")
        t0 = time.time()
        for i, ep in enumerate(dom_eps):
            prompt = build_rank_prompt(support_block, ep.get("problem_pddl",""), dom_pddl)
            resp = call_llm(host, model, prompt)
            if resp.startswith("ERROR:"):
                n_errors += 1
            score = parse_score(resp)
            scores.append(score if score is not None else -1)
            if i < 5:
                print(f"    [sample {i}] score={score}  raw_response={resp[:60]!r}")
            if (i+1) % 20 == 0:
                print(f"    {i+1}/{len(dom_eps)}  elapsed={time.time()-t0:.0f}s  errors={n_errors}")
                if n_errors >= 10:
                    print(f"    *** ABORTING DOMAIN: {n_errors} HTTP errors in "
                          f"first {i+1} calls — Ollama/model likely down ***")
                    break

        scores = np.array(scores, dtype=float)
        ns_arr = np.array([e.get("n_steps", -1) for e in dom_eps], dtype=float)
        valid = (scores >= 0) & (ns_arr > 0)

        print(f"    Score distribution: unique={np.unique(scores[valid])[:10]} "
              f"(showing up to 10 unique values)")
        print(f"    Score std: {scores[valid].std():.2f}  "
              f"mean: {scores[valid].mean():.2f}")

        if valid.sum() > 5 and scores[valid].std() > 0:
            rho, p = stats.spearmanr(scores[valid], ns_arr[valid])
        else:
            rho, p = 0.0, 1.0
            print(f"    WARNING: constant or near-constant scores, rho undefined -> set to 0")

        results[dom] = {
            "rho": float(abs(rho)), "p": float(p),
            "n_valid": int(valid.sum()), "n_total": len(dom_eps),
            "n_parse_failures": int((scores < 0).sum()),
        }
        print(f"  {dom}: |rho|={abs(rho):.3f} (n_valid={valid.sum()}/{len(dom_eps)})")

    return results


def run_pairwise_mode(host, model, eps, tt, y_ns, n_support, n_query, n_pairs=100):
    """Pairwise comparison: more robust for LLMs than absolute scoring."""
    rng = np.random.default_rng(42)
    train_domains = ["depot", "rovers", "satellite"]
    train_idx = np.where(np.isin(tt, train_domains))[0]
    support_idx = rng.choice(train_idx, min(n_support, len(train_idx)), replace=False)
    support_eps = [eps[i] for i in support_idx]
    support_block = build_support_block(support_eps)

    results = {}
    for dom in TEST_DOMAINS:
        dom_idx = np.where(tt == dom)[0][:n_query]
        dom_eps = [eps[i] for i in dom_idx]
        dom_pddl = dom_eps[0].get("domain_pddl", "") if dom_eps else ""
        ns_arr = np.array([e.get("n_steps", -1) for e in dom_eps], dtype=float)

        # Sample random pairs, get pairwise comparisons, build a score via
        # win-count (Copeland-style ranking from pairwise judgments)
        n = len(dom_eps)
        win_count = np.zeros(n)
        compare_count = np.zeros(n)

        pairs = [(rng.integers(0,n), rng.integers(0,n)) for _ in range(n_pairs)]
        pairs = [(a,b) for a,b in pairs if a != b]

        print(f"\n  Domain: {dom} ({len(pairs)} pairwise comparisons)")
        t0 = time.time()
        for i, (a, b) in enumerate(pairs):
            prob_a = dom_eps[a].get("problem_pddl", "")
            prob_b = dom_eps[b].get("problem_pddl", "")
            prompt = build_pairwise_prompt(support_block, prob_a, prob_b, dom_pddl)
            resp = call_llm(host, model, prompt)
            resp_clean = resp.strip().upper()
            if resp.startswith("ERROR:"):
                continue  # skip failed calls entirely, don't count as compared
            compare_count[a] += 1; compare_count[b] += 1
            if "A" in resp_clean[:5] and "B" not in resp_clean[:5]:
                win_count[a] += 1
            elif "B" in resp_clean[:5] and "A" not in resp_clean[:5]:
                win_count[b] += 1
            # ties/unparseable: no win assigned

            if (i+1) % 20 == 0:
                print(f"    {i+1}/{len(pairs)}  elapsed={time.time()-t0:.0f}s")

        # Score = win rate (proxy for relative difficulty rank)
        scores = np.divide(win_count, compare_count,
                           out=np.zeros_like(win_count), where=compare_count>0)
        valid = (compare_count > 0) & (ns_arr > 0)

        if valid.sum() > 5:
            rho, p = stats.spearmanr(scores[valid], ns_arr[valid])
        else:
            rho, p = 0.0, 1.0

        results[dom] = {
            "rho": float(abs(rho)), "p": float(p),
            "n_valid": int(valid.sum()), "n_total": n,
            "n_pairs": len(pairs),
        }
        print(f"  {dom}: |rho|={abs(rho):.3f} (n_valid={valid.sum()}/{n}, "
              f"{len(pairs)} pairs)")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["rank", "pairwise"], default="rank")
    p.add_argument("--n_support", type=int, default=20)
    p.add_argument("--n_query", type=int, default=200)
    p.add_argument("--n_pairs", type=int, default=100)
    args = p.parse_args()

    eps = json.loads((ROOT/"data"/"planning"/"episodes.json").read_text())
    tt = np.array([e.get("task_type","") for e in eps])
    y_ns = np.array([e.get("n_steps",-1) for e in eps], dtype=float)

    print(f"=== LLM-only difficulty baseline: {args.model} ({args.mode} mode) ===")
    print(f"Support set: {args.n_support} labeled source-domain examples")
    print(f"Query set: {args.n_query} instances per test domain")

    if args.mode == "rank":
        results = run_rank_mode(args.host, args.model, eps, tt, y_ns,
                                 args.n_support, args.n_query)
    else:
        results = run_pairwise_mode(args.host, args.model, eps, tt, y_ns,
                                     args.n_support, args.n_query, args.n_pairs)

    mean_rho = np.mean([v["rho"] for v in results.values()])

    print(f"\n=== SUMMARY (LLM-only baseline, {args.mode}) ===")
    for dom, v in results.items():
        print(f"  {dom:<22} |rho|={v['rho']:.3f}")
    print(f"  Mean |rho|: {mean_rho:.3f}")
    print(f"\n  Compare to ARC: mean |rho|=0.737 (Table 2)")
    print(f"  Compare to |O|: mean |rho|=0.658 (Table 2)")

    safe_model = args.model.replace(":","_")
    out_path = RES / f"llm_difficulty_baseline_{args.mode}_{safe_model}.json"
    out_path.write_text(json.dumps({
        "model": args.model, "mode": args.mode,
        "n_support": args.n_support, "results": results,
        "mean_rho": float(mean_rho),
    }, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
