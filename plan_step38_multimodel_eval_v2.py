"""
plan_step38_multimodel_eval_v2.py
====================================
Evaluate any Ollama model on BW/LOG/MBW using the REAL PDDL validator
(state simulation + precondition checking), not syntax-only matching.

Uses plan_validator.py, extracted from plan_step15_qwen72b_eval.py --
the validator that produced the paper's Table 7 numbers.

USAGE:
  python plan_step38_multimodel_eval_v2.py \
      --host http://sg016:11434 \
      --model mistral:7b \
      --n 200

Output: results_planning/multi_model_main_eval_v2/{model}_eval_instances.jsonl
        results_planning/multi_model_main_eval_v2/{model}_summary.json
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

from plan_validator import is_valid, parse_plan_lines

ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results_planning"
OUT_DIR = RES / "multi_model_main_eval_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def call_llm(host, model, prompt, timeout=90, keep_alive="30m"):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": 800, "temperature": 0.0}
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=payload,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
            return resp.get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def run_domain(host, model, domain, n, timeout, eps):
    domain_eps = [e for e in eps if e.get("task_type","")==domain][:n]
    results = []
    n_valid = 0
    t0 = time.time()
    reason_counts = {}

    for i, ep in enumerate(domain_eps):
        dom_pddl  = ep.get("domain_pddl", "")
        prob_pddl = ep.get("problem_pddl", "")
        actions = re.findall(r":action\s+(\S+)", dom_pddl)

        prompt = (
            "You are an expert AI planning assistant. Solve this "
            "classical planning problem.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Output ONLY the final plan, one action per line: "
            "(action-name param1 param2 ...)\n"
            "- Do NOT include any explanation after the plan.\n"
            "- If unsolvable, output exactly: NO_PLAN\n\n"
            f"=== DOMAIN ===\n{dom_pddl}\n\n"
            f"=== PROBLEM ===\n{prob_pddl}\n\n"
            f"Available actions: {', '.join(actions)}\n"
            "Now solve the problem:\n"
        )

        resp = call_llm(host, model, prompt, timeout=timeout)
        valid, reason = is_valid(resp, ep, dom_pddl)
        n_valid += int(valid)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        results.append({
            "domain": domain, "instance_id": ep.get("instance_id", i),
            "valid_plan": valid, "reason": reason,
            "n_steps": ep.get("n_steps", -1),
            "raw_response": resp[:300],
        })

        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            rate = n_valid / (i+1)
            print(f"    [{domain}] {i+1}/{len(domain_eps)}  "
                  f"valid={rate:.1%}  elapsed={elapsed:.0f}s", flush=True)

    solve_rate = n_valid / len(domain_eps) if domain_eps else 0
    return results, solve_rate, reason_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--domains", default="blocksworld,logistics,mystery_blocksworld")
    args = p.parse_args()

    domains = args.domains.split(",")
    eps = json.loads((ROOT/"data"/"planning"/"episodes.json").read_text())

    safe_name = args.model.replace(":", "_").replace("/", "_")
    jsonl_path = OUT_DIR / f"{safe_name}_eval_instances.jsonl"
    summary_path = OUT_DIR / f"{safe_name}_summary.json"

    print(f"=== Evaluating {args.model} on {domains} (N={args.n} each) ===")
    print("Using REAL state-simulating validator (precondition + goal checks)")

    all_results = []
    summary = {}
    for dom in domains:
        print(f"\n  Running {dom}...")
        t0 = time.time()
        results, solve_rate, reason_counts = run_domain(
            args.host, args.model, dom, args.n, args.timeout, eps)
        elapsed = time.time() - t0
        all_results.extend(results)
        summary[dom] = {
            "solve_rate": solve_rate, "n": len(results),
            "elapsed_s": elapsed,
            "avg_latency_s": elapsed/len(results) if results else 0,
            "reason_counts": reason_counts,
        }
        print(f"  {dom}: {solve_rate:.1%} solved ({len(results)} instances, "
              f"{elapsed:.0f}s)")
        print(f"    Reasons: {reason_counts}")

    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    summary_path.write_text(json.dumps({
        "model": args.model, "domains": summary
    }, indent=2))

    print(f"\n=== SUMMARY: {args.model} (real validator) ===")
    for dom, s in summary.items():
        print(f"  {dom:<22} {s['solve_rate']*100:>6.1f}%")
    print(f"\nSaved → {jsonl_path}")
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()
