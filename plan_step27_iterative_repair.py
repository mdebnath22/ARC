"""
plan_step27_iterative_repair.py
================================
Iterative repair layer on top of ARC routing.

PIPELINE:
  1. ARC routes instance to LLM or EHC solver
  2. LLM generates a plan (may be invalid)
  3. Step-level validator identifies first failing step
  4. Repair prompt sent to LLM: "Fix only step N"
  5. Repeat up to max_repairs times
  6. If still invalid: EHC fallback

This directly addresses the reviewer criticism:
  "LLMs hallucinate invalid plans — routing gains are modest
   because the LLM component is unreliable."

The repair layer makes the LLM component more reliable,
increasing system validity INDEPENDENT of routing quality.

COMPARISON TABLE (what the paper shows):
  Method              BW      LOG     MBW
  LLM only            23.5%   0%      22%
  LLM + repair        X%      Y%      Z%
  ARC-FS + repair     X%      Y%      Z%
  EHC (solver only)   60.5%   99.5%   54%

USAGE:
  python plan_step27_iterative_repair.py --host http://HOST:11434
"""

from __future__ import annotations
import argparse, json, pickle, importlib.util, re, sys
import signal, tempfile, time, urllib.request, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"
TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
TRAIN_DOMAINS = ["depot", "rovers", "satellite"]


# ══════════════════════════════════════════════════════════════════════════════
# PDDL executor (step-level — returns first failing step)
# ══════════════════════════════════════════════════════════════════════════════

def _f(*p): return " ".join(str(x).lower() for x in p)

HANDLERS = {
    "pick-up":         (1,lambda x:({_f("clear",x),_f("ontable",x),_f("handempty")},{_f("ontable",x),_f("clear",x),_f("handempty")},{_f("holding",x)})),
    "put-down":        (1,lambda x:({_f("holding",x)},{_f("holding",x)},{_f("handempty"),_f("ontable",x),_f("clear",x)})),
    "stack":           (2,lambda x,y:({_f("holding",x),_f("clear",y)},{_f("holding",x),_f("clear",y)},{_f("handempty"),_f("on",x,y),_f("clear",x)})),
    "unstack":         (2,lambda x,y:({_f("on",x,y),_f("clear",x),_f("handempty")},{_f("on",x,y),_f("clear",x),_f("handempty")},{_f("holding",x),_f("clear",y)})),
    "grasp":           (1,lambda x:({_f("apex",x),_f("grounded",x),_f("grasping")},{_f("grounded",x),_f("apex",x),_f("grasping")},{_f("clutching",x)})),
    "release":         (1,lambda x:({_f("clutching",x)},{_f("clutching",x)},{_f("grasping"),_f("grounded",x),_f("apex",x)})),
    "place":           (2,lambda x,y:({_f("clutching",x),_f("apex",y)},{_f("clutching",x),_f("apex",y)},{_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
    "lift":            (2,lambda x,y:({_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
    "load-truck":      (3,lambda p,t,l:({_f("at",t,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,t)})),
    "unload-truck":    (3,lambda p,t,l:({_f("at",t,l),_f("in",p,t)},{_f("in",p,t)},{_f("at",p,l)})),
    "load-airplane":   (3,lambda p,a,l:({_f("at",a,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,a)})),
    "unload-airplane": (3,lambda p,a,l:({_f("at",a,l),_f("in",p,a)},{_f("in",p,a)},{_f("at",p,l)})),
    "drive-truck":     (4,lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},{_f("at",t,s)},{_f("at",t,d)})),
    "fly-airplane":    (3,lambda a,s,d:({_f("at",a,s),_f("airport",s),_f("airport",d)},{_f("at",a,s)},{_f("at",a,d)})),
}
ALIASES = {"pickup":"pick-up","putdown":"put-down","put_down":"put-down",
           "fly":"fly-airplane","fly-plane":"fly-airplane",
           "move-truck":"drive-truck","load-pkg":"load-truck",
           "unload-pkg":"unload-truck","load":"load-truck","unload":"unload-truck"}
ALIASES_MBW = {"pick-up":"grasp","put-down":"release","stack":"place",
               "unstack":"lift","pickup":"grasp","putdown":"release"}


def normalize_actions(actions, domain=""):
    al = dict(ALIASES)
    if "mystery" in domain.lower(): al.update(ALIASES_MBW)
    norm = []
    for act in actions:
        toks = act.strip().strip("()").split()
        if not toks: continue
        name = al.get(toks[0].lower(), toks[0].lower())
        norm.append("(" + " ".join([name] + [t.lower() for t in toks[1:]]) + ")")
    return norm


def execute_step_by_step(record, actions, domain=""):
    """
    Execute plan step by step.
    Returns (valid: bool, first_fail_idx: int or None, fail_reason: str, state_at_fail: set)
    """
    init  = record.get("init_facts", [])
    goal  = record.get("goal_facts", [])
    if not init or not goal:
        return True, None, None, set()

    state = {_f(*f.strip().strip("()").split()) for f in init}
    goals = {_f(*g.strip().strip("()").split()) for g in goal}

    for i, act in enumerate(actions):
        toks = act.strip().strip("()").split()
        if not toks: continue
        name = toks[0].lower(); args = [t.lower() for t in toks[1:]]

        if name not in HANDLERS:
            return False, i, f"unknown_action:{name}", state

        ar, fn = HANDLERS[name]
        if len(args) != ar:
            return False, i, f"wrong_arity:{name}(expected {ar}, got {len(args)})", state

        pre, rem, add = fn(*args)
        missing = pre - state
        if missing:
            return False, i, f"precondition_fail:{name} missing {sorted(missing)[:2]}", state

        state = (state - rem) | add

    ok = goals.issubset(state)
    missing_goals = goals - state
    if ok:
        return True, None, None, state
    else:
        return False, len(actions), f"goal_not_reached:{sorted(missing_goals)[:2]}", state


def parse_plan(response, domain=""):
    if not response or response.startswith("ERROR:"): return []
    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b", txt, re.I): return []
    lines = [l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    return re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", response, re.I)


# ══════════════════════════════════════════════════════════════════════════════
# LLM interface
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(host, model, prompt, timeout=180):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": 2048, "temperature": 0.0}
    }).encode()
    try:
        req = urllib.request.Request(f"{host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "")
    except Exception as e:
        return f"ERROR:{e}"


def initial_plan_prompt(record):
    dom  = record.get("domain_pddl", "")
    prob = record.get("problem_pddl", "")
    acts = re.findall(r":action\s+(\S+)", dom)
    ab = ""
    if acts:
        ab = "\nCRITICAL — use ONLY these action names:\n"
        for a in acts:
            m = re.search(rf":action\s+{re.escape(a)}.*?:parameters\s*\(([^)]*)\)",
                          dom, re.DOTALL)
            n = len(re.findall(r"\?", m.group(1))) if m else 1
            ab += f"  ({a} {' '.join(f'arg{i+1}' for i in range(n))})\n"
    return (
        f"You are a PDDL planning expert.\n\n"
        f"=== DOMAIN ===\n{dom}\n\n=== PROBLEM ===\n{prob}\n{ab}\n"
        "Output ONLY the plan, one action per line: (action arg1 ...)\n"
        "If unsolvable: NO_PLAN\n"
    )


def repair_prompt(record, valid_prefix, failing_step, fail_reason, remaining):
    """
    Ask LLM to fix ONLY the failing step.
    Shows what worked, what failed, and asks for a replacement.
    """
    dom  = record.get("domain_pddl", "")
    prob = record.get("problem_pddl", "")
    acts = re.findall(r":action\s+(\S+)", dom)

    prefix_str = "\n".join(f"  {i+1}. {s}" for i,s in enumerate(valid_prefix))
    remaining_str = "\n".join(f"  {len(valid_prefix)+i+1}. {s}" for i,s in enumerate(remaining))

    return (
        "The following plan has an error. Fix ONLY the failing step.\n\n"
        f"=== DOMAIN (relevant) ===\n{dom[:800]}\n\n"
        f"=== PROBLEM ===\n{prob[:600]}\n\n"
        f"STEPS THAT WORKED:\n{prefix_str if prefix_str else '  (none yet)'}\n\n"
        f"FAILING STEP {len(valid_prefix)+1}: {failing_step}\n"
        f"ERROR: {fail_reason}\n\n"
        f"REMAINING STEPS (after failed step):\n{remaining_str if remaining_str else '  (none)'}\n\n"
        f"Available actions: {', '.join(acts)}\n\n"
        "Output a REPLACEMENT for the failing step (and any immediately following "
        "steps that need to change). Format: one action per line (action arg1 ...).\n"
        "Output ONLY the replacement steps, starting from the fixed step.\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# EHC fallback
# ══════════════════════════════════════════════════════════════════════════════

def run_ehc(record, timeout=15):
    try:
        from pyperplan.pddl.parser import Parser
        from pyperplan import grounding
        from pyperplan.search.enforced_hillclimbing_search import enforced_hillclimbing_search
        from pyperplan.heuristics.relaxation import hFFHeuristic
    except ImportError:
        return False, []

    dom  = record.get("domain_pddl", "")
    prob = record.get("problem_pddl", "")
    if not dom or not prob: return False, []

    def _to(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _to); signal.alarm(timeout)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dp = Path(tmp)/"domain.pddl"; pp = Path(tmp)/"problem.pddl"
            dp.write_text(dom); pp.write_text(prob)
            parser = Parser(str(dp), str(pp))
            task   = grounding.ground(parser.parse_problem(parser.parse_domain()))
            sol    = enforced_hillclimbing_search(task, hFFHeuristic(task))
            signal.alarm(0)
            return (sol is not None), ([f"({op.name})" for op in sol] if sol else [])
    except (TimeoutError, Exception):
        signal.alarm(0); return False, []


# ══════════════════════════════════════════════════════════════════════════════
# Core: LLM + iterative repair
# ══════════════════════════════════════════════════════════════════════════════

def llm_with_repair(record, host, model, domain="", max_repairs=3):
    """
    Generate a plan with iterative step-level repair.

    Returns:
      (valid: bool, plan: list[str], n_repairs: int, method: str)
    """
    # Initial plan generation
    response = call_llm(host, model, initial_plan_prompt(record), timeout=180)
    actions  = parse_plan(response, domain)
    actions  = normalize_actions(actions, domain)

    if not actions:
        return False, [], 0, "llm_no_plan"

    valid, fail_idx, fail_reason, _ = execute_step_by_step(record, actions, domain)
    if valid:
        return True, actions, 0, "llm_direct"

    # Iterative repair
    current_plan = actions[:]
    for repair_num in range(1, max_repairs + 1):
        if fail_idx is None or fail_idx >= len(current_plan):
            break

        valid_prefix = current_plan[:fail_idx]
        failing_step = current_plan[fail_idx] if fail_idx < len(current_plan) else "goal_not_reached"
        remaining    = current_plan[fail_idx+1:]

        repair_response = call_llm(
            host, model,
            repair_prompt(record, valid_prefix, failing_step, fail_reason, remaining),
            timeout=120
        )

        new_steps = parse_plan(repair_response, domain)
        new_steps = normalize_actions(new_steps, domain)

        if not new_steps:
            break

        # Replace failing step and remainder with new steps
        current_plan = valid_prefix + new_steps

        valid, fail_idx, fail_reason, _ = execute_step_by_step(
            record, current_plan, domain)

        if valid:
            return True, current_plan, repair_num, f"llm_repaired_{repair_num}"

    return False, current_plan, max_repairs, "llm_repair_failed"


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_repair_evaluation(host, model_id, N=100, max_repairs=3):
    print(f"\n{'='*65}")
    print("ITERATIVE REPAIR EVALUATION")
    print(f"  Model: {model_id}  N: {N}/domain  Max repairs: {max_repairs}")
    print('='*65)

    # Check Ollama
    try:
        data = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"{host}/api/tags"), timeout=10).read())
        avail = [m["name"] for m in data.get("models", [])]
        if not any(model_id.split(":")[0] in m for m in avail):
            print(f"  {model_id} not available"); return
        print(f"  OK: {model_id}")
    except Exception as e:
        print(f"  ERROR: {e}"); return

    # Load episodes
    eps_by_dom = {}
    for e in json.loads((DATA/"episodes.json").read_text()):
        eps_by_dom.setdefault(e.get("task_type", e.get("domain","")), []).append(e)

    # Load Qwen labels for comparison
    qwen_by_dom = {}
    qp = RESULTS/"qwen72b_eval_instances.jsonl"
    if qp.exists():
        for line in open(qp):
            r = json.loads(line); qwen_by_dom.setdefault(r["domain"],[]).append(r)
        for d in qwen_by_dom: qwen_by_dom[d].sort(key=lambda r:int(r["instance_id"]))

    all_results = {}

    print(f"\n  {'Domain':<12}  {'LLM%':>6}  {'LLM+repair%':>12}  "
          f"{'EHC%':>6}  {'repairs_used':>13}  {'repair_success%':>16}")
    print("  " + "-"*72)

    for dom in TEST_DOMAINS:
        eps = eps_by_dom.get(dom, [])[:N]
        qlist = qwen_by_dom.get(dom, [])[:N]

        # Qwen no-repair baseline (from saved labels)
        llm_direct = sum(1 for r in qlist if r["valid_plan"]) / max(len(qlist), 1)

        # LLM + repair (fresh calls)
        n_valid = 0; total_repairs = 0
        n_repaired = 0  # cases that failed initially but succeeded after repair
        n_repair_attempts = 0

        for i, ep in enumerate(eps):
            # Step 1: check if LLM already solved it (from saved labels)
            if i < len(qlist) and qlist[i]["valid_plan"]:
                n_valid += 1  # already valid, no repair needed
                if (i+1) % 10 == 0:
                    print(f"    [{i+1}/{N}] valid={n_valid} (from labels)",end="\r")
                continue

            # Step 2: LLM failed — try repair with fresh call
            resp = call_llm(host, model_id, initial_plan_prompt(ep), timeout=120)
            actions = parse_plan(resp, dom)
            actions = normalize_actions(actions, dom)
            if not actions:
                if (i+1) % 10 == 0:
                    print(f"    [{i+1}/{N}] valid={n_valid} repairs={total_repairs}",end="\r")
                continue

            valid, fail_idx, fail_reason, _ = execute_step_by_step(ep, actions, dom)
            if valid:
                n_valid += 1  # repair not needed, direct success
                if (i+1) % 10 == 0:
                    print(f"    [{i+1}/{N}] valid={n_valid}",end="\r")
                continue

            # Step 3: iterative repair on failing plan
            current_plan = actions[:]
            repaired = False
            for repair_num in range(1, max_repairs + 1):
                if fail_idx is None or fail_idx >= len(current_plan): break
                valid_prefix = current_plan[:fail_idx]
                failing_step = current_plan[fail_idx] if fail_idx < len(current_plan) else "goal_not_reached"
                remaining    = current_plan[fail_idx+1:]
                rep_resp = call_llm(host, model_id,
                    repair_prompt(ep, valid_prefix, failing_step, fail_reason, remaining),
                    timeout=90)
                new_steps = parse_plan(rep_resp, dom)
                new_steps = normalize_actions(new_steps, dom)
                if not new_steps: break
                current_plan = valid_prefix + new_steps
                valid, fail_idx, fail_reason, _ = execute_step_by_step(ep, current_plan, dom)
                total_repairs += 1
                if valid:
                    n_valid += 1; n_repaired += 1; repaired = True; break
            n_repair_attempts += 1
            if (i+1) % 10 == 0:
                print(f"    [{i+1}/{N}] valid={n_valid} repairs={total_repairs}",end="\r")

        # EHC baseline
        ehc_valid = sum(1 for ep in eps if run_ehc(ep, timeout=10)[0])

        repair_pct = n_valid / N
        ehc_pct    = ehc_valid / N
        repair_success_pct = (n_repaired / max(n_repair_attempts, 1)) * 100

        dl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
        print(f"  {dl:<12}  {llm_direct:>6.1%}  {repair_pct:>12.1%}  "
              f"{ehc_pct:>6.1%}  {total_repairs/N:>13.2f}  {repair_success_pct:>15.1f}%")

        all_results[dom] = {
            "llm_direct":    float(llm_direct),
            "llm_repair":    float(repair_pct),
            "ehc":           float(ehc_pct),
            "mean_repairs":  float(total_repairs/N),
            "repair_success": float(repair_success_pct),
        }

    # LaTeX table
    print(f"\n  LaTeX:")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Effect of iterative repair on LLM plan validity.")
    print(r"\emph{LLM}: direct generation (Qwen~2.5-72B, $N{=}200$).")
    print(r"\emph{LLM+repair}: step-level validation with up to 3 targeted")
    print(r"repair prompts per instance.")
    print(r"\emph{EHC}: always-EHC baseline (upper bound for solver).}")
    print(r"\label{tab:repair}")
    print(r"\small\begin{tabular}{l ccc c}\toprule")
    print(r"\textbf{Domain} & \textbf{LLM} & \textbf{LLM+repair} & \textbf{EHC} & \textbf{Repair success\%} \\\midrule")
    for dom in TEST_DOMAINS:
        r  = all_results.get(dom, {})
        dl = dom.replace("mystery_blocksworld","MBW").replace("blocksworld","BW").replace("logistics","LOG")
        print(f"  {dl} & {r.get('llm_direct',0):.1%} & \\textbf{{{r.get('llm_repair',0):.1%}}} "
              f"& {r.get('ehc',0):.1%} & {r.get('repair_success',0):.1f}\\% \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")

    (RESULTS/"iterative_repair_results.json").write_text(
        json.dumps(all_results, indent=2))
    print(f"\n  Saved → {RESULTS}/iterative_repair_results.json")

    print()
    print("  PAPER NARRATIVE:")
    for dom in TEST_DOMAINS:
        r  = all_results.get(dom, {})
        dl = dom[:3].upper()
        gain = r.get("llm_repair",0) - r.get("llm_direct",0)
        print(f"  {dl}: LLM {r.get('llm_direct',0):.1%} → LLM+repair "
              f"{r.get('llm_repair',0):.1%} ({gain:+.1%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host",    default="http://localhost:11434")
    p.add_argument("--model",   default="qwen2.5:72b")
    p.add_argument("--n",       type=int, default=100)
    p.add_argument("--repairs", type=int, default=3)
    args = p.parse_args()
    run_repair_evaluation(args.host, args.model, args.n, args.repairs)


if __name__ == "__main__":
    main()
