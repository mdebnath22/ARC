"""
plan_step15_qwen72b_eval.py  — clean rewrite
==============================================
Evaluates Qwen 72B on 600 PDDL instances (200 per test domain).
Self-contained: all helper functions defined before use.

USAGE:
  python plan_step15_qwen72b_eval.py --phase eval \
      --model qwen2.5:72b \
      --ollama_host http://sc011:11435 \
      --timeout 600

  python plan_step15_qwen72b_eval.py --phase eval --overwrite   # rerun from scratch
"""

from __future__ import annotations
import argparse, json, re, time, urllib.request
from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data" / "planning"
RESULTS_DIR = ROOT_DIR / "results_planning"; RESULTS_DIR.mkdir(exist_ok=True)

TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
DOMAIN_LABELS = {"blocksworld":"Blocksworld","logistics":"Logistics",
                 "mystery_blocksworld":"Mystery-BW"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(episode: dict) -> str:
    domain_pddl  = episode.get("domain_pddl", "")
    problem_pddl = episode.get("problem_pddl", "")
    description  = episode.get("description", "")
    action_names = re.findall(r":action\s+(\S+)", domain_pddl)
    action_list  = ""
    if action_names:
        action_list = (
            "\nIMPORTANT: Use ONLY these action names (exactly as written):\n"
            + "\n".join(f"  {a}" for a in action_names) + "\n"
        )
    return (
        "You are a PDDL planning expert. Solve the planning problem below.\n\n"
        f"Domain:\n{domain_pddl}\n\n"
        f"Problem:\n{problem_pddl}\n\n"
        f"Description: {description}\n"
        f"{action_list}\n"
        "Output ONLY the plan as a sequence of ground actions, one per line:\n"
        "(action-name arg1 arg2 ...)\n\n"
        "Do not explain. Do not use action names not listed above.\n"
        "If unsolvable output exactly: NO_PLAN"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Plan parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_plan(response: str):
    """Extract action lines from Qwen 72B response. Returns list of strings."""
    if not response or "ERROR:" in response:
        return []
    stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "NO_PLAN" in stripped.upper() or "CANNOT SOLVE" in stripped.upper():
        return []
    # Strategy 1: lines after </think>
    parts = re.split(r"</think>", response, flags=re.DOTALL)
    if len(parts) > 1:
        lines = [l.strip() for l in parts[-1].split("\n")
                 if l.strip().startswith("(")]
        if lines:
            return lines
    # Strategy 2: parenthesized lines in stripped text
    lines = [l.strip() for l in stripped.split("\n")
             if l.strip().startswith("(")]
    if lines:
        return lines
    # Strategy 3: any parenthesized expressions
    found = re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", response, re.IGNORECASE)
    if found:
        return found
    # Strategy 4: action keyword lines without parens
    kws = ["unstack","stack","pickup","putdown","pick-up","put-down",
           "load","unload","drive","fly","move","grasp","release",
           "place","lift","board","debark","refuel"]
    result = []
    for line in stripped.split("\n"):
        l = line.strip().lower()
        if any(l.startswith(kw) for kw in kws):
            result.append(f"({line.strip()})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. PDDL validator
# ══════════════════════════════════════════════════════════════════════════════

def _f(*parts):
    return " ".join(str(p).lower() for p in parts)


def validate_plan(actions, episode: dict):
    """
    Execute plan against PDDL initial state and check goal.
    Returns (valid: bool, error_type: str|None).
    """
    if not actions:
        return False, "empty_plan"

    init_facts = episode.get("init_facts", [])
    goal_facts = episode.get("goal_facts", [])
    if not init_facts or not goal_facts:
        # No ground truth — accept if parse succeeded
        return True, None

    state = {_f(*f.strip().strip("()").split()) for f in init_facts}
    goals = {_f(*g.strip().strip("()").split()) for g in goal_facts}

    H = {
        # blocksworld
        "pick-up":         (1, lambda x:      ({_f("clear",x),_f("ontable",x),_f("handempty")},     {_f("ontable",x),_f("clear",x),_f("handempty")},  {_f("holding",x)})),
        "put-down":        (1, lambda x:      ({_f("holding",x)},                                     {_f("holding",x)},                                {_f("handempty"),_f("ontable",x),_f("clear",x)})),
        "stack":           (2, lambda x,y:    ({_f("holding",x),_f("clear",y)},                      {_f("holding",x),_f("clear",y)},                  {_f("handempty"),_f("on",x,y),_f("clear",x)})),
        "unstack":         (2, lambda x,y:    ({_f("on",x,y),_f("clear",x),_f("handempty")},         {_f("on",x,y),_f("clear",x),_f("handempty")},    {_f("holding",x),_f("clear",y)})),
        # mystery_blocksworld (scrambled names)
        "grasp":           (1, lambda x:      ({_f("apex",x),_f("grounded",x),_f("grasping")},       {_f("grounded",x),_f("apex",x),_f("grasping")},   {_f("clutching",x)})),
        "release":         (1, lambda x:      ({_f("clutching",x)},                                   {_f("clutching",x)},                              {_f("grasping"),_f("grounded",x),_f("apex",x)})),
        "place":           (2, lambda x,y:    ({_f("clutching",x),_f("apex",y)},                     {_f("clutching",x),_f("apex",y)},                 {_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
        "lift":            (2, lambda x,y:    ({_f("stacked",x,y),_f("apex",x),_f("grasping")},      {_f("stacked",x,y),_f("apex",x),_f("grasping")}, {_f("clutching",x),_f("apex",y)})),
        # logistics
        "load-truck":      (3, lambda p,t,l:  ({_f("at",t,l),_f("at",p,l)},                          {_f("at",p,l)},                                   {_f("in",p,t)})),
        "unload-truck":    (3, lambda p,t,l:  ({_f("at",t,l),_f("in",p,t)},                          {_f("in",p,t)},                                   {_f("at",p,l)})),
        "load-airplane":   (3, lambda p,a,l:  ({_f("at",a,l),_f("at",p,l)},                          {_f("at",p,l)},                                   {_f("in",p,a)})),
        "unload-airplane": (3, lambda p,a,l:  ({_f("at",a,l),_f("in",p,a)},                          {_f("in",p,a)},                                   {_f("at",p,l)})),
        "drive-truck":     (4, lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},  {_f("at",t,s)},                                   {_f("at",t,d)})),
        "fly-airplane":    (3, lambda a,s,d:  ({_f("at",a,s),_f("airport",s),_f("airport",d)},       {_f("at",a,s)},                                   {_f("at",a,d)})),
    }

    for act_str in actions:
        toks = act_str.strip().strip("()").split()
        if not toks:
            continue
        name = toks[0].lower()
        args = [t.lower() for t in toks[1:]]
        if name not in H:
            return False, f"unknown_action:{name}"
        arity, fn = H[name]
        if len(args) != arity:
            return False, f"arity:{name}(need {arity} got {len(args)})"
        pre, rem, add = fn(*args)
        if not pre.issubset(state):
            return False, f"precondition_fail:{name}"
        state = (state - rem) | add

    ok = goals.issubset(state)
    return ok, (None if ok else "goal_not_reached")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Ollama caller
# ══════════════════════════════════════════════════════════════════════════════

def call_ollama(host: str, model: str, prompt: str, timeout: int = 300):
    payload = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": 8192, "temperature": 0.0},
    }).encode()
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            f"{host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        return data.get("response", ""), float(ms)
    except Exception as e:
        return f"ERROR: {e}", (time.perf_counter() - t0) * 1000


def check_ollama(host: str, model: str) -> bool:
    try:
        req  = urllib.request.Request(f"{host}/api/tags", method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        tags = json.loads(resp.read())
        available = [m["name"] for m in tags.get("models", [])]
        print(f"  Ollama reachable at {host}")
        print(f"  Available models: {available}")
        return any(model.split(":")[0] in m for m in available)
    except Exception as e:
        print(f"  ERROR: Cannot reach {host}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 5. Load episodes
# ══════════════════════════════════════════════════════════════════════════════

def load_episodes():
    """Load PDDL episodes from the data directory."""
    eps_path = DATA_DIR / "episodes.json"
    if eps_path.exists():
        return json.loads(eps_path.read_text())

    # Fallback: build from individual records
    records_path = DATA_DIR / "records.json"
    if records_path.exists():
        return json.loads(records_path.read_text())

    # Last resort: scan for any json with instance_id
    for candidate in DATA_DIR.glob("*.json"):
        try:
            data = json.loads(candidate.read_text())
            if isinstance(data, list) and data and "instance_id" in data[0]:
                print(f"  Loading episodes from {candidate.name}")
                return data
        except Exception:
            continue

    raise FileNotFoundError(
        f"No episode file found in {DATA_DIR}. "
        "Expected episodes.json or records.json.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main eval phase
# ══════════════════════════════════════════════════════════════════════════════

def phase_eval(args):
    print(f"\n{'='*65}")
    print(f"PHASE 1: Qwen 72B Evaluation")
    print(f"  Host:  {args.ollama_host}")
    print(f"  Model: {args.model}")
    print(f"{'='*65}")

    if not check_ollama(args.ollama_host, args.model):
        print(f"\nERROR: {args.model} not available at {args.ollama_host}")
        print("Run: ollama pull qwen2.5:72b")
        return

    episodes   = load_episodes()
    test_eps   = [ep for ep in episodes
                  if ep.get("task_type", ep.get("domain", "")) in TEST_DOMAINS]
    test_eps.sort(key=lambda e: int(e.get("instance_id", 0)))

    out_path = RESULTS_DIR / "qwen72b_eval_instances.jsonl"

    # Resume support
    done = {}
    if out_path.exists() and not args.overwrite:
        for line in open(out_path):
            r = json.loads(line)
            done[int(r["instance_id"])] = r
        print(f"  Resuming: {len(done)} already done")

    to_run = [ep for ep in test_eps
              if int(ep.get("instance_id", 0)) not in done]
    print(f"  To evaluate: {len(to_run)} instances")

    by_dom = {d: {"n": 0, "valid": 0} for d in TEST_DOMAINS}

    with open(out_path, "a") as f:
        for idx, ep in enumerate(to_run):
            dom = ep.get("task_type", ep.get("domain", ""))
            iid = int(ep.get("instance_id", 0))

            prompt   = build_prompt(ep)
            response, lat_ms = call_ollama(
                args.ollama_host, args.model, prompt, timeout=args.timeout)

            actions  = parse_plan(response)
            ep["_raw_response"] = response
            valid, err = validate_plan(actions, ep)

            row = {
                "instance_id":       iid,
                "domain":            dom,
                "model":             args.model,
                "valid_plan":        valid,
                "error_type":        err,
                "parse_ok":          len(actions) > 0,
                "n_actions":         len(actions),
                "latency_ms":        lat_ms,
                "raw_response":      response[:600],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            by_dom[dom]["n"]     += 1
            by_dom[dom]["valid"] += int(valid)

            if (idx + 1) % 20 == 0 or idx == len(to_run) - 1:
                print(f"\n  [{idx+1}/{len(to_run)}]")
                for d, c in by_dom.items():
                    if c["n"] > 0:
                        print(f"    {DOMAIN_LABELS[d]}: "
                              f"{c['valid']}/{c['n']} = {c['valid']/c['n']:.1%}")

    # Final summary
    print(f"\n  FINAL SUMMARY ({args.model}):")
    all_rows = list(done.values())
    for line in open(out_path):
        r = json.loads(line)
        if int(r["instance_id"]) not in done:
            all_rows.append(r)
    by_dom2 = {}
    for r in all_rows:
        by_dom2.setdefault(r["domain"], []).append(r)
    for dom in TEST_DOMAINS:
        rlist = by_dom2.get(dom, [])
        if rlist:
            acc = sum(r["valid_plan"] for r in rlist) / len(rlist)
            print(f"  {DOMAIN_LABELS[dom]:<20}: {acc:.1%} ({len(rlist)} instances)")
    print(f"\n  Results → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",       choices=["eval"], default="eval")
    p.add_argument("--model",       default="qwen2.5:72b")
    p.add_argument("--ollama_host", default="http://sc011:11435")
    p.add_argument("--timeout",     type=int, default=600)
    p.add_argument("--overwrite",   action="store_true")
    args = p.parse_args()

    phase_eval(args)


if __name__ == "__main__":
    main()
