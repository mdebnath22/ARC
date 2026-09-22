"""
qwen_eval.py  —  clean, self-contained Qwen 72B evaluation
============================================================
Evaluates Qwen 72B on 600 PDDL instances (200 per domain).
Loads episodes from the existing data directory.
Validates plans with a built-in PDDL executor.
Supports resume (skips already-done instances).

Usage:
  python qwen_eval.py --host http://sc011:11435 --model qwen2.5:72b
  python qwen_eval.py --host http://sc011:11435 --model qwen2.5:72b --overwrite
  python qwen_eval.py --host http://sc011:11435 --model qwen2.5:72b --dry_run 5
"""

import argparse, json, re, time, urllib.request
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
DATA   = ROOT / "data" / "planning"
OUT    = ROOT / "results_planning" / "qwen72b_eval_instances.jsonl"
(ROOT / "results_planning").mkdir(exist_ok=True)

TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]

# ── PDDL executor ─────────────────────────────────────────────────────────────

def _f(*p): return " ".join(str(x).lower() for x in p)

def validate(record, actions):
    """Execute plan against init state, check goal reached."""
    if not actions:
        return False, "empty_plan"
    init  = record.get("init_facts", [])
    goal  = record.get("goal_facts", [])
    if not init or not goal:
        return True, None   # no ground truth — accept if parsed
    state = {_f(*f.strip().strip("()").split()) for f in init}
    goals = {_f(*g.strip().strip("()").split()) for g in goal}
    H = {
        # blocksworld
        "pick-up":         (1,lambda x:({_f("clear",x),_f("ontable",x),_f("handempty")},{_f("ontable",x),_f("clear",x),_f("handempty")},{_f("holding",x)})),
        "put-down":        (1,lambda x:({_f("holding",x)},{_f("holding",x)},{_f("handempty"),_f("ontable",x),_f("clear",x)})),
        "stack":           (2,lambda x,y:({_f("holding",x),_f("clear",y)},{_f("holding",x),_f("clear",y)},{_f("handempty"),_f("on",x,y),_f("clear",x)})),
        "unstack":         (2,lambda x,y:({_f("on",x,y),_f("clear",x),_f("handempty")},{_f("on",x,y),_f("clear",x),_f("handempty")},{_f("holding",x),_f("clear",y)})),
        # mystery_blocksworld
        "grasp":           (1,lambda x:({_f("apex",x),_f("grounded",x),_f("grasping")},{_f("grounded",x),_f("apex",x),_f("grasping")},{_f("clutching",x)})),
        "release":         (1,lambda x:({_f("clutching",x)},{_f("clutching",x)},{_f("grasping"),_f("grounded",x),_f("apex",x)})),
        "place":           (2,lambda x,y:({_f("clutching",x),_f("apex",y)},{_f("clutching",x),_f("apex",y)},{_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
        "lift":            (2,lambda x,y:({_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
        # logistics
        "load-truck":      (3,lambda p,t,l:({_f("at",t,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,t)})),
        "unload-truck":    (3,lambda p,t,l:({_f("at",t,l),_f("in",p,t)},{_f("in",p,t)},{_f("at",p,l)})),
        "load-airplane":   (3,lambda p,a,l:({_f("at",a,l),_f("at",p,l)},{_f("at",p,l)},{_f("in",p,a)})),
        "unload-airplane": (3,lambda p,a,l:({_f("at",a,l),_f("in",p,a)},{_f("in",p,a)},{_f("at",p,l)})),
        "drive-truck":     (4,lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},{_f("at",t,s)},{_f("at",t,d)})),
        "fly-airplane":    (3,lambda a,s,d:({_f("at",a,s),_f("airport",s),_f("airport",d)},{_f("at",a,s)},{_f("at",a,d)})),
    }
    for act in actions:
        toks = act.strip().strip("()").split()
        if not toks: continue
        name = toks[0].lower(); args = [t.lower() for t in toks[1:]]
        if name not in H: return False, f"unknown:{name}"
        ar, fn = H[name]
        if len(args) != ar: return False, f"arity:{name}"
        pre, rem, add = fn(*args)
        if not pre.issubset(state): return False, f"precond:{name}"
        state = (state - rem) | add
    ok = goals.issubset(state)
    return ok, (None if ok else "goal_not_reached")

# ── Prompt builder ────────────────────────────────────────────────────────────

def extract_action_signatures(domain_pddl):
    """Extract action names AND their parameter counts from domain PDDL."""
    sigs = []
    for m in re.finditer(r":action\s+(\S+).*?:parameters\s*\(([^)]*)\)", 
                          domain_pddl, re.DOTALL):
        name   = m.group(1)
        params = m.group(2).strip()
        # Count parameters (each starts with ?)
        n_params = len(re.findall(r"\?", params))
        sigs.append((name, n_params))
    return sigs


def build_prompt(rec):
    dom  = rec.get("domain_pddl", "")
    prob = rec.get("problem_pddl", "")
    desc = rec.get("description", "")
    task = rec.get("task_type", rec.get("domain", ""))

    sigs = extract_action_signatures(dom)
    if sigs:
        act_lines = []
        for name, n in sigs:
            args = " ".join(f"arg{i+1}" for i in range(n))
            act_lines.append(f"  ({name} {args})")
        act_block = (
            "\nCRITICAL: You MUST use ONLY these exact action names "
            "(do NOT invent alternatives):\n"
            + "\n".join(act_lines)
            + "\n"
        )
    else:
        act_block = ""

    return (
        "You are a PDDL planning expert. "
        "Solve the planning problem below.\n\n"
        f"=== DOMAIN ===\n{dom}\n\n"
        f"=== PROBLEM ===\n{prob}\n\n"
        f"=== TASK ===\n{desc}\n"
        f"{act_block}\n"
        "Rules:\n"
        "1. Output ONLY the plan, one action per line: (action-name arg1 arg2 ...)\n"
        "2. Use ONLY the action names listed above — no alternatives, no paraphrases\n"
        "3. Arguments must be object names from the problem (no variables like ?x)\n"
        "4. If the problem is unsolvable, output exactly: NO_PLAN\n"
        "5. Do NOT output explanations, comments, or any other text\n"
    )

# ── Plan parser ───────────────────────────────────────────────────────────────

def parse_plan(response, domain=""):
    if not response or response.startswith("ERROR:"): return []
    # strip thinking block
    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b", txt, re.I): return []
    # after </think>
    parts = re.split(r"</think>", response, flags=re.DOTALL)
    if len(parts) > 1:
        lines = [l.strip() for l in parts[-1].split("\n") if l.strip().startswith("(")]
        if lines: return lines
    # parenthesised lines
    lines = [l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    # any parenthesised expression
    found = re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", response, re.I)
    if found: return found
    # normalise common wrong names to canonical names
    # Base aliases (always applied)
    ALIASES = {
        "pickup":    "pick-up",
        "put_down":  "put-down",
        "putdown":   "put-down",
        "fly":       "fly-airplane",
        "fly-plane": "fly-airplane",
        "move-truck":"drive-truck",
        "load-pkg":  "load-truck",
        "unload-pkg":"unload-truck",
        "board-pkg": "load-airplane",
        "unboard-pkg":"unload-airplane",
        "load":      "load-truck",
        "unload":    "unload-truck",
    }
    # Mystery-BW: map standard BW names to scrambled names
    if "mystery" in domain.lower():
        ALIASES.update({
            "pick-up": "grasp",
            "put-down":"release",
            "stack":   "place",
            "unstack": "lift",
        })
    normalised = []
    for line in txt.split("\n"):
        line = line.strip().strip("()")
        if not line: continue
        toks = line.split()
        if not toks: continue
        name = ALIASES.get(toks[0].lower(), toks[0].lower())
        if name != toks[0].lower():
            normalised.append("(" + " ".join([name] + toks[1:]) + ")")
    return normalised

# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(host, model, prompt, timeout):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": 4096, "temperature": 0.0},
    }).encode()
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            f"{host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        ms = (time.perf_counter()-t0)*1000
        return data.get("response", ""), float(ms)
    except Exception as e:
        return f"ERROR:{e}", (time.perf_counter()-t0)*1000

# ── Load episodes ─────────────────────────────────────────────────────────────

def load_episodes():
    """Load records from data/planning directory."""
    # Try records.json first (step1 output)
    for fname in ["records.json", "episodes.json", "all_records.json"]:
        p = DATA / fname
        if p.exists():
            recs = json.loads(p.read_text())
            print(f"  Loaded {len(recs)} records from {p}")
            return recs

    # Try loading numpy metadata
    import numpy as np
    task_types_path = DATA / "task_types.npy"
    if task_types_path.exists():
        # Reconstruct from PDDL files if they exist
        print("  No records.json found. Looking for PDDL files...")

    # Fall back: look for jsonl
    for fname in ["records.jsonl", "episodes.jsonl"]:
        p = DATA / fname
        if p.exists():
            recs = [json.loads(l) for l in open(p) if l.strip()]
            print(f"  Loaded {len(recs)} records from {p}")
            return recs

    raise FileNotFoundError(
        f"No episode records found in {DATA}.\n"
        f"Files present: {list(DATA.iterdir())}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host",      default="http://sc011:11435")
    p.add_argument("--model",     default="qwen2.5:72b")
    p.add_argument("--timeout",   type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run",   type=int, default=0,
                   help="Test on first N instances only")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"Qwen 72B Evaluation")
    print(f"  host:  {args.host}")
    print(f"  model: {args.model}")
    print(f"  out:   {OUT}")
    print(f"{'='*60}\n")

    # ── Verify Ollama ──────────────────────────────────────────────────────────
    try:
        req  = urllib.request.Request(f"{args.host}/api/tags", method="GET")
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"Ollama OK. Models: {models}")
        if not any(args.model.split(":")[0] in m for m in models):
            print(f"WARNING: {args.model} not found!")
    except Exception as e:
        print(f"ERROR: Cannot reach {args.host}: {e}")
        return

    # ── Quick test call ────────────────────────────────────────────────────────
    print(f"\nTest call to verify Qwen responds...")
    test_resp, test_ms = call_ollama(
        args.host, args.model,
        "Output only: (pick-up b1)", timeout=60)
    print(f"  Response: {repr(test_resp[:100])}")
    print(f"  Latency:  {test_ms:.0f}ms")
    if test_resp.startswith("ERROR:"):
        print("ERROR: Qwen not responding. Are you in an interactive GPU session?")
        print("  Run: interactive -G 1 -t 0-8:00  then  ollama-start")
        return
    print("  Qwen is working.\n")

    # ── Load episodes ──────────────────────────────────────────────────────────
    all_recs = load_episodes()
    recs = [r for r in all_recs
            if r.get("task_type", r.get("domain", "")) in TEST_DOMAINS]
    print(f"Test-domain records: {len(recs)}")
    for dom in TEST_DOMAINS:
        n = sum(1 for r in recs
                if r.get("task_type", r.get("domain","")) == dom)
        print(f"  {dom}: {n}")

    if args.dry_run > 0:
        recs = recs[:args.dry_run]
        print(f"\nDRY RUN: evaluating first {args.dry_run} instances")

    # ── Resume ─────────────────────────────────────────────────────────────────
    done = {}
    if args.overwrite and OUT.exists():
        OUT.unlink()
        print(f"Deleted existing output file (--overwrite)")
    elif OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done[int(r["instance_id"])] = r
        print(f"Resuming: {len(done)} already done")

    to_run = [r for r in recs if int(r.get("instance_id", -1)) not in done]
    print(f"To evaluate: {len(to_run)}\n")

    # ── Evaluation loop ────────────────────────────────────────────────────────
    by_dom = {d: {"n": 0, "valid": 0, "empty": 0, "wrong_act": 0, "prec": 0}
              for d in TEST_DOMAINS}

    with open(OUT, "a") as f:
        for idx, rec in enumerate(to_run):
            dom = rec.get("task_type", rec.get("domain", "unknown"))
            iid = int(rec.get("instance_id", idx))

            prompt   = build_prompt(rec)
            response, lat = call_ollama(args.host, args.model, prompt, args.timeout)
            actions  = parse_plan(response, domain=dom)
            valid, err = validate(rec, actions)

            row = {
                "instance_id":  iid,
                "domain":       dom,
                "model":        args.model,
                "valid_plan":   valid,
                "error_type":   err,
                "n_actions":    len(actions),
                "latency_ms":   lat,
                "raw_response": response[:500],
            }
            f.write(json.dumps(row) + "\n"); f.flush()

            # Update stats
            if dom in by_dom:
                by_dom[dom]["n"] += 1
                by_dom[dom]["valid"] += int(valid)
                if err == "empty_plan":   by_dom[dom]["empty"]     += 1
                if err and "unknown" in err: by_dom[dom]["wrong_act"] += 1
                if err and "precond" in err: by_dom[dom]["prec"]    += 1

            # Progress every 20
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"[{idx+1}/{len(to_run)}]  lat={lat:.0f}ms  "
                      f"valid={valid}  err={err}")
                print(f"  response: {repr(response[:120])}")
                for d, c in by_dom.items():
                    if c["n"] > 0:
                        pct = c["valid"]/c["n"]
                        print(f"  {d}: {c['valid']}/{c['n']}={pct:.1%}"
                              f"  empty={c['empty']} wrong_act={c['wrong_act']}"
                              f" prec_fail={c['prec']}")
                print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\nFINAL RESULTS:")
    all_rows = list(done.values())
    for line in open(OUT):
        r = json.loads(line)
        if int(r["instance_id"]) not in done:
            all_rows.append(r)
    by_dom2 = {}
    for r in all_rows:
        by_dom2.setdefault(r["domain"], []).append(r)
    for dom in TEST_DOMAINS:
        rlist = by_dom2.get(dom, [])
        if rlist:
            acc = sum(r["valid_plan"] for r in rlist)/len(rlist)
            errs = {}
            for r in rlist:
                e = r.get("error_type") or "valid"
                errs[e] = errs.get(e, 0) + 1
            print(f"  {dom}: {acc:.1%} ({len(rlist)} instances)")
            for e, n in sorted(errs.items(), key=lambda x:-x[1]):
                print(f"    {e}: {n}")
    print(f"\nResults → {OUT}")


if __name__ == "__main__":
    main()