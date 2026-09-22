"""
diagnose_step19.py
==================
Run on sc011 to diagnose why all 1800 instances show 0% validity.
Checks: Ollama connection, raw response format, parser, validator.

Usage:
  python diagnose_step19.py --host http://sc011:11434 --model qwen2.5:72b
"""
import argparse, json, re, urllib.request, time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "planning_ipc"

def call_ollama(host, model, prompt, timeout=120):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": 2048, "temperature": 0.0},
    }).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        ms = (time.perf_counter()-t0)*1000
        return data.get("response",""), float(ms), data.get("done",False)
    except Exception as e:
        return f"ERROR:{e}", (time.perf_counter()-t0)*1000, False

def _f(*parts): return " ".join(str(p).lower() for p in parts)

def parse_plan(response):
    if not response or "ERROR:" in response: return []
    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not txt: return []
    # After </think>
    parts = re.split(r"</think>", response, flags=re.DOTALL)
    if len(parts) > 1:
        lines = [l.strip() for l in parts[-1].split("\n") if l.strip().startswith("(")]
        if lines: return lines
    lines = [l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    found = re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", response, re.I)
    return found

def validate(record, actions):
    if not actions: return False, "empty_plan"
    state = {_f(*f.strip().strip("()").split()) for f in record.get("init_facts",[])}
    goals = {_f(*g.strip().strip("()").split()) for g in record.get("goal_facts",[])}
    if not state or not goals: return True, None
    H = {
        "pick-up":   (1,lambda x:({_f("clear",x),_f("ontable",x),_f("handempty")},{_f("ontable",x),_f("clear",x),_f("handempty")},{_f("holding",x)})),
        "put-down":  (1,lambda x:({_f("holding",x)},{_f("holding",x)},{_f("handempty"),_f("ontable",x),_f("clear",x)})),
        "stack":     (2,lambda x,y:({_f("holding",x),_f("clear",y)},{_f("holding",x),_f("clear",y)},{_f("handempty"),_f("on",x,y),_f("clear",x)})),
        "unstack":   (2,lambda x,y:({_f("on",x,y),_f("clear",x),_f("handempty")},{_f("on",x,y),_f("clear",x),_f("handempty")},{_f("holding",x),_f("clear",y)})),
        "grasp":     (1,lambda x:({_f("apex",x),_f("grounded",x),_f("grasping")},{_f("grounded",x),_f("apex",x),_f("grasping")},{_f("clutching",x)})),
        "release":   (1,lambda x:({_f("clutching",x)},{_f("clutching",x)},{_f("grasping"),_f("grounded",x),_f("apex",x)})),
        "place":     (2,lambda x,y:({_f("clutching",x),_f("apex",y)},{_f("clutching",x),_f("apex",y)},{_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
        "lift":      (2,lambda x,y:({_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
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
        name=toks[0].lower(); args=[t.lower() for t in toks[1:]]
        if name not in H: return False, f"unknown_action:{name}"
        ar,fn = H[name]
        if len(args)!=ar: return False, f"arity:{name}(need {ar} got {len(args)})"
        pre,rem,add = fn(*args)
        if not pre.issubset(state): return False, f"precondition_fail:{name}"
        state=(state-rem)|add
    ok=goals.issubset(state)
    return ok,(None if ok else "goal_not_reached")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="http://sc011:11434")
    p.add_argument("--model", default="qwen2.5:72b")
    args = p.parse_args()

    # ── Step A: validator self-test ───────────────────────────────────────────
    print("Step A: Validator self-test")
    fake_rec = {
        "task_type": "blocksworld",
        "init_facts": ["(ontable b1)","(clear b1)","(ontable b2)","(clear b2)","(handempty)"],
        "goal_facts":  ["(on b1 b2)"],
    }
    good_plan = ["(pick-up b1)","(stack b1 b2)"]
    ok, err = validate(fake_rec, good_plan)
    print(f"  Known-good plan: valid={ok} err={err}  {'PASS' if ok else 'FAIL'}")

    bad_plan = ["(stack b1 b2)"]
    ok2, err2 = validate(fake_rec, bad_plan)
    print(f"  Known-bad plan:  valid={ok2} err={err2}  {'PASS' if not ok2 else 'FAIL'}")

    # ── Step B: Check what's in the eval output already ──────────────────────
    eval_path = ROOT_DIR / "results_planning_ipc" / "qwen_ipc_eval.jsonl"
    if eval_path.exists():
        rows = [json.loads(l) for l in open(eval_path) if l.strip()]
        print(f"\nStep B: Existing eval output ({len(rows)} rows)")
        # Show first 5 rows
        for r in rows[:5]:
            resp = r.get("raw_response","")
            print(f"  id={r['instance_id']} dom={r['domain']} "
                  f"valid={r['valid_plan']} err={r['error_type']}")
            print(f"  n_actions={r['n_actions']} latency={r['latency_ms']:.0f}ms")
            print(f"  raw_response: {repr(resp[:200])}")
            print()
        # Check if responses are empty vs validator rejecting
        empty = sum(1 for r in rows if r.get("error_type")=="empty_plan")
        unknown = sum(1 for r in rows if r.get("error_type","").startswith("unknown_action"))
        prec = sum(1 for r in rows if r.get("error_type","").startswith("precondition"))
        arity = sum(1 for r in rows if r.get("error_type","").startswith("arity"))
        parse_err = sum(1 for r in rows if r.get("error_type")=="parse_error")
        print(f"  Error breakdown ({len(rows)} total):")
        print(f"    empty_plan:        {empty}")
        print(f"    unknown_action:    {unknown}")
        print(f"    precondition_fail: {prec}")
        print(f"    arity_mismatch:    {arity}")
        print(f"    parse_error:       {parse_err}")

    # ── Step C: Load first BW record and call Qwen ───────────────────────────
    labeled = DATA_DIR / "records_ipc_labeled.json"
    if not labeled.exists():
        print("\nStep C: records_ipc_labeled.json not found"); return

    records = json.loads(labeled.read_text())
    bw_recs = [r for r in records if r["task_type"]=="blocksworld"]
    # Use the easiest one (n_objects=2)
    easy = min(bw_recs, key=lambda r: r["n_objects"])
    print(f"\nStep C: Testing on easiest BW instance (n_objects={easy['n_objects']})")

    import re as _re
    action_names = _re.findall(r":action\s+(\S+)", easy.get("domain_pddl",""))
    action_list = "\nIMPORTANT: Use ONLY these action names:\n" + \
                  "\n".join(f"  {a}" for a in action_names) + "\n"

    prompt = (
        "You are a PDDL planner. Solve the problem below.\n\n"
        f"=== DOMAIN ===\n{easy['domain_pddl']}\n\n"
        f"=== PROBLEM ===\n{easy['problem_pddl']}\n\n"
        f"=== DESCRIPTION ===\n{easy['description']}\n"
        f"{action_list}\n"
        "Output ONLY the plan, one action per line: (action arg1 arg2 ...)\n"
        "If unsolvable output: NO_PLAN"
    )

    print(f"  Calling {args.model} at {args.host}...")
    response, ms, done = call_ollama(args.host, args.model, prompt, timeout=120)
    print(f"  Latency: {ms:.0f}ms  done={done}")
    print(f"  Raw response:\n---\n{response}\n---")

    # ── Step D: parse and validate ────────────────────────────────────────────
    print("\nStep D: Parse and validate")
    actions = parse_plan(response)
    print(f"  Parsed actions: {actions}")
    if actions:
        ok3, err3 = validate(easy, actions)
        print(f"  valid={ok3}  err={err3}")
    else:
        print("  No actions parsed — check raw response above")

    # ── Step E: check init_facts/goal_facts in record ─────────────────────────
    print(f"\nStep E: Record fields for instance {easy['instance_id']}")
    print(f"  init_facts: {easy.get('init_facts',[])[:5]}...")
    print(f"  goal_facts: {easy.get('goal_facts',[])}")
    print(f"  n_objects:  {easy['n_objects']}")
    print(f"  valid:      {easy.get('valid')}  n_steps: {easy.get('n_steps')}")


if __name__ == "__main__":
    main()
