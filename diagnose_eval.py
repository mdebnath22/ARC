"""
diagnose_eval.py — run on cluster to diagnose Qwen eval.
Usage: python diagnose_eval.py --host http://sc015:11435 --model qwen2.5:72b
"""
import argparse, json, re, time, urllib.request
from typing import List, Optional, Tuple

# ── Parser ─────────────────────────────────────────────────────────────────
PAREN_RE = re.compile(r"\(([^()\n]+)\)")
FENCE_RE  = re.compile(r"```[a-zA-Z0-9_-]*")
SKIP      = {"define","domain","problem","requirements","predicates","action",
             "parameters","precondition","effect","objects","init","goal",
             "types","and","not","or","when","forall","exists","strips"}

def _tok(toks):
    if not toks or len(toks) > 6: return None
    if any(("?" in t) or (":" in t) for t in toks): return None
    h = toks[0].lower()
    if h in SKIP: return None
    args = [t.lower() for t in toks[1:]]
    return h, args, "("+  " ".join([h]+args) +")"

def extract_actions(raw: str):
    if not raw: return [], False, "empty_plan"
    txt = FENCE_RE.sub("", raw)
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
    if not txt or txt.lower().strip(".!") in {"no-plan","no plan","no_plan","unsat","unsolvable"}:
        return [], False, "empty_plan"
    acts = []
    for line in txt.splitlines():
        line = re.sub(r"^\d+[.)\-]\s*", "", line.strip())
        if not line: continue
        m = re.match(r"^\(([^()]+)\)\s*$", line)
        if m:
            p = _tok(m.group(1).split())
            if p: acts.append(p); continue
        m2 = re.match(r"^([a-z][a-z0-9_\-]*)(.*)$", line, re.I)
        if m2:
            p = _tok([m2.group(1)] + m2.group(2).split())
            if p: acts.append(p)
    if acts: return acts, True, None
    for m in PAREN_RE.finditer(txt):
        p = _tok(m.group(1).split())
        if p: acts.append(p)
    return (acts, True, None) if acts else ([], False, "parse_error")

# ── Validator ──────────────────────────────────────────────────────────────
def _f(*p): return " ".join(str(x).lower() for x in p)

def validate_plan_pddl(domain, init_facts, goal_facts, parsed):
    state = {_f(*f.strip().strip("()").split()) for f in init_facts}
    goals = {_f(*g.strip().strip("()").split()) for g in goal_facts}
    H = {
        "pick-up":         (1, lambda x:      ({_f("clear",x),_f("ontable",x),_f("handempty")},     {_f("ontable",x),_f("clear",x),_f("handempty")}, {_f("holding",x)})),
        "put-down":        (1, lambda x:      ({_f("holding",x)},                                    {_f("holding",x)},                               {_f("handempty"),_f("ontable",x),_f("clear",x)})),
        "stack":           (2, lambda x,y:    ({_f("holding",x),_f("clear",y)},                     {_f("holding",x),_f("clear",y)},                 {_f("handempty"),_f("on",x,y),_f("clear",x)})),
        "unstack":         (2, lambda x,y:    ({_f("on",x,y),_f("clear",x),_f("handempty")},        {_f("on",x,y),_f("clear",x),_f("handempty")},   {_f("holding",x),_f("clear",y)})),
        "grasp":           (1, lambda x:      ({_f("apex",x),_f("grounded",x),_f("grasping")},      {_f("grounded",x),_f("apex",x),_f("grasping")},  {_f("clutching",x)})),
        "release":         (1, lambda x:      ({_f("clutching",x)},                                  {_f("clutching",x)},                             {_f("grasping"),_f("grounded",x),_f("apex",x)})),
        "place":           (2, lambda x,y:    ({_f("clutching",x),_f("apex",y)},                    {_f("clutching",x),_f("apex",y)},                {_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
        "lift":            (2, lambda x,y:    ({_f("stacked",x,y),_f("apex",x),_f("grasping")},     {_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
        "load-truck":      (3, lambda p,t,l:  ({_f("at",t,l),_f("at",p,l)},                         {_f("at",p,l)},                                  {_f("in",p,t)})),
        "unload-truck":    (3, lambda p,t,l:  ({_f("at",t,l),_f("in",p,t)},                         {_f("in",p,t)},                                  {_f("at",p,l)})),
        "load-airplane":   (3, lambda p,a,l:  ({_f("at",a,l),_f("at",p,l)},                         {_f("at",p,l)},                                  {_f("in",p,a)})),
        "unload-airplane": (3, lambda p,a,l:  ({_f("at",a,l),_f("in",p,a)},                         {_f("in",p,a)},                                  {_f("at",p,l)})),
        "drive-truck":     (4, lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)}, {_f("at",t,s)},                                  {_f("at",t,d)})),
        "fly-airplane":    (3, lambda a,s,d:  ({_f("at",a,s),_f("airport",s),_f("airport",d)},      {_f("at",a,s)},                                  {_f("at",a,d)})),
    }
    for name, args, canon in parsed:
        if name not in H: return False, f"unknown_action:{name}"
        ar, fn = H[name]
        if len(args) != ar: return False, f"arity:{name}"
        pre, rem, add = fn(*args)
        if not pre.issubset(state): return False, f"precondition_fail:{name}"
        state = (state - rem) | add
    ok = goals.issubset(state)
    return ok, (None if ok else "goal_not_reached")


BW_DOMAIN = """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (on ?x ?y)(ontable ?x)(clear ?x)(handempty)(holding ?x))
  (:action pick-up :parameters (?x)
    :precondition (and (clear ?x)(ontable ?x)(handempty))
    :effect (and (not (ontable ?x))(not (clear ?x))(not (handempty))(holding ?x)))
  (:action put-down :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x))(handempty)(ontable ?x)(clear ?x)))
  (:action stack :parameters (?x ?y)
    :precondition (and (holding ?x)(clear ?y))
    :effect (and (not (holding ?x))(not (clear ?y))(handempty)(on ?x ?y)(clear ?x)))
  (:action unstack :parameters (?x ?y)
    :precondition (and (on ?x ?y)(clear ?x)(handempty))
    :effect (and (not (on ?x ?y))(not (handempty))(holding ?x)(clear ?y))))"""

BW_PROBLEM = """(define (problem test) (:domain blocksworld)
  (:objects b1 b2)
  (:init (ontable b1)(clear b1)(ontable b2)(clear b2)(handempty))
  (:goal (and (on b1 b2))))"""

BW_INIT = ["(ontable b1)","(clear b1)","(ontable b2)","(clear b2)","(handempty)"]
BW_GOAL = ["(on b1 b2)"]

def call_ollama(host, model, prompt, timeout=120):
    payload = json.dumps({"model":model,"prompt":prompt,"stream":False,
        "options":{"num_predict":512,"temperature":0.0}}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(f"{host}/api/generate", data=payload,
        headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data.get("response",""), (time.perf_counter()-t0)*1000

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="http://localhost:11434")
    p.add_argument("--model", default="qwen2.5:72b")
    args = p.parse_args()

    # Step A: parser self-test (no Ollama needed)
    print("Step A: Parser + validator self-test")
    parsed, ok, err = extract_actions("(pick-up b1)\n(stack b1 b2)")
    valid, verr = validate_plan_pddl("blocksworld", BW_INIT, BW_GOAL, parsed) if ok else (False,"parse_fail")
    status = "PASS" if (ok and valid) else "FAIL"
    print(f"  [{status}] parse_ok={ok} valid={valid} err={verr}")

    # Step B: call Ollama
    import re as _re
    action_names = _re.findall(r":action\s+(\S+)", BW_DOMAIN)
    prompt = (f"{BW_DOMAIN}\n\n{BW_PROBLEM}\n\n"
              f"IMPORTANT: Use ONLY these action names:\n"
              + "\n".join(f"  {a}" for a in action_names)
              + "\n\nOutput ONLY the plan, one action per line: (action arg1 ...)\n"
              "If unsolvable: NO_PLAN")
    print(f"\nStep B: Calling {args.model} at {args.host}")
    try:
        response, ms = call_ollama(args.host, args.model, prompt)
        print(f"  Latency: {ms:.0f}ms")
        print(f"  Raw response:\n---\n{response}\n---")
    except Exception as e:
        print(f"  ERROR: {e}"); return

    # Step C: parse + validate
    print("\nStep C: Parse + validate Qwen response")
    parsed, ok, err = extract_actions(response)
    print(f"  parse_ok={ok}  err={err}  actions={parsed}")
    if ok and parsed:
        valid, verr = validate_plan_pddl("blocksworld", BW_INIT, BW_GOAL, parsed)
        print(f"  valid={valid}  err={verr}")

if __name__ == "__main__":
    main()
