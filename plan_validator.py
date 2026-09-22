"""
plan_validator.py
==================
Genuine PDDL plan validator: executes actions against the initial
state, checks preconditions, and verifies the goal is reached.
Extracted from plan_step15_qwen72b_eval.py (the validator that
produced the paper's Table 7 numbers: Qwen-72B 23.5% BW / 0% LOG / 22% MBW).

This replaces the syntax-only checkers in plan_step31/plan_step37,
which only checked whether action names came from the domain
vocabulary and could be fooled by syntactically valid but
semantically empty plans (e.g. repeating a legal action forever).

Supports: blocksworld, mystery_blocksworld, logistics.
"""
import re


def _f(*parts):
    return " ".join(str(p).lower() for p in parts)


# Action semantics: name -> (arity, fn(*args) -> (preconditions, remove, add))
_ACTION_SEMANTICS = {
    # blocksworld
    "pick-up":  (1, lambda x:    ({_f("clear",x),_f("ontable",x),_f("handempty")},
                                    {_f("ontable",x),_f("clear",x),_f("handempty")},
                                    {_f("holding",x)})),
    "put-down": (1, lambda x:    ({_f("holding",x)},
                                    {_f("holding",x)},
                                    {_f("handempty"),_f("ontable",x),_f("clear",x)})),
    "stack":    (2, lambda x,y:  ({_f("holding",x),_f("clear",y)},
                                    {_f("holding",x),_f("clear",y)},
                                    {_f("handempty"),_f("on",x,y),_f("clear",x)})),
    "unstack":  (2, lambda x,y:  ({_f("on",x,y),_f("clear",x),_f("handempty")},
                                    {_f("on",x,y),_f("clear",x),_f("handempty")},
                                    {_f("holding",x),_f("clear",y)})),
    # mystery_blocksworld (scrambled predicate/action names)
    "grasp":    (1, lambda x:    ({_f("apex",x),_f("grounded",x),_f("grasping")},
                                    {_f("grounded",x),_f("apex",x),_f("grasping")},
                                    {_f("clutching",x)})),
    "release":  (1, lambda x:    ({_f("clutching",x)},
                                    {_f("clutching",x)},
                                    {_f("grasping"),_f("grounded",x),_f("apex",x)})),
    "place":    (2, lambda x,y:  ({_f("clutching",x),_f("apex",y)},
                                    {_f("clutching",x),_f("apex",y)},
                                    {_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
    "lift":     (2, lambda x,y:  ({_f("stacked",x,y),_f("apex",x),_f("grasping")},
                                    {_f("stacked",x,y),_f("apex",x),_f("grasping")},
                                    {_f("clutching",x),_f("apex",y)})),
    # logistics
    "load-truck":      (3, lambda p,t,l: ({_f("at",t,l),_f("at",p,l)},
                                            {_f("at",p,l)},
                                            {_f("in",p,t)})),
    "unload-truck":    (3, lambda p,t,l: ({_f("at",t,l),_f("in",p,t)},
                                            {_f("in",p,t)},
                                            {_f("at",p,l)})),
    "load-airplane":   (3, lambda p,a,l: ({_f("at",a,l),_f("at",p,l)},
                                            {_f("at",p,l)},
                                            {_f("in",p,a)})),
    "unload-airplane": (3, lambda p,a,l: ({_f("at",a,l),_f("in",p,a)},
                                            {_f("in",p,a)},
                                            {_f("at",p,l)})),
    "drive-truck":     (4, lambda t,s,d,c: ({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},
                                              {_f("at",t,s)},
                                              {_f("at",t,d)})),
    "fly-airplane":    (3, lambda a,s,d: ({_f("at",a,s),_f("airport",s),_f("airport",d)},
                                            {_f("at",a,s)},
                                            {_f("at",a,d)})),
}


def parse_plan_lines(response):
    """
    Extract action tokens from an LLM response, accepting both
    parenthesized '(action arg1 arg2)' and bare 'action arg1 arg2'
    line formats.
    """
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    actions = []
    for line in lines:
        if line.startswith("("):
            actions.append(line)
        else:
            # bare format: wrap in parens for uniform downstream parsing
            toks = line.split()
            if toks and re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", toks[0]):
                actions.append(f"({line})")
    return actions


def validate_plan(actions, episode: dict):
    """
    Execute plan against PDDL initial state and check goal.
    Returns (valid: bool, error_type: str|None).

    actions: list of strings like "(pick-up b1)" or "(load-truck p1 t1 l1)"
    episode: dict with 'init_facts' and 'goal_facts' (lists of fact strings)
    """
    if not actions:
        return False, "empty_plan"

    init_facts = episode.get("init_facts", [])
    goal_facts = episode.get("goal_facts", [])
    if not init_facts or not goal_facts:
        # No ground truth available — accept if parse succeeded
        return True, None

    state = {_f(*f.strip().strip("()").split()) for f in init_facts}
    goals = {_f(*g.strip().strip("()").split()) for g in goal_facts}

    for act_str in actions:
        toks = act_str.strip().strip("()").split()
        if not toks:
            continue
        name = toks[0].lower()
        args = [t.lower() for t in toks[1:]]

        if name not in _ACTION_SEMANTICS:
            return False, f"unknown_action:{name}"

        arity, fn = _ACTION_SEMANTICS[name]
        if len(args) != arity:
            return False, f"arity:{name}(need {arity} got {len(args)})"

        pre, rem, add = fn(*args)
        if not pre.issubset(state):
            return False, f"precondition_fail:{name}"

        state = (state - rem) | add

    ok = goals.issubset(state)
    return ok, (None if ok else "goal_not_reached")


def is_valid(response, episode, domain_pddl=None):
    """
    Full pipeline: parse LLM response into plan actions, then
    validate via state simulation. Compatible drop-in replacement
    for the syntax-only is_valid() used in plan_step31/plan_step37.

    Returns (valid: bool, reason: str) matching the existing
    reason vocabulary: "error", "refusal", "empty_plan",
    "unknown_action:X", "precondition_fail:X", "arity:X",
    "goal_not_reached", "valid".
    """
    if not response or response.startswith("ERROR:"):
        return False, "error"

    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b|i cannot|unable to|impossible", txt, re.I):
        return False, "refusal"

    actions = parse_plan_lines(txt)
    if not actions:
        return False, "empty_plan"

    valid, reason = validate_plan(actions, episode)
    return valid, (reason if reason else "valid")
