"""
revalidate.py
=============
Revalidates existing Qwen eval results using fixed parser.
No new Qwen calls — just re-parses existing raw_response fields.

Usage:
  python revalidate.py                          # both files
  python revalidate.py --file qwen72b_eval_instances.jsonl
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def _f(*p): return " ".join(str(x).lower() for x in p)

ACTION_KWS = {
    'pick-up','put-down','stack','unstack','pickup','putdown','put_down',
    'load-truck','unload-truck','load-airplane','unload-airplane',
    'drive-truck','fly-airplane','fly','fly-plane','load','unload',
    'grasp','release','place','lift',
    'move','move-truck','load-pkg','unload-pkg','board-pkg','unboard-pkg',
}

ALIASES_COMMON = {
    "pickup":"pick-up","putdown":"put-down","put_down":"put-down",
    "fly":"fly-airplane","fly-plane":"fly-airplane",
    "move-truck":"drive-truck",
    "load-pkg":"load-truck","unload-pkg":"unload-truck",
    "board-pkg":"load-airplane","unboard-pkg":"unload-airplane",
    "load":"load-truck","unload":"unload-truck",
}
ALIASES_MBW = {
    "pick-up":"grasp","put-down":"release","stack":"place","unstack":"lift",
    "pickup":"grasp","putdown":"release",
}

def parse_plan(response, domain=""):
    if not response or response.startswith("ERROR:"): return []
    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if re.search(r"\bNO[_\s-]PLAN\b", txt, re.I): return []

    # Strategy 1: after </think>
    parts = re.split(r"</think>", response, flags=re.DOTALL)
    if len(parts) > 1:
        lines = [l.strip() for l in parts[-1].split("\n") if l.strip().startswith("(")]
        if lines: return lines

    # Strategy 2: parenthesised lines
    lines = [l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines

    # Strategy 3: any parenthesised expressions
    found = re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", response, re.I)
    if found: return found

    # Strategy 4: keyword lines WITHOUT parens
    result = []
    for line in txt.split("\n"):
        line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        toks = line.split()
        if toks and toks[0].lower() in ACTION_KWS:
            result.append("(" + line + ")")
    return result


def apply_aliases(actions, domain):
    aliases = dict(ALIASES_COMMON)
    if "mystery" in domain.lower():
        aliases.update(ALIASES_MBW)
    result = []
    for act in actions:
        toks = act.strip().strip("()").split()
        if not toks: continue
        name = aliases.get(toks[0].lower(), toks[0].lower())
        result.append("(" + " ".join([name] + toks[1:]) + ")")
    return result


def validate(record, actions):
    if not actions: return False, "empty_plan"
    init  = record.get("init_facts", [])
    goal  = record.get("goal_facts", [])
    if not init or not goal: return True, None
    state = {_f(*f.strip().strip("()").split()) for f in init}
    goals = {_f(*g.strip().strip("()").split()) for g in goal}
    H = {
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
    for act in actions:
        toks = act.strip().strip("()").split()
        if not toks: continue
        name=toks[0].lower(); args=[t.lower() for t in toks[1:]]
        if name not in H: return False, f"unknown:{name}"
        ar,fn=H[name]
        if len(args)!=ar: return False, f"arity:{name}"
        pre,rem,add=fn(*args)
        if not pre.issubset(state): return False, f"precond:{name}"
        state=(state-rem)|add
    ok=goals.issubset(state)
    return ok,(None if ok else "goal_not_reached")


def revalidate_file(path, records_path=None):
    if not path.exists():
        print(f"  Not found: {path}")
        return

    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"\n{path.name}: {len(rows)} rows")

    # Load ground-truth records if available (for init/goal facts)
    records_by_id = {}
    if records_path and records_path.exists():
        recs = json.loads(records_path.read_text())
        for r in recs:
            iid = int(r.get('instance_id', -1))
            dom = r.get('task_type', r.get('domain', ''))
            records_by_id[(dom, iid)] = r
            records_by_id[iid] = r  # fallback for 200-instance dataset
        print(f"  Loaded {len(recs)} ground-truth records")

    updated = []
    by_dom  = {}

    for row in rows:
        dom  = row['domain']
        iid  = int(row['instance_id'])
        resp = row.get('raw_response', '')

        # Re-parse with fixed parser
        actions = parse_plan(resp, dom)
        actions = apply_aliases(actions, dom)

        # Get ground-truth record for validation
        rec = records_by_id.get((dom, iid), records_by_id.get(iid, row))
        valid, err = validate(rec, actions)

        row['valid_plan']  = valid
        row['error_type']  = err
        row['n_actions']   = len(actions)
        updated.append(row)
        by_dom.setdefault(dom, []).append(row)

    # Write back
    out = path.parent / (path.stem + '_revalidated.jsonl')
    with open(out, 'w') as f:
        for row in updated:
            f.write(json.dumps(row) + '\n')

    print(f"  Results after revalidation:")
    for dom in ['blocksworld','logistics','mystery_blocksworld']:
        rlist = by_dom.get(dom, [])
        if not rlist: continue
        valid = sum(r['valid_plan'] for r in rlist)
        errs  = {}
        for r in rlist:
            e = r.get('error_type') or 'valid'
            errs[e] = errs.get(e,0)+1
        top = sorted(errs.items(), key=lambda x:-x[1])[:3]
        print(f"  {dom}: {valid}/{len(rlist)} = {valid/len(rlist):.1%}  {top}")
    print(f"  Saved → {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='both')
    args = p.parse_args()

    RESULTS     = ROOT / 'results_planning'
    RESULTS_IPC = ROOT / 'results_planning_ipc'
    DATA        = ROOT / 'data' / 'planning'
    DATA_IPC    = ROOT / 'data' / 'planning_ipc'

    if args.file == 'both' or 'qwen72b' in args.file:
        # Try to load original records for init/goal facts
        recs_path = DATA / 'episodes.json'
        revalidate_file(RESULTS / 'qwen72b_eval_instances.jsonl', recs_path)

    if args.file == 'both' or 'ipc' in args.file:
        recs_path = DATA_IPC / 'records_ipc_labeled.json'
        revalidate_file(RESULTS_IPC / 'qwen_ipc_eval.jsonl', recs_path)


if __name__ == '__main__':
    main()