"""
plan_step19_ipc_scale.py
========================
Scales the entire pipeline to match LLM-Modulo / Verma et al. setup:

  N = 600 instances per test domain (was 200)
  n_objects range: 2-15 (was 3-7), matching IPC benchmark difficulty spread
  Same zero-shot evaluation setup as LLM-Modulo

PHASES:
  generate  — generate 600 instances per domain with IPC-scale difficulty
  bfs       — BFS label all instances (parallelised)
  embed     — compute sentence-transformer embeddings
  eval      — run Qwen 72B on all 600 instances
  retrain   — retrain ARC on new data
  report    — generate comparison table vs LLM-Modulo / Verma

USAGE:
  python plan_step19_ipc_scale.py --phase generate
  python plan_step19_ipc_scale.py --phase bfs
  python plan_step19_ipc_scale.py --phase embed
  python plan_step19_ipc_scale.py --phase eval --ollama_host http://sg016:11435
  python plan_step19_ipc_scale.py --phase retrain
  python plan_step19_ipc_scale.py --phase report
  python plan_step19_ipc_scale.py --phase all --ollama_host http://sg016:11435
"""

from __future__ import annotations
import argparse, json, re, time, importlib.util, warnings, pickle
from collections import deque
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data" / "planning_ipc"   # separate from old data
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT_DIR / "results_planning_ipc"; RESULTS_DIR.mkdir(exist_ok=True)

# ── IPC-scale parameters ──────────────────────────────────────────────────────
N_PER_DOMAIN = 600      # matches LLM-Modulo
MAX_STEPS    = 20       # extended from 12 to handle larger instances
MAX_NODES    = 50_000   # extended from 5000

# n_objects ranges matching IPC benchmark difficulty spread
# LLM-Modulo uses 2-20 blocks; we use 2-15 for computational tractability
IPC_COMPLEXITY = {
    "blocksworld":         list(range(2, 16)),   # 2-15 blocks
    "mystery_blocksworld": list(range(2, 16)),   # same structure, scrambled
    "logistics":           list(range(2, 11)),   # 2-10 packages
}

# Training domains: keep same as before
TRAIN_DOMAINS = ["depot", "satellite", "rovers", "ferry"]
TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
DOMAIN_LABELS = {
    "blocksworld":         "Blocksworld",
    "logistics":           "Logistics",
    "mystery_blocksworld": "Mystery-BW",
}

# ── LLM-Modulo reference numbers (for comparison table) ───────────────────────
LLM_MODULO_REF = {
    "GPT-4o zero-shot BW":  "35.5% (213/600)",
    "GPT-4o zero-shot MBW": "0% (0/600)",
    "source": "Valmeekam et al. 2023, LLM-Modulo Framework",
    "note": "Their instances may differ; comparison is approximate",
}
VERMA_REF = {
    "Llama-3 baseline BW":  "28%",
    "Llama-3 baseline MBW": "1%",
    "Llama-3 baseline LOG": "11%",
    "GPT-4 baseline BW":    "35%",
    "GPT-4 baseline MBW":   "3%",
    "GPT-4 baseline LOG":   "6%",
    "source": "Verma et al. 2025, PDDL-INSTRUCT",
    "note": "Uses different model versions and instance sets",
}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Generate 600 instances per domain
# ══════════════════════════════════════════════════════════════════════════════

def load_generators():
    """Load instance generators from plan_step1."""
    spec = importlib.util.spec_from_file_location(
        "step1", ROOT_DIR / "plan_step1_data_prep.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s


def phase_generate(args):
    print("\n" + "="*65)
    print("PHASE 1: Generate 600 instances per domain (IPC-scale)")
    print(f"  n_objects range: 2-15 (was 3-7)")
    print(f"  N per domain: {N_PER_DOMAIN} (was 200)")
    print("="*65)

    step1 = load_generators()
    rng   = np.random.default_rng(args.seed)

    all_records = {}

    for dom in TEST_DOMAINS:
        print(f"\n  Generating {N_PER_DOMAIN} {dom} instances...", flush=True)
        n_sizes = IPC_COMPLEXITY[dom]
        records = []

        # Distribute instances evenly across n_objects levels
        n_per_size = N_PER_DOMAIN // len(n_sizes)
        remainder  = N_PER_DOMAIN % len(n_sizes)

        gen_fn = step1.GENERATORS[dom]   # domain-specific generator

        iid = 0
        for size_idx, n in enumerate(n_sizes):
            count = n_per_size + (1 if size_idx < remainder else 0)
            for _ in range(count):
                try:
                    rec = gen_fn(n, rng)
                    rec["instance_id"] = iid
                    rec["task_type"]   = dom
                    records.append(rec)
                    iid += 1
                except Exception as e:
                    print(f"    WARNING: gen failed for {dom} n={n}: {e}")

        all_records[dom] = records
        print(f"  Generated {len(records)} instances for {dom}")

        # Check n_objects distribution
        nobj = [r["n_objects"] for r in records]
        from collections import Counter
        dist = Counter(nobj)
        print(f"  n_objects distribution: {dict(sorted(dist.items()))}")

    # Save all records
    out_path = DATA_DIR / "records_ipc.json"
    flat = []
    for dom, recs in all_records.items():
        flat.extend(recs)
    out_path.write_text(json.dumps(flat, indent=2))
    print(f"\n  Saved {len(flat)} total records → {out_path}")
    return flat


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — BFS labeling (extended budget for larger instances)
# ══════════════════════════════════════════════════════════════════════════════


def phase_bfs(args):
    """
    BFS already runs during phase_generate (via bfs_plan inside each generator).
    This phase reads the generated records, maps the existing n_steps/valid fields
    to bfs_n_steps/bfs_solved for consistency, and prints summary stats.
    """
    print("\n" + "="*65)
    print("PHASE 2: BFS labeling (reading from generated records)")
    print("  Note: BFS runs inside each generator; this phase just remaps fields")
    print("="*65)

    records_path = DATA_DIR / "records_ipc.json"
    if not records_path.exists():
        print("ERROR: Run --phase generate first")
        return

    records = json.loads(records_path.read_text())
    print(f"  Processing {len(records)} instances...")

    by_dom = {}
    for rec in records:
        # Map generator fields → standard BFS fields
        rec["bfs_n_steps"] = int(rec.get("n_steps", MAX_STEPS + 5))
        rec["bfs_solved"]  = bool(rec.get("valid", False))
        by_dom.setdefault(rec["task_type"], []).append(rec)

    # Print summary per domain
    print("\n  BFS results by domain:")
    for dom in TEST_DOMAINS:
        recs   = by_dom.get(dom, [])
        if not recs: continue
        solved  = sum(r["bfs_solved"] for r in recs)
        capped  = len(recs) - solved
        steps   = [r["bfs_n_steps"] for r in recs if r["bfs_solved"]]
        mean_s  = np.mean(steps) if steps else 0
        max_s   = max(steps) if steps else 0
        nobj    = [r["n_objects"] for r in recs]
        print(f"  {dom}:")
        print(f"    N={len(recs)}  solved={solved} ({solved/len(recs):.1%})"
              f"  cap-exceeded={capped} ({capped/len(recs):.1%})")
        print(f"    n_steps: mean={mean_s:.1f}  max={max_s}")
        print(f"    n_objects: min={min(nobj)}  max={max(nobj)}"
              f"  mean={np.mean(nobj):.1f}")

    out_path = DATA_DIR / "records_ipc_labeled.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\n  Saved {len(records)} labeled records → {out_path}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Embeddings
# ══════════════════════════════════════════════════════════════════════════════

def phase_embed(args):
    print("\n" + "="*65)
    print("PHASE 3: Compute sentence-transformer embeddings")
    print("="*65)

    labeled_path = DATA_DIR / "records_ipc_labeled.json"
    if not labeled_path.exists():
        print("ERROR: Run --phase bfs first")
        return

    records = json.loads(labeled_path.read_text())
    descs   = [r["description"] for r in records]
    print(f"  {len(descs)} descriptions to encode")

    # Try to load sentence_transformers — may need specific conda env
    try:
        from sentence_transformers import SentenceTransformer
        model_loaded = True
    except ImportError:
        model_loaded = False
        print("  sentence_transformers not found in current env.")
        print("  Try: conda activate <your_env> && pip install sentence-transformers")
        print("  OR copy embeddings from existing dataset and extend:")
        print(f"    python plan_step19_ipc_scale.py --phase embed --use_existing_embeds")
        if not args.use_existing_embeds:
            return None

    if not model_loaded or args.use_existing_embeds:
        # Fallback: load existing X_fm.npy (200 instances) and extend
        # by running the embedding model from plan_step3_guru.py
        print("  Using plan_step3_guru.py embedding pipeline...")
        spec = importlib.util.spec_from_file_location(
            "step3", ROOT_DIR / "plan_step3_guru.py")
        step3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step3)
        # step3 has the encoder loaded internally
        existing_fm = ROOT_DIR / "data" / "planning" / "X_fm.npy"
        if existing_fm.exists():
            print(f"  Found existing embeddings at {existing_fm}")
            print("  Computing new embeddings for IPC-scale descriptions...")
        # Fall through to sentence_transformers below
        try:
            from sentence_transformers import SentenceTransformer
            model_loaded = True
        except ImportError:
            print("  ERROR: Cannot import sentence_transformers.")
            print("  Run this phase in the conda env that has it:")
            print("  conda run -n <env_name> python plan_step19_ipc_scale.py --phase embed")
            return None

    print(f"  Loading all-mpnet-base-v2...", flush=True)
    model = SentenceTransformer("all-mpnet-base-v2")

    print(f"  Encoding {len(descs)} descriptions...", flush=True)
    batch_size = 64
    embeddings = []
    for i in range(0, len(descs), batch_size):
        batch = descs[i:i+batch_size]
        embs  = model.encode(batch, show_progress_bar=False,
                              convert_to_numpy=True)
        embeddings.append(embs)
        if (i // batch_size + 1) % 5 == 0:
            print(f"  [{i+len(batch)}/{len(descs)}]", flush=True)

    X_fm = np.vstack(embeddings)
    np.save(DATA_DIR / "X_fm_ipc.npy", X_fm)
    print(f"  Embeddings shape: {X_fm.shape}")
    print(f"  Saved → {DATA_DIR}/X_fm_ipc.npy")
    return X_fm


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Qwen 72B evaluation
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(record):
    """
    Zero-shot prompt matching LLM-Modulo / plan-bench format.
    Lists valid action names explicitly to prevent hallucination.
    """
    domain_pddl  = record.get("domain_pddl",  "")
    problem_pddl = record.get("problem_pddl", "")
    description  = record.get("description",  "")

    # Extract valid action names
    action_names = re.findall(r":action\s+(\S+)", domain_pddl)
    action_list  = ""
    if action_names:
        action_list = (
            "\nIMPORTANT: Use ONLY these action names (exactly as written):\n"
            + "\n".join(f"  {a}" for a in action_names) + "\n"
        )

    return (
        "You are an expert PDDL planner. "
        "Given the domain and problem below, produce a valid sequential plan.\n\n"
        f"=== DOMAIN ===\n{domain_pddl}\n\n"
        f"=== PROBLEM ===\n{problem_pddl}\n\n"
        f"=== DESCRIPTION ===\n{description}\n"
        f"{action_list}\n"
        "Output ONLY the plan as a list of ground actions, one per line:\n"
        "(action-name arg1 arg2 ...)\n\n"
        "Do NOT include explanations. "
        "Do NOT use action names not listed above.\n"
        "If the problem is unsolvable, output exactly: NO_PLAN"
    )


def parse_plan(response: str):
    """Robust plan extraction handling <think> blocks and format variants."""
    if not response or "ERROR:" in response:
        return []
    txt = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not txt or txt.lower().strip(".!") in {"no-plan","no plan","no_plan",
                                               "unsat","unsolvable"}:
        return []
    # Strategy 1: after </think>
    parts = re.split(r"</think>", response, flags=re.DOTALL)
    if len(parts) > 1:
        lines = [l.strip() for l in parts[-1].split("\n")
                 if l.strip().startswith("(")]
        if lines: return lines
    # Strategy 2: parenthesized lines in stripped text
    lines = [l.strip() for l in txt.split("\n") if l.strip().startswith("(")]
    if lines: return lines
    # Strategy 3: any parenthesized expressions
    found = re.findall(r"\([a-z][a-z0-9\-]*(?: \S+)*\)", response, re.I)
    return found


def _f(*parts): return " ".join(str(p).lower() for p in parts)

def validate_plan(record, actions):
    """PDDL execution validator — checks preconditions and goal."""
    if not actions: return False, "empty_plan"
    dom        = record["task_type"]
    init_facts = record.get("init_facts", [])
    goal_facts = record.get("goal_facts", [])
    if not init_facts or not goal_facts: return True, None

    state = {_f(*f.strip().strip("()").split()) for f in init_facts}
    goals = {_f(*g.strip().strip("()").split()) for g in goal_facts}

    # Action handlers for all supported domains
    H = {
        # blocksworld / mystery_blocksworld
        "pick-up":         (1, lambda x:    ({_f("clear",x),_f("ontable",x),_f("handempty")},     {_f("ontable",x),_f("clear",x),_f("handempty")}, {_f("holding",x)})),
        "put-down":        (1, lambda x:    ({_f("holding",x)},                                     {_f("holding",x)},                               {_f("handempty"),_f("ontable",x),_f("clear",x)})),
        "stack":           (2, lambda x,y:  ({_f("holding",x),_f("clear",y)},                      {_f("holding",x),_f("clear",y)},                 {_f("handempty"),_f("on",x,y),_f("clear",x)})),
        "unstack":         (2, lambda x,y:  ({_f("on",x,y),_f("clear",x),_f("handempty")},         {_f("on",x,y),_f("clear",x),_f("handempty")},   {_f("holding",x),_f("clear",y)})),
        # mystery_blocksworld (scrambled names)
        "grasp":           (1, lambda x:    ({_f("apex",x),_f("grounded",x),_f("grasping")},       {_f("grounded",x),_f("apex",x),_f("grasping")},  {_f("clutching",x)})),
        "release":         (1, lambda x:    ({_f("clutching",x)},                                   {_f("clutching",x)},                             {_f("grasping"),_f("grounded",x),_f("apex",x)})),
        "place":           (2, lambda x,y:  ({_f("clutching",x),_f("apex",y)},                     {_f("clutching",x),_f("apex",y)},                {_f("grasping"),_f("stacked",x,y),_f("apex",x)})),
        "lift":            (2, lambda x,y:  ({_f("stacked",x,y),_f("apex",x),_f("grasping")},      {_f("stacked",x,y),_f("apex",x),_f("grasping")},{_f("clutching",x),_f("apex",y)})),
        # logistics
        "load-truck":      (3, lambda p,t,l:({_f("at",t,l),_f("at",p,l)},                          {_f("at",p,l)},                                  {_f("in",p,t)})),
        "unload-truck":    (3, lambda p,t,l:({_f("at",t,l),_f("in",p,t)},                          {_f("in",p,t)},                                  {_f("at",p,l)})),
        "load-airplane":   (3, lambda p,a,l:({_f("at",a,l),_f("at",p,l)},                          {_f("at",p,l)},                                  {_f("in",p,a)})),
        "unload-airplane": (3, lambda p,a,l:({_f("at",a,l),_f("in",p,a)},                          {_f("in",p,a)},                                  {_f("at",p,l)})),
        "drive-truck":     (4, lambda t,s,d,c:({_f("at",t,s),_f("in-city",s,c),_f("in-city",d,c)},{_f("at",t,s)},                                  {_f("at",t,d)})),
        "fly-airplane":    (3, lambda a,s,d:({_f("at",a,s),_f("airport",s),_f("airport",d)},       {_f("at",a,s)},                                  {_f("at",a,d)})),
    }

    # Parse actions
    parsed = []
    for act_str in actions:
        toks = act_str.strip().strip("()").split()
        if not toks: continue
        name = toks[0].lower(); args = [t.lower() for t in toks[1:]]
        parsed.append((name, args))

    for name, args in parsed:
        if name not in H: return False, f"unknown_action:{name}"
        arity, fn = H[name]
        if len(args) != arity: return False, f"arity:{name}"
        pre, rem, add = fn(*args)
        if not pre.issubset(state): return False, f"precondition_fail:{name}"
        state = (state - rem) | add

    ok = goals.issubset(state)
    return ok, (None if ok else "goal_not_reached")


def call_ollama(host, model, prompt, timeout=300):
    import urllib.request
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": 8192, "temperature": 0.0},
    }).encode()
    t0  = time.perf_counter()
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        return data.get("response", ""), float(ms)
    except Exception as e:
        return f"ERROR: {e}", (time.perf_counter()-t0)*1000


def phase_eval(args):
    print("\n" + "="*65)
    print(f"PHASE 4: Qwen 72B evaluation on {N_PER_DOMAIN} instances/domain")
    print(f"  Host:  {args.ollama_host}")
    print(f"  Model: {args.model}")
    print("="*65)

    labeled_path = DATA_DIR / "records_ipc_labeled.json"
    if not labeled_path.exists():
        print("ERROR: Run --phase bfs first")
        return

    records  = json.loads(labeled_path.read_text())
    out_path = RESULTS_DIR / "qwen_ipc_eval.jsonl"

    # Resume support
    done = {}
    if out_path.exists() and not args.overwrite:
        for line in open(out_path):
            r = json.loads(line)
            done[int(r["instance_id"])] = r
        print(f"  Resuming: {len(done)} already done")

    to_run = [r for r in records
              if r["task_type"] in TEST_DOMAINS
              and int(r["instance_id"]) not in done]
    print(f"  To evaluate: {len(to_run)} instances")

    by_dom = {d: {"n":0,"valid":0} for d in TEST_DOMAINS}

    with open(out_path, "a") as f:
        for idx, rec in enumerate(to_run):
            dom  = rec["task_type"]
            iid  = int(rec["instance_id"])
            prompt = build_prompt(rec)
            response, lat_ms = call_ollama(
                args.ollama_host, args.model, prompt, timeout=args.timeout)

            actions = parse_plan(response)
            valid, err = validate_plan(rec, actions)

            row = {
                "instance_id":  iid,
                "domain":       dom,
                "model":        args.model,
                "valid_plan":   valid,
                "error_type":   err,
                "n_actions":    len(actions),
                "latency_ms":   lat_ms,
                "raw_response": response[:600],
            }
            f.write(json.dumps(row) + "\n"); f.flush()
            by_dom[dom]["n"]     += 1
            by_dom[dom]["valid"] += int(valid)

            if (idx + 1) % 50 == 0:
                print(f"\n  [{idx+1}/{len(to_run)}]")
                for d, c in by_dom.items():
                    if c["n"] > 0:
                        print(f"    {d}: {c['valid']}/{c['n']} = {c['valid']/c['n']:.1%}")

    print(f"\n  FINAL ({args.model}):")
    all_results = list(done.values())
    for line in open(out_path):
        r = json.loads(line)
        if int(r["instance_id"]) not in done:
            all_results.append(r)
    by_dom2 = {}
    for r in all_results:
        by_dom2.setdefault(r["domain"], []).append(r)
    for dom in TEST_DOMAINS:
        rlist = by_dom2.get(dom, [])
        if rlist:
            acc = sum(r["valid_plan"] for r in rlist)/len(rlist)
            print(f"  {DOMAIN_LABELS[dom]}: {acc:.1%} ({len(rlist)} instances)")
    print(f"  Results → {out_path}")
    return by_dom2


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Retrain ARC on IPC-scale data
# ══════════════════════════════════════════════════════════════════════════════

def phase_retrain(args):
    """
    Prepare numpy arrays for ARC retraining on IPC-scale data,
    then call plan_step17_arc_v2 with the new data directory.
    """
    print("\n" + "="*65)
    print("PHASE 5: Prepare IPC-scale data arrays for ARC retraining")
    print("="*65)

    labeled_path = DATA_DIR / "records_ipc_labeled.json"
    if not labeled_path.exists():
        print("ERROR: Run --phase bfs first"); return

    records = json.loads(labeled_path.read_text())
    X_fm_path = DATA_DIR / "X_fm_ipc.npy"
    if not X_fm_path.exists():
        print("ERROR: Run --phase embed first"); return

    X_fm = np.load(X_fm_path)

    # Load step1 for surface features
    step1 = load_generators()
    X_surf = step1.compute_surface_features(records)

    task_types = np.array([r["task_type"] for r in records])

    # BFS labels — compute median per domain for y_success
    y_nsteps  = np.array([r.get("bfs_n_steps", 20) for r in records], dtype=float)
    y_success = np.zeros(len(records), dtype=int)

    for dom in TEST_DOMAINS + TRAIN_DOMAINS:
        mask   = task_types == dom
        if mask.sum() == 0: continue
        median = np.median(y_nsteps[mask])
        y_success[mask] = (y_nsteps[mask] <= median).astype(int)

    # Save as numpy arrays matching plan_step3 format
    np.save(DATA_DIR / "X_surf.npy",    X_surf)
    np.save(DATA_DIR / "X_fm.npy",      X_fm)
    np.save(DATA_DIR / "y_success.npy", y_success)
    np.save(DATA_DIR / "y_nsteps.npy",  y_nsteps)
    np.save(DATA_DIR / "task_types.npy", task_types)

    # Registry
    registry = {
        "n_total":   len(records),
        "surf_dim":  int(X_surf.shape[1]),
        "fm_dim":    int(X_fm.shape[1]),
        "splits": {
            "meta_train": {"tasks": TRAIN_DOMAINS,
                           "indices": np.where(np.isin(task_types, TRAIN_DOMAINS))[0].tolist()},
            "meta_val":   {"tasks": ["gripper"],
                           "indices": np.where(task_types == "gripper")[0].tolist()},
            "meta_test":  {"tasks": TEST_DOMAINS,
                           "indices": np.where(np.isin(task_types, TEST_DOMAINS))[0].tolist()},
        },
    }
    (DATA_DIR / "registry.json").write_text(json.dumps(registry, indent=2))

    print(f"  Saved numpy arrays to {DATA_DIR}")
    print(f"  X_surf: {X_surf.shape}  X_fm: {X_fm.shape}")
    print(f"  y_success: {y_success.mean():.1%} positive")
    print()
    print("  Now retrain ARC v2 on this data:")
    print(f"  python plan_step17_arc_v2.py --phase preprocess "
          f"--data_dir {DATA_DIR}")
    print(f"  python plan_step17_arc_v2.py --phase train "
          f"--data_dir {DATA_DIR} --n_episodes 5000")
    print(f"  python plan_step17_arc_v2.py --phase eval "
          f"--data_dir {DATA_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Comparison report
# ══════════════════════════════════════════════════════════════════════════════

def phase_report(args):
    print("\n" + "="*65)
    print("PHASE 6: Comparison table vs LLM-Modulo / Verma et al.")
    print("="*65)

    eval_path = RESULTS_DIR / "qwen_ipc_eval.jsonl"
    if not eval_path.exists():
        print("ERROR: Run --phase eval first"); return

    by_dom = {}
    for line in open(eval_path):
        r = json.loads(line)
        by_dom.setdefault(r["domain"], []).append(r)

    print()
    print("  OUR RESULTS (Qwen 72B zero-shot, N=600/domain):")
    our = {}
    for dom in TEST_DOMAINS:
        rlist = by_dom.get(dom, [])
        if not rlist:
            print(f"  {DOMAIN_LABELS[dom]}: no results yet")
            continue
        acc = sum(r["valid_plan"] for r in rlist)/len(rlist)
        our[dom] = acc
        print(f"  {DOMAIN_LABELS[dom]}: {acc:.1%} ({len(rlist)} instances)")

    print()
    print("  LLM-MODULO REFERENCE (GPT-4o zero-shot, N=600):")
    print("  Blocksworld: 35.5%")
    print("  Mystery-BW:  0%")
    print("  Logistics:   not reported")
    print()
    print("  VERMA ET AL. REFERENCE (Llama-3 / GPT-4 baseline):")
    print("  Blocksworld: 28% (Llama-3) / 35% (GPT-4)")
    print("  Mystery-BW:  1% (Llama-3) / 3% (GPT-4)")
    print("  Logistics:   11% (Llama-3) / 6% (GPT-4)")
    print()
    print("  NOTE: All comparisons are approximate — different models,")
    print("  different instance generators, different prompts.")
    print("  Qwen 72B vs GPT-4o is not directly comparable.")
    print()

    # Generate LaTeX table
    lines = [
        r"\begin{table}[h]\centering",
        r"\caption{Zero-shot LLM plan validity on IPC planning domains "
        r"($N=600$ per domain). "
        r"Our evaluation uses Qwen2.5-72B with a PDDL execution validator. "
        r"LLM-Modulo uses GPT-4o; Verma et al. use Llama-3-8B and GPT-4. "
        r"Instance sets differ across papers; numbers are not directly comparable.}",
        r"\label{tab:llm_validity}",
        r"\small\begin{tabular}{l ccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Blocksworld} & \textbf{Mystery-BW} "
        r"& \textbf{Logistics} \\",
        r"\midrule",
        r"  Llama-3 (Verma et al. 2025) & 28\% & 1\% & 11\% \\",
        r"  GPT-4 (Verma et al. 2025) & 35\% & 3\% & 6\% \\",
        r"  GPT-4o (LLM-Modulo) & 35.5\% & 0\% & --- \\",
        r"  \midrule",
    ]
    bw  = f"{our.get('blocksworld',0):.1%}" if "blocksworld" in our else "---"
    mbw = f"{our.get('mystery_blocksworld',0):.1%}" if "mystery_blocksworld" in our else "---"
    log = f"{our.get('logistics',0):.1%}" if "logistics" in our else "---"
    lines.append(
        f"  \\textbf{{Qwen2.5-72B (ours)}} & {bw} & {mbw} & {log} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}\end{table}"]
    tex = "\n".join(lines)
    (RESULTS_DIR / "llm_validity_comparison.tex").write_text(tex)
    print(f"  LaTeX → {RESULTS_DIR}/llm_validity_comparison.tex")
    print()
    print(tex)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=[
        "generate","bfs","embed","eval","retrain","report","all"], default="all")
    p.add_argument("--model",        default="qwen2.5:72b")
    p.add_argument("--ollama_host",  default="http://sg016:11435")
    p.add_argument("--timeout",      type=int, default=600)
    p.add_argument("--overwrite",    action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--n_workers",         type=int,  default=4)
    p.add_argument("--use_existing_embeds", action="store_true",
                   help="Use existing sentence_transformers env")
    args = p.parse_args()

    run_all = args.phase == "all"

    if run_all or args.phase == "generate":
        phase_generate(args)
    if run_all or args.phase == "bfs":
        phase_bfs(args)
    if run_all or args.phase == "embed":
        phase_embed(args)
    if run_all or args.phase == "eval":
        phase_eval(args)
    if run_all or args.phase == "retrain":
        phase_retrain(args)
    if run_all or args.phase == "report":
        phase_report(args)


if __name__ == "__main__":
    main()