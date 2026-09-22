"""
plan_step1_data_prep_real.py
============================
Builds X_surf, X_fm, y_* arrays from real ScienceWorld trajectories
collected by plan_step1_collect_trajectories.py  (or loaded from
the ETO HuggingFace dataset as an alternative — see --source eto).

FM embedding input = goal_text + initial_observation_text
  → genuinely varies within task type (different objects, rooms, states)
  → FM knows about object affordances, causal structure, goal semantics
  → surface features (lexical stats) are orthogonal

This is the correct analog of the protein setup:
  protein:  sequence → ESM2 embedding, physicochemical → surface features
  planning: goal+obs → mpnet embedding, lexical stats → surface features

USAGE:
  python plan_step1_data_prep_real.py               # from collected trajectories
  python plan_step1_data_prep_real.py --source eto  # from HuggingFace ETO dataset
  python plan_step1_data_prep_real.py --force       # recompute embeddings
"""

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/planning")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEST_TASKS = [
    "grow-fruit", "measure-melting-point-unknown-substance",
    "test-conductivity", "find-animal",
]
VAL_TASKS = [
    "the-same-2", "chemistry-mix", "chemistry-mix-paint-tertiary-color",
]

# ── ETO dataset loader ────────────────────────────────────────────────────────

def load_eto_dataset():
    import json as _json
    import re as _re
    from huggingface_hub import hf_hub_download

    print("Downloading sciworld_sft.json from agent-eto/eto-sft-trajectory...")
    local_path = hf_hub_download(
        repo_id="agent-eto/eto-sft-trajectory",
        filename="data/sciworld_sft.json",
        repo_type="dataset",
    )

    with open(local_path) as f:
        raw = _json.load(f)
    print(f"  Raw records: {len(raw)}")

    TASK_KEYWORD_MAP = [
        ("determine if",                          "test-conductivity"),
        ("electrically conduct",                  "test-conductivity-of-unknown-substances"),
        ("measure the melting point of the unknown", "measure-melting-point-unknown-substance"),
        ("measure the melting point",             "measure-melting-point-known-substance"),
        ("melting point",                         "measure-melting-point"),
        ("boil",                                  "boil"),
        ("melt",                                  "melt"),
        ("freeze",                                "freeze"),
        ("change the state of matter",            "change-the-state-of-matter-of"),
        ("grow a fruit",                          "grow-fruit"),
        ("grow a",                                "grow-plant"),
        ("find a(n) animal",                      "find-animal"),
        ("find a(n) non-living",                  "find-non-living-thing"),
        ("find a(n) living",                      "find-living-thing"),
        ("find a(n) plant",                       "find-plant"),
        ("find a(n)",                             "find-living-thing"),
        ("tertiary",                              "chemistry-mix-paint-tertiary-color"),
        ("mix the paint",                         "chemistry-mix-paint-secondary-color"),
        ("mix",                                   "chemistry-mix"),
        ("use the thermometer",                   "use-thermometer"),
        ("thermometer",                           "use-thermometer"),
        ("renewable",                             "power-component-renewable-vs-nonrenewable-energy"),
        ("power the",                             "power-component"),
        ("lifespan",                              "lifespan-longest"),
        ("shortest lifespan",                     "lifespan-shortest"),
        ("life stage",                            "identify-life-stages-1"),
        ("life cycle",                            "identify-life-stages-2"),
        ("inclined plane",                        "inclined-plane-determine-angle"),
        ("friction",                              "inclined-plane-friction-named-surface"),
        ("the same type",                         "the-same-2"),
        ("the same",                              "the-same-1"),
    ]

    def extract_task_type(goal_text):
        gl = goal_text.lower()
        for keyword, task in TASK_KEYWORD_MAP:
            if keyword in gl:
                return task
        return "unknown"

    def infer_success(conversations, goal_text):
        """
        Task-aware success inference.
        For conductivity tasks: success = agent moved item to blue box.
        For find tasks: success = agent focused on correct item.
        For melt/boil/freeze: success = last observation mentions state change.
        For thermometer: success = agent read the thermometer.
        Default: trajectory length heuristic (longer = more likely successful).
        """
        goal_lower = goal_text.lower()

        # Collect all turn texts
        all_human = [t.get("value","") for t in conversations if t.get("from")=="human"]
        all_gpt   = [t.get("value","") for t in conversations if t.get("from")=="gpt"]
        last_obs  = all_human[-1].lower() if all_human else ""
        last_acts = " ".join(all_gpt[-3:]).lower() if all_gpt else ""
        full_text = " ".join(all_human + all_gpt).lower()

        # Conductivity: placed in blue box = conductive = task done correctly
        if "electrically conductive" in goal_lower or "determine if" in goal_lower:
            if "move" in last_acts and "blue box" in last_acts:
                return 1
            if "move" in last_acts and "orange box" in last_acts:
                return 1  # either placement = task completed
            if "blue box" in last_obs or "orange box" in last_obs:
                # check if item is now IN the box
                if "containing" in last_obs and (
                    "blue box" in last_obs or "orange box" in last_obs
                ):
                    return 1
            return 0

        # Find tasks: success = focus action on correct item in last turns
        if "find a(n)" in goal_lower:
            if "focus on" in last_acts:
                return 1
            if "move" in last_acts and ("red box" in last_acts or "box" in last_acts):
                return 1
            return 0

        # Melt/boil/freeze: last observation mentions state change
        if "melt" in goal_lower or "boil" in goal_lower or "freeze" in goal_lower:
            state_words = ["melted","boiling","frozen","liquid","gas","steam",
                           "is melting","has melted","is boiling","is frozen"]
            if any(w in last_obs for w in state_words):
                return 1
            # Also check if agent reported task done
            if "task complete" in last_acts or "done" in last_acts:
                return 1
            return 0

        # Thermometer: success = agent read a temperature
        if "thermometer" in goal_lower or "temperature" in goal_lower:
            if "read thermometer" in last_acts or "temperature" in last_obs:
                return 1
            # Check if a number appears in last observation (temp reading)
            if _re.search(r'\d+\s*(degree|°|celsius|c\b)', last_obs):
                return 1
            return 0

        # Lifespan: agent identified correct organism
        if "lifespan" in goal_lower:
            if "focus on" in last_acts or "move" in last_acts:
                return 1
            return 0

        # Grow tasks: plant appeared or grew
        if "grow" in goal_lower:
            if "fruit" in last_obs or "flower" in last_obs or "grew" in last_obs:
                return 1
            return 0

        # Default: if agent took at least 5 actions and last action is not
        # an exploration action, assume partial success
        n_gpt = len(all_gpt)
        if n_gpt >= 5 and last_acts and "look around" not in last_acts:
            return 1
        return 0

    def get_first_real_observation(conversations):
        """
        Extract the first meaningful observation (turn[4] or later).
        Skip the generic env description (turn[0]) and 'OK' turns.
        """
        obs_turns = []
        for turn in conversations:
            if turn.get("from") != "human":
                continue
            val = turn.get("value", "")
            # Skip the generic environment description
            if "there are several rooms:" in val.lower():
                continue
            # Skip very short turns
            if len(val) < 30:
                continue
            # This is a real observation or task description
            obs_turns.append(val.strip())
            if len(obs_turns) >= 2:
                break
        return " [OBS] ".join(obs_turns)

    records = []
    instance_id = 0
    skipped = 0

    for row in raw:
        convs = row.get("conversations", [])
        if len(convs) < 3:
            skipped += 1
            continue

        # Find goal text (first human turn containing "your task")
        goal_text = ""
        for turn in convs:
            val = turn.get("value", "")
            if turn.get("from") == "human" and "your task" in val.lower():
                goal_text = val.strip()
                break

        if not goal_text:
            skipped += 1
            continue

        task_type = extract_task_type(goal_text)

        # instance_text = goal + first real observation
        # (NOT the generic room list — that's identical across all records)
        first_obs = get_first_real_observation(convs)

        # instance_text is what gets FM-embedded
        # goal encodes WHAT to do; first_obs encodes WHERE and WITH WHAT
        # Together they vary genuinely within each task type
        instance_text = f"{goal_text} [OBS] {first_obs}".strip()

        n_steps = sum(1 for t in convs if t.get("from") == "gpt")
        success = infer_success(convs, goal_text)

        records.append({
            "instance_id":      instance_id,
            "task_type":        task_type,
            "task_variant":     str(row.get("id", instance_id)),
            "goal_text":        goal_text,
            "observation_text": first_obs[:400],
            "instance_text":    instance_text,
            "n_steps":          max(1, n_steps),
            "final_score":      float(success),
            "success":          success,
        })
        instance_id += 1

    print(f"  Parsed {len(records)} records  (skipped {skipped})")

    from collections import Counter
    task_counts = Counter(r["task_type"] for r in records)
    print(f"  Task types found ({len(task_counts)}):")
    for task, cnt in task_counts.most_common():
        sr = sum(r["success"] for r in records
                 if r["task_type"] == task) / max(cnt, 1)
        print(f"    {task:<55} n={cnt:4d}  SR={sr:.2f}")

    print(f"  Overall success rate: "
          f"{sum(r['success'] for r in records)/max(len(records),1):.3f}")

    return records

# ── Surface features — 21-dim density/ratio only ─────────────────────────────
# These are computed from instance_text (goal + initial obs).
# They capture lexical surface properties only.
# FM embeddings will capture object affordances, goal semantics,
# causal structure — information that surface features cannot.

SCIENCE_OBJECTS = [
    "thermometer", "beaker", "pot", "container", "battery", "wire", "bulb",
    "soil", "seed", "water", "heat", "ice", "temperature", "friction",
    "circuit", "conductor", "insulator", "melting", "boiling", "freezing",
    "solid", "liquid", "gas", "organism", "plant", "animal", "lifespan",
    "paint", "color", "substance", "reaction", "candle", "metal", "wood",
    "rubber", "glass", "knife", "stove", "fridge", "table", "box", "jar",
]

ACTION_VERBS = [
    "find", "locate", "place", "heat", "cool", "measure", "record", "test",
    "mix", "combine", "connect", "identify", "compare", "verify", "observe",
    "calculate", "determine", "pick", "put", "move", "go", "look", "open",
    "close", "pour", "use", "take", "drop", "examine", "read",
]


def compute_surface_features(texts: list) -> np.ndarray:
    """21-dim surface features from instance text."""
    N = len(texts)
    feats = np.zeros((N, 21), dtype=np.float32)

    for i, text in enumerate(texts):
        words = text.lower().split()
        sents = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        vocab = set(words)
        nw    = max(len(words), 1)

        feats[i,  0] = len(vocab) / nw
        oc = sum(1 for o in SCIENCE_OBJECTS if o in text.lower())
        feats[i,  1] = oc
        feats[i,  2] = oc / nw
        vc = sum(1 for v in ACTION_VERBS if v in words)
        feats[i,  3] = vc
        feats[i,  4] = vc / nw
        seq = ["then","next","first","second","finally","after",
               "before","until","once","while","subsequently"]
        feats[i,  5] = sum(1 for w in words if w in seq)
        feats[i,  6] = text.count(",") / nw
        feats[i,  7] = text.count(".") / nw
        feats[i,  8] = sum(1 for w in words if len(w) > 9)
        feats[i,  9] = sum(1 for w in words if len(w) > 9) / nw
        neg = ["not","no","never","without","except","unless","can't","cannot"]
        feats[i, 10] = sum(1 for w in words if w in neg) / nw
        qnt = ["each","every","all","multiple","several","three","five","two"]
        feats[i, 11] = sum(1 for w in words if w in qnt) / nw
        counts = Counter(words); total = sum(counts.values())
        probs  = np.array([v / total for v in counts.values()])
        feats[i, 12] = float(-np.sum(probs * np.log(probs + 1e-10)))
        feats[i, 13] = len(re.findall(r'\b\d+\b', text)) / nw
        feats[i, 14] = len(re.findall(r'\b\d+\.\d+\b', text)) / nw
        feats[i, 15] = int("your task" in text.lower())
        feats[i, 16] = int("you need" in text.lower())
        feats[i, 17] = int("you must" in text.lower())
        feats[i, 18] = sum(1 for p in ["first","then","next","finally"]
                           if p in text.lower())
        feats[i, 19] = int(len(sents) >= 3)
        feats[i, 20] = len(text) / 500.0   # normalized raw length (weak proxy, not level-deterministic)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def compute_fm_embeddings(texts, model_name="all-mpnet-base-v2", batch_size=64):
    from sentence_transformers import SentenceTransformer
    print(f"    Loading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"    Encoding {len(texts)} texts...")
    emb = model.encode(texts, batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["collected", "eto"], default="collected",
                        help="'collected' = use plan_step1_collect_trajectories.py output; "
                             "'eto' = load from HuggingFace agent-eto/eto-sft-trajectory")
    parser.add_argument("--fm_model", default="all-mpnet-base-v2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # ── Load records ──────────────────────────────────────────────────────────
    ep_path = DATA_DIR / "episodes.json"
    if ep_path.exists() and not args.force:
        print(f"Loading cached episodes from {ep_path}")
        records = json.loads(ep_path.read_text())
    elif args.source == "eto":
        records = load_eto_dataset()
        KNOWN_TASKS = {
            "boil", "melt", "freeze", "change-the-state-of-matter-of",
            "measure-melting-point-known-substance", "measure-melting-point-unknown-substance",
            "grow-plant", "grow-fruit",
            "find-living-thing", "find-non-living-thing", "find-plant", "find-animal",
            "chemistry-mix", "chemistry-mix-paint-secondary-color", "chemistry-mix-paint-tertiary-color",
            "use-thermometer", "measure-melting-point",
            "power-component", "power-component-renewable-vs-nonrenewable-energy",
            "test-conductivity", "test-conductivity-of-unknown-substances",
            "lifespan-longest", "lifespan-shortest",
            "identify-life-stages-1", "identify-life-stages-2",
            "inclined-plane-determine-angle", "inclined-plane-friction-named-surface",
            "inclined-plane-friction-unnamed-surface",
            "the-same-1", "the-same-2",
        }
        
        def normalize_task_type(tt):
            # Direct match
            if tt in KNOWN_TASKS:
                return tt
            # Partial match — pick longest known task that is a substring
            candidates = [k for k in KNOWN_TASKS if k in tt or tt in k]
            if candidates:
                return max(candidates, key=len)
            return tt
        
        for r in records:
            r["task_type"] = normalize_task_type(r["task_type"])
        ep_path.write_text(json.dumps(records, indent=2))
        print(f"Saved {len(records)} ETO records → {ep_path}")
    else:
        raw_path = DATA_DIR / "trajectories_raw.json"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"No trajectories found at {raw_path}. "
                f"Run plan_step1_collect_trajectories.py first, "
                f"or use --source eto to load from HuggingFace."
            )
        records = json.loads(raw_path.read_text())
        ep_path.write_text(json.dumps(records, indent=2))
        print(f"Loaded {len(records)} real trajectories from {raw_path}")

    print(f"\nDataset summary:")
    print(f"  Total:       {len(records)}")
    print(f"  Success rate:{np.mean([r['success'] for r in records]):.3f}")
    print(f"  Task types:  {len(set(r['task_type'] for r in records))}")

    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_type"]].append(r)
    print(f"\nPer-task breakdown (first 5):")
    for tt in list(by_task.keys())[:5]:
        n  = len(by_task[tt])
        sr = np.mean([r["success"] for r in by_task[tt]])
        print(f"  {tt:<55} n={n:3d}  SR={sr:.2f}")

    texts     = [r["instance_text"] for r in records]
    y_success = np.array([r["success"] for r in records], dtype=np.int64)
    y_nsteps  = np.array([r["n_steps"]  for r in records], dtype=np.float32)
    task_arr  = np.array([r["task_type"] for r in records])

    # ── Surface features ──────────────────────────────────────────────────────
    sf_path = DATA_DIR / "X_surf.npy"
    if sf_path.exists() and not args.force:
        print(f"\nLoading cached surface features")
        X_surf = np.load(sf_path)
    else:
        print(f"\nComputing surface features (21-dim)...")
        X_surf = compute_surface_features(texts)
        np.save(sf_path, X_surf)
        print(f"  Saved → {sf_path}  shape={X_surf.shape}")

    # Diagnostic: surface → task type accuracy (should be moderate, not 1.0)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        sc  = StandardScaler()
        le  = LabelEncoder()
        y_t = le.fit_transform(task_arr)
        # Only run if we have >1 task type AND >1 success class
        if len(np.unique(y_t)) > 1 and len(np.unique(y_success)) > 1:
            acc = LogisticRegression(max_iter=1000).fit(
                sc.fit_transform(X_surf), y_t
            ).score(sc.transform(X_surf), y_t)
            print(f"  Surface → task_type accuracy: {acc:.3f}  (want < 0.80)")
        else:
            print(f"  Surface → task_type accuracy: skipped "
                  f"(only {len(np.unique(y_t))} task types, "
                  f"{len(np.unique(y_success))} success classes)")
    except ImportError:
        pass

    # Key diagnostic: within-task surface variance (should be > 0 now)
    print(f"\n  Within-task surface feature std (first 3 tasks):")
    for tt in list(set(task_arr))[:3]:
        mask = (task_arr == tt)
        wstd = X_surf[mask].std(axis=0).mean()
        print(f"    {tt[:40]:<40} {wstd:.4f}  {'✓ varies' if wstd > 0.05 else '✗ flat'}")

    # ── FM embeddings ─────────────────────────────────────────────────────────
    fm_path = DATA_DIR / "X_fm.npy"
    if fm_path.exists() and not args.force:
        print(f"\nLoading cached FM embeddings")
        X_fm = np.load(fm_path)
    else:
        print(f"\nComputing sentence transformer embeddings (768-dim)...")
        X_fm = compute_fm_embeddings(texts, model_name=args.fm_model)
        np.save(fm_path, X_fm)
        print(f"  Saved → {fm_path}  shape={X_fm.shape}")

    # Key diagnostic: within-task cosine similarity (should be < 0.9 now)
    print(f"\n  Within-task FM cosine similarity (first 3 tasks):")
    for tt in list(set(task_arr))[:3]:
        mask = (task_arr == tt)
        Xe   = X_fm[mask]
        cos  = float((Xe @ Xe.T).mean())
        print(f"    {tt[:40]:<40} {cos:.4f}  {'✓ variation' if cos < 0.90 else '✗ too similar'}")

    # ── Save labels and splits ────────────────────────────────────────────────
    np.save(DATA_DIR / "y_success.npy", y_success)
    np.save(DATA_DIR / "y_nsteps.npy",  y_nsteps)
    np.save(DATA_DIR / "task_types.npy", task_arr)

    unique_tasks = list(np.unique(task_arr))
    test_tasks   = [t for t in TEST_TASKS if t in unique_tasks]
    val_tasks    = [t for t in VAL_TASKS  if t in unique_tasks]
    train_tasks  = [t for t in unique_tasks
                    if t not in test_tasks and t not in val_tasks]

    def get_idx(tasks):
        ts = set(tasks)
        return [i for i, t in enumerate(task_arr) if t in ts]

    splits = {
        "meta_train": {"tasks": train_tasks, "indices": get_idx(train_tasks)},
        "meta_val":   {"tasks": val_tasks,   "indices": get_idx(val_tasks)},
        "meta_test":  {"tasks": test_tasks,  "indices": get_idx(test_tasks)},
    }

    registry = {
        "n_total":      len(records),
        "n_task_types": len(unique_tasks),
        "surf_dim":     int(X_surf.shape[1]),
        "fm_dim":       int(X_fm.shape[1]),
        "fm_model":     args.fm_model,
        "data_source":  args.source,
        "splits": {k: {"tasks": v["tasks"], "n_instances": len(v["indices"])}
                   for k, v in splits.items()},
        "task_types":   unique_tasks,
        "labels":       ["success", "n_steps"],
    }

    (DATA_DIR / "registry.json").write_text(json.dumps(registry, indent=2))
    for name, data in splits.items():
        np.save(DATA_DIR / f"idx_{name}.npy", np.array(data["indices"]))

    print(f"\nMeta-splits:")
    for name, data in splits.items():
        print(f"  {name:<12}: {len(data['tasks'])} task types, "
              f"{len(data['indices'])} instances")

    print(f"\nAll data saved to {DATA_DIR}/")
    print(f"Next: python plan_step2_rplm_baseline.py")


if __name__ == "__main__":
    main()