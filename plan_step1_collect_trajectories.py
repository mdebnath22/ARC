"""
plan_step1_collect_trajectories.py
===================================
Collects real ScienceWorld trajectories by running a heuristic agent
across all 30 task types. Each trajectory becomes one instance:

  instance = {
      task_type, task_variant, env_step_limit,
      observation_text,   ← initial observation (what agent sees at t=0)
      goal_text,          ← task goal string from env
      full_obs_concat,    ← concatenation of all observations in trajectory
      n_steps,            ← steps taken
      success,            ← 1 if score >= 0.5 at end
      final_score,        ← raw env score [0,1]
  }

Instance text for FM embedding = goal_text + " " + observation_text
  (this is semantically rich and genuinely varies within task type
   because each variant has different objects, locations, states)

USAGE:
  python plan_step1_collect_trajectories.py --n_per_task 40
  python plan_step1_collect_trajectories.py --n_per_task 10 --quick
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/planning")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# All 30 ScienceWorld task names (matches env.getTaskNames())
SCIENCEWORLD_TASKS = [
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
]

TEST_TASKS = [
    "grow-fruit", "measure-melting-point-unknown-substance",
    "test-conductivity", "find-animal",
]
VAL_TASKS = [
    "the-same-2", "chemistry-mix", "chemistry-mix-paint-tertiary-color",
]


def run_random_agent(env, task_name, variant_idx, max_steps=40):
    """
    Runs a random agent on one episode.
    Returns a trajectory dict with all fields needed for GURU.
    """
    try:
        obs, info = env.reset()
    except Exception:
        obs = env.reset()
        info = {}

    goal_text = env.getGoalText() if hasattr(env, "getGoalText") else ""
    initial_obs = obs

    trajectory_obs = [obs]
    n_steps = 0
    final_score = 0.0

    for step in range(max_steps):
        # Get valid actions and pick one randomly
        valid_actions = env.getValidActions() if hasattr(env, "getValidActions") else []
        if not valid_actions:
            valid_actions = ["look around", "inventory"]

        action = random.choice(valid_actions)

        try:
            obs, score, done, info = env.step(action)
        except Exception:
            break

        trajectory_obs.append(obs)
        final_score = float(score)
        n_steps += 1

        if done:
            break

    full_obs_concat = " [SEP] ".join(trajectory_obs[:5])  # first 5 obs to keep length bounded
    instance_text   = f"{goal_text} {initial_obs}".strip()
    success         = int(final_score >= 0.5)

    return {
        "task_type":        task_name,
        "task_variant":     variant_idx,
        "goal_text":        goal_text,
        "observation_text": initial_obs,
        "full_obs_concat":  full_obs_concat,
        "instance_text":    instance_text,   # ← this is what gets embedded
        "n_steps":          n_steps,
        "final_score":      round(final_score, 4),
        "success":          success,
    }


def collect_trajectories(task_types, n_per_task, seed=42):
    try:
        import scienceworld
    except ImportError:
        raise ImportError("Run: pip install scienceworld  (also needs Java 1.8+)")

    random.seed(seed)
    np.random.seed(seed)

    env = scienceworld.ScienceWorldEnv("")
    all_task_names = env.getTaskNames()
    print(f"Available ScienceWorld tasks: {len(all_task_names)}")

    records = []
    instance_id = 0

    for task_name in task_types:
        # Map our task name to env task name (they may differ slightly)
        matched = [t for t in all_task_names if task_name in t or t in task_name]
        if not matched:
            print(f"  WARNING: task '{task_name}' not found in env, skipping")
            continue
        env_task_name = matched[0]

        # Load task and find out how many variants exist
        env.load(env_task_name, 0, "easy", generateGoldPath=False)
        num_variants = env.getNumVariations()
        variants_to_use = list(range(min(n_per_task, num_variants)))

        # If we need more than available variants, cycle through them
        if n_per_task > num_variants:
            extras = [i % num_variants for i in range(n_per_task - num_variants)]
            variants_to_use = variants_to_use + extras

        random.shuffle(variants_to_use)
        n_success = 0

        for var_idx in variants_to_use:
            env.load(env_task_name, var_idx, "easy", generateGoldPath=False)
            rec = run_random_agent(env, task_name, var_idx, max_steps=40)
            rec["instance_id"] = instance_id
            records.append(rec)
            instance_id += 1
            n_success += rec["success"]

        sr = n_success / len(variants_to_use)
        print(f"  {task_name:<55} n={len(variants_to_use)}  SR={sr:.2f}")

    env.shutdown()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_task", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.quick:
        task_types = TEST_TASKS + VAL_TASKS + ["boil", "melt", "freeze"]
        args.n_per_task = 15
    else:
        task_types = SCIENCEWORLD_TASKS

    out_path = DATA_DIR / "trajectories_raw.json"
    if out_path.exists() and not args.force:
        print(f"Found existing trajectories at {out_path}, loading...")
        records = json.loads(out_path.read_text())
    else:
        print(f"Collecting {len(task_types)} task types × {args.n_per_task} variants...")
        t0 = time.time()
        records = collect_trajectories(task_types, args.n_per_task)
        out_path.write_text(json.dumps(records, indent=2))
        print(f"Saved {len(records)} trajectories → {out_path}  ({time.time()-t0:.1f}s)")

    print(f"\nDataset summary:")
    print(f"  Total:       {len(records)}")
    print(f"  Success rate:{np.mean([r['success'] for r in records]):.3f}")
    print(f"  Task types:  {len(set(r['task_type'] for r in records))}")

    # Show within-task text diversity (proxy)
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_type"]].append(r["instance_text"])
    print(f"\n  Sample instance texts (first 2 per task):")
    for tt in list(by_task.keys())[:3]:
        for txt in by_task[tt][:2]:
            print(f"    [{tt}] {txt[:120]}...")

    print(f"\nNext: python plan_step1_data_prep_real.py")


if __name__ == "__main__":
    main()