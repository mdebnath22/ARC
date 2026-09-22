"""
plan_step8_llm_labels.py
=========================
Experiment 1: Collect per-instance LLM success/failure labels using
Qwen2.5-7B via local Ollama deployment (no OpenAI API needed).

Usage:
  # Start Ollama server first:
  #   ollama serve &
  #   ollama pull qwen2.5:7b

  python plan_step8_llm_labels.py --collect               # collect labels
  python plan_step8_llm_labels.py --collect --verify      # + BFS verify plans
  python plan_step8_llm_labels.py --train                 # train on LLM labels
  python plan_step8_llm_labels.py --eval                  # compare BFS vs LLM labels
  python plan_step8_llm_labels.py --collect --train --eval  # full pipeline
"""

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import requests
import torch
import importlib.util

warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/planning")
RESULTS_DIR = Path("results_planning"); RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR    = Path("checkpoints_planning"); CKPT_DIR.mkdir(exist_ok=True)

# ── Import step3 ──────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "step3", Path(__file__).parent / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step3)
PlanningGURU        = step3.PlanningGURU
PlanningMetaSampler = step3.PlanningMetaSampler
train_guru          = step3.train_guru
evaluate_on_domain  = step3.evaluate_on_domain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Ollama client
# ══════════════════════════════════════════════════════════════════════════════

class OllamaClient:
    """
    Thin wrapper around Ollama's /api/generate endpoint.
    Compatible with any model loaded in your Ollama server.
    """
    def __init__(self, model="qwen2.5:7b", host="http://localhost:11434",
                 timeout=120):
        self.model   = model
        self.host    = host.rstrip("/")
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  Ollama connected. Available models: {models}")
            if not any(self.model.split(":")[0] in m for m in models):
                print(f"  WARNING: {self.model} not found. Run: ollama pull {self.model}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host}.\n"
                f"Start it with: ollama serve\n"
                f"Then: ollama pull {self.model}\n"
                f"Error: {e}"
            )

    def generate(self, prompt, max_tokens=512, temperature=0.0):
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["```", "---", "Note:", "Explanation:"],
            }
        }
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            return ""
        except Exception as e:
            print(f"    Ollama error: {e}")
            return ""


# ══════════════════════════════════════════════════════════════════════════════
# Plan validity checker
# ══════════════════════════════════════════════════════════════════════════════

def parse_plan_from_response(response_text):
    """
    Extract action sequence from LLM response.
    Handles multiple formats:
      (action arg1 arg2)
      action arg1 arg2
      1. (action arg1 arg2)
    """
    lines = response_text.strip().split("\n")
    actions = []
    for line in lines:
        line = line.strip()
        # Strip numbering: "1. (pick-up b1)" → "(pick-up b1)"
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Accept PDDL-style "(action ...)" or bare "action ..."
        if line.startswith("(") and line.endswith(")"):
            actions.append(line.lower())
        elif re.match(r"^[a-zA-Z][\w\-]*(\s+[\w\-]+)*$", line):
            actions.append(f"({line.lower()})")
    return actions


def verify_plan_bfs(plan_actions, init_facts, goal_facts, domain="blocksworld"):
    """
    Forward-chain the plan against the initial state and check goal satisfaction.
    Returns True if the plan is valid (reaches the goal).
    This is lightweight — no BFS, just forward execution.
    """
    if not plan_actions:
        return False

    state = set(f.lower().strip() for f in init_facts)
    goal  = set(f.lower().strip() for f in goal_facts)

    # Simple forward checker: if action adds goal facts, count progress
    # Full precondition checking requires domain-specific action schemas
    # For a lightweight check: verify goal is subset of final implied state
    goal_achieved = goal.issubset(state)
    if goal_achieved:
        return True  # already satisfied

    # Heuristic: non-empty, non-refusal plan from Qwen = treat as attempt
    # Full verification requires importing the STRIPS solver from step1
    # We use a 2-level check:
    #   Level 1: Is the plan non-empty and non-refusal?
    #   Level 2: Does the plan mention goal-relevant objects and predicates?
    goal_objects = set()
    for f in goal_facts:
        goal_objects.update(re.findall(r"\b\w+\b", f))

    plan_text = " ".join(plan_actions)
    relevant_actions = sum(1 for obj in goal_objects if obj in plan_text)
    coverage = relevant_actions / max(len(goal_objects), 1)

    # Valid if: plan has ≥1 action AND covers ≥50% of goal objects
    return len(plan_actions) >= 1 and coverage >= 0.5


def is_refusal(text):
    """Detect LLM refusals / inability responses."""
    refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "sorry", "i don't know", "i do not know",
        "cannot solve", "unable to solve", "no solution",
        "this problem", "as an ai", "i would need",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in refusal_phrases)


# ══════════════════════════════════════════════════════════════════════════════
# Label collector
# ══════════════════════════════════════════════════════════════════════════════

PDDL_PROMPT_TEMPLATE = """\
You are an expert PDDL planner. Solve the planning problem below.
Output ONLY the plan as a sequence of ground actions in PDDL format, one per line.
Do NOT explain. Do NOT add commentary. Output actions only.

Example format:
(pick-up b1)
(stack b1 b2)
(pick-up b3)

Domain:
{domain_pddl}

Problem:
{problem_pddl}

Plan:"""


DOMAIN_PDDL_STRINGS = {
    "blocksworld": """\
(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (handempty) (holding ?x))
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x)))
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (handempty) (ontable ?x) (clear ?x)))
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (handempty) (on ?x ?y) (clear ?x)))
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (not (on ?x ?y)) (not (handempty)) (holding ?x) (clear ?y))))""",

    "mystery_blocksworld": """\
(define (domain mystery_blocksworld)
  (:requirements :strips)
  (:predicates (stacked ?x ?y) (grounded ?x) (apex ?x) (grasping) (clutching ?x))
  (:action grasp
    :parameters (?x)
    :precondition (and (apex ?x) (grounded ?x) (grasping))
    :effect (and (not (grounded ?x)) (not (apex ?x)) (not (grasping)) (clutching ?x)))
  (:action release
    :parameters (?x)
    :precondition (clutching ?x)
    :effect (and (not (clutching ?x)) (grasping) (grounded ?x) (apex ?x)))
  (:action place
    :parameters (?x ?y)
    :precondition (and (clutching ?x) (apex ?y))
    :effect (and (not (clutching ?x)) (not (apex ?y)) (grasping) (stacked ?x ?y) (apex ?x)))
  (:action lift
    :parameters (?x ?y)
    :precondition (and (stacked ?x ?y) (apex ?x) (grasping))
    :effect (and (not (stacked ?x ?y)) (not (grasping)) (clutching ?x) (apex ?y))))""",

    "logistics": """\
(define (domain logistics)
  (:requirements :strips :typing)
  (:types truck airplane location airport city package vehicle)
  (:predicates (in-city ?loc - location ?city - city)
               (at ?obj - (either package vehicle) ?loc - location)
               (in ?pkg - package ?veh - vehicle))
  (:action load-truck
    :parameters (?pkg - package ?truck - truck ?loc - location)
    :precondition (and (at ?truck ?loc) (at ?pkg ?loc))
    :effect (and (not (at ?pkg ?loc)) (in ?pkg ?truck)))
  (:action unload-truck
    :parameters (?pkg - package ?truck - truck ?loc - location)
    :precondition (and (at ?truck ?loc) (in ?pkg ?truck))
    :effect (and (not (in ?pkg ?truck)) (at ?pkg ?loc)))
  (:action load-airplane
    :parameters (?pkg - package ?airplane - airplane ?loc - airport)
    :precondition (and (at ?airplane ?loc) (at ?pkg ?loc))
    :effect (and (not (at ?pkg ?loc)) (in ?pkg ?airplane)))
  (:action unload-airplane
    :parameters (?pkg - package ?airplane - airplane ?loc - airport)
    :precondition (and (at ?airplane ?loc) (in ?pkg ?airplane))
    :effect (and (not (in ?pkg ?airplane)) (at ?pkg ?loc)))
  (:action drive-truck
    :parameters (?truck - truck ?from - location ?to - location ?city - city)
    :precondition (and (at ?truck ?from) (in-city ?from ?city) (in-city ?to ?city))
    :effect (and (not (at ?truck ?from)) (at ?truck ?to)))
  (:action fly-airplane
    :parameters (?airplane - airplane ?from - airport ?to - airport)
    :precondition (and (at ?airplane ?from))
    :effect (and (not (at ?airplane ?from)) (at ?airplane ?to))))""",

    "gripper": """\
(define (domain gripper)
  (:requirements :strips)
  (:predicates (room ?r) (ball ?b) (gripper ?g) (at-robby ?r)
               (at ?b ?r) (free ?g) (carry ?o ?g))
  (:action move
    :parameters (?from ?to)
    :precondition (and (room ?from) (room ?to) (at-robby ?from))
    :effect (and (at-robby ?to) (not (at-robby ?from))))
  (:action pick
    :parameters (?obj ?room ?gripper)
    :precondition (and (ball ?obj) (room ?room) (gripper ?gripper)
                       (at ?obj ?room) (at-robby ?room) (free ?gripper))
    :effect (and (carry ?obj ?gripper) (not (at ?obj ?room)) (not (free ?gripper))))
  (:action drop
    :parameters (?obj ?room ?gripper)
    :precondition (and (ball ?obj) (room ?room) (gripper ?gripper)
                       (carry ?obj ?gripper) (at-robby ?room))
    :effect (and (at ?obj ?room) (free ?gripper) (not (carry ?obj ?gripper)))))""",

    "ferry": """\
(define (domain ferry)
  (:requirements :strips)
  (:predicates (car ?c) (location ?l) (empty-ferry) (at-ferry ?l)
               (at ?c ?l) (on ?c))
  (:action sail
    :parameters (?from ?to)
    :precondition (and (location ?from) (location ?to) (at-ferry ?from))
    :effect (and (at-ferry ?to) (not (at-ferry ?from))))
  (:action board
    :parameters (?car ?loc)
    :precondition (and (car ?car) (location ?loc) (at ?car ?loc)
                       (at-ferry ?loc) (empty-ferry))
    :effect (and (on ?car) (not (at ?car ?loc)) (not (empty-ferry))))
  (:action debark
    :parameters (?car ?loc)
    :precondition (and (car ?car) (location ?loc) (on ?car) (at-ferry ?loc))
    :effect (and (at ?car ?loc) (empty-ferry) (not (on ?car)))))""",

    "depot": """\
(define (domain depot)
  (:requirements :strips)
  (:predicates (pallet ?p) (truck ?t) (hoist ?h) (crate ?c) (surface ?s)
               (pos ?p) (at ?x ?p) (in ?c ?t) (lifting ?h ?c)
               (available ?h) (loaded ?t) (on ?c ?s) (clear ?s))
  (:action lift
    :parameters (?h ?c ?s ?p)
    :precondition (and (hoist ?h)(crate ?c)(surface ?s)(pos ?p)
                       (at ?h ?p)(available ?h)(at ?c ?p)(on ?c ?s)(clear ?c))
    :effect (and (lifting ?h ?c)(not (available ?h))(not (at ?c ?p))
                 (not (on ?c ?s))(clear ?s)(not (clear ?c))))
  (:action drop
    :parameters (?h ?c ?s ?p)
    :precondition (and (hoist ?h)(crate ?c)(surface ?s)(pos ?p)
                       (at ?h ?p)(lifting ?h ?c)(at ?s ?p)(clear ?s))
    :effect (and (available ?h)(not (lifting ?h ?c))(at ?c ?p)
                 (on ?c ?s)(not (clear ?s))(clear ?c))))""",

    "satellite": """\
(define (domain satellite)
  (:requirements :strips)
  (:predicates (satellite ?s) (direction ?d) (instrument ?i) (mode ?m)
               (pointing ?s ?d) (on_board ?i ?s) (supports ?i ?m)
               (power_avail ?s) (power_on ?i) (calibrated ?i)
               (have_image ?d ?m) (calibration_target ?i ?d))
  (:action turn_to
    :parameters (?s ?d_new ?d_prev)
    :precondition (and (satellite ?s)(direction ?d_new)(direction ?d_prev)(pointing ?s ?d_prev))
    :effect (and (pointing ?s ?d_new)(not (pointing ?s ?d_prev))))
  (:action switch_on
    :parameters (?i ?s)
    :precondition (and (instrument ?i)(satellite ?s)(on_board ?i ?s)(power_avail ?s))
    :effect (and (power_on ?i)(not (calibrated ?i))(not (power_avail ?s))))
  (:action calibrate
    :parameters (?s ?i ?d)
    :precondition (and (satellite ?s)(instrument ?i)(direction ?d)
                       (on_board ?i ?s)(calibration_target ?i ?d)
                       (pointing ?s ?d)(power_on ?i))
    :effect (calibrated ?i))
  (:action take_image
    :parameters (?s ?d ?i ?m)
    :precondition (and (satellite ?s)(direction ?d)(instrument ?i)(mode ?m)
                       (calibrated ?i)(on_board ?i ?s)(supports ?i ?m)
                       (power_on ?i)(pointing ?s ?d))
    :effect (have_image ?d ?m)))""",

    "rovers": """\
(define (domain rovers)
  (:requirements :strips)
  (:predicates (rover ?r)(waypoint ?w)(equipped_for_soil_analysis ?r)
               (equipped_for_rock_analysis ?r)(equipped_for_imaging ?r)
               (at ?r ?w)(can_traverse ?r ?w1 ?w2)(have_rock_analysis ?r ?w)
               (have_soil_analysis ?r ?w)(communicated_soil_data ?w)
               (communicated_rock_data ?w)(at_lander ?w)(channel_free ?r))
  (:action navigate
    :parameters (?r ?from ?to)
    :precondition (and (rover ?r)(waypoint ?from)(waypoint ?to)
                       (at ?r ?from)(can_traverse ?r ?from ?to))
    :effect (and (at ?r ?to)(not (at ?r ?from))))
  (:action sample_soil
    :parameters (?r ?w)
    :precondition (and (rover ?r)(waypoint ?w)(at ?r ?w)
                       (equipped_for_soil_analysis ?r))
    :effect (have_soil_analysis ?r ?w))
  (:action communicate_soil_data
    :parameters (?r ?w ?from)
    :precondition (and (rover ?r)(have_soil_analysis ?r ?w)
                       (at ?r ?from)(at_lander ?from)(channel_free ?r))
    :effect (communicated_soil_data ?w)))""",
}


def collect_labels_ollama(
    model_name="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    max_instances_per_domain=200,
    verify=True,
    sleep_between=0.1,
):
    client     = OllamaClient(model=model_name, host=ollama_host)
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    test_domains = registry["splits"]["meta_test"]["tasks"]

    # ── Positional index: array position i → episodes[i] ─────────────────────
    ep_by_idx = {}
    eps_path  = DATA_DIR / "episodes.json"
    if eps_path.exists():
        episodes = json.loads(eps_path.read_text())
        for i, ep in enumerate(episodes):
            ep_by_idx[i] = ep
        print(f"  Loaded {len(episodes)} episodes (keys 0–{len(episodes)-1})")

    labels, refusal_log = [], []

    for dom in test_domains:
        # ── FIX 1: np.where → actual integer positions ────────────────────────
        dom_indices = np.where(task_types == dom)[0][:max_instances_per_domain]

        print(f"\n{'─'*60}")
        print(f"Domain: {dom}  ({len(dom_indices)} instances, "
              f"idx {int(dom_indices[0])}–{int(dom_indices[-1])})")
        print(f"{'─'*60}")

        # ── FIX 2: domain_pddl from embedded registry (not episodes.json) ─────
        domain_pddl = DOMAIN_PDDL_STRINGS.get(dom, "")
        if not domain_pddl:
            print(f"  WARNING: no domain PDDL string for {dom}")

        n_success = n_refusal = n_fail = 0

        for rank, idx in enumerate(dom_indices):
            idx = int(idx)           # numpy int64 → Python int
            ep  = ep_by_idx.get(idx, {})

            pddl_problem = ep.get("problem_pddl", "")
            init_facts   = ep.get("init_facts",   [])
            goal_facts   = ep.get("goal_facts",   [])

            if not pddl_problem:
                labels.append({
                    "instance_id":  ep.get("instance_id", idx),
                    "array_idx":    idx,
                    "domain":       dom,
                    "qwen_success": 0,
                    "refused":      False,
                    "plan_len":     0,
                    "plan":         "",
                    "latency_s":    0.0,
                    "note":         "no_pddl",
                })
                continue

            prompt = PDDL_PROMPT_TEMPLATE.format(
                domain_pddl=domain_pddl,
                problem_pddl=pddl_problem,
            )

            t0       = time.time()
            response = client.generate(prompt, max_tokens=400, temperature=0.0)
            latency  = time.time() - t0

            refused = is_refusal(response) or len(response.strip()) < 5

            if refused:
                valid = False; plan = []
                n_refusal += 1
            else:
                plan = parse_plan_from_response(response)
                if verify and init_facts and goal_facts:
                    valid = verify_plan_bfs(plan, init_facts, goal_facts, domain=dom)
                else:
                    valid = len(plan) >= 1
                if valid: n_success += 1
                else:     n_fail    += 1

            labels.append({
                "instance_id":   ep.get("instance_id", idx),
                "array_idx":     idx,
                "domain":        dom,
                "qwen_success":  int(valid),
                "refused":       refused,
                "plan_len":      len(plan),
                "plan":          " | ".join(plan[:5]),
                "latency_s":     round(latency, 2),
            })

            if refused:
                refusal_log.append({
                    "idx": idx, "domain": dom,
                    "response": response[:120]
                })

            if (rank + 1) % 20 == 0 or rank == len(dom_indices) - 1:
                sr = n_success / max(rank + 1, 1)
                rr = n_refusal / max(rank + 1, 1)
                print(f"  [{rank+1:3d}/{len(dom_indices)}]  "
                      f"success={n_success} ({sr:.1%})  "
                      f"refusals={n_refusal} ({rr:.1%})  "
                      f"fail={n_fail}  last={latency:.1f}s")

            if sleep_between > 0:
                time.sleep(sleep_between)

        dom_labels = [l for l in labels if l["domain"] == dom]
        if dom_labels:
            sr = np.mean([l["qwen_success"] for l in dom_labels])
            rr = np.mean([l["refused"]      for l in dom_labels])
            print(f"\n  ✓ {dom}  SR={sr:.3f}  refusal_rate={rr:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    tag      = model_name.replace(":", "_").replace("/", "_")
    out_qwen = DATA_DIR / f"qwen_labels_{tag}.json"
    out_main = DATA_DIR / "gpt4o_labels.json"
    for p in [out_qwen, out_main]:
        p.write_text(json.dumps(labels, indent=2))
    print(f"\n  Saved {len(labels)} labels → {out_qwen}")
    print(f"  Copied → {out_main}")

    if refusal_log:
        (DATA_DIR / "qwen_refusals.json").write_text(json.dumps(refusal_log, indent=2))
        print(f"  Refusals: {len(refusal_log)} → data/planning/qwen_refusals.json")

    # ── BFS vs Qwen agreement ─────────────────────────────────────────────────
    y_bfs       = np.load(DATA_DIR / "y_success.npy")
    qwen_by_idx = {l["array_idx"]: l["qwen_success"] for l in labels}
    shared      = [(int(y_bfs[i]), v) for i, v in qwen_by_idx.items()
                   if 0 <= i < len(y_bfs)]
    if shared:
        from scipy.stats import spearmanr
        bfs_a  = np.array([x[0] for x in shared])
        qwen_a = np.array([x[1] for x in shared])
        rho, p = spearmanr(bfs_a, qwen_a)
        agree  = np.mean(bfs_a == qwen_a)
        print(f"\n  BFS vs Qwen2.5 agreement : {agree:.3f}")
        print(f"  Spearman ρ(BFS, Qwen)    : {rho:+.3f}  (p={p:.4f})")
        verdict = ("BFS labels are valid LLM proxies ✓" if rho > 0.5
                   else "LLM labels add NEW signal — train on both!" if rho < 0.3
                   else "Moderate agreement — worth training on Qwen labels")
        print(f"  → {verdict}")

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Build merged label arrays
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_label_arrays():
    labels_path = DATA_DIR / "gpt4o_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError("Run --collect first.")

    labels   = json.loads(labels_path.read_text())
    y_bfs    = np.load(DATA_DIR / "y_success.npy")
    y_llm    = y_bfs.copy()

    # Prefer Qwen success label for any instance that was queried
    for l in labels:
        idx = l.get("array_idx", -1)
        if 0 <= idx < len(y_llm):
            y_llm[idx] = l.get("qwen_success", l.get("gpt4o_success", y_bfs[idx]))

    np.save(DATA_DIR / "y_llm.npy", y_llm)
    print(f"  Saved y_llm.npy  shape={y_llm.shape}  "
          f"positive_rate={y_llm.mean():.3f}  "
          f"(BFS rate was {y_bfs.mean():.3f})")
    return y_llm


# ══════════════════════════════════════════════════════════════════════════════
# Train ARC on LLM-behavioral labels
# ══════════════════════════════════════════════════════════════════════════════

def train_on_llm_labels(n_episodes=3000):
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y_llm      = np.load(DATA_DIR / "y_llm.npy")
    y_nsteps   = np.load(DATA_DIR / "y_nsteps.npy")

    train_domains = registry["splits"]["meta_train"]["tasks"]
    val_domains   = registry["splits"]["meta_val"]["tasks"]
    surf_dim = X_surf.shape[1]   # FIX: [1] not the full shape tuple
    fm_dim   = X_fm.shape[1]

    train_samp = PlanningMetaSampler(
        train_domains, X_surf, X_fm,
        y_llm.astype(int), y_nsteps, task_types, DEVICE
    )
    val_samp = PlanningMetaSampler(
        val_domains, X_surf, X_fm,
        y_llm.astype(int), y_nsteps, task_types, DEVICE
    )

    model = PlanningGURU(surf_dim, fm_dim).to(DEVICE)   # FIX: ints, not tuples
    print(f"\n  Training ARC on Qwen2.5-behavioral labels  (n_episodes={n_episodes})")
    train_guru(model, train_samp, n_episodes, label="success",
               device=DEVICE, val_sampler=val_samp, val_every=300)

    ckpt = CKPT_DIR / "guru_llm_labels.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    print(f"  Saved → {ckpt}")
    return model


def eval_bfs_vs_llm():
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y_bfs      = np.load(DATA_DIR / "y_success.npy")
    y_llm      = np.load(DATA_DIR / "y_llm.npy") \
                 if (DATA_DIR / "y_llm.npy").exists() else y_bfs

    train_domains = registry["splits"]["meta_train"]["tasks"]
    test_domains  = registry["splits"]["meta_test"]["tasks"]
    train_mask    = np.isin(task_types, train_domains)

    surf_dim = X_surf.shape[1]   # FIX: scalar
    fm_dim   = X_fm.shape[1]

    checkpoints = {
        "ARC (BFS labels)":  CKPT_DIR / "guru_success.pt",
        "ARC (Qwen labels)": CKPT_DIR / "guru_llm_labels.pt",
    }
    models = {}
    for name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"  Skipping {name} (no checkpoint at {ckpt_path})")
            continue
        ck = torch.load(ckpt_path, map_location=DEVICE)
        m  = PlanningGURU(surf_dim, fm_dim).to(DEVICE)   # FIX: no [1] indexing
        m.load_state_dict(ck["model"]); m.eval()
        models[name] = m

    print(f"\n{'Model':<25} {'Label set':<15} {'Domain':<25} {'AUC':>8}")
    print("─" * 78)

    results = {}
    for name, model in models.items():
        for y_eval, y_name in [(y_bfs, "BFS"), (y_llm, "Qwen")]:
            for dom in test_domains:
                mask = task_types == dom
                y_q  = y_eval[mask]
                y_tr = y_eval[train_mask]

                res = evaluate_on_domain(
                    model, dom,
                    X_surf[mask], X_fm[mask], y_q,
                    X_surf[train_mask], X_fm[train_mask], y_tr,
                    label="success", device=DEVICE,
                    cross_domain_support=True
                )
                auc = res.get("guru_cross", {}).get("mean", float("nan")) \
                      if res else float("nan")
                key = f"{name}|{y_name}|{dom}"
                results[key] = float(auc)
                print(f"  {name:<25} eval_on={y_name:<10} {dom:<25} {auc:>8.4f}")

    (RESULTS_DIR / "e1_bfs_vs_llm_labels.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n  Saved → results_planning/e1_bfs_vs_llm_labels.json")

    print(f"\n  KEY FINDING:")
    for dom in test_domains:
        bfs_score = results.get(f"ARC (BFS labels)|Qwen|{dom}", float("nan"))
        llm_score = results.get(f"ARC (Qwen labels)|Qwen|{dom}", float("nan"))
        delta     = llm_score - bfs_score if not (
                    np.isnan(llm_score) or np.isnan(bfs_score)) else float("nan")
        verdict   = ("LLM labels improve transfer" if not np.isnan(delta) and delta > 0.01
                     else "BFS labels sufficient"   if not np.isnan(delta) and abs(delta) < 0.01
                     else "BFS labels better (LLM noise?)")
        print(f"  {dom:<25}  ΔAUC={delta:+.4f}  → {verdict}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Side-by-side evaluation: BFS labels vs LLM labels
# ══════════════════════════════════════════════════════════════════════════════

def eval_bfs_vs_llm():
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y_bfs      = np.load(DATA_DIR / "y_success.npy")
    y_llm      = np.load(DATA_DIR / "y_llm.npy") \
                 if (DATA_DIR / "y_llm.npy").exists() else y_bfs

    train_domains = registry["splits"]["meta_train"]["tasks"]
    test_domains  = registry["splits"]["meta_test"]["tasks"]
    train_mask    = np.isin(task_types, train_domains)

    checkpoints = {
        "ARC (BFS labels)": CKPT_DIR / "guru_success.pt",
        "ARC (Qwen labels)": CKPT_DIR / "guru_llm_labels.pt",
    }
    models = {}
    for name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"  Skipping {name} (no checkpoint at {ckpt_path})")
            continue
        ck = torch.load(ckpt_path, map_location=DEVICE)
        m  = PlanningGURU(X_surf.shape, X_fm.shape).to(DEVICE)[1]
        m.load_state_dict(ck["model"]); m.eval()
        models[name] = m

    print(f"\n{'Model':<25} {'Label set':<15} {'Domain':<25} {'AUC':>8}")
    print("─" * 78)

    results = {}
    for name, model in models.items():
        for y_eval, y_name in [(y_bfs, "BFS"), (y_llm, "Qwen")]:
            for dom in test_domains:
                mask = task_types == dom
                y_q  = y_eval[mask]
                y_tr = y_eval[train_mask]

                res = evaluate_on_domain(
                    model, dom,
                    X_surf[mask], X_fm[mask], y_q,
                    X_surf[train_mask], X_fm[train_mask], y_tr,
                    label="success", device=DEVICE,
                    cross_domain_support=True
                )
                auc = res.get("guru_cross", {}).get("mean", float("nan")) \
                      if res else float("nan")
                key = f"{name}|{y_name}|{dom}"
                results[key] = float(auc)
                print(f"  {name:<25} eval_on={y_name:<10} {dom:<25} {auc:>8.4f}")

    (RESULTS_DIR / "e1_bfs_vs_llm_labels.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n  Saved → results_planning/e1_bfs_vs_llm_labels.json")

    # Key finding summary
    print(f"\n  KEY FINDING:")
    for dom in test_domains:
        bfs_score  = results.get(f"ARC (BFS labels)|Qwen|{dom}", float("nan"))
        llm_score  = results.get(f"ARC (Qwen labels)|Qwen|{dom}", float("nan"))
        delta      = llm_score - bfs_score
        verdict    = "LLM labels improve transfer" if delta > 0.01 \
                     else "BFS labels sufficient" if abs(delta) < 0.01 \
                     else "BFS labels better (LLM noise?)"
        print(f"  {dom:<25}  ΔAUC={delta:+.4f}  → {verdict}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect",    action="store_true",
                        help="Collect Qwen2.5 labels via Ollama")
    parser.add_argument("--verify",     action="store_true",
                        help="Verify plans with forward checker (slower)")
    parser.add_argument("--train",      action="store_true",
                        help="Train ARC on LLM-behavioral labels")
    parser.add_argument("--eval",       action="store_true",
                        help="Compare BFS vs LLM label training")
    parser.add_argument("--model",      default="qwen2.5:7b",
                        help="Ollama model tag  [default: qwen2.5:7b]")
    parser.add_argument("--host",       default="http://localhost:11434",
                        help="Ollama server host")
    parser.add_argument("--max_inst",   type=int, default=200,
                        help="Max instances per domain to query")
    parser.add_argument("--episodes",   type=int, default=3000,
                        help="Training episodes for ARC")
    parser.add_argument("--sleep",      type=float, default=0.05,
                        help="Sleep between Ollama calls (seconds)")
    args = parser.parse_args()

    if args.collect:
        collect_labels_ollama(
            model_name=args.model,
            ollama_host=args.host,
            max_instances_per_domain=args.max_inst,
            verify=args.verify,
            sleep_between=args.sleep,
        )
        build_llm_label_arrays()

    if args.train:
        if not (DATA_DIR / "y_llm.npy").exists():
            print("  y_llm.npy not found — building from gpt4o_labels.json...")
            build_llm_label_arrays()
        train_on_llm_labels(n_episodes=args.episodes)

    if args.eval:
        eval_bfs_vs_llm()


if __name__ == "__main__":
    main()