"""
plan_step8_gpt4o_eval.py
========================
Run deterministic planning evaluation on the planning dataset (OpenAI or Mistral
chat completions) and produce per-instance outcomes plus aggregate metrics.

Default scope: meta_test domains from data/planning/registry.json.

Outputs (OpenAI default):
  - results_planning/gpt4o_eval_instances.jsonl
  - results_planning/gpt4o_eval_summary.json

Outputs (``--provider mistral`` default):
  - results_planning/mistral_eval_instances.jsonl
  - results_planning/mistral_eval_summary.json

Each JSONL row includes:
  instance_id, domain, raw_response, parsed_actions, parse_ok, exec_ok,
  goal_ok, valid_plan, error_type, first_fail_step_idx, first_fail_action,
  missing_preconditions, latency_ms, prompt_tokens, completion_tokens,
  total_tokens

Usage:
  python plan_step8_gpt4o_eval.py
  python plan_step8_gpt4o_eval.py --provider mistral
  python plan_step8_gpt4o_eval.py --provider mistral --model mistral-small-latest
  python plan_step8_gpt4o_eval.py --max-instances 6
  python plan_step8_gpt4o_eval.py --prompt-profile balanced_hard
  python plan_step8_gpt4o_eval.py --force
  python plan_step8_gpt4o_eval.py --self-test

Requires ``mistralai`` when using ``--provider mistral`` (``pip install mistralai``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

try:
    from mistralai import Mistral
    from mistralai.models.sdkerror import SDKError as MistralSDKError
except ImportError:
    Mistral = None  # type: ignore[misc, assignment]
    MistralSDKError = None  # type: ignore[misc, assignment]


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "planning"
RESULTS_DIR = ROOT_DIR / "results_planning"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_INSTANCE_OUT = RESULTS_DIR / "gpt4o_eval_instances.jsonl"
DEFAULT_SUMMARY_OUT = RESULTS_DIR / "gpt4o_eval_summary.json"
MISTRAL_DEFAULT_INSTANCE_OUT = RESULTS_DIR / "mistral_eval_instances.jsonl"
MISTRAL_DEFAULT_SUMMARY_OUT = RESULTS_DIR / "mistral_eval_summary.json"

DEFAULT_MODEL_OPENAI = "gpt-4o"
DEFAULT_MODEL_MISTRAL = "mistral-large-latest"

SUPPORTED_DOMAINS = {"blocksworld", "mystery_blocksworld", "logistics"}

# episodes.json intentionally drops domain_pddl for file size/caching.
# Keep exact domain text here so GPT sees the same operator schema used at data gen.
DOMAIN_PDDL_MAP = {
    "blocksworld": """
(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (handempty) (holding ?x))
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x)) (not (handempty))))
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x))))
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y))))
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))))
""".strip(),
    "mystery_blocksworld": """
(define (domain mystery_blocksworld)
  (:requirements :strips)
  (:predicates (stacked ?x ?y) (grounded ?x) (apex ?x) (grasping) (clutching ?x))
  (:action grasp
    :parameters (?x)
    :precondition (and (apex ?x) (grounded ?x) (grasping))
    :effect (and (clutching ?x) (not (grounded ?x)) (not (apex ?x)) (not (grasping))))
  (:action release
    :parameters (?x)
    :precondition (clutching ?x)
    :effect (and (grounded ?x) (apex ?x) (grasping) (not (clutching ?x))))
  (:action place
    :parameters (?x ?y)
    :precondition (and (clutching ?x) (apex ?y))
    :effect (and (stacked ?x ?y) (apex ?x) (grasping) (not (clutching ?x)) (not (apex ?y))))
  (:action lift
    :parameters (?x ?y)
    :precondition (and (stacked ?x ?y) (apex ?x) (grasping))
    :effect (and (clutching ?x) (apex ?y) (not (stacked ?x ?y)) (not (grasping)))))
""".strip(),
    "logistics": """
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
    :effect (and (not (at ?airplane ?from)) (at ?airplane ?to))))
""".strip(),
}

SYSTEM_PROMPT_LEGACY = (
    "You are a precise PDDL planner. Return only a grounded action sequence. "
    "Each action must be exactly one parenthesized line like (pick-up b1). "
    "Do not include explanations, markdown, or extra text."
)

SYSTEM_PROMPT_BALANCED = (
    "You are a strict PDDL planner. "
    "Output only valid grounded actions in exact schema form. "
    "No prose, no numbering, no markdown."
)

SYSTEM_PROMPT_BALANCED_HARD = (
    "You are a strict STRIPS planner. "
    "Silently parse objects, init facts, and goals. "
    "Internally simulate state transitions from :init. "
    "Emit only executable grounded actions in exact schema form. "
    "No prose, no numbering, no markdown."
)

USER_PROMPT_TEMPLATE_LEGACY = """Solve this planning problem and output only the plan.

Constraints:
- Output one grounded action per line.
- Use only actions defined in the domain.
- If no valid plan is found, output exactly: NO-PLAN

Domain PDDL:
{domain_pddl}

Problem PDDL:
{problem_pddl}
"""

USER_PROMPT_TEMPLATE_BALANCED = """Solve this planning problem and output only a grounded plan.

Output contract (strict):
- Output must be either:
  1) one or more lines of `(action arg1 arg2 ...)`, OR
  2) exactly `NO-PLAN`
- Do not output any explanation, markdown, bullets, numbering, or comments.
- Use action names exactly as listed below (no aliases/synonyms).
- Use only object symbols present in the problem.
- Prefer best-effort executable plans; return `NO-PLAN` only if no valid plan can be found.

Allowed action signatures:
{action_signatures}

Domain PDDL:
{domain_pddl}

Problem PDDL:
{problem_pddl}
"""

USER_PROMPT_TEMPLATE_BALANCED_HARD = """Solve this planning problem and output only a grounded plan.

Output contract (strict):
- Output must be either:
  1) one or more lines of `(action arg1 arg2 ...)`, OR
  2) exactly `NO-PLAN`
- Do not output any explanation, markdown, bullets, numbering, or comments.
- Use action names exactly as listed below (no aliases/synonyms).
- Use only object symbols present in the problem.

Internal execution checklist (do not print):
1) Parse object symbols, init facts, and goal facts.
2) Maintain state S initialized to :init.
3) Before each action, verify all preconditions hold in S.
4) Apply add/delete effects to update S after each action.
5) Continue until all goals hold in S.
6) If you cannot construct a valid executable plan, return exactly `NO-PLAN`.

Domain-specific anti-drift rules:
- logistics:
  - Never use `load-*`/`unload-*` unless carrier and package preconditions hold at that location.
  - For `drive-truck(truck, src, dst, city)`, both src and dst must satisfy `in-city(..., city)`.
- blocksworld:
  - `pick-up(x)` requires `(clear x) (ontable x) (handempty)`.
  - `unstack(x y)` requires `(on x y) (clear x) (handempty)`.
- mystery_blocksworld:
  - Enforce exact preconditions/effects from the provided predicates at every step.

Allowed action signatures:
{action_signatures}

Domain PDDL:
{domain_pddl}

Problem PDDL:
{problem_pddl}
"""

PAREN_ACTION_RE = re.compile(r"\(([^()\n]+)\)")
CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?")
NO_PLAN_MARKERS = {"no-plan", "no plan", "unsat", "unsolvable", "none"}
LINE_ACTION_RE = re.compile(r"^\(\s*([a-zA-Z][a-zA-Z0-9_-]*)\s*([^()]*)\)$")
LEADING_ENUM_RE = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)")
NO_PLAN_ALIASES = {"no plan found", "no valid plan", "no valid plan found"}
FILTER_HEADS = {"and", "not", "define", "problem", "domain", "objects", "init", "goal"}


def _fact(pred: str, *args: str) -> str:
    parts = [pred] + list(args)
    return "(" + " ".join(parts) + ")"


def _norm_fact(fact: str) -> str:
    txt = fact.strip().lower()
    if txt.startswith("(") and txt.endswith(")"):
        txt = txt[1:-1]
    txt = " ".join(txt.split())
    return f"({txt})"


def resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p


def resolve_default_output_paths(
    prompt_profile: str,
    max_instances: int,
    provider: str = "openai",
) -> Tuple[Path, Path]:
    prefix = "gpt4o" if provider == "openai" else "mistral"
    if prompt_profile != "balanced_hard":
        return (
            RESULTS_DIR / f"{prefix}_eval_instances.jsonl",
            RESULTS_DIR / f"{prefix}_eval_summary.json",
        )
    suffix = "_smoke" if max_instances and max_instances > 0 else ""
    return (
        RESULTS_DIR / f"{prefix}_eval_instances_balanced_hard{suffix}.jsonl",
        RESULTS_DIR / f"{prefix}_eval_summary_balanced_hard{suffix}.json",
    )


def load_records(scope: str = "meta_test", data_dir: Path = DEFAULT_DATA_DIR) -> List[dict]:
    episodes_path = data_dir / "episodes.json"
    registry_path = data_dir / "registry.json"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing dataset: {episodes_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing registry: {registry_path}")

    records = json.loads(episodes_path.read_text())
    registry = json.loads(registry_path.read_text())
    meta_test = set(registry.get("test_domains", []))

    if scope == "meta_test":
        out = [r for r in records if r.get("task_type") in meta_test]
    elif scope == "all":
        out = records
    else:
        raise ValueError(f"Unsupported scope: {scope}")

    out = [r for r in out if r.get("task_type") in SUPPORTED_DOMAINS]
    out.sort(key=lambda r: int(r.get("instance_id", -1)))
    return out


def balanced_subset(records: List[dict], max_instances: int) -> List[dict]:
    if max_instances <= 0 or len(records) <= max_instances:
        return records

    by_dom: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_dom[r["task_type"]].append(r)

    domains = sorted(by_dom.keys())
    picked: List[dict] = []
    idx = {d: 0 for d in domains}
    while len(picked) < max_instances:
        progress = False
        for dom in domains:
            i = idx[dom]
            if i < len(by_dom[dom]) and len(picked) < max_instances:
                picked.append(by_dom[dom][i])
                idx[dom] += 1
                progress = True
        if not progress:
            break
    picked.sort(key=lambda r: int(r.get("instance_id", -1)))
    return picked


def load_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def get_domain_pddl(record: dict) -> str:
    if record.get("domain_pddl"):
        return str(record["domain_pddl"])
    dom = str(record.get("task_type") or record.get("domain") or "").strip()
    if dom in DOMAIN_PDDL_MAP:
        return DOMAIN_PDDL_MAP[dom]
    raise KeyError(f"Missing domain PDDL for domain={dom}")


def action_signatures_for_domain(domain: str) -> str:
    table = {
        "blocksworld": [
            "pick-up(obj)",
            "put-down(obj)",
            "stack(obj, target)",
            "unstack(obj, target)",
        ],
        "mystery_blocksworld": [
            "grasp(obj)",
            "release(obj)",
            "place(obj, target)",
            "lift(obj, target)",
        ],
        "logistics": [
            "load-truck(pkg, truck, loc)",
            "unload-truck(pkg, truck, loc)",
            "load-airplane(pkg, airplane, airport)",
            "unload-airplane(pkg, airplane, airport)",
            "drive-truck(truck, src, dst, city)",
            "fly-airplane(airplane, src_airport, dst_airport)",
        ],
    }
    if domain not in table:
        raise KeyError(f"No action-signature table for domain={domain}")
    return "\n".join(f"- {x}" for x in table[domain])


def build_prompts(prompt_profile: str, domain: str, domain_pddl: str, problem_pddl: str) -> Tuple[str, str]:
    if prompt_profile == "legacy":
        return (
            SYSTEM_PROMPT_LEGACY,
            USER_PROMPT_TEMPLATE_LEGACY.format(
                domain_pddl=domain_pddl,
                problem_pddl=problem_pddl,
            ),
        )
    if prompt_profile == "balanced":
        return (
            SYSTEM_PROMPT_BALANCED,
            USER_PROMPT_TEMPLATE_BALANCED.format(
                action_signatures=action_signatures_for_domain(domain),
                domain_pddl=domain_pddl,
                problem_pddl=problem_pddl,
            ),
        )
    if prompt_profile == "balanced_hard":
        return (
            SYSTEM_PROMPT_BALANCED_HARD,
            USER_PROMPT_TEMPLATE_BALANCED_HARD.format(
                action_signatures=action_signatures_for_domain(domain),
                domain_pddl=domain_pddl,
                problem_pddl=problem_pddl,
            ),
        )
    raise ValueError(f"Unsupported prompt profile: {prompt_profile}")


def _parse_action_tokens(toks: List[str]) -> Optional[Tuple[str, List[str], str]]:
    if len(toks) == 0 or len(toks) > 5:
        return None
    if any(("?" in t) or (":" in t) for t in toks):
        return None
    head = toks[0].lower()
    if head in FILTER_HEADS:
        return None
    args = [t.lower() for t in toks[1:]]
    canonical = "(" + " ".join([head] + args) + ")"
    return head, args, canonical


def _strip_code_fences(txt: str) -> str:
    lines = []
    for raw_line in txt.splitlines():
        if CODE_FENCE_RE.match(raw_line.strip()):
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def _extract_line_actions(txt: str) -> List[Tuple[str, List[str], str]]:
    actions: List[Tuple[str, List[str], str]] = []
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = LEADING_ENUM_RE.sub("", line).strip()
        m = LINE_ACTION_RE.match(line)
        if not m:
            continue
        head = m.group(1)
        rest = m.group(2).strip()
        toks = [head] + ([t for t in rest.split() if t] if rest else [])
        parsed = _parse_action_tokens(toks)
        if parsed is not None:
            actions.append(parsed)
    return actions


def _extract_fallback_actions(txt: str) -> List[Tuple[str, List[str], str]]:
    actions: List[Tuple[str, List[str], str]] = []
    for m in PAREN_ACTION_RE.finditer(txt):
        start, end = m.span()
        prev_ch = txt[start - 1] if start > 0 else ""
        next_ch = txt[end] if end < len(txt) else ""
        # Avoid nested captures like ((pick-up b1)).
        if prev_ch == "(" or next_ch == ")":
            continue
        inner = " ".join(m.group(1).strip().split())
        if not inner:
            continue
        toks = inner.split()
        parsed = _parse_action_tokens(toks)
        if parsed is not None:
            actions.append(parsed)
    return actions


def _is_no_plan_text(txt: str) -> bool:
    normalized = " ".join(txt.lower().replace("`", " ").split()).strip()
    normalized = normalized.strip(".!;:")
    if normalized in NO_PLAN_MARKERS:
        return True
    return normalized in NO_PLAN_ALIASES


def extract_actions(raw: str) -> Tuple[List[Tuple[str, List[str], str]], bool, Optional[str]]:
    if raw is None:
        return [], False, "parse_error"

    txt = _strip_code_fences(raw.strip())

    line_actions = _extract_line_actions(txt)
    if line_actions:
        return line_actions, True, None

    regex_actions = _extract_fallback_actions(txt)
    if regex_actions:
        return regex_actions, True, None

    if _is_no_plan_text(txt):
        return [], False, "empty_plan"
    return [], False, "parse_error"


def apply_action(domain: str, state: set, action_name: str, args: List[str]) -> Tuple[bool, Optional[str], set]:
    s = set(state)

    def require(preconds: Iterable[str]) -> bool:
        return set(preconds).issubset(s)

    if domain == "blocksworld":
        if action_name == "pick-up":
            if len(args) != 1:
                return False, "arity_mismatch", state
            x = args[0]
            pre = {_fact("clear", x), _fact("ontable", x), _fact("handempty")}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("ontable", x), _fact("clear", x), _fact("handempty")}
            s |= {_fact("holding", x)}
            return True, None, s
        if action_name == "put-down":
            if len(args) != 1:
                return False, "arity_mismatch", state
            x = args[0]
            pre = {_fact("holding", x)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("holding", x)}
            s |= {_fact("handempty"), _fact("ontable", x), _fact("clear", x)}
            return True, None, s
        if action_name == "stack":
            if len(args) != 2:
                return False, "arity_mismatch", state
            x, y = args
            pre = {_fact("holding", x), _fact("clear", y)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("holding", x), _fact("clear", y)}
            s |= {_fact("handempty"), _fact("on", x, y), _fact("clear", x)}
            return True, None, s
        if action_name == "unstack":
            if len(args) != 2:
                return False, "arity_mismatch", state
            x, y = args
            pre = {_fact("on", x, y), _fact("clear", x), _fact("handempty")}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("on", x, y), _fact("clear", x), _fact("handempty")}
            s |= {_fact("holding", x), _fact("clear", y)}
            return True, None, s
        return False, "unknown_action", state

    if domain == "mystery_blocksworld":
        if action_name == "grasp":
            if len(args) != 1:
                return False, "arity_mismatch", state
            x = args[0]
            pre = {_fact("apex", x), _fact("grounded", x), _fact("grasping")}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("grounded", x), _fact("apex", x), _fact("grasping")}
            s |= {_fact("clutching", x)}
            return True, None, s
        if action_name == "release":
            if len(args) != 1:
                return False, "arity_mismatch", state
            x = args[0]
            pre = {_fact("clutching", x)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("clutching", x)}
            s |= {_fact("grasping"), _fact("grounded", x), _fact("apex", x)}
            return True, None, s
        if action_name == "place":
            if len(args) != 2:
                return False, "arity_mismatch", state
            x, y = args
            pre = {_fact("clutching", x), _fact("apex", y)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("clutching", x), _fact("apex", y)}
            s |= {_fact("grasping"), _fact("stacked", x, y), _fact("apex", x)}
            return True, None, s
        if action_name == "lift":
            if len(args) != 2:
                return False, "arity_mismatch", state
            x, y = args
            pre = {_fact("stacked", x, y), _fact("apex", x), _fact("grasping")}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("stacked", x, y), _fact("grasping")}
            s |= {_fact("clutching", x), _fact("apex", y)}
            return True, None, s
        return False, "unknown_action", state

    if domain == "logistics":
        if action_name == "load-truck":
            if len(args) != 3:
                return False, "arity_mismatch", state
            pkg, truck, loc = args
            pre = {_fact("at", truck, loc), _fact("at", pkg, loc)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("at", pkg, loc)}
            s |= {_fact("in", pkg, truck)}
            return True, None, s
        if action_name == "unload-truck":
            if len(args) != 3:
                return False, "arity_mismatch", state
            pkg, truck, loc = args
            pre = {_fact("at", truck, loc), _fact("in", pkg, truck)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("in", pkg, truck)}
            s |= {_fact("at", pkg, loc)}
            return True, None, s
        if action_name == "load-airplane":
            if len(args) != 3:
                return False, "arity_mismatch", state
            pkg, plane, loc = args
            pre = {_fact("at", plane, loc), _fact("at", pkg, loc)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("at", pkg, loc)}
            s |= {_fact("in", pkg, plane)}
            return True, None, s
        if action_name == "unload-airplane":
            if len(args) != 3:
                return False, "arity_mismatch", state
            pkg, plane, loc = args
            pre = {_fact("at", plane, loc), _fact("in", pkg, plane)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("in", pkg, plane)}
            s |= {_fact("at", pkg, loc)}
            return True, None, s
        if action_name == "drive-truck":
            if len(args) != 4:
                return False, "arity_mismatch", state
            truck, src, dst, city = args
            pre = {
                _fact("at", truck, src),
                _fact("in-city", src, city),
                _fact("in-city", dst, city),
            }
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("at", truck, src)}
            s |= {_fact("at", truck, dst)}
            return True, None, s
        if action_name == "fly-airplane":
            if len(args) != 3:
                return False, "arity_mismatch", state
            plane, src, dst = args
            pre = {_fact("at", plane, src)}
            if not require(pre):
                return False, "precondition_fail", state
            s -= {_fact("at", plane, src)}
            s |= {_fact("at", plane, dst)}
            return True, None, s
        return False, "unknown_action", state

    return False, "unknown_action", state


def expected_preconditions(domain: str, action_name: str, args: List[str]) -> Tuple[Optional[str], Optional[set]]:
    if domain == "blocksworld":
        if action_name == "pick-up":
            if len(args) != 1:
                return "arity_mismatch", None
            x = args[0]
            return None, {_fact("clear", x), _fact("ontable", x), _fact("handempty")}
        if action_name == "put-down":
            if len(args) != 1:
                return "arity_mismatch", None
            x = args[0]
            return None, {_fact("holding", x)}
        if action_name == "stack":
            if len(args) != 2:
                return "arity_mismatch", None
            x, y = args
            return None, {_fact("holding", x), _fact("clear", y)}
        if action_name == "unstack":
            if len(args) != 2:
                return "arity_mismatch", None
            x, y = args
            return None, {_fact("on", x, y), _fact("clear", x), _fact("handempty")}
        return "unknown_action", None

    if domain == "mystery_blocksworld":
        if action_name == "grasp":
            if len(args) != 1:
                return "arity_mismatch", None
            x = args[0]
            return None, {_fact("apex", x), _fact("grounded", x), _fact("grasping")}
        if action_name == "release":
            if len(args) != 1:
                return "arity_mismatch", None
            x = args[0]
            return None, {_fact("clutching", x)}
        if action_name == "place":
            if len(args) != 2:
                return "arity_mismatch", None
            x, y = args
            return None, {_fact("clutching", x), _fact("apex", y)}
        if action_name == "lift":
            if len(args) != 2:
                return "arity_mismatch", None
            x, y = args
            return None, {_fact("stacked", x, y), _fact("apex", x), _fact("grasping")}
        return "unknown_action", None

    if domain == "logistics":
        if action_name == "load-truck":
            if len(args) != 3:
                return "arity_mismatch", None
            pkg, truck, loc = args
            return None, {_fact("at", truck, loc), _fact("at", pkg, loc)}
        if action_name == "unload-truck":
            if len(args) != 3:
                return "arity_mismatch", None
            pkg, truck, loc = args
            return None, {_fact("at", truck, loc), _fact("in", pkg, truck)}
        if action_name == "load-airplane":
            if len(args) != 3:
                return "arity_mismatch", None
            pkg, plane, loc = args
            return None, {_fact("at", plane, loc), _fact("at", pkg, loc)}
        if action_name == "unload-airplane":
            if len(args) != 3:
                return "arity_mismatch", None
            pkg, plane, loc = args
            return None, {_fact("at", plane, loc), _fact("in", pkg, plane)}
        if action_name == "drive-truck":
            if len(args) != 4:
                return "arity_mismatch", None
            truck, src, dst, city = args
            return None, {_fact("at", truck, src), _fact("in-city", src, city), _fact("in-city", dst, city)}
        if action_name == "fly-airplane":
            if len(args) != 3:
                return "arity_mismatch", None
            plane, src, _dst = args
            return None, {_fact("at", plane, src)}
        return "unknown_action", None

    return "unknown_action", None


def validate_plan(
    domain: str,
    init_facts: Iterable[str],
    goal_facts: Iterable[str],
    parsed_actions: List[Tuple[str, List[str], str]],
) -> Tuple[bool, bool, Optional[str], Optional[dict]]:
    if len(parsed_actions) == 0:
        return False, False, "empty_plan", None

    state = {_norm_fact(f) for f in init_facts}
    goals = {_norm_fact(g) for g in goal_facts}

    for step_idx, (action_name, args, canonical) in enumerate(parsed_actions, start=1):
        ok, err, next_state = apply_action(domain, state, action_name, args)
        if not ok:
            fail = {
                "first_fail_step_idx": step_idx,
                "first_fail_action": canonical,
                "missing_preconditions": None,
            }
            if err == "precondition_fail":
                pre_err, preconds = expected_preconditions(domain, action_name, args)
                if pre_err is None and preconds is not None:
                    missing = sorted([p for p in preconds if p not in state])
                    fail["missing_preconditions"] = missing
            return False, False, err, fail
        state = next_state

    if goals.issubset(state):
        return True, True, None, None
    return True, False, "goal_not_reached", None


def call_gpt4o_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
    retry_base_s: float,
) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            msg = resp.choices[0].message.content if resp.choices else ""
            usage = resp.usage
            p_tok = int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else None
            c_tok = int(usage.completion_tokens) if usage and usage.completion_tokens is not None else None
            t_tok = int(usage.total_tokens) if usage and usage.total_tokens is not None else None
            return msg or "", p_tok, c_tok, t_tok
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt >= max_retries:
                raise e
            wait_s = retry_base_s * (2 ** attempt)
            time.sleep(wait_s)
        except APIError:
            raise

    raise RuntimeError("Unreachable retry path")


def _mistral_transient(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError)):
        return True
    if MistralSDKError is not None and isinstance(exc, MistralSDKError):
        sc = getattr(exc, "status_code", -1)
        return sc in (408, 425, 429, 500, 502, 503, 504)
    return False


def call_mistral_with_retry(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
    retry_base_s: float,
) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.complete(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            msg = ""
            if resp.choices:
                ch0 = resp.choices[0]
                raw_content = getattr(ch0.message, "content", None)
                if isinstance(raw_content, str):
                    msg = raw_content
                elif raw_content:
                    parts: List[str] = []
                    for chunk in raw_content:
                        t = getattr(chunk, "text", None) or getattr(chunk, "content", None)
                        if t:
                            parts.append(str(t))
                    msg = "".join(parts)
            usage = getattr(resp, "usage", None)
            p_tok = int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else None
            c_tok = int(usage.completion_tokens) if usage and usage.completion_tokens is not None else None
            t_tok = int(usage.total_tokens) if usage and usage.total_tokens is not None else None
            return msg or "", p_tok, c_tok, t_tok
        except Exception as e:
            if _mistral_transient(e):
                if attempt >= max_retries:
                    raise e
                wait_s = retry_base_s * (2**attempt)
                time.sleep(wait_s)
                continue
            raise

    raise RuntimeError("Unreachable retry path")


def summarize_rows(
    rows: List[dict],
    price_in_per_mtok: Optional[float],
    price_out_per_mtok: Optional[float],
) -> dict:
    by_dom: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)

    def _summary(sub_rows: List[dict]) -> dict:
        n = len(sub_rows)
        if n == 0:
            return {}
        valid = [1.0 if r.get("valid_plan") else 0.0 for r in sub_rows]
        parse = [1.0 if r.get("parse_ok") else 0.0 for r in sub_rows]
        exec_ok = [1.0 if r.get("exec_ok") else 0.0 for r in sub_rows]
        goal = [1.0 if r.get("goal_ok") else 0.0 for r in sub_rows]
        plan_lens = [len(r.get("parsed_actions", [])) for r in sub_rows]
        lat = [float(r.get("latency_ms", 0.0)) for r in sub_rows]
        p_tok = sum(int(r.get("prompt_tokens", 0) or 0) for r in sub_rows)
        c_tok = sum(int(r.get("completion_tokens", 0) or 0) for r in sub_rows)
        t_tok = sum(int(r.get("total_tokens", 0) or 0) for r in sub_rows)
        err_counts = Counter((r.get("error_type") or "none") for r in sub_rows)

        out = {
            "n_instances": n,
            "validity_rate": float(sum(valid) / n),
            "parse_rate": float(sum(parse) / n),
            "exec_rate": float(sum(exec_ok) / n),
            "goal_rate": float(sum(goal) / n),
            "avg_pred_plan_len": float(sum(plan_lens) / n),
            "latency_ms": {
                "mean": float(sum(lat) / n),
                "median": float(statistics.median(lat)),
                "p95": float(_percentile(lat, 95)),
            },
            "tokens": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
            },
            "error_counts": dict(err_counts),
        }
        if price_in_per_mtok is not None and price_out_per_mtok is not None:
            out["estimated_cost_usd"] = float(
                (p_tok / 1_000_000.0) * price_in_per_mtok
                + (c_tok / 1_000_000.0) * price_out_per_mtok
            )
        return out

    domain_summaries = {d: _summary(v) for d, v in sorted(by_dom.items())}
    global_summary = _summary(rows)
    return {
        "global": global_summary,
        "domains": domain_summaries,
    }


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(xs[lo])
    frac = rank - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def run_self_test() -> None:
    # Parse checks
    raw = "```\n(pick-up b1)\n(stack b1 b2)\n```"
    acts, ok, err = extract_actions(raw)
    assert ok and err is None
    assert [a[2] for a in acts] == ["(pick-up b1)", "(stack b1 b2)"]

    raw_numbered = "1. (pick-up b1)\n2) (stack b1 b2)"
    acts, ok, err = extract_actions(raw_numbered)
    assert ok and err is None
    assert [a[2] for a in acts] == ["(pick-up b1)", "(stack b1 b2)"]

    raw_nested = "((pick-up b1))"
    acts, ok, err = extract_actions(raw_nested)
    assert (not ok) and err == "parse_error" and len(acts) == 0

    raw_prose = "I cannot compute this exactly right now."
    acts, ok, err = extract_actions(raw_prose)
    assert (not ok) and err == "parse_error" and len(acts) == 0

    raw_empty = "NO-PLAN"
    acts, ok, err = extract_actions(raw_empty)
    assert (not ok) and err == "empty_plan" and len(acts) == 0

    # Blocksworld valid
    init_bw = ["(ontable b1)", "(ontable b2)", "(clear b1)", "(clear b2)", "(handempty)"]
    goal_bw = ["(on b1 b2)"]
    plan_bw = [("pick-up", ["b1"], "(pick-up b1)"), ("stack", ["b1", "b2"], "(stack b1 b2)")]
    exec_ok, goal_ok, err, diag = validate_plan("blocksworld", init_bw, goal_bw, plan_bw)
    assert exec_ok and goal_ok and err is None
    assert diag is None

    # Mystery-BW valid
    init_m = ["(grounded b1)", "(grounded b2)", "(apex b1)", "(apex b2)", "(grasping)"]
    goal_m = ["(stacked b1 b2)"]
    plan_m = [("grasp", ["b1"], "(grasp b1)"), ("place", ["b1", "b2"], "(place b1 b2)")]
    exec_ok, goal_ok, err, diag = validate_plan("mystery_blocksworld", init_m, goal_m, plan_m)
    assert exec_ok and goal_ok and err is None
    assert diag is None

    # Logistics valid
    init_l = [
        "(at truck1 loc1)",
        "(at pkg1 loc1)",
        "(in-city loc1 city1)",
        "(in-city loc2 city1)",
    ]
    goal_l = ["(at pkg1 loc2)"]
    plan_l = [
        ("load-truck", ["pkg1", "truck1", "loc1"], "(load-truck pkg1 truck1 loc1)"),
        ("drive-truck", ["truck1", "loc1", "loc2", "city1"], "(drive-truck truck1 loc1 loc2 city1)"),
        ("unload-truck", ["pkg1", "truck1", "loc2"], "(unload-truck pkg1 truck1 loc2)"),
    ]
    exec_ok, goal_ok, err, diag = validate_plan("logistics", init_l, goal_l, plan_l)
    assert exec_ok and goal_ok and err is None
    assert diag is None

    # Invalid checks by taxonomy
    exec_ok, goal_ok, err, diag = validate_plan("blocksworld", init_bw, goal_bw, [("unknown", ["b1"], "(unknown b1)")])
    assert (not exec_ok) and (not goal_ok) and err == "unknown_action"
    assert diag and diag["first_fail_step_idx"] == 1 and diag["first_fail_action"] == "(unknown b1)"

    exec_ok, goal_ok, err, diag = validate_plan("logistics", init_l, goal_l, [("load-truck", ["pkg1"], "(load-truck pkg1)")])
    assert (not exec_ok) and (not goal_ok) and err == "arity_mismatch"
    assert diag and diag["first_fail_step_idx"] == 1 and diag["first_fail_action"] == "(load-truck pkg1)"

    exec_ok, goal_ok, err, diag = validate_plan(
        "mystery_blocksworld",
        init_m,
        goal_m,
        [("place", ["b1", "b2"], "(place b1 b2)")],
    )
    assert (not exec_ok) and (not goal_ok) and err == "precondition_fail"
    assert diag and "(clutching b1)" in (diag.get("missing_preconditions") or [])

    print("Self-test passed: parser + validators + taxonomy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["openai", "mistral"],
        default="openai",
        help="LLM API: OpenAI Chat Completions or Mistral chat.complete.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Model id. Default: gpt-4o (openai) or mistral-large-latest (mistral).",
    )
    parser.add_argument("--scope", choices=["meta_test", "all"], default="meta_test")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--prompt-profile",
        choices=["legacy", "balanced", "balanced_hard"],
        default="balanced",
    )
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=1.5)
    parser.add_argument(
        "--instance-out",
        default=None,
        help="Per-instance JSONL path. Default depends on --provider. If set to that "
             "provider's default path and profile=balanced_hard, uses profile-tagged filenames.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Summary JSON path. Default depends on --provider.",
    )
    parser.add_argument("--price-in-per-mtok", type=float, default=None)
    parser.add_argument("--price-out-per-mtok", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    load_dotenv()
    provider = args.provider
    resolved_model = args.model or (
        DEFAULT_MODEL_OPENAI if provider == "openai" else DEFAULT_MODEL_MISTRAL
    )

    openai_client: Optional[OpenAI] = None
    mistral_client: Any = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in your environment or .env file.")
        openai_client = OpenAI(api_key=api_key)
    else:
        if Mistral is None:
            raise RuntimeError(
                "mistralai is not installed. Run: pip install mistralai"
            )
        m_key = os.getenv("MISTRAL_API_KEY")
        if not m_key:
            raise RuntimeError("MISTRAL_API_KEY is missing. Put it in your environment or .env file.")
        mistral_client = Mistral(api_key=m_key)

    data_dir = resolve_data_dir(args.data_dir)
    records = load_records(scope=args.scope, data_dir=data_dir)
    records = balanced_subset(records, args.max_instances) if args.max_instances else records

    default_instance, default_summary = resolve_default_output_paths(
        args.prompt_profile, args.max_instances, provider
    )
    instance_out_arg = args.instance_out
    summary_out_arg = args.summary_out
    if instance_out_arg is None:
        instance_out_arg = str(default_instance)
    elif provider == "openai" and instance_out_arg == str(DEFAULT_INSTANCE_OUT):
        instance_out_arg = str(default_instance)
    elif provider == "mistral" and instance_out_arg == str(MISTRAL_DEFAULT_INSTANCE_OUT):
        instance_out_arg = str(default_instance)

    if summary_out_arg is None:
        summary_out_arg = str(default_summary)
    elif provider == "openai" and summary_out_arg == str(DEFAULT_SUMMARY_OUT):
        summary_out_arg = str(default_summary)
    elif provider == "mistral" and summary_out_arg == str(MISTRAL_DEFAULT_SUMMARY_OUT):
        summary_out_arg = str(default_summary)

    out_jsonl = Path(instance_out_arg)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(summary_out_arg)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    if args.force and out_jsonl.exists():
        out_jsonl.unlink()

    existing_rows = load_existing_rows(out_jsonl)
    done_ids = {int(r["instance_id"]) for r in existing_rows if "instance_id" in r}

    target_ids = {int(r["instance_id"]) for r in records}
    remaining = [r for r in records if int(r["instance_id"]) not in done_ids]

    print("=" * 72)
    print("Step 8 - Planning Eval (LLM)")
    print(f"  Provider: {provider}")
    print(f"  Model: {resolved_model}")
    print(f"  Prompt profile: {args.prompt_profile}")
    print(f"  Scope: {args.scope}")
    print(f"  Data dir: {data_dir}")
    print(f"  Selected instances: {len(records)}")
    print(f"  Already done: {len(records) - len(remaining)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  JSONL: {out_jsonl}")
    print(f"  Summary: {out_summary}")
    print("=" * 72)

    with out_jsonl.open("a") as f:
        for i, rec in enumerate(remaining, start=1):
            dom = rec["task_type"]
            iid = int(rec["instance_id"])
            start_t = time.time()
            raw_text = ""
            p_tok = c_tok = t_tok = None
            parse_ok = exec_ok = goal_ok = valid_plan = False
            parsed_actions: List[str] = []
            error_type: Optional[str] = None
            first_fail_step_idx: Optional[int] = None
            first_fail_action: Optional[str] = None
            missing_preconditions: Optional[List[str]] = None

            domain_pddl = get_domain_pddl(rec)
            system_prompt, prompt = build_prompts(
                prompt_profile=args.prompt_profile,
                domain=dom,
                domain_pddl=domain_pddl,
                problem_pddl=rec["problem_pddl"],
            )

            try:
                if provider == "openai":
                    assert openai_client is not None
                    raw_text, p_tok, c_tok, t_tok = call_gpt4o_with_retry(
                        client=openai_client,
                        model=resolved_model,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        max_retries=args.max_retries,
                        retry_base_s=args.retry_base_seconds,
                    )
                else:
                    assert mistral_client is not None
                    raw_text, p_tok, c_tok, t_tok = call_mistral_with_retry(
                        client=mistral_client,
                        model=resolved_model,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        max_retries=args.max_retries,
                        retry_base_s=args.retry_base_seconds,
                    )

                parsed, parse_ok, parse_err = extract_actions(raw_text)
                parsed_actions = [x[2] for x in parsed]
                if not parse_ok:
                    error_type = parse_err
                else:
                    exec_ok, goal_ok, val_err, fail_diag = validate_plan(
                        domain=dom,
                        init_facts=rec["init_facts"],
                        goal_facts=rec["goal_facts"],
                        parsed_actions=parsed,
                    )
                    error_type = val_err
                    valid_plan = bool(exec_ok and goal_ok)
                    if fail_diag:
                        first_fail_step_idx = fail_diag.get("first_fail_step_idx")
                        first_fail_action = fail_diag.get("first_fail_action")
                        missing_preconditions = fail_diag.get("missing_preconditions")

            except Exception as e:
                error_type = "api_error"
                raw_text = f"API_ERROR: {type(e).__name__}: {e}"

            latency_ms = float((time.time() - start_t) * 1000.0)
            row = {
                "instance_id": iid,
                "domain": dom,
                "raw_response": raw_text,
                "parsed_actions": parsed_actions,
                "parse_ok": bool(parse_ok),
                "exec_ok": bool(exec_ok),
                "goal_ok": bool(goal_ok),
                "valid_plan": bool(valid_plan),
                "error_type": error_type,
                "first_fail_step_idx": first_fail_step_idx,
                "first_fail_action": first_fail_action,
                "missing_preconditions": missing_preconditions,
                "prompt_profile": args.prompt_profile,
                "latency_ms": latency_ms,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            status = "OK" if valid_plan else f"FAIL({error_type})"
            print(f"[{i:03d}/{len(remaining):03d}] iid={iid:4d} dom={dom:20s} -> {status}")

    final_rows = load_existing_rows(out_jsonl)
    final_rows = [r for r in final_rows if int(r.get("instance_id", -1)) in target_ids]
    summary = summarize_rows(
        final_rows,
        price_in_per_mtok=args.price_in_per_mtok,
        price_out_per_mtok=args.price_out_per_mtok,
    )
    summary["meta"] = {
        "provider": provider,
        "model": resolved_model,
        "prompt_profile": args.prompt_profile,
        "scope": args.scope,
        "data_dir": str(data_dir),
        "n_selected": len(records),
        "n_evaluated": len(final_rows),
        "instance_out": str(out_jsonl),
    }
    out_summary.write_text(json.dumps(summary, indent=2))

    print("\nSummary:")
    if summary.get("global"):
        g = summary["global"]
        print(
            "  Global: "
            f"valid={g['validity_rate']:.1%}, "
            f"parse={g['parse_rate']:.1%}, "
            f"exec={g['exec_rate']:.1%}, "
            f"goal={g['goal_rate']:.1%}, "
            f"tokens={g['tokens']['total_tokens']}"
        )
        if "estimated_cost_usd" in g:
            print(f"  Estimated cost: ${g['estimated_cost_usd']:.4f}")
    print(f"  Wrote summary -> {out_summary}")


if __name__ == "__main__":
    main()
