#!/usr/bin/env python3
"""
exp_hint_cd.py
==============
Hint-CD baseline: reference-dictionary-aware generation.

Since Ollama doesn't expose logit biases, we implement a proxy:
  HINT-CD: generate full XML, then apply SP2 ONLY (reference repair)
           as the sole post-processing step — no SP3/SP4/SP5.
  This directly answers: "what if constrained decoding tracked joint names?"
  and is equivalent to a reference-dictionary baseline.

Also computes the theoretical Hint-CD ceiling analytically from Exp 1 data.

Conditions (all use same LLM, same prompt):
  BASELINE_RAW:  No post-processing (already have this from Exp 3)
  HINT_CD:       SP2 only (reference repair, no semantic/physics/constraint fix)
  LAYERED_VAL:   Full SP1-SP5 (already have this from Exp 3)

Expected result:
  HINT_CD reduces R1 errors by ~69% but leaves E/C/P failures untouched.
  Feasibility: ~15% + (34.3% × 69%) = ~39%
  Validator still needed for the remaining 57pp gap.

Usage:
    python3 exp_hint_cd.py \
        --ollama-host "$OLLAMA_HOST" \
        --model qwen2.5:7b \
        --samples 50 \
        --output-dir validator_paper_results/exp_hint_cd
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mujoco; HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from validator_paper_exp1_taxonomy import (
    build_ollama_caller, TASK_PROMPTS, SYSTEM_PROMPTS, detect_errors
)
from mjcf_autopatcher import patch_xml as _full_patch

# ─────────────────────────────────────────────────────────────────────────────
# SP2 only (reference repair, no other strategies)
# ─────────────────────────────────────────────────────────────────────────────

def sp2_only(xml: str) -> tuple[str, int]:
    """Apply only SP2 (reference name remapping). Return (repaired_xml, n_fixes)."""
    if not xml:
        return xml, 0
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return xml, 0  # S2_MALFORMED — SP2 can't help

    wb  = root.find('worldbody')
    act = root.find('actuator')
    if wb is None or act is None:
        return xml, 0

    # Extract all declared joint names
    joint_names = {j.get('name', '') for j in wb.iter('joint') if j.get('name')}
    if not joint_names:
        return xml, 0

    fixes = 0
    for motor in act.iter('motor'):
        ref = motor.get('joint', '')
        if ref and ref not in joint_names:
            # Find closest by normalised edit distance
            best = min(joint_names,
                       key=lambda j: _edit_dist(ref, j) / max(len(ref), len(j), 1))
            dist = _edit_dist(ref, best) / max(len(ref), len(best), 1)
            if dist <= 0.5:
                motor.set('joint', best)
                fixes += 1

    return ET.tostring(root, encoding='unicode'), fixes


def _edit_dist(a: str, b: str) -> int:
    """Levenshtein distance."""
    if a == b: return 0
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            prev, dp[j] = dp[j], (prev if a[i-1]==b[j-1]
                                   else 1 + min(prev, dp[j], dp[j-1]))
    return dp[n]


def _extract(raw: str) -> Optional[str]:
    raw = re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```(?:xml|mujoco)?\s*', '', raw)
    raw = re.sub(r'```', '', raw)
    m = re.search(r'<mujoco[\s\S]*?</mujoco>', raw, re.IGNORECASE)
    return m.group(0) if m else None


def _is_valid(xml: str) -> bool:
    if not HAS_MUJOCO or not xml: return False
    try:
        m = mujoco.MjModel.from_xml_string(xml)
        return m.nu > 0 and m.nbody > 1
    except Exception:
        return False


def _get_errors(xml: str, task: str) -> List[str]:
    try:
        _, errs = detect_errors(xml, 'probe', task, 0)
        return [e.get('error_code', str(e)) if isinstance(e, dict) else str(e)
                for e in errs]
    except Exception:
        return []


@dataclass
class HintCDTrial:
    task:           str
    trial:          int
    # Raw
    raw_feasible:   bool
    raw_errors:     List[str]
    has_r1_raw:     bool
    # Hint-CD (SP2 only)
    hintcd_feasible:bool
    hintcd_errors:  List[str]
    has_r1_hintcd:  bool
    n_refs_fixed:   int
    # Full validator (SP1-SP5)
    full_feasible:  bool
    # Timing
    latency_s:      float


def run_trial(llm, task: str, trial: int) -> HintCDTrial:
    t0 = time.time()
    sys_p = SYSTEM_PROMPTS.get('unconstrained',
        'Output only complete valid MuJoCo XML. No explanation. No markdown.')
    usr_p = TASK_PROMPTS[task]

    try:
        raw = llm([{'role':'system','content':sys_p},
                   {'role':'user',  'content':usr_p}])
    except Exception:
        raw = ''

    xml = _extract(raw) or ''

    # Raw
    raw_valid  = _is_valid(xml)
    raw_errors = _get_errors(xml, task) if xml else ['S1_NO_XML']
    has_r1_raw = any('R1' in e for e in raw_errors)

    # Hint-CD: SP2 only
    hintcd_xml, n_fixed = sp2_only(xml) if xml else ('', 0)
    hintcd_valid  = _is_valid(hintcd_xml)
    hintcd_errors = _get_errors(hintcd_xml, task) if hintcd_xml else raw_errors
    has_r1_hintcd = any('R1' in e for e in hintcd_errors)

    # Full validator
    full_xml, _ = _full_patch(xml, task_name=task) if xml else ('', None)
    full_valid  = _is_valid(full_xml) if full_xml else False

    return HintCDTrial(
        task=task, trial=trial,
        raw_feasible=raw_valid,   raw_errors=raw_errors,   has_r1_raw=has_r1_raw,
        hintcd_feasible=hintcd_valid, hintcd_errors=hintcd_errors,
        has_r1_hintcd=has_r1_hintcd, n_refs_fixed=n_fixed,
        full_feasible=full_valid,
        latency_s=time.time()-t0,
    )


def report(trials: List[HintCDTrial], output_dir: str) -> None:
    print('\n' + '='*70)
    print('HINT-CD BASELINE RESULTS')
    print('='*70)

    tasks = sorted(set(t.task for t in trials))
    for task in tasks:
        ts = [t for t in trials if t.task == task]
        raw_f  = np.mean([t.raw_feasible    for t in ts])
        hcd_f  = np.mean([t.hintcd_feasible for t in ts])
        full_f = np.mean([t.full_feasible   for t in ts])
        r1_raw = np.mean([t.has_r1_raw      for t in ts])
        r1_hcd = np.mean([t.has_r1_hintcd   for t in ts])
        r1_red = (r1_raw - r1_hcd) / max(r1_raw, 0.001)
        print(f'\n  {task}:')
        print(f'    RAW feasibility:     {raw_f:.1%}')
        print(f'    HINT-CD feasibility: {hcd_f:.1%}  '
              f'(+{(hcd_f-raw_f)*100:.1f}pp over raw)')
        print(f'    FULL-VAL feasibility:{full_f:.1%}  '
              f'(+{(full_f-hcd_f)*100:.1f}pp over Hint-CD)')
        print(f'    R1 rate: raw={r1_raw:.1%} → hint-cd={r1_hcd:.1%} '
              f'({r1_red:.0%} reduction)')

    # Aggregate
    raw_f  = np.mean([t.raw_feasible    for t in trials])
    hcd_f  = np.mean([t.hintcd_feasible for t in trials])
    full_f = np.mean([t.full_feasible   for t in trials])
    r1_raw = np.mean([t.has_r1_raw      for t in trials])
    r1_hcd = np.mean([t.has_r1_hintcd   for t in trials])

    print(f'\n  AVERAGE (all tasks):')
    print(f'    RAW:     {raw_f:.1%}')
    print(f'    HINT-CD: {hcd_f:.1%}  (+{(hcd_f-raw_f)*100:.1f}pp)')
    print(f'    FULL-VAL:{full_f:.1%}  (+{(full_f-hcd_f)*100:.1f}pp)')
    print(f'    Hint-CD closes {(hcd_f-raw_f)/(full_f-raw_f)*100:.0f}% of the gap; '
          f'validator covers the remaining {(full_f-hcd_f)/(full_f-raw_f)*100:.0f}%')

    # Theoretical ceiling from Exp 1
    print(f'\n  THEORETICAL HINT-CD CEILING (from Exp 1 numbers):')
    r1_rate  = 0.343
    sp2_rec  = 0.69
    raw_base = 0.15
    theory   = raw_base + (1 - raw_base) * r1_rate * sp2_rec
    print(f'    = raw_feas + (1-raw_feas) × R1_rate × SP2_recovery')
    print(f'    = {raw_base:.0%} + {(1-raw_base):.0%} × {r1_rate:.1%} × {sp2_rec:.0%}')
    print(f'    = {theory:.1%}  (vs LAYERED_VALIDATOR 96%)')
    print(f'    Gap: {(0.96-theory)*100:.1f}pp — covered by SP3/SP4/SP5')

    # LaTeX table
    print(f'\n  LaTeX row for Table 3:')
    print(f'  Hint-CD (reference dict.) & {hcd_f:.0%} & 1.0 & '
          f'{full_f:.0%} & 1.3 \\\\')
    print(f'  \\quad [R1 reduction: {r1_raw:.0%}→{r1_hcd:.0%}] & & & &')

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'hint_cd_results.json')
    with open(path, 'w') as f:
        json.dump([asdict(t) for t in trials], f, indent=2)
    print(f'\n  Results → {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ollama-host', required=True)
    parser.add_argument('--model',       default='qwen2.5:7b')
    parser.add_argument('--samples',     type=int, default=50)
    parser.add_argument('--output-dir',  default='validator_paper_results/exp_hint_cd')
    parser.add_argument('--tasks',
        default='locomotion_forward,jumping,swimming,box_pushing,crawling')
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',')]
    llm   = build_ollama_caller(args.ollama_host, args.model)
    all_trials: List[HintCDTrial] = []
    total = len(tasks) * args.samples

    for task in tasks:
        print(f'\n=== Task: {task} ===')
        for i in range(args.samples):
            t = run_trial(llm, task, i)
            all_trials.append(t)
            done = len(all_trials)
            print(f'  [{done:3d}/{total}] raw={t.raw_feasible} '
                  f'hintcd={t.hintcd_feasible} full={t.full_feasible} '
                  f'refs_fixed={t.n_refs_fixed} R1_gone={t.has_r1_raw and not t.has_r1_hintcd}')

    report(all_trials, args.output_dir)


if __name__ == '__main__':
    main()
