#!/usr/bin/env python3
"""
exp_urdf_chunked.py
===================
URDF chunked generation for qwen2.5:72b.

Tests whether the 72b model's URDF output-mode failure (outputting prose
instead of XML) is due to serialization instability rather than a
fundamental model limitation.

Conditions:
  ZERO_SHOT:  Standard single-prompt URDF generation (baseline, replicates Exp 4)
  ONE_SHOT:   Single URDF example in prompt (few-shot)
  CHUNKED:    Two-step generation:
                Step 1: Generate <robot> header + all <link> elements
                Step 2: Generate <joint> elements conditioned on Step 1 output

Usage:
    python3 exp_urdf_chunked.py \
        --ollama-host "$OLLAMA_HOST" \
        --model qwen2.5:72b \
        --samples 50 \
        --output-dir validator_paper_results/exp_urdf_chunked
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator_paper_exp1_taxonomy import build_ollama_caller

# ── URDF validation ───────────────────────────────────────────────────────────

def _has_xml(text: str) -> bool:
    return bool(re.search(r'<robot', text, re.IGNORECASE))

def _extract_urdf(raw: str) -> Optional[str]:
    raw = re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```(?:xml|urdf)?\s*', '', raw)
    raw = re.sub(r'```', '', raw)
    m = re.search(r'<robot[\s\S]*?</robot>', raw, re.IGNORECASE)
    return m.group(0) if m else None

def _urdf_parseable(xml: str) -> bool:
    if not xml: return False
    try: ET.fromstring(xml); return True
    except: return False

def _detect_urdf_errors(xml: str) -> List[str]:
    if not xml: return ['U_S1_NO_XML']
    errors = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ['U_S2_PARSE_FAIL']
    if root.tag.lower() != 'robot': errors.append('U_S3_NO_MODEL')
    links = list(root.findall('link'))
    if not links: errors.append('U_S4_NO_LINK')
    joints = list(root.findall('joint'))
    if not joints: errors.append('U_S5_NO_JOINT')
    # Check joint references
    link_names = {l.get('name','') for l in links}
    for j in joints:
        for ref_tag in ['parent','child']:
            ref = j.find(ref_tag)
            if ref is not None and ref.get('link','') not in link_names:
                errors.append('U_R1_BAD_JOINT_REF'); break
    # Check inertial
    for l in links:
        if l.find('inertial') is None:
            errors.append('U_P1_NO_INERTIAL'); break
    return errors

def _urdf_valid(xml: str) -> bool:
    return _urdf_parseable(xml) and not _detect_urdf_errors(xml)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYS_PROMPT = ("You are a robot engineer. Output only a complete valid URDF XML. "
               "No markdown. No explanation. Start with <robot name=")

TASK_PROMPT = {
    'urdf_arm': (
        "Generate a URDF for a 6-DOF robot arm with 7 links (base + 6 segments) "
        "and 6 revolute joints. Each link must have <inertial>, <visual>, and "
        "<collision> elements. Each joint must have valid <parent> and <child> links."
    ),
    'urdf_walker': (
        "Generate a URDF for a 4-legged walking robot with 9 links (body + 2 segments "
        "per leg) and 8 revolute joints. Each link must have <inertial> properties."
    ),
}

CHUNK1_PROMPT = {
    'urdf_arm': (
        "Generate only the <robot> opening tag and all <link> elements for a "
        "6-DOF robot arm. Include <inertial>, <visual>, <collision> for each link. "
        "Do NOT include joints yet. End with the last </link> tag."
    ),
    'urdf_walker': (
        "Generate only the <robot> opening tag and all <link> elements for a "
        "4-legged robot. Include <inertial>, <visual>, <collision> for each link. "
        "Do NOT include joints yet. End with the last </link> tag."
    ),
}

CHUNK2_PROMPT = {
    'urdf_arm': (
        "Now add all 6 revolute <joint> elements to the robot above. "
        "Each joint must reference exactly the link names defined above. "
        "Then close with </robot>. Output only the joint elements and </robot>."
    ),
    'urdf_walker': (
        "Now add all 8 revolute <joint> elements to the robot above. "
        "Each joint must reference exactly the link names defined above. "
        "Then close with </robot>. Output only the joint elements and </robot>."
    ),
}

ONESHOT_EXAMPLE = """Example URDF structure:
<robot name="example">
  <link name="base_link">
    <inertial><mass value="1.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <inertial><mass value="0.5"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><cylinder radius="0.02" length="0.2"/></geometry></visual>
    <collision><geometry><cylinder radius="0.02" length="0.2"/></geometry></collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
"""


# ── Experiment runner ─────────────────────────────────────────────────────────

@dataclass
class ChunkedTrial:
    task:               str
    condition:          str      # ZERO_SHOT | ONE_SHOT | CHUNKED
    trial:              int
    has_xml:            bool     # did model output any XML at all
    parseable:          bool     # XML is well-formed
    valid:              bool     # passes all URDF checks
    error_codes:        List[str]
    n_links:            int      # number of links in output
    n_joints:           int      # number of joints in output
    chunk1_ok:          bool     # (CHUNKED only) step 1 produced links
    chunk2_ok:          bool     # (CHUNKED only) step 2 produced joints
    latency_s:          float


def run_zero_shot(llm, task: str, trial: int) -> ChunkedTrial:
    t0 = time.time()
    try:
        raw = llm([{'role':'system','content':SYS_PROMPT},
                   {'role':'user',  'content':TASK_PROMPT[task]}])
    except Exception: raw = ''
    xml = _extract_urdf(raw)
    errs = _detect_urdf_errors(xml) if xml else ['U_S1_NO_XML']
    nl = len(ET.fromstring(xml).findall('link')) if xml and _urdf_parseable(xml) else 0
    nj = len(ET.fromstring(xml).findall('joint')) if xml and _urdf_parseable(xml) else 0
    return ChunkedTrial(task, 'ZERO_SHOT', trial,
                         has_xml=bool(xml), parseable=_urdf_parseable(xml) if xml else False,
                         valid=_urdf_valid(xml) if xml else False,
                         error_codes=errs, n_links=nl, n_joints=nj,
                         chunk1_ok=False, chunk2_ok=False,
                         latency_s=time.time()-t0)


def run_one_shot(llm, task: str, trial: int) -> ChunkedTrial:
    t0 = time.time()
    prompt = ONESHOT_EXAMPLE + '\n\nNow generate:\n' + TASK_PROMPT[task]
    try:
        raw = llm([{'role':'system','content':SYS_PROMPT},
                   {'role':'user',  'content':prompt}])
    except Exception: raw = ''
    xml = _extract_urdf(raw)
    errs = _detect_urdf_errors(xml) if xml else ['U_S1_NO_XML']
    nl = len(ET.fromstring(xml).findall('link')) if xml and _urdf_parseable(xml) else 0
    nj = len(ET.fromstring(xml).findall('joint')) if xml and _urdf_parseable(xml) else 0
    return ChunkedTrial(task, 'ONE_SHOT', trial,
                         has_xml=bool(xml), parseable=_urdf_parseable(xml) if xml else False,
                         valid=_urdf_valid(xml) if xml else False,
                         error_codes=errs, n_links=nl, n_joints=nj,
                         chunk1_ok=False, chunk2_ok=False,
                         latency_s=time.time()-t0)


def run_chunked(llm, task: str, trial: int) -> ChunkedTrial:
    t0 = time.time()

    # Step 1: links only
    try:
        raw1 = llm([{'role':'system','content':SYS_PROMPT},
                    {'role':'user',  'content':CHUNK1_PROMPT[task]}])
    except Exception: raw1 = ''
    chunk1_ok = _has_xml(raw1)

    # Extract declared link names from step 1
    link_names = []
    try:
        # Wrap in robot tag to parse
        test_xml = f'<robot name="tmp">{raw1}</robot>'
        root = ET.fromstring(test_xml)
        link_names = [l.get('name','') for l in root.findall('link') if l.get('name')]
    except Exception:
        pass

    # Step 2: joints conditioned on step 1
    joint_hint = ''
    if link_names:
        joint_hint = (f'\nDeclared links (use EXACTLY these names): '
                      f'{", ".join(link_names)}\n')
    try:
        raw2 = llm([
            {'role':'system',    'content':SYS_PROMPT},
            {'role':'user',      'content':CHUNK1_PROMPT[task]},
            {'role':'assistant', 'content':raw1},
            {'role':'user',      'content':joint_hint + CHUNK2_PROMPT[task]},
        ])
    except Exception: raw2 = ''
    chunk2_ok = bool(re.search(r'<joint', raw2, re.IGNORECASE))

    # Combine
    combined = raw1 + '\n' + raw2
    if not combined.strip().startswith('<robot'):
        combined = '<robot name="combined">' + combined
    if not combined.strip().endswith('</robot>'):
        combined = combined + '\n</robot>'
    xml = _extract_urdf(combined)
    errs = _detect_urdf_errors(xml) if xml else ['U_S1_NO_XML']
    nl = len(ET.fromstring(xml).findall('link'))  if xml and _urdf_parseable(xml) else 0
    nj = len(ET.fromstring(xml).findall('joint')) if xml and _urdf_parseable(xml) else 0
    return ChunkedTrial(task, 'CHUNKED', trial,
                         has_xml=bool(xml), parseable=_urdf_parseable(xml) if xml else False,
                         valid=_urdf_valid(xml) if xml else False,
                         error_codes=errs, n_links=nl, n_joints=nj,
                         chunk1_ok=chunk1_ok, chunk2_ok=chunk2_ok,
                         latency_s=time.time()-t0)


def report(trials: List[ChunkedTrial], output_dir: str):
    print('\n' + '='*70)
    print('URDF CHUNKED GENERATION RESULTS')
    print('='*70)

    tasks = sorted(set(t.task for t in trials))
    conditions = ['ZERO_SHOT','ONE_SHOT','CHUNKED']

    for task in tasks:
        print(f'\n  Task: {task}')
        print(f"  {'Condition':<12} {'XML%':>6} {'Parse%':>7} {'Valid%':>7} "
              f"{'R1%':>6} {'Links':>6} {'Joints':>7}")
        print(f'  {"-"*58}')
        for cond in conditions:
            ts = [t for t in trials if t.task == task and t.condition == cond]
            if not ts: continue
            xml_r   = np.mean([t.has_xml    for t in ts])
            parse_r = np.mean([t.parseable  for t in ts])
            valid_r = np.mean([t.valid      for t in ts])
            r1_r    = np.mean([any('R1' in e for e in t.error_codes) for t in ts])
            nl      = np.mean([t.n_links    for t in ts])
            nj      = np.mean([t.n_joints   for t in ts])
            print(f'  {cond:<12} {xml_r:>6.1%} {parse_r:>7.1%} {valid_r:>7.1%} '
                  f'{r1_r:>6.1%} {nl:>6.1f} {nj:>7.1f}')

    # Top error codes per condition
    print('\n  Top errors by condition (CHUNKED):')
    from collections import Counter
    chunked = [t for t in trials if t.condition == 'CHUNKED']
    all_errs = [e for t in chunked for e in t.error_codes]
    for code, cnt in Counter(all_errs).most_common(5):
        print(f'    {code}: {cnt}')

    # Conclusion
    zero_v  = np.mean([t.valid for t in trials if t.condition == 'ZERO_SHOT'])
    chunk_v = np.mean([t.valid for t in trials if t.condition == 'CHUNKED'])
    print(f'\n  ZERO_SHOT valid: {zero_v:.1%}  →  CHUNKED valid: {chunk_v:.1%}')
    if chunk_v > zero_v + 0.10:
        print('  → Chunked generation significantly improves validity')
        print('    → Failure IS serialization instability, not model limitation')
        print('    → Mitigation: two-step generation reduces output-mode failures')
    else:
        print('  → Chunked generation does not significantly improve validity')
        print('    → Failure is likely a deeper instruction-following issue')
        print('    → Mitigation: few-shot exemplars (ONE_SHOT) more effective')

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'urdf_chunked_results.json')
    with open(path, 'w') as f:
        json.dump([asdict(t) for t in trials], f, indent=2)
    print(f'\n  Results → {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ollama-host', required=True)
    parser.add_argument('--model',       default='qwen2.5:72b')
    parser.add_argument('--samples',     type=int, default=50)
    parser.add_argument('--output-dir',
        default='validator_paper_results/exp_urdf_chunked')
    parser.add_argument('--tasks', default='urdf_arm,urdf_walker')
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',')]
    llm   = build_ollama_caller(args.ollama_host, args.model)
    all_trials: List[ChunkedTrial] = []
    total = len(tasks) * args.samples * 3  # 3 conditions

    for task in tasks:
        print(f'\n=== Task: {task} ===')
        for i in range(args.samples):
            for cond, fn in [('ZERO_SHOT', run_zero_shot),
                              ('ONE_SHOT',  run_one_shot),
                              ('CHUNKED',   run_chunked)]:
                t = fn(llm, task, i)
                all_trials.append(t)
                done = len(all_trials)
                print(f'  [{done:3d}/{total}] {cond:<10} xml={t.has_xml} '
                      f'valid={t.valid} errors={t.error_codes[:2]}')

    report(all_trials, args.output_dir)


if __name__ == '__main__':
    main()
