#!/usr/bin/env python3
"""
eval_depot_n200.py

ARC routing evaluation on Depot using Qwen2.5-7B via Ollama.
Uses an existing ARC checkpoint; no retraining.

Key fixes:
- Uses actual dataset filenames in data/planning/
- Uses actual checkpoint directory checkpoints_planning/
- Supports --ckpt properly
- Supports both --label success and --label n_steps
- Uses ARC naming in user-facing output
"""

import argparse
import json
import subprocess
from pathlib import Path
import importlib.util

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Import ARC model class from plan_step3_guru.py
# ---------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "step3", Path(__file__).parent / "plan_step3_guru.py"
)
step3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step3)

PlanningARC = step3.PlanningGURU


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/planning")
CKPT_DIR = Path("checkpoints_planning")
RESULTS_DIR = Path("results/planning")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = """You are a PDDL planning expert.
Solve the following PDDL planning problem using ONLY the actions defined in the domain.
Output ONLY the plan. Each line must be exactly one action in this format:
(action-name arg1 arg2 ...)
Use parentheses. No quotes. No numbering. No explanation. No markdown.

Domain:
{domain}

Problem:
{problem}
"""


# ---------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------
def call_ollama(problem_pddl: str, model_name: str, max_tokens: int, temperature: float, domain_pddl: str = "") -> str:
    payload = {
        "model": model_name,
        "prompt": PROMPT_TEMPLATE.format(domain=domain_pddl, problem=problem_pddl),
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            "http://10.139.126.18:11434/api/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    try:
        return json.loads(result.stdout).get("response", "").strip()
    except Exception:
        return ""


def parse_plan(raw: str) -> list[str]:
    import re
    actions = []
    for line in raw.splitlines():
        s = line.strip().lower()
        # Standard PDDL format: (action arg1 arg2)
        if s.startswith("(") and s.endswith(")"):
            actions.append(s)
            continue
        # Fallback: ["action" "arg1" "arg2"] or "action" "arg1" tokens
        tokens = re.findall(r'[\w][\w\-]*', s)
        if len(tokens) >= 1 and not any(w in tokens[0] for w in ("the","a","and","or","is","to","of")):
            actions.append("(" + " ".join(tokens) + ")")
    return actions


def validate_plan(plan: list[str], domain_pddl: str, problem_pddl: str) -> bool:
    """Forward-chain the plan through the PDDL state to check true validity."""
    if not plan:
        return False
    import re

    def parse_facts(block):
        return set(re.findall(r"\(([^()]+)\)", block))

    def parse_section(pddl, tag):
        m = re.search(rf"\(:{tag}\s+(.*?)\)\s*(?=\(:|$)", pddl, re.S | re.I)
        return m.group(1).strip() if m else ""

    # Parse init facts and goal facts from problem PDDL
    init_block  = parse_section(problem_pddl, "init")
    goal_block  = parse_section(problem_pddl, "goal")
    init_facts  = parse_facts("(" + init_block + ")")
    goal_facts  = parse_facts("(" + goal_block.replace("and","") + ")")

    # Parse actions from domain PDDL using paren-balanced extractor
    actions = {}
    def extract_balanced(s, start):
        """Extract balanced paren expression starting at index of first '('."""
        idx = s.find("(", start)
        if idx < 0: return "", start
        depth, i = 0, idx
        while i < len(s):
            if s[i] == "(": depth += 1
            elif s[i] == ")":
                depth -= 1
                if depth == 0: return s[idx:i+1], i+1
            i += 1
        return s[idx:], len(s)

    for m in re.finditer(r"\(:action\s+(\S+)(.*?)(?=\(:action|$)", domain_pddl, re.S | re.I):
        name = m.group(1).lower()
        body = m.group(2)
        pm = re.search(r":parameters\s*\(([^)]*)\)", body, re.I)
        if not pm: continue
        params = pm.group(1).split()
        pre_start = body.lower().find(":precondition")
        eff_start = body.lower().find(":effect")
        if pre_start < 0 or eff_start < 0: continue
        pre_str, _ = extract_balanced(body, pre_start)
        eff_str, _ = extract_balanced(body, eff_start)
        actions[name] = {"params": params, "pre": pre_str, "eff": eff_str}

    if not actions:
        return True  # can't validate without domain

    def ground(template, param_names, arg_values):
        result = template
        for p, a in zip(param_names, arg_values):
            # ?param boundaries: preceded by non-word or start, followed by non-word or end
            result = re.sub(rf"(?<![\w?]){re.escape(p)}(?![\w-])", a, result)
        return result

    def get_literals(grounded_str):
        pos = set(re.findall(r"(?<!not\s)(?<!not)\(([^()]+)\)", grounded_str))
        neg = set(re.findall(r"\(not\s+\(([^()]+)\)\)", grounded_str))
        return pos - neg, neg

    state = set(init_facts)

    for step in plan:
        step = step.strip().strip("()")
        # Strip any parens wrapping individual args e.g. (move (room1) (room2))
        # Apply repeatedly until stable (handles nested parens)
        prev = None
        while prev != step:
            prev = step
            step = re.sub(r"\(([^()]+)\)", r"\1", step)
        parts = step.split()
        if not parts:
            return False
        name = parts[0].lower()
        args = parts[1:]

        if name not in actions:
            return False

        act = actions[name]
        if len(args) != len(act["params"]):
            return False

        grounded_pre = ground(act["pre"], act["params"], args).lower()
        pre_pos, pre_neg = get_literals(grounded_pre)

        if not pre_pos.issubset(state):
            return False
        if pre_neg & state:
            return False

        grounded_eff = ground(act["eff"], act["params"], args).lower()
        eff_pos, eff_neg = get_literals(grounded_eff)
        state = (state - eff_neg) | eff_pos

    return goal_facts.issubset(state)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data():
    Xsurf = np.load(DATA_DIR / "X_surf.npy")
    Xfm = np.load(DATA_DIR / "X_fm.npy")
    ysuccess = np.load(DATA_DIR / "y_success.npy")
    ynsteps = np.load(DATA_DIR / "y_nsteps.npy").astype(float)
    tasktypes = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry = json.loads((DATA_DIR / "registry.json").read_text())
    return Xsurf, Xfm, ysuccess, ynsteps, tasktypes, registry


def get_split_tasks(registry, split_name: str):
    splits = registry["splits"]
    if split_name in splits:
        return splits[split_name]["tasks"]
    alt = split_name.replace("-", "_") if "-" in split_name else split_name.replace("_", "-")
    if alt in splits:
        return splits[alt]["tasks"]
    raise KeyError(f"Could not find split '{split_name}' in registry['splits']")


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------
def default_ckpt_for_label(label: str) -> Path:
    if label == "n_steps":
        return CKPT_DIR / "guru_n_steps.pt"
    if label == "success":
        return CKPT_DIR / "guru_success.pt"
    raise ValueError(f"Unsupported label: {label}")


def load_domain_problems(domain: str, n_target: int):
    episodes = json.loads((DATA_DIR / "episodes.json").read_text())
    tasktypes = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    mask = tasktypes == domain
    idxs = np.where(mask)[0][:n_target]

    if len(idxs) == 0:
        raise FileNotFoundError(f"No episodes found for domain={domain}")

    problems = []
    for idx in idxs:
        ep = episodes[idx]
        domain_pddl = ep.get("domainpddl", "")
        problem_pddl = ep.get("problempddl") or ep.get("problem_pddl") or ep.get("pddl", "")
        problems.append({
            "id": str(ep.get("instanceid", idx)),
            "pddl": problem_pddl,
            "domain": domain_pddl,
        })

    return problems, problems[0]["domain"]


# ---------------------------------------------------------------------
# ARC scoring
# ---------------------------------------------------------------------
def fit_residual(Xptr, Xetr, Xpte, Xete):
    ncomp = max(2, min(20, Xptr.shape[0] // 10, Xptr.shape[1]))
    pred = Pipeline(
        [
            ("pca", PCA(n_components=ncomp)),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pred.fit(Xptr, Xetr)
    return Xetr - pred.predict(Xptr), Xete - pred.predict(Xpte)


def arc_score_domain(model, domain, Xsurf, Xfm, ynsteps, tasktypes, train_domains):
    mask = tasktypes == domain
    train_mask = np.isin(tasktypes, train_domains)

    Xsurf_q = Xsurf[mask]
    Xfm_q = Xfm[mask]
    Xsurf_s = Xsurf[train_mask]
    Xfm_s = Xfm[train_mask]

    scs = StandardScaler().fit(Xsurf_s)
    sce = StandardScaler().fit(Xfm_s)

    Xs_n = scs.transform(Xsurf_s)
    Xq_n = scs.transform(Xsurf_q)
    Xe_n = sce.transform(Xfm_s)
    Xeq_n = sce.transform(Xfm_q)

    ncomp = max(2, min(20, len(Xs_n) // 10, Xs_n.shape[1]))
    pred = Pipeline(
        [
            ("pca", PCA(n_components=ncomp)),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pred.fit(Xs_n, Xe_n)

    Xr_s = Xe_n - pred.predict(Xs_n)
    Xr_q = Xeq_n - pred.predict(Xq_n)

    rng = np.random.default_rng(42)
    sup_idx = rng.choice(len(Xsurf_s), min(60, len(Xsurf_s)), replace=False)

    Ssn = Xs_n[sup_idx]
    SV = np.hstack([Ssn, Xr_s[sup_idx]])

    Ssn_t = torch.FloatTensor(Ssn).to(DEVICE)
    SV_t = torch.FloatTensor(SV).to(DEVICE)

    scores = []
    with torch.no_grad():
        for i in range(len(Xsurf_q)):
            out, _, _ = model(
                torch.FloatTensor(Xq_n[i]).to(DEVICE),
                torch.FloatTensor(Xeq_n[i]).to(DEVICE),
                torch.FloatTensor(Xr_q[i]).to(DEVICE),
                Ssn_t,
                None,
                SV_t,
                head="reg",
            )
            scores.append(float(out.cpu()))

    return np.array(scores), mask


def routing_validity(scores, llm_valid, budget=0.5):
    n = len(scores)
    n_llm = int(n * budget)

    easy = np.argsort(scores)[:n_llm]
    hard = np.argsort(scores)[n_llm:]

    return float(llm_valid[easy].sum() + len(hard)) / n


# ---------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------
def load_domain_problems(domain: str, n_target: int):
    problem_dir = DATA_DIR / domain / "problems"
    domain_file = DATA_DIR / domain / "domain.pddl"

    if not problem_dir.exists():
        raise FileNotFoundError(f"Problems not found at {problem_dir}")

    domain_pddl = domain_file.read_text() if domain_file.exists() else ""
    problem_files = sorted(problem_dir.glob("*.pddl"))[:n_target]

    problems = [
        {
            "id": pf.stem,
            "pddl": pf.read_text(),
            "domain": domain_pddl,
        }
        for pf in problem_files
    ]
    return problems, domain_pddl


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def run_routing_eval(
    domain: str,
    n_target: int,
    model,
    Xsurf,
    Xfm,
    ynsteps,
    tasktypes,
    train_domains,
    ollama_model: str,
    max_tokens: int,
    temperature: float,
):
    print(f"\n{'=' * 60}")
    print(f"Domain: {domain.upper()}  N={n_target}  LLM={ollama_model}")
    print(f"{'=' * 60}")

    arc_scores, mask = arc_score_domain(
        model, domain, Xsurf, Xfm, ynsteps, tasktypes, train_domains
    )

    nobj_scores = Xsurf[mask][:, 0]

    problems, _ = load_domain_problems(domain, n_target)
    n_eval = min(len(problems), int(mask.sum()), n_target)
    print(f"Instances available: {int(mask.sum())}  evaluating: {n_eval}")

    llm_valid = np.zeros(n_eval, dtype=int)
    llm_raw = []
    n_empty = 0

    for i, prob in enumerate(problems[:n_eval]):
        raw = call_ollama(prob["pddl"], ollama_model, max_tokens, temperature, domain_pddl=prob["domain"])
        plan = parse_plan(raw)
        valid = int(validate_plan(plan, prob["domain"], prob["pddl"]))

        llm_valid[i] = valid
        llm_raw.append(
            {
                "id": prob["id"],
                "valid": valid,
                "plan": plan,
                "raw": raw[:300],
            }
        )

        if not plan:
            n_empty += 1

        if (i + 1) % 20 == 0:
            print(
                f"  [{i+1}/{n_eval}]  "
                f"validity: {llm_valid[:i+1].mean():.1%}  "
                f"empty: {n_empty}"
            )

    llm_acc = float(llm_valid.mean()) if n_eval > 0 else 0.0
    arc_v = routing_validity(arc_scores[:n_eval], llm_valid)
    nobj_v = routing_validity(nobj_scores[:n_eval], llm_valid)
    rand_v = routing_validity(
        np.random.default_rng(42).permutation(n_eval).astype(float),
        llm_valid,
    )

    rng = np.random.default_rng(42)
    deltas_ar, deltas_an = [], []

    for _ in range(1000):
        idx = rng.choice(n_eval, n_eval, replace=True)
        a = routing_validity(arc_scores[idx], llm_valid[idx])
        r = routing_validity(rng.permutation(n_eval).astype(float), llm_valid[idx])
        o = routing_validity(nobj_scores[idx], llm_valid[idx])
        deltas_ar.append(a - r)
        deltas_an.append(a - o)

    ci_ar = (float(np.percentile(deltas_ar, 2.5)), float(np.percentile(deltas_ar, 97.5)))
    ci_an = (float(np.percentile(deltas_an, 2.5)), float(np.percentile(deltas_an, 97.5)))

    rho_arc, p_arc = spearmanr(arc_scores[:n_eval], llm_valid)
    rho_nobj, p_nobj = spearmanr(nobj_scores[:n_eval], llm_valid)

    print(f"\n{'─' * 60}")
    print(f"LLM accuracy  : {llm_acc:.1%}  empty: {n_empty}/{n_eval}")
    print(f"ARC routing   : {arc_v:.1%}  Δ vs random: {arc_v-rand_v:+.1%}  CI: ({ci_ar[0]:+.1%}, {ci_ar[1]:+.1%})")
    print(f"Obj-count     : {nobj_v:.1%}  Δ vs random: {nobj_v-rand_v:+.1%}")
    print(f"Random        : {rand_v:.1%}")
    print(f"ARC Δ vs nobj : {arc_v-nobj_v:+.1%}  CI: ({ci_an[0]:+.1%}, {ci_an[1]:+.1%})")
    print(f"ρ(ARC,valid)  : {rho_arc:+.3f}  p={p_arc:.4f}")
    print(f"ρ(nobj,valid) : {rho_nobj:+.3f}  p={p_nobj:.4f}")

    if llm_acc < 0.15:
        print("WARNING: LLM accuracy < 15% — routing may not help in this regime.")

    result = {
        "domain": domain,
        "n_eval": n_eval,
        "llm_model": ollama_model,
        "llm_accuracy": llm_acc,
        "n_empty": int(n_empty),
        "routing": {
            "arc": float(arc_v),
            "obj_count": float(nobj_v),
            "random": float(rand_v),
            "delta_arc_over_random": float(arc_v - rand_v),
            "delta_arc_over_nobj": float(arc_v - nobj_v),
            "ci_arc_over_random": list(ci_ar),
            "ci_arc_over_nobj": list(ci_an),
        },
        "spearman": {
            "arc_vs_valid": {"rho": float(rho_arc), "p": float(p_arc)},
            "nobj_vs_valid": {"rho": float(rho_nobj), "p": float(p_nobj)},
        },
        "llm_raw": llm_raw,
    }

    out_path = RESULTS_DIR / f"{domain}_n{n_eval}_routing.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {out_path}")

    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="depot")
    parser.add_argument("--n_target", type=int, default=200)
    parser.add_argument("--label", choices=["success", "n_steps"], default="n_steps")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--ollama_model", default="qwen2.5:7b")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()



def load_checkpoint(label="n_steps", ckpt_path=None):
    from pathlib import Path
    ckpt_path = Path(ckpt_path) if ckpt_path is not None else Path(f"checkpoints_planning/arc_{label}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    fm_key   = "query_enc.net.0.weight"
    surf_key = "key_enc.net.0.weight"
    if fm_key not in state or surf_key not in state:
        raise KeyError(f"Cannot infer dims. Keys: {list(state.keys())[:20]}")
    fm_dim   = state[fm_key].shape[1]
    surf_dim = state[surf_key].shape[1]
    model = PlanningARC(surf_dim=surf_dim, fm_dim=fm_dim).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"ARC checkpoint loaded from: {ckpt_path}")
    return model, ckpt_path

def main():
    args = parse_args()

    Xsurf, Xfm, ysuccess, ynsteps, tasktypes, registry = load_data()
    train_domains = get_split_tasks(registry, "meta-train")

    model, ckpt_path = load_checkpoint(label=args.label, ckpt_path=args.ckpt)
    print(f"ARC checkpoint loaded from: {ckpt_path}")
    print(f"Device: {DEVICE}")
    print(f"Train domains: {train_domains}")

    run_routing_eval(
        domain=args.domain,
        n_target=args.n_target,
        model=model,
        Xsurf=Xsurf,
        Xfm=Xfm,
        ynsteps=ynsteps,
        tasktypes=tasktypes,
        train_domains=train_domains,
        ollama_model=args.ollama_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()