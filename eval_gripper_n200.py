#!/usr/bin/env python3
"""
eval_gripper_n200.py
Gripper routing evaluation N=200 using ARC checkpoint + Ollama LLM.
Zero-shot: gripper is the meta-val domain, not seen during ARC training.
"""

import argparse
import json
import re
import subprocess
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn

# ── Constants ────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR    = Path("data/planning")
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

# ── ARC model (mirrors plan_step3_guru.py architecture) ──────────
class QueryMLP(nn.Module):
    def __init__(self, fm_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fm_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, d_model), nn.LayerNorm(d_model))
    def forward(self, x): return self.net(x)

class KeyMLP(nn.Module):
    def __init__(self, surf_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(surf_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, d_model), nn.LayerNorm(d_model))
    def forward(self, x): return self.net(x)

class PlanningARC(nn.Module):
    def __init__(self, surf_dim, fm_dim, d_model=128):
        super().__init__()
        self.scale     = d_model ** -0.5
        self.log_temp  = nn.Parameter(torch.zeros(1))
        self.query_enc = QueryMLP(fm_dim, d_model)
        self.key_enc   = KeyMLP(surf_dim, d_model)
        self.value_proj = nn.Sequential(
            nn.Linear(surf_dim + fm_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(surf_dim + d_model + fm_dim, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05))
        self.head_cls = nn.Linear(128, 2)
        self.head_reg = nn.Linear(128, 1)

    def attend(self, q_fm, S_surf, S_V):
        single = q_fm.dim() == 1
        if single: q_fm = q_fm.unsqueeze(0)
        q     = self.query_enc(q_fm)
        k     = self.key_enc(S_surf)
        v     = self.value_proj(S_V)
        temp  = torch.exp(-self.log_temp.clamp(-2.3, 2.3))
        alpha = torch.softmax(torch.matmul(q, k.T) * self.scale * temp, dim=-1)
        z     = torch.matmul(alpha, v)
        if single: z, alpha = z.squeeze(0), alpha.squeeze(0)
        return z, alpha

    def forward(self, q_surf, q_fm, q_resid, S_surf, S_fm, S_V, head="cls"):
        z, alpha = self.attend(q_fm, S_surf, S_V)
        fused    = torch.cat([q_surf, z, q_resid], dim=-1)
        feats    = self.fusion(fused)
        if head == "cls":
            return self.head_cls(feats), feats, alpha
        else:
            return self.head_reg(feats).squeeze(-1), feats, alpha


# ── Checkpoint loader ────────────────────────────────────────────
def load_checkpoint(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck    = torch.load(ckpt_path, map_location=DEVICE)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    fm_dim   = state["query_enc.net.0.weight"].shape[1]
    surf_dim = state["key_enc.net.0.weight"].shape[1]
    model    = PlanningARC(surf_dim=surf_dim, fm_dim=fm_dim).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"ARC checkpoint loaded from: {ckpt_path}")
    return model


# ── Data loader ──────────────────────────────────────────────────
def load_data():
    Xsurf     = np.load(DATA_DIR / "X_surf.npy")
    Xfm       = np.load(DATA_DIR / "X_fm.npy")
    ysuccess  = np.load(DATA_DIR / "y_success.npy")
    ynsteps   = np.load(DATA_DIR / "y_nsteps.npy").astype(float)
    tasktypes = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry  = json.loads((DATA_DIR / "registry.json").read_text())
    return Xsurf, Xfm, ysuccess, ynsteps, tasktypes, registry


# ── ARC scoring ──────────────────────────────────────────────────
def arc_score_domain(model, domain, Xsurf, Xfm, ynsteps, tasktypes, train_domains):
    mask       = tasktypes == domain
    train_mask = np.isin(tasktypes, train_domains)

    Xsurf_q = Xsurf[mask];      Xfm_q = Xfm[mask]
    Xsurf_s = Xsurf[train_mask]; Xfm_s = Xfm[train_mask]
    y_s     = ynsteps[train_mask]

    scs = StandardScaler().fit(Xsurf_s)
    sce = StandardScaler().fit(Xfm_s)
    Xs_n  = scs.transform(Xsurf_s);  Xq_n  = scs.transform(Xsurf_q)
    Xe_n  = sce.transform(Xfm_s);    Xeq_n = sce.transform(Xfm_q)

    ncomp = max(2, min(20, len(Xs_n) // 10, Xs_n.shape[1]))
    pred  = Pipeline([("pca", PCA(n_components=ncomp)),
                      ("ridge", Ridge(alpha=1.0))])
    pred.fit(Xs_n, Xe_n)
    Xr_s = Xe_n  - pred.predict(Xs_n)
    Xr_q = Xeq_n - pred.predict(Xq_n)

    rng     = np.random.default_rng(42)
    sup_idx = rng.choice(len(Xsurf_s), min(60, len(Xsurf_s)), replace=False)
    Ssn = Xs_n[sup_idx]
    SV  = np.hstack([Ssn, Xr_s[sup_idx]])

    Ssn_t = torch.FloatTensor(Ssn).to(DEVICE)
    SV_t  = torch.FloatTensor(SV).to(DEVICE)

    scores = []
    with torch.no_grad():
        for i in range(len(Xsurf_q)):
            q_s  = torch.FloatTensor(Xq_n[i]).to(DEVICE)
            q_fm = torch.FloatTensor(Xeq_n[i]).to(DEVICE)
            q_r  = torch.FloatTensor(Xr_q[i]).to(DEVICE)
            out, _, _ = model(q_s, q_fm, q_r, Ssn_t, None, SV_t, head="reg")
            scores.append(float(out.cpu()))
    return np.array(scores), mask


# ── Ollama call ──────────────────────────────────────────────────
def call_ollama(problem_pddl: str, domain_pddl: str,
                model_name: str, max_tokens: int, temperature: float,
                ollama_host: str) -> str:
    payload = {
        "model": model_name,
        "prompt": PROMPT_TEMPLATE.format(domain=domain_pddl, problem=problem_pddl),
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    result = subprocess.run(
        ["curl", "-s", "-X", "POST",
         f"http://{ollama_host}/api/generate",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, check=False)
    try:
        return json.loads(result.stdout).get("response", "").strip()
    except Exception:
        return ""


# ── Plan parser ──────────────────────────────────────────────────
def parse_plan(raw: str) -> list:
    actions = []
    for line in raw.splitlines():
        s = line.strip().lower()
        if s.startswith("(") and s.endswith(")"):
            actions.append(s)
            continue
        tokens = re.findall(r'[\w][\w\-]*', s)
        if tokens and not any(w == tokens[0] for w in
                              ("the","a","and","or","is","to","of","you","note")):
            actions.append("(" + " ".join(tokens) + ")")
    return actions


# ── Forward-chaining validator ───────────────────────────────────
def validate_plan(plan: list, domain_pddl: str, problem_pddl: str) -> bool:
    if not plan:
        return False

    def extract_section(pddl, tag):
        pat = re.search(rf"\(:{tag}\b", pddl, re.I)
        if not pat: return ""
        depth, i = 0, pat.start()
        while i < len(pddl):
            if pddl[i] == "(": depth += 1
            elif pddl[i] == ")":
                depth -= 1
                if depth == 0: return pddl[pat.start()+1:i]
            i += 1
        return ""

    def parse_facts(block):
        return set(re.findall(r"\(([^()]+)\)", block))

    def extract_balanced(s, start):
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

    init_facts = parse_facts(extract_section(problem_pddl, "init"))
    goal_facts = parse_facts(extract_section(problem_pddl, "goal")
                             .replace("and", "").replace(":goal", ""))

    # Parse actions
    actions = {}
    for m in re.finditer(r"\(:action\s+(\S+)(.*?)(?=\(:action|$)",
                         domain_pddl, re.S | re.I):
        name = m.group(1).lower()
        body = m.group(2)
        pm   = re.search(r":parameters\s*\(([^)]*)\)", body, re.I)
        if not pm: continue
        params = pm.group(2).split() if False else pm.group(1).split()
        pre_str, _ = extract_balanced(body, body.lower().find(":precondition"))
        eff_str, _ = extract_balanced(body, body.lower().find(":effect"))
        actions[name] = {"params": params, "pre": pre_str, "eff": eff_str}

    if not actions:
        return True

    def ground(template, param_names, arg_values):
        result = template
        for p, a in zip(param_names, arg_values):
            result = re.sub(rf"(?<![\\w?]){re.escape(p)}(?![\\w-])", a, result)
        return result

    def get_literals(s):
        pos = set(re.findall(r"\(([^()]+)\)", s))
        neg = set(re.findall(r"\(not\s+\(([^()]+)\)\)", s))
        return pos - neg, neg

    state = set(init_facts)
    for step in plan:
        # Strip paren-wrapped args: (move (room1) (room2)) → move room1 room2
        s = step.strip()
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\(([^()]+)\)", r"\1", s)
        parts = s.strip("()").split()
        if not parts: return False
        name, args = parts[0], parts[1:]
        if name not in actions: return False
        act = actions[name]
        if len(args) != len(act["params"]): return False
        grounded_pre = ground(act["pre"], act["params"], args).lower()
        grounded_eff = ground(act["eff"], act["params"], args).lower()
        pre_pos, pre_neg = get_literals(grounded_pre)
        if not pre_pos.issubset(state): return False
        if pre_neg & state: return False
        eff_pos, eff_neg = get_literals(grounded_eff)
        state = (state - eff_neg) | eff_pos

    return goal_facts.issubset(state)


# ── Routing helpers ──────────────────────────────────────────────
def routing_validity(scores, llm_valid, budget=0.5):
    n        = len(scores)
    n_llm    = int(n * budget)
    easy_idx = np.argsort(scores)[:n_llm]
    hard_idx = np.argsort(scores)[n_llm:]
    return float(llm_valid[easy_idx].sum() + len(hard_idx)) / n


# ── Problem loader ───────────────────────────────────────────────
def load_domain_problems(domain: str, n_target: int):
    problem_dir = DATA_DIR / domain / "problems"
    domain_file = DATA_DIR / domain / "domain.pddl"
    if not problem_dir.exists():
        raise FileNotFoundError(f"Problems not found at {problem_dir}")
    domain_pddl    = domain_file.read_text() if domain_file.exists() else ""
    problem_files  = sorted(problem_dir.glob("*.pddl"))[:n_target]
    return [{"id": pf.stem, "pddl": pf.read_text(), "domain": domain_pddl}
            for pf in problem_files], domain_pddl


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         default="checkpoints_planning/guru_n_steps.pt")
    parser.add_argument("--domain",       default="gripper")
    parser.add_argument("--n_target",     type=int,   default=200)
    parser.add_argument("--ollama_model", default="qwen2.5:7b")
    parser.add_argument("--ollama_host",  default="10.139.126.18:11434")
    parser.add_argument("--max_tokens",   type=int,   default=512)
    parser.add_argument("--temperature",  type=float, default=0.0)
    parser.add_argument('--shuffle_instances', action='store_true',
                        help='Shuffle instance order before evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    args = parser.parse_args()

    model = load_checkpoint(args.ckpt)
    Xsurf, Xfm, ysuccess, ynsteps, tasktypes, registry = load_data()

    train_domains = registry["splits"]["meta_train"]["tasks"]
    print(f"Device: {DEVICE}")
    print(f"Train domains: {train_domains}")

    arc_scores, mask = arc_score_domain(
        model, args.domain, Xsurf, Xfm, ynsteps, tasktypes, train_domains)
    nobj_scores = Xsurf[mask][:, 0]

    problems, _ = load_domain_problems(args.domain, args.n_target)
    if args.shuffle_instances:
        import random
        random.seed(args.seed)
        combined = list(zip(problems,
                            arc_scores[:len(problems)],
                            nobj_scores[:len(problems)]))
        random.shuffle(combined)
        problems, arc_scores_list, nobj_scores_list = zip(*combined)
        problems = list(problems)
        arc_scores  = np.array(arc_scores_list)
        nobj_scores = np.array(nobj_scores_list)
        print(f"Shuffled instances with seed={args.seed}")
    n_eval = min(len(problems), int(mask.sum()), args.n_target)

    print(f"\n{'='*60}")
    print(f"Domain: {args.domain.upper()}  N={args.n_target}  "
          f"LLM={args.ollama_model}")
    print(f"{'='*60}")
    print(f"Instances available: {int(mask.sum())}  evaluating: {n_eval}")

    llm_valid = np.zeros(n_eval, dtype=int)
    n_empty   = 0

    for i, prob in enumerate(problems[:n_eval]):
        raw   = call_ollama(prob["pddl"], prob["domain"], args.ollama_model,
                            args.max_tokens, args.temperature, args.ollama_host)
        plan  = parse_plan(raw)
        valid = int(validate_plan(plan, prob["domain"], prob["pddl"]))
        llm_valid[i] = valid
        if not plan: n_empty += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_eval}]  "
                  f"validity: {llm_valid[:i+1].mean():.1%}  empty: {n_empty}")

    llm_acc = float(llm_valid.mean())
    arc_v   = routing_validity(arc_scores[:n_eval], llm_valid)
    nobj_v  = routing_validity(nobj_scores[:n_eval], llm_valid)
    rand_v  = routing_validity(
        np.random.default_rng(42).permutation(n_eval).astype(float), llm_valid)

    # Bootstrap CIs
    rng_b = np.random.default_rng(0)
    d_ar, d_an = [], []
    for _ in range(1000):
        idx  = rng_b.choice(n_eval, n_eval, replace=True)
        a    = routing_validity(arc_scores[idx],   llm_valid[idx])
        r    = routing_validity(rng_b.permutation(n_eval).astype(float), llm_valid[idx])
        o    = routing_validity(nobj_scores[idx],  llm_valid[idx])
        d_ar.append(a - r);  d_an.append(a - o)

    rho_arc,  p_arc  = spearmanr(arc_scores[:n_eval],  llm_valid)
    rho_nobj, p_nobj = spearmanr(nobj_scores[:n_eval], llm_valid)

    print(f"\n{'─'*60}")
    print(f"LLM accuracy  : {llm_acc:.1%}  empty: {n_empty}/{n_eval}")
    print(f"ARC routing   : {arc_v:.1%}  "
          f"Δ vs random: {arc_v-rand_v:+.1%}  "
          f"CI: ({np.percentile(d_ar,2.5):+.1%}, {np.percentile(d_ar,97.5):+.1%})")
    print(f"Obj-count     : {nobj_v:.1%}  Δ vs random: {nobj_v-rand_v:+.1%}")
    print(f"Random        : {rand_v:.1%}")
    print(f"ARC Δ vs nobj : {arc_v-nobj_v:+.1%}  "
          f"CI: ({np.percentile(d_an,2.5):+.1%}, {np.percentile(d_an,97.5):+.1%})")
    print(f"ρ(ARC,valid)  : {rho_arc:+.3f}  p={p_arc:.4f}")
    print(f"ρ(nobj,valid) : {rho_nobj:+.3f}  p={p_nobj:.4f}")

    if llm_acc < 0.15:
        print("WARNING: LLM accuracy < 15% — routing may not help in this regime.")

    out = {
        "domain": args.domain, "n_eval": n_eval,
        "llm_model": args.ollama_model, "llm_accuracy": llm_acc,
        "n_empty": int(n_empty),
        "routing": {
            "arc": arc_v, "obj_count": nobj_v, "random": rand_v,
            "delta_arc_over_random": arc_v - rand_v,
            "delta_arc_over_nobj":   arc_v - nobj_v,
            "ci_arc_over_random": [float(np.percentile(d_ar, 2.5)),
                                   float(np.percentile(d_ar, 97.5))],
            "ci_arc_over_nobj":   [float(np.percentile(d_an, 2.5)),
                                   float(np.percentile(d_an, 97.5))],
        },
        "spearman": {
            "arc_vs_valid":  {"rho": rho_arc,  "p": p_arc},
            "nobj_vs_valid": {"rho": rho_nobj, "p": p_nobj},
        },
        "raw": {
            "arc_scores":  arc_scores[:n_eval].tolist(),
            "nobj_scores": nobj_scores[:n_eval].tolist(),
            "llm_valid":   llm_valid.tolist(),
        },
    }
    out_path = RESULTS_DIR / f"{args.domain}_n{n_eval}_routing.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()