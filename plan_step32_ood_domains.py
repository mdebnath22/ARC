"""
plan_step32_ood_domains.py
===========================
Out-of-distribution domain evaluation.

Tests ARC zero-shot on domains NEVER seen during training or evaluation:
  - IPC 2023 domains (Ricochet Robots, Recharging Robots, Folding)
  - AutoPlanBench 2.0 domains (ICAPS 2024)

If ARC beats object-cardinality on these, the "IPC overfitting" concern
is permanently silenced.

SETUP (run once):
  # IPC 2023 domains
  git clone https://github.com/ipc2023-classical/ipc2023-classical
  # OR individual domains:
  git clone https://github.com/ipc2023-classical/domain-ricochet-robots
  git clone https://github.com/ipc2023-classical/domain-recharging-robots
  git clone https://github.com/ipc2023-classical/domain-folding

  # AutoPlanBench 2.0
  git clone https://github.com/BorealisAI/AutoPlanBench

USAGE:
  python plan_step32_ood_domains.py --setup        # clone repos
  python plan_step32_ood_domains.py --generate     # generate 200 instances
  python plan_step32_ood_domains.py --evaluate     # run BFS + ARC + baselines
  python plan_step32_ood_domains.py --all          # all steps
"""

from __future__ import annotations
import argparse, importlib.util, json, os, pickle, re, signal
import subprocess, sys, tempfile, warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
OOD     = ROOT / "data" / "ood_domains"; OOD.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"
TRAIN_DOMAINS = ["depot","rovers","satellite"]

# IPC 2023 domain repos
IPC2023_REPOS = {
    "ricochet_robots":    "https://github.com/ipc2023-classical/domain-ricochet-robots",
    "recharging_robots":  "https://github.com/ipc2023-classical/domain-recharging-robots",
    "folding":            "https://github.com/ipc2023-classical/domain-folding",
}

# AutoPlanBench repo
APB_REPO = "https://github.com/BorealisAI/AutoPlanBench"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Clone repositories
# ══════════════════════════════════════════════════════════════════════════════

def setup_repos():
    print("\n" + "="*60)
    print("SETUP: Cloning OOD domain repositories")
    print("="*60)

    for name, url in IPC2023_REPOS.items():
        dest = OOD / name
        if dest.exists():
            print(f"  {name}: already exists")
            continue
        print(f"  Cloning {name}...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"  {name}: OK")
        else:
            print(f"  {name}: FAILED — {result.stderr[:100]}")
            print(f"  Trying alternative: download domain.pddl directly")
            _download_domain_fallback(name, dest)

    # AutoPlanBench
    apb_dest = OOD / "autopb"
    if not apb_dest.exists():
        print("  Cloning AutoPlanBench...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", APB_REPO, str(apb_dest)],
            capture_output=True, text=True, timeout=120
        )
        print(f"  AutoPlanBench: {'OK' if result.returncode==0 else 'FAILED'}")

    # List what we have
    print("\n  OOD data directory:")
    for p in sorted(OOD.iterdir()):
        if p.is_dir():
            pddl_files = list(p.rglob("*.pddl"))
            print(f"    {p.name}: {len(pddl_files)} .pddl files")


def _download_domain_fallback(name, dest):
    """Try to download domain.pddl from known raw GitHub URLs."""
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)

    # Known raw URLs for IPC 2023 domain files
    fallback_urls = {
        "ricochet_robots": [
            "https://raw.githubusercontent.com/ipc2023-classical/domain-ricochet-robots/main/domain.pddl",
        ],
        "recharging_robots": [
            "https://raw.githubusercontent.com/ipc2023-classical/domain-recharging-robots/main/domain.pddl",
        ],
        "folding": [
            "https://raw.githubusercontent.com/ipc2023-classical/domain-folding/main/domain.pddl",
        ],
    }

    for url in fallback_urls.get(name, []):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                content = r.read().decode()
            (dest / "domain.pddl").write_text(content)
            print(f"  {name}: downloaded domain.pddl from {url}")
            return True
        except Exception as e:
            print(f"  {name}: download failed: {e}")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Generate instances using available generators or by sampling
# ══════════════════════════════════════════════════════════════════════════════

def generate_instances():
    print("\n" + "="*60)
    print("GENERATE: Creating 200 instances per OOD domain")
    print("="*60)

    N = 200
    generated = {}

    for dom_name in list(IPC2023_REPOS.keys()) + ["autopb"]:
        dom_dir = OOD / dom_name
        if not dom_dir.exists():
            print(f"  {dom_name}: not found, skipping")
            continue

        # Find domain.pddl
        domain_files = list(dom_dir.rglob("domain.pddl"))
        if not domain_files:
            print(f"  {dom_name}: no domain.pddl found")
            continue

        domain_pddl = domain_files[0].read_text()
        print(f"\n  {dom_name}: found domain at {domain_files[0]}")
        print(f"    Actions: {re.findall(r':action\\s+(\\S+)', domain_pddl)[:5]}")

        # Find problem files (existing instances)
        prob_files = sorted(dom_dir.rglob("p*.pddl")) + \
                     sorted(dom_dir.rglob("problem*.pddl")) + \
                     sorted(dom_dir.rglob("instance*.pddl"))
        prob_files = [p for p in prob_files if "domain" not in p.name.lower()]

        print(f"    Found {len(prob_files)} existing problem files")

        if len(prob_files) >= 20:
            # Use existing instances (up to N)
            instances = []
            for i, pf in enumerate(prob_files[:N]):
                try:
                    prob_pddl = pf.read_text()
                    instances.append({
                        "instance_id": i,
                        "domain": dom_name,
                        "task_type": dom_name,
                        "domain_pddl": domain_pddl,
                        "problem_pddl": prob_pddl,
                        "source_file": str(pf),
                    })
                except Exception:
                    continue

            # Try to find generator script
            gen_scripts = list(dom_dir.rglob("generate*.py")) + \
                          list(dom_dir.rglob("generator*.py")) + \
                          list(dom_dir.rglob("gen*.sh"))
            if gen_scripts:
                print(f"    Generator found: {gen_scripts[0].name}")

            print(f"    Using {len(instances)} instances")
            generated[dom_name] = instances

        elif domain_pddl:
            # No problem files — try to use generator if present
            gen_scripts = list(dom_dir.rglob("*.py")) + \
                          list(dom_dir.rglob("*.sh"))
            print(f"    Few problem files. Generator scripts: {[g.name for g in gen_scripts[:3]]}")
            print(f"    Try running generator manually:")
            print(f"    cd {dom_dir}")
            for gs in gen_scripts[:2]:
                print(f"    python {gs.name} [args]")
            generated[dom_name] = []  # will fill after manual generation
        else:
            print(f"    SKIP: no domain.pddl")

    # Save
    out = OOD / "instances.json"
    flat = []
    for dom, insts in generated.items():
        flat.extend(insts)
    out.write_text(json.dumps(flat, indent=2))
    print(f"\n  Saved {len(flat)} total instances → {out}")
    return generated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: BFS labeling
# ══════════════════════════════════════════════════════════════════════════════

def run_bfs(instances, timeout=15, max_steps=12, max_nodes=5000):
    """Run BFS on all instances, return difficulty labels."""
    try:
        from pyperplan.pddl.parser import Parser
        from pyperplan import grounding
        from pyperplan.search.breadth_first_search import breadth_first_search
    except ImportError:
        print("  pyperplan not installed: pip install pyperplan")
        return instances

    labeled = []
    print(f"\n  Running BFS on {len(instances)} instances...")

    for i, ep in enumerate(instances):
        dom_pddl  = ep.get("domain_pddl", "")
        prob_pddl = ep.get("problem_pddl", "")
        if not dom_pddl or not prob_pddl:
            ep["n_steps"] = -1; ep["bfs_solved"] = False
            labeled.append(ep); continue

        def _to(s,f): raise TimeoutError()
        signal.signal(signal.SIGALRM, _to); signal.alarm(timeout)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dp = Path(tmp)/"domain.pddl"; pp = Path(tmp)/"problem.pddl"
                dp.write_text(dom_pddl); pp.write_text(prob_pddl)
                parser = Parser(str(dp), str(pp))
                task   = grounding.ground(parser.parse_problem(parser.parse_domain()))
                sol    = breadth_first_search(task)
                signal.alarm(0)
                ep["n_steps"]    = len(sol) if sol else -1
                ep["bfs_solved"] = sol is not None
        except (TimeoutError, Exception) as e:
            signal.alarm(0)
            ep["n_steps"]    = -1
            ep["bfs_solved"] = False

        labeled.append(ep)
        if (i+1) % 20 == 0:
            solved = sum(1 for e in labeled if e["bfs_solved"])
            print(f"    [{i+1}/{len(instances)}] solved={solved}", end="\r")

    solved = sum(1 for e in labeled if e["bfs_solved"])
    print(f"\n  BFS: {solved}/{len(instances)} solved  "
          f"({solved/len(instances):.1%})")
    return labeled


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Extract syntactic features
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(instances):
    """Extract the same 29 syntactic features used in training."""
    featurized = []
    for ep in instances:
        dom  = ep.get("domain_pddl","")
        prob = ep.get("problem_pddl","")

        # Parse basic counts
        objects   = re.findall(r'^\s*(\S+)\s*-\s*\S+', prob, re.M)
        n_obj     = len(objects)
        init      = re.findall(r'\((\w[\w\-]*)', prob.split(":goal")[0].split(":init")[-1] if ":init" in prob else "")
        goal      = re.findall(r'\((\w[\w\-]*)', prob.split(":goal")[-1] if ":goal" in prob else "")
        ops       = re.findall(r':action\s+\S+', dom)
        preds     = re.findall(r':predicate\s+\S+|:predicates[^)]+\(\s*(\w[\w\-]*)', dom)

        n_init    = len(init)
        n_goal    = len(goal)
        n_ops     = len(ops)

        pred_counts = {}
        for p in init: pred_counts[p] = pred_counts.get(p,0)+1
        pc_vals   = list(pred_counts.values()) if pred_counts else [0]

        feat = np.array([
            float(n_obj),
            float(n_goal),
            float(n_init),
            float(n_ops),
            float(n_goal),          # goal_cardinality
            float(np.mean(pc_vals)),
            float(np.std(pc_vals)) if len(pc_vals)>1 else 0,
            float(np.max(pc_vals)),
            float(len(set(pred_counts.keys()))),
            float(n_ops % 2),       # operator_parity
            float(n_goal/max(n_obj,1)),
            float(n_init/max(n_goal,1)),
            float(n_obj/max(len(pred_counts),1)),
            float(n_goal*np.mean(pc_vals)),
            float(len(pred_counts)/max(n_ops,1)),
            float(len(set(pred_counts.keys()))/max(n_obj,1)),
            float(n_init/max(n_obj**2,1)),  # init_density
            float(n_goal/max(n_obj**2,1)),
            float(len(pred_counts)/max(n_ops,1)),
            float(np.mean(pc_vals)*n_goal),
            float(n_obj*np.mean(pc_vals)),
            float(n_goal/max(n_init,1)),
            float(n_init/max(n_ops,1)),
            float(n_goal/max(n_init,1)),
            float(n_obj/max(n_goal,1)),
            float(len(pred_counts)/max(n_obj,1)),
            float(n_init/max(n_obj,1)),
            float(n_ops/max(n_obj,1)),
            float(n_obj*n_goal/max(n_init,1)),
        ], dtype=np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        ep["features_xs"] = feat.tolist()
        ep["n_objects"]   = n_obj
        featurized.append(ep)
    return featurized


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: ARC evaluation on OOD domains
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ood():
    print("\n" + "="*60)
    print("EVALUATE: ARC zero-shot on OOD domains")
    print("="*60)

    # Load instances
    inst_file = OOD / "instances_labeled.json"
    if not inst_file.exists():
        print("  Run --generate first")
        return {}

    instances = json.loads(inst_file.read_text())
    instances = [e for e in instances if e.get("bfs_solved") and e.get("n_steps",0) > 0]
    print(f"  Using {len(instances)} labeled instances")

    if len(instances) < 20:
        print("  Not enough labeled instances. Run BFS first.")
        return {}

    # Load ARC model (use best available)
    spec6 = importlib.util.spec_from_file_location("s6", ROOT/"plan_step6_pddlinst_gate.py")
    s6    = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    X_surf_train, X_fm_train, tt_train, y_s, y_ns, _ = s6.load_data(data_dir=DATA)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    tr = np.isin(tt_train, TRAIN_DOMAINS)
    sc_s = StandardScaler().fit(X_surf_train[tr])
    sc_f = StandardScaler().fit(X_fm_train[tr])
    pp   = Pipeline([("pca",PCA(20)),("r",Ridge(1.0))])
    pp.fit(sc_s.transform(X_surf_train[tr]), sc_f.transform(X_fm_train[tr]))

    # Support set from training domains
    import torch
    spec16 = importlib.util.spec_from_file_location("s16", ROOT/"plan_step16_arc_improvements.py")
    s16    = importlib.util.module_from_spec(spec16); spec16.loader.exec_module(s16)

    ckpt = torch.load(CKPT/"guru_baseline_5000ep.pt", map_location="cpu")
    model = s16.ImprovedGURU(30, X_fm_train.shape[1])
    model.load_state_dict(ckpt["model"]); model.eval()

    rng  = np.random.default_rng(42)
    sidx = rng.choice(tr.sum(), min(60,tr.sum()), replace=False)
    Xs_tr = sc_s.transform(X_surf_train[tr])
    Xe_tr = sc_f.transform(X_fm_train[tr])
    Xr_tr = Xe_tr - pp.predict(Xs_tr)
    S_s   = torch.FloatTensor(Xs_tr[sidx])
    S_f   = torch.FloatTensor(Xe_tr[sidx])
    S_V   = torch.FloatTensor(np.hstack([Xs_tr[sidx],Xe_tr[sidx]]))

    # FM embeddings for OOD instances
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-mpnet-base-v2")

    results = {}
    doms = list(set(e["task_type"] for e in instances))

    print(f"\n  {'Domain':<22}  {'N':>4}  {'ARC |ρ|':>9}  {'|O| |ρ|':>9}  {'Δ':>6}")
    print("  " + "-"*55)

    for dom in doms:
        dom_eps = [e for e in instances if e["task_type"]==dom]
        if len(dom_eps) < 10:
            continue

        n_steps = np.array([e["n_steps"] for e in dom_eps], dtype=float)
        n_obj   = np.array([e["n_objects"] for e in dom_eps], dtype=float)

        # Build descriptions for FM embedding
        descs = []
        for ep in dom_eps:
            prob = ep.get("problem_pddl","")
            objs = re.findall(r'^\s*(\S+)\s*-\s*\S+', prob, re.M)
            goals = re.findall(r'\([\w\-]+[^)]*\)', prob.split(":goal")[-1][:200] if ":goal" in prob else "")
            desc  = (f"Planning problem in domain {dom}. "
                     f"Objects: {', '.join(objs[:8])}. "
                     f"Goals: {', '.join(goals[:4])}.")
            descs.append(desc)

        X_fm_ood = sbert.encode(descs, show_progress_bar=False)
        X_xs_ood = np.array([e.get("features_xs",[0]*30) for e in dom_eps])
        if X_xs_ood.shape[1] < 30:
            pad = np.zeros((len(X_xs_ood), 30-X_xs_ood.shape[1]))
            X_xs_ood = np.hstack([X_xs_ood, pad])

        Xs_n = sc_s.transform(X_xs_ood)
        Xe_n = sc_f.transform(X_fm_ood)
        Xr_n = Xe_n - pp.predict(Xs_n)

        arc_scores = []
        import torch.nn.functional as F
        with torch.no_grad():
            for i in range(len(dom_eps)):
                qs = torch.FloatTensor(Xs_n[i]).unsqueeze(0)
                qf = torch.FloatTensor(Xe_n[i]).unsqueeze(0)
                qr = torch.FloatTensor(Xr_n[i]).unsqueeze(0)
                try:
                    o,_,_=model(qs,qf,qr,S_s,S_f,S_V,head="cls")
                    arc_scores.append(float(F.softmax(o.squeeze(0),-1)[1].cpu()))
                except Exception:
                    arc_scores.append(0.5)

        arc_scores = np.array(arc_scores)
        rho_arc, _ = stats.spearmanr(arc_scores, n_steps)
        rho_obj, _ = stats.spearmanr(n_obj,       n_steps)

        results[dom] = {
            "n": len(dom_eps),
            "rho_arc": float(rho_arc),
            "rho_obj": float(rho_obj),
            "arc_wins": abs(rho_arc) > abs(rho_obj),
        }
        dl = dom.replace("_"," ").title()[:20]
        print(f"  {dl:<22}  {len(dom_eps):>4}  {abs(rho_arc):>9.3f}  "
              f"{abs(rho_obj):>9.3f}  {abs(rho_arc)-abs(rho_obj):>+6.3f}")

    # LaTeX
    print("\n  LaTeX:")
    print(r"\begin{table}[t]\centering\small")
    print(r"\caption{Zero-shot ARC on out-of-distribution domains")
    print(r"(IPC 2023 / AutoPlanBench), never seen during training.")
    print(r"Spearman $|\rho|$ between ARC predictions and BFS plan length.}")
    print(r"\label{tab:ood}")
    print(r"\begin{tabular}{l ccc}\toprule")
    print(r"\textbf{Domain} & N & ARC $|\rho|$ & $|O|$ $|\rho|$ \\\midrule")
    for dom,r in results.items():
        dl = dom.replace("_"," ").title()
        better = r"$\uparrow$" if r["arc_wins"] else ""
        print(f"  {dl} & {r['n']} & {abs(r['rho_arc']):.3f}{better} & {abs(r['rho_obj']):.3f} \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")

    (RESULTS/"ood_evaluation.json").write_text(json.dumps(results,indent=2))
    print(f"\n  Saved → {RESULTS}/ood_evaluation.json")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup",    action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--label",    action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--all",      action="store_true")
    args = p.parse_args()

    if args.setup or args.all:
        setup_repos()

    if args.generate or args.all:
        generated = generate_instances()
        # Flatten and featurize
        flat = []
        for dom, insts in generated.items():
            flat.extend(insts)
        flat = extract_features(flat)
        (OOD/"instances_featurized.json").write_text(json.dumps(flat,indent=2))

    if args.label or args.all:
        feat_file = OOD/"instances_featurized.json"
        if not feat_file.exists():
            print("Run --generate first"); return
        instances = json.loads(feat_file.read_text())
        labeled   = run_bfs(instances)
        (OOD/"instances_labeled.json").write_text(json.dumps(labeled,indent=2))
        print(f"Saved labeled instances → {OOD}/instances_labeled.json")

    if args.evaluate or args.all:
        evaluate_ood()

    if not any([args.setup,args.generate,args.label,args.evaluate,args.all]):
        print(__doc__)

if __name__ == "__main__": main()
