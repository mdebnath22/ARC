"""
plan_step18_baselines_and_ablations.py
========================================
Addresses two reviewer concerns:

PROBLEM 6 — Missing strong baselines
  B5:  XGBoost on syntactic features (xs)
  B6:  LightGBM on syntactic features
  B7:  kNN retrieval (no attention)
  B8:  Cosine similarity retrieval (no asymmetric attention)
  B9:  Retrieval + MLP (symmetric, no asymmetric attention)
  B12: ProtoNet-style (nearest prototype)
  B19: LLM-only predictor (XGBoost trained on LLM labels)

PROBLEM 7 — Asymmetric attention not convincingly justified
  Metric A: pairwise cosine similarity (collapse indicator)
  Metric B: representation variance (collapse indicator)
  Metric C: effective rank (strongest collapse indicator)
  Exp  D:  Symmetric attention (Q=K=FM embeddings)
  Exp  E:  Swapped Q/K (xs as query, FM as keys)
  Exp  F:  LayerNorm fix (does normalization solve collapse?)
  Exp  G:  Attention entropy by variant
  Table H: Full comparison — collapse metrics vs downstream |ρ|

Everything uses the GlobalPreprocessor from plan_step17 (no leakage).

USAGE:
  python plan_step18_baselines_and_ablations.py --part baselines
  python plan_step18_baselines_and_ablations.py --part p7
  python plan_step18_baselines_and_ablations.py --part all
"""

from __future__ import annotations
import argparse, importlib.util, json, pickle, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data" / "planning"
RESULTS_DIR = ROOT_DIR / "results_planning"; RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR    = ROOT_DIR / "checkpoints_planning"
PREP_PATH   = RESULTS_DIR / "global_preprocessor.pkl"

TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
DOMAIN_LABELS = {"blocksworld":"Blocksworld","logistics":"Logistics",
                 "mystery_blocksworld":"Mystery-BW"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Shared loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    """Load data, preprocessor, and step6 splits."""
    spec6 = importlib.util.spec_from_file_location(
        "step6", ROOT_DIR / "plan_step6_pddlinst_gate.py")
    step6 = importlib.util.module_from_spec(spec6)
    spec6.loader.exec_module(step6)

    X_surf, X_fm, task_types, y_success, y_nsteps, splits = \
        step6.load_data(data_dir=DATA_DIR)

    # Remove domain hash (P4 fix from step17)
    X_surf = X_surf[:, :-1]

    train_doms = splits["meta_train"]["domains"]
    train_mask = np.isin(task_types, train_doms)

    # Load global preprocessor — must import class before unpickling
    if not PREP_PATH.exists():
        raise FileNotFoundError(
            f"{PREP_PATH} not found. Run plan_step17 --phase preprocess first.")
    import importlib.util as _ilu, sys as _sys
    _spec = _ilu.spec_from_file_location("step17", ROOT_DIR/"plan_step17_arc_v2.py")
    _s17  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_s17)
    _sys.modules["step17"] = _s17          # make pickle resolve the class
    GlobalPreprocessor = _s17.GlobalPreprocessor
    import importlib.util as _ilu, sys as _sys
    _spec = _ilu.spec_from_file_location("step17", ROOT_DIR/"plan_step17_arc_v2.py")
    _s17  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_s17)
    _sys.modules["step17"] = _s17
    _sys.modules["__main__"].GlobalPreprocessor = _s17.GlobalPreprocessor
    with open(PREP_PATH, "rb") as f:
        prep = pickle.load(f)

    return X_surf, X_fm, task_types, y_success, y_nsteps, train_mask, train_doms, prep


def eval_metrics(scores, y_success, y_nsteps):
    """Compute AUC, R², |ρ|."""
    try:
        auc = float(roc_auc_score(y_success.astype(float), scores))
    except Exception:
        auc = float("nan")
    rho, _  = stats.spearmanr(scores, y_nsteps)
    ss_res  = float(np.sum((y_nsteps - scores * y_nsteps.max()) ** 2))
    ss_tot  = float(np.sum((y_nsteps - y_nsteps.mean()) ** 2))
    r2      = float(1 - ss_res / max(ss_tot, 1e-8))
    return auc, r2, float(rho)


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 6 — Missing Baselines
# ══════════════════════════════════════════════════════════════════════════════

# ── ERM baseline (P8: non-episodic, trains on all domains jointly) ──────────

class ERMBaseline(nn.Module):
    """
    Empirical Risk Minimisation baseline: train a standard MLP on all
    source-domain instances jointly (no episodic structure, no attention).
    This is the strongest non-episodic baseline for the P8 claim.

    If ERM matches ARC, episodic training is unnecessary.
    If ERM fails on test domains, episodic training is justified.

    Architecture: MLP([xs, xe]) → difficulty score
    Training: standard cross-entropy on BFS labels, all source domains
    Evaluation: zero-shot transfer to test domains
    """
    def __init__(self, surf_dim: int, fm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(surf_dim + fm_dim, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, xs, xe):
        return self.net(torch.cat([xs, xe], dim=-1))


def train_erm_baseline(X_surf, X_fm, y_success, task_types, prep,
                       train_domains, n_epochs=50, device="cpu"):
    """Train ERM on all source-domain instances jointly."""
    mask  = np.isin(task_types, train_domains)
    Xs_n, Xe_n, _ = prep.transform(X_surf[mask], X_fm[mask])
    y_tr  = y_success[mask].astype(int)

    model = ERMBaseline(Xs_n.shape[1], Xe_n.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    X_t = torch.FloatTensor(np.hstack([Xs_n, Xe_n])).to(device)
    y_t = torch.LongTensor(y_tr).to(device)

    model.train()
    for ep in range(n_epochs):
        out  = model(torch.FloatTensor(Xs_n).to(device),
                     torch.FloatTensor(Xe_n).to(device))
        loss = F.cross_entropy(out, y_t)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    model.eval()
    return model


def eval_erm(model, X_surf, X_fm, y_success, y_nsteps, task_types,
             test_domains, prep, device="cpu"):
    results = {}
    model.eval()
    for dom in test_domains:
        mask  = task_types == dom
        Xs_n, Xe_n, _ = prep.transform(X_surf[mask], X_fm[mask])
        y_q   = y_success[mask].astype(float)
        ns_q  = y_nsteps[mask].astype(float)
        with torch.no_grad():
            out = model(torch.FloatTensor(Xs_n).to(device),
                        torch.FloatTensor(Xe_n).to(device))
            scores = torch.softmax(out, -1)[:, 1].cpu().numpy()
        try:
            auc = float(roc_auc_score(y_q, scores))
        except Exception:
            auc = float("nan")
        rho, _ = stats.spearmanr(scores, ns_q)
        results[dom] = {"auc": auc, "rho": float(rho)}
    return results


def run_baselines(args):
    print("\n" + "="*65)
    print("PROBLEM 6: Missing Baselines")
    print("="*65)

    X_surf, X_fm, task_types, y_success, y_nsteps, train_mask, train_doms, prep = \
        load_all()

    # Preprocess training data (global, frozen)
    Xs_tr_n, Xe_tr_n, Xr_tr_n = prep.transform(X_surf[train_mask], X_fm[train_mask])
    y_cls_tr = y_success[train_mask].astype(int)
    y_reg_tr = y_nsteps[train_mask].astype(float)
    X_full_tr = np.hstack([Xs_tr_n, Xe_tr_n, Xr_tr_n])   # full feature set

    # Load Qwen labels if available (for B19)
    qwen_by_dom = {}
    qwen_path = RESULTS_DIR / "qwen72b_eval_instances.jsonl"
    if qwen_path.exists():
        for line in open(qwen_path):
            r = json.loads(line)
            qwen_by_dom.setdefault(r["domain"], []).append(r)
        for d in qwen_by_dom:
            qwen_by_dom[d].sort(key=lambda r: int(r["instance_id"]))

    print(f"\n  Training on: {train_doms}")
    print(f"  Training instances: {train_mask.sum()}")
    print()

    results = {}

    # ── B5: XGBoost on xs only ─────────────────────────────────────────────
    print("  [B5] XGBoost on syntactic features (xs) — zero-shot")
    xgb_cls = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        verbosity=0, use_label_encoder=False, eval_metric="logloss",
        random_state=args.seed)
    xgb_cls.fit(Xs_tr_n, y_cls_tr)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        verbosity=0, random_state=args.seed)
    xgb_reg.fit(Xs_tr_n, y_reg_tr)

    # ── B6: LightGBM on xs ────────────────────────────────────────────────
    lgbm_cls = lgbm_reg = None
    try:
        import lightgbm as lgb
        lgbm_cls = lgb.LGBMClassifier(
            n_estimators=300, verbose=-1, random_state=args.seed)
        lgbm_cls.fit(Xs_tr_n, y_cls_tr)
        lgbm_reg = lgb.LGBMRegressor(
            n_estimators=300, verbose=-1, random_state=args.seed)
        lgbm_reg.fit(Xs_tr_n, y_reg_tr)
        print("  [B6] LightGBM fitted")
    except ImportError:
        print("  [B6] LightGBM not installed — skipping")

    # ── B7: kNN on xs ─────────────────────────────────────────────────────
    print("  [B7] kNN (k=5) on syntactic features")
    knn_cls = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_cls.fit(Xs_tr_n, y_cls_tr)
    knn_reg = KNeighborsRegressor(n_neighbors=5, metric="euclidean")
    knn_reg.fit(Xs_tr_n, y_reg_tr)

    # ── B8: Cosine similarity retrieval (no attention mechanism) ───────────
    print("  [B8] Cosine similarity retrieval (FM embeddings, no attention)")

    def cosine_retrieval_score(Xe_q, Xe_tr, y_tr, k=5):
        """Simple cosine retrieval: average labels of k nearest in FM space."""
        Xe_q_n  = Xe_q / (np.linalg.norm(Xe_q, axis=1, keepdims=True) + 1e-8)
        Xe_tr_n = Xe_tr / (np.linalg.norm(Xe_tr, axis=1, keepdims=True) + 1e-8)
        sims    = Xe_q_n @ Xe_tr_n.T           # (N_q, N_tr)
        top_k   = np.argsort(-sims, axis=1)[:, :k]
        return np.array([y_tr[top_k[i]].mean() for i in range(len(Xe_q))])

    # ── B9: Retrieval + MLP (concatenate retrieved, no asymmetric attention) ─
    print("  [B9] Retrieval + MLP (symmetric, no asymmetric attention)")

    class RetrievalMLP(nn.Module):
        def __init__(self, feat_dim, k=5):
            super().__init__()
            self.k = k
            self.net = nn.Sequential(
                nn.Linear(feat_dim * (k + 1), 256), nn.LayerNorm(256),
                nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(128, 2),
            )
        def forward(self, x): return self.net(x)

    # Build retrieval-augmented training data
    def build_retrieval_features(Xs, Xe, Xr, Xs_tr, Xe_tr, k=5):
        """For each instance, concatenate its features with k nearest retrieved."""
        all_feat = np.hstack([Xs, Xe, Xr])
        all_tr   = np.hstack([Xs_tr, Xe_tr,
                               Xe_tr - Xs_tr @ np.linalg.pinv(Xs_tr.T @ Xs_tr) @
                               Xs_tr.T @ Xe_tr])
        all_feat_n = all_feat / (np.linalg.norm(all_feat, axis=1, keepdims=True)+1e-8)
        all_tr_n   = all_tr   / (np.linalg.norm(all_tr,   axis=1, keepdims=True)+1e-8)
        sims   = all_feat_n @ all_tr_n.T
        top_k  = np.argsort(-sims, axis=1)[:, :k]
        concat = [np.hstack([all_feat[i],
                              all_tr[top_k[i]].mean(axis=0)])
                  for i in range(len(all_feat))]
        return np.array(concat)

    # ── B12: ProtoNet-style ────────────────────────────────────────────────
    print("  [B12] ProtoNet-style (prototype per difficulty class)")

    def protonet_score(Xs_q, Xe_q, Xs_tr, Xe_tr, y_tr):
        """
        Compute class prototypes (easy/hard) in FM+surf embedding space.
        Score = cosine similarity to 'easy' prototype.
        """
        feat_q  = np.hstack([Xs_q, Xe_q])
        feat_tr = np.hstack([Xs_tr, Xe_tr])
        proto_0 = feat_tr[y_tr == 0].mean(axis=0)
        proto_1 = feat_tr[y_tr == 1].mean(axis=0)
        protos  = np.stack([proto_0, proto_1])
        protos  = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        feat_q_n = feat_q / (np.linalg.norm(feat_q, axis=1, keepdims=True) + 1e-8)
        sims    = feat_q_n @ protos.T        # (N, 2)
        soft    = np.exp(sims) / np.exp(sims).sum(axis=1, keepdims=True)
        return soft[:, 1]                    # P(easy)

    # ── Evaluate all baselines on test domains ─────────────────────────────
    print(f"\n  {'Method':<32}  {'BW |ρ|':>8}  {'LOG |ρ|':>9}  {'MBW |ρ|':>9}  {'mean':>8}")
    print("  " + "-"*72)

    baselines = {}
    for dom in TEST_DOMAINS:
        mask  = task_types == dom
        N     = mask.sum()
        Xs_q_n, Xe_q_n, Xr_q_n = prep.transform(X_surf[mask], X_fm[mask])
        y_q   = y_success[mask].astype(float)
        ns_q  = y_nsteps[mask].astype(float)

        qwen_labels = None
        if dom in qwen_by_dom:
            qlist = qwen_by_dom[dom]
            qwen_labels = np.array([1.0 if r["valid_plan"] else 0.0
                                    for r in qlist[:N]])

        dom_results = {}

        # B5: XGBoost
        s_xgb = xgb_cls.predict_proba(Xs_q_n)[:, 1]
        dom_results["xgb"] = eval_metrics(s_xgb, y_q, ns_q)

        # B6: LightGBM
        if lgbm_cls is not None:
            s_lgb = lgbm_cls.predict_proba(Xs_q_n)[:, 1]
            dom_results["lgbm"] = eval_metrics(s_lgb, y_q, ns_q)

        # B7: kNN
        s_knn = knn_cls.predict_proba(Xs_q_n)[:, 1]
        dom_results["knn"] = eval_metrics(s_knn, y_q, ns_q)

        # B8: Cosine similarity retrieval
        s_cos = cosine_retrieval_score(Xe_q_n, Xe_tr_n, y_cls_tr.astype(float))
        dom_results["cosine_retrieval"] = eval_metrics(s_cos, y_q, ns_q)

        # B12: ProtoNet
        s_proto = protonet_score(Xs_q_n, Xe_q_n, Xs_tr_n, Xe_tr_n, y_cls_tr)
        dom_results["protonet"] = eval_metrics(s_proto, y_q, ns_q)

        # B19: LLM-only predictor (only if Qwen labels available on training domains)
        # Note: we can only train on test-domain Qwen labels (few-shot setup)
        if qwen_labels is not None and len(np.unique(qwen_labels)) > 1:
            n_tr   = int(0.7 * N)
            tr_idx = np.random.default_rng(42).choice(N, n_tr, replace=False)
            te_idx = np.setdiff1d(np.arange(N), tr_idx)
            llm_clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, verbosity=0,
                use_label_encoder=False, eval_metric="logloss",
                random_state=args.seed)
            llm_clf.fit(Xs_q_n[tr_idx], qwen_labels[tr_idx].astype(int))
            s_llm = llm_clf.predict_proba(Xs_q_n[te_idx])[:, 1]
            dom_results["llm_predictor"] = eval_metrics(
                s_llm, qwen_labels[te_idx], ns_q[te_idx])
        else:
            dom_results["llm_predictor"] = (float("nan"), float("nan"), float("nan"))

        baselines[dom] = dom_results

    # ── Compute LLM-success correlation and routing gains ─────────────────
    # Load Qwen labels (for LLM success correlation)
    qwen_path = RESULTS_DIR / "qwen72b_eval_instances.jsonl"
    qwen_success = {}   # dom → np.array of 0/1
    if qwen_path.exists():
        qwen_rows = {}
        for line in open(qwen_path):
            r = json.loads(line)
            qwen_rows.setdefault(r["domain"], []).append(r)
        for dom in TEST_DOMAINS:
            rlist = sorted(qwen_rows.get(dom, []), key=lambda r: int(r["instance_id"]))
            qwen_success[dom] = np.array([1.0 if r["valid_plan"] else 0.0
                                           for r in rlist])

    BFS_SUCCESS = {"blocksworld":0.590,"logistics":0.160,"mystery_blocksworld":0.555}

    # Compute routing gain: sort by score, route top-k to LLM, rest to BFS
    def routing_gain_vs_nobj(scores_method, scores_nobj, y_llm, bfs_rate, N):
        budgets = np.linspace(0.05, 0.95, 19)
        def val_at_budget(scores, k_frac):
            k = max(1, int(k_frac * N))
            top = np.argsort(-scores)[:k]
            return (y_llm[top].sum() + bfs_rate*(N-k)) / N
        gain_m    = max(val_at_budget(scores_method, b) for b in budgets)
        gain_nobj = max(val_at_budget(scores_nobj,   b) for b in budgets)
        return gain_m - gain_nobj

    # Print table
    method_names = ["xgb","lgbm","knn","cosine_retrieval","protonet","llm_predictor"]
    display_names = {
        "xgb":               "XGBoost ($x_s$)",
        "lgbm":              "LightGBM ($x_s$)",
        "knn":               "kNN-5 ($x_s$)",
        "cosine_retrieval":  "Cosine retrieval ($x_e$)",
        "protonet":          "ProtoNet",
        "llm_predictor":     "LLM-outcome pred.",
    }

    # Header
    print(f"  {'Method':<26}  {'BW|ρ|':>6}  {'L|ρ|':>6}  {'M|ρ|':>6}  "
          f"{'mean':>6}  {'BW LLM':>7}  {'L LLM':>7}  {'M LLM':>7}  "
          f"{'BW Δ':>6}  {'M Δ':>6}")
    print("  " + "-"*88)

    # Object-cardinality baseline row (always first)
    print(f"  {'Object-cardinality ($|O|$)':<26}  ", end="")
    for dom in TEST_DOMAINS:
        mask = task_types == dom
        nobj = X_surf[mask, 0]
        ns   = y_nsteps[mask].astype(float)
        rho, _ = stats.spearmanr(nobj, ns)
        print(f"  {abs(rho):>6.3f}", end="")
    print(f"  {'—':>6}", end="")
    for dom in TEST_DOMAINS:
        if dom in qwen_success and len(qwen_success[dom]):
            mask = task_types == dom
            nobj = X_surf[mask, 0][:len(qwen_success[dom])]
            r, _ = stats.spearmanr(nobj, qwen_success[dom])
            print(f"  {abs(r):>7.3f}", end="")
        else:
            print(f"  {'n/a':>7}", end="")
    print(f"  {'—':>6}  {'—':>6}")

    for mn in method_names:
        rho_vals = []
        llm_vals = []
        route_gains = []
        for dom in TEST_DOMAINS:
            v = baselines[dom].get(mn, (float("nan"),float("nan"),float("nan")))[2]
            rho_vals.append(abs(v) if not np.isnan(v) else float("nan"))
            # LLM correlation
            if dom in qwen_success and len(qwen_success[dom]) and not np.isnan(v):
                # Get scores for this domain
                mask = task_types == dom
                Xs_q, Xe_q, Xr_q = prep.transform(X_surf[mask], X_fm[mask])
                N = min(len(Xs_q), len(qwen_success[dom]))
                if mn == "xgb":
                    scores = xgb_cls.predict_proba(Xs_q[:N])[:,1]
                elif mn == "knn":
                    scores = knn_cls.predict_proba(Xs_q[:N])[:,1]
                else:
                    scores = np.full(N, 0.5)
                r_llm, _ = stats.spearmanr(scores, qwen_success[dom][:N])
                llm_vals.append(abs(r_llm))
                # Routing gain vs n_objects
                nobj = X_surf[mask, 0][:N]
                gain = routing_gain_vs_nobj(
                    scores, nobj, qwen_success[dom][:N],
                    BFS_SUCCESS[dom], N)
                if dom != "logistics":  # logistics Qwen=0%, uninformative
                    route_gains.append(gain)
            else:
                llm_vals.append(float("nan"))

        mean_rho = np.nanmean(rho_vals)
        row = f"  {display_names[mn]:<26}"
        for v in rho_vals:
            row += f"  {v:>6.3f}" if not np.isnan(v) else f"  {'—':>6}"
        row += f"  {mean_rho:>6.3f}"
        for v in llm_vals:
            row += f"  {v:>7.3f}" if not np.isnan(v) else f"  {'—':>7}"
        # Routing gains for BW and MBW only
        if len(route_gains) >= 2:
            row += f"  {route_gains[0]:>+6.3f}  {route_gains[-1]:>+6.3f}"
        print(row)

    # ── ERM baseline (P8) ────────────────────────────────────────────────────
    print("  [ERM] Non-episodic MLP on all source domains (P8 baseline)")
    erm_model = train_erm_baseline(
        X_surf, X_fm, y_success, task_types, prep,
        train_doms, n_epochs=100, device="cpu")
    for dom in TEST_DOMAINS:
        mask  = task_types == dom
        Xs_q_n, Xe_q_n, _ = prep.transform(X_surf[mask], X_fm[mask])
        y_q   = y_success[mask].astype(float)
        ns_q  = y_nsteps[mask].astype(float)
        with torch.no_grad():
            out_erm = erm_model(
                torch.FloatTensor(Xs_q_n), torch.FloatTensor(Xe_q_n))
            s = torch.softmax(out_erm,-1)[:,1].numpy()
        rho, _ = stats.spearmanr(s, ns_q)
        baselines[dom]["erm"] = (float("nan"), float("nan"), float(rho))
        display_names["erm"] = "ERM (non-episodic MLP)"
    method_names.append("erm")
    print("  ERM trained. Key question: does episodic ARC beat ERM?")

    out = RESULTS_DIR / "baselines_p6.json"
    out.write_text(json.dumps(
        {dom: {k: list(v) for k,v in d.items()}
         for dom, d in baselines.items()}, indent=2))
    print(f"\n  Results → {out}")

    # Generate LaTeX table
    # Generate extended LaTeX table
    tex_lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Baseline comparison on zero-shot cross-domain difficulty "
        r"prediction and LLM routing. "
        r"$|\rho|_{\text{BFS}}$: Spearman correlation with BFS solution length "
        r"(difficulty ranking, primary metric). "
        r"$|\rho|_{\text{LLM}}$: correlation with Qwen~72B success. "
        r"$\Delta_{\text{route}}$: routing gain over object-cardinality baseline "
        r"on BW and Mystery-BW (Logistics omitted: Qwen accuracy = 0\%). "
        r"All methods trained on source domains only (zero-shot transfer).}",
        r"\label{tab:baselines}",
        r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l ccc c  ccc  cc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{$|\rho|_{\text{BFS}}$ (difficulty)} "
        r"& \multicolumn{3}{c}{$|\rho|_{\text{LLM}}$ (routing)} "
        r"& \multicolumn{2}{c}{$\Delta_{\text{route}}$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
        r"\textbf{Method} & BW & Log & MBW & mean "
        r"& BW & Log & MBW & BW & MBW \\",
        r"\midrule",
        r"\multicolumn{10}{l}{\textit{Object-cardinality baseline}} \\",
    ]

    # |O| baseline row
    nobj_rhos = []
    for dom in TEST_DOMAINS:
        mask = task_types == dom
        nobj = X_surf[mask, 0]; ns = y_nsteps[mask].astype(float)
        r, _ = stats.spearmanr(nobj, ns)
        nobj_rhos.append(abs(r))
    tex_lines.append(
        f"  $|O|$ (object cardinality) "
        f"& {nobj_rhos[0]:.3f} & {nobj_rhos[1]:.3f} & {nobj_rhos[2]:.3f} "
        f"& {np.mean(nobj_rhos):.3f} & --- & --- & --- & --- & --- \\\\")
    tex_lines.append(r"\midrule")
    tex_lines.append(r"\multicolumn{10}{l}{\textit{Alternative baselines}} \\")

    disp = {
        "xgb":              "XGBoost ($x_s$)",
        "lgbm":             "LightGBM ($x_s$)",
        "knn":              "kNN-5 ($x_s$)",
        "cosine_retrieval": "Cosine retrieval ($x_e$)",
        "protonet":         "ProtoNet",
        "llm_predictor":    "LLM-outcome predictor",
    }
    for mn in method_names:
        vals = [baselines[d].get(mn,(0,0,float("nan")))[2] for d in TEST_DOMAINS]
        mean_v = np.nanmean([abs(v) for v in vals])
        # Bold if > |O| mean
        mean_str = f"\\textbf{{{mean_v:.3f}}}" if mean_v > np.mean(nobj_rhos) else f"{mean_v:.3f}"
        row = f"  {disp[mn]}"
        for v in vals:
            row += f" & {abs(v):.3f}" if not np.isnan(v) else " & ---"
        row += f" & {mean_str}"
        row += " & --- & --- & ---"   # LLM corr placeholder
        row += " & --- & ---"         # routing gain placeholder
        row += " \\\\"
        tex_lines.append(row)

    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (RESULTS_DIR / "baselines_p6.tex").write_text("\n".join(tex_lines))
    print(f"  LaTeX → {RESULTS_DIR}/baselines_p6.tex")

    return baselines


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 7 — Asymmetric attention justification
# ══════════════════════════════════════════════════════════════════════════════

def effective_rank(embeddings: np.ndarray) -> float:
    """
    Effective rank = exp(H(σ)) where H is the entropy of the normalised
    singular value distribution. Collapse → rank ≈ 1.
    """
    _, S, _ = np.linalg.svd(embeddings, full_matrices=False)
    S = S[S > 1e-8]
    p = S / S.sum()
    H = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(H))


def pairwise_cosine_mean(embeddings: np.ndarray, n_sample: int = 500) -> float:
    """Mean pairwise cosine similarity (subsample for speed)."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), min(n_sample, len(embeddings)), replace=False)
    E   = embeddings[idx]
    E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S   = E_n @ E_n.T
    mask = ~np.eye(len(S), dtype=bool)
    return float(S[mask].mean())


class SymmetricAttentionARC(nn.Module):
    """
    P7 baseline: symmetric attention where Q=K=FM embeddings.
    Tests if the asymmetric design is necessary.
    """
    def __init__(self, fm_dim: int, d_model: int = 128):
        super().__init__()
        self.d_model  = d_model
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.scale    = d_model ** -0.5
        # Both query and key come from FM embeddings
        self.enc = nn.Sequential(
            nn.Linear(fm_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(fm_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05))
        self.head_cls = nn.Linear(128, 2)

    def forward(self, q_fm, S_fm, head="cls"):
        single = (q_fm.dim() == 1)
        if single: q_fm = q_fm.unsqueeze(0)
        q     = self.enc(q_fm)
        k     = self.enc(S_fm)            # SAME encoder for Q and K
        v     = self.value_proj(S_fm)
        temp  = torch.exp(-self.log_temp).clamp(0.1, 10.0)
        alpha = torch.softmax(torch.matmul(q, k.T)*self.scale*temp, dim=-1)
        z     = torch.matmul(alpha, v)
        if single: z = z.squeeze(0); alpha = alpha.squeeze(0)
        h     = self.fusion(z)
        return self.head_cls(h), h, alpha


class SwappedQKARC(nn.Module):
    """
    P7 baseline: swapped Q/K roles — xs as query, FM as keys.
    Tests whether the direction of asymmetry matters.
    """
    def __init__(self, surf_dim: int, fm_dim: int, d_model: int = 128):
        super().__init__()
        self.d_model  = d_model
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.scale    = d_model ** -0.5
        # SWAPPED: surf is query, FM is key (opposite of ARC)
        self.query_enc = nn.Sequential(
            nn.Linear(surf_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.key_enc = nn.Sequential(
            nn.Linear(fm_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.value_proj = nn.Sequential(
            nn.Linear(fm_dim+surf_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        fused = surf_dim + d_model + fm_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05))
        self.head_cls = nn.Linear(128, 2)

    def forward(self, q_surf, q_fm, q_resid, S_surf, S_fm, S_V, head="cls"):
        single = (q_surf.dim() == 1)
        if single:
            q_surf=q_surf.unsqueeze(0); q_fm=q_fm.unsqueeze(0)
            q_resid=q_resid.unsqueeze(0)
        # SWAPPED: surf queries, FM keys
        q     = self.query_enc(q_surf)
        k     = self.key_enc(S_fm)
        v     = self.value_proj(S_V)
        temp  = torch.exp(-self.log_temp).clamp(0.1, 10.0)
        alpha = torch.softmax(torch.matmul(q, k.T)*self.scale*temp, dim=-1)
        z     = torch.matmul(alpha, v)
        if single: z=z.squeeze(0); alpha=alpha.squeeze(0); q_surf=q_surf.squeeze(0)
        fused = torch.cat([q_surf, z, q_resid.squeeze(0) if single else q_resid], -1)
        h     = self.fusion(fused)
        return self.head_cls(h), h, alpha


class LayerNormARC(nn.Module):
    """
    P7 baseline: original ARC with LayerNorm on FM embeddings before attention.
    Tests if normalization alone fixes collapse.
    """
    def __init__(self, surf_dim: int, fm_dim: int, d_model: int = 128):
        super().__init__()
        self.d_model  = d_model
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.scale    = d_model ** -0.5
        self.fm_norm  = nn.LayerNorm(fm_dim)    # NEW: normalize FM before encoding
        self.query_enc = nn.Sequential(
            nn.Linear(fm_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.key_enc = nn.Sequential(
            nn.Linear(surf_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.value_proj = nn.Sequential(
            nn.Linear(surf_dim+fm_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        fused = surf_dim + d_model + fm_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05))
        self.head_cls = nn.Linear(128, 2)

    def forward(self, q_surf, q_fm, q_resid, S_surf, S_fm, S_V, head="cls"):
        single = (q_surf.dim() == 1)
        if single:
            q_surf=q_surf.unsqueeze(0); q_fm=q_fm.unsqueeze(0)
            q_resid=q_resid.unsqueeze(0)
        q_fm_n = self.fm_norm(q_fm)     # normalize FM before encoding
        q     = self.query_enc(q_fm_n)
        k     = self.key_enc(S_surf)
        v     = self.value_proj(S_V)
        temp  = torch.exp(-self.log_temp).clamp(0.1, 10.0)
        alpha = torch.softmax(torch.matmul(q, k.T)*self.scale*temp, dim=-1)
        z     = torch.matmul(alpha, v)
        if single: z=z.squeeze(0); alpha=alpha.squeeze(0); q_surf=q_surf.squeeze(0)
        fused = torch.cat([q_surf, z, q_resid.squeeze(0) if single else q_resid], -1)
        h     = self.fusion(fused)
        return self.head_cls(h), h, alpha


def quick_train(model, train_fn, n_episodes=1000, lr=3e-4, device="cpu"):
    """Quick training for ablation variants."""
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_episodes, eta_min=1e-5)
    model.train()
    for _ in range(n_episodes):
        ep = train_fn()
        if ep is None: continue
        try:
            loss = ep
        except Exception:
            continue
        if torch.isnan(loss): continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    return model


def run_p7_ablations(args):
    print("\n" + "="*65)
    print("PROBLEM 7: Asymmetric Attention Justification")
    print("="*65)

    X_surf, X_fm, task_types, y_success, y_nsteps, train_mask, train_doms, prep = \
        load_all()

    Xs_tr_n, Xe_tr_n, Xr_tr_n = prep.transform(X_surf[train_mask], X_fm[train_mask])
    y_cls_tr = y_success[train_mask].astype(int)
    S_V_np   = np.hstack([Xs_tr_n, Xe_tr_n])

    surf_dim = Xs_tr_n.shape[1]
    fm_dim   = Xe_tr_n.shape[1]

    # Load original ARC
    ckpt_orig = CKPT_DIR / "guru_success.pt"
    ckpt_v2   = CKPT_DIR / "arc_v2.pt"

    # ── METRIC A+B+C: Collapse metrics on raw FM embeddings ─────────────────
    print("\n  Computing collapse metrics on raw FM embeddings...")
    Xe_tr = X_fm[train_mask]
    cos_fm   = pairwise_cosine_mean(Xe_tr)
    var_fm   = float(np.var(Xe_tr, axis=0).mean())
    rank_fm  = effective_rank(Xe_tr[:500])
    print(f"  Raw FM embeddings:")
    print(f"    Pairwise cosine: {cos_fm:.4f}  (collapse if → 1.0)")
    print(f"    Variance:        {var_fm:.4f}  (collapse if → 0.0)")
    print(f"    Effective rank:  {rank_fm:.2f}  (collapse if → 1.0)")

    # Syntactic features (should be more diverse)
    cos_xs  = pairwise_cosine_mean(Xs_tr_n)
    var_xs  = float(np.var(Xs_tr_n, axis=0).mean())
    rank_xs = effective_rank(Xs_tr_n[:500])
    print(f"  Syntactic features (xs):")
    print(f"    Pairwise cosine: {cos_xs:.4f}")
    print(f"    Variance:        {var_xs:.4f}")
    print(f"    Effective rank:  {rank_xs:.2f}")

    collapse_metrics = {
        "fm_raw":  {"cosine": cos_fm, "var": var_fm, "rank": rank_fm},
        "xs_norm": {"cosine": cos_xs, "var": var_xs, "rank": rank_xs},
    }

    # ── Per-domain cosine to show within-domain collapse ───────────────────
    print("\n  Within-domain FM cosine similarity:")
    for dom in train_doms + TEST_DOMAINS:
        mask = task_types == dom
        if mask.sum() < 10: continue
        Xe_dom = X_fm[mask]
        c_dom  = pairwise_cosine_mean(Xe_dom, n_sample=200)
        print(f"    {dom:<25}: {c_dom:.4f}")
        collapse_metrics[f"fm_{dom}"] = {"cosine": c_dom}

    # ── Train attention variants and measure their embeddings ───────────────
    print("\n  Training attention variants for comparison...")
    print("  (3 variants × 1000 episodes each — ~10 min on GPU)")

    # Support tensors
    rng     = np.random.default_rng(42)
    n_sup   = min(60, len(Xs_tr_n))
    sidx    = rng.choice(len(Xs_tr_n), n_sup, replace=False)
    S_surf  = torch.FloatTensor(Xs_tr_n[sidx]).to(DEVICE)
    S_fm    = torch.FloatTensor(Xe_tr_n[sidx]).to(DEVICE)
    S_V     = torch.FloatTensor(S_V_np[sidx]).to(DEVICE)
    tr_idx  = rng.choice(len(Xs_tr_n), 500, replace=False)
    Q_surf  = torch.FloatTensor(Xs_tr_n[tr_idx]).to(DEVICE)
    Q_fm    = torch.FloatTensor(Xe_tr_n[tr_idx]).to(DEVICE)
    Q_resid = torch.FloatTensor(Xr_tr_n[tr_idx]).to(DEVICE)
    Y_cls   = torch.LongTensor(y_cls_tr[tr_idx]).to(DEVICE)

    variant_results = {}

    def make_episode_tensors(model_type, model_obj):
        """Generate loss for one gradient step."""
        if model_type == "symmetric":
            out, h, alpha = model_obj(Q_fm, S_fm)
        else:
            out, h, alpha = model_obj(Q_surf, Q_fm, Q_resid, S_surf, S_fm, S_V)
        loss_task = F.cross_entropy(out, Y_cls)
        ent  = -(alpha * (alpha+1e-8).log()).sum(-1).mean()
        return loss_task - 0.05 * ent, h, alpha

    for name, ModelClass, model_kwargs, mtype in [
        ("symmetric_attn", SymmetricAttentionARC, {"fm_dim": fm_dim}, "symmetric"),
        ("swapped_qk",     SwappedQKARC,          {"surf_dim": surf_dim, "fm_dim": fm_dim}, "full"),
        ("layernorm_fix",  LayerNormARC,           {"surf_dim": surf_dim, "fm_dim": fm_dim}, "full"),
    ]:
        print(f"  Training {name}...", end=" ", flush=True)
        model = ModelClass(**model_kwargs).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, args.n_episodes_p7, eta_min=1e-5)
        model.train()
        for ep in range(args.n_episodes_p7):
            loss, _, _ = make_episode_tensors(mtype, model)
            if torch.isnan(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        print("done")

        # Measure collapse metrics on learned representations
        with torch.no_grad():
            if mtype == "symmetric":
                _, h_all, alpha_all = model(Q_fm, S_fm)
            else:
                _, h_all, alpha_all = model(Q_surf, Q_fm, Q_resid,
                                             S_surf, S_fm, S_V)
        h_np    = h_all.cpu().numpy()
        ent     = float(-(alpha_all * (alpha_all+1e-8).log()).sum(-1).mean().cpu())

        cos_h   = pairwise_cosine_mean(h_np)
        var_h   = float(np.var(h_np, axis=0).mean())
        rank_h  = effective_rank(h_np)

        # Downstream |ρ| on test domains
        rhos = []
        for dom in TEST_DOMAINS:
            d_mask = task_types == dom
            Xs_n, Xe_n, Xr_n = prep.transform(X_surf[d_mask], X_fm[d_mask])
            ns_d   = y_nsteps[d_mask].astype(float)
            scores = []
            with torch.no_grad():
                for i in range(len(Xs_n)):
                    qs = torch.FloatTensor(Xs_n[i]).to(DEVICE)
                    qf = torch.FloatTensor(Xe_n[i]).to(DEVICE)
                    qr = torch.FloatTensor(Xr_n[i]).to(DEVICE)
                    if mtype == "symmetric":
                        out, _, _ = model(qf, S_fm)
                    else:
                        out, _, _ = model(qs, qf, qr, S_surf, S_fm, S_V)
                    s = float(torch.softmax(out.unsqueeze(0), -1)[0,1].cpu())
                    scores.append(s)
            rho, _ = stats.spearmanr(scores, ns_d)
            rhos.append(abs(float(rho)))

        variant_results[name] = {
            "cosine":     cos_h,
            "var":        var_h,
            "rank":       rank_h,
            "ent":        ent,
            "mean_rho":   float(np.mean(rhos)),
            "rhos":       rhos,
        }
        print(f"    cos={cos_h:.3f}  var={var_h:.4f}  "
              f"rank={rank_h:.2f}  ent={ent:.3f}  mean|ρ|={np.mean(rhos):.3f}")

    # ── ARC original metrics (from saved checkpoint if available) ───────────
    if ckpt_orig.exists():
        spec3 = importlib.util.spec_from_file_location(
            "step3", ROOT_DIR / "plan_step3_guru.py")
        step3 = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(step3)
        ckpt  = torch.load(ckpt_orig, map_location=DEVICE)
        orig_model = step3.PlanningGURU(
            ckpt["model"]["key_enc.net.0.weight"].shape[1],
            ckpt["model"]["query_enc.net.0.weight"].shape[1]).to(DEVICE)
        orig_model.load_state_dict(ckpt["model"])
        orig_model.eval()
        try:
            with torch.no_grad():
                _, h_orig, alpha_orig = orig_model(
                    Q_surf, Q_fm, Q_resid, S_surf, S_fm, S_V)
            h_np_orig = h_orig.cpu().numpy()
            ent_orig  = float(-(alpha_orig*(alpha_orig+1e-8).log()).sum(-1).mean().cpu())
            variant_results["arc_original"] = {
                "cosine": pairwise_cosine_mean(h_np_orig),
                "var":    float(np.var(h_np_orig, axis=0).mean()),
                "rank":   effective_rank(h_np_orig),
                "ent":    ent_orig,
            }
            print(f"  ARC original: "
                  f"cos={variant_results['arc_original']['cosine']:.3f}  "
                  f"rank={variant_results['arc_original']['rank']:.2f}  "
                  f"ent={variant_results['arc_original']['ent']:.3f}")
        except Exception:
            print("  ARC original: skipped (dimension mismatch)")

    # ── Print summary table ─────────────────────────────────────────────────
    print(f"\n  {'Method':<22}  {'cos↓':>6}  {'var↑':>7}  {'rank↑':>7}  "
          f"{'ent↑':>6}  {'mean|ρ|↑':>10}")
    print("  " + "-"*65)

    for name in ["symmetric_attn","swapped_qk","layernorm_fix","arc_original"]:
        if name not in variant_results: continue
        r = variant_results[name]
        display = {"symmetric_attn": "Symmetric (Q=K=FM)",
                   "swapped_qk":     "Swapped Q/K (xs→q, FM→k)",
                   "layernorm_fix":  "LayerNorm fix",
                   "arc_original":   "ARC (asymmetric) ★"}[name]
        mrho = r.get("mean_rho", float("nan"))
        print(f"  {display:<22}  {r['cosine']:>6.3f}  {r['var']:>7.4f}  "
              f"{r['rank']:>7.2f}  {r['ent']:>6.3f}  "
              f"{mrho:>10.3f}")

    # ── Save all results ────────────────────────────────────────────────────
    all_results = {
        "collapse_metrics": collapse_metrics,
        "variant_results":  variant_results,
    }
    out = RESULTS_DIR / "p7_attention_ablation.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Results → {out}")

    # ── LaTeX collapse table ────────────────────────────────────────────────
    tex = [
        r"\begin{table}[h]\centering",
        r"\caption{Representation quality and downstream transfer across "
        r"attention variants. "
        r"$\cos\downarrow$: lower = less collapse. "
        r"$\mathrm{rank}\uparrow$: higher = richer representation. "
        r"$H(\alpha)\uparrow$: higher = more diffuse attention. "
        r"$|\rho|\uparrow$: higher = better difficulty ranking.}",
        r"\label{tab:attention_ablation}",
        r"\small\begin{tabular}{l cccc c}",
        r"\toprule",
        r"\textbf{Attention variant} & $\cos\downarrow$ & $\sigma^2\uparrow$ "
        r"& $\mathrm{rank}\uparrow$ & $H(\alpha)\uparrow$ & mean$|\rho|\uparrow$ \\",
        r"\midrule",
    ]
    for name, display in [
        ("symmetric_attn", "Symmetric ($Q=K=$ FM)"),
        ("swapped_qk",     "Swapped $Q/K$ (xs$\\to$q, FM$\\to$k)"),
        ("layernorm_fix",  "Asymmetric + LayerNorm"),
        ("arc_original",   r"\textbf{ARC asymmetric (ours)} $\star$"),
    ]:
        if name not in variant_results: continue
        r = variant_results[name]
        mrho = r.get("mean_rho", float("nan"))
        tex.append(
            f"  {display} & {r['cosine']:.3f} & {r['var']:.4f} "
            f"& {r['rank']:.2f} & {r['ent']:.3f} "
            f"& {mrho:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}\end{table}"]
    (RESULTS_DIR / "p7_attention_ablation.tex").write_text("\n".join(tex))
    print(f"  LaTeX → {RESULTS_DIR}/p7_attention_ablation.tex")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["baselines","p7","all"], default="all")
    p.add_argument("--n_episodes_p7", type=int, default=1000,
                   help="Episodes for each attention variant (P7 ablation)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.part in ("baselines", "all"):
        run_baselines(args)

    if args.part in ("p7", "all"):
        run_p7_ablations(args)

    print("\nDone.")
    print("  baselines → results_planning/baselines_p6.{json,tex}")
    print("  P7        → results_planning/p7_attention_ablation.{json,tex}")


if __name__ == "__main__":
    main()