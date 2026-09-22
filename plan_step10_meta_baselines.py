"""
plan_step10_meta_baselines.py
==============================
Experiment 3: Proper meta-learning baselines.
  - Prototypical Networks (Snell et al., 2017) for difficulty prediction
  - MAML (Finn et al., 2017) applied to planning difficulty

Uses the SAME feature set as ARC (surf + FM residual) for a fair comparison.

Usage:
  python plan_step10_meta_baselines.py --baseline proto
  python plan_step10_meta_baselines.py --baseline maml
  python plan_step10_meta_baselines.py --baseline all
"""

import argparse
import json
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
import importlib.util

warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/planning")
RESULTS_DIR = Path("results_planning"); RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR    = Path("checkpoints_planning")

spec = importlib.util.spec_from_file_location("step3", Path(__file__).parent / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(step3)
PlanningMetaSampler = step3.PlanningMetaSampler
evaluate_on_domain  = step3.evaluate_on_domain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Prototypical Network for difficulty prediction
# ══════════════════════════════════════════════════════════════════════════════

class ProtoEncoder(nn.Module):
    def __init__(self, in_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, d_model), nn.LayerNorm(d_model),
        )
    def forward(self, x): return self.net(x)


class ProtoNet(nn.Module):
    """
    Prototypical Network adapted for regression/classification of planning difficulty.
    For classification: prototype = mean embedding per class.
    For regression: prototype = scalar prediction via weighted nearest-support.
    """
    def __init__(self, surf_dim, fm_dim, d_model=128):
        super().__init__()
        self.encoder = ProtoEncoder(surf_dim + fm_dim, d_model)

    def forward_cls(self, query_feats, support_feats, support_labels):
        q_emb  = self.encoder(query_feats)
        s_emb  = self.encoder(support_feats)
        # Compute class prototypes (mean embedding per class)
        classes = support_labels.unique()
        protos  = torch.stack([s_emb[support_labels == c].mean(0) for c in classes])
        # Distance to prototypes → log-softmax
        dists = torch.cdist(q_emb.unsqueeze(0), protos.unsqueeze(0)).squeeze(0)
        return -dists, classes  # logits (neg distance)

    def forward_reg(self, query_feats, support_feats, support_targets):
        q_emb = self.encoder(query_feats)
        s_emb = self.encoder(support_feats)
        # Soft nearest-neighbor regression
        dists = torch.cdist(q_emb, s_emb)
        weights = torch.softmax(-dists, dim=-1)
        preds = (weights * support_targets.unsqueeze(0)).sum(-1)
        return preds


def train_proto(n_episodes=3000, label="success"):
    X_surf = np.load(DATA_DIR / "X_surf.npy")
    X_fm   = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry = json.loads((DATA_DIR / "registry.json").read_text())
    y_success = np.load(DATA_DIR / "y_success.npy")
    y_nsteps  = np.load(DATA_DIR / "y_nsteps.npy")

    train_domains = registry["splits"]["meta_train"]["tasks"]
    val_domains   = registry["splits"]["meta_val"]["tasks"]
    surf_dim, fm_dim = X_surf.shape[1], X_fm.shape[1]

    sampler = PlanningMetaSampler(
        train_domains, X_surf, X_fm,
        y_success.astype(int), y_nsteps, task_types, DEVICE
    )

    model = ProtoNet(surf_dim, fm_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)

    best_loss, best_state = float("inf"), None
    print(f"\nTraining ProtoNet ({label}, {n_episodes} episodes)...")

    for ep in range(n_episodes):
        ep_data = sampler.sample_episode(label=label)
        # Concatenate surf + fm residual as input features
        s_feats = torch.cat([ep_data["S_surf"], ep_data["S_V"][:, X_surf.shape[1]:]], dim=-1)
        q_feats = torch.cat([ep_data["Q_surf"], ep_data["Q_resid"]], dim=-1)

        if label == "success":
            logits, classes = model.forward_cls(q_feats, s_feats, ep_data["Y_cls"])
            # Remap class indices
            y_remapped = torch.zeros_like(ep_data["Y_cls"])
            for ci, c in enumerate(classes):
                y_remapped[ep_data["Y_cls"] == c] = ci
            loss = F.cross_entropy(logits, y_remapped)
        else:
            preds = model.forward_reg(q_feats, s_feats, ep_data["Y_reg"])
            loss  = F.mse_loss(preds, ep_data["Y_reg"])

        if torch.isnan(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % (n_episodes // 10) == 0:
            print(f"  Ep {ep+1}/{n_episodes}  loss={loss.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)
    ckpt = CKPT_DIR / f"proto_{label}.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    print(f"  Saved → {ckpt}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAML for difficulty prediction
# ══════════════════════════════════════════════════════════════════════════════

class MAMLHead(nn.Module):
    """Small network for MAML inner-loop adaptation."""
    def __init__(self, in_dim, out_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)


def maml_inner_update(model, support_feats, support_labels, lr_inner=0.01,
                      n_steps=5, label="success"):
    """
    MAML inner loop: gradient steps on support set.
    Returns adapted parameters (does NOT modify original model).
    """
    fast_weights = {n: p.clone() for n, p in model.named_parameters()}

    for _ in range(n_steps):
        logits = _forward_with_weights(model, support_feats, fast_weights)
        if label == "success":
            loss = F.cross_entropy(logits, support_labels)
        else:
            loss = F.mse_loss(logits.squeeze(-1), support_labels.float())
        grads = torch.autograd.grad(loss, fast_weights.values(),
                                    create_graph=True, allow_unused=True)
        fast_weights = {
            n: p - lr_inner * (g if g is not None else torch.zeros_like(p))
            for (n, p), g in zip(fast_weights.items(), grads)
        }
    return fast_weights


def _forward_with_weights(model, x, weights):
    """Forward pass using custom weights dict (for MAML)."""
    # Simple two-layer forward with custom weights
    w1 = weights["net.0.weight"]; b1 = weights["net.0.bias"]
    w2 = weights["net.2.weight"]; b2 = weights["net.2.bias"]
    h  = F.relu(F.linear(x, w1, b1))
    return F.linear(h, w2, b2)


def train_maml(n_episodes=3000, label="success", lr_outer=1e-3, lr_inner=0.01, n_inner=5):
    X_surf = np.load(DATA_DIR / "X_surf.npy")
    X_fm   = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y_success  = np.load(DATA_DIR / "y_success.npy")
    y_nsteps   = np.load(DATA_DIR / "y_nsteps.npy")

    train_domains = registry["splits"]["meta_train"]["tasks"]
    sampler = PlanningMetaSampler(
        train_domains, X_surf, X_fm,
        y_success.astype(int), y_nsteps, task_types, DEVICE
    )

    in_dim  = X_surf.shape[1] + X_fm.shape[1]  # surf + residual
    out_dim = 2 if label == "success" else 1
    model   = MAMLHead(in_dim, out_dim).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr_outer)

    print(f"\nTraining MAML ({label}, {n_episodes} episodes)...")
    best_loss, best_state = float("inf"), None

    for ep in range(n_episodes):
        ep_data = sampler.sample_episode(label=label)
        s_feats = torch.cat([ep_data["S_surf"], ep_data["S_V"][:, X_surf.shape[1]:]], dim=-1)
        q_feats = torch.cat([ep_data["Q_surf"], ep_data["Q_resid"]], dim=-1)

        y_s = ep_data["Y_cls"] if label == "success" else ep_data["Y_reg"]
        y_q = ep_data["Y_cls"] if label == "success" else ep_data["Y_reg"]

        # Inner loop: adapt on support set
        fast_w = maml_inner_update(model, s_feats, y_s,
                                   lr_inner=lr_inner, n_steps=n_inner, label=label)
        # Outer loop: evaluate on query set with adapted weights
        q_logits = _forward_with_weights(model, q_feats, fast_w)
        if label == "success":
            outer_loss = F.cross_entropy(q_logits, y_q)
        else:
            outer_loss = F.mse_loss(q_logits.squeeze(-1), y_q.float())

        if torch.isnan(outer_loss): continue
        opt.zero_grad(); outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if outer_loss.item() < best_loss:
            best_loss  = outer_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % (n_episodes // 10) == 0:
            print(f"  Ep {ep+1}/{n_episodes}  outer_loss={outer_loss.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)
    ckpt = CKPT_DIR / f"maml_{label}.pt"
    torch.save({"model": model.state_dict(), "in_dim": in_dim, "out_dim": out_dim}, ckpt)
    print(f"  Saved → {ckpt}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation: compare ProtoNet, MAML, and ARC
# ══════════════════════════════════════════════════════════════════════════════

def eval_all_baselines(label="success"):
    X_surf = np.load(DATA_DIR / "X_surf.npy")
    X_fm   = np.load(DATA_DIR / "X_fm.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    y = np.load(DATA_DIR / "y_success.npy") if label == "success" \
        else np.load(DATA_DIR / "y_nsteps.npy")

    test_domains  = registry["splits"]["meta_test"]["tasks"]
    train_domains = registry["splits"]["meta_train"]["tasks"]
    train_mask    = np.isin(task_types, train_domains)
    metric = "AUC" if label == "success" else "R²"

    # Load ARC checkpoint
    from step3 import PlanningGURU as ARCM
    arc_ckpt = CKPT_DIR / f"guru_{label}.pt"
    arc = None
    if arc_ckpt.exists():
        ck = torch.load(arc_ckpt, map_location=DEVICE)
        arc = ARCM(X_surf.shape[1], X_fm.shape[1]).to(DEVICE)
        arc.load_state_dict(ck["model"]); arc.eval()

    results = {}
    print(f"\n{'Domain':<25} {'ProtoNet':>10} {'MAML':>10} {'ARC':>10}  ({metric})")
    print("-" * 60)

    for dom in test_domains:
        mask = task_types == dom
        y_q  = y[mask]; y_tr = y[train_mask]
        row  = {}

        # ARC score
        if arc is not None:
            res = evaluate_on_domain(
                arc, dom, X_surf[mask], X_fm[mask], y_q,
                X_surf[train_mask], X_fm[train_mask], y_tr,
                label=label, device=DEVICE, cross_domain_support=True
            )
            row["arc"] = res.get("guru_cross", {}).get("mean", float("nan")) if res else float("nan")

        # ProtoNet and MAML: simple XGBoost probe on their learned embeddings
        for baseline, ckpt_name in [("proto", f"proto_{label}.pt"), ("maml", f"maml_{label}.pt")]:
            ckpt_path = CKPT_DIR / ckpt_name
            if not ckpt_path.exists():
                row[baseline] = float("nan"); continue

            ck = torch.load(ckpt_path, map_location=DEVICE)
            if baseline == "proto":
                m = ProtoNet(X_surf.shape[1], X_fm.shape[1]).to(DEVICE)
            else:
                in_dim  = ck["in_dim"]; out_dim = ck["out_dim"]
                m = MAMLHead(in_dim, out_dim).to(DEVICE)
            m.load_state_dict(ck["model"]); m.eval()

            sc_s = StandardScaler().fit(X_surf[train_mask])
            sc_e = StandardScaler().fit(X_fm[train_mask])
            Xs_s = sc_s.transform(X_surf[train_mask]); Xs_q = sc_s.transform(X_surf[mask])
            Xe_s = sc_e.transform(X_fm[train_mask]);   Xe_q = sc_e.transform(X_fm[mask])

            # Compute residuals
            n_comp = max(2, min(20, len(Xs_s) // 10, Xs_s.shape[1]))
            pred = Pipeline([("pca", PCA(n_comp)), ("ridge", Ridge())])
            pred.fit(Xs_s, Xe_s)
            Xr_s = Xe_s - pred.predict(Xs_s)
            Xr_q = Xe_q - pred.predict(Xs_q)
            feats_s = np.hstack([Xs_s, Xr_s]); feats_q = np.hstack([Xs_q, Xr_q])

            with torch.no_grad():
                if baseline == "proto":
                    emb_s = m.encoder(torch.FloatTensor(feats_s).to(DEVICE)).cpu().numpy()
                    emb_q = m.encoder(torch.FloatTensor(feats_q).to(DEVICE)).cpu().numpy()
                else:
                    # MAML: forward through net.0 layer for embedding
                    w1 = m.net[0].weight.data.cpu().numpy()
                    b1 = m.net[0].bias.data.cpu().numpy()
                    emb_s = np.maximum(0, feats_s @ w1.T + b1)
                    emb_q = np.maximum(0, feats_q @ w1.T + b1)

            import xgboost as xgb
            sc2 = StandardScaler()
            A = sc2.fit_transform(emb_s); B = sc2.transform(emb_q)
            if label == "success":
                clf = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                        verbosity=0, eval_metric="logloss",
                                        random_state=42)
                clf.fit(A, y_tr.astype(int))
                try:
                    score = roc_auc_score(y_q, clf.predict_proba(B)[:, 1])
                except:
                    score = float("nan")
            else:
                reg = xgb.XGBRegressor(n_estimators=200, max_depth=4, verbosity=0, random_state=42)
                reg.fit(A, y_tr)
                score = r2_score(y_q, reg.predict(B))
            row[baseline] = float(score)

        results[dom] = row
        print(f"  {dom:<23} {row.get('proto', float('nan')):>10.4f} "
              f"{row.get('maml', float('nan')):>10.4f} {row.get('arc', float('nan')):>10.4f}")

    (RESULTS_DIR / f"e3_meta_baselines_{label}.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → results_planning/e3_meta_baselines_{label}.json")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["proto", "maml", "all"], default="all")
    parser.add_argument("--label",    choices=["success", "n_steps"], default="success")
    parser.add_argument("--episodes", type=int, default=3000)
    args = parser.parse_args()

    if args.baseline in ("proto", "all"):
        train_proto(n_episodes=args.episodes, label=args.label)
    if args.baseline in ("maml", "all"):
        train_maml(n_episodes=args.episodes, label=args.label)
    eval_all_baselines(label=args.label)


if __name__ == "__main__":
    main()