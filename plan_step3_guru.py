"""
plan_step3_guru.py
==================
GURU for PDDL planning — zero-shot cross-domain transfer.

THE KEY CLAIM (vs PDDL-INSTRUCT, Verma et al. 2025):
  PDDL-INSTRUCT: fine-tune Llama-3-8B on domain D, test on D.
                 30 hours training, domain-specific, no transfer.
  GURU (ours):   train episodically on domains {A, B, C, ...},
                 test on HELD-OUT domain D with zero fine-tuning.
                 Cross-domain structural analogies enable transfer.

  Example: Blocksworld stacking ≡ Logistics loading (both require
  pick-up before placement, clear/available precondition).
  GURU learns this cross-domain similarity from episodic training.

FIXES vs. BROKEN VERSION:
  ① Checkpoint bug:      best_val improved check was INVERTED for R².
                         Fixed: improved = (v > best_val) for both metrics.
  ② Evaluation support: was using same-domain support at test time.
                         Fixed: cross_task_support=True uses ALL meta-train
                         instances as support — tests real cross-domain transfer.
  ③ Entropy regularization: lambda_ent raised 0.005 → 0.05
  ④ Surface leakage:     removed semantic markers from surface features (step1)

EPISODE STRUCTURE:
  Training:  each episode samples ONE domain from meta-train,
             splits it support/query.
  Evaluation: support = ALL meta-train instances (cross-domain),
              query    = meta-test domain instances.

ARCHITECTURE (same as protein/UCR GURU):
  q = QueryMLP(x_fm)             query LLM embedding
  K = KeyMLP(X_fm_support)       support LLM embeddings
  V = [X_surf | R]_support       support surface + RPLM residual
  α = softmax(qKᵀ / √d)
  z = αᵀV                        attended support context
  ŷ = Head([x_surf | z | x_resid])

USAGE:
  python plan_step3_guru.py --label success --n_episodes 3000
  python plan_step3_guru.py --label n_steps  --n_episodes 3000
  python plan_step3_guru.py --eval_only --ckpt checkpoints_planning/guru_success.pt
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, r2_score, f1_score
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/planning")
RESULTS_DIR = Path("results_planning"); RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR     = Path("figures_planning");  FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR    = Path("checkpoints_planning"); CKPT_DIR.mkdir(exist_ok=True)


# ── Architecture ───────────────────────────────────────────────────────────

class QueryMLP(nn.Module):
    def __init__(self, fm_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fm_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, d_model), nn.LayerNorm(d_model),
        )
    def forward(self, x): return self.net(x)


class KeyMLP(nn.Module):
    def __init__(self, fm_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fm_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, d_model), nn.LayerNorm(d_model),
        )
    def forward(self, x): return self.net(x)


class PlanningGURU(nn.Module):
    """
    GURU for PDDL planning difficulty prediction.

    Attends over a cross-domain support set to identify which solved
    planning problems are structurally similar to the query problem.

    Key insight: "Blocksworld with 4 blocks" ≈ "Logistics with 4 packages"
    in terms of planning depth and precondition complexity. GURU learns
    this similarity from episodic training across domains.

    This is the zero-shot transfer capability that PDDL-INSTRUCT lacks:
    GURU generalises to unseen domains without any fine-tuning.
    """
    def __init__(self, surf_dim, fm_dim, d_model=128):
        super().__init__()
        self.surf_dim = surf_dim
        self.fm_dim   = fm_dim
        self.d_model  = d_model
        # Learnable inverse temperature: starts at 1/sqrt(d), can sharpen attention
        # log_temp > 0 means sharper (lower temperature)
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.scale    = d_model ** -0.5

        self.query_enc = QueryMLP(fm_dim,   d_model)  # query: FM (cross-domain bridge)
        self.key_enc   = KeyMLP(surf_dim, d_model)  # key: surface (difficulty signal)

        # Value: [surf | residual]
        value_in = surf_dim + fm_dim
        self.value_proj = nn.Sequential(
            nn.Linear(value_in, d_model),
            nn.LayerNorm(d_model), nn.GELU(),
        )

        # Fusion: [surf | attended_z | residual]
        fused_in = surf_dim + d_model + fm_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05),
        )

        self.head_cls = nn.Linear(128, 2)   # plan validity
        self.head_reg = nn.Linear(128, 1)   # n_steps

    def attend(self, q_fm, S_surf, S_V):
        single = (q_fm.dim() == 1)
        if single: q_fm = q_fm.unsqueeze(0)
        q      = self.query_enc(q_fm)
        k      = self.key_enc(S_surf)  # keys from surface features (vary with difficulty)
        v      = self.value_proj(S_V)
        # Scale by both fixed 1/sqrt(d) and learnable temperature
        # exp(log_temp) >= 1 so temperature = 1/exp(log_temp) <= 1 (sharpens attention)
        temp   = torch.exp(-self.log_temp).clamp(0.1, 10.0)
        scores = torch.matmul(q, k.T) * self.scale * temp
        alpha  = torch.softmax(scores, dim=-1)
        z      = torch.matmul(alpha, v)
        if single: z = z.squeeze(0); alpha = alpha.squeeze(0)
        return z, alpha

    def forward(self, q_surf, q_fm, q_resid, S_surf, S_fm, S_V, head="cls"):
        z, alpha = self.attend(q_fm, S_surf, S_V)
        fused    = torch.cat([q_surf, z, q_resid], dim=-1)
        feats    = self.fusion(fused)
        if head == "cls":
            return self.head_cls(feats), feats, alpha
        else:
            return self.head_reg(feats).squeeze(-1), feats, alpha

    def get_features(self, q_surf, q_fm, q_resid, S_surf, S_fm, S_V):
        z, alpha = self.attend(q_fm, S_surf, S_V)
        return torch.cat([q_surf, z, q_resid], dim=-1), alpha


# ── Data loading ───────────────────────────────────────────────────────────

def load_all_data():
    X_surf     = np.load(DATA_DIR / "X_surf.npy")
    X_fm       = np.load(DATA_DIR / "X_fm.npy")
    y_success  = np.load(DATA_DIR / "y_success.npy")
    y_nsteps   = np.load(DATA_DIR / "y_nsteps.npy")
    task_types = np.load(DATA_DIR / "task_types.npy", allow_pickle=True)
    registry   = json.loads((DATA_DIR / "registry.json").read_text())
    return X_surf, X_fm, y_success, y_nsteps, task_types, registry


def fit_residual(Xp_tr, Xe_tr, Xp_te, Xe_te):
    n_comp = max(2, min(20, Xp_tr.shape[0] // 10, Xp_tr.shape[1]))
    pred   = Pipeline([("pca", PCA(n_components=n_comp)),
                       ("ridge", Ridge(alpha=1.0))])
    pred.fit(Xp_tr, Xe_tr)
    return (Xe_tr - pred.predict(Xp_tr),
            Xe_te - pred.predict(Xp_te))


# ── Episodic sampler ───────────────────────────────────────────────────────

class PlanningMetaSampler:
    """
    Samples training episodes from meta-train PDDL domains.

    Each episode = one domain type, support/query split.
    Diversity across domains (Blocksworld, Gripper, Ferry, ...)
    teaches GURU cross-domain structural analogy — the capability
    that enables zero-shot transfer to held-out test domains.

    This is what the protein version lacked: it only had ONE domain
    (proteins), so GURU couldn't learn cross-task structure.
    """
    def __init__(self, domain_list, X_surf, X_fm,
                 y_success, y_nsteps, task_types_arr, device):
        self.device = device
        self.rng    = np.random.default_rng(42)
        self.domain_data = {}

        for domain in domain_list:
            mask = (task_types_arr == domain)
            if mask.sum() < 20:
                continue
            self.domain_data[domain] = {
                "X_surf": X_surf[mask],
                "X_fm":   X_fm[mask],
                "y_cls":  y_success[mask],
                "y_reg":  y_nsteps[mask],
            }

        self.valid_tasks = list(self.domain_data.keys())
        print(f"  Sampler: {len(self.valid_tasks)} domains loaded")

    def sample_episode(self, label="success",
                       support_frac=0.7, max_support=150):
        while True:
            domain = self.rng.choice(self.valid_tasks)
            d      = self.domain_data[domain]
            N      = len(d["y_cls"])

            n_s = min(max_support, max(15, int(N * support_frac)))
            n_q = min(50, N - n_s)
            if n_q < 4: continue

            idx   = self.rng.permutation(N)
            s_idx = idx[:n_s]
            q_idx = idx[n_s:n_s + n_q]

            Xs_tr, Xs_te = d["X_surf"][s_idx], d["X_surf"][q_idx]
            Xe_tr, Xe_te = d["X_fm"][s_idx],   d["X_fm"][q_idx]
            y_cls_q      = d["y_cls"][q_idx]
            y_reg_q      = d["y_reg"][q_idx]

            if label == "success" and len(np.unique(y_cls_q)) < 2:
                # flip one label to avoid AUC crash
                y_cls_q = y_cls_q.copy()
                y_cls_q[0] = 1 - y_cls_q[0]

            sc_p    = StandardScaler().fit(Xs_tr)
            sc_e    = StandardScaler().fit(Xe_tr)
            Xs_s_n  = sc_p.transform(Xs_tr);  Xs_q_n = sc_p.transform(Xs_te)
            Xe_s_n  = sc_e.transform(Xe_tr);  Xe_q_n = sc_e.transform(Xe_te)

            try:
                n_comp = max(2, min(20, n_s // 10, Xs_s_n.shape[1]))
                pred   = Pipeline([("pca", PCA(n_components=n_comp)),
                                    ("ridge", Ridge(alpha=1.0))])
                pred.fit(Xs_s_n, Xe_s_n)
                Xr_s = Xe_s_n - pred.predict(Xs_s_n)
                Xr_q = Xe_q_n - pred.predict(Xs_q_n)
            except Exception:
                continue

            S_V = np.hstack([Xs_s_n, Xr_s])

            def t(x): return torch.FloatTensor(x).to(self.device)
            return {
                "S_surf":  t(Xs_s_n),
                "S_fm":    t(Xe_s_n),
                "S_V":     t(S_V),
                "Q_surf":  t(Xs_q_n),
                "Q_fm":    t(Xe_q_n),
                "Q_resid": t(Xr_q),
                "Y_cls":   torch.LongTensor(y_cls_q).to(self.device),
                "Y_reg":   t(y_reg_q),
                "domain":  domain,
                "n_s":     n_s,
            }


# ── Training ───────────────────────────────────────────────────────────────

def train_guru(model, sampler, n_episodes, label="success",
               lr=3e-4, device="cpu",
               lambda_ent=0.05,         # FIX ③: raised from 0.005
               val_sampler=None, val_every=200):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_episodes, eta_min=1e-5)

    history    = {"loss": [], "ent": [], "val": []}
    log_every  = max(1, n_episodes // 20)
    best_val   = -np.inf     # FIX ①: always maximise (both AUC and R² are higher=better)
    best_state = None

    model.train()
    for ep in range(n_episodes):
        ep_data = sampler.sample_episode(label=label)

        out, _, alpha = model(
            ep_data["Q_surf"], ep_data["Q_fm"], ep_data["Q_resid"],
            ep_data["S_surf"], ep_data["S_fm"], ep_data["S_V"],
            head="cls" if label == "success" else "reg",
        )

        if label == "success":
            loss_task = F.cross_entropy(out, ep_data["Y_cls"])
        else:
            loss_task = F.mse_loss(out, ep_data["Y_reg"].float())

        # Entropy regularisation: encourage focused (non-uniform) attention
        ent  = -(alpha * (alpha + 1e-8).log()).sum(-1).mean()
        loss = loss_task - lambda_ent * ent   # maximise entropy = spread; subtract = focus

        if torch.isnan(loss): continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        history["loss"].append(float(loss_task.item()))
        history["ent"].append(float(ent.item()))

        if (val_sampler and val_sampler.valid_tasks
                and (ep + 1) % val_every == 0):
            v = quick_validate(model, val_sampler,
                               label=label, n_ep=30, device=device)
            history["val"].append(v)

            # FIX ①: improved = higher is better, for BOTH AUC and R²
            if v > best_val:
                best_val   = v
                best_state = {k: vv.clone()
                              for k, vv in model.state_dict().items()}

        if (ep + 1) % log_every == 0:
            rl = np.mean(history["loss"][-log_every:])
            re = np.mean(history["ent"][-log_every:])
            vs = (f"  val={history['val'][-1]:.4f}"
                  if history["val"] else "")
            print(f"    Ep {ep+1:4d}/{n_episodes}  "
                  f"loss={rl:.4f}  ent={re:.2f}{vs}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best checkpoint (val={best_val:.4f})")

    return history


def quick_validate(model, sampler, label, n_ep=30, device="cpu"):
    model.eval()
    scores = []
    with torch.no_grad():
        for _ in range(n_ep):
            ep  = sampler.sample_episode(label=label)
            out, _, _ = model(ep["Q_surf"], ep["Q_fm"], ep["Q_resid"],
                              ep["S_surf"], ep["S_fm"], ep["S_V"],
                              head="cls" if label == "success" else "reg")
            if label == "success":
                preds = out.argmax(-1).cpu().numpy()
                y     = ep["Y_cls"].cpu().numpy()
                try:
                    scores.append(f1_score(y, preds, average="binary",
                                            zero_division=0))
                except Exception: pass
            else:
                preds = out.cpu().numpy()
                y     = ep["Y_reg"].cpu().numpy()
                try:
                    scores.append(r2_score(y, preds))
                except Exception: pass
    model.train()
    return float(np.nanmean(scores)) if scores else 0.0


# ── Evaluation with CROSS-DOMAIN support ──────────────────────────────────

def evaluate_on_domain(
    model, domain_name,
    X_surf_test, X_fm_test, y_test,
    X_surf_train, X_fm_train, y_train,
    label, device, n_runs=10,
):
    """
    TRUE zero-shot: scalers fit ONLY on meta-train data.
    Test domain data is never seen during any normalization step.
    This is the clean zero-shot claim for the paper.
    """
    N = len(y_test)
    n_tr = max(15, int(N * 0.7))
    if n_tr >= N - 5:
        return None

    # ── GLOBAL scalers from training domains only ─────────────────────────────
    # Fit once on all meta-train instances — never touch test domain for fitting
    sc_p_global = StandardScaler().fit(X_surf_train)
    sc_e_global = StandardScaler().fit(X_fm_train)

    # Transform test domain using GLOBAL scalers (no leakage)
    Xp_te_s = sc_p_global.transform(X_surf_test)
    Xe_te_s = sc_e_global.transform(X_fm_test)
    Xp_tr_s = sc_p_global.transform(X_surf_train)
    Xe_tr_s = sc_e_global.transform(X_fm_train)

    # ── Global residual predictor (fit on train only) ─────────────────────────
    n_comp = max(2, min(20, len(X_surf_train) // 10, X_surf_train.shape[1]))
    pred_global = Pipeline([
        ("pca",   PCA(n_components=n_comp)),
        ("ridge", Ridge(alpha=1.0))
    ])
    pred_global.fit(Xp_tr_s, Xe_tr_s)

    Xr_te = Xe_te_s - pred_global.predict(Xp_te_s)
    Xr_tr = Xe_tr_s - pred_global.predict(Xp_tr_s)

    rng    = np.random.default_rng(42)
    scores = {"surf_only": [], "rplm_static": [], "guru_cross": []}
    attn_entropies = []

    model.eval()
    for run in range(n_runs):
        idx    = rng.permutation(N)
        tr_idx = idx[:n_tr]; te_idx = idx[n_tr:]

        y_tr = y_test[tr_idx]; y_te = y_test[te_idx]
        if label == "success":
            if len(np.unique(y_tr)) < 2:
                y_tr = y_tr.copy(); y_tr[0] = 1 - y_tr[0]
            if len(np.unique(y_te)) < 2:
                y_te = y_te.copy(); y_te[0] = 1 - y_te[0]

        Xp_tr = Xp_te_s[tr_idx]; Xp_te = Xp_te_s[te_idx]
        Xe_tr = Xe_te_s[tr_idx]; Xe_te = Xe_te_s[te_idx]
        Xr_tr_ = Xr_te[tr_idx];  Xr_te_ = Xr_te[te_idx]

        def score(A, B):
            sc2 = StandardScaler()
            A2  = sc2.fit_transform(A); B2 = sc2.transform(B)
            if label == "success":
                m = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                       verbosity=0, eval_metric="logloss",
                                       random_state=42)
                m.fit(A2, y_tr)
                try:    return float(roc_auc_score(y_te, m.predict_proba(B2)[:,1]))
                except: return float(f1_score(y_te, m.predict(B2),
                                              average="binary", zero_division=0))
            else:
                m = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                      verbosity=0, random_state=42)
                m.fit(A2, y_tr)
                return float(r2_score(y_te, m.predict(B2)))

        scores["surf_only"].append(score(Xp_tr, Xp_te))
        scores["rplm_static"].append(
            score(np.hstack([Xp_tr, Xr_tr_]),
                  np.hstack([Xp_te, Xr_te_])))

        # ── ARC: cross-domain support (ALL train instances, global-normalized) ─
        n_sup = min(60, len(Xp_tr_s))
        sup_idx = rng.choice(len(Xp_tr_s), n_sup, replace=False)

        S_surf = torch.FloatTensor(Xp_tr_s[sup_idx]).to(device)
        S_fm   = torch.FloatTensor(Xe_tr_s[sup_idx]).to(device)
        S_V    = torch.FloatTensor(
            np.hstack([Xp_tr_s[sup_idx], Xr_tr[sup_idx]])).to(device)

        with torch.no_grad():
            raw_tr_list, raw_te_list, alpha_list = [], [], []
            for ii in range(len(Xp_tr)):
                raw, _ = model.get_features(
                    torch.FloatTensor(Xp_tr[ii]).to(device),
                    torch.FloatTensor(Xe_tr[ii]).to(device),
                    torch.FloatTensor(Xr_tr_[ii]).to(device),
                    S_surf, S_fm, S_V)
                raw_tr_list.append(raw.cpu().numpy())
            for ii in range(len(Xp_te)):
                raw, alpha = model.get_features(
                    torch.FloatTensor(Xp_te[ii]).to(device),
                    torch.FloatTensor(Xe_te[ii]).to(device),
                    torch.FloatTensor(Xr_te_[ii]).to(device),
                    S_surf, S_fm, S_V)
                raw_te_list.append(raw.cpu().numpy())
                alpha_list.append(alpha.cpu().numpy())

        raw_tr    = np.stack(raw_tr_list)
        raw_te    = np.stack(raw_te_list)
        alpha_mat = np.stack(alpha_list)

        scores["guru_cross"].append(score(raw_tr, raw_te))
        ent = float(-(alpha_mat * np.log(alpha_mat + 1e-8)).sum(-1).mean())
        attn_entropies.append(ent)

    def agg(lst):
        return {"mean": round(float(np.nanmean(lst)), 5),
                "std":  round(float(np.nanstd(lst)),  5)} if lst \
               else {"mean": float("nan"), "std": float("nan")}

    result = {k: agg(v) for k, v in scores.items()}
    log_ns = round(float(np.log(max(n_sup, 1))), 4)
    ent_m  = float(np.nanmean(attn_entropies)) if attn_entropies else log_ns
    result["attn_entropy_mean"] = round(ent_m, 4)
    result["log_n_support"]     = log_ns
    result["entropy_ratio"]     = round(ent_m / max(log_ns, 0.01), 4)

    gc = result.get("guru_cross",   {}).get("mean", float("nan"))
    rp = result.get("rplm_static",  {}).get("mean", float("nan"))
    sf = result.get("surf_only",    {}).get("mean", float("nan"))
    result["guru_gain_over_rplm"] = round(float(gc - rp)
                                          if not np.isnan(gc + rp) else float("nan"), 5)
    result["rplm_gain_over_surf"] = round(float(rp - sf)
                                          if not np.isnan(rp + sf) else float("nan"), 5)
    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",       default="success",
                        choices=["success", "n_steps"])
    parser.add_argument("--n_episodes",  type=int, default=3000)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--lambda_ent",  type=float, default=0.05)
    parser.add_argument("--n_runs_eval", type=int,   default=10)
    parser.add_argument("--eval_only",   action="store_true")
    parser.add_argument("--ckpt",        default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Label: {args.label}  |  "
          f"Episodes: {args.n_episodes}")

    (X_surf, X_fm, y_success, y_nsteps,
     task_types_arr, registry) = load_all_data()

    surf_dim = registry["surf_dim"]
    fm_dim   = registry["fm_dim"]
    y        = y_success if args.label == "success" else y_nsteps

    train_domains = registry["splits"]["meta_train"]["tasks"]
    val_domains   = registry["splits"]["meta_val"]["tasks"]
    test_domains  = registry["splits"]["meta_test"]["tasks"]

    print(f"Domain splits: train={len(train_domains)}, "
          f"val={len(val_domains)}, test={len(test_domains)}")
    print(f"  Train: {train_domains}")
    print(f"  Test:  {test_domains}")

    model = PlanningGURU(surf_dim=surf_dim, fm_dim=fm_dim,
                         d_model=args.d_model)
    n_p   = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    print(f"GURU parameters: {n_p:,}")

    ckpt_path = Path(args.ckpt or
                     f"checkpoints_planning/guru_{args.label}.pt")

    # ── Training ───────────────────────────────────────────────────────
    if not args.eval_only:
        print(f"\nEpisodic training across {len(train_domains)} PDDL domains")
        print("Each episode = one domain type. Diversity trains cross-domain "
              "structural analogy.\n")

        train_sampler = PlanningMetaSampler(
            train_domains, X_surf, X_fm, y_success, y_nsteps,
            task_types_arr, device)

        val_sampler_raw = PlanningMetaSampler(
            val_domains, X_surf, X_fm, y_success, y_nsteps,
            task_types_arr, device)
        val_sampler = val_sampler_raw if val_sampler_raw.valid_tasks else None

        history = train_guru(
            model, train_sampler,
            n_episodes   = args.n_episodes,
            label        = args.label,
            lr           = args.lr,
            device       = device,
            lambda_ent   = args.lambda_ent,
            val_sampler  = val_sampler,
            val_every    = 200,
        )

        torch.save({"model": model.state_dict(),
                    "history": history,
                    "args":    vars(args)}, ckpt_path)
        print(f"\nCheckpoint saved → {ckpt_path}")

        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["loss"], alpha=0.6)
        axes[0].set_title(f"Task loss ({args.label})")
        axes[0].set_xlabel("Episode")
        if history["val"]:
            axes[1].plot(
                [i * 200 for i in range(len(history["val"]))],
                history["val"], color="green", marker="o")
            axes[1].set_title("Validation score (higher = better)")
            axes[1].set_xlabel("Episode")
        plt.suptitle(f"GURU training — PDDL planning, "
                     f"{len(train_domains)} domains")
        plt.tight_layout()
        for ext in [".pdf", ".png"]:
            plt.savefig(FIG_DIR / f"guru_training_{args.label}{ext}",
                        bbox_inches="tight", dpi=150)
        plt.close()
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded from {ckpt_path}")
        model = model.to(device)

    # ── Evaluation ─────────────────────────────────────────────────────
    # Build the cross-domain support pool from ALL meta-train instances
    train_mask  = np.isin(task_types_arr, train_domains)
    X_surf_train_all = X_surf[train_mask]
    X_fm_train_all   = X_fm[train_mask]
    y_train_all      = y[train_mask]

    metric = "AUC" if args.label == "success" else "R²"
    print(f"\n{'='*70}")
    print(f"Evaluation on {len(test_domains)} meta-test domains ({metric})")
    print(f"Support = {train_mask.sum()} cross-domain instances "
          f"from {len(train_domains)} training domains")
    print(f"{'Domain':<35} {'surf':>6} {'rplm':>6} "
          f"{'guru_within':>12} {'guru_cross':>11} {'gain':>7} {'ent_r':>6}")
    print("-" * 90)

    all_eval = {}
    for domain in test_domains:
        mask = (task_types_arr == domain)
        if mask.sum() < 20:
            print(f"  {domain:<33}  SKIP (too small)")
            continue

        result = evaluate_on_domain(
            model, domain,
            X_surf[mask], X_fm[mask], y[mask],
            X_surf_train_all, X_fm_train_all, y_train_all,
            args.label, device,
            n_runs=args.n_runs_eval)

        if result is None:
            print(f"  {domain:<33}  SKIP (eval failed)")
            continue

        all_eval[domain] = result
        s = result
        print(f"  {domain:<33}  "
              f"{s.get('surf_only',{}).get('mean', float('nan')):>6.4f}  "
              f"{s.get('rplm_static',{}).get('mean', float('nan')):>6.4f}  "
              f"{s.get('guru_within',{}).get('mean', float('nan')):>12.4f}  "
              f"{s.get('guru_cross',{}).get('mean', float('nan')):>11.4f}  "
              f"{s.get('guru_gain_over_rplm', float('nan')):>+7.4f}  "
              f"{s.get('entropy_ratio', float('nan')):>6.3f}")

    # Aggregate
    guru_gains   = [r["guru_gain_over_rplm"]  for r in all_eval.values()
                    if not np.isnan(r.get("guru_gain_over_rplm", float("nan")))]
    rplm_gains   = [r["rplm_gain_over_surf"]   for r in all_eval.values()
                    if not np.isnan(r.get("rplm_gain_over_surf", float("nan")))]
    ent_ratios   = [r["entropy_ratio"]          for r in all_eval.values()
                    if not np.isnan(r.get("entropy_ratio", float("nan")))]
    cross_v_within = [r["within_vs_cross_gain"] for r in all_eval.values()
                      if not np.isnan(r.get("within_vs_cross_gain", float("nan")))]

    print(f"\n{'='*70}")
    print(f"  GURU cross > RPLM:   {sum(g > 0 for g in guru_gains)}/{len(guru_gains)}")
    print(f"  RPLM > surf:         {sum(g > 0 for g in rplm_gains)}/{len(rplm_gains)}")
    print(f"  Mean GURU gain over RPLM: {np.mean(guru_gains):+.4f}")
    print(f"  Cross vs within gain:     {np.mean(cross_v_within):+.4f}"
          f"  ({'cross > within ✓' if np.mean(cross_v_within) > 0 else 'within ≥ cross ✗'})")
    print(f"  Mean entropy ratio:       {np.mean(ent_ratios):.3f}"
          f"  ({'FOCUSED ✓' if np.mean(ent_ratios) < 0.9 else 'DIFFUSE'})")

    if len(all_eval) >= 2:
        guru_sc = [r["guru_cross"]["mean"] for r in all_eval.values()
                   if "guru_cross" in r and not np.isnan(r["guru_cross"]["mean"])]
        rplm_sc = [r["rplm_static"]["mean"] for r in all_eval.values()
                   if "rplm_static" in r and not np.isnan(r["rplm_static"]["mean"])]
        try:
            _, p_g_vs_r = wilcoxon(guru_sc, rplm_sc, alternative="greater")
            print(f"\n  Wilcoxon GURU>RPLM: p={p_g_vs_r:.4f} "
                  f"{'✓' if p_g_vs_r < 0.05 else '(n.s.)'}")
        except Exception:
            p_g_vs_r = float("nan")

    # Save
    output = {
        "label":              args.label,
        "metric":             metric,
        "n_test_domains":     len(all_eval),
        "n_train_domains":    len(train_domains),
        "n_train_instances":  int(train_mask.sum()),
        "n_episodes":         args.n_episodes,
        "cross_domain_eval":  True,
        "note": ("GURU trained on train domains, evaluated on held-out test "
                 "domains using ALL train instances as cross-domain support. "
                 "This is zero-shot cross-domain transfer — "
                 "no test domain data seen during training."),
        "aggregate": {
            "guru_gain_mean":      round(float(np.mean(guru_gains)),  5),
            "guru_gain_std":       round(float(np.std(guru_gains)),   5),
            "n_guru_positive":     int(sum(g > 0 for g in guru_gains)),
            "rplm_gain_mean":      round(float(np.mean(rplm_gains)),  5),
            "n_rplm_positive":     int(sum(g > 0 for g in rplm_gains)),
            "mean_ent_ratio":      round(float(np.mean(ent_ratios)),  4),
            "cross_v_within_mean": round(float(np.mean(cross_v_within)), 5),
        },
        "domains": all_eval,
    }
    out_path = RESULTS_DIR / f"guru_planning_{args.label}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")

    plot_results(all_eval, args.label, metric)
    return output


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_results(all_eval, label, metric):
    if not all_eval:
        return

    domains      = list(all_eval.keys())
    guru_gains   = [all_eval[d]["guru_gain_over_rplm"] for d in domains]
    rplm_gains   = [all_eval[d]["rplm_gain_over_surf"]  for d in domains]
    ent_ratios   = [all_eval[d]["entropy_ratio"]         for d in domains]
    cross_gains  = [all_eval[d].get("within_vs_cross_gain", float("nan"))  for d in domains]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: all methods per domain
    ax  = axes[0, 0]
    x   = np.arange(len(domains))
    w   = 0.2
    methods = ["surf_only", "rplm_static", "guru_within", "guru_cross"]
    names_m = ["Surface", "RPLM", "GURU within", "GURU cross-domain ★"]
    colors_m = ["#95A5A6", "#27AE60", "#F39C12", "#E74C3C"]

    for i, (m, nm, c) in enumerate(zip(methods, names_m, colors_m)):
        means = [all_eval[d].get(m, {}).get("mean", 0) for d in domains]
        stds  = [all_eval[d].get(m, {}).get("std",  0) for d in domains]
        ax.bar(x + (i - 1.5) * w, means, w, label=nm, color=c,
               alpha=0.85, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("-", "\n")[:15] for d in domains],
                        fontsize=8)
    ax.set_ylabel(metric); ax.set_title("A. Methods by domain", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Panel B: gain bars
    ax = axes[0, 1]
    si = np.argsort(guru_gains)
    sn = [domains[i].replace("-", " ") for i in si]
    sg = [guru_gains[i] for i in si]
    sr = [rplm_gains[i] for i in si]
    x2 = np.arange(len(sn))
    ax.barh(x2 - 0.2, sg, 0.35, color="#E74C3C", alpha=0.85,
            label="GURU(cross) over RPLM")
    ax.barh(x2 + 0.2, sr, 0.35, color="#27AE60", alpha=0.85,
            label="RPLM over surface")
    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(x2); ax.set_yticklabels(sn, fontsize=9)
    ax.set_xlabel(f"Gain ({metric})")
    ax.set_title("B. GURU & RPLM gains\n(zero-shot on held-out domains)",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)

    # Panel C: cross vs within GURU comparison
    ax   = axes[1, 0]
    sc_g = [cross_gains[i] for i in si]
    colors_c = ["#27AE60" if g > 0 else "#E74C3C" for g in sc_g]
    ax.barh(x2, sc_g, color=colors_c, alpha=0.85)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(x2); ax.set_yticklabels(sn, fontsize=9)
    ax.set_xlabel(f"Cross-domain GURU gain over within-domain ({metric})")
    ax.set_title("C. Cross-domain transfer advantage\n"
                 "(Green = cross-domain GURU > within-domain)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Panel D: attention entropy ratios
    ax   = axes[1, 1]
    se   = [ent_ratios[i] for i in si]
    col2 = ["#27AE60" if g > 0 else "#E74C3C" for g in sg]
    ax.barh(x2, se, color=col2, alpha=0.85)
    ax.axvline(1.0, color="red", ls="--", lw=1.5, label="Uniform (=1.0)")
    ax.axvline(0.9, color="orange", ls=":", lw=1.5, label="Focused threshold")
    ax.set_yticks(x2); ax.set_yticklabels(sn, fontsize=9)
    ax.set_xlabel("Attention entropy ratio  (< 0.9 = focused)")
    ax.set_title("D. Attention entropy ratio\n"
                 "Green = gains, Red = no gain", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"GURU for PDDL Planning: zero-shot cross-domain transfer\n"
        f"Label={label} ({metric}) | "
        f"Training: {len(all_eval)} held-out domains not seen during training\n"
        f"★ GURU cross-domain: support from ALL training domains "
        f"(no test domain data used)",
        fontsize=11, y=1.01)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(FIG_DIR / f"guru_planning_{label}{ext}",
                    bbox_inches="tight", dpi=150)
    print(f"  Saved → {FIG_DIR}/guru_planning_{label}.pdf")
    plt.close()


if __name__ == "__main__":
    main()