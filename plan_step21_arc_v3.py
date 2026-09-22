"""
plan_step21_arc_v3.py
======================
ARC v3: three architectural changes that directly address why zero-shot gains are small.

CHANGE 1 — Contrastive loss as PRIMARY objective
  Old: L = L_bfs + 0.3*L_reg + 0.1*L_gate
  New: L = L_contrastive + 0.1*L_bfs + 0.1*L_reg
  Rationale: step16 ablation proved contrastive alone gives 0.670 mean |ρ|
  vs baseline 0.347. The main model never fully leveraged this.

CHANGE 2 — Cross-domain contrastive alignment
  Old contrastive: positives/negatives within same episode (same domain)
  New: explicitly sample cross-domain pairs:
    Positive = instance from DIFFERENT domain with similar n_steps (|Δ| < 2)
    Negative = instance from DIFFERENT domain with very different n_steps (|Δ| > 6)
  This directly trains the representation to be domain-invariant at equal difficulty.
  Fixes Mystery-BW instability where the model fails to map across domains.

CHANGE 3 — Pairwise ranking loss instead of regression
  Old: MSE(predicted_nsteps, true_nsteps) — optimises absolute values
  New: L_rank = -log σ(f(x_i) - f(x_j)) for all pairs where n_steps_i > n_steps_j
  Directly optimises Spearman ρ (ranking), which is the evaluation metric.
  Much stronger cross-domain transfer than regression.

USAGE:
  python plan_step21_arc_v3.py --phase train
  python plan_step21_arc_v3.py --phase eval
  python plan_step21_arc_v3.py --phase compare   # v2 vs v3 side-by-side
  python plan_step21_arc_v3.py --phase all
"""

from __future__ import annotations
import argparse, importlib.util, json, pickle, sys, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)
CKPT    = ROOT / "checkpoints_planning"; CKPT.mkdir(exist_ok=True)

TEST_DOMAINS  = ["blocksworld", "logistics", "mystery_blocksworld"]
TRAIN_DOMAINS = ["depot", "rovers", "satellite"]   # ferry optional if generated
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Load dependencies
# ══════════════════════════════════════════════════════════════════════════════

def load_deps():
    spec17 = importlib.util.spec_from_file_location("step17", ROOT/"plan_step17_arc_v2.py")
    s17    = importlib.util.module_from_spec(spec17); spec17.loader.exec_module(s17)
    sys.modules["step17"] = s17
    sys.modules["__main__"].GlobalPreprocessor = s17.GlobalPreprocessor

    spec6 = importlib.util.spec_from_file_location("step6", ROOT/"plan_step6_pddlinst_gate.py")
    s6    = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)

    prep_path = RESULTS / "global_preprocessor.pkl"
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)

    X_surf, X_fm, task_types, y_success, y_nsteps, splits = s6.load_data(data_dir=DATA)
    X_surf = X_surf[:, :-1]  # remove domain hash

    # Update train_domains if ferry exists
    if "ferry" in set(task_types):
        global TRAIN_DOMAINS
        TRAIN_DOMAINS = ["depot", "rovers", "satellite", "ferry"]
        print(f"  Ferry found — using 4 train domains")

    return s17, prep, X_surf, X_fm, task_types, y_success, y_nsteps


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 3 — Pairwise ranking loss
# ══════════════════════════════════════════════════════════════════════════════

class PairwiseRankingLoss(nn.Module):
    """
    Bradley-Terry pairwise ranking loss.
    For all pairs (i,j) where n_steps_i > n_steps_j:
      L = -log σ(score_i - score_j)
    score_i > score_j means "i is harder than j"

    Directly optimises Spearman ρ ranking, unlike MSE which optimises
    absolute value prediction.

    n_pairs_per_batch: subsample pairs to keep O(N) not O(N²)
    """
    def __init__(self, n_pairs: int = 64, margin: float = 0.0):
        super().__init__()
        self.n_pairs = n_pairs
        self.margin  = margin

    def forward(self, scores: torch.Tensor, n_steps: torch.Tensor) -> torch.Tensor:
        N = len(scores)
        if N < 2:
            return torch.tensor(0.0, device=scores.device)

        # Sample pairs where n_steps differs
        n_pairs = min(self.n_pairs, N * (N - 1) // 2)
        i_idx   = torch.randint(0, N, (n_pairs,), device=scores.device)
        j_idx   = torch.randint(0, N, (n_pairs,), device=scores.device)
        same    = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % N

        diff_ns = n_steps[i_idx] - n_steps[j_idx]   # > 0 means i harder
        diff_sc = scores[i_idx]  - scores[j_idx]     # should also be > 0

        # Only keep pairs with actual difficulty difference
        valid = diff_ns.abs() > 0.5
        if valid.sum() < 2:
            return torch.tensor(0.0, device=scores.device)

        # sign: +1 if i harder, -1 if j harder
        sign  = diff_ns[valid].sign()
        loss  = -F.logsigmoid(sign * (diff_sc[valid] - self.margin))
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2 — Cross-domain contrastive loss
# ══════════════════════════════════════════════════════════════════════════════

class CrossDomainContrastiveLoss(nn.Module):
    """
    Cross-domain InfoNCE: for each anchor from domain A, find positives
    from DIFFERENT domains with similar difficulty and negatives from
    different domains with very different difficulty.

    This forces the QueryMLP to produce embeddings where cross-domain
    instances of equal difficulty are close — directly addressing
    the Mystery-BW transfer failure.

    pos_thresh: |Δn_steps| < pos_thresh → same difficulty across domains
    neg_thresh: |Δn_steps| > neg_thresh → different difficulty
    temperature: contrastive temperature
    """
    def __init__(self, pos_thresh=2.0, neg_thresh=6.0, temperature=0.07):
        super().__init__()
        self.pos_thresh  = pos_thresh
        self.neg_thresh  = neg_thresh
        self.temperature = temperature

    def forward(self,
                embeddings: torch.Tensor,   # (B, d) query embeddings
                n_steps:    torch.Tensor,   # (B,)
                domains:    list,           # list of domain strings, len=B
               ) -> torch.Tensor:
        B = len(embeddings)
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device)

        q = F.normalize(embeddings, p=2, dim=-1)   # (B, d)
        sim = torch.matmul(q, q.T) / self.temperature  # (B, B)

        ns   = n_steps.float()
        diff = (ns.unsqueeze(1) - ns.unsqueeze(0)).abs()   # (B, B)

        # Cross-domain mask: True when i and j are from DIFFERENT domains
        diff_dom = torch.tensor(
            [[domains[i] != domains[j] for j in range(B)] for i in range(B)],
            dtype=torch.bool, device=embeddings.device)

        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)

        # Cross-domain positives: different domain, similar difficulty
        pos_mask = diff_dom & (diff < self.pos_thresh) & ~eye
        # Cross-domain negatives: different domain, very different difficulty
        neg_mask = diff_dom & (diff > self.neg_thresh)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        loss_terms = []
        for i in range(B):
            pos_i = pos_mask[i]
            neg_i = neg_mask[i]
            if pos_i.sum() == 0 or neg_i.sum() == 0:
                continue
            sim_pos = sim[i][pos_i].exp().mean()
            sim_neg = sim[i][neg_i].exp().mean()
            if sim_pos + sim_neg < 1e-8:
                continue
            loss_terms.append(-torch.log(sim_pos / (sim_pos + sim_neg) + 1e-8))

        if not loss_terms:
            return torch.tensor(0.0, device=embeddings.device)
        return torch.stack(loss_terms).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Multi-domain episode sampler (needed for cross-domain contrastive)
# ══════════════════════════════════════════════════════════════════════════════

class MultiDomainBatch:
    """
    For contrastive training: sample a batch spanning MULTIPLE domains.
    Each batch contains instances from all training domains, enabling
    cross-domain positive/negative pair construction.
    """
    def __init__(self, X_surf, X_fm, y_nsteps, task_types,
                 train_domains, prep, n_per_domain=8, device="cpu"):
        self.prep          = prep
        self.train_domains = train_domains
        self.n_per_domain  = n_per_domain
        self.device        = device

        # Index by domain
        self.domain_data = {}
        for dom in train_domains:
            mask = task_types == dom
            if mask.sum() == 0:
                continue
            Xs_n, Xe_n, Xr_n = prep.transform(X_surf[mask], X_fm[mask])
            self.domain_data[dom] = {
                "Xs": Xs_n, "Xe": Xe_n, "Xr": Xr_n,
                "ns": y_nsteps[mask].astype(float),
            }

    def sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        all_Xs, all_Xe, all_Xr, all_ns, all_doms = [], [], [], [], []
        for dom, d in self.domain_data.items():
            N    = len(d["Xs"])
            n    = min(self.n_per_domain, N)
            idx  = rng.choice(N, n, replace=False)
            all_Xs.append(d["Xs"][idx])
            all_Xe.append(d["Xe"][idx])
            all_Xr.append(d["Xr"][idx])
            all_ns.append(d["ns"][idx])
            all_doms.extend([dom] * n)

        def t(x): return torch.FloatTensor(x).to(self.device)
        return {
            "Q_surf":   t(np.vstack(all_Xs)),
            "Q_fm":     t(np.vstack(all_Xe)),
            "Q_resid":  t(np.vstack(all_Xr)),
            "Y_reg":    t(np.concatenate(all_ns)),
            "domains":  all_doms,
        }


def build_support(X_surf, X_fm, task_types, prep, train_domains, device):
    """Build the global support set (same as step17)."""
    mask  = np.isin(task_types, train_domains)
    Xs_n, Xe_n, _ = prep.transform(X_surf[mask], X_fm[mask])
    rng   = np.random.default_rng(42)
    n_s   = min(60, mask.sum())
    sidx  = rng.choice(mask.sum(), n_s, replace=False)
    S_V   = np.hstack([Xs_n[sidx], Xe_n[sidx]])
    return (torch.FloatTensor(Xs_n[sidx]).to(device),
            torch.FloatTensor(Xe_n[sidx]).to(device),
            torch.FloatTensor(S_V).to(device))


# ══════════════════════════════════════════════════════════════════════════════
# Training with contrastive-primary objective
# ══════════════════════════════════════════════════════════════════════════════

def phase_train(args, s17, prep, X_surf, X_fm, task_types, y_success, y_nsteps):
    print("\n" + "="*65)
    print("ARC v3 TRAINING")
    print("  Change 1: contrastive is PRIMARY objective")
    print("  Change 2: cross-domain contrastive alignment")
    print("  Change 3: pairwise ranking loss replaces regression")
    print("="*65)

    surf_dim  = X_surf.shape[1]
    fm_dim    = X_fm.shape[1]
    print(f"  surf_dim={surf_dim}  fm_dim={fm_dim}  train_domains={TRAIN_DOMAINS}")

    model     = s17.ARCv2(surf_dim, fm_dim).to(DEVICE)
    opt       = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.n_episodes, eta_min=1e-5)

    # Loss functions
    cross_dom_contrast = CrossDomainContrastiveLoss(
        pos_thresh=2.0, neg_thresh=6.0, temperature=0.07)
    ranking_loss       = PairwiseRankingLoss(n_pairs=64)
    pu_loss_fn         = s17.NNPULoss(
        prior=float((y_nsteps[np.isin(task_types, TRAIN_DOMAINS)] <= 12).mean()))

    # Loss weights (CHANGE 1: contrastive dominates)
    lam_contrast = 1.0    # PRIMARY
    lam_bfs      = 0.1    # secondary
    lam_rank     = 0.2    # ranking replaces regression (CHANGE 3)
    lam_gate     = 0.3
    lam_ent      = 0.05

    # Multi-domain sampler (CHANGE 2: cross-domain batches)
    multi_sampler = MultiDomainBatch(
        X_surf, X_fm, y_nsteps, task_types,
        TRAIN_DOMAINS, prep,
        n_per_domain=max(8, 32 // len(TRAIN_DOMAINS)),
        device=DEVICE)

    # Single-domain sampler for BFS head
    spec1 = importlib.util.spec_from_file_location("step3", ROOT/"plan_step3_guru.py")
    s3    = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(s3)
    bfs_sampler = s3.PlanningMetaSampler(
        TRAIN_DOMAINS, X_surf, X_fm, y_success, y_nsteps, task_types, DEVICE)

    rng        = np.random.default_rng(args.seed)
    history    = {"loss_contrast":[], "loss_rank":[], "loss_bfs":[], "gate_c":[]}
    log_every  = max(1, args.n_episodes // 20)

    model.train()
    for ep in range(args.n_episodes):
        # ── CHANGE 2: cross-domain batch ──────────────────────────────────────
        batch = multi_sampler.sample(rng)
        Q_surf = batch["Q_surf"]
        Q_fm   = batch["Q_fm"]
        Q_resid= batch["Q_resid"]
        Y_reg  = batch["Y_reg"]
        domains= batch["domains"]

        # Get support set for attention
        S_surf, S_fm, S_V = build_support(X_surf, X_fm, task_types,
                                           prep, TRAIN_DOMAINS, DEVICE)

        # Forward pass — regression head
        out_reg, h, alpha, c = model(Q_surf, Q_fm, Q_resid,
                                     S_surf, S_fm, S_V,
                                     head="reg", return_gate=True)

        # ── CHANGE 1+2: cross-domain contrastive on query embeddings ──────────
        q_embeds = model.query_enc(Q_fm)  # (B, d) — raw QueryMLP output
        loss_c   = cross_dom_contrast(q_embeds, Y_reg, domains)

        # ── CHANGE 3: pairwise ranking loss ───────────────────────────────────
        loss_r   = ranking_loss(out_reg, Y_reg)

        # BFS head (light weight)
        ep_bfs = bfs_sampler.sample_episode(label="success")
        if ep_bfs is not None:
            out_bfs, h_bfs, alpha_bfs, c_bfs = model(
                ep_bfs["Q_surf"], ep_bfs["Q_fm"], ep_bfs["Q_resid"],
                ep_bfs["S_surf"], ep_bfs["S_fm"], ep_bfs["S_V"],
                head="bfs", return_gate=True)
            # PU loss with cap-exceeded mask
            cap_mask = ep_bfs["Y_reg"] > 12
            loss_bfs = pu_loss_fn(out_bfs, ep_bfs["Y_cls"], cap_mask)
        else:
            loss_bfs = torch.tensor(0.0, device=DEVICE)

        # Gate regularisation
        loss_gate = F.mse_loss(c, torch.zeros_like(c))   # push toward 0 (complex)

        # Entropy regularisation
        ent  = -(alpha * (alpha + 1e-8).log()).sum(-1).mean()

        # CHANGE 1: contrastive is dominant
        loss = (lam_contrast * loss_c
                + lam_rank    * loss_r
                + lam_bfs     * loss_bfs
                + lam_gate    * loss_gate
                - lam_ent     * ent)

        if torch.isnan(loss):
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        history["loss_contrast"].append(float(loss_c.item()))
        history["loss_rank"].append(float(loss_r.item()))
        history["loss_bfs"].append(float(loss_bfs.item()))
        history["gate_c"].append(float(c.mean().item()))

        if (ep + 1) % log_every == 0:
            n = log_every
            print(f"  ep {ep+1}/{args.n_episodes}  "
                  f"contrast={np.mean(history['loss_contrast'][-n:]):.4f}  "
                  f"rank={np.mean(history['loss_rank'][-n:]):.4f}  "
                  f"bfs={np.mean(history['loss_bfs'][-n:]):.4f}  "
                  f"gate_c={np.mean(history['gate_c'][-n:]):.3f}")

    torch.save({"model": model.state_dict(),
                "surf_dim": surf_dim, "fm_dim": fm_dim},
               CKPT / "arc_v3.pt")
    print(f"  Saved → {CKPT}/arc_v3.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, prep, X_surf, X_fm, task_types, y_nsteps,
             label="v3", S_surf=None, S_fm=None, S_V=None):
    model.eval()
    if S_surf is None:
        S_surf, S_fm, S_V = build_support(
            X_surf, X_fm, task_types, prep, TRAIN_DOMAINS, DEVICE)

    results = {}
    print(f"\n  [{label}]")
    print(f"  {'Domain':<22}  {'|ρ|':>8}  {'R²':>8}  {'AUC':>8}")
    print("  " + "-"*50)

    for dom in TEST_DOMAINS:
        mask  = task_types == dom
        Xs_n, Xe_n, Xr_n = prep.transform(X_surf[mask], X_fm[mask])
        ns_q  = y_nsteps[mask].astype(float)

        scores = []
        with torch.no_grad():
            for i in range(len(Xs_n)):
                qs = torch.FloatTensor(Xs_n[i]).unsqueeze(0).to(DEVICE)
                qf = torch.FloatTensor(Xe_n[i]).unsqueeze(0).to(DEVICE)
                qr = torch.FloatTensor(Xr_n[i]).unsqueeze(0).to(DEVICE)
                out, _, _ = model(qs, qf, qr, S_surf, S_fm, S_V, head="reg")
                scores.append(float(out.squeeze().cpu()))
        scores = np.array(scores)

        rho, _  = stats.spearmanr(scores, ns_q)
        ss_res  = float(np.sum((ns_q - scores) ** 2))
        ss_tot  = float(np.sum((ns_q - ns_q.mean()) ** 2))
        r2      = float(1 - ss_res / max(ss_tot, 1e-8))
        try:
            y_bin = (ns_q <= np.median(ns_q)).astype(int)
            auc   = float(roc_auc_score(y_bin, -scores))
        except Exception:
            auc   = float("nan")

        print(f"  {dom:<22}  {abs(rho):>8.4f}  {r2:>8.4f}  {auc:>8.4f}")
        results[dom] = {"rho": float(rho), "r2": r2, "auc": auc}

    mean_rho = np.mean([abs(r["rho"]) for r in results.values()])
    print(f"  {'mean |ρ|':<22}  {mean_rho:>8.4f}")
    results["mean_rho"] = float(mean_rho)
    return results


def phase_eval(args, s17, prep, X_surf, X_fm, task_types, y_nsteps):
    print("\n" + "="*65)
    print("EVALUATION: ARC v3 vs ARC v2")
    print("="*65)

    S_surf, S_fm, S_V = build_support(
        X_surf, X_fm, task_types, prep, TRAIN_DOMAINS, DEVICE)

    # Load v3
    ckpt_v3 = CKPT / "arc_v3.pt"
    if not ckpt_v3.exists():
        print("ERROR: arc_v3.pt not found. Run --phase train first.")
        return

    ckpt = torch.load(ckpt_v3, map_location=DEVICE)
    v3   = s17.ARCv2(ckpt["surf_dim"], ckpt["fm_dim"]).to(DEVICE)
    v3.load_state_dict(ckpt["model"])
    r_v3 = evaluate(v3, prep, X_surf, X_fm, task_types, y_nsteps,
                    label="ARC v3 (contrastive+ranking)", S_surf=S_surf,
                    S_fm=S_fm, S_V=S_V)

    # Load v2 for comparison
    ckpt_v2 = CKPT / "arc_v2.pt"
    if ckpt_v2.exists():
        ckpt2 = torch.load(ckpt_v2, map_location=DEVICE)
        v2    = s17.ARCv2(ckpt2["surf_dim"], ckpt2["fm_dim"]).to(DEVICE)
        v2.load_state_dict(ckpt2["model"])
        r_v2 = evaluate(v2, prep, X_surf, X_fm, task_types, y_nsteps,
                        label="ARC v2 (baseline)", S_surf=S_surf,
                        S_fm=S_fm, S_V=S_V)

        print(f"\n  DELTA (v3 - v2):")
        for dom in TEST_DOMAINS:
            d = abs(r_v3[dom]["rho"]) - abs(r_v2[dom]["rho"])
            print(f"    {dom}: {d:+.4f}")
        d_mean = r_v3["mean_rho"] - r_v2["mean_rho"]
        print(f"    mean:   {d_mean:+.4f}")

    # Save
    (RESULTS / "arc_v3_eval.json").write_text(json.dumps(r_v3, indent=2))
    print(f"\n  Results → {RESULTS}/arc_v3_eval.json")
    return r_v3


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["train","eval","all"], default="all")
    p.add_argument("--n_episodes", type=int, default=3000)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    print(f"\nARC v3  —  device={DEVICE}")
    s17, prep, X_surf, X_fm, task_types, y_success, y_nsteps = load_deps()

    model = None
    if args.phase in ("train", "all"):
        model = phase_train(args, s17, prep, X_surf, X_fm,
                            task_types, y_success, y_nsteps)

    if args.phase in ("eval", "all"):
        phase_eval(args, s17, prep, X_surf, X_fm, task_types, y_nsteps)


if __name__ == "__main__":
    main()