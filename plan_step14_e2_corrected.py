"""
plan_step14_e2_corrected.py
============================
Corrected E2 quintile stratification experiment.

PROBLEM WITH ORIGINAL (plan_step5_bulletproof.py run_e2_stratification):
  get_guru_scores_per_instance() trains XGBoost on 70% of the TEST
  DOMAIN's BFS labels (y_success). The resulting quintile scores are
  supervised by test-domain ground truth. Of course BFS-easy instances
  have high GPT-4o validity — this is a property of the dataset, not ARC.
  Any model, even n_objects alone, would produce the same Spearman ρ.

CORRECT PROTOCOL:
  1. Extract ARC attention features using model.get_features() with
     a TRAINING-DOMAIN support set only.
  2. Train a linear probe on TRAINING DOMAIN features only.
  3. Apply to test domain — zero-shot, no test labels used.
  4. Assign quintiles by those zero-shot scores.
  5. Report GPT-4o validity per quintile.

Also reports n_objects quintiles as a baseline for comparison:
  If n_objects gives the same Spearman ρ as ARC, then ARC adds nothing
  to the quintile analysis beyond what object count provides.

USAGE:
  python plan_step14_e2_corrected.py
"""

import json
import importlib.util
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
import torch
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data" / "planning"
RESULTS_DIR = ROOT_DIR / "results_planning"; RESULTS_DIR.mkdir(exist_ok=True)

DOMAIN_LABELS = {
    "blocksworld":         "Blocksworld",
    "logistics":           "Logistics",
    "mystery_blocksworld": "Mystery-BW",
}
TEST_DOMAINS = ["blocksworld", "logistics", "mystery_blocksworld"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Load everything ───────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "step3", ROOT_DIR / "plan_step3_guru.py")
step3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step3)

print("Loading data and model...")
X_surf, X_fm, y_success, y_nsteps, task_types, registry = step3.load_all_data()

splits     = registry["splits"]
meta_train = splits.get("meta_train", splits.get("train", {})) if isinstance(splits, dict) else splits

if isinstance(meta_train, dict) and "domains" in meta_train:
    train_doms = meta_train["domains"]           # original expected schema
    train_mask = np.isin(task_types, train_doms)

elif isinstance(meta_train, (list, np.ndarray)) and isinstance(
        (meta_train[0] if isinstance(meta_train, list) else meta_train.flat[0]), str):
    train_doms = list(meta_train)                # flat list of domain strings
    train_mask = np.isin(task_types, train_doms)

elif isinstance(meta_train, (list, np.ndarray)) and len(meta_train) == len(task_types):
    train_mask = np.asarray(meta_train, dtype=bool)   # boolean/index mask
    train_doms = list(np.unique(task_types[train_mask]))

else:
    # fallback: treat all non-test domains as training
    train_mask = ~np.isin(task_types, TEST_DOMAINS)
    train_doms = list(np.unique(task_types[train_mask]))

X_surf_s  = X_surf[train_mask]
X_fm_s    = X_fm[train_mask]
y_s       = y_success[train_mask]
ns_s      = y_nsteps[train_mask]

ckpt  = torch.load(ROOT_DIR / "checkpoints_planning" / "guru_success.pt",
                   map_location=DEVICE)
state = ckpt["model"]
model = step3.PlanningGURU(
    state["key_enc.net.0.weight"].shape[1],
    state["query_enc.net.0.weight"].shape[1]).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"  Model loaded. Training domains: {train_doms}")

# ── Build training-domain support set ─────────────────────────────────────────
rng     = np.random.default_rng(42)
n_sup   = min(60, len(X_surf_s))
sup_idx = rng.choice(len(X_surf_s), n_sup, replace=False)
Xs_sup  = X_surf_s[sup_idx]
Xe_sup  = X_fm_s[sup_idx]

sc_p_sup = StandardScaler().fit(Xs_sup)
sc_e_sup = StandardScaler().fit(Xe_sup)
Xs_sup_n = sc_p_sup.transform(Xs_sup)
Xe_sup_n = sc_e_sup.transform(Xe_sup)
n_comp   = max(2, min(20, n_sup // 10, Xs_sup_n.shape[1]))
res_pipe = Pipeline([("pca", PCA(n_components=n_comp)),
                     ("ridge", Ridge(alpha=1.0))])
res_pipe.fit(Xs_sup_n, Xe_sup_n)
Xr_sup_n = Xe_sup_n - res_pipe.predict(Xs_sup_n)

S_surf = torch.FloatTensor(Xs_sup_n).to(DEVICE)
S_fm   = torch.FloatTensor(Xe_sup_n).to(DEVICE)
S_V    = torch.FloatTensor(np.hstack([Xs_sup_n, Xr_sup_n])).to(DEVICE)


def get_arc_features_zero_shot(Xs_q, Xf_q):
    """
    Extract ARC attention features for query instances.
    Uses ONLY training-domain support — no test labels.
    Returns Nx128 feature matrix.
    """
    Xs_n = sc_p_sup.transform(Xs_q)
    Xe_n = sc_e_sup.transform(Xf_q)
    Xr_n = Xe_n - res_pipe.predict(Xs_n)

    feats = []
    with torch.no_grad():
        for i in range(len(Xs_n)):
            feat, _ = model.get_features(
                torch.FloatTensor(Xs_n[i]).to(DEVICE),
                torch.FloatTensor(Xe_n[i]).to(DEVICE),
                torch.FloatTensor(Xr_n[i]).to(DEVICE),
                S_surf, S_fm, S_V)
            feats.append(feat.cpu().numpy())
    return np.stack(feats)


# ── Train zero-shot probe on TRAINING domains only ────────────────────────────
print("Extracting ARC features for training domains...", end=" ", flush=True)
F_train = get_arc_features_zero_shot(X_surf_s, X_fm_s)
sc_f    = StandardScaler().fit(F_train)
F_tr_n  = sc_f.transform(F_train)

# Logistic regression probe trained on training-domain BFS labels
# This is the "routing classifier" that transfers to test domains
probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
probe.fit(F_tr_n, y_s.astype(int))
print("done")

# ── Load GPT-4o labels ────────────────────────────────────────────────────────
gpt4o_path = ROOT_DIR / "gpt4o_eval_instances.jsonl"
by_dom = {}
for line in open(gpt4o_path):
    r = json.loads(line)
    by_dom.setdefault(r["domain"], []).append(r)
for dom in by_dom:
    by_dom[dom].sort(key=lambda r: int(r["instance_id"]))
print(f"Loaded GPT-4o labels: "
      f"{sum(len(v) for v in by_dom.values())} instances\n")


# ── Run corrected E2 ──────────────────────────────────────────────────────────
print("CORRECTED E2: Zero-shot ARC quintiles vs GPT-4o validity")
print("="*70)
print("(No test-domain labels used at any point in scoring)")
print()

header = (f"  {'Domain':<14}  {'Q1':>7}  {'Q2':>7}  {'Q3':>7}  "
          f"{'Q4':>7}  {'Q5':>7}  {'ρ(ARC)':>8}  {'ρ(nobj)':>8}")
print(header)
print("  " + "-"*70)

all_results = {}

for dom in TEST_DOMAINS:
    mask   = task_types == dom
    Xs_q   = X_surf[mask]
    Xf_q   = X_fm[mask]
    n_obj  = Xs_q[:, 0]
    N      = mask.sum()

    # GPT-4o labels
    y_gpt = np.array([1.0 if r["valid_plan"] else 0.0
                      for r in by_dom.get(dom, [])[:N]])

    # ── ARC zero-shot scores (NO test labels) ─────────────────────────────
    F_te    = get_arc_features_zero_shot(Xs_q, Xf_q)
    F_te_n  = sc_f.transform(F_te)
    arc_scores = probe.predict_proba(F_te_n)[:, 1]

    # ── n_objects scores (baseline) ────────────────────────────────────────
    # Fewer objects = easier = higher P(easy)
    nobj_scores = -n_obj.astype(float)

    # ── Assign equal-size quintiles ────────────────────────────────────────
    def quintile_gpt_rates(scores, y_gpt, n_q=5):
        """Sort by score descending (easiest first), split into n_q equal groups."""
        order  = np.argsort(scores)[::-1]
        groups = np.array_split(order, n_q)
        rates  = [float(y_gpt[g].mean()) if len(g) > 0 else float("nan")
                  for g in groups]
        counts = [len(g) for g in groups]
        return rates, counts

    arc_rates,  arc_counts  = quintile_gpt_rates(arc_scores,  y_gpt)
    nobj_rates, nobj_counts = quintile_gpt_rates(nobj_scores, y_gpt)

    # ── Spearman ρ on quintile means ───────────────────────────────────────
    rho_arc,  p_arc  = spearmanr([1,2,3,4,5], arc_rates)
    rho_nobj, p_nobj = spearmanr([1,2,3,4,5], nobj_rates)

    lbl  = DOMAIN_LABELS[dom]
    q_str = "  ".join(f"{r:>6.1%}" for r in arc_rates)
    print(f"  {lbl:<14}  {q_str}  {rho_arc:>+8.3f}  {rho_nobj:>+8.3f}")

    all_results[dom] = {
        "arc": {
            "quintile_rates":  arc_rates,
            "quintile_counts": arc_counts,
            "spearman_rho":    float(rho_arc),
            "spearman_p":      float(p_arc),
        },
        "nobj": {
            "quintile_rates":  nobj_rates,
            "quintile_counts": nobj_counts,
            "spearman_rho":    float(rho_nobj),
            "spearman_p":      float(p_nobj),
        },
        "overall_gpt4o": float(y_gpt.mean()),
        "zero_shot":     True,
        "note": "ARC scored by logistic regression trained on training "
                "domains only. No test-domain labels used at any point.",
    }

print()
print("n_objects baseline (same quintile partitioning):")
print(header)
print("  " + "-"*70)
for dom in TEST_DOMAINS:
    r     = all_results[dom]["nobj"]
    lbl   = DOMAIN_LABELS[dom]
    q_str = "  ".join(f"{v:>6.1%}" for v in r["quintile_rates"])
    rho_arc = all_results[dom]["arc"]["spearman_rho"]
    print(f"  {lbl:<14}  {q_str}  {'':>8}  {r['spearman_rho']:>+8.3f}")

print()
print("INTERPRETATION:")
print("  ρ(ARC):  Spearman correlation using zero-shot ARC scores")
print("  ρ(nobj): Spearman correlation using n_objects alone")
print("  If ρ(ARC) ≈ ρ(nobj): ARC adds nothing beyond object count")
print("  If ρ(ARC) > ρ(nobj): ARC provides additional signal")
print()
print("KEY: Q1=100% GPT-4o validity is only meaningful if ARC quintiles")
print("     were assigned without using test-domain BFS labels.")
print("     This corrected experiment uses no test labels.")

# ── LaTeX table ───────────────────────────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(
    r"\caption{E2 (corrected): Real GPT-4o valid-plan rate by ARC difficulty "
    r"quintile. Q1 is easiest (highest ARC P(easy)); Q5 is hardest. "
    r"\textbf{Zero-shot}: ARC scores derived from a logistic regression "
    r"trained on training domains only; no test-domain labels used at any point. "
    r"$\rho(\text{ARC})$ and $\rho(\text{n-obj})$ are Spearman correlations "
    r"between quintile rank and GPT-4o validity rate.}")
lines.append(r"\label{tab:e2_corrected}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{l ccccc cc}")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Domain} & Q1 & Q2 & Q3 & Q4 & Q5 "
    r"& $\rho$\textbf{(ARC)} & $\rho$\textbf{(n-obj)} \\")
lines.append(r"\midrule")

for dom in TEST_DOMAINS:
    r    = all_results[dom]
    lbl  = DOMAIN_LABELS[dom]
    qs   = " & ".join(f"{v:.1%}" for v in r["arc"]["quintile_rates"])
    lines.append(
        f"  {lbl} & {qs} & "
        f"${r['arc']['spearman_rho']:+.3f}$ & "
        f"${r['nobj']['spearman_rho']:+.3f}$ \\\\")

lines.append(r"\bottomrule")
lines.append(r"\multicolumn{8}{p{0.98\linewidth}}{\footnotesize")
lines.append(
    r"Quintile sizes are equal ($N/5$ instances each). "
    r"ARC scores from logistic regression trained on Depot, Satellite, "
    r"Rovers, Ferry (training domains); no Blocksworld, Logistics, or "
    r"Mystery-BW labels used. "
    r"$\rho(\text{n-obj})$ uses object count alone as the difficulty "
    r"signal for comparison. "
    r"If $\rho(\text{ARC}) \approx \rho(\text{n-obj})$, object count "
    r"is sufficient; if $\rho(\text{ARC}) > \rho(\text{n-obj})$, "
    r"ARC provides additional difficulty signal beyond problem size.}")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
out_tex = RESULTS_DIR / "e2_corrected.tex"
out_tex.write_text(tex)
out_json = RESULTS_DIR / "e2_corrected.json"
out_json.write_text(json.dumps(all_results, indent=2))

print(f"\n  LaTeX → {out_tex}")
print(f"  JSON  → {out_json}")
print()
print("Compare these ρ values against the original (supervised) table.")
print("The honest question: does ARC zero-shot predict GPT-4o difficulty")
print("better than n_objects alone?")
