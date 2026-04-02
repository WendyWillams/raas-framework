"""
RAAS: Risk-Aware Adaptive Screening Framework
==============================================
Reproducible experiment code for:
  "A Risk-Aware Adaptive Screening Framework for Psychiatric Assessment:
   Development and Validation Study"

Dataset
-------
DASS-42 public dataset (Kaggle):
https://www.kaggle.com/datasets/lucasgreenwell/depression-anxiety-stress-scales-responses

Download data.csv and place it in the same directory as this script,
or pass --data_path to specify the file location.

Requirements
------------
    pip install numpy pandas scikit-learn

Usage
-----
    python raas_experiment.py
    python raas_experiment.py --data_path /path/to/data.csv
    python raas_experiment.py --n_bootstrap 2000 --theta 0.75 --tau 1.2
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, recall_score

warnings.filterwarnings("ignore")


# ── Constants ─────────────────────────────────────────────────────────────────

# DASS-42 subscale item numbers (1-based, matching Q{n}A column names)
DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
ANXIETY_ITEMS    = [2, 4,  7,  9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
STRESS_ITEMS     = [1, 6,  8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

# Rule engine signal items: 3 per dimension, based on conceptual alignment
# with core symptoms of each DASS subscale
SIGNAL_ITEMS = {
    "dep": [3, 5, 10],  # no positive affect; loss of motivation; no future hope
    "anx": [2, 4,  7],  # dry mouth; breathing difficulties; trembling
    "str": [1, 6,  8],  # easily irritated; overreaction; difficulty relaxing
}

# TIPI personality subscales correlated with emotional instability
TIPI_FEATURES = ["TIPI4", "TIPI9"]

# DASS-42 official moderate-severity lower bounds (0-3 per item scale × 14 items)
# Converted to 1-4 scale by adding 14 (14 items x 1 per item).
# The 60th-percentile thresholds (dep=39, anx=32, str=38) exceed these bounds
# by 11, 8, and 5 points respectively, indicating that positive labels
# correspond to moderate-to-severe symptomatology.
OFFICIAL_MODERATE_1TO4_SCALE = {"dep": 28, "anx": 24, "str": 33}


# ── Data loading and preprocessing ───────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    q_cols = [f"Q{i}A" for i in range(1, 43) if f"Q{i}A" in df.columns]
    for c in q_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=q_cols)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df = df[(df["age"] >= 13) & (df["age"] <= 80)]
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df):,} valid samples.")
    return df


def build_labels(df):
    """
    Multi-label targets: label=1 if subscale sum > 60th percentile.
    Thresholds exceed DASS official moderate lower bounds (converted to 1-4
    scale), so positive labels correspond to moderate-to-severe presentations.
    """
    q_cols = [f"Q{i}A" for i in range(1, 43) if f"Q{i}A" in df.columns]
    dep_s = df[[f"Q{i}A" for i in DEPRESSION_ITEMS if f"Q{i}A" in q_cols]].sum(axis=1)
    anx_s = df[[f"Q{i}A" for i in ANXIETY_ITEMS    if f"Q{i}A" in q_cols]].sum(axis=1)
    str_s = df[[f"Q{i}A" for i in STRESS_ITEMS     if f"Q{i}A" in q_cols]].sum(axis=1)

    p60 = {
        "dep": dep_s.quantile(0.60),
        "anx": anx_s.quantile(0.60),
        "str": str_s.quantile(0.60),
    }

    print("\nLabel thresholds (60th percentile, 1-4 scale):")
    for d, v in p60.items():
        off = OFFICIAL_MODERATE_1TO4_SCALE[d]
        print(f"  {d}: {v:.0f}  (official moderate lower bound: {off}, diff: +{v-off:.0f})")

    y = np.column_stack([
        (dep_s > p60["dep"]).astype(int),
        (anx_s > p60["anx"]).astype(int),
        (str_s > p60["str"]).astype(int),
    ])

    print(f"\nPositive rate: dep={y[:,0].mean():.1%}  anx={y[:,1].mean():.1%}  str={y[:,2].mean():.1%}")
    combos, counts = np.unique(y, axis=0, return_counts=True)
    n = len(y)
    print("Label distribution:")
    for c, cnt in sorted(zip(map(tuple, combos), counts), key=lambda x: -x[1]):
        print(f"  {list(c)}: {cnt/n:.1%}  (n={cnt:,})")
    return y


def build_features(df):
    """Feature matrix: 42 item responses + demographics + TIPI subscales."""
    q_cols = [f"Q{i}A" for i in range(1, 43) if f"Q{i}A" in df.columns]
    feat_cols = list(q_cols)
    for col in ["age", "gender"] + TIPI_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
            feat_cols.append(col)
    return df[feat_cols].values.astype(float), feat_cols


# ── Evaluation metrics ────────────────────────────────────────────────────────

def recall_at_k(y_true, y_pred):
    """
    Primary safety metric.
    For each patient: proportion of genuinely needed dimensions correctly
    recommended. Average across all patients.
    Recall@K = 1.0 means zero omissions.
    """
    scores = []
    for yt, yp in zip(y_true, y_pred):
        needed = yt.sum()
        scores.append(1.0 if needed == 0 else np.logical_and(yt, yp).sum() / needed)
    return float(np.mean(scores))


def bootstrap_ci(y_true, y_pred, n_bootstrap=2000, seed=42, alpha=0.05):
    """95% bootstrap confidence interval for Recall@K."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot.append(recall_at_k(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


def evaluate(y_true, y_pred, name, compute_ci=False, n_bootstrap=2000):
    acc  = accuracy_score(y_true, y_pred)
    mac  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rak  = recall_at_k(y_true, y_pred)
    dims = y_pred.sum(axis=1).mean()
    red  = 1 - dims / 3.0
    r = dict(method=name, avg_dims=dims, reduction=red,
             accuracy=acc, macro_recall=mac, recall_at_k=rak)
    if compute_ci:
        lo, hi = bootstrap_ci(y_true, y_pred, n_bootstrap=n_bootstrap)
        r.update(ci_lo=lo, ci_hi=hi)
    return r


# ── Rule engine ───────────────────────────────────────────────────────────────

def get_signal_indices(q_cols):
    """Map signal item numbers to column indices in the feature matrix."""
    col_idx = {col: i for i, col in enumerate(q_cols)}
    return {
        dim: [col_idx[f"Q{q}A"] for q in items if f"Q{q}A" in col_idx]
        for dim, items in SIGNAL_ITEMS.items()
    }


def rule_predict(X_q, sig_idx, tau=1.2):
    """
    Rule engine: prune dimension if signal-item mean < tau.
    Safety fallback: always retain at least one dimension (stress).
    X_q : first 42 columns of feature matrix (Q1A..Q42A).
    """
    n = len(X_q)
    kept = np.ones((n, 3), dtype=int)
    for di, dim in enumerate(["dep", "anx", "str"]):
        kept[:, di] = (X_q[:, sig_idx[dim]].mean(axis=1) >= tau).astype(int)
    kept[kept.sum(axis=1) == 0, 2] = 1  # safety fallback
    return kept


# ── Adaptive switching ────────────────────────────────────────────────────────

def adaptive_predict(y_ml, rule, proba, theta=0.75):
    """
    Adaptive confidence switching:
      conf >= theta → use GBM prediction directly (high-confidence path)
      conf <  theta → apply min(GBM, rule)         (low-confidence path)

    In practice, rule recommendations are >= GBM for 99.8% of low-confidence
    samples (tau=1.2 is conservative), so min() = GBM for nearly all cases.
    The rule engine's value is providing an interpretable reasoning trace
    for low-confidence patients, not changing the recommendation outcome.
    """
    conf = proba.max(axis=1)
    y_out = y_ml.copy()
    low = conf < theta
    y_out[low] = np.minimum(y_ml[low], rule[low])
    return y_out


# ── Experiments ───────────────────────────────────────────────────────────────

def exp1_method_comparison(X_tr, X_te, y_tr, y_te, feat_cols,
                            tau, theta, n_bootstrap, seed):
    print("\n" + "="*65)
    print("Experiment 1: Method Comparison (Table 1)")
    print("="*65)

    sig_idx = get_signal_indices(feat_cols[:42])
    X_te_q  = X_te[:, :42]

    # Train models
    print("Training LR  ...", end=" ", flush=True)
    lr = MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=seed))
    lr.fit(X_tr, y_tr)
    print("RF  ...", end=" ", flush=True)
    rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1))
    rf.fit(X_tr, y_tr)
    print("GBM ...", end=" ", flush=True)
    gbm = MultiOutputClassifier(GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed))
    gbm.fit(X_tr, y_tr)
    print("done.")

    proba  = np.column_stack([e.predict_proba(X_te)[:,1] for e in gbm.estimators_])
    y_gbm  = gbm.predict(X_te)
    rule   = rule_predict(X_te_q, sig_idx, tau=tau)
    y_adapt = adaptive_predict(y_gbm, rule, proba, theta=theta)

    results = [
        evaluate(y_te, np.ones_like(y_te),   "Full-scale baseline"),
        evaluate(y_te, lr.predict(X_te),      "Logistic Regression"),
        evaluate(y_te, rf.predict(X_te),      "Random Forest"),
        evaluate(y_te, y_gbm,                 "GBM only"),
        evaluate(y_te, np.minimum(y_gbm, rule), "Rule+GBM (no adaptation)"),
        evaluate(y_te, y_adapt, "Adaptive Rule+GBM (proposed)",
                 compute_ci=True, n_bootstrap=n_bootstrap),
    ]

    hdr = f"{'Method':<34} {'Dims':>6} {'Red':>7} {'Acc':>7} {'MacRec':>7} {'R@K':>7}"
    print("\n" + hdr)
    print("-"*65)
    for r in results:
        ci = f"  95%CI[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]" if "ci_lo" in r else ""
        print(f"{r['method']:<34} {r['avg_dims']:>6.2f} {r['reduction']:>7.1%} "
              f"{r['accuracy']:>7.4f} {r['macro_recall']:>7.4f} {r['recall_at_k']:>7.4f}{ci}")

    conf = proba.max(axis=1)
    print(f"\nAt theta={theta}: high-conf rate={( conf>=theta).mean():.1%}  "
          f"low-conf rate={(conf<theta).mean():.1%}")

    return gbm, proba, y_gbm, rule, y_adapt


def exp2_threshold_sensitivity(y_te, y_gbm, rule, proba,
                                thetas=(0.50,0.60,0.70,0.75,0.80,0.90,1.00)):
    print("\n" + "="*65)
    print("Experiment 2: Adaptive Threshold Sensitivity (Table 2)")
    print("="*65)
    conf = proba.max(axis=1)
    print(f"\n{'theta':>10} {'R@K':>8} {'Reduction':>10} {'HighConf%':>11}")
    print("-"*45)
    for t in thetas:
        y_a = adaptive_predict(y_gbm, rule, proba, theta=t)
        rak = recall_at_k(y_te, y_a)
        dims = y_a.sum(axis=1).mean()
        hc = (conf >= t).mean()
        mark = " *" if t == 0.75 else ""
        print(f"{t:>10.2f}{mark:2} {rak:>8.4f} {1-dims/3:>10.1%} {hc:>11.1%}")


def exp3_signal_ablation(X_te, y_te, proba, feat_cols,
                          theta=0.75, n_runs=10, seed=42):
    print("\n" + "="*65)
    print("Experiment 3: Signal Item Ablation")
    print("="*65)
    rng = np.random.default_rng(seed)
    sig_idx = get_signal_indices(feat_cols[:42])
    X_te_q  = X_te[:, :42]
    y_gbm   = (proba >= 0.5).astype(int)
    n_q     = 42

    def run(rule_te):
        y_a = adaptive_predict(y_gbm, rule_te, proba, theta=theta)
        return recall_at_k(y_te, y_a), y_a.sum(axis=1).mean()

    # Expert
    rak_e, d_e = run(rule_predict(X_te_q, sig_idx))

    # Random (n_runs)
    raks_r, dims_r = [], []
    for _ in range(n_runs):
        rand = {dim: list(rng.choice(n_q, 3, replace=False)) for dim in ["dep","anx","str"]}
        rak_r, d_r = run(rule_predict(X_te_q, rand))
        raks_r.append(rak_r); dims_r.append(d_r)

    # No rule
    rak_n, d_n = recall_at_k(y_te, y_gbm), y_gbm.sum(axis=1).mean()

    print(f"\n{'Configuration':<35} {'R@K':>8} {'AvgDims':>9} {'Reduction':>10}")
    print("-"*65)
    print(f"{'Expert signal items':<35} {rak_e:>8.4f} {d_e:>9.2f} {1-d_e/3:>10.1%}")
    print(f"{'Random items (mean, n='+str(n_runs)+')':<35} "
          f"{np.mean(raks_r):>8.4f} {np.mean(dims_r):>9.2f} {1-np.mean(dims_r)/3:>10.1%}")
    print(f"  (std R@K = {np.std(raks_r):.4f})")
    print(f"{'No rule layer (pure GBM)':<35} {rak_n:>8.4f} {d_n:>9.2f} {1-d_n/3:>10.1%}")


def exp4_case_study(X_te, y_te, proba, feat_cols,
                     theta=0.75, tau=1.2, idx=13):
    print("\n" + "="*65)
    print(f"Experiment 4: Real Case Study (sample index={idx})")
    print("="*65)
    sig_idx = get_signal_indices(feat_cols[:42])
    X_te_q  = X_te[:, :42]
    rule    = rule_predict(X_te_q, sig_idx, tau=tau)
    y_gbm   = (proba >= 0.5).astype(int)
    y_adapt = adaptive_predict(y_gbm, rule, proba, theta=theta)
    conf    = proba.max(axis=1)

    q = X_te_q[idx]
    path = "LOW-confidence → rule engine triggered" if conf[idx] < theta else "HIGH-confidence → GBM direct"
    print(f"\nConfidence: {conf[idx]:.3f}  ({path})")

    dim_names = {"dep": "Depression", "anx": "Anxiety", "str": "Stress"}
    q_labels  = {"dep": ["Q3","Q5","Q10"], "anx": ["Q2","Q4","Q7"], "str": ["Q1","Q6","Q8"]}
    for di, dim in enumerate(["dep","anx","str"]):
        scores = q[sig_idx[dim]]
        mean   = scores.mean()
        act    = "RETAINED" if mean >= tau else "PRUNED"
        print(f"  {dim_names[dim]} {q_labels[dim]}: scores={[int(s) for s in scores]}, "
              f"mean={mean:.2f} → {act}")

    print(f"\n  GBM probabilities: dep={proba[idx,0]:.3f} anx={proba[idx,1]:.3f} str={proba[idx,2]:.3f}")
    print(f"  GBM prediction:    {y_gbm[idx].tolist()}")
    print(f"  Rule recommendation:{rule[idx].tolist()}")
    print(f"  Adaptive output:   {y_adapt[idx].tolist()}  [= min(GBM, rule)]")
    print(f"  True label:        {y_te[idx].tolist()}")
    print(f"  Correct: {np.array_equal(y_adapt[idx], y_te[idx])}")

    # Rule modification rate among low-confidence samples
    low_mask = conf < theta
    changed  = sum(
        not np.array_equal(y_gbm[i], rule[i])
        for i in np.where(low_mask)[0]
    )
    n_low = low_mask.sum()
    print(f"\n  Low-confidence samples (conf<{theta}): n={n_low}")
    print(f"  Rule differs from GBM in: {changed} ({changed/n_low:.1%})")
    print(f"  But min(GBM,rule)=GBM because rule is always >= GBM:")
    rule_more = sum((rule[i] >= y_gbm[i]).all() for i in np.where(low_mask)[0])
    print(f"    rule >= GBM in {rule_more}/{n_low} ({rule_more/n_low:.1%}) low-conf samples")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   default="data.csv")
    parser.add_argument("--tau",         type=float, default=1.2)
    parser.add_argument("--theta",       type=float, default=0.75)
    parser.add_argument("--n_bootstrap", type=int,   default=2000)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print("RAAS Experiment Runner")
    print(f"tau={args.tau}  theta={args.theta}  n_bootstrap={args.n_bootstrap}  seed={args.seed}\n")

    df = load_data(args.data_path)
    y  = build_labels(df)
    X, feat_cols = build_features(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    print(f"\nTrain: {len(X_tr):,}   Test: {len(X_te):,}")

    gbm, proba, y_gbm, rule, y_adapt = exp1_method_comparison(
        X_tr, X_te, y_tr, y_te, feat_cols,
        args.tau, args.theta, args.n_bootstrap, args.seed
    )
    exp2_threshold_sensitivity(y_te, y_gbm, rule, proba)
    exp3_signal_ablation(X_te, y_te, proba, feat_cols, theta=args.theta, seed=args.seed)
    exp4_case_study(X_te, y_te, proba, feat_cols, theta=args.theta, tau=args.tau)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
