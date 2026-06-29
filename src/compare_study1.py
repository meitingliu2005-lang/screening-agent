"""
Study 1 final comparison: TF-IDF+LR baseline vs. zero-shot LLM vs. RAG
LLM, all evaluated on the same papers with the same metrics.

This is the actual deliverable of Study 1 — the three individual
scorer scripts each produce one number; this is what answers the
research question (does retrieval improve relevance judgments,
compared to a classic baseline and a zero-shot LLM?).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)
from scipy.stats import binomtest
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SCORERS = {
    "TF-IDF + LR": ("baseline_predictions.csv", "tfidf_lr_proba"),
    "Zero-shot LLM": ("zero_shot_predictions.csv", "zero_shot_proba"),
    "RAG LLM": ("rag_predictions.csv", "rag_proba"),
    "PyTorch MLP": ("pytorch_predictions.csv", "pytorch_proba"),
}


def mcnemar_test(y, pred_a, pred_b, name_a, name_b):
    """Paired significance test: are two classifiers' errors on the same
    papers distinguishable from noise, or could the observed gap just be
    chance given this sample size?

    McNemar's test only looks at papers where the two classifiers
    DISAGREE (one right, one wrong) and asks whether that disagreement
    is skewed toward one model more than a coin flip would predict.
    Papers both got right or both got wrong are uninformative for this
    question, so they're excluded — which is exactly why this is the
    right test here: with only 40 positives, most of the signal about
    "which model is actually better" lives in a small set of disagreement
    cases, and eyeballing aggregate metrics can't tell you whether that
    small set is even large enough to draw a conclusion from.
    """
    correct_a = (pred_a == y)
    correct_b = (pred_b == y)

    b_right_a_wrong = int(((~correct_a) & correct_b).sum())
    a_right_b_wrong = int((correct_a & (~correct_b)).sum())
    n_discordant = b_right_a_wrong + a_right_b_wrong

    if n_discordant == 0:
        return {"comparison": f"{name_a} vs {name_b}", "n_discordant": 0,
                "p_value": 1.0, "verdict": "identical predictions"}

    result = binomtest(min(b_right_a_wrong, a_right_b_wrong), n_discordant, 0.5)
    p = result.pvalue
    verdict = "significant difference (p<0.05)" if p < 0.05 else "not distinguishable from noise"

    return {
        "comparison": f"{name_a} vs {name_b}",
        f"{name_a}_only_correct": a_right_b_wrong,
        f"{name_b}_only_correct": b_right_a_wrong,
        "n_discordant": n_discordant,
        "p_value": round(p, 4),
        "verdict": verdict,
    }


def run_significance_tests():
    loaded = {}
    for name, (filename, col) in SCORERS.items():
        path = RESULTS_DIR / filename
        if path.exists():
            df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            loaded[name] = df

    names = list(loaded.keys())
    if len(names) < 2:
        return

    print("\nPairwise significance (McNemar's test, threshold=0.5):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            df_a, df_b = loaded[name_a], loaded[name_b]
            merged = df_a[["id", "label_included"]].merge(
                df_b[["id"]], on="id"
            )
            y = merged["label_included"].values
            col_a = SCORERS[name_a][1]
            col_b = SCORERS[name_b][1]
            pred_a = (df_a.set_index("id").loc[merged["id"], col_a].values >= 0.5).astype(int)
            pred_b = (df_b.set_index("id").loc[merged["id"], col_b].values >= 0.5).astype(int)

            result = mcnemar_test(y, pred_a, pred_b, name_a, name_b)
            print(f"  {result['comparison']}: n_discordant={result['n_discordant']}, "
                  f"p={result['p_value']} -> {result['verdict']}")


def bootstrap_metric_diff(y, proba_a, proba_b, metric_fn, name_a, name_b, metric_name,
                           n_bootstrap=2000, random_state=42):
    """Bootstrap confidence interval on the DIFFERENCE in a chosen metric
    between two scorers (e.g. recall, or PR-AUC) — not raw correctness.

    This is the test McNemar's can't give you: McNemar's asks "which
    model is more often right at a 0.5 threshold," which under class
    imbalance can point the opposite direction from "which model is
    better at the metric that actually matters." Resampling papers with
    replacement many times and recomputing the metric gap each time
    gives a confidence interval on that gap directly — if the interval
    excludes zero, the difference is real; if it includes zero, the
    observed gap could just be this particular sample.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    diffs = []
    attempts = 0
    max_attempts = n_bootstrap * 3

    while len(diffs) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        idx = rng.randint(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue  # need both classes present to compute these metrics
        try:
            metric_a = metric_fn(y_b, proba_a[idx])
            metric_b = metric_fn(y_b, proba_b[idx])
        except Exception:
            continue
        diffs.append(metric_b - metric_a)

    if not diffs:
        return None

    diffs = np.array(diffs)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    significant = not (lower <= 0 <= upper)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "metric": metric_name,
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_95_low": round(float(lower), 4),
        "ci_95_high": round(float(upper), 4),
        "n_valid_bootstrap": len(diffs),
        "verdict": "significant (95% CI excludes 0)" if significant else "not distinguishable from noise",
    }


def run_bootstrap_tests(n_bootstrap=2000):
    loaded = {}
    for name, (filename, col) in SCORERS.items():
        path = RESULTS_DIR / filename
        if path.exists():
            df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            loaded[name] = df

    names = list(loaded.keys())
    if len(names) < 2:
        return

    metric_fns = {
        "recall": lambda y, p: recall_score(y, (p >= 0.5).astype(int), zero_division=0),
        "pr_auc": lambda y, p: average_precision_score(y, p),
    }

    print(f"\nBootstrap CI on metric differences ({n_bootstrap} resamples, 95% CI):")
    rows = []
    for metric_name, metric_fn in metric_fns.items():
        print(f"\n  Metric: {metric_name}  (positive mean_diff means the second-named scorer is higher)")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a, name_b = names[i], names[j]
                df_a, df_b = loaded[name_a], loaded[name_b]
                merged_ids = df_a[["id", "label_included"]].merge(df_b[["id"]], on="id")
                y = merged_ids["label_included"].values
                col_a, col_b = SCORERS[name_a][1], SCORERS[name_b][1]
                proba_a = df_a.set_index("id").loc[merged_ids["id"], col_a].values
                proba_b = df_b.set_index("id").loc[merged_ids["id"], col_b].values

                result = bootstrap_metric_diff(
                    y, proba_a, proba_b, metric_fn, name_a, name_b, metric_name,
                    n_bootstrap=n_bootstrap,
                )
                if result is None:
                    continue
                rows.append(result)
                print(f"    {result['comparison']}: mean diff={result['mean_diff']:+.3f}, "
                      f"95% CI=[{result['ci_95_low']:+.3f}, {result['ci_95_high']:+.3f}] "
                      f"-> {result['verdict']}")

    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "study1_bootstrap_tests.csv", index=False)
        print(f"\nSaved: {RESULTS_DIR / 'study1_bootstrap_tests.csv'}")


def compare():
    rows = []
    plt.figure(figsize=(6, 5))

    base_labels = None
    for name, (filename, col) in SCORERS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"Skipping {name}: {filename} not found yet")
            continue

        df = pd.read_csv(path)
        y = df["label_included"].values
        proba = df[col].values

        if base_labels is None:
            base_labels = y.mean()

        y_pred = (proba >= 0.5).astype(int)
        rows.append({
            "scorer": name,
            "n": len(df),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "pr_auc": average_precision_score(y, proba),
        })

        prec, rec, _ = precision_recall_curve(y, proba)
        plt.plot(rec, prec, label=f"{name} (AP={rows[-1]['pr_auc']:.3f})")

    if not rows:
        print("No scorer results found yet — run baseline.py, zero_shot.py, and rag.py first.")
        return

    if base_labels is not None:
        plt.axhline(base_labels, color="gray", linestyle="--", label="Random baseline")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "study1_summary.csv", index=False)

    print("\nStudy 1 — scorer comparison")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Study 1 — Precision-Recall, all scorers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "study1_comparison.png", dpi=150)
    print(f"\nSaved: {RESULTS_DIR / 'study1_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'study1_comparison.png'}")

    run_significance_tests()
    run_bootstrap_tests()


if __name__ == "__main__":
    compare()
