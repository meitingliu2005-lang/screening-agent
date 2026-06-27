"""
Study 1, scorer 1/3: TF-IDF + logistic regression baseline.

This is the classic published baseline for systematic-review screening
(the same family of model ASReview's own benchmarks compare against).
It needs no API calls and no embeddings, so it's the fastest scorer to
run and sets the number the other two scorers have to beat.

Evaluation approach: 5-fold stratified cross-validation, not a single
train/test split. With ~270 records and ~40 positives, one held-out
split would have too few positive examples for trustworthy metrics.
Cross-validation also means every paper gets exactly one out-of-fold
prediction, so we get a metric over the *whole* dataset rather than a
small slice of it.

IMPORTANT for fairness across scorers: the fold assignments are saved
to results/cv_folds.csv with a fixed random_state. The zero-shot and
RAG scorers (built next) will reuse these exact same fold assignments,
so all three methods are evaluated on identical splits.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)
import matplotlib.pyplot as plt

from data import load_review

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
N_FOLDS = 5
RANDOM_STATE = 42


def run_baseline():
    df = load_review()
    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).values
    y = df["label_included"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_proba = np.zeros(len(df))
    fold_id = np.zeros(len(df), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X_train = vectorizer.fit_transform(texts[train_idx])
        X_test = vectorizer.transform(texts[test_idx])

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_train, y[train_idx])

        oof_proba[test_idx] = clf.predict_proba(X_test)[:, 1]
        fold_id[test_idx] = fold

    # Save fold assignments so other scorers can reuse the exact same splits
    splits_df = pd.DataFrame({"id": df["id"], "fold": fold_id})
    RESULTS_DIR.mkdir(exist_ok=True)
    splits_df.to_csv(RESULTS_DIR / "cv_folds.csv", index=False)

    # Save predictions
    preds_df = pd.DataFrame({
        "id": df["id"],
        "label_included": y,
        "tfidf_lr_proba": oof_proba,
    })
    preds_df.to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)

    # Metrics at default 0.5 threshold
    y_pred = (oof_proba >= 0.5).astype(int)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    pr_auc = average_precision_score(y, oof_proba)

    print("TF-IDF + Logistic Regression — 5-fold CV, out-of-fold metrics")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  PR-AUC:    {pr_auc:.3f}")
    print(f"  (n={len(df)}, positives={int(y.sum())})")

    # Precision-recall curve plot
    prec, rec, _ = precision_recall_curve(y, oof_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"TF-IDF + LR (AP={pr_auc:.3f})")
    plt.axhline(y.mean(), color="gray", linestyle="--", label="Random baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Study 1 — Precision-Recall (TF-IDF + LR baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_pr_curve.png", dpi=150)
    print(f"\nSaved: {RESULTS_DIR / 'baseline_predictions.csv'}")
    print(f"Saved: {RESULTS_DIR / 'cv_folds.csv'}")
    print(f"Saved: {RESULTS_DIR / 'baseline_pr_curve.png'}")

    return preds_df


if __name__ == "__main__":
    run_baseline()