"""
Study 1, scorer 2/3: zero-shot LLM scorer.

Same dataset, same evaluation metrics as the TF-IDF + logistic
regression baseline — the only thing that changes is the scoring
method. No retrieved examples, no training data: the model judges each
paper purely from the review's stated scope and the paper's own
title/abstract.

This needs an OPENAI_API_KEY set as an environment variable (or in a
local .env file, not committed to git).
"""

import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

from data import load_review, DATASET_NAME
from llm_common import get_review_context, build_screening_messages, call_model_with_retry, DEFAULT_MODEL

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "zero_shot_predictions.csv"


def run_zero_shot(limit: int = None):
    """Run the zero-shot scorer over the dataset.

    Args:
        limit: if set, only scores the first N papers — useful for a
            cheap smoke test before committing to all 270 API calls.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in a .env file in the "
            "project root: OPENAI_API_KEY=sk-..."
        )

    client = OpenAI()
    df = load_review()
    if limit:
        df = df.head(limit).copy()

    review_context = get_review_context(DATASET_NAME)

    # Resume support: if a partial run already exists, skip papers we've
    # already scored. Useful since 270 sequential API calls is long
    # enough that a dropped connection partway through shouldn't mean
    # starting over.
    done_ids = set()
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["id"])
        print(f"Resuming: {len(done_ids)} papers already scored")

    results = []
    start = time.time()

    for i, row in df.iterrows():
        if row["id"] in done_ids:
            continue

        messages = build_screening_messages(review_context, row["title"], row["abstract"])
        prob, reasoning = call_model_with_retry(client, messages, model=DEFAULT_MODEL)

        results.append({
            "id": row["id"],
            "label_included": row["label_included"],
            "zero_shot_proba": prob,
            "reasoning": reasoning,
        })

        # Save incrementally every 20 papers, not just at the end
        if len(results) % 20 == 0:
            elapsed = time.time() - start
            print(f"  scored {len(results)}/{len(df) - len(done_ids)} new papers ({elapsed:.0f}s elapsed)")
            _append_results(results)
            results = []

    if results:
        _append_results(results)

    print(f"\nDone in {time.time() - start:.0f}s")
    _print_metrics()


def _append_results(new_results: list):
    new_df = pd.DataFrame(new_results)
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    RESULTS_DIR.mkdir(exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)


def _print_metrics():
    preds = pd.read_csv(OUTPUT_PATH)
    y = preds["label_included"].values
    proba = preds["zero_shot_proba"].values

    y_pred = (proba >= 0.5).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    pr_auc = average_precision_score(y, proba)

    print(f"\nZero-shot LLM ({DEFAULT_MODEL}) — n={len(preds)}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  PR-AUC:    {pr_auc:.3f}")

    prec, rec, _ = precision_recall_curve(y, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"Zero-shot LLM (AP={pr_auc:.3f})")
    plt.axhline(y.mean(), color="gray", linestyle="--", label="Random baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Study 1 — Precision-Recall (zero-shot LLM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "zero_shot_pr_curve.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'zero_shot_pr_curve.png'}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_zero_shot(limit=n)
