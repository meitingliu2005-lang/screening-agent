"""
Study 1, scorer 3/3: RAG scorer.

Identical to the zero-shot scorer in every respect except one: each
screening decision is made with retrieved similar papers (and their
known labels) included as context. Keeping everything else constant
— same prompt structure, same model, same metrics, same CV folds —
means any performance difference vs. zero-shot is attributable to
retrieval specifically, which is the actual question Study 1 asks.

Leakage control: for a paper in fold i, retrieved examples are drawn
ONLY from papers NOT in fold i — the same train/test separation the
TF-IDF baseline uses. Without this, the RAG scorer would effectively
get to peek at its own test set.
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
from embed import get_embeddings, retrieve_similar

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FOLDS_PATH = RESULTS_DIR / "cv_folds.csv"
OUTPUT_PATH = RESULTS_DIR / "rag_predictions.csv"
K_EXAMPLES = 5


def run_rag(limit: int = None):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in a .env file in the "
            "project root: OPENAI_API_KEY=sk-..."
        )
    if not FOLDS_PATH.exists():
        raise RuntimeError(
            "results/cv_folds.csv not found — run baseline.py first, it "
            "generates the fold assignments this scorer reuses."
        )

    client = OpenAI()
    df = load_review()
    folds = pd.read_csv(FOLDS_PATH)
    df = df.merge(folds, on="id")  # adds a "fold" column, aligned by id

    print("Getting embeddings (cached after first run)...")
    embeddings = get_embeddings(df)
    fold_arr = df["fold"].values

    review_context = get_review_context(DATASET_NAME)

    target_df = df.head(limit) if limit else df

    done_ids = set()
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["id"])
        print(f"Resuming: {len(done_ids)} papers already scored")

    results = []
    total_scored = 0
    n_to_score = len(target_df) - len(done_ids.intersection(set(target_df["id"])))
    start = time.time()

    for i, row in target_df.iterrows():
        if row["id"] in done_ids:
            continue

        exclude_mask = (fold_arr == row["fold"])
        similar_idx = retrieve_similar(i, embeddings, exclude_mask, k=K_EXAMPLES)
        examples = [
            {
                "title": df.iloc[j]["title"],
                "abstract": df.iloc[j]["abstract"],
                "label_included": df.iloc[j]["label_included"],
            }
            for j in similar_idx
        ]

        messages = build_screening_messages(
            review_context, row["title"], row["abstract"], examples=examples
        )
        prob, reasoning = call_model_with_retry(client, messages, model=DEFAULT_MODEL)

        results.append({
            "id": row["id"],
            "label_included": row["label_included"],
            "rag_proba": prob,
            "reasoning": reasoning,
        })
        total_scored += 1

        if len(results) >= 20:
            elapsed = time.time() - start
            print(f"  scored {total_scored}/{n_to_score} new papers ({elapsed:.0f}s elapsed)")
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
    proba = preds["rag_proba"].values

    y_pred = (proba >= 0.5).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    pr_auc = average_precision_score(y, proba)

    print(f"\nRAG LLM ({DEFAULT_MODEL}, k={K_EXAMPLES}) — n={len(preds)}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  PR-AUC:    {pr_auc:.3f}")

    prec, rec, _ = precision_recall_curve(y, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"RAG LLM (AP={pr_auc:.3f})")
    plt.axhline(y.mean(), color="gray", linestyle="--", label="Random baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Study 1 — Precision-Recall (RAG LLM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rag_pr_curve.png", dpi=150)
    print(f"Saved: {RESULTS_DIR / 'rag_pr_curve.png'}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_rag(limit=n)
