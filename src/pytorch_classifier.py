"""
Study 1, scorer 4/4: PyTorch MLP classifier on embeddings.

Unlike the other scorers, this one needs no API calls at all — it
trains directly on the embeddings already cached by embed.py during
the RAG scorer's run. A small neural net on top of the same embeddings
used for RAG's retrieval step, evaluated through the same 5-fold CV as
the baseline.

Overfitting risk worth being explicit about: these embeddings are
1536-dimensional, and each training fold has only ~216 examples — that
ratio is exactly the setup that overfits a careless neural net. Three
specific guards against that, not just dropout for its own sake:
  1. A single small hidden layer (32 units) — minimal capacity to fit
  2. Weight decay (L2) on the optimizer
  3. Early stopping on a held-out validation slice carved out of each
     fold's training data, restoring the best-validation-loss weights
     rather than just running a fixed number of epochs
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)
import matplotlib.pyplot as plt

from data import load_review
from embed import get_embeddings

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FOLDS_PATH = RESULTS_DIR / "cv_folds.csv"
OUTPUT_PATH = RESULTS_DIR / "pytorch_predictions.csv"

HIDDEN_DIM = 32
MAX_EPOCHS = 200
PATIENCE = 20
RANDOM_STATE = 42


class ScreeningMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits, not probabilities


def train_one_fold(X_train, y_train, X_val, y_val, input_dim):
    torch.manual_seed(RANDOM_STATE)
    model = ScreeningMLP(input_dim)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def run_pytorch_classifier():
    df = load_review()
    folds = pd.read_csv(FOLDS_PATH)
    df = df.merge(folds, on="id")

    print("Getting embeddings (uses the cache from the RAG step — no API calls here)...")
    embeddings = get_embeddings(df)
    y = df["label_included"].values
    fold_arr = df["fold"].values

    oof_proba = np.zeros(len(df))

    for fold in sorted(df["fold"].unique()):
        train_mask = fold_arr != fold
        test_mask = fold_arr == fold

        X_train_full = embeddings[train_mask]
        y_train_full = y[train_mask]
        X_test = embeddings[test_mask]

        # Carve a validation slice out of this fold's training data, for
        # early stopping -- the test fold itself is never touched until
        # final prediction.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.15,
            stratify=y_train_full, random_state=RANDOM_STATE,
        )

        model = train_one_fold(X_train, y_train, X_val, y_val, embeddings.shape[1])

        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_test, dtype=torch.float32))
            test_proba = torch.sigmoid(test_logits).numpy()

        oof_proba[test_mask] = test_proba
        print(f"  fold {fold}: trained on {len(X_train)} (+{len(X_val)} val), "
              f"predicted {test_mask.sum()}")

    preds_df = pd.DataFrame({
        "id": df["id"],
        "label_included": y,
        "pytorch_proba": oof_proba,
    })
    RESULTS_DIR.mkdir(exist_ok=True)
    preds_df.to_csv(OUTPUT_PATH, index=False)

    y_pred = (oof_proba >= 0.5).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    pr_auc = average_precision_score(y, oof_proba)

    print(f"\nPyTorch MLP (embeddings, {HIDDEN_DIM}-unit hidden layer) — 5-fold CV, n={len(df)}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  PR-AUC:    {pr_auc:.3f}")

    prec, rec, _ = precision_recall_curve(y, oof_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"PyTorch MLP (AP={pr_auc:.3f})")
    plt.axhline(y.mean(), color="gray", linestyle="--", label="Random baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Study 1 — Precision-Recall (PyTorch MLP on embeddings)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pytorch_pr_curve.png", dpi=150)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {RESULTS_DIR / 'pytorch_pr_curve.png'}")


if __name__ == "__main__":
    run_pytorch_classifier()
