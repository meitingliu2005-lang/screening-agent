"""
Data acquisition for Study 1 (classification quality).

Loads one SYNERGY systematic-review dataset and caches it locally as a
clean parquet file with: id, title, abstract, label_included.

Dataset: Sep_2021 — "The rodent object-in-context task: A systematic
review and meta-analysis of important variables" (271 records, 40
included, ~14.8% positive rate).
"""

from pathlib import Path
import pandas as pd
from synergy_dataset import Dataset

DATASET_NAME = "Sep_2021"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / f"{DATASET_NAME}.parquet"


def load_review(force_refresh: bool = False) -> pd.DataFrame:
    """Load the chosen review, using a local cache if available."""
    if CACHE_PATH.exists() and not force_refresh:
        return pd.read_parquet(CACHE_PATH)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    d = Dataset(DATASET_NAME)
    df = d.to_frame().reset_index().rename(columns={"openalex_id": "id"})

    keep_cols = ["id", "title", "abstract", "label_included"]
    df = df[keep_cols].dropna(subset=["abstract"]).reset_index(drop=True)

    df.to_parquet(CACHE_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_review(force_refresh=True)
    print(f"Loaded {len(df)} records (after dropping missing abstracts)")
    print(f"Positive (included) rate: {df['label_included'].mean():.2%}")
    print("\nSample row:")
    print(df.iloc[0][["title", "label_included"]])
    print(f"\nCached to: {CACHE_PATH}")
