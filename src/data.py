"""
Data acquisition for Study 1 (classification quality).

Loads one SYNERGY systematic-review dataset and caches it locally as a
clean parquet file with: id, title, abstract, label_included.

Dataset: Sep_2021 — "The rodent object-in-context task: A systematic
review and meta-analysis of important variables" (271 records, 40
included, ~14.8% positive rate).
"""

from pathlib import Path
import urllib.request
import pandas as pd
from synergy_dataset import Dataset
from synergy_dataset.base import _get_path_raw_dataset, SYNERGY_VERSION

DATASET_NAME = "Sep_2021"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / f"{DATASET_NAME}.csv"

# Files needed for one review. The package's default download grabs the
# ENTIRE 26-review archive (~430MB) just to get one review's ~900KB of
# data, and a single unresumable urlopen().read() over that much data is
# also fragile on a flaky connection. Instead we fetch only the handful
# of files this one review actually needs, directly from the repo.
REVIEW_FILES = [
    "labels.csv",
    "metadata.json",
    "metadata_publication.json",
    "works_1.zip",
]
RAW_BASE = (
    f"https://raw.githubusercontent.com/asreview/synergy-dataset/"
    f"v{SYNERGY_VERSION}/{DATASET_NAME}"
)


def _ensure_raw_dataset_downloaded():
    """Download just this review's files if they aren't present yet."""
    review_dir = Path(_get_path_raw_dataset(), DATASET_NAME)
    labels_file = review_dir / "labels.csv"
    if labels_file.exists():
        return

    review_dir.mkdir(parents=True, exist_ok=True)
    for fname in REVIEW_FILES:
        url = f"{RAW_BASE}/{fname}"
        dest = review_dir / fname
        print(f"  fetching {fname}...")
        urllib.request.urlretrieve(url, dest)


def load_review(force_refresh: bool = False) -> pd.DataFrame:
    """Load the chosen review, using a local cache if available."""
    if CACHE_PATH.exists() and not force_refresh:
        return pd.read_csv(CACHE_PATH)

    _ensure_raw_dataset_downloaded()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    d = Dataset(DATASET_NAME)
    df = d.to_frame().reset_index().rename(columns={"openalex_id": "id"})

    keep_cols = ["id", "title", "abstract", "label_included"]
    df = df[keep_cols].dropna(subset=["abstract"]).reset_index(drop=True)

    df.to_csv(CACHE_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_review(force_refresh=True)
    print(f"Loaded {len(df)} records (after dropping missing abstracts)")
    print(f"Positive (included) rate: {df['label_included'].mean():.2%}")
    print("\nSample row:")
    print(df.iloc[0][["title", "label_included"]])
    print(f"\nCached to: {CACHE_PATH}")