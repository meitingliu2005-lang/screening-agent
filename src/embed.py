"""
Embeddings for the RAG scorer.

Uses OpenAI's text-embedding-3-small ($0.02 per 1M tokens — for 270
short abstracts this costs a fraction of a cent) rather than a local
model. Deliberately skipping sentence-transformers/chromadb here: for
270 vectors, an in-memory cosine-similarity search is just as correct
as a full vector DB, without an extra multi-GB local install.

Embeddings are cached to disk since they never change for a fixed
dataset — no reason to pay for or wait on the same embedding twice.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "results" / "embeddings.npy"
CACHE_IDS_PATH = ROOT / "results" / "embeddings_ids.csv"


def _embed_batch_with_retry(client, batch: list, max_retries: int = 4) -> list:
    """Call the embeddings API with retry/backoff — same protection
    call_model_with_retry gives the chat completion calls. A single
    dropped connection shouldn't crash an otherwise-working run.
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)


def get_embeddings(df: pd.DataFrame, force_refresh: bool = False) -> np.ndarray:
    """Return an (n_papers, dim) array of embeddings for title+abstract,
    in the same row order as df. Uses a local cache keyed by paper id.
    """
    if CACHE_PATH.exists() and CACHE_IDS_PATH.exists() and not force_refresh:
        cached_ids = pd.read_csv(CACHE_IDS_PATH)["id"].tolist()
        if cached_ids == df["id"].tolist():
            return np.load(CACHE_PATH)
        # Fall through and recompute if ids don't match (e.g. different review)

    # A generous timeout (default is 10s, too tight for a flaky connection)
    client = OpenAI(timeout=60.0)
    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = _embed_batch_with_retry(client, batch)
        all_embeddings.extend(batch_embeddings)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)} papers")

    embeddings = np.array(all_embeddings)

    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.save(CACHE_PATH, embeddings)
    df[["id"]].to_csv(CACHE_IDS_PATH, index=False)

    return embeddings


def retrieve_similar(query_idx: int, embeddings: np.ndarray, exclude_mask: np.ndarray,
                      k: int = 5) -> list:
    """Return indices of the k most similar papers to embeddings[query_idx],
    excluding any index where exclude_mask is True (e.g. same CV fold,
    to prevent leakage).
    """
    # Cosine similarity via normalized dot product
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = norm @ norm[query_idx]

    sims = sims.copy()
    sims[exclude_mask] = -np.inf
    sims[query_idx] = -np.inf  # never retrieve itself

    top_k_idx = np.argsort(sims)[::-1][:k]
    return top_k_idx.tolist()
