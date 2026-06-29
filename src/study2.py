"""
Study 2: selection efficiency.

Study 1 fixed the sample order and varied the SCORING method. Study 2
does the opposite: fixes the relevance signal (the zero-shot LLM's
already-computed scores, plus the cached embeddings) and varies the
ORDER papers get screened in. Question: does adaptively choosing what
to screen next reach high recall faster than a fixed/passive order?

Needs ZERO new API calls -- everything here reuses Study 1's saved
artifacts (zero_shot_predictions.csv, embeddings.npy).

Leakage discipline: embeddings are known for every paper from the
start (they come from title/abstract text, not from labels). Only the
LABEL of a paper is "hidden" until that paper is selected for
screening. Every policy below picks the next paper using only
embeddings of unscreened papers and labels of ALREADY-screened ones --
never the label of a paper before it's "screened."
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from data import load_review
from embed import get_embeddings

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ZERO_SHOT_PATH = RESULTS_DIR / "zero_shot_predictions.csv"

N_RANDOM_RUNS = 100
N_SEEDED_RUNS = 50
SEED_SIZE = 6
RETRAIN_EVERY = 5
RECALL_TARGET = 0.95
RANDOM_STATE = 42
N_BOOTSTRAP = 1000


def recall_curve(order: np.ndarray, y: np.ndarray) -> np.ndarray:
    """recall_curve[i] = recall after screening order[:i+1]."""
    n_pos = y.sum()
    found_cumulative = np.cumsum(y[order])
    return found_cumulative / n_pos


def papers_to_reach_target(curve: np.ndarray, target: float = RECALL_TARGET) -> int:
    """First N (1-indexed) at which recall >= target."""
    hits = np.where(curve >= target)[0]
    return int(hits[0] + 1) if len(hits) else len(curve)


def _build_bootstrap_sequence(order: np.ndarray, resample_counts: np.ndarray) -> np.ndarray:
    """Expand a policy's fixed preference order into a longer sequence
    where each original paper appears resample_counts[paper] times,
    preserving the policy's relative ordering. This is how a fixed
    ranking gets evaluated against a bootstrap-resampled population.
    """
    seq = []
    for idx in order:
        c = resample_counts[idx]
        if c:
            seq.extend([idx] * c)
    return np.array(seq)


def bootstrap_significance(all_orders: dict, y: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                            random_state: int = 123):
    """Paired bootstrap over PAPERS (not policy randomness): resample the
    270 papers with replacement, and evaluate every policy's already-
    computed preference order against that same resampled population.
    The same resample is shared across all policies within an iteration,
    so pairwise differences are paired comparisons -- directly analogous
    to the bootstrap test already validated in compare_study1.py.

    For stochastic policies (which have multiple precomputed orders),
    one of their existing orders is randomly selected each iteration --
    so the resulting CI reflects BOTH sources of uncertainty (the
    policy's own randomness AND sample composition) in one number. For
    the deterministic greedy policy, only sample-composition uncertainty
    applies, since there's only one order to choose from.

    Note on scope: this tests robustness of each policy's ALREADY-
    OBSERVED preference ordering to sampling variation. It does not
    re-run the full adaptive decision loop under each resampled
    population (that would require redoing the active-learning
    simulation per bootstrap draw, a much larger compute cost) -- this
    is the same "evaluate a fixed ranking under resampling" approach
    used for the PR-curve bootstrap in Study 1.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    names = list(all_orders.keys())

    per_policy_wss = {name: [] for name in names}
    pairwise_diffs = {f"{a} vs {b}": [] for i, a in enumerate(names) for b in names[i + 1:]}

    attempts = 0
    valid_iters = 0
    max_attempts = n_bootstrap * 3

    while valid_iters < n_bootstrap and attempts < max_attempts:
        attempts += 1
        idx = rng.randint(0, n, size=n)
        resample_counts = np.bincount(idx, minlength=n)

        iter_wss = {}
        skip_iteration = False
        for name in names:
            orders = all_orders[name]
            chosen_order = orders[rng.randint(0, len(orders))]
            seq = _build_bootstrap_sequence(chosen_order, resample_counts)
            y_seq = y[seq]
            n_pos = y_seq.sum()
            if n_pos == 0:
                skip_iteration = True  # this resample has zero positives -- undefined recall
                break
            curve = np.cumsum(y_seq) / n_pos
            n_needed = papers_to_reach_target(curve)
            iter_wss[name] = 1 - n_needed / len(seq)

        if skip_iteration:
            continue

        valid_iters += 1
        for name in names:
            per_policy_wss[name].append(iter_wss[name])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairwise_diffs[f"{a} vs {b}"].append(iter_wss[b] - iter_wss[a])

    policy_rows = []
    for name in names:
        vals = np.array(per_policy_wss[name])
        policy_rows.append({
            "policy": name,
            "wss_mean": round(float(vals.mean()), 4),
            "wss_ci_low": round(float(np.percentile(vals, 2.5)), 4),
            "wss_ci_high": round(float(np.percentile(vals, 97.5)), 4),
        })

    pairwise_rows = []
    for comparison, diffs in pairwise_diffs.items():
        diffs = np.array(diffs)
        lower, upper = np.percentile(diffs, [2.5, 97.5])
        significant = not (lower <= 0 <= upper)
        pairwise_rows.append({
            "comparison": comparison,
            "mean_diff": round(float(diffs.mean()), 4),
            "ci_95_low": round(float(lower), 4),
            "ci_95_high": round(float(upper), 4),
            "verdict": "significant (CI excludes 0)" if significant else "not distinguishable from noise",
        })

    return pd.DataFrame(policy_rows), pd.DataFrame(pairwise_rows), valid_iters


def policy_random(n: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.permutation(n)


def policy_greedy_score(score: np.ndarray) -> np.ndarray:
    """Screen in order of the zero-shot LLM's score, highest first.
    Uses ONLY the precomputed score -- never touches labels.
    """
    return np.argsort(-score, kind="stable")


def _stratified_seed(y: np.ndarray, seed_size: int, rng: np.random.RandomState) -> list:
    """A random seed set guaranteed to contain at least one positive and
    one negative -- otherwise relevance sampling has no relevant centroid
    to start from, and uncertainty sampling can't fit a binary classifier.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    seed = list(rng.choice(pos_idx, size=1, replace=False))
    seed += list(rng.choice(neg_idx, size=seed_size - 1, replace=False))
    rng.shuffle(seed)
    return seed


def policy_relevance_sampling(embeddings: np.ndarray, y: np.ndarray, seed: list,
                               rng: np.random.RandomState) -> np.ndarray:
    """Adaptive: always pick the unscreened paper most similar (cosine)
    to the centroid of relevant papers found so far. Falls back to a
    random pick if no relevant papers have been found yet.
    """
    n = len(y)
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    order = list(seed)
    remaining = set(range(n)) - set(order)

    while remaining:
        relevant_so_far = [i for i in order if y[i] == 1]  # only ALREADY-screened labels
        remaining_list = list(remaining)

        if not relevant_so_far:
            next_idx = rng.choice(remaining_list)
        else:
            centroid = norm_emb[relevant_so_far].mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            sims = norm_emb[remaining_list] @ centroid  # only uses remaining EMBEDDINGS, not labels
            next_idx = remaining_list[int(np.argmax(sims))]

        order.append(next_idx)
        remaining.discard(next_idx)

    return np.array(order)


def policy_uncertainty_sampling(embeddings: np.ndarray, y: np.ndarray, seed: list,
                                 rng: np.random.RandomState,
                                 retrain_every: int = RETRAIN_EVERY) -> np.ndarray:
    """Adaptive: maintain a logistic regression on embeddings, retrained
    periodically on revealed labels, always pick the unscreened paper
    closest to its decision boundary (most uncertain) next.
    """
    n = len(y)
    order = list(seed)
    remaining = set(range(n)) - set(order)
    model = None
    steps_since_retrain = retrain_every  # force a retrain on the first step

    while remaining:
        if steps_since_retrain >= retrain_every:
            model = LogisticRegression(class_weight="balanced", max_iter=500)
            model.fit(embeddings[order], y[order])  # only ALREADY-screened labels
            steps_since_retrain = 0

        remaining_list = list(remaining)
        proba = model.predict_proba(embeddings[remaining_list])[:, 1]  # only remaining EMBEDDINGS
        uncertainty = -np.abs(proba - 0.5)
        next_idx = remaining_list[int(np.argmax(uncertainty))]

        order.append(next_idx)
        remaining.discard(next_idx)
        steps_since_retrain += 1

    return np.array(order)


def run_study2():
    df = load_review()
    y = df["label_included"].values
    n = len(df)

    zero_shot = pd.read_csv(ZERO_SHOT_PATH)
    score_map = zero_shot.set_index("id")["zero_shot_proba"]
    score = df["id"].map(score_map).values

    print("Getting embeddings (cached from the RAG step — no API calls here)...")
    embeddings = get_embeddings(df)

    results = {}
    all_orders = {}

    # --- Random order (averaged over many runs) ---
    rng = np.random.RandomState(RANDOM_STATE)
    random_orders = [policy_random(n, rng) for _ in range(N_RANDOM_RUNS)]
    results["Random order"] = np.array([recall_curve(o, y) for o in random_orders])
    all_orders["Random order"] = random_orders

    # --- Greedy by LLM score (deterministic, single curve) ---
    order = policy_greedy_score(score)
    results["Greedy by LLM score"] = recall_curve(order, y)[None, :]
    all_orders["Greedy by LLM score"] = [order]

    # --- Relevance sampling (averaged, same seeds as uncertainty sampling for a fair pairing) ---
    rng = np.random.RandomState(RANDOM_STATE)
    seeds = [_stratified_seed(y, SEED_SIZE, rng) for _ in range(N_SEEDED_RUNS)]

    rel_orders = []
    for seed in seeds:
        run_rng = np.random.RandomState(rng.randint(0, 1_000_000))
        rel_orders.append(policy_relevance_sampling(embeddings, y, seed, run_rng))
    results["Relevance sampling"] = np.array([recall_curve(o, y) for o in rel_orders])
    all_orders["Relevance sampling"] = rel_orders

    # --- Uncertainty sampling (same seeds, paired comparison) ---
    unc_orders = []
    for seed in seeds:
        unc_orders.append(policy_uncertainty_sampling(embeddings, y, seed, np.random.RandomState(RANDOM_STATE)))
    results["Uncertainty sampling"] = np.array([recall_curve(o, y) for o in unc_orders])
    all_orders["Uncertainty sampling"] = unc_orders

    # --- Summary: WSS@95 ---
    summary_rows = []
    plt.figure(figsize=(7, 5))
    x = np.arange(1, n + 1)

    for name, curves in results.items():
        mean_curve = curves.mean(axis=0)
        n_needed = [papers_to_reach_target(c) for c in curves]
        mean_n_needed = float(np.mean(n_needed))
        wss95 = 1 - mean_n_needed / n

        summary_rows.append({
            "policy": name,
            "mean_papers_to_95pct_recall": round(mean_n_needed, 1),
            "wss_at_95": round(wss95, 4),
        })

        plt.plot(x, mean_curve, label=name)
        if curves.shape[0] > 1:
            lower = np.percentile(curves, 10, axis=0)
            upper = np.percentile(curves, 90, axis=0)
            plt.fill_between(x, lower, upper, alpha=0.15)

    plt.axhline(RECALL_TARGET, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Papers screened")
    plt.ylabel("Recall (fraction of true relevant papers found)")
    plt.title("Study 2 — Recall vs. papers screened, by selection policy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "study2_recall_curves.png", dpi=150)

    summary = pd.DataFrame(summary_rows).sort_values("wss_at_95", ascending=False)
    summary.to_csv(RESULTS_DIR / "study2_summary.csv", index=False)

    print(f"\nStudy 2 — selection policy comparison (n={n}, {int(y.sum())} relevant)")
    print(summary.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR / 'study2_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'study2_recall_curves.png'}")

    print(f"\nBootstrap significance testing ({N_BOOTSTRAP} paper-resampled iterations)...")
    policy_df, pairwise_df, valid_iters = bootstrap_significance(all_orders, y)
    print(f"(valid iterations: {valid_iters}/{N_BOOTSTRAP})\n")

    print("Per-policy WSS@95 with 95% CI:")
    print(policy_df.to_string(index=False))
    print("\nPairwise differences (95% CI on the gap, positive = second-named policy higher):")
    print(pairwise_df.to_string(index=False))

    policy_df.to_csv(RESULTS_DIR / "study2_policy_ci.csv", index=False)
    pairwise_df.to_csv(RESULTS_DIR / "study2_pairwise_significance.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'study2_policy_ci.csv'}")
    print(f"Saved: {RESULTS_DIR / 'study2_pairwise_significance.csv'}")


if __name__ == "__main__":
    run_study2()
