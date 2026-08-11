"""
Q4 (part 1): Official ranking metrics — AUC, MRR, nDCG@5, nDCG@10.

Each metric is computed PER IMPRESSION (given the shown articles'
true click labels and the model's scores for them), then averaged
across impressions — this is the standard MIND/EB-NeRD leaderboard
protocol (macro-average over impressions, not one global AUC over
all rows pooled together).

Impressions where AUC is undefined (all labels the same, e.g. no
clicks or all clicks) are skipped for AUC specifically, but still
contribute to MRR/nDCG since those remain well-defined.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """AUC for one impression. Returns None if undefined (only one
    class present — can't rank clicked vs non-clicked if there's
    only one kind of label)."""
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Reciprocal Rank for one impression: 1 / rank of the first
    relevant (clicked) item, where rank is 1-indexed position after
    sorting by y_score descending. 0 if no relevant item at all."""
    order = np.argsort(y_score)[::-1]
    ranked_true = y_true[order]
    relevant_positions = np.where(ranked_true == 1)[0]
    if len(relevant_positions) == 0:
        return 0.0
    first_relevant_rank = relevant_positions[0] + 1  # 1-indexed
    return 1.0 / first_relevant_rank


def dcg_at_k(ranked_relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at k, given relevance labels already
    sorted in the order the model ranked them (descending by score)."""
    ranked_relevance = ranked_relevance[:k]
    discounts = np.log2(np.arange(2, len(ranked_relevance) + 2))
    return float(np.sum(ranked_relevance / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Normalized DCG at k for one impression."""
    order = np.argsort(y_score)[::-1]
    ranked_relevance = y_true[order]

    dcg = dcg_at_k(ranked_relevance, k)

    ideal_relevance = np.sort(y_true)[::-1]
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_impressions(impressions: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    impressions: list of (y_true, y_score) pairs, one per impression.
    y_true: binary array (1=clicked, 0=not clicked) for the shown articles.
    y_score: model's score for each shown article, same order as y_true.

    Returns per-metric arrays (not just the mean) so callers can feed
    them straight into bootstrap.py for confidence intervals, plus the
    mean of each for a quick summary.
    """
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

    for y_true, y_score in impressions:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        auc = auc_score(y_true, y_score)
        if auc is not None:
            aucs.append(auc)

        mrrs.append(mrr_score(y_true, y_score))
        ndcg5s.append(ndcg_at_k(y_true, y_score, 5))
        ndcg10s.append(ndcg_at_k(y_true, y_score, 10))

    return {
        "auc": {"values": np.array(aucs), "mean": float(np.mean(aucs)) if aucs else None},
        "mrr": {"values": np.array(mrrs), "mean": float(np.mean(mrrs)) if mrrs else None},
        "ndcg@5": {"values": np.array(ndcg5s), "mean": float(np.mean(ndcg5s)) if ndcg5s else None},
        "ndcg@10": {"values": np.array(ndcg10s), "mean": float(np.mean(ndcg10s)) if ndcg10s else None},
    }


if __name__ == "__main__":
    # Synthetic self-test — doesn't need real data, just checks the
    # metrics behave sanely before wiring them to BM25/embeddings.
    rng = np.random.default_rng(0)

    synthetic_impressions = []
    for _ in range(200):
        n = rng.integers(3, 15)
        y_true = np.zeros(n)
        y_true[rng.integers(0, n)] = 1  # exactly one click, like MIND/EB-NeRD impressions
        y_score = rng.random(n)
        synthetic_impressions.append((y_true, y_score))

    results = evaluate_impressions(synthetic_impressions)
    print("Synthetic random-scoring sanity check (expect AUC~0.5, low MRR/nDCG):")
    for name, r in results.items():
        print(f"  {name}: {r['mean']:.4f}")

    # a "perfect" model that always scores the true clicked item highest
    perfect_impressions = []
    for y_true, _ in synthetic_impressions:
        y_score = y_true.copy() + rng.random(len(y_true)) * 0.01  # clicked item gets highest score
        perfect_impressions.append((y_true, y_score))

    results = evaluate_impressions(perfect_impressions)
    print("\nSynthetic near-perfect scoring (expect all metrics near 1.0):")
    for name, r in results.items():
        print(f"  {name}: {r['mean']:.4f}")