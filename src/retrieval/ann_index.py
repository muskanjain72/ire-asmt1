"""
Q3 (part 2): ANN index + semantic candidate retrieval.

Builds a FAISS index over article embeddings, represents a user as
the mean-pooled embedding of their (leakage-safe) clicked history,
and retrieves top-K nearest articles by cosine similarity.

Uses IndexFlatIP (brute-force inner product over L2-normalized
vectors == cosine similarity). At this corpus size (11k-51k articles)
brute-force is fast enough; FAISS is used anyway since it's the
standard tool and the same code would scale to a real ANN index
(IVF/HNSW) with a one-line change if the corpus grew much larger —
worth naming in the Q6 "where it breaks at 10x" discussion.
"""

import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from feature_store import FeatureStore
from embeddings import get_or_compute_embeddings


class ANNIndex:
    def __init__(self, embeddings: dict[str, np.ndarray]):
        self.article_ids = list(embeddings.keys())
        matrix = np.stack([embeddings[aid] for aid in self.article_ids]).astype(np.float32)
        faiss.normalize_L2(matrix)  # so inner product == cosine similarity

        self.dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)

    def retrieve_topk(self, query_vector: np.ndarray, k: int) -> list[str]:
        if query_vector is None:
            return []
        q = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        k = min(k, len(self.article_ids))
        _, indices = self.index.search(q, k)
        return [self.article_ids[i] for i in indices[0] if i != -1]

    def retrieve_topk_batch(self, query_vectors: list, k: int) -> list:
        """Batched retrieval — much faster than looping retrieve_topk."""
        valid_idx = [i for i, v in enumerate(query_vectors) if v is not None]
        if not valid_idx:
            return [[] for _ in query_vectors]

        stacked = np.stack([query_vectors[i] for i in valid_idx]).astype(np.float32)
        faiss.normalize_L2(stacked)
        k = min(k, len(self.article_ids))
        _, indices = self.index.search(stacked, k)

        out = [[] for _ in query_vectors]
        for row, orig_idx in enumerate(valid_idx):
            out[orig_idx] = [self.article_ids[i] for i in indices[row] if i != -1]
        return out


def user_vector(article_ids: list, embeddings: dict):
    """Mean-pool the embeddings of a user's clicked history. Returns
    None if the user has no history with a known embedding (cold-start
    or history points to articles outside this embedding set)."""
    vecs = [embeddings[aid] for aid in article_ids if aid in embeddings]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def evaluate_recall_at_k(
    ann_index: ANNIndex,
    embeddings: dict,
    feature_store: FeatureStore,
    impressions: pd.DataFrame,
    ks: list = [50, 100, 200],
    sample_size: int = 500,
    seed: int = 42,
) -> dict:
    """Same protocol as bm25.evaluate_recall_at_k, for apples-to-apples
    comparison: sample clicked impressions, build the user's query
    (here: mean-pooled embedding of leakage-safe history), check if
    the truly-clicked article is in the top-K."""
    clicked = impressions[impressions["clicked"] == 1]
    if sample_size and len(clicked) > sample_size:
        clicked = clicked.sample(sample_size, random_state=seed)

    max_k = max(ks)

    query_vectors = []
    for row in clicked.itertuples(index=False):
        hist_ids = feature_store.get_user_history(row.user_id, row.timestamp)
        query_vectors.append(user_vector(hist_ids, embeddings))

    all_candidates = ann_index.retrieve_topk_batch(query_vectors, max_k)

    hits = {k: 0 for k in ks}
    total = 0
    for row, candidates in zip(clicked.itertuples(index=False), all_candidates):
        if not candidates:
            continue
        total += 1
        for k in ks:
            if row.article_id in candidates[:k]:
                hits[k] += 1

    return {k: (hits[k] / total if total > 0 else 0.0) for k in ks}


def run_for_dataset(dataset: str, ks: list, sample_size: int) -> None:
    print(f"\n=== {dataset.upper()} ===")

    store = FeatureStore("data/splits", dataset)
    print("Loading/computing article embeddings...")
    embeddings = get_or_compute_embeddings(dataset, store)

    print(f"Building FAISS index over {len(embeddings)} article embeddings...")
    index = ANNIndex(embeddings)

    val_impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
    print(f"Evaluating on val split (sampling up to {sample_size} clicked impressions)...")

    recall = evaluate_recall_at_k(index, embeddings, store, val_impressions, ks, sample_size)
    for k in ks:
        print(f"  Recall@{k}: {recall[k]:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    ks = [50, 100, 200]
    datasets = ["mind", "ebnerd"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        run_for_dataset(dataset, ks, args.sample_size)