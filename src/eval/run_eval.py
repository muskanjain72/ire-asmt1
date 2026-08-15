"""
Q4 (part 5): Run the evaluation harness on real BM25 and embedding
retrieval results.

For each sampled impression:
  1. Reconstruct the actual shown-article set + click labels (by
     grouping the impressions table on impression_id)
  2. Build the query (BM25: history text, embeddings: mean-pooled
     history vector) from LEAKAGE-SAFE history only
  3. Score just those shown candidates with BM25 and with embeddings
  4. Feed (y_true, y_score) into metrics.evaluate_impressions
  5. Bootstrap 95% CIs on every metric
  6. Break down by cold-start vs warm user (reusing the same
     threshold as slicing.py / Q3)
  7. Beyond-accuracy (diversity/novelty/coverage) computed separately,
     using each method's top-K retrieval over the FULL corpus per
     user (not the impression subset — these describe candidate-
     generation quality, not within-impression reranking)

Usage:
    python src/eval/run_eval.py --dataset both --sample-size 300
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))

from feature_store import FeatureStore
from bm25 import BM25Index
from ann_index import ANNIndex, user_vector, score_candidates_batch as ann_score_candidates_batch
from embeddings import get_or_compute_embeddings

from metrics import evaluate_impressions
from bootstrap import bootstrap_all_metrics
from beyond_accuracy import intra_list_diversity, compute_item_popularity, novelty, catalog_coverage


def build_impression_groups(impressions: pd.DataFrame, sample_size: int, seed: int = 42):
    """
    Group the exploded impressions table back into one entry per
    impression: (impression_id, user_id, timestamp, article_ids, y_true).
    Only keeps impressions that have at least one click (needed for
    MRR/nDCG to be meaningful) and at least 2 candidates (needed for
    AUC to be defined).
    """
    groups = []
    for impression_id, group in impressions.groupby("impression_id"):
        y_true = group["clicked"].values
        if y_true.sum() == 0 or len(set(y_true)) < 2:
            continue
        groups.append({
            "impression_id": impression_id,
            "user_id": group["user_id"].iloc[0],
            "timestamp": group["timestamp"].iloc[0],
            "article_ids": group["article_id"].tolist(),
            "y_true": y_true,
        })

    if sample_size and len(groups) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(groups), sample_size, replace=False)
        groups = [groups[i] for i in idx]

    return groups


def run_bm25_eval(store: FeatureStore, bm25_index: BM25Index, groups: list, cold_start_threshold: int = 5):
    queries = [store.user_history_text(g["user_id"], g["timestamp"]) for g in groups]
    candidate_lists = [g["article_ids"] for g in groups]
    score_dicts = bm25_index.score_candidates_batch(queries, candidate_lists)

    return _finalize_scores(store, groups, score_dicts, cold_start_threshold)


def run_embedding_eval(store: FeatureStore, embeddings: dict, groups: list, cold_start_threshold: int = 5):
    query_vectors = []
    for g in groups:
        hist_ids = store.get_user_history(g["user_id"], g["timestamp"])
        query_vectors.append(user_vector(hist_ids, embeddings))
    candidate_lists = [g["article_ids"] for g in groups]
    score_dicts = ann_score_candidates_batch(query_vectors, candidate_lists, embeddings)

    return _finalize_scores(store, groups, score_dicts, cold_start_threshold)


def _finalize_scores(store: FeatureStore, groups: list, score_dicts: list, cold_start_threshold: int):
    """Shared logic: turn per-candidate score dicts into (y_true, y_score)
    pairs, tagged with cold_start/warm slice, skipping impressions where
    scoring failed (empty query -> all-zero scores)."""
    impressions_data = []
    slices = []

    for g, score_dict in zip(groups, score_dicts):
        if not score_dict or all(v == 0.0 for v in score_dict.values()):
            continue  # no usable signal (e.g. cold-start user, empty query)

        y_score = np.array([score_dict.get(aid, 0.0) for aid in g["article_ids"]])
        impressions_data.append((g["y_true"], y_score))

        n_clicks = store.get_user_recency_features(g["user_id"], g["timestamp"])["click_count_total"]
        slices.append("cold_start" if n_clicks < cold_start_threshold else "warm")

    return impressions_data, slices


def report_metrics(label: str, impressions_data: list, slices: list):
    print(f"\n--- {label} ---")
    if not impressions_data:
        print("  No usable impressions (all skipped — likely no history/embeddings).")
        return

    results = evaluate_impressions(impressions_data)
    cis = bootstrap_all_metrics(results)

    print(f"  n={len(impressions_data)}")
    for metric_name, ci in cis.items():
        if ci["mean"] is None:
            print(f"  {metric_name}: n/a")
        else:
            print(f"  {metric_name}: {ci['mean']:.4f}  95% CI=[{ci['lower']:.4f}, {ci['upper']:.4f}]  (n={ci['n']})")

    # slice breakdown
    slices = np.array(slices)
    for slice_name in ["cold_start", "warm"]:
        mask = slices == slice_name
        if mask.sum() == 0:
            print(f"  [{slice_name}] no samples")
            continue
        sliced_data = [d for d, m in zip(impressions_data, mask) if m]
        sliced_results = evaluate_impressions(sliced_data)
        print(f"  [{slice_name}] n={mask.sum()}: "
              + ", ".join(f"{k}={v['mean']:.4f}" if v["mean"] is not None else f"{k}=n/a"
                          for k, v in sliced_results.items()))


def report_beyond_accuracy(label: str, store: FeatureStore, groups: list, retrieve_fn, embeddings: dict,
                            train_impressions: pd.DataFrame, k: int = 20):
    """retrieve_fn(user_id, timestamp) -> list[article_id], the method's
    own top-K retrieval (full corpus), used to assess diversity/novelty/coverage
    of what each method WOULD recommend, independent of the impression-level scoring above."""
    print(f"\n--- {label} beyond-accuracy (top-{k}) ---")
    popularity = compute_item_popularity(train_impressions)
    total_users = train_impressions["user_id"].nunique()

    all_lists = []
    diversities, novelties = [], []
    for g in groups:
        candidates = retrieve_fn(g["user_id"], g["timestamp"])
        if not candidates:
            continue
        all_lists.append(candidates)
        div = intra_list_diversity(candidates, embeddings)
        if div is not None:
            diversities.append(div)
        novelties.append(novelty(candidates, popularity, total_users))

    catalog_size = len(store.articles)
    coverage = catalog_coverage(all_lists, catalog_size)

    print(f"  avg diversity: {np.mean(diversities):.4f}" if diversities else "  avg diversity: n/a")
    print(f"  avg novelty: {np.mean(novelties):.4f}" if novelties else "  avg novelty: n/a")
    covered_count = len(set().union(*all_lists)) if all_lists else 0
    print(f"  catalog coverage: {coverage:.4f} ({covered_count}/{catalog_size})")


def run_for_dataset(dataset: str, sample_size: int):
    print(f"\n{'=' * 60}\n{dataset.upper()}\n{'=' * 60}")

    store = FeatureStore("data/splits", dataset)
    val_impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
    train_impressions = pd.read_parquet(f"data/splits/{dataset}/train/impressions.parquet")

    print("Building impression groups...")
    groups = build_impression_groups(val_impressions, sample_size)
    print(f"  {len(groups)} usable impressions (>=1 click, >=2 candidates)")

    print("Building BM25 index...")
    bm25_index = BM25Index(store.articles.reset_index())

    print("Loading embeddings + building ANN index...")
    embeddings = get_or_compute_embeddings(dataset, store)
    ann_index = ANNIndex(embeddings)

    # --- accuracy metrics (AUC/MRR/nDCG) ---
    bm25_data, bm25_slices = run_bm25_eval(store, bm25_index, groups)
    report_metrics("BM25", bm25_data, bm25_slices)

    embed_data, embed_slices = run_embedding_eval(store, embeddings, groups)
    report_metrics("Embeddings", embed_data, embed_slices)

    # --- beyond-accuracy metrics ---
    def bm25_retrieve(user_id, ts):
        query = store.user_history_text(user_id, ts)
        return bm25_index.retrieve_topk(query, 20) if query.strip() else []

    def embed_retrieve(user_id, ts):
        hist_ids = store.get_user_history(user_id, ts)
        qvec = user_vector(hist_ids, embeddings)
        return ann_index.retrieve_topk(qvec, 20) if qvec is not None else []

    report_beyond_accuracy("BM25", store, groups, bm25_retrieve, embeddings, train_impressions)
    report_beyond_accuracy("Embeddings", store, groups, embed_retrieve, embeddings, train_impressions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=300,
                         help="number of impressions to evaluate (kept modest — BM25 "
                              "candidate scoring does a full-corpus ranking per impression)")
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        run_for_dataset(dataset, args.sample_size)