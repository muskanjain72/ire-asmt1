# # """
# # Q2: Lexical candidate generation via BM25.

# # Builds an inverted-index-backed BM25 ranker over each dataset's
# # article titles + abstracts, uses feature_store.user_history_text()
# # to turn a user's leakage-safe click history into a query, retrieves
# # top-K candidates, and reports Recall@K for K in {50, 100, 200}.

# # Uses the rank_bm25 library (Okapi BM25 — the exact formula: TF
# # saturation via k1, length normalization via b) rather than a
# # hand-rolled index, since the goal here is correct application +
# # evaluation, not reimplementing the scoring math.

# # NOTE ON SCALE: MIND's val split has ~576k impressions against a
# # 51k-article corpus. Scoring every impression against the full corpus
# # in pure Python is slow, so evaluate_recall_at_k samples a subset of
# # impressions by default (see --sample-size). This is a real scale
# # limitation worth naming explicitly in the Q6 design note.
# # """

# # import argparse
# # import re
# # import sys
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # from rank_bm25 import BM25Okapi

# # sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
# # from feature_store import FeatureStore


# # _TOKEN_RE = re.compile(r"[a-zA-ZæøåÆØÅ0-9]+")


# # def tokenize(text: str) -> list[str]:
# #     """Simple lowercase alphanumeric tokenizer. Works for both English
# #     (MIND) and Danish (EB-NeRD) since we keep Danish-specific letters."""
# #     if not text:
# #         return []
# #     return _TOKEN_RE.findall(text.lower())


# # class BM25Index:
# #     def __init__(self, articles: pd.DataFrame):
# #         """
# #         articles: the unified `articles` table (article_id, title,
# #         abstract, ...). Indexes over title + abstract.
# #         """
# #         self.article_ids = articles["article_id"].tolist()
# #         corpus_text = (articles["title"].fillna("") + " " + articles["abstract"].fillna(""))
# #         self.tokenized_corpus = [tokenize(t) for t in corpus_text]
# #         self.bm25 = BM25Okapi(self.tokenized_corpus)

# #     def retrieve_topk(self, query_text: str, k: int) -> list[str]:
# #         """Return up to k article_ids ranked by BM25 score, descending."""
# #         query_tokens = tokenize(query_text)
# #         if not query_tokens:
# #             return []
# #         scores = self.bm25.get_scores(query_tokens)
# #         top_idx = np.argsort(scores)[::-1][:k]
# #         return [self.article_ids[i] for i in top_idx]


# # def evaluate_recall_at_k(
# #     bm25_index: BM25Index,
# #     feature_store: FeatureStore,
# #     impressions: pd.DataFrame,
# #     ks: list[int] = [50, 100, 200],
# #     sample_size: int = 2000,
# #     seed: int = 42,
# # ) -> dict[int, float]:
# #     """
# #     Recall@K: of the impressions where the user actually clicked
# #     something, what fraction of the time does the clicked article
# #     appear in our top-K BM25 candidates for that impression's query?

# #     Query = concatenated titles of the user's leakage-safe history
# #     (feature_store.user_history_text), i.e. what BM25 sees is built
# #     only from clicks strictly before this impression's timestamp.

# #     Only impressions with a positive click are used, since Recall@K
# #     is undefined for impressions with no relevant item.
# #     """
# #     clicked = impressions[impressions["clicked"] == 1]
# #     if sample_size and len(clicked) > sample_size:
# #         clicked = clicked.sample(sample_size, random_state=seed)

# #     max_k = max(ks)
# #     hits = {k: 0 for k in ks}
# #     total = 0

# #     for row in clicked.itertuples(index=False):
# #         query = feature_store.user_history_text(row.user_id, row.timestamp)
# #         if not query:
# #             continue  # cold-start user with no history — BM25 query would be empty

# #         candidates = bm25_index.retrieve_topk(query, max_k)
# #         total += 1
# #         for k in ks:
# #             if row.article_id in candidates[:k]:
# #                 hits[k] += 1

# #     return {k: (hits[k] / total if total > 0 else 0.0) for k in ks}


# # def run_for_dataset(dataset: str, ks: list[int], sample_size: int) -> None:
# #     print(f"\n=== {dataset.upper()} ===")

# #     store = FeatureStore("data/splits", dataset)
# #     print(f"Building BM25 index over {len(store.articles)} articles...")
# #     bm25_index = BM25Index(store.articles.reset_index())

# #     val_impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
# #     print(f"Evaluating on val split ({len(val_impressions)} impressions, "
# #           f"sampling up to {sample_size} clicked ones)...")

# #     recall = evaluate_recall_at_k(bm25_index, store, val_impressions, ks, sample_size)
# #     for k in ks:
# #         print(f"  Recall@{k}: {recall[k]:.4f}")


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--dataset", choices=["mind", "ebnerd", "both"], default="both")
# #     parser.add_argument("--sample-size", type=int, default=2000)
# #     args = parser.parse_args()

# #     ks = [50, 100, 200]
# #     datasets = ["mind", "ebnerd"] if args.dataset == "both" else [args.dataset]

# #     for dataset in datasets:
# #         run_for_dataset(dataset, ks, args.sample_size)


# #initial commented
# #from rank_bm25 , we shifted to bm25s , since it was slow for larger datasets

# """
# Q2: Lexical candidate generation via BM25.

# Builds an inverted-index-backed BM25 ranker over each dataset's
# article titles + abstracts, uses feature_store.user_history_text()
# to turn a user's leakage-safe click history into a query, retrieves
# top-K candidates, and reports Recall@K for K in {50, 100, 200}.

# Uses the rank_bm25 library (Okapi BM25 — the exact formula: TF
# saturation via k1, length normalization via b) rather than a
# hand-rolled index, since the goal here is correct application +
# evaluation, not reimplementing the scoring math.

# NOTE ON SCALE: MIND's val split has ~576k impressions against a
# 51k-article corpus. Scoring every impression against the full corpus
# in pure Python is slow, so evaluate_recall_at_k samples a subset of
# impressions by default (see --sample-size). This is a real scale
# limitation worth naming explicitly in the Q6 design note.
# """



# import argparse
# import re
# import sys
# from pathlib import Path

# import bm25s
# import numpy as np
# import pandas as pd

# sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
# from feature_store import FeatureStore


# _TOKEN_RE = re.compile(r"[a-zA-ZæøåÆØÅ0-9]+")


# def tokenize(text: str) -> list[str]:
#     """Simple lowercase alphanumeric tokenizer. Works for both English
#     (MIND) and Danish (EB-NeRD) since we keep Danish-specific letters."""
#     if not text:
#         return []
#     return _TOKEN_RE.findall(text.lower())


# class BM25Index:
#     """
#     Vectorized BM25 index using the bm25s library (sparse-matrix
#     scoring) instead of rank_bm25's pure-Python per-token scan —
#     orders of magnitude faster at this corpus size (50k+ docs,
#     thousands of queries with long concatenated-history text).
#     """

#     def __init__(self, articles: pd.DataFrame):
#         """
#         articles: the unified `articles` table (article_id, title,
#         abstract, ...). Indexes over title + abstract.
#         """
#         self.article_ids = articles["article_id"].tolist()
#         corpus_text = (articles["title"].fillna("") + " " + articles["abstract"].fillna(""))
#         tokenized_corpus = bm25s.tokenize(corpus_text.tolist(), stopwords=None, show_progress=False)
#         self.bm25 = bm25s.BM25()
#         self.bm25.index(tokenized_corpus, show_progress=False)

#     def retrieve_topk(self, query_text: str, k: int) -> list[str]:
#         """Return up to k article_ids ranked by BM25 score, descending."""
#         if not query_text.strip():
#             return []
#         query_tokens = bm25s.tokenize([query_text], stopwords=None, show_progress=False)
#         k = min(k, len(self.article_ids))
#         results, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)
#         return [self.article_ids[i] for i in results[0]]

#     def retrieve_topk_batch(self, query_texts: list[str], k: int) -> list[list[str]]:
#         """Batched version — much faster than calling retrieve_topk in a
#         loop, since bm25s vectorizes scoring across all queries at once."""
#         non_empty_idx = [i for i, q in enumerate(query_texts) if q.strip()]
#         if not non_empty_idx:
#             return [[] for _ in query_texts]

#         non_empty_queries = [query_texts[i] for i in non_empty_idx]
#         query_tokens = bm25s.tokenize(non_empty_queries, stopwords=None, show_progress=False)
#         k = min(k, len(self.article_ids))
#         results, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)

#         out = [[] for _ in query_texts]
#         for row, orig_idx in enumerate(non_empty_idx):
#             out[orig_idx] = [self.article_ids[i] for i in results[row]]
#         return out


# # =====================================================================
# # MANUAL BM25 IMPLEMENTATION (For Assignment Requirements)
# # 
# # Instructions:
# # 1. Uncomment this block of code (the ManualBM25Index class).
# # 2. In `run_for_dataset`, replace:
# #        bm25_index = BM25Index(store.articles.reset_index())
# #    with:
# #        bm25_index = ManualBM25Index(store.articles.reset_index())
# # 3. You will also need to add `import math` and `from collections import defaultdict, Counter` 
# #    at the top of this file.
# # =====================================================================

# # import math
# # from collections import defaultdict, Counter
# # 
# # class ManualBM25Index:
# #     """
# #     Manual inverted index implementation of BM25 from scratch,
# #     using no external scoring libraries like rank_bm25 or bm25s.
# #     """
# #     def __init__(self, articles: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
# #         self.k1 = k1
# #         self.b = b
# #         self.article_ids = articles["article_id"].tolist()
# #         
# #         print("  Building manual inverted index...")
# #         corpus_text = (articles["title"].fillna("") + " " + articles["abstract"].fillna(""))
# #         
# #         # 1. Tokenize corpus
# #         tokenized_corpus = [tokenize(t) for t in corpus_text]
# #         self.N = len(tokenized_corpus)
# #         self.avgdl = sum(len(doc) for doc in tokenized_corpus) / self.N if self.N > 0 else 0
# #         
# #         # 2. Build inverted index and compute document frequencies (DF)
# #         # inverted_index maps: term -> list of (doc_index, term_freq)
# #         self.inverted_index = defaultdict(list)
# #         self.df = defaultdict(int)
# #         self.doc_lengths = []
# #         
# #         for doc_idx, tokens in enumerate(tokenized_corpus):
# #             self.doc_lengths.append(len(tokens))
# #             term_counts = Counter(tokens)
# #             for term, count in term_counts.items():
# #                 self.inverted_index[term].append((doc_idx, count))
# #                 self.df[term] += 1
# #                 
# #         # 3. Compute IDF for each term in the vocabulary
# #         self.idf = {}
# #         for term, df_count in self.df.items():
# #             # Standard BM25 IDF formula
# #             idf_val = math.log(1 + (self.N - df_count + 0.5) / (df_count + 0.5))
# #             self.idf[term] = idf_val
# # 
# #     def retrieve_topk(self, query_text: str, k: int) -> list[str]:
# #         query_tokens = tokenize(query_text)
# #         if not query_tokens:
# #             return []
# #             
# #         doc_scores = defaultdict(float)
# #         
# #         # Calculate BM25 score for each query term
# #         for term in query_tokens:
# #             if term not in self.inverted_index:
# #                 continue
# #             idf_val = self.idf[term]
# #             
# #             # Iterate through all documents containing the term
# #             for doc_idx, tf in self.inverted_index[term]:
# #                 dl = self.doc_lengths[doc_idx]
# #                 numerator = tf * (self.k1 + 1)
# #                 denominator = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
# #                 doc_scores[doc_idx] += idf_val * (numerator / denominator)
# #                 
# #         if not doc_scores:
# #             return []
# #             
# #         # Sort documents by score descending
# #         top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
# #         return [self.article_ids[doc_idx] for doc_idx, score in top_docs]
# # 
# #     def retrieve_topk_batch(self, query_texts: list[str], k: int) -> list[list[str]]:
# #         # Fallback to looping for the manual implementation
# #         return [self.retrieve_topk(q, k) for q in query_texts]
# # 
# # =====================================================================


# def evaluate_recall_at_k(
#     bm25_index: BM25Index,
#     feature_store: FeatureStore,
#     impressions: pd.DataFrame,
#     ks: list[int] = [50, 100, 200],
#     sample_size: int = 2000,
#     seed: int = 42,
# ) -> dict[int, float]:
#     """
#     Recall@K: of the impressions where the user actually clicked
#     something, what fraction of the time does the clicked article
#     appear in our top-K BM25 candidates for that impression's query?

#     Query = concatenated titles of the user's leakage-safe history
#     (feature_store.user_history_text), i.e. what BM25 sees is built
#     only from clicks strictly before this impression's timestamp.

#     Only impressions with a positive click are used, since Recall@K
#     is undefined for impressions with no relevant item.
#     """
#     clicked = impressions[impressions["clicked"] == 1]
#     if sample_size and len(clicked) > sample_size:
#         clicked = clicked.sample(sample_size, random_state=seed)

#     max_k = max(ks)

#     # build all queries first, then score them in one batched call —
#     # this is what actually makes bm25s fast (vectorized across queries)
#     queries = [
#         feature_store.user_history_text(row.user_id, row.timestamp)
#         for row in clicked.itertuples(index=False)
#     ]
#     all_candidates = bm25_index.retrieve_topk_batch(queries, max_k)

#     hits = {k: 0 for k in ks}
#     total = 0
#     for row, candidates in zip(clicked.itertuples(index=False), all_candidates):
#         if not candidates:
#             continue  # cold-start user with no history — query was empty
#         total += 1
#         for k in ks:
#             if row.article_id in candidates[:k]:
#                 hits[k] += 1

#     return {k: (hits[k] / total if total > 0 else 0.0) for k in ks}


# def run_for_dataset(dataset: str, ks: list[int], sample_size: int) -> None:
#     print(f"\n=== {dataset.upper()} ===")

#     store = FeatureStore("data/splits", dataset)
#     print(f"Building BM25 index over {len(store.articles)} articles...")
#     bm25_index = BM25Index(store.articles.reset_index())

#     val_impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
#     print(f"Evaluating on val split ({len(val_impressions)} impressions, "
#           f"sampling up to {sample_size} clicked ones)...")

#     recall = evaluate_recall_at_k(bm25_index, store, val_impressions, ks, sample_size)
#     for k in ks:
#         print(f"  Recall@{k}: {recall[k]:.4f}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", choices=["mind", "ebnerd", "both"], default="both")
#     parser.add_argument("--sample-size", type=int, default=2000)
#     args = parser.parse_args()

#     ks = [50, 100, 200]
#     datasets = ["mind", "ebnerd"] if args.dataset == "both" else [args.dataset]

#     for dataset in datasets:
#         run_for_dataset(dataset, ks, args.sample_size)


"""
Q2: Lexical candidate generation via BM25.

Builds an inverted-index-backed BM25 ranker over each dataset's
article titles + abstracts, uses feature_store.user_history_text()
to turn a user's leakage-safe click history into a query, retrieves
top-K candidates, and reports Recall@K for K in {50, 100, 200}.

Uses the rank_bm25 library (Okapi BM25 — the exact formula: TF
saturation via k1, length normalization via b) rather than a
hand-rolled index, since the goal here is correct application +
evaluation, not reimplementing the scoring math.

NOTE ON SCALE: MIND's val split has ~576k impressions against a
51k-article corpus. Scoring every impression against the full corpus
in pure Python is slow, so evaluate_recall_at_k samples a subset of
impressions by default (see --sample-size). This is a real scale
limitation worth naming explicitly in the Q6 design note.
"""

import argparse
import re
import sys
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from feature_store import FeatureStore


_TOKEN_RE = re.compile(r"[a-zA-ZæøåÆØÅ0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer. Works for both English
    (MIND) and Danish (EB-NeRD) since we keep Danish-specific letters."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    Vectorized BM25 index using the bm25s library (sparse-matrix
    scoring) instead of rank_bm25's pure-Python per-token scan —
    orders of magnitude faster at this corpus size (50k+ docs,
    thousands of queries with long concatenated-history text).
    """

    def __init__(self, articles: pd.DataFrame):
        """
        articles: the unified `articles` table (article_id, title,
        abstract, ...). Indexes over title + abstract.
        """
        self.article_ids = articles["article_id"].tolist()
        corpus_text = (articles["title"].fillna("") + " " + articles["abstract"].fillna(""))
        tokenized_corpus = bm25s.tokenize(corpus_text.tolist(), stopwords=None, show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokenized_corpus, show_progress=False)

    def retrieve_topk(self, query_text: str, k: int) -> list[str]:
        """Return up to k article_ids ranked by BM25 score, descending."""
        if not query_text.strip():
            return []
        query_tokens = bm25s.tokenize([query_text], stopwords=None, show_progress=False)
        k = min(k, len(self.article_ids))
        results, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)
        return [self.article_ids[i] for i in results[0]]

    def retrieve_topk_batch(self, query_texts: list[str], k: int) -> list[list[str]]:
        """Batched version — much faster than calling retrieve_topk in a
        loop, since bm25s vectorizes scoring across all queries at once."""
        non_empty_idx = [i for i, q in enumerate(query_texts) if q.strip()]
        if not non_empty_idx:
            return [[] for _ in query_texts]

        non_empty_queries = [query_texts[i] for i in non_empty_idx]
        query_tokens = bm25s.tokenize(non_empty_queries, stopwords=None, show_progress=False)
        k = min(k, len(self.article_ids))
        results, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)

        out = [[] for _ in query_texts]
        for row, orig_idx in enumerate(non_empty_idx):
            out[orig_idx] = [self.article_ids[i] for i in results[row]]
        return out


def evaluate_recall_at_k(
    bm25_index: BM25Index,
    feature_store: FeatureStore,
    impressions: pd.DataFrame,
    ks: list[int] = [50, 100, 200],
    sample_size: int = 2000,
    seed: int = 42,
    cold_start_threshold: int = 5,
) -> dict:
    """
    Recall@K, computed overall AND split by cold-start vs warm user
    (using feature_store's leakage-safe click_count_total, same
    threshold as slicing.py). Returns:
      {
        "overall": {k: recall, ...},
        "cold_start": {k: recall, ...},
        "warm": {k: recall, ...},
        "n_overall": int, "n_cold_start": int, "n_warm": int,
      }
    """
    clicked = impressions[impressions["clicked"] == 1]
    if sample_size and len(clicked) > sample_size:
        clicked = clicked.sample(sample_size, random_state=seed)

    max_k = max(ks)

    queries = []
    user_slices = []
    for row in clicked.itertuples(index=False):
        queries.append(feature_store.user_history_text(row.user_id, row.timestamp))
        n_clicks = feature_store.get_user_recency_features(
            row.user_id, row.timestamp
        )["click_count_total"]
        user_slices.append("cold_start" if n_clicks < cold_start_threshold else "warm")

    all_candidates = bm25_index.retrieve_topk_batch(queries, max_k)

    hits = {"overall": {k: 0 for k in ks}, "cold_start": {k: 0 for k in ks}, "warm": {k: 0 for k in ks}}
    totals = {"overall": 0, "cold_start": 0, "warm": 0}

    for row, candidates, slice_name in zip(clicked.itertuples(index=False), all_candidates, user_slices):
        if not candidates:
            continue
        totals["overall"] += 1
        totals[slice_name] += 1
        for k in ks:
            hit = row.article_id in candidates[:k]
            if hit:
                hits["overall"][k] += 1
                hits[slice_name][k] += 1

    result = {"n_overall": totals["overall"], "n_cold_start": totals["cold_start"], "n_warm": totals["warm"]}
    for group in ["overall", "cold_start", "warm"]:
        n = totals[group]
        result[group] = {k: (hits[group][k] / n if n > 0 else None) for k in ks}
    return result


def run_for_dataset(dataset: str, ks: list[int], sample_size: int) -> None:
    print(f"\n=== {dataset.upper()} ===")

    store = FeatureStore("data/splits", dataset)
    print(f"Building BM25 index over {len(store.articles)} articles...")
    bm25_index = BM25Index(store.articles.reset_index())

    val_impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
    print(f"Evaluating on val split ({len(val_impressions)} impressions, "
          f"sampling up to {sample_size} clicked ones)...")

    recall = evaluate_recall_at_k(bm25_index, store, val_impressions, ks, sample_size)
    print(f"  n: overall={recall['n_overall']}, cold_start={recall['n_cold_start']}, warm={recall['n_warm']}")
    for group in ["overall", "cold_start", "warm"]:
        print(f"  [{group}]")
        for k in ks:
            val = recall[group][k]
            print(f"    Recall@{k}: {val:.4f}" if val is not None else f"    Recall@{k}: n/a (no samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=2000)
    args = parser.parse_args()

    ks = [50, 100, 200]
    datasets = ["mind", "ebnerd"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        run_for_dataset(dataset, ks, args.sample_size)