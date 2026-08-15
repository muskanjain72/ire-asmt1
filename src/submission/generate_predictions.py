"""
Q5: Generate prediction.txt for MIND Codabench submission.

Uses embeddings (not BM25) as the scoring method, since Q4's real
evaluation showed embeddings clearly outperform BM25 for exactly this
task on MIND (AUC 0.677 vs 0.568, MRR 0.429 vs 0.299 — see run_eval.py
results). Also far more tractable at this scale: cosine similarity is
cheap; BM25 subset-scoring 2.37M impressions with our current
full-corpus-ranking approach would be impractically slow.

Output format (per MIND's official Codabench spec):
  ImpressionID [rank_of_candidate_1,rank_of_candidate_2,...]
  Ranks are 1-indexed, listed in the ORIGINAL candidate order from
  the test file — not resorted. Rank 1 = highest predicted relevance.

Usage:
    python src/submission/generate_predictions.py --test-limit 1000   # smoke test first!
    python src/submission/generate_predictions.py                     # full run
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path("data/raw/mind/test/MINDlarge_test")
EMBEDDINGS_CACHE = Path("data/raw/mind/test/test_article_embeddings.parquet")
OUTPUT_PATH = Path("outputs/predictions/prediction.txt")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def load_test_articles() -> pd.DataFrame:
    """Parse MINDlarge_test's news.tsv (same schema as MINDsmall's)."""
    cols = ["news_id", "category", "subcategory", "title", "abstract",
            "url", "title_entities", "abstract_entities"]
    df = pd.read_csv(TEST_DIR / "news.tsv", sep="\t", header=None, names=cols,
                      dtype=str, keep_default_na=False)
    return df


def compute_or_load_embeddings(articles: pd.DataFrame) -> dict[str, np.ndarray]:
    if EMBEDDINGS_CACHE.exists():
        print(f"Loading cached test-article embeddings from {EMBEDDINGS_CACHE}")
        df = pd.read_parquet(EMBEDDINGS_CACHE)
        return {row.news_id: np.array(row.embedding, dtype=np.float32) for row in df.itertuples(index=False)}

    from sentence_transformers import SentenceTransformer

    print("Encoding test articles with sentence-transformers "
          "(this is the slow step — ~30 min expected for ~121k articles)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
    vectors = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

    embeddings = {nid: vec.astype(np.float32) for nid, vec in zip(articles["news_id"], vectors)}

    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame({
        "news_id": list(embeddings.keys()),
        "embedding": [v.tolist() for v in embeddings.values()],
    })
    cache_df.to_parquet(EMBEDDINGS_CACHE, index=False)
    print(f"Cached embeddings to {EMBEDDINGS_CACHE}")

    return embeddings


def user_vector(history_ids: list[str], embeddings: dict, fallback: np.ndarray) -> np.ndarray:
    vecs = [embeddings[nid] for nid in history_ids if nid in embeddings]
    if not vecs:
        return fallback  # cold-start / unknown history — use global mean as a neutral guess
    return np.mean(vecs, axis=0)


def rank_candidates(query_vec: np.ndarray, candidate_ids: list[str], embeddings: dict) -> list[int]:
    """Return 1-indexed ranks, one per candidate, IN THE ORIGINAL candidate order."""
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    scores = []
    for cid in candidate_ids:
        if cid in embeddings:
            c = embeddings[cid]
            c_norm = c / (np.linalg.norm(c) + 1e-10)
            scores.append(float(np.dot(q_norm, c_norm)))
        else:
            scores.append(-1e9)  # unknown article — rank it last

    # rank 1 = highest score. argsort ascending on -score gives descending score order.
    order = np.argsort(-np.array(scores))  # order[i] = index of the i-th best candidate
    ranks = np.empty(len(candidate_ids), dtype=int)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-limit", type=int, default=None,
                         help="only process first N impressions — use this for a smoke test before the full run")
    args = parser.parse_args()

    print("Loading test articles...")
    articles = load_test_articles()
    print(f"  {len(articles)} articles")

    embeddings = compute_or_load_embeddings(articles)
    global_mean = np.mean(list(embeddings.values()), axis=0)

    print("Streaming through behaviors.tsv...")
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    behaviors_iter = pd.read_csv(
        TEST_DIR / "behaviors.tsv", sep="\t", header=None, names=cols,
        dtype=str, keep_default_na=False, chunksize=10000,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(OUTPUT_PATH, "w") as f:
        for chunk in behaviors_iter:
            for row in chunk.itertuples(index=False):
                if args.test_limit and n_written >= args.test_limit:
                    print(f"Reached --test-limit={args.test_limit}, stopping early.")
                    print(f"Wrote {n_written} lines to {OUTPUT_PATH}")
                    return

                history_ids = row.history.split() if row.history else []
                candidate_ids = row.impressions.split() if row.impressions else []
                if not candidate_ids:
                    continue  # shouldn't happen, but skip defensively rather than crash

                qvec = user_vector(history_ids, embeddings, global_mean)
                ranks = rank_candidates(qvec, candidate_ids, embeddings)

                rank_str = ",".join(str(r) for r in ranks)
                f.write(f"{row.impression_id} [{rank_str}]\n")
                n_written += 1

                if n_written % 100000 == 0:
                    print(f"  ...{n_written} impressions processed")

    print(f"Done. Wrote {n_written} lines to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
