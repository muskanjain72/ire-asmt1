"""
Q5: Generate predictions.txt for EB-NeRD (RecSys 2024) Codabench submission.

Uses the provided Word2Vec article embeddings (document_vector.parquet)
— no need to compute new ones, since its 125,541 articles already
matches the test set's article count exactly.

Output format (per EB-NeRD's official Codabench spec):
  ImpressionID [rank_of_candidate_1,rank_of_candidate_2,...]
  Ranks are 1-indexed, listed in the ORIGINAL candidate order — not
  resorted. File must be named exactly `predictions.txt` (plural —
  different from MIND's `prediction.txt`).

Usage:
    python src/submission/generate_predictions_ebnerd.py --test-limit 1000
    python src/submission/generate_predictions_ebnerd.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TESTSET_DIR = Path("data/raw/ebnerd/testset/ebnerd_testset")
WORD2VEC_PATH = Path("data/raw/ebnerd/word2vec/Ekstra_Bladet_word2vec/document_vector.parquet")
OUTPUT_PATH = Path("outputs/predictions/predictions.txt")


def load_embeddings() -> dict:
    df = pd.read_parquet(WORD2VEC_PATH)
    return {row.article_id: np.array(row.document_vector, dtype=np.float32) for row in df.itertuples(index=False)}


def load_history() -> dict:
    """Returns {user_id: [article_id, ...]} — the full clicked-article list
    per user, from history.parquet's article_id_fixed column."""
    df = pd.read_parquet(TESTSET_DIR / "test" / "history.parquet")
    assert "article_id_fixed" in df.columns, (
        f"Expected column 'article_id_fixed' not found. Actual columns: {df.columns.tolist()}"
    )
    return {row.user_id: (row.article_id_fixed if row.article_id_fixed is not None else [])
            for row in df.itertuples(index=False)}


def user_vector(article_ids: list, embeddings: dict, fallback: np.ndarray) -> np.ndarray:
    vecs = [embeddings[aid] for aid in article_ids if aid in embeddings]
    if not vecs:
        return fallback
    return np.mean(vecs, axis=0)


def rank_candidates(query_vec: np.ndarray, candidate_ids: list, embeddings: dict) -> list:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    scores = []
    for cid in candidate_ids:
        if cid in embeddings:
            c = embeddings[cid]
            c_norm = c / (np.linalg.norm(c) + 1e-10)
            scores.append(float(np.dot(q_norm, c_norm)))
        else:
            scores.append(-1e9)

    order = np.argsort(-np.array(scores))
    ranks = np.empty(len(candidate_ids), dtype=int)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args()

    print("Loading Word2Vec article embeddings...")
    embeddings = load_embeddings()
    print(f"  {len(embeddings)} article embeddings")
    global_mean = np.mean(list(embeddings.values()), axis=0)

    print("Loading user history...")
    history = load_history()
    print(f"  {len(history)} users with history")

    behaviors_path = TESTSET_DIR / "test" / "behaviors.parquet"

    # Cheap metadata-only schema check — does NOT load row data into memory
    import pyarrow.parquet as pq
    schema_names = pq.ParquetFile(behaviors_path).schema.names
    print(f"  behaviors schema (raw, list columns show as nested names): {schema_names}")

    candidate_col = "article_ids_inview"

    print("Streaming through behaviors.parquet in batches (low-memory mode)...")
    parquet_file = pq.ParquetFile(behaviors_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    schema_confirmed = False

    with open(OUTPUT_PATH, "w") as f:
        for batch in parquet_file.iter_batches(batch_size=5000):
            chunk = batch.to_pandas()

            if not schema_confirmed:
                assert candidate_col in chunk.columns, (
                    f"Expected column '{candidate_col}' not found. "
                    f"Actual columns: {chunk.columns.tolist()}. "
                    f"Update generate_predictions_ebnerd.py's candidate_col variable to match."
                )
                schema_confirmed = True

            for row in chunk.itertuples(index=False):
                if args.test_limit and n_written >= args.test_limit:
                    print(f"Reached --test-limit={args.test_limit}, stopping early.")
                    print(f"Wrote {n_written} lines to {OUTPUT_PATH}")
                    return

                candidate_ids = getattr(row, candidate_col)
                if candidate_ids is None or len(candidate_ids) == 0:
                    continue

                hist_ids = history.get(row.user_id, [])
                qvec = user_vector(hist_ids, embeddings, global_mean)
                ranks = rank_candidates(qvec, list(candidate_ids), embeddings)

                rank_str = ",".join(str(r) for r in ranks)
                f.write(f"{row.impression_id} [{rank_str}]\n")
                n_written += 1

                if n_written % 500000 == 0:
                    print(f"  ...{n_written} impressions processed")

    print(f"Done. Wrote {n_written} lines to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
