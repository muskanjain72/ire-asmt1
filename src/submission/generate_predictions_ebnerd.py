"""
Q5: Generate predictions.txt for EB-NeRD (RecSys 2024) Codabench submission.

Uses the provided Word2Vec article embeddings (document_vector.parquet)
-- no need to compute new ones, since its 125,541 articles already
matches the test set's article count exactly.

Paths default to this repo's local layout, but can be overridden via
CLI flags. The large parquet inputs are streamed in batches; the script
keeps only the lookup tables required for within-impression ranking.

Output format (per EB-NeRD's official Codabench spec):
  ImpressionID [rank_of_candidate_1,rank_of_candidate_2,...]
  Ranks are 1-indexed, listed in the ORIGINAL candidate order -- not
  resorted. File must be named exactly `predictions.txt` (plural --
  different from MIND's `prediction.txt`).

Usage:
    # local (default paths, matches this repo's data/raw/ layout)
    python src/submission/generate_predictions_ebnerd.py --test-limit 1000
    python src/submission/generate_predictions_ebnerd.py

    # Colab / any environment with a different layout
    python generate_predictions_ebnerd.py \
        --testset-dir testset/ebnerd_testset \
        --word2vec-path word2vec/Ekstra_Bladet_word2vec/document_vector.parquet \
        --output outputs/predictions/predictions.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_TESTSET_DIR = Path("data/raw/ebnerd/testset/ebnerd_testset")
DEFAULT_WORD2VEC_PATH = Path("data/raw/ebnerd/word2vec/Ekstra_Bladet_word2vec/document_vector.parquet")
DEFAULT_OUTPUT_PATH = Path("outputs/predictions/predictions.txt")
DEFAULT_BATCH_SIZE = 5000
EPSILON = 1e-10
MISSING_SCORE = -1e9


def as_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + EPSILON)


def require_columns(parquet_file: pq.ParquetFile, path: Path, columns: set[str]) -> None:
    missing = columns.difference(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(
            f"{path} is missing expected column(s): {sorted(missing)}. "
            f"Actual columns: {parquet_file.schema_arrow.names}"
        )


def load_embeddings(word2vec_path: Path, batch_size: int) -> tuple[dict, np.ndarray]:
    """Stream article vectors from parquet, keeping the normalized lookup table needed for scoring."""
    parquet_file = pq.ParquetFile(word2vec_path)
    require_columns(parquet_file, word2vec_path, {"article_id", "document_vector"})

    embeddings = {}
    vector_sum = None
    n_vectors = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["article_id", "document_vector"],
    ):
        chunk = batch.to_pandas()
        for row in chunk.itertuples(index=False):
            vector = np.asarray(row.document_vector, dtype=np.float32)
            embeddings[row.article_id] = normalize(vector)
            vector_sum = vector.copy() if vector_sum is None else vector_sum + vector
            n_vectors += 1

    if n_vectors == 0:
        raise ValueError(f"No vectors found in {word2vec_path}")

    return embeddings, normalize(vector_sum / n_vectors)


def load_user_vectors(
    testset_dir: Path,
    embeddings: dict,
    batch_size: int,
) -> dict:
    """Stream history.parquet and keep one normalized profile vector per user."""
    history_path = testset_dir / "test" / "history.parquet"
    parquet_file = pq.ParquetFile(history_path)
    require_columns(parquet_file, history_path, {"user_id", "article_id_fixed"})

    user_vectors = {}
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["user_id", "article_id_fixed"],
    ):
        chunk = batch.to_pandas()
        for row in chunk.itertuples(index=False):
            vectors = [embeddings[aid] for aid in as_list(row.article_id_fixed) if aid in embeddings]
            if vectors:
                user_vectors[row.user_id] = normalize(np.mean(vectors, axis=0))

    return user_vectors


def rank_candidates(query_vec: np.ndarray, candidate_ids: list, embeddings: dict) -> list[int]:
    scores = np.full(len(candidate_ids), MISSING_SCORE, dtype=np.float32)
    for idx, cid in enumerate(candidate_ids):
        if cid in embeddings:
            scores[idx] = np.dot(query_vec, embeddings[cid])

    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(candidate_ids), dtype=int)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--testset-dir", type=Path, default=DEFAULT_TESTSET_DIR,
                        help="path to the unzipped ebnerd_testset folder")
    parser.add_argument("--word2vec-path", type=Path, default=DEFAULT_WORD2VEC_PATH,
                        help="path to document_vector.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="output path for predictions.txt")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="parquet rows to process per streaming batch")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    print("Streaming Word2Vec article embeddings...")
    embeddings, global_mean = load_embeddings(args.word2vec_path, args.batch_size)
    print(f"  {len(embeddings)} article embeddings")

    print("Streaming user history into compact user vectors...")
    user_vectors = load_user_vectors(args.testset_dir, embeddings, args.batch_size)
    print(f"  {len(user_vectors)} users with usable history vectors")

    behaviors_path = args.testset_dir / "test" / "behaviors.parquet"
    candidate_col = "article_ids_inview"
    parquet_file = pq.ParquetFile(behaviors_path)
    require_columns(parquet_file, behaviors_path, {"impression_id", "user_id", candidate_col})
    print(f"  behaviors schema: {parquet_file.schema_arrow.names}")

    print("Streaming through behaviors.parquet in batches (low-memory mode)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(args.output, "w") as f:
        for batch in parquet_file.iter_batches(
            batch_size=args.batch_size,
            columns=["impression_id", "user_id", candidate_col],
        ):
            chunk = batch.to_pandas()

            for row in chunk.itertuples(index=False):
                if args.test_limit is not None and n_written >= args.test_limit:
                    print(f"Reached --test-limit={args.test_limit}, stopping early.")
                    print(f"Wrote {n_written} lines to {args.output}")
                    return

                candidate_ids = as_list(getattr(row, candidate_col))
                if not candidate_ids:
                    continue

                qvec = user_vectors.get(row.user_id, global_mean)
                ranks = rank_candidates(qvec, candidate_ids, embeddings)

                rank_str = ",".join(str(r) for r in ranks)
                f.write(f"{row.impression_id} [{rank_str}]\n")
                n_written += 1

                if n_written % 500000 == 0:
                    print(f"  ...{n_written} impressions processed")

    print(f"Done. Wrote {n_written} lines to {args.output}")


if __name__ == "__main__":
    main()
