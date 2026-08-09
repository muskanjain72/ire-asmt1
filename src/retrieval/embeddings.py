"""
Q3 (part 1): Load or compute article embeddings.

EB-NeRD: use the provided Word2Vec document embeddings
(data/raw/ebnerd/word2vec/Ekstra_Bladet_word2vec/document_vector.parquet)
— no need to train our own since Ekstra Bladet already published
per-article vectors.

MIND: no embeddings are provided, so we compute our own using a
lightweight sentence-transformers model over title + abstract.

Results are cached to data/splits/<dataset>/article_embeddings.parquet
so this only needs to run once per dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from feature_store import FeatureStore

EBNERD_WORD2VEC_PATH = "data/raw/ebnerd/word2vec/Ekstra_Bladet_word2vec/document_vector.parquet"
MIND_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_ebnerd_embeddings(store: FeatureStore) -> dict[str, np.ndarray]:
    """
    Load Ekstra Bladet's provided Word2Vec article embeddings and
    re-key them with our unified ebnerd_ article_id prefix.
    """
    df = pd.read_parquet(EBNERD_WORD2VEC_PATH)
    embeddings = {}
    for row in df.itertuples(index=False):
        article_id = f"ebnerd_{row.article_id}"
        if article_id in store.articles.index:  # only keep articles we actually have
            embeddings[article_id] = np.array(row.document_vector, dtype=np.float32)
    return embeddings


def compute_mind_embeddings(
    store: FeatureStore,
    model_name: str = MIND_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """
    Compute sentence embeddings for MIND articles from title + abstract,
    since MIND provides no pretrained article embeddings.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    articles = store.articles.reset_index()
    texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()

    print(f"  Encoding {len(texts)} articles (batch_size={batch_size})...")
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )

    return {
        article_id: vec.astype(np.float32)
        for article_id, vec in zip(articles["article_id"], vectors)
    }


def get_or_compute_embeddings(dataset: str, store: FeatureStore, cache_dir: str = "data/splits") -> dict[str, np.ndarray]:
    """
    Load cached embeddings if present, otherwise compute/load and cache.
    """
    cache_path = Path(cache_dir) / dataset / "article_embeddings.parquet"

    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        df = pd.read_parquet(cache_path)
        return {row.article_id: np.array(row.embedding, dtype=np.float32) for row in df.itertuples(index=False)}

    if dataset == "ebnerd":
        embeddings = load_ebnerd_embeddings(store)
    elif dataset == "mind":
        embeddings = compute_mind_embeddings(store)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # cache to disk
    cache_df = pd.DataFrame({
        "article_id": list(embeddings.keys()),
        "embedding": [v.tolist() for v in embeddings.values()],
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(embeddings)} embeddings to {cache_path}")

    return embeddings


if __name__ == "__main__":
    for dataset in ["mind", "ebnerd"]:
        print(f"\n=== {dataset.upper()} ===")
        store = FeatureStore("data/splits", dataset)
        embeddings = get_or_compute_embeddings(dataset, store)
        sample_id = next(iter(embeddings))
        print(f"  {len(embeddings)} article embeddings ready, "
              f"dim={len(embeddings[sample_id])}")