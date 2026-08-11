"""
Q4 (part 2): Beyond-accuracy metrics — diversity, novelty, coverage.

These evaluate properties of the RECOMMENDED LIST itself, not just
whether the clicked item was found. Computed over each impression's
top-K recommended candidates (not the full inview set).
"""

import numpy as np
import pandas as pd


def intra_list_diversity(candidate_ids: list, embeddings: dict) -> float:
    """
    Average pairwise dissimilarity (1 - cosine similarity) among a
    single list's recommended items. Higher = more diverse list.
    Returns None if fewer than 2 candidates have known embeddings.
    """
    vecs = [embeddings[cid] for cid in candidate_ids if cid in embeddings]
    if len(vecs) < 2:
        return None

    matrix = np.stack(vecs)
    norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    sim_matrix = norm @ norm.T

    n = len(vecs)
    upper_tri_idx = np.triu_indices(n, k=1)
    avg_similarity = sim_matrix[upper_tri_idx].mean()
    return float(1 - avg_similarity)


def compute_item_popularity(train_impressions: pd.DataFrame) -> dict:
    """
    Popularity = fraction of training impressions in which an article
    was clicked, out of all impressions it appeared in. Used as the
    basis for novelty (self-information of an item).
    """
    clicks = train_impressions.groupby("article_id")["clicked"].sum()
    shows = train_impressions.groupby("article_id")["clicked"].count()
    popularity = (clicks / shows).to_dict()
    return popularity


def novelty(candidate_ids: list, popularity: dict, total_users: int) -> float:
    """
    Novelty via self-information: -log2(popularity). Rare/unpopular
    items score higher (more "novel"). Unknown items are treated as
    maximally novel (never seen in training).
    """
    scores = []
    for cid in candidate_ids:
        p = popularity.get(cid, 1.0 / max(total_users, 1))  # unseen -> treat as rare
        p = max(p, 1e-6)  # avoid log(0)
        scores.append(-np.log2(p))
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(all_recommended_lists: list, catalog_size: int) -> float:
    """
    Fraction of the full article catalog that appears in AT LEAST ONE
    recommended list across all evaluated impressions/users.
    """
    recommended_union = set()
    for candidates in all_recommended_lists:
        recommended_union.update(candidates)
    return len(recommended_union) / catalog_size if catalog_size > 0 else 0.0


if __name__ == "__main__":
    # Synthetic self-test
    rng = np.random.default_rng(0)
    fake_embeddings = {f"a{i}": rng.random(16) for i in range(100)}

    diverse_list = [f"a{i}" for i in rng.choice(100, 10, replace=False)]
    print("Diversity (random list, expect moderate value):",
          intra_list_diversity(diverse_list, fake_embeddings))

    similar_list = ["a1", "a1", "a1"]  # degenerate — same item repeated
    # (not realistic, but shows the function handles near-identical vectors)
    print("Diversity (same item repeated, expect near 0):",
          intra_list_diversity(similar_list, fake_embeddings))

    fake_popularity = {f"a{i}": rng.random() * 0.1 for i in range(100)}
    print("\nNovelty of a random candidate list:",
          novelty(diverse_list, fake_popularity, total_users=1000))

    all_lists = [[f"a{i}" for i in rng.choice(100, 10, replace=False)] for _ in range(50)]
    print("\nCatalog coverage across 50 lists of 10 (out of 100 items):",
          catalog_coverage(all_lists, catalog_size=100))