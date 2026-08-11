"""
Q4 (part 3): Slicing — cold-start vs warm users, head vs tail articles.

Given an impressions dataframe and a FeatureStore, tags each
impression with a slice label so metrics can be computed separately
per slice (e.g. "does BM25 do better for warm users than cold-start
users?").
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from feature_store import FeatureStore


def tag_cold_start_vs_warm(
    impressions: pd.DataFrame,
    feature_store: FeatureStore,
    cold_start_threshold: int = 5,
) -> pd.DataFrame:
    """
    Adds a `user_slice` column: "cold_start" if the user has fewer
    than `cold_start_threshold` safe historical clicks at the time of
    this impression, else "warm".
    """
    df = impressions.copy()

    def _slice_for_row(row):
        n_clicks = feature_store.get_user_recency_features(
            row["user_id"], row["timestamp"]
        )["click_count_total"]
        return "cold_start" if n_clicks < cold_start_threshold else "warm"

    df["user_slice"] = df.apply(_slice_for_row, axis=1)
    return df


def tag_head_vs_tail(
    impressions: pd.DataFrame,
    train_impressions: pd.DataFrame,
    head_percentile: float = 0.8,
) -> pd.DataFrame:
    """
    Adds an `article_slice` column: "head" if the article's popularity
    (total clicks in train_impressions) is at or above the given
    percentile, else "tail". Popularity is computed from TRAIN data
    only, to avoid leaking val/test-period popularity into the slice
    definition.
    """
    df = impressions.copy()

    article_clicks = train_impressions.groupby("article_id")["clicked"].sum()
    threshold = article_clicks.quantile(head_percentile)

    def _slice_for_article(article_id):
        clicks = article_clicks.get(article_id, 0)
        return "head" if clicks >= threshold else "tail"

    df["article_slice"] = df["article_id"].apply(_slice_for_article)
    return df


def summarize_slice_sizes(df: pd.DataFrame, slice_col: str) -> dict:
    """Quick sanity check: how many impressions fall in each slice."""
    return df[slice_col].value_counts().to_dict()


if __name__ == "__main__":
    for dataset in ["mind", "ebnerd"]:
        print(f"\n=== {dataset.upper()} ===")
        store = FeatureStore("data/splits", dataset)

        val_impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
        train_impressions = pd.read_parquet(f"data/splits/{dataset}/train/impressions.parquet")

        # sample for speed — full val set can be large
        sample = val_impressions.sample(min(500, len(val_impressions)), random_state=42)

        tagged = tag_cold_start_vs_warm(sample, store)
        print("User slice sizes:", summarize_slice_sizes(tagged, "user_slice"))

        tagged2 = tag_head_vs_tail(sample, train_impressions)
        print("Article slice sizes:", summarize_slice_sizes(tagged2, "article_slice"))