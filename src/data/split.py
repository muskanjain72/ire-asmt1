

"""
Temporal train/val/test split for interaction data.

Never split interaction data randomly — a random split lets future
clicks leak into training. Instead we cut by time: earliest
impressions -> train, a middle window -> val, latest impressions -> test.

IMPORTANT DESIGN NOTE (fixed after a real bug):
We do NOT pre-filter click_history into a separate "clean" table per
split. An earlier version tried to do this with a single per-user
global cutoff and produced 0 valid history rows for train — because
whether a history row is "safe" to use depends on the SPECIFIC
impression being predicted, not a single per-split cutoff per user.

Instead: click_history is kept as one full table per dataset, and the
leakage-safe filtering happens dynamically at query time in
feature_store.py's get_user_history(user_id, before_timestamp). This
is also the architecturally correct place for it — a feature store is
supposed to be queried per-request, not pre-baked per split.
"""

import pandas as pd
from pathlib import Path


def temporal_split(
    impressions: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split an impressions table into train/val/test by time.
    Boundaries are chosen on the TIME RANGE, not row count.
    """
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1

    df = impressions.sort_values(timestamp_col).reset_index(drop=True)

    t_min = df[timestamp_col].min()
    t_max = df[timestamp_col].max()
    total_span = t_max - t_min

    test_cutoff = t_max - total_span * test_frac
    val_cutoff = test_cutoff - total_span * val_frac

    train_df = df[df[timestamp_col] < val_cutoff].reset_index(drop=True)
    val_df = df[(df[timestamp_col] >= val_cutoff) & (df[timestamp_col] < test_cutoff)].reset_index(drop=True)
    test_df = df[df[timestamp_col] >= test_cutoff].reset_index(drop=True)

    return train_df, val_df, test_df


def build_splits_for_dataset(
    processed_dir: str | Path,
    dataset_prefix: str,
    articles_files: list[str],
    impressions_files: list[str],
    click_history_files: list[str],
    out_dir: str | Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> None:
    """
    Load processed parquet files, merge, temporally split the
    IMPRESSIONS only, and save:
      out_dir/<dataset>/articles.parquet            (shared, not split)
      out_dir/<dataset>/click_history_full.parquet  (shared, not split)
      out_dir/<dataset>/{train,val,test}/impressions.parquet
    """
    processed_dir = Path(processed_dir)
    dataset_out = Path(out_dir) / dataset_prefix
    dataset_out.mkdir(parents=True, exist_ok=True)

    articles = pd.concat(
        [pd.read_parquet(processed_dir / f) for f in articles_files]
    ).drop_duplicates(subset="article_id").reset_index(drop=True)

    impressions = pd.concat(
        [pd.read_parquet(processed_dir / f) for f in impressions_files]
    ).drop_duplicates().reset_index(drop=True)

    click_history = pd.concat(
        [pd.read_parquet(processed_dir / f) for f in click_history_files]
    ).drop_duplicates().reset_index(drop=True)

    # articles and full click_history are shared across all splits —
    # filtering by time happens dynamically at query time instead
    articles.to_parquet(dataset_out / "articles.parquet", index=False)
    click_history.to_parquet(dataset_out / "click_history_full.parquet", index=False)

    train_imp, val_imp, test_imp = temporal_split(impressions, val_frac, test_frac)

    print(f"[{dataset_prefix}] impressions -> train={len(train_imp)}, "
          f"val={len(val_imp)}, test={len(test_imp)}")
    print(f"  articles={len(articles)}, click_history_full={len(click_history)}")

    for split_name, split_imp in [("train", train_imp), ("val", val_imp), ("test", test_imp)]:
        split_dir = dataset_out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_imp.to_parquet(split_dir / "impressions.parquet", index=False)


if __name__ == "__main__":
    PROCESSED = "data/processed"
    OUT = "data/splits"

    build_splits_for_dataset(
        processed_dir=PROCESSED,
        dataset_prefix="mind",
        articles_files=["mind_articles.parquet"],
        impressions_files=["mind_impressions.parquet"],
        click_history_files=["mind_click_history.parquet"],
        out_dir=OUT,
    )

    build_splits_for_dataset(
        processed_dir=PROCESSED,
        dataset_prefix="ebnerd",
        articles_files=["ebnerd_train_articles.parquet"],
        impressions_files=[
            "ebnerd_train_impressions.parquet",
            "ebnerd_validation_impressions.parquet",
        ],
        click_history_files=[
            "ebnerd_train_click_history.parquet",
            "ebnerd_validation_click_history.parquet",
        ],
        out_dir=OUT,
    )

    print(f"\nSaved all splits to {OUT}/")