"""
Parse raw EB-NeRD dataset files into the unified schema used across
this project: articles, impressions, click_history.

Raw EB-NeRD files:
  articles.parquet   -> article_id, title, subtitle, body, category,
                         subcategory, category_str, published_time, ...
                         (one file, shared across train/validation)
  behaviors.parquet  -> impression_id, article_id, impression_time,
                         article_ids_inview, article_ids_clicked,
                         user_id, ... (per split: train/ or validation/)
  history.parquet    -> user_id, article_id_fixed, impression_time_fixed,
                         scroll_percentage_fixed, read_time_fixed
                         (list-valued columns, one row per user; per split)

Unlike MIND's packed strings ("N123-1 N456-0"), EB-NeRD already
separates shown vs. clicked articles into two list columns
(article_ids_inview, article_ids_clicked), which makes impressions
parsing simpler here.

history.parquet stores parallel list columns per user
(article_id_fixed[i] happened at impression_time_fixed[i]) — these
must be zipped together, not just exploded independently.
"""

import pandas as pd
from pathlib import Path


def _prefix(id_, prefix: str = "ebnerd_") -> str:
    """Namespace an ID so it can't collide with MIND IDs."""
    return f"{prefix}{id_}"


def load_articles(articles_parquet_path: str | Path) -> pd.DataFrame:
    """
    Parse articles.parquet into the unified `articles` schema.

    Returns columns:
      article_id, dataset, title, abstract, body, category,
      subcategory, published_time
    """
    df = pd.read_parquet(articles_parquet_path)

    articles = pd.DataFrame({
        "article_id": df["article_id"].astype(str).apply(_prefix),
        "dataset": "ebnerd",
        "title": df["title"].fillna(""),
        "abstract": df["subtitle"].fillna(""),  # EB-NeRD's subtitle ~ MIND's abstract
        "body": df["body"].fillna(""),
        "category": df["category_str"].fillna(""),  # human-readable, matches MIND's string category
        "subcategory": df.get("subcategory", pd.Series([""] * len(df))).fillna(""),
        "published_time": pd.to_datetime(df["published_time"], errors="coerce"),
    })
    return articles.drop_duplicates(subset="article_id").reset_index(drop=True)


def load_impressions(behaviors_parquet_path: str | Path) -> pd.DataFrame:
    """
    Parse behaviors.parquet into the unified `impressions` schema.
    One output row per (impression, article) pair.

    Returns columns:
      impression_id, user_id, timestamp, article_id, clicked
    """
    df = pd.read_parquet(behaviors_parquet_path)
    df["timestamp"] = pd.to_datetime(df["impression_time"], errors="coerce")

    rows = []
    for row in df.itertuples(index=False):
        inview = row.article_ids_inview if row.article_ids_inview is not None else []
        clicked_set = set(row.article_ids_clicked) if row.article_ids_clicked is not None else set()

        for article_id in inview:
            rows.append({
                "impression_id": _prefix(row.impression_id, "ebnerd_imp_"),
                "user_id": _prefix(row.user_id, "ebnerd_user_"),
                "timestamp": row.timestamp,
                "article_id": _prefix(article_id),
                "clicked": int(article_id in clicked_set),
            })

    return pd.DataFrame(rows)


def load_click_history(history_parquet_path: str | Path) -> pd.DataFrame:
    """
    Parse history.parquet into the unified `click_history` schema.

    article_id_fixed and impression_time_fixed are PARALLEL list
    columns per user — article_id_fixed[i] was clicked at
    impression_time_fixed[i]. We zip them together per row.

    Unlike MIND, EB-NeRD gives real per-click timestamps here, so
    this table is more precise than MIND's history (which only has
    an upper-bound timestamp per click).

    Returns columns:
      user_id, article_id, timestamp
    """
    df = pd.read_parquet(history_parquet_path)

    rows = []
    for row in df.itertuples(index=False):
        user_id = _prefix(row.user_id, "ebnerd_user_")
        article_ids = row.article_id_fixed if row.article_id_fixed is not None else []
        timestamps = row.impression_time_fixed if row.impression_time_fixed is not None else []

        # defensive: lists should be same length, but don't crash if not
        n = min(len(article_ids), len(timestamps))
        for i in range(n):
            rows.append({
                "user_id": user_id,
                "article_id": _prefix(article_ids[i]),
                "timestamp": pd.to_datetime(timestamps[i], errors="coerce"),
            })

    history = pd.DataFrame(rows)
    return history.drop_duplicates(subset=["user_id", "article_id", "timestamp"]).reset_index(drop=True)


def parse_ebnerd_bundle(bundle_dir: str | Path, split: str = "train") -> dict[str, pd.DataFrame]:
    """
    Parse one EB-NeRD bundle end to end.

    bundle_dir should contain articles.parquet at the top level and
    a `split` subfolder (e.g. "train" or "validation") containing
    behaviors.parquet and history.parquet.
    """
    bundle_dir = Path(bundle_dir)
    articles = load_articles(bundle_dir / "articles.parquet")
    impressions = load_impressions(bundle_dir / split / "behaviors.parquet")
    click_history = load_click_history(bundle_dir / split / "history.parquet")
    return {
        "articles": articles,
        "impressions": impressions,
        "click_history": click_history,
    }


if __name__ == "__main__":
    import sys

    # Usage: python parse_ebnerd.py data/raw/ebnerd/demo train data/processed
    bundle_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ebnerd/demo"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = parse_ebnerd_bundle(bundle_dir, split)
    for name, df in tables.items():
        print(f"{name}: {len(df)} rows")
        print(df.head(3))
        print()
        df.to_parquet(out_dir / f"ebnerd_{split}_{name}.parquet", index=False)

    print(f"Saved parsed tables to {out_dir}/")