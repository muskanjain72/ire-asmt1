"""
Parse raw MIND dataset files into the unified schema used across
this project: articles, impressions, click_history.

Raw MIND files (per split, e.g. train/ or dev/):
  news.tsv       -> news_id, category, subcategory, title, abstract,
                     url, title_entities, abstract_entities
  behaviors.tsv  -> impression_id, user_id, time, history, impressions

`impressions` column in behaviors.tsv looks like:
  "N12345-1 N67890-0 N11111-0"
  where each token is "<news_id>-<clicked 0/1>"

`history` column looks like a space-separated list of news_ids the
user clicked before this impression, e.g. "N222 N333 N444".
MIND does not give per-item timestamps for history, so we upper-bound
each history click's timestamp with the impression's own timestamp.
"""

import pandas as pd  
# for working with tabular data
from pathlib import Path


def _prefix(id_: str, prefix: str = "mind_") -> str:
    """Namespace an ID so it can't collide with EB-NeRD IDs."""
    # mind is the default option ig 
    return f"{prefix}{id_}"


def load_articles(news_tsv_path: str | Path) -> pd.DataFrame:
    """
    Parse news.tsv into the unified `articles` schema.

    Returns columns:
      article_id, dataset, title, abstract, body, category,
      subcategory, published_time
    """
    cols = [
        "news_id", "category", "subcategory", "title", "abstract",
        "url", "title_entities", "abstract_entities",
    ]
    df = pd.read_csv(news_tsv_path, sep="\t", header=None, names=cols,
                      dtype=str, keep_default_na=False)
    # reading news.tsv, dtype=str -> read the valus as str

    articles = pd.DataFrame({
        "article_id": df["news_id"].apply(_prefix),
        "dataset": "mind",
        "title": df["title"].fillna(""),
        "abstract": df["abstract"].fillna(""),
        "body": "",  # MIND does not provide body text
        "category": df["category"].fillna(""),
        "subcategory": df["subcategory"].fillna(""),
        "published_time": pd.NaT,  # MIND does not provide this -> not a time
    })
    return articles.drop_duplicates(subset="article_id").reset_index(drop=True)


def load_impressions(behaviors_tsv_path: str | Path) -> pd.DataFrame:
    """
    Parse behaviors.tsv into the unified `impressions` schema.
    One output row per (impression, article) pair.

    Returns columns:
      impression_id, user_id, timestamp, article_id, clicked
    """
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(behaviors_tsv_path, sep="\t", header=None, names=cols,
                      dtype=str, keep_default_na=False)

    df["timestamp"] = pd.to_datetime(df["time"])  
    # convert time to pandas date-time format for comparison 

    rows = []
    # loop through each bhavior record
    for row in df.itertuples(index=False): 
        if not row.impressions:
            continue
        # split the impression string and process each article individually
        for token in row.impressions.split():
            news_id, clicked = token.rsplit("-", 1) 
            # "N123-1" -> separate newsid and clicked
            # rsplit splits form right
            rows.append({
                "impression_id": _prefix(row.impression_id, "mind_imp_"),
                "user_id": _prefix(row.user_id, "mind_user_"),
                "timestamp": row.timestamp,
                "article_id": _prefix(news_id),
                "clicked": int(clicked),
            })

    return pd.DataFrame(rows)


def load_click_history(behaviors_tsv_path: str | Path) -> pd.DataFrame:
    """
    Parse the `history` column of behaviors.tsv into the unified
    `click_history` schema.

    MIND does not timestamp individual history clicks, so each
    history row is stamped with the impression's own timestamp as
    an upper bound (i.e. "this click happened at or before this
    impression"). This is a known limitation — flag it in the
    design note.

    Returns columns:
      user_id, article_id, timestamp
    """
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(behaviors_tsv_path, sep="\t", header=None, names=cols,
                      dtype=str, keep_default_na=False)
    df["timestamp"] = pd.to_datetime(df["time"])

    rows = []
    for row in df.itertuples(index=False):
        if not row.history:
            continue
        user_id = _prefix(row.user_id, "mind_user_")
        for news_id in row.history.split():
            # loop through history
            rows.append({
                "user_id": user_id,
                "article_id": _prefix(news_id),
                "timestamp": row.timestamp,
            })

    history = pd.DataFrame(rows)
    # for each user, keep only one occurrence of an article
    return history.drop_duplicates(subset=["user_id", "article_id"]).reset_index(drop=True)


def parse_mind_split(split_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Parse one MIND split (train/ or dev/) end to end.

    split_dir should contain news.tsv and behaviors.tsv.
    """
    split_dir = Path(split_dir)
    articles = load_articles(split_dir / "news.tsv")
    impressions = load_impressions(split_dir / "behaviors.tsv")
    click_history = load_click_history(split_dir / "behaviors.tsv")
    return {
        "articles": articles,
        "impressions": impressions,
        "click_history": click_history,
    }


if __name__ == "__main__":
    import sys

    # Usage: python parse_mind.py data/raw/mind/train data/processed/mind_train
    split_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mind/train"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed")
    # use the second argument as output directory , otherwise data/processed
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = parse_mind_split(split_dir)
    for name, df in tables.items():
        print(f"{name}: {len(df)} rows")
        print(df.head(3))
        # print first 3 lines for debugging
        print()
        df.to_parquet(out_dir / f"mind_{name}.parquet", index=False)
        # save as paraquet and dont save the pandas row-number index as an extra coloumn

    print(f"Saved parsed tables to {out_dir}/")