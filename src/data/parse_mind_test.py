"""
Parse MINDlarge_test raw behaviors.tsv into a test-time impressions
parquet consumable by generate_predictions.py.

Difference from parse_mind.py's train/dev parsing: the official test
set's `impressions` column has NO click labels — each token is a bare
news_id, not "news_id-0/1" (labels are what's being predicted, so
they're withheld). Everything else (column layout, ID prefixing) is
identical to train/dev, so this reuses parse_mind._prefix to guarantee
IDs line up with the feature store / embeddings built from train/dev.

Usage:
    python src/data/parse_mind_test.py data/raw/mind/test/MINDlarge_test data/processed/mind_test_impressions.parquet
"""

""" 
parse_mind.py
    ↓
TRAIN / DEV
    ↓
we KNOW clicked/not-clicked
    ↓
used for retrieval + evaluation


parse_mind_test.py
    ↓
MIND LARGE TEST
    ↓
we DON'T KNOW clicked/not-clicked
    ↓
used to GENERATE FINAL PREDICTIONS
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parse_mind import _prefix  # reuse same prefixing scheme as train/dev


def parse_mind_test_split(split_dir: str | Path) -> pd.DataFrame:
    """
    Returns one row per impression_id:
      impression_id, user_id, timestamp, article_ids_inview (list[str])

    This shape matches what generate_predictions.py looks for when the
    "already grouped" branch runs (article_ids_inview present).
    """
    split_dir = Path(split_dir)
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(
        split_dir / "behaviors.tsv", sep="\t", header=None, names=cols,
        dtype=str, keep_default_na=False,
    )
    df["timestamp"] = pd.to_datetime(df["time"])

    rows = []
    for row in df.itertuples(index=False):
        if not row.impressions:
            continue

        article_ids = []
        for tok in row.impressions.split():
            # Defensive: some redistributed copies of the test set keep a
            # dummy "-0" suffix on every token instead of dropping labels
            # entirely. MIND news_ids never contain "-", so splitting on
            # it is safe either way.
            news_id = tok.split("-")[0] if "-" in tok else tok
            article_ids.append(_prefix(news_id))

        rows.append({
            "impression_id": _prefix(row.impression_id, "mind_imp_"),
            "user_id": _prefix(row.user_id, "mind_user_"),
            "timestamp": row.timestamp,
            "article_ids_inview": article_ids,
        })
        # one impression + the lsit of articles that were shown

    return pd.DataFrame(rows)


if __name__ == "__main__":
    split_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mind/test/MINDlarge_test"
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/mind_test_impressions.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_impressions = parse_mind_test_split(split_dir)
    print(f"Parsed {len(test_impressions)} test impressions from {split_dir}")
    print(test_impressions.head(3))

    avg_candidates = test_impressions["article_ids_inview"].apply(len).mean()
    print(f"Avg candidates per impression: {avg_candidates:.1f}")

    test_impressions.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")