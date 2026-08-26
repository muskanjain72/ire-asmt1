# """
# Feature store: a thin lookup layer over the processed parquet files.

# Rather than baking a "safe" click history into static per-split
# files (which turned out to be the wrong design — see split.py's
# docstring), this module loads the full articles + click_history
# tables once per dataset and answers point queries:
#   - get_article_features(article_id)
#   - get_user_history(user_id, before_timestamp)  <- the leakage-safe one

# get_user_history is the single place that enforces "no future click":
# callers MUST pass the timestamp of the impression they're predicting,
# and only history strictly before that timestamp is returned. This is
# also exactly what tests/test_no_leakage.py checks against.
# """

# import pandas as pd
# from pathlib import Path


# class FeatureStore:
#     def __init__(self, splits_dir: str | Path, dataset: str):
#         """
#         dataset: "mind" or "ebnerd"
#         Expects splits_dir/<dataset>/articles.parquet and
#         splits_dir/<dataset>/click_history_full.parquet to exist
#         (produced by split.py).
#         """
#         base = Path(splits_dir) / dataset

#         self.articles = pd.read_parquet(base / "articles.parquet").set_index("article_id")
#         self.click_history = pd.read_parquet(base / "click_history_full.parquet")

#         # index by user for fast repeated lookups
#         self.click_history = self.click_history.sort_values("timestamp")
#         self._history_by_user = {
#             user_id: group[["article_id", "timestamp"]].reset_index(drop=True)
#             for user_id, group in self.click_history.groupby("user_id")
#         }

#     def get_article_features(self, article_id: str) -> dict:
#         """Return a dict of an article's features, or {} if unknown."""
#         if article_id not in self.articles.index:
#             return {}
#         return self.articles.loc[article_id].to_dict()

#     def get_user_history(self, user_id: str, before_timestamp) -> list[str]:
#         """
#         Return the list of article_ids this user clicked strictly
#         before `before_timestamp`. This is the leakage boundary —
#         always call this with the timestamp of the impression you
#         are about to predict, never with "now" or a split-wide cutoff.
#         """
#         if user_id not in self._history_by_user:
#             return []
#         user_hist = self._history_by_user[user_id]
#         safe = user_hist[user_hist["timestamp"] < before_timestamp]
#         return safe["article_id"].tolist()

#     def user_history_text(self, user_id: str, before_timestamp) -> str:
#         """
#         Convenience for BM25 (Q2): concatenate the titles of a user's
#         safe click history into a single query string.
#         """
#         article_ids = self.get_user_history(user_id, before_timestamp)
#         titles = [
#             self.articles.loc[aid, "title"]
#             for aid in article_ids
#             if aid in self.articles.index
#         ]
#         return " ".join(titles)


# if __name__ == "__main__":
#     # smoke test
#     for dataset in ["mind", "ebnerd"]:
#         fs = FeatureStore("data/splits", dataset)
#         print(f"[{dataset}] articles={len(fs.articles)}, "
#               f"users_with_history={len(fs._history_by_user)}")

#         impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
#         sample = impressions.iloc[0]
#         hist = fs.get_user_history(sample["user_id"], sample["timestamp"])
#         print(f"  sample user={sample['user_id']}, "
#               f"impression_time={sample['timestamp']}, "
#               f"safe_history_len={len(hist)}")


#initial one commented out

"""
Feature store: a thin lookup layer over the processed parquet files.

Rather than baking a "safe" click history into static per-split
files (which turned out to be the wrong design — see split.py's
docstring), this module loads the full articles + click_history
tables once per dataset and answers point queries:
  - get_article_features(article_id)
  - get_user_history(user_id, before_timestamp)  <- the leakage-safe one

get_user_history is the single place that enforces "no future click":
callers MUST pass the timestamp of the impression they're predicting,
and only history strictly before that timestamp is returned. This is
also exactly what tests/test_no_leakage.py checks against.
"""

import pandas as pd
from pathlib import Path


class FeatureStore:
    def __init__(self, splits_dir: str | Path, dataset: str):
        """
        dataset: "mind" or "ebnerd"
        Expects splits_dir/<dataset>/articles.parquet and
        splits_dir/<dataset>/click_history_full.parquet to exist
        (produced by split.py).
        """
        base = Path(splits_dir) / dataset

        self.articles = pd.read_parquet(base / "articles.parquet").set_index("article_id")
        self.click_history = pd.read_parquet(base / "click_history_full.parquet")

        # index by user for fast repeated lookups
        self.click_history = self.click_history.sort_values("timestamp")
        self._history_by_user = {
            user_id: group[["article_id", "timestamp"]].reset_index(drop=True)
            for user_id, group in self.click_history.groupby("user_id")
        }

    def get_article_features(self, article_id: str) -> dict:
        """Return a dict of an article's features, or {} if unknown."""
        if article_id not in self.articles.index:
            return {}
        return self.articles.loc[article_id].to_dict()

    def get_user_history(self, user_id: str, before_timestamp) -> list[str]:
        """
        Return the list of article_ids this user clicked strictly
        before `before_timestamp`. This is the leakage boundary —
        always call this with the timestamp of the impression you
        are about to predict, never with "now" or a split-wide cutoff.
        """
        if user_id not in self._history_by_user:
            return []
        user_hist = self._history_by_user[user_id]
        safe = user_hist[user_hist["timestamp"] < before_timestamp]
        return safe["article_id"].tolist()

    def user_history_text(self, user_id: str, before_timestamp) -> str:
        """
        Convenience for BM25 (Q2): concatenate the titles of a user's
        safe click history into a single query string.
        """
        article_ids = self.get_user_history(user_id, before_timestamp)
        titles = [
            self.articles.loc[aid, "title"]
            for aid in article_ids
            if aid in self.articles.index
        ]
        return " ".join(titles)

    def get_user_recency_features(self, user_id: str, before_timestamp) -> dict:
        """
        Compute simple recency/activity features for a user, using
        only history strictly before `before_timestamp` (same leakage
        boundary as get_user_history).

        Returns:
          click_count_total: total safe historical clicks
          click_count_last_1d / last_7d / last_30d: activity in recent windows
          hours_since_last_click: recency of most recent safe click
                                   (None if the user has no safe history)
        """
        if user_id not in self._history_by_user:
            return {
                "click_count_total": 0,
                "click_count_last_1d": 0,
                "click_count_last_7d": 0,
                "click_count_last_30d": 0,
                "hours_since_last_click": None,
            }

        user_hist = self._history_by_user[user_id]
        safe = user_hist[user_hist["timestamp"] < before_timestamp]

        if len(safe) == 0:
            return {
                "click_count_total": 0,
                "click_count_last_1d": 0,
                "click_count_last_7d": 0,
                "click_count_last_30d": 0,
                "hours_since_last_click": None,
            }

        deltas = before_timestamp - safe["timestamp"]

        last_click_delta = deltas.min()
        hours_since_last_click = last_click_delta.total_seconds() / 3600.0

        return {
            "click_count_total": int(len(safe)),
            "click_count_last_1d": int((deltas <= pd.Timedelta(days=1)).sum()),
            "click_count_last_7d": int((deltas <= pd.Timedelta(days=7)).sum()),
            "click_count_last_30d": int((deltas <= pd.Timedelta(days=30)).sum()),
            "hours_since_last_click": hours_since_last_click,
        }


if __name__ == "__main__":
    # smoke test
    for dataset in ["mind", "ebnerd"]:
        fs = FeatureStore("data/splits", dataset)
        print(f"[{dataset}] articles={len(fs.articles)}, "
              f"users_with_history={len(fs._history_by_user)}")

        impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
        sample = impressions.iloc[0]
        hist = fs.get_user_history(sample["user_id"], sample["timestamp"])
        recency = fs.get_user_recency_features(sample["user_id"], sample["timestamp"])
        print(f"  sample user={sample['user_id']}, "
              f"impression_time={sample['timestamp']}, "
              f"safe_history_len={len(hist)}")
        print(f"  recency features: {recency}")