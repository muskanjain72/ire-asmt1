"""
Q9: Anti-gaming — enforce the behaviour-window boundary.

Asserts that get_user_history() and get_user_recency_features() never
return a click that happened at or after the impression timestamp
they were queried with. This is the core leakage invariant the whole
project depends on: retrieval queries (BM25 history text, embedding
user vectors) and feature-store lookups are only ever built from
history strictly before the impression being predicted.

Run with: pytest tests/test_no_leakage.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))
from feature_store import FeatureStore


DATASETS = ["mind", "ebnerd"]
SAMPLE_SIZE = 300


@pytest.mark.parametrize("dataset", DATASETS)
def test_get_user_history_has_no_future_clicks(dataset):
    """For a sample of real impressions, every article returned by
    get_user_history must have been clicked strictly before that
    impression's own timestamp."""
    store = FeatureStore("data/splits", dataset)
    impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")

    sample = impressions.sample(min(SAMPLE_SIZE, len(impressions)), random_state=42)

    violations = []
    n_checked = 0

    for row in sample.itertuples(index=False):
        history_ids = store.get_user_history(row.user_id, row.timestamp)
        if not history_ids:
            continue

        # look up the raw click_history table directly to get each
        # returned article's actual click timestamp for this user
        user_hist = store._history_by_user.get(row.user_id)
        if user_hist is None:
            continue

        matched = user_hist[user_hist["article_id"].isin(history_ids)]
        n_checked += 1

        future_clicks = matched[matched["timestamp"] >= row.timestamp]
        if len(future_clicks) > 0:
            violations.append({
                "user_id": row.user_id,
                "impression_timestamp": row.timestamp,
                "leaked_rows": future_clicks.to_dict("records"),
            })

    assert n_checked > 0, (
        f"No impressions in the {dataset} val sample had any history — "
        f"test is not actually exercising the leakage check. Investigate sampling."
    )
    assert not violations, (
        f"LEAKAGE DETECTED in {dataset}: {len(violations)} impression(s) had at least one "
        f"history article with a click timestamp >= the impression's own timestamp. "
        f"First violation: {violations[0]}"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_get_user_recency_features_consistent_with_history(dataset):
    """click_count_total from get_user_recency_features must exactly
    match the length of get_user_history's returned list — both must
    apply the identical leakage boundary."""
    store = FeatureStore("data/splits", dataset)
    impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")

    sample = impressions.sample(min(SAMPLE_SIZE, len(impressions)), random_state=42)

    for row in sample.itertuples(index=False):
        history_ids = store.get_user_history(row.user_id, row.timestamp)
        recency = store.get_user_recency_features(row.user_id, row.timestamp)

        assert recency["click_count_total"] == len(history_ids), (
            f"Mismatch for user={row.user_id}, impression_time={row.timestamp}: "
            f"get_user_history returned {len(history_ids)} articles but "
            f"get_user_recency_features counted {recency['click_count_total']}. "
            f"Both must apply the same leakage boundary."
        )


@pytest.mark.parametrize("dataset", DATASETS)
def test_synthetic_future_click_is_excluded(dataset):
    """Direct unit test with a synthetic case: manually construct a
    history entry timestamped AFTER a fake impression, and confirm
    get_user_history correctly excludes it."""
    store = FeatureStore("data/splits", dataset)

    # grab a real user who has at least one history row, so we can
    # test against their real earliest click as an anchor point
    if not store._history_by_user:
        pytest.skip(f"No users with history in {dataset} — cannot run synthetic test")

    user_id = next(iter(store._history_by_user))
    user_hist = store._history_by_user[user_id]

    earliest_click_time = user_hist["timestamp"].min()
    latest_click_time = user_hist["timestamp"].max()

    # query with a timestamp BEFORE this user's earliest click —
    # should return no history at all
    before_everything = earliest_click_time - pd.Timedelta(days=1)
    result = store.get_user_history(user_id, before_everything)
    assert result == [], (
        f"Expected no history when querying before this user's earliest click, got {result}"
    )

    # query with a timestamp AFTER their latest click — should return
    # their full history (nothing excluded)
    after_everything = latest_click_time + pd.Timedelta(days=1)
    result = store.get_user_history(user_id, after_everything)
    assert len(result) == len(user_hist), (
        f"Expected all {len(user_hist)} history rows when querying after the latest click, "
        f"got {len(result)}"
    )