"""
One-command rebuild: raw files -> unified schema -> temporal splits ->
ready-to-use feature store.

Usage:
    python build_pipeline.py              # full pipeline, all steps
    python build_pipeline.py --skip-download   # reuse existing data/raw/
    python build_pipeline.py --skip-mind       # EB-NeRD only
    python build_pipeline.py --skip-ebnerd     # MIND only

This is Q1 requirement #5 — a single script that rebuilds everything
from raw files. Each stage's own module can still be run standalone
for debugging (as we did throughout development), but this is the
one entrypoint a fresh clone should use.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "data"))

import download as download_mod  # noqa: E402
import parse_mind  # noqa: E402
import parse_ebnerd  # noqa: E402
import split as split_mod  # noqa: E402
import feature_store as fs_mod  # noqa: E402
# linting instruction in python -> module level import not at top of file


def run_download(args) -> None:
    print("\n" + "=" * 60)
    print("STEP 1/4: Download raw data")
    print("=" * 60)
    if args.skip_download:
        print("  --skip-download set, assuming data/raw/ already populated")
        return
    download_mod.download_mind()
    download_mod.download_ebnerd(
        include_small=not args.skip_ebnerd_small,
        include_embeddings=not args.skip_embeddings,
    )


def run_parse(args) -> None:
    print("\n" + "=" * 60)
    print("STEP 2/4: Parse into unified schema")
    print("=" * 60)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_mind:
        print("\n--- MIND ---")
        tables = parse_mind.parse_mind_split("data/raw/mind/MINDsmall_train")
        for name, df in tables.items():
            print(f"  {name}: {len(df)} rows")
            df.to_parquet(out_dir / f"mind_{name}.parquet", index=False)

    if not args.skip_ebnerd:
        print("\n--- EB-NeRD ---")
        for split_name in ["train", "validation"]:
            tables = parse_ebnerd.parse_ebnerd_bundle("data/raw/ebnerd/demo", split_name)
            for name, df in tables.items():
                print(f"  {split_name}/{name}: {len(df)} rows")
                df.to_parquet(out_dir / f"ebnerd_{split_name}_{name}.parquet", index=False)


def run_split(args) -> None:
    print("\n" + "=" * 60)
    print("STEP 3/4: Temporal train/val/test split")
    print("=" * 60)

    processed = "data/processed"
    out = "data/splits"

    if not args.skip_mind:
        split_mod.build_splits_for_dataset(
            processed_dir=processed,
            dataset_prefix="mind",
            articles_files=["mind_articles.parquet"],
            impressions_files=["mind_impressions.parquet"],
            click_history_files=["mind_click_history.parquet"],
            out_dir=out,
        )

    if not args.skip_ebnerd:
        split_mod.build_splits_for_dataset(
            processed_dir=processed,
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
            out_dir=out,
        )


def run_feature_store_check(args) -> None:
    print("\n" + "=" * 60)
    print("STEP 4/4: Feature store sanity check")
    print("=" * 60)

    import pandas as pd

    datasets = []
    if not args.skip_mind:
        datasets.append("mind")
    if not args.skip_ebnerd:
        datasets.append("ebnerd")

    for dataset in datasets:
        store = fs_mod.FeatureStore("data/splits", dataset)
        print(f"[{dataset}] articles={len(store.articles)}, "
              f"users_with_history={len(store._history_by_user)}")

        impressions = pd.read_parquet(f"data/splits/{dataset}/val/impressions.parquet")
        sample = impressions.iloc[0]
        hist = store.get_user_history(sample["user_id"], sample["timestamp"])
        recency = store.get_user_recency_features(sample["user_id"], sample["timestamp"])
        print(f"  sample: safe_history_len={len(hist)}, recency={recency}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true",
                         help="reuse existing data/raw/ instead of re-downloading")
    parser.add_argument("--skip-mind", action="store_true")
    parser.add_argument("--skip-ebnerd", action="store_true")
    parser.add_argument("--skip-ebnerd-small", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    run_download(args)
    run_parse(args)
    run_split(args)
    run_feature_store_check(args)

    print("\n" + "=" * 60)
    print("Pipeline complete. Processed data in data/processed/, "
          "splits + feature store in data/splits/")
    print("=" * 60)


if __name__ == "__main__":
    main()