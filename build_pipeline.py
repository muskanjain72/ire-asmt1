"""
One-command rebuild: raw files -> unified schema -> temporal splits ->
ready-to-use feature store.

Usage:
    python build_pipeline.py              # full pipeline, all steps
    python build_pipeline.py --skip-download   # reuse existing data/raw/
    python build_pipeline.py --skip-mind       # EB-NeRD only
    python build_pipeline.py --skip-ebnerd     # MIND only
    python build_pipeline.py --include-mind-testset      # opt into MINDlarge_test
    python build_pipeline.py --include-ebnerd-small      # opt into EB-NeRD small
    python build_pipeline.py --include-ebnerd-testset    # opt into EB-NeRD testset
    python build_pipeline.py --include-ebnerd-large      # opt into EB-NeRD large
    python build_pipeline.py --include-embeddings        # opt into EB-NeRD Word2Vec

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
    if not args.skip_mind:
        download_mod.download_mind(
            include_testset=args.include_mind_testset and not args.skip_mind_testset,
        )
    else:
        print("  --skip-mind set, skipping MIND download")

    if not args.skip_ebnerd:
        download_mod.download_ebnerd(
            include_small=args.include_ebnerd_small and not args.skip_ebnerd_small,
            include_large=args.include_ebnerd_large,
            include_testset=args.include_ebnerd_testset and not args.skip_ebnerd_testset,
            include_embeddings=args.include_embeddings and not args.skip_embeddings,
        )
    else:
        print("  --skip-ebnerd set, skipping EB-NeRD download")


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
    parser.add_argument("--include-mind-testset", action="store_true",
                         help="download MINDlarge_test; disabled by default")
    parser.add_argument("--skip-mind-testset", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-ebnerd-small", action="store_true",
                         help="download EB-NeRD small; disabled by default")
    parser.add_argument("--skip-ebnerd-small", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-ebnerd-large", action="store_true",
                         help="download EB-NeRD large; disabled by default because it is very large")
    parser.add_argument("--include-ebnerd-testset", action="store_true",
                         help="download EB-NeRD test set; disabled by default")
    parser.add_argument("--skip-ebnerd-testset", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-embeddings", action="store_true",
                         help="download EB-NeRD Word2Vec embeddings; disabled by default")
    parser.add_argument("--skip-embeddings", action="store_true", help=argparse.SUPPRESS)
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