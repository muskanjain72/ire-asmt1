"""
Download raw data files for MIND and EB-NeRD.

Small/demo sets are useful for development and debugging.
The large test sets are required for Codabench submissions.

MIND: downloaded via the Kaggle API for MIND-small and HuggingFace
for MINDlarge_test.

EB-NeRD: downloaded directly from the public S3 bucket, no auth needed.

Usage:
    python src/data/download.py
    python src/data/download.py --skip-mind-testset
    python src/data/download.py --skip-ebnerd-small
    python src/data/download.py --skip-ebnerd-testset
    python src/data/download.py --include-ebnerd-large
    python src/data/download.py --skip-embeddings
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
MIND_DIR = RAW_DIR / "mind"
EBNERD_DIR = RAW_DIR / "ebnerd"

EBNERD_BASE_URL = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"


def _run(cmd: list[str], **kwargs) -> None:
    """Run a shell command and fail fast if it exits with an error."""
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _wget(url: str, dest_dir: Path) -> Path:
    """Download a file into dest_dir with wget, unless it already exists."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"  already have {dest_path}, skipping download")
        return dest_path

    _run(["wget", "-q", "--show-progress", url, "-O", str(dest_path)])
    return dest_path


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile:
        print(
            f"  ERROR: {zip_path} is not a valid zip (corrupted download). "
            f"Delete it and re-run this script."
        )
        raise


# def download_mind(
#     dest_dir: Path = MIND_DIR,
#     include_testset: bool = True,
# ) -> None:
#     """
#     Download MIND-small using the Kaggle API and, optionally,
#     MINDlarge_test using the HuggingFace CLI.
#     """
#     print("=== MIND-small (via Kaggle) ===")

#     kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"

#     if not kaggle_creds.exists():
#         print(
#             "  Kaggle credentials not found at ~/.kaggle/kaggle.json.\n"
#             "  Skipping MIND-small download."
#         )
#         return

#     try:
#         import kaggle  # noqa: F401
#     except ImportError:
#         print("  'kaggle' package not installed. Run: pip install kaggle")
#         return

#     dest_dir.mkdir(parents=True, exist_ok=True)

#     zip_path = dest_dir / "mind-news-dataset.zip"

#     if not zip_path.exists():
#         _run([
#             "kaggle",
#             "datasets",
#             "download",
#             "-d",
#             "arashnic/mind-news-dataset",
#             "-p",
#             str(dest_dir),
#         ])
#     else:
#         print(f"  already have {zip_path}, skipping download")

#     _unzip(zip_path, dest_dir)

#     # Remove stray duplicate folder from the Kaggle mirror
#     stray = dest_dir / "news.tsv"
#     if stray.is_dir():
#         shutil.rmtree(stray)

#     train_dir = dest_dir / "MINDsmall_train"

#     if (train_dir / "news.tsv").exists() and \
#        (train_dir / "behaviors.tsv").exists():
#         print(f"  MIND-small ready at {train_dir}")
#     else:
#         print(
#             f"  WARNING: expected files not found under {train_dir} — "
#             f"inspect {dest_dir} manually."
#         )

#     if not include_testset:
#         print("  MINDlarge_test skipped")
#         return

#     # ---------------------------------------------------------
#     # MINDlarge_test - required for Codabench submission
#     # ---------------------------------------------------------
#     print("=== MINDlarge_test ===")

#     test_dir = dest_dir / "test"
#     test_dir.mkdir(parents=True, exist_ok=True)

#     test_zip = test_dir / "MINDlarge_test.zip"

#     if not test_zip.exists():
#         _run([
#             "hf",
#             "download",
#             "yjw1029/MIND",
#             "--repo-type",
#             "dataset",
#             "--include",
#             "MINDlarge_test.zip",
#             "--local-dir",
#             str(test_dir),
#         ])
#     else:
#         print(f"  already have {test_zip}, skipping download")

#     _unzip(test_zip, test_dir / "MINDlarge_test")

#     print(f"  MINDlarge_test ready at {test_dir / 'MINDlarge_test'}")


def download_mind(
    dest_dir: Path = MIND_DIR,
    include_testset: bool = True,
) -> None:
    """
    Download MIND-small using the Kaggle API and, optionally,
    MINDlarge_test using the HuggingFace CLI.
    """
    print("=== MIND-small (via Kaggle) ===")

    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "mind-news-dataset.zip"

    if not zip_path.exists():
        _run([
            "kaggle",
            "datasets",
            "download",
            "-d",
            "arashnic/mind-news-dataset",
            "-p",
            str(dest_dir),
        ])
    else:
        print(f"  already have {zip_path}, skipping download")

    _unzip(zip_path, dest_dir)

    # Remove stray duplicate folder from the Kaggle mirror
    stray = dest_dir / "news.tsv"
    if stray.is_dir():
        shutil.rmtree(stray)

    train_dir = dest_dir / "MINDsmall_train"

    if (train_dir / "news.tsv").exists() and \
       (train_dir / "behaviors.tsv").exists():
        print(f"  MIND-small ready at {train_dir}")
    else:
        print(
            f"  WARNING: expected files not found under {train_dir} — "
            f"inspect {dest_dir} manually."
        )

    if not include_testset:
        print("  MINDlarge_test skipped")
        return

    # ---------------------------------------------------------
    # MINDlarge_test - required for Codabench submission
    # ---------------------------------------------------------
    print("=== MINDlarge_test ===")

    test_dir = dest_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_zip = test_dir / "MINDlarge_test.zip"

    if not test_zip.exists():
        _run([
            "hf",
            "download",
            "yjw1029/MIND",
            "--repo-type",
            "dataset",
            "--include",
            "MINDlarge_test.zip",
            "--local-dir",
            str(test_dir),
        ])
    else:
        print(f"  already have {test_zip}, skipping download")

    _unzip(test_zip, test_dir / "MINDlarge_test")

    print(f"  MINDlarge_test ready at {test_dir / 'MINDlarge_test'}")

def download_ebnerd(
    dest_dir: Path = EBNERD_DIR,
    include_small: bool = True,
    include_large: bool = False,
    include_testset: bool = True,
    include_embeddings: bool = True,
) -> None:
    """
    Download EB-NeRD demo/small data, optional large data,
    required test set, and optional Word2Vec embeddings.
    """
    print("=== EB-NeRD (via S3, no auth needed) ===")

    # Demo
    demo_zip = _wget(
        f"{EBNERD_BASE_URL}/ebnerd_demo.zip",
        dest_dir,
    )
    _unzip(demo_zip, dest_dir / "demo")
    print(f"  demo ready at {dest_dir / 'demo'}")

    # Small
    if include_small:
        small_zip = _wget(
            f"{EBNERD_BASE_URL}/ebnerd_small.zip",
            dest_dir,
        )
        _unzip(small_zip, dest_dir / "small")
        print(f"  small ready at {dest_dir / 'small'}")

    # Large - optional full-scale training
    if include_large:
        large_zip = _wget(
            f"{EBNERD_BASE_URL}/ebnerd_large.zip",
            dest_dir,
        )
        _unzip(large_zip, dest_dir / "large")
        print(f"  large ready at {dest_dir / 'large'}")

        articles_zip = _wget(
            f"{EBNERD_BASE_URL}/artifacts/articles_large_only.zip",
            dest_dir,
        )
        _unzip(articles_zip, dest_dir / "large")
        print(f"  large articles ready at {dest_dir / 'large'}")

    # Test set - required for Codabench
    if include_testset:
        testset_zip = _wget(
            f"{EBNERD_BASE_URL}/ebnerd_testset.zip",
            dest_dir,
        )
        _unzip(testset_zip, dest_dir / "testset")
        print(f"  testset ready at {dest_dir / 'testset'}")

    # Word2Vec embeddings
    if include_embeddings:
        emb_zip = _wget(
            f"{EBNERD_BASE_URL}/artifacts/Ekstra_Bladet_word2vec.zip",
            dest_dir,
        )
        _unzip(emb_zip, dest_dir / "word2vec")
        print(f"  word2vec embeddings ready at {dest_dir / 'word2vec'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--skip-mind",
        action="store_true",
        help="skip all MIND downloads",
    )

    parser.add_argument(
        "--skip-ebnerd",
        action="store_true",
        help="skip all EB-NeRD downloads",
    )

    parser.add_argument(
        "--skip-mind-testset",
        action="store_true",
        help="skip MINDlarge_test",
    )

    parser.add_argument(
        "--skip-ebnerd-small",
        action="store_true",
        help="skip EB-NeRD small",
    )

    parser.add_argument(
        "--include-ebnerd-large",
        action="store_true",
        help="download EB-NeRD large; disabled by default because it is very large",
    )

    parser.add_argument(
        "--skip-ebnerd-testset",
        action="store_true",
        help="skip EB-NeRD test set",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="skip Word2Vec embeddings",
    )

    args = parser.parse_args()

    if not args.skip_mind:
        download_mind(include_testset=not args.skip_mind_testset)

    if not args.skip_ebnerd:
        download_ebnerd(
            include_small=not args.skip_ebnerd_small,
            include_large=args.include_ebnerd_large,
            include_testset=not args.skip_ebnerd_testset,
            include_embeddings=not args.skip_embeddings,
        )

    print("\nDone. Raw data under data/raw/")


if __name__ == "__main__":
    main()