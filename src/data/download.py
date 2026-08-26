"""
Download raw data files for MIND-small and EB-NeRD demo/small.

MIND: downloaded via the Kaggle API (the official HuggingFace repo
is gated and the original Azure blob no longer allows public access
as of Aug 2026 — both were tried and failed during development).
Requires a Kaggle API token at ~/.kaggle/kaggle.json
(create one at https://www.kaggle.com/settings -> API -> Create New Token).

EB-NeRD: downloaded directly from the public S3 bucket, no auth needed.

Usage:
    python src/data/download.py                # download everything
    python src/data/download.py --skip-ebnerd-small       # demo only
    python src/data/download.py --skip-embeddings         # skip word2vec (133MB)
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
    """Run a shell command and fail fast if the command exits with an error."""
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _wget(url: str, dest_dir: Path) -> Path:
    """
    Download a file into dest_dir with wget, unless the file already exists.
    Returns the local path so callers can pass it directly to unzip/extract steps.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename
    if dest_path.exists():
        print(f"  already have {dest_path}, skipping download")
        return dest_path
    _run(["wget", "-q", "--show-progress", url, "-O", str(dest_path)])
    return dest_path


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file into dest_dir and surface a clear message for corrupt downloads."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    except zipfile.BadZipFile:
        print(f"  ERROR: {zip_path} is not a valid zip (corrupted download). "
              f"Delete it and re-run this script.")
        raise


def download_mind(dest_dir: Path = MIND_DIR) -> None:
    """
    Download and extract the MIND-small dataset using the Kaggle API.

    The download is skipped when Kaggle credentials or the kaggle package are missing,
    because this dataset source requires authenticated Kaggle access.
    """
    print("=== MIND-small (via Kaggle) ===")
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        print(
            "  Kaggle credentials not found at ~/.kaggle/kaggle.json.\n"
            "  Create a token at https://www.kaggle.com/settings -> API -> "
            "'Create New Token', then place the downloaded kaggle.json at "
            "~/.kaggle/kaggle.json and chmod 600 it. Skipping MIND download."
        )
        return

    try:
        import kaggle  # noqa: F401 (import triggers credential validation)
    except ImportError:
        print("  'kaggle' package not installed. Run: pip install kaggle")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "mind-news-dataset.zip"
    if not zip_path.exists():
        _run([
            "kaggle", "datasets", "download",
            "-d", "arashnic/mind-news-dataset",
            "-p", str(dest_dir),
        ])
    else:
        print(f"  already have {zip_path}, skipping download")

    _unzip(zip_path, dest_dir)

    # this mirror also drops a stray duplicate news.tsv/ folder — clean it up
    stray = dest_dir / "news.tsv"
    if stray.is_dir():
        shutil.rmtree(stray)

    train_dir = dest_dir / "MINDsmall_train"
    if (train_dir / "news.tsv").exists() and (train_dir / "behaviors.tsv").exists():
        print(f"  MIND ready at {train_dir}")
    else:
        print(f"  WARNING: expected files not found under {train_dir} — "
              f"inspect {dest_dir} manually.")


def download_ebnerd(
    dest_dir: Path = EBNERD_DIR,
    include_small: bool = True,
    include_embeddings: bool = True,
) -> None:
    """
    Download and extract EB-NeRD demo data, with optional small split and embeddings.

    EB-NeRD files are public S3 downloads, so this path does not require credentials.
    """
    print("=== EB-NeRD (via S3, no auth needed) ===")

    demo_zip = _wget(f"{EBNERD_BASE_URL}/ebnerd_demo.zip", dest_dir)
    _unzip(demo_zip, dest_dir / "demo")
    print(f"  demo ready at {dest_dir / 'demo'}")

    if include_small:
        small_zip = _wget(f"{EBNERD_BASE_URL}/ebnerd_small.zip", dest_dir)
        _unzip(small_zip, dest_dir / "small")
        print(f"  small ready at {dest_dir / 'small'}")

    if include_embeddings:
        emb_zip = _wget(
            f"{EBNERD_BASE_URL}/artifacts/Ekstra_Bladet_word2vec.zip", dest_dir
        )
        _unzip(emb_zip, dest_dir / "word2vec")
        print(f"  word2vec embeddings ready at {dest_dir / 'word2vec'}")


def main():
    """Parse command-line flags and run the selected dataset download steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mind", action="store_true")
    parser.add_argument("--skip-ebnerd", action="store_true")
    parser.add_argument("--skip-ebnerd-small", action="store_true",
                         help="only download EB-NeRD demo, not the larger 'small' bundle")
    parser.add_argument("--skip-embeddings", action="store_true",
                         help="skip the word2vec embeddings download (133MB)")
    args = parser.parse_args()

    if not args.skip_mind:
        download_mind()
    if not args.skip_ebnerd:
        download_ebnerd(
            include_small=not args.skip_ebnerd_small,
            include_embeddings=not args.skip_embeddings,
        )

    print("\nDone. Raw data under data/raw/")


if __name__ == "__main__":
    main()