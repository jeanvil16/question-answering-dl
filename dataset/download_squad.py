"""Download the official SQuAD v1.1 dataset (~30 MB train + ~5 MB dev).

Usage:
    python dataset/download_squad.py              # downloads both files
    python dataset/download_squad.py --which train

After downloading, train on the full dataset:
    python training/train.py --train-file dataset/train-v1.1.json \
        --val-file dataset/dev-v1.1.json --epochs 2 --batch-size 64 \
        --max-context-len 320 --max-question-len 32 --min-token-freq 2
"""

import argparse
import urllib.request
from pathlib import Path

URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


def progress(block_num, block_size, total_size):
    done = min(1.0, block_num * block_size / max(total_size, 1))
    print(f"\r  downloading... {done:6.1%}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["train", "dev", "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    targets = list(URLS) if args.which == "all" else [args.which]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in targets:
        dest = args.out_dir / f"{name}-v1.1.json"
        print(f"[get] {name}: {dest.name}")
        urllib.request.urlretrieve(URLS[name], dest, reporthook=progress)
        print(f"\n[ok] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
