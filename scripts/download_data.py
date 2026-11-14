"""Download the HAM10000 dermoscopy dataset from a public HuggingFace mirror.

The official Kaggle distribution requires an account; this mirror is ungated and
contains the same 10,015 images plus the original metadata CSV.

Usage:
    python scripts/download_data.py [--root data/raw/ham7]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

REPO = "marmal88/skin_cancer"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def list_parquet_files() -> list[str]:
    url = f"https://huggingface.co/api/datasets/{REPO}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    return [
        item["rfilename"]
        for item in payload["siblings"]
        if item["rfilename"].endswith(".parquet")
    ]


def download(url: str, destination: str) -> None:
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"skip   {destination} ({os.path.getsize(destination) / 1e6:.0f} MB)")
        return
    temporary = destination + ".part"
    print(f"get    {url}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)
    print(f"saved  {destination} ({os.path.getsize(destination) / 1e6:.0f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/ham7")
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)

    download(f"{BASE}/HAM10000_metadata.csv", os.path.join(args.root, "HAM10000_metadata.csv"))
    for path in list_parquet_files():
        download(f"{BASE}/{path}", os.path.join(args.root, os.path.basename(path)))

    print("done")


if __name__ == "__main__":
    main()
