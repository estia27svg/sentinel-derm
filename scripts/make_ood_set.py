"""Build a standard out-of-distribution negative set from the CIFAR-10 test set.

CIFAR-10 is the conventional negative set in the OOD-detection literature, which
makes the AUROC comparable. It is an *easy* benchmark - natural images look
nothing like dermoscopy - so we report it as a sanity check rather than as
evidence that the gate will catch a hard shift such as clinical photographs.

Usage:
    python scripts/make_ood_set.py --count 1000
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import urllib.request

import pandas as pd
from PIL import Image

URL = "https://huggingface.co/datasets/djghosh/wds_cifar10_test/resolve/main/test/0.tar"
TARGET = 256


def download(url: str, destination: str) -> None:
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".part"
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--root", default="data/ood")
    args = parser.parse_args()

    image_dir = os.path.join(args.root, "cifar")
    os.makedirs(image_dir, exist_ok=True)
    archive = os.path.join("data", "raw", "cifar_wds", "test_0.tar")
    download(URL, archive)

    rows = []
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with Image.open(io.BytesIO(handle.read())) as image:
                image = image.convert("RGB").resize((TARGET, TARGET), Image.Resampling.BICUBIC)
                image_id = f"cifar_{len(rows):05d}"
                path = os.path.join(image_dir, f"{image_id}.jpg")
                image.save(path, "JPEG", quality=92)
            rows.append({"image_id": image_id, "path": path.replace("\\", "/"), "label": 0, "split": "ood"})
            if len(rows) >= args.count:
                break

    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(args.root, "index.csv"), index=False)
    print(f"wrote {len(frame)} OOD images -> {image_dir}")


if __name__ == "__main__":
    main()
