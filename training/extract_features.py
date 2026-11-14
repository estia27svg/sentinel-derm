"""Cache backbone embeddings for every image in the prepared dataset.

Usage:
    python training/extract_features.py --backbone efficientnet_b0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.backbone import FeatureExtractor, build_transform

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class ImageIndex(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, self.labels[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--index", default="data/processed/index.csv")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="for quick benchmarks")
    parser.add_argument("--source", default="", help="label stored alongside OOD embeddings")
    parser.add_argument("--out-name", default="", help="override the output filename")
    parser.add_argument("--tta", action="store_true", help="average with a horizontal flip")
    args = parser.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(0)

    frame = pd.read_csv(args.index)
    if args.limit:
        frame = frame.sample(n=min(args.limit, len(frame)), random_state=0).reset_index(drop=True)
    transform = build_transform(args.image_size)
    loader = DataLoader(
        ImageIndex(frame, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    model = FeatureExtractor(args.backbone, pretrained=True)
    model.eval()

    embeddings: list[np.ndarray] = []
    started = time.perf_counter()
    seen = 0
    with torch.inference_mode():
        for pixels, _ in loader:
            pooled, _ = model(pixels)
            if args.tta:
                flipped, _ = model(torch.flip(pixels, dims=[3]))
                pooled = (pooled + flipped) / 2.0
            embeddings.append(pooled.numpy().astype(np.float32))
            seen += pixels.shape[0]
            if seen % 512 == 0 or seen == len(frame):
                elapsed = time.perf_counter() - started
                print(
                    f"  {seen}/{len(frame)} images  {seen / elapsed:.1f} img/s  {elapsed:.0f}s",
                    flush=True,
                )

    matrix = np.concatenate(embeddings, axis=0)
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_tta" if args.tta else ""
    filename = args.out_name or f"embeddings_{args.backbone}{suffix}.npz"
    if not filename.endswith(".npz"):
        filename += ".npz"
    out_path = os.path.join(args.out_dir, filename)
    np.savez_compressed(
        out_path,
        embeddings=matrix,
        labels=frame["label"].to_numpy(dtype=np.int64),
        splits=frame["split"].to_numpy(dtype=str),
        image_ids=frame["image_id"].to_numpy(dtype=str),
        source=np.array(args.source) if args.source else np.array(""),
    )
    elapsed = time.perf_counter() - started
    print(
        f"saved {matrix.shape} embeddings -> {out_path} "
        f"({matrix.shape[1]}-d, {elapsed:.0f}s, {len(frame) / elapsed:.1f} img/s)"
    )


if __name__ == "__main__":
    main()
