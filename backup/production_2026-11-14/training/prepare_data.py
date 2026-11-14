"""Turn the HAM10000 parquet shards into an on-disk image folder plus splits.

Two data-integrity issues are handled here:

1. The public mirror we download from has overlapping train/validation/test
   shards: 3,339 images appear more than once. We deduplicate on ``image_id``.
2. HAM10000 contains several photographs of the same lesion (7,470 lesions,
   10,015 images). Splitting by image leaks near-duplicates into the test set,
   so splits are made by *lesion* and stratified per diagnosis.

Usage:
    python training/prepare_data.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic keratosis / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
DX_ALIASES = {
    "melanoma": "mel",
    "mel": "mel",
    "melanocytic_nevi": "nv",
    "melanocytic_nevus": "nv",
    "nevus": "nv",
    "nv": "nv",
    "benign_keratosis-like_lesions": "bkl",
    "benign_keratosis": "bkl",
    "bkl": "bkl",
    "basal_cell_carcinoma": "bcc",
    "bcc": "bcc",
    "actinic_keratoses": "akiec",
    "actinic_keratosis": "akiec",
    "akiec": "akiec",
    "vascular_lesions": "vasc",
    "vascular_lesion": "vasc",
    "vasc": "vasc",
    "dermatofibroma": "df",
    "df": "df",
}
EXPECTED = {"akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142}

IMAGE_SIZE = 256
SEED = 42


def normalise_dx(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    if key not in DX_ALIASES:
        raise ValueError(f"unknown diagnosis label: {value!r}")
    return DX_ALIASES[key]


def iter_shard_rows(root: str):
    """Yield (image_id, lesion_id, dx_code, raw_bytes, extra_metadata)."""
    for name in sorted(os.listdir(root)):
        if not name.endswith(".parquet"):
            continue
        parquet = pq.ParquetFile(os.path.join(root, name))
        available = set(parquet.schema_arrow.names)
        columns = [column for column in ("image", "image_id", "lesion_id", "dx", "age", "sex", "localization") if column in available]
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                image = row.get("image") or {}
                raw = image.get("bytes")
                image_id = row.get("image_id") or os.path.splitext(os.path.basename(image.get("path") or ""))[0]
                if not raw or not image_id:
                    continue
                yield (
                    image_id,
                    row.get("lesion_id") or image_id,
                    normalise_dx(row.get("dx")),
                    raw,
                    {
                        "age": row.get("age"),
                        "sex": row.get("sex"),
                        "localization": row.get("localization"),
                    },
                )


def save_image(raw: bytes, destination: str) -> float:
    """Write the resized image and return a sharpness score (edge variance)."""
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        image.save(destination, "JPEG", quality=92)
        edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float64)
        return float(edges[2:-2, 2:-2].var())


def lesion_splits(meta: pd.DataFrame) -> dict[str, str]:
    """Assign every lesion to exactly one split, stratified by diagnosis."""
    assignment: dict[str, str] = {}
    by_class: dict[str, list[str]] = defaultdict(list)
    for lesion_id, dx in meta.groupby("lesion_id")["dx"].first().items():
        by_class[dx].append(lesion_id)

    for dx, lesions in by_class.items():
        lesions = sorted(lesions)
        train, rest = train_test_split(lesions, train_size=0.6, random_state=SEED)
        cal, test = train_test_split(rest, test_size=0.5, random_state=SEED)
        for lesion in train:
            assignment[lesion] = "train"
        for lesion in cal:
            assignment[lesion] = "cal"
        for lesion in test:
            assignment[lesion] = "test"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/ham7")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--splits", default="data/splits")
    args = parser.parse_args()

    image_dir = os.path.join(args.out, "images")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(args.splits, exist_ok=True)

    rows = []
    seen: set[str] = set()
    duplicates = 0
    for image_id, lesion_id, dx, raw, extra in iter_shard_rows(args.raw):
        if image_id in seen:
            duplicates += 1
            continue
        seen.add(image_id)

        destination = os.path.join(image_dir, f"{image_id}.jpg")
        if os.path.exists(destination):
            with Image.open(destination) as existing:
                edges = np.asarray(existing.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float64)
                sharpness = float(edges[2:-2, 2:-2].var())
        else:
            sharpness = save_image(raw, destination)

        rows.append(
            {
                "image_id": image_id,
                "lesion_id": lesion_id,
                "dx": dx,
                "label": CLASS_NAMES.index(dx),
                "path": os.path.relpath(destination).replace("\\", "/"),
                "sharpness": sharpness,
                "age": extra["age"],
                "sex": extra["sex"],
                "localization": extra["localization"],
            }
        )
        if len(rows) % 1000 == 0:
            print(f"  prepared {len(rows)} images", flush=True)

    index = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    print(f"deduplicated {duplicates} repeated image rows")

    counts = index["dx"].value_counts().to_dict()
    unexpected = {
        name: {"expected": EXPECTED[name], "found": int(counts.get(name, 0))}
        for name in EXPECTED
        if int(counts.get(name, 0)) != EXPECTED[name]
    }
    if unexpected:
        raise ValueError(f"class counts do not match HAM10000: {unexpected}")

    assignment = lesion_splits(index[["lesion_id", "dx"]])
    index["split"] = index["lesion_id"].map(assignment)
    if index["split"].isna().any():
        raise ValueError("some lesions received no split assignment")

    index.to_csv(os.path.join(args.out, "index.csv"), index=False)

    summary = {"seed": SEED, "image_size": IMAGE_SIZE, "total": int(len(index)), "splits": {}}
    for split, frame in index.groupby("split"):
        frame.to_csv(os.path.join(args.splits, f"{split}.csv"), index=False)
        summary["splits"][split] = {
            "images": int(len(frame)),
            "lesions": int(frame["lesion_id"].nunique()),
            "class_counts": frame["dx"].value_counts().to_dict(),
        }
    with open(os.path.join(args.splits, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    train_sharpness = index.loc[index["split"] == "train", "sharpness"]
    quality = {
        "sharpness_p01": float(np.percentile(train_sharpness, 1.0)),
        "sharpness_p05": float(np.percentile(train_sharpness, 5.0)),
        "sharpness_median": float(np.median(train_sharpness)),
        "min_dimension": 96,
    }
    with open(os.path.join(args.out, "quality.json"), "w", encoding="utf-8") as handle:
        json.dump(quality, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"images written: {len(index)} -> {image_dir}")


if __name__ == "__main__":
    main()
