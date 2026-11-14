"""Create the locked lesion-level split used by every experiment.

The production split has already been used to report results, so experiments run
on a fresh split with a different, fixed seed. Groups are lesions: every image of
a lesion lands in exactly one split, which prevents near-duplicate leakage.

Usage:
    python experiments/make_split.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 20261114
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "data", "processed", "index.csv")
OUT_DIR = os.path.join(ROOT, "experiments", "splits")


def main() -> None:
    frame = pd.read_csv(INDEX)
    os.makedirs(OUT_DIR, exist_ok=True)

    by_class: dict[str, list[str]] = defaultdict(list)
    for lesion_id, dx in frame.groupby("lesion_id")["dx"].first().items():
        by_class[dx].append(lesion_id)

    assignment: dict[str, str] = {}
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

    frame["split"] = frame["lesion_id"].map(assignment)
    if frame["split"].isna().any():
        raise ValueError("unassigned lesions")

    # Lesion-level separation check: no lesion may appear in two splits.
    per_lesion = frame.groupby("lesion_id")["split"].nunique()
    if (per_lesion > 1).any():
        raise ValueError("lesion leakage between splits")

    frame[["image_id", "lesion_id", "dx", "label", "path", "split"]].to_csv(
        os.path.join(OUT_DIR, "split.csv"), index=False
    )

    summary = {"seed": SEED, "split_unit": "lesion", "total_images": int(len(frame)), "splits": {}}
    for name, part in frame.groupby("split"):
        summary["splits"][name] = {
            "images": int(len(part)),
            "lesions": int(part["lesion_id"].nunique()),
            "class_counts": {k: int(v) for k, v in part["dx"].value_counts().items()},
        }
    summary["overlap_lesions"] = int((per_lesion > 1).sum())
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
