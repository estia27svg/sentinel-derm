"""Train the linear head, calibrate the conformal layer, and evaluate everything.

Everything reported here is computed on the held-out test split, which is split
by lesion and never touched during model selection or calibration.

Usage:
    python training/train.py --embeddings artifacts/embeddings_efficientnet_b0.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import ALPHA, CLASS_NAMES, TARGET_COVERAGE
from training import conformal, metrics, ood


def load_embeddings(path: str):
    payload = np.load(path, allow_pickle=False)
    return (
        payload["embeddings"].astype(np.float64),
        payload["labels"].astype(int),
        payload["splits"].astype(str),
        payload["image_ids"].astype(str),
    )


def select_hyperparameters(features: np.ndarray, labels: np.ndarray) -> tuple[float, str | None, list[dict]]:
    """Grid search on the training split only, scored by macro F1."""
    grid = []
    best = (-1.0, 1.0, None)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for weight in (None, "balanced"):
        for c_value in (0.01, 0.1, 1.0):
            model = LogisticRegression(
                C=c_value,
                class_weight=weight,
                max_iter=1000,
                tol=1e-4,
            )
            scores = cross_val_score(model, features, labels, cv=folds, scoring="f1_macro", n_jobs=1)
            mean_score = float(scores.mean())
            grid.append({"C": c_value, "class_weight": weight, "cv_macro_f1": mean_score})
            print(f"  C={c_value:<5} weight={str(weight):<9} cv_macro_f1={mean_score:.4f}", flush=True)
            if mean_score > best[0]:
                best = (mean_score, c_value, weight)
    return best[1], best[2], grid


def pick_samples(probs, pred_sets, labels, image_ids, eligible_mask, limit=12):
    """Choose demo cases that show off confident, ambiguous and rare decisions."""
    sizes = pred_sets.sum(axis=1)
    chosen = []
    used = set()

    def add(candidates, tag, count):
        for index in candidates:
            if len(chosen) >= limit or count <= 0:
                return
            if index in used:
                continue
            used.add(index)
            chosen.append(
                {
                    "image_id": str(image_ids[index]),
                    "true_class": CLASS_NAMES[labels[index]],
                    "tag": tag,
                    "set_size": int(sizes[index]),
                    "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs[index])},
                }
            )
            count -= 1

    predicted = probs.argmax(axis=1)
    correct = predicted == labels

    def ordered(mask):
        subset = np.where(eligible_mask & mask)[0]
        return subset[np.argsort(-probs[subset].max(axis=1))]

    melanoma = labels == CLASS_NAMES.index("mel")
    add(ordered((sizes == 1) & correct & melanoma), "clear melanoma", 3)
    add(ordered((sizes == 1) & correct & (labels == CLASS_NAMES.index("nv"))), "clear nevus", 3)
    add(ordered(sizes >= 2), "ambiguous", 4)
    add(ordered((sizes == 1) & correct & (labels == CLASS_NAMES.index("bcc"))), "basal cell carcinoma", 2)
    add(ordered((sizes == 1) & correct & (labels == CLASS_NAMES.index("bkl"))), "benign keratosis", 2)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="artifacts/embeddings_efficientnet_b0.npz")
    parser.add_argument("--ood-embeddings", default="", help="optional npz with OOD embeddings")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--class-weight", default="balanced")
    args = parser.parse_args()

    started = time.perf_counter()
    os.makedirs(args.artifacts, exist_ok=True)
    embeddings, labels, splits, image_ids = load_embeddings(args.embeddings)
    print(f"embeddings: {embeddings.shape}, classes: {np.bincount(labels, minlength=7).tolist()}")

    train_mask = splits == "train"
    cal_mask = splits == "cal"
    test_mask = splits == "test"
    print(f"split sizes -> train {train_mask.sum()}, cal {cal_mask.sum()}, test {test_mask.sum()}")

    scaler = StandardScaler().fit(embeddings[train_mask])
    features = scaler.transform(embeddings)

    if args.skip_search:
        c_value, weight, grid = args.C, (None if args.class_weight == "none" else args.class_weight), []
    else:
        print("model selection (training split only):")
        c_value, weight, grid = select_hyperparameters(features[train_mask], labels[train_mask])
    print(f"chosen: C={c_value} class_weight={weight}")

    model = LogisticRegression(C=c_value, class_weight=weight, max_iter=2000, tol=1e-5)
    model.fit(features[train_mask], labels[train_mask])

    probs = model.predict_proba(features)
    probs_cal, labels_cal = probs[cal_mask], labels[cal_mask]
    probs_test, labels_test = probs[test_mask], labels[test_mask]

    quantiles = conformal.class_conditional_quantiles(probs_cal, labels_cal, args.alpha, len(CLASS_NAMES))
    sets_all = conformal.prediction_sets(probs, quantiles)
    sets_test = sets_all[test_mask]
    stats = conformal.set_statistics(sets_test, labels_test)

    print("calibrating OOD detector")
    ood_model = ood.fit_mahalanobis(features[train_mask], labels[train_mask], len(CLASS_NAMES))
    ood_cal_scores = ood.mahalanobis_scores(features[cal_mask], ood_model["means"], ood_model["precision"])
    ood_test_scores = ood.mahalanobis_scores(features[test_mask], ood_model["means"], ood_model["precision"])
    ood_all_scores = ood.mahalanobis_scores(features, ood_model["means"], ood_model["precision"])
    threshold_95 = ood.threshold_at_percentile(ood_cal_scores, 95.0)
    threshold_99 = ood.threshold_at_percentile(ood_cal_scores, 99.0)

    ood_metrics = {"threshold_95": threshold_95, "threshold_99": threshold_99, "auroc": None, "source": None}
    if args.ood_embeddings:
        ood_payload = np.load(args.ood_embeddings, allow_pickle=False)
        ood_features = scaler.transform(ood_payload["embeddings"].astype(np.float64))
        ood_scores = ood.mahalanobis_scores(ood_features, ood_model["means"], ood_model["precision"])
        comparison = ood.evaluate(ood_scores, ood_test_scores)
        ood_metrics.update(comparison)
        ood_metrics["source"] = str(ood_payload["source"]) if "source" in ood_payload else "external"
        ood_metrics["rejection_rate_at_95"] = float((ood_scores > threshold_95).mean())
        print(f"OOD AUROC: {comparison['auroc']:.4f} on {comparison['n_negatives']} negatives")

    classifier = metrics.classification_metrics(probs_test, labels_test)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alpha": args.alpha,
        "target_coverage": TARGET_COVERAGE if args.alpha == ALPHA else 1 - args.alpha,
        "backbone": args.backbone,
        "hyperparameters": {"C": c_value, "class_weight": weight, "grid": grid},
        "dataset": {
            "name": "HAM10000",
            "total_images": int(len(labels)),
            "splits": {
                name: int((splits == name).sum()) for name in ("train", "cal", "test")
            },
            "split_unit": "lesion",
        },
        "classifier": classifier,
        "conformal": {
            **{key: value for key, value in stats.items() if key != "coverage_gap"},
            "target_coverage": 1 - args.alpha,
            "coverage_by_class": metrics.per_class_coverage(sets_test, labels_test),
            "set_size_histogram": metrics.set_size_histogram(sets_test),
            "quantiles": {name: float(q) for name, q in zip(CLASS_NAMES, quantiles)},
        },
        "risk_coverage": metrics.risk_coverage(probs_test, labels_test),
        "coverage_sweep": metrics.coverage_sweep(probs_cal, labels_cal, probs_test, labels_test),
        "ood": ood_metrics,
        "latency": {"note": "measured by the serving layer at request time"},
    }

    np.savez_compressed(
        os.path.join(args.artifacts, "head.npz"),
        coef=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float32),
        classes=model.classes_.astype(np.int64),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        backbone=np.array(args.backbone),
    )
    np.savez_compressed(
        os.path.join(args.artifacts, "ood.npz"),
        means=ood_model["means"].astype(np.float32),
        precision=ood_model["precision"].astype(np.float32),
        threshold_95=np.array(threshold_95, dtype=np.float32),
        threshold_99=np.array(threshold_99, dtype=np.float32),
        shrinkage=np.array(ood_model["shrinkage"], dtype=np.float32),
    )
    with open(os.path.join(args.artifacts, "conformal.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "alpha": args.alpha,
                "target_coverage": 1 - args.alpha,
                "quantiles": [float(q) for q in quantiles],
                "n_calibration": int(cal_mask.sum()),
                "method": "class-conditional split conformal, score = 1 - p(y|x)",
            },
            handle,
            indent=2,
        )
    with open(os.path.join(args.artifacts, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    samples = pick_samples(
        probs,
        sets_all,
        labels,
        image_ids,
        test_mask & (ood_all_scores <= threshold_99),
    )
    with open(os.path.join(args.artifacts, "samples.json"), "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)

    plots = metrics.save_plots(result, sets_test, os.path.join(args.artifacts, "plots"))
    print(f"plots: {[os.path.basename(p) for p in plots]}")

    print("\n--- held-out test results ---")
    print(f"accuracy            {classifier['accuracy']:.4f}")
    print(f"macro F1            {classifier['macro_f1']:.4f}")
    print(f"calibration (ECE)   {classifier['ece']:.4f}")
    print(f"coverage @ alpha    {stats['coverage']:.4f} (target {1 - args.alpha:.2f})")
    print(f"avg set size        {stats['avg_set_size']:.2f}")
    print(f"singleton rate      {stats['singleton_rate']:.4f}")
    print(f"selective accuracy  {stats['selective_accuracy']}")
    print(f"melanoma coverage   {metrics.per_class_coverage(sets_test, labels_test)['mel']}")
    print(f"elapsed             {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
