"""Locked evaluation protocol for model improvements.

Rules enforced here:
  * model selection uses GroupKFold over *lesions* on the training split only;
  * the calibration split is used only for conformal quantiles and OOD thresholds;
  * the test split is scored once, after the candidate is frozen;
  * uncertainty is reported as lesion-cluster bootstrap intervals.

Usage:
    python experiments/harness.py --embeddings convnext_tiny=experiments/artifacts/embeddings_convnext_tiny.npz \
        --ood-embeddings convnext_tiny=experiments/artifacts/embeddings_ood_convnext_tiny.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training import conformal, ood

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TAU_GRID = (0.0, 0.25, 0.5, 1.0)
SEED = 20261114


# --------------------------------------------------------------------- loading

def load_split() -> pd.DataFrame:
    frame = pd.read_csv(os.path.join(ROOT, "experiments", "splits", "split.csv"))
    return frame.set_index("image_id")


def load_embeddings(specs: list[str], split: pd.DataFrame):
    matrices, reference_ids = [], None
    for spec in specs:
        label, path = spec.split("=", 1)
        payload = np.load(path, allow_pickle=False)
        ids = payload["image_ids"].astype(str)
        if reference_ids is None:
            reference_ids = ids
            matrix = payload["embeddings"].astype(np.float64)
        else:
            if not np.array_equal(ids, reference_ids):
                order = pd.Index(ids)
                if set(ids) != set(reference_ids):
                    raise ValueError(f"{path} covers a different image set")
                matrix = payload["embeddings"].astype(np.float64)[[order.get_loc(i) for i in reference_ids]]
            else:
                matrix = payload["embeddings"].astype(np.float64)
        matrices.append(matrix)
        print(f"  loaded {label}: {matrix.shape}")

    meta = split.loc[reference_ids]
    labels = meta["label"].to_numpy(dtype=int)
    # Guard against silent misalignment between embeddings and metadata.
    for spec in specs:
        _, path = spec.split("=", 1)
        stored = np.load(path, allow_pickle=False)["labels"].astype(int)
        aligned = stored if len(stored) == len(reference_ids) else stored[[pd.Index(np.load(path)["image_ids"].astype(str)).get_loc(i) for i in reference_ids]]
        if not np.array_equal(aligned, labels):
            raise ValueError(f"label mismatch for {path}")

    features = np.concatenate(matrices, axis=1) if len(matrices) > 1 else matrices[0]
    return (
        features,
        labels,
        meta["lesion_id"].to_numpy(dtype=str),
        meta["split"].to_numpy(dtype=str),
        reference_ids,
    )


# ----------------------------------------------------------------------- stats

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (None, None)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def class_table(probs: np.ndarray, labels: np.ndarray, pred_sets: np.ndarray) -> dict:
    predictions = probs.argmax(axis=1)
    table = {}
    for index, name in enumerate(CLASS_NAMES):
        positive = labels == index
        predicted = predictions == index
        tp = int((positive & predicted).sum())
        fn = int((positive & ~predicted).sum())
        fp = int((~positive & predicted).sum())
        tn = int((~positive & ~predicted).sum())
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        specificity = tn / (tn + fp) if tn + fp else None
        npv = tn / (tn + fn) if tn + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        covered = int(pred_sets[positive, index].sum()) if positive.any() else 0
        low, high = wilson_interval(covered, int(positive.sum()))
        table[name] = {
            "support": int(positive.sum()),
            "sensitivity": recall,
            "specificity": specificity,
            "ppv": precision,
            "npv": npv,
            "f1": f1,
            "conformal_coverage": covered / int(positive.sum()) if positive.any() else None,
            "coverage_ci95": [low, high],
        }
    return table


def overall_metrics(probs: np.ndarray, labels: np.ndarray, pred_sets: np.ndarray) -> dict:
    predictions = probs.argmax(axis=1)
    table = class_table(probs, labels, pred_sets)
    recalls = [v["sensitivity"] for v in table.values() if v["sensitivity"] is not None]
    sizes = pred_sets.sum(axis=1)
    covered = pred_sets[np.arange(len(labels)), labels]
    singleton = sizes == 1
    singleton_correct = (predictions[singleton] == labels[singleton]).mean() if singleton.any() else None
    f1_values = [v["f1"] for v in table.values()]
    return {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "ece": expected_calibration_error(probs, labels),
        "conformal_coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float(singleton.mean()),
        "empty_set_rate": float((sizes == 0).mean()),
        "selective_accuracy": float(singleton_correct) if singleton_correct is not None else None,
        "classes": table,
        "confusion_matrix": np.array(
            [[int(((labels == i) & (predictions == j)).sum()) for j in range(len(CLASS_NAMES))] for i in range(len(CLASS_NAMES))]
        ).tolist(),
    }


def cluster_bootstrap(probs, labels, pred_sets, lesions, iterations=1000, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(lesions)
    index_by_lesion = {lesion: np.where(lesions == lesion)[0] for lesion in unique}
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "conformal_coverage", "avg_set_size",
            "singleton_rate", "selective_accuracy")
    draws = {key: [] for key in keys}
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_lesion[lesion] for lesion in sampled])
        result = overall_metrics(probs[rows], labels[rows], pred_sets[rows])
        for key in keys:
            value = result[key]
            if value is not None:
                draws[key].append(value)
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None
        for key, values in draws.items()
    }


def per_class_bootstrap(probs, labels, pred_sets, lesions, iterations=1000) -> dict:
    rng = np.random.default_rng(SEED + 1)
    unique = np.unique(lesions)
    index_by_lesion = {lesion: np.where(lesions == lesion)[0] for lesion in unique}
    ranges = {name: [] for name in CLASS_NAMES}
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_lesion[lesion] for lesion in sampled])
        result = class_table(probs[rows], labels[rows], pred_sets[rows])
        for name in CLASS_NAMES:
            ranges[name].append(result[name]["sensitivity"])
    return {
        name: [float(np.percentile([v for v in values if v is not None], 2.5)),
               float(np.percentile([v for v in values if v is not None], 97.5))]
        if any(v is not None for v in values) else None
        for name, values in ranges.items()
    }


# ------------------------------------------------------------------ adjustment

def prior_adjusted(probs: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    if tau == 0.0:
        return probs
    adjusted = probs / np.power(priors, tau)[None, :]
    return adjusted / adjusted.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------- CV run

def cross_validate(features, labels, lesions) -> tuple[list[dict], dict]:
    priors = np.bincount(labels, minlength=len(CLASS_NAMES)) / len(labels)
    folds = list(GroupKFold(n_splits=5).split(features, labels, groups=lesions))
    rows = []
    for weight in (None, "balanced"):
        for c_value in C_GRID:
            probs = np.zeros((len(labels), len(CLASS_NAMES)))
            for train_index, val_index in folds:
                model = LogisticRegression(C=c_value, class_weight=weight, max_iter=2000, tol=1e-4)
                model.fit(features[train_index], labels[train_index])
                probs[val_index] = model.predict_proba(features[val_index])
            taus = TAU_GRID if weight is None else (0.0,)
            for tau in taus:
                adjusted = prior_adjusted(probs, priors, tau)
                metrics = overall_metrics(adjusted, labels, np.zeros((len(labels), len(CLASS_NAMES)), dtype=bool))
                row = {
                    "C": c_value,
                    "class_weight": weight,
                    "prior_tau": tau,
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "accuracy": metrics["accuracy"],
                    "ece": metrics["ece"],
                    "mel_recall": metrics["classes"]["mel"]["sensitivity"],
                    "nv_recall": metrics["classes"]["nv"]["sensitivity"],
                    "per_class_recall": {name: metrics["classes"][name]["sensitivity"] for name in CLASS_NAMES},
                }
                rows.append(row)
                print(
                    f"  C={c_value:<6} weight={str(weight):<9} tau={tau:<5} "
                    f"macroF1={row['macro_f1']:.4f} melR={row['mel_recall']:.4f} "
                    f"balAcc={row['balanced_accuracy']:.4f} acc={row['accuracy']:.4f}",
                    flush=True,
                )
    baseline = next(
        row for row in rows if row["C"] == 0.003 and row["class_weight"] is None and row["prior_tau"] == 0.0
    )
    return rows, baseline


def select(rows: list[dict], baseline: dict) -> dict:
    """Macro-F1 selection that may not regress melanoma recall beyond 2 points."""
    floor = baseline["mel_recall"] - 0.02
    eligible = [row for row in rows if row["mel_recall"] >= floor]
    if not eligible:
        eligible = rows
    best = max(eligible, key=lambda row: (row["macro_f1"], row["mel_recall"]))
    best_mel = max(rows, key=lambda row: (row["mel_recall"], row["macro_f1"]))
    return {"primary": best, "melanoma_max": best_mel, "floor": floor}


# ---------------------------------------------------------------------- OOD

def evaluate_ood(embeddings, labels, lesions, splits, ood_specs) -> dict:
    train = splits == "train"
    cal = splits == "cal"
    test = splits == "test"

    scaler = StandardScaler().fit(embeddings[train])
    features = scaler.transform(embeddings)
    model = ood.fit_mahalanobis(features[train], labels[train], len(CLASS_NAMES))
    cal_scores = ood.mahalanobis_scores(features[cal], model["means"], model["precision"])
    test_scores = ood.mahalanobis_scores(features[test], model["means"], model["precision"])
    threshold_95 = ood.threshold_at_percentile(cal_scores, 95.0)
    threshold_99 = ood.threshold_at_percentile(cal_scores, 99.0)

    result = {
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
        "in_distribution_rejection_rate_95": float((test_scores > threshold_95).mean()),
        "in_distribution_rejection_rate_99": float((test_scores > threshold_99).mean()),
        "sources": {},
    }
    for spec in ood_specs or []:
        label, path = spec.split("=", 1)
        payload = np.load(path, allow_pickle=False)
        ood_features = scaler.transform(payload["embeddings"].astype(np.float64))
        scores = ood.mahalanobis_scores(ood_features, model["means"], model["precision"])
        comparison = ood.evaluate(scores, test_scores)
        result["sources"][label] = {
            **comparison,
            "rejection_rate_at_95": float((scores > threshold_95).mean()),
            "rejection_rate_at_99": float((scores > threshold_99).mean()),
        }
    return result


# -------------------------------------------------------------------- reporting

def summarize_row(rows, key):
    return next(row for row in rows if all(row[k] == v for k, v in key.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="append", required=True, help="label=path, repeatable")
    parser.add_argument("--ood-embeddings", action="append", default=[])
    parser.add_argument("--name", default="experiment")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--artifacts", default="experiments/artifacts")
    parser.add_argument("--reports", default="experiments/reports")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--cv-only", action="store_true", help="stop after model selection; never touch test")
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--class-weight", default=None)
    parser.add_argument("--prior-tau", type=float, default=0.0)
    parser.add_argument("--ood-only", action="store_true")
    args = parser.parse_args()

    split = load_split()
    embeddings, labels, lesions, splits, image_ids = load_embeddings(args.embeddings, split)
    print(f"features {embeddings.shape}, train {(splits == 'train').sum()}, cal {(splits == 'cal').sum()}, test {(splits == 'test').sum()}")

    if args.ood_only:
        result = evaluate_ood(embeddings, labels, lesions, splits, args.ood_embeddings)
        print(json.dumps(result, indent=2))
        return

    train_mask, cal_mask, test_mask = splits == "train", splits == "cal", splits == "test"
    scaler = StandardScaler().fit(embeddings[train_mask])
    features = scaler.transform(embeddings)

    if not args.skip_cv:
        print("lesion-level 5-fold CV on train:")
        rows, baseline = cross_validate(features[train_mask], labels[train_mask], lesions[train_mask])
        choice = select(rows, baseline)
        print(f"baseline (C=0.003, none): macroF1={baseline['macro_f1']:.4f} melR={baseline['mel_recall']:.4f}")
        print(f"selected: {choice['primary']}")
        if args.cv_only:
            os.makedirs(args.reports, exist_ok=True)
            path = os.path.join(args.reports, f"{args.name}_cv.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "name": args.name,
                        "features": [spec.split("=", 1)[0] for spec in args.embeddings],
                        "dimension": int(embeddings.shape[1]),
                        "rows": rows,
                        "baseline": baseline,
                        "selected": choice,
                    },
                    handle,
                    indent=2,
                )
            print(f"wrote {path} (test split untouched)")
            return
    else:
        if args.C is None:
            raise SystemExit("--skip-cv requires --C")
        rows, baseline = [], None
        choice = {
            "primary": {"C": args.C, "class_weight": args.class_weight, "prior_tau": args.prior_tau}
        }
        print(f"using frozen config: {choice['primary']}")

    chosen = choice["primary"]
    weight = None if chosen["class_weight"] in (None, "none") else chosen["class_weight"]
    priors = np.bincount(labels[train_mask], minlength=len(CLASS_NAMES)) / int(train_mask.sum())

    model = LogisticRegression(C=chosen["C"], class_weight=weight, max_iter=2000, tol=1e-4)
    model.fit(features[train_mask], labels[train_mask])
    probs = prior_adjusted(model.predict_proba(features), priors, chosen["prior_tau"])

    quantiles = conformal.class_conditional_quantiles(probs[cal_mask], labels[cal_mask], 0.1, len(CLASS_NAMES))
    pred_sets = conformal.prediction_sets(probs, quantiles)

    metrics = overall_metrics(probs[test_mask], labels[test_mask], pred_sets[test_mask])
    intervals = cluster_bootstrap(
        probs[test_mask], labels[test_mask], pred_sets[test_mask], lesions[test_mask], args.iterations
    )
    class_intervals = per_class_bootstrap(
        probs[test_mask], labels[test_mask], pred_sets[test_mask], lesions[test_mask], args.iterations
    )
    ood_metrics = evaluate_ood(embeddings, labels, lesions, splits, args.ood_embeddings)

    result = {
        "name": args.name,
        "protocol": {
            "selection": "lesion-level 5-fold CV on train, macro F1 with melanoma-recall floor",
            "calibration": "class-conditional split conformal on cal only",
            "test": "scored once after freezing",
            "bootstrap": f"lesion-cluster bootstrap, {args.iterations} resamples",
        },
        "features": [spec.split("=", 1)[0] for spec in args.embeddings],
        "dimension": int(embeddings.shape[1]),
        "split": {name: int((splits == name).sum()) for name in ("train", "cal", "test")},
        "cv": {"rows": rows, "baseline": baseline, "selected": choice},
        "chosen": chosen,
        "test": metrics,
        "test_ci95": intervals,
        "test_per_class_ci95": class_intervals,
        "conformal": {
            "alpha": 0.1,
            "quantiles": {name: float(q) for name, q in zip(CLASS_NAMES, quantiles)},
            "per_class_coverage": {name: metrics["classes"][name]["conformal_coverage"] for name in CLASS_NAMES},
        },
        "ood": ood_metrics,
    }

    os.makedirs(args.artifacts, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.artifacts, f"head_{args.name}.npz"),
        coef=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float32),
        classes=model.classes_.astype(np.int64),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        backbone=np.array("+".join(a.split("=", 1)[0] for a in args.embeddings)),
        prior_tau=np.array(chosen["prior_tau"], dtype=np.float32),
    )
    with open(os.path.join(args.artifacts, f"conformal_{args.name}.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "alpha": 0.1,
                "quantiles": [float(q) for q in quantiles],
                "n_calibration": int(cal_mask.sum()),
                "method": "class-conditional split conformal, score = 1 - p(y|x)",
            },
            handle,
            indent=2,
        )
    with open(os.path.join(args.reports, f"{args.name}.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\n--- locked test set (single evaluation) ---")
    print(f"accuracy          {metrics['accuracy']:.4f}  CI {intervals['accuracy']}")
    print(f"balanced accuracy {metrics['balanced_accuracy']:.4f}  CI {intervals['balanced_accuracy']}")
    print(f"macro F1          {metrics['macro_f1']:.4f}  CI {intervals['macro_f1']}")
    print(f"melanoma recall   {metrics['classes']['mel']['sensitivity']:.4f}  CI {class_intervals['mel']}")
    print(f"conformal cover   {metrics['conformal_coverage']:.4f}  CI {intervals['conformal_coverage']}")
    print(f"avg set size      {metrics['avg_set_size']:.3f}")
    print("per class:")
    for name in CLASS_NAMES:
        entry = metrics["classes"][name]
        ci = class_intervals[name]
        print(
            f"  {name:<6} n={entry['support']:<5} sens={entry['sensitivity']:.3f} "
            f"spec={entry['specificity']:.3f} ppv={entry['ppv']:.3f} f1={entry['f1']:.3f} "
            f"cov={entry['conformal_coverage']:.3f} sensCI={ci}"
        )
    print(json.dumps(ood_metrics, indent=2))


if __name__ == "__main__":
    main()
