"""Metric computations and figures used by the evaluation dashboard."""

from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, log_loss

from training.conformal import class_conditional_quantiles, prediction_sets

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if not in_bin.any():
            continue
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(in_bin.sum()),
                "confidence": float(confidence[in_bin].mean()),
                "accuracy": float(correct[in_bin].mean()),
            }
        )
    return bins


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    predictions = probs.argmax(axis=1)
    confusion = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums > 0)

    recall = {}
    precision = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = confusion[index, index]
        recall[name] = float(true_positive / row_sums[index, 0]) if row_sums[index, 0] else None
        column_total = confusion[:, index].sum()
        precision[name] = float(true_positive / column_total) if column_total else None

    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=list(range(len(CLASS_NAMES))), zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "nll": float(log_loss(labels, np.clip(probs, 1e-9, 1.0), labels=list(range(len(CLASS_NAMES))))),
        "ece": expected_calibration_error(probs, labels),
        "per_class_recall": recall,
        "per_class_precision": precision,
        "support": {name: int(row_sums[index, 0]) for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_normalized": normalized.round(4).tolist(),
        "reliability": reliability_bins(probs, labels),
    }


def risk_coverage(probs: np.ndarray, labels: np.ndarray, points: int = 25) -> dict:
    """Accuracy among the most-confident fraction of cases."""
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels
    order = np.argsort(-confidence)

    coverages, accuracies = [], []
    for coverage in np.linspace(0.05, 1.0, points):
        keep = max(1, int(round(coverage * len(labels))))
        selected = order[:keep]
        coverages.append(float(keep / len(labels)))
        accuracies.append(float(correct[selected].mean()))
    return {"coverage": coverages, "accuracy": accuracies}


def per_class_coverage(pred_sets: np.ndarray, labels: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        members = labels == index
        if not members.any():
            result[name] = None
            continue
        result[name] = float(pred_sets[members, index].mean())
    return result


def coverage_sweep(probs_cal, labels_cal, probs_test, labels_test, alphas=None) -> dict:
    """Empirical coverage on the test split as the target coverage varies."""
    if alphas is None:
        alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    empirical, target, sizes = [], [], []
    for alpha in alphas:
        quantiles = class_conditional_quantiles(probs_cal, labels_cal, alpha, len(CLASS_NAMES))
        sets = prediction_sets(probs_test, quantiles)
        covered = sets[np.arange(len(labels_test)), labels_test]
        empirical.append(float(covered.mean()))
        target.append(float(1.0 - alpha))
        sizes.append(float(sets.sum(axis=1).mean()))
    return {"alpha": [float(a) for a in alphas], "empirical": empirical, "target": target, "avg_set_size": sizes}


def set_size_histogram(pred_sets: np.ndarray) -> dict:
    sizes = pred_sets.sum(axis=1)
    counts = {str(size): int((sizes == size).sum()) for size in range(0, len(CLASS_NAMES) + 1)}
    return counts


def save_plots(metrics: dict, prediction_sets_test: np.ndarray, output_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper = "#FBF8F0"
    plt.rcParams.update(
        {
            "figure.facecolor": paper,
            "axes.facecolor": paper,
            "savefig.facecolor": paper,
            "text.color": "#2F3A33",
            "axes.labelcolor": "#6F7568",
            "axes.edgecolor": "#CFC3AE",
            "xtick.color": "#7A8074",
            "ytick.color": "#7A8074",
            "axes.titlecolor": "#24382E",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E7DFD0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )
    primary = "#24523F"
    sage = "#8FA68E"
    reference = "#A79A87"
    os.makedirs(output_dir, exist_ok=True)
    written = []

    def finish(figure, name):
        path = os.path.join(output_dir, name)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
        written.append(path)

    # Reliability diagram
    bins = metrics["classifier"]["reliability"]
    if bins:
        figure, axis = plt.subplots(figsize=(5, 5), dpi=160)
        axis.plot([0, 1], [0, 1], "--", color=reference, lw=1, label="perfect calibration")
        axis.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            "o-",
            color=primary,
            lw=2,
            ms=4,
            label="model",
        )
        axis.set_xlim(0.2, 1.0)
        axis.set_xlabel("predicted confidence")
        axis.set_ylabel("observed accuracy")
        axis.set_title("Reliability diagram (test split)")
        axis.legend(frameon=False, loc="upper left")
        finish(figure, "reliability.png")

    # Risk-coverage curve
    curve = metrics["risk_coverage"]
    figure, axis = plt.subplots(figsize=(5, 4), dpi=160)
    axis.plot(curve["coverage"], curve["accuracy"], color=primary, lw=2)
    axis.set_xlabel("fraction of cases answered")
    axis.set_ylabel("accuracy")
    axis.set_title("Risk-coverage trade-off")
    axis.set_ylim(0, 1.02)
    finish(figure, "risk_coverage.png")

    # Prediction set sizes
    sizes = prediction_sets_test.sum(axis=1)
    figure, axis = plt.subplots(figsize=(5, 4), dpi=160)
    axis.hist(
        sizes,
        bins=np.arange(-0.5, len(CLASS_NAMES) + 1.5, 1),
        color=sage,
        edgecolor=primary,
        linewidth=0.8,
        rwidth=0.72,
    )
    axis.set_xlabel("prediction set size")
    axis.set_ylabel("cases")
    axis.set_title("How often the model commits (test split)")
    finish(figure, "set_sizes.png")

    # Confusion matrix
    matrix = np.asarray(metrics["classifier"]["confusion_matrix_normalized"])
    from matplotlib.colors import LinearSegmentedColormap

    bone = LinearSegmentedColormap.from_list("bone_forest", ["#F7F2E7", "#CBD5C4", "#7E9C83", "#24523F"])
    figure, axis = plt.subplots(figsize=(6, 5), dpi=160)
    image = axis.imshow(matrix, cmap=bone, vmin=0, vmax=1)
    axis.grid(False)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title("Confusion matrix (row-normalised)")
    figure.colorbar(image, ax=axis, fraction=0.046)
    finish(figure, "confusion.png")

    # Coverage guarantee sweep
    sweep = metrics["coverage_sweep"]
    figure, axis = plt.subplots(figsize=(5, 4), dpi=160)
    axis.plot(sweep["target"], sweep["target"], "--", color=reference, lw=1, label="target coverage")
    axis.plot(sweep["target"], sweep["empirical"], "o-", color=primary, lw=2, ms=5, label="empirical coverage")
    axis.set_xlabel("target coverage")
    axis.set_ylabel("empirical coverage")
    axis.set_title("Coverage guarantee holds across targets")
    axis.legend(frameon=False, loc="lower right")
    finish(figure, "coverage_sweep.png")

    return written
