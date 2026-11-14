"""Class-conditional split conformal prediction for multi-class image triage.

Nonconformity score
-------------------
For a held-out point (x, y) we use the "one minus probability" score

    s(x, y) = 1 - p_hat(y | x)

Calibration
-----------
For every class c we collect the scores of calibration points with label c and
take the ceil((n_c + 1)(1 - alpha)) / n_c empirical quantile, capped at 1.0.
This is the standard finite-sample correction from Angelopoulos & Bates (2023),
applied per class (Mondrian / class-conditional conformal prediction).

Prediction set
--------------
    C(x) = { c : 1 - p_hat(c | x) <= q_c }

which is equivalent to thresholding each class probability at 1 - q_c.

Guarantees
----------
With exchangeable calibration and test data, P(Y in C(X)) >= 1 - alpha on
average, and for each class with enough calibration samples the per-class
marginal coverage is also >= 1 - alpha. Class-conditional calibration matters
here: melanoma is rare, and a marginal (pooled) threshold would systematically
under-cover it.
"""

from __future__ import annotations

import numpy as np


def nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return 1 - p_hat(y | x) for each row."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    return 1.0 - probs[np.arange(len(labels)), labels]


def class_conditional_quantiles(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    n_classes: int,
) -> np.ndarray:
    """Per-class conformal quantiles of the nonconformity score."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    scores = nonconformity_scores(probs, labels)
    labels = np.asarray(labels, dtype=int)
    quantiles = np.full(n_classes, 1.0, dtype=np.float64)

    for c in range(n_classes):
        class_scores = np.sort(scores[labels == c])
        n_c = class_scores.size
        if n_c == 0:
            quantiles[c] = 1.0
            continue

        level = np.ceil((n_c + 1) * (1.0 - alpha)) / n_c
        if level >= 1.0:
            quantiles[c] = float(class_scores[-1])
            continue

        # "higher" gives the ceil((n+1)(1-alpha))-th smallest score.
        quantiles[c] = float(np.quantile(class_scores, level, method="higher"))

    return quantiles


def prediction_sets(probs: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Boolean matrix [n_samples, n_classes]; True means the class is included."""
    probs = np.asarray(probs, dtype=np.float64)
    return probs >= (1.0 - quantiles[None, :])


def set_statistics(pred_sets: np.ndarray, labels: np.ndarray) -> dict:
    """Coverage, set-size and abstention diagnostics for a batch."""
    pred_sets = np.asarray(pred_sets, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    sizes = pred_sets.sum(axis=1)
    covered = pred_sets[np.arange(len(labels)), labels]

    singleton = sizes == 1
    empty = sizes == 0
    abstained = sizes >= 2

    singleton_correct = np.zeros(len(labels), dtype=bool)
    if singleton.any():
        predicted = pred_sets[singleton].argmax(axis=1)
        singleton_correct[singleton] = predicted == labels[singleton]

    return {
        "n": int(len(labels)),
        "coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float(singleton.mean()),
        "empty_set_rate": float(empty.mean()),
        "abstention_rate": float(abstained.mean()),
        "selective_accuracy": (
            float(singleton_correct.sum() / singleton.sum()) if singleton.any() else None
        ),
        "coverage_gap": float(covered.mean() - 0.0),
    }
