"""Out-of-distribution detection in embedding space.

A dermatology triage model that quietly classifies a photo of a kitchen counter
as "melanoma" is worse than useless. We therefore score every input by its
Mahalanobis distance to the nearest class-conditional embedding centre, fit on
the training split, and reject inputs whose score exceeds a threshold calibrated
to keep the in-distribution false-rejection rate at a fixed level.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


def fit_mahalanobis(embeddings: np.ndarray, labels: np.ndarray, n_classes: int):
    """Fit class centres and a shared, shrunk precision matrix."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    means = np.zeros((n_classes, embeddings.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        members = embeddings[labels == c]
        if len(members) == 0:
            raise ValueError(f"class {c} has no training examples")
        means[c] = members.mean(axis=0)

    centered = embeddings - means[labels]
    covariance = LedoitWolf().fit(centered)
    return {
        "means": means,
        "precision": covariance.precision_,
        "shrinkage": float(covariance.shrinkage_),
    }


def mahalanobis_scores(
    embeddings: np.ndarray, means: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """Minimum squared Mahalanobis distance to any class centre."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    distances = np.empty((len(embeddings), len(means)), dtype=np.float64)
    for index, centre in enumerate(means):
        delta = embeddings - centre
        distances[:, index] = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return distances.min(axis=1)


def threshold_at_percentile(scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(scores, percentile))


def evaluate(negative_scores: np.ndarray, positive_scores: np.ndarray) -> dict:
    """AUROC of the OOD score, treating negatives as the positive class."""
    labels = np.concatenate([np.ones(len(negative_scores)), np.zeros(len(positive_scores))])
    scores = np.concatenate([negative_scores, positive_scores])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "n_negatives": int(len(negative_scores)),
        "n_positives": int(len(positive_scores)),
        "negative_median": float(np.median(negative_scores)),
        "positive_median": float(np.median(positive_scores)),
    }
