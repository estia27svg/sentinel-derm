"""Sanity checks for the conformal layer using synthetic probabilities.

Run: python -m pytest tests/ -q
"""

import numpy as np

from training.conformal import (
    class_conditional_quantiles,
    prediction_sets,
    set_statistics,
)


def _synthetic(n=6000, k=7, seed=0, class_skew=(0.5, 0.2, 0.12, 0.08, 0.05, 0.04, 0.01)):
    rng = np.random.default_rng(seed)
    labels = rng.choice(k, size=n, p=np.asarray(class_skew) / np.sum(class_skew))
    # Signal strength varies per sample: the model is confident on "easy" rows.
    confidence = rng.uniform(0.15, 0.95, size=n)
    logits = rng.normal(0, 0.6, size=(n, k))
    logits[np.arange(n), labels] += 4.0 * confidence
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, labels


def test_coverage_is_close_to_target():
    probs, labels = _synthetic(seed=1)
    n_cal = len(labels) // 2
    q = class_conditional_quantiles(probs[:n_cal], labels[:n_cal], alpha=0.1, n_classes=7)
    sets = prediction_sets(probs[n_cal:], q)
    stats = set_statistics(sets, labels[n_cal:])
    assert 0.87 <= stats["coverage"] <= 0.93, stats


def test_sets_are_never_smaller_than_one_on_average():
    probs, labels = _synthetic(seed=2)
    n_cal = len(labels) // 2
    q = class_conditional_quantiles(probs[:n_cal], labels[:n_cal], alpha=0.1, n_classes=7)
    sets = prediction_sets(probs[n_cal:], q)
    assert sets.sum(axis=1).mean() >= 1.0


def test_alpha_near_zero_achieves_near_full_coverage():
    probs, labels = _synthetic(n=2000, seed=3)
    n_cal = len(labels) // 2
    q = class_conditional_quantiles(probs[:n_cal], labels[:n_cal], alpha=1e-9, n_classes=7)
    sets = prediction_sets(probs[n_cal:], q)
    # With a 1 - p score, alpha -> 0 guarantees coverage, not that every class
    # enters every set: the quantile saturates at the largest calibration score.
    assert set_statistics(sets, labels[n_cal:])["coverage"] >= 0.99


def test_singleton_decisions_are_more_accurate_than_all_decisions():
    probs, labels = _synthetic(seed=4)
    n_cal = len(labels) // 2
    q = class_conditional_quantiles(probs[:n_cal], labels[:n_cal], alpha=0.1, n_classes=7)
    sets = prediction_sets(probs[n_cal:], q)
    stats = set_statistics(sets, labels[n_cal:])
    argmax_acc = float((probs[n_cal:].argmax(axis=1) == labels[n_cal:]).mean())
    assert stats["selective_accuracy"] >= argmax_acc
