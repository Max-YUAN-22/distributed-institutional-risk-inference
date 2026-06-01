"""Smoke tests for evaluation metrics."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    headline_metrics, per_class_metrics, confusion, early_warning_lead,
)


def test_headline_metrics_perfect():
    y = np.array([0, 1, 2, 3, 0, 1])
    m = headline_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["cohen_kappa"] == 1.0


def test_per_class_keys():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    pc = per_class_metrics(y_true, y_pred)
    assert set(pc.keys()) == {"Normal", "Blue", "Yellow", "Red"}


def test_confusion_shape():
    y = np.array([0, 1, 2, 3])
    assert confusion(y, y).shape == (4, 4)


def test_early_warning_lead_basic():
    preds = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
    gt    = np.array([3, 2])
    out = early_warning_lead(preds, gt)
    assert out["n_evaluated"] == 2
    assert out["mean_lead_semesters"] == 1.0
