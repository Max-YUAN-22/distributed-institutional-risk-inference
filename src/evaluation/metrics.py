"""Evaluation metrics used throughout the manuscript."""
from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)


CLASS_NAMES: List[str] = ["Normal", "Blue", "Yellow", "Red"]


def headline_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: "np.ndarray | None" = None) -> Dict[str, float]:
    """Return the manuscript's headline metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred,
                                                weights="quadratic")),
    }
    if y_prob is not None:
        try:
            metrics["auc_ovr_macro"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            ))
        except ValueError:
            metrics["auc_ovr_macro"] = float("nan")
    return metrics


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray
                       ) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0,
    )
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        out[name] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred,
                            labels=list(range(len(CLASS_NAMES))))


def early_warning_lead(predicted_per_semester: np.ndarray,
                       actual_warning_semester: np.ndarray) -> Dict[str, float]:
    """Mean lead time between first non-Normal prediction and ground truth."""
    leads = []
    for pred_seq, gt_sem in zip(predicted_per_semester, actual_warning_semester):
        first_alert = np.argmax(pred_seq > 0) if (pred_seq > 0).any() else -1
        if first_alert >= 0 and gt_sem >= 0:
            leads.append(gt_sem - first_alert)
    leads = np.array(leads, dtype=float)
    return {
        "mean_lead_semesters": float(leads.mean()) if leads.size else 0.0,
        "median_lead_semesters": float(np.median(leads)) if leads.size else 0.0,
        "n_evaluated": int(leads.size),
    }
