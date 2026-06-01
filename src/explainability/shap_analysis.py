"""SHAP explanations for the LSTM-Attention model.

Implements the source-ablation and feature-attribution analyses
reported in §4.4 of the manuscript.  Uses ``shap.DeepExplainer`` for
the TensorFlow build of the model and falls back to ``KernelExplainer``
when the deep API is not available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _require_shap():
    if not _SHAP_AVAILABLE:
        raise ImportError("`shap` is required for shap_analysis.py.")


def explain_lstm_attention(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str] | None = None,
    output_dir: str | Path = "shap_out",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for an LSTM-Attention model.

    Returns
    -------
    shap_values  : (N, T, F, K) for K classes
    feature_imp  : (F,) — mean |SHAP| collapsed across timesteps & samples
    """
    _require_shap()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        explainer = shap.DeepExplainer(model, X_background[:200])
        shap_values = explainer.shap_values(X_explain[:500])
    except Exception:  # fall back to KernelExplainer on a flattened model
        def _flat_predict(x_flat: np.ndarray) -> np.ndarray:
            x = x_flat.reshape((-1, X_background.shape[1], X_background.shape[2]))
            mask = (x.sum(-1) != 0).astype(np.float32)
            return model.predict([x, mask])

        explainer = shap.KernelExplainer(
            _flat_predict,
            X_background.reshape(X_background.shape[0], -1)[:100],
        )
        shap_values = explainer.shap_values(
            X_explain.reshape(X_explain.shape[0], -1)[:100], nsamples=200,
        )

    shap_arr = np.array(shap_values)
    feature_imp = np.abs(shap_arr).mean(axis=(0, 1, 2))[: X_explain.shape[2]]
    np.save(out_dir / "shap_values.npy", shap_arr)
    np.save(out_dir / "feature_importance.npy", feature_imp)
    if feature_names is not None:
        ranked = sorted(
            zip(feature_names, feature_imp), key=lambda t: -t[1],
        )
        with (out_dir / "feature_importance_ranked.txt").open("w") as f:
            for name, val in ranked:
                f.write(f"{val:.6f}\t{name}\n")
    return shap_arr, feature_imp


def source_ablation_importance(
    feature_imp: np.ndarray,
    feature_to_source: list[str],
) -> dict[str, float]:
    """Aggregate per-feature SHAP magnitudes by source system.

    Used to derive the 46.6 % industry-university attribution
    reported in §4.4.
    """
    sources = sorted(set(feature_to_source))
    totals = {s: 0.0 for s in sources}
    for imp, src in zip(feature_imp, feature_to_source):
        totals[src] += float(imp)
    grand = sum(totals.values()) or 1.0
    return {s: t / grand for s, t in totals.items()}
