"""Baseline models: LR, RF, XGBoost, vanilla LSTM, GRU.

Each baseline accepts the same (X, mask, y) inputs as the LSTM-Attention
model.  Non-sequential models flatten X into (N, T*F) by concatenating
masked time-steps and zero-padding missing slots.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def _flatten(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Concatenate (N, T, F) into (N, T*F); zero-mask missing cells."""
    Xm = X * mask[..., None].astype(X.dtype)
    return Xm.reshape(Xm.shape[0], -1)


def logistic_regression(
    X: np.ndarray, mask: np.ndarray, y: np.ndarray, **kw
) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs",
        multi_class="multinomial", **kw,
    )
    clf.fit(_flatten(X, mask), y)
    return clf


def random_forest(
    X: np.ndarray, mask: np.ndarray, y: np.ndarray, n_estimators: int = 300, **kw
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=42, **kw,
    )
    clf.fit(_flatten(X, mask), y)
    return clf


def xgboost(
    X: np.ndarray, mask: np.ndarray, y: np.ndarray, n_estimators: int = 500, **kw
):
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost not installed.")
    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
        objective="multi:softprob", num_class=len(np.unique(y)),
        eval_metric="mlogloss", tree_method="hist", n_jobs=-1, random_state=42,
        **kw,
    )
    clf.fit(_flatten(X, mask), y)
    return clf


# ----------------------------------------------------------------------
# Recurrent baselines (vanilla LSTM, GRU) — defined for whichever DL
# backend is available.
# ----------------------------------------------------------------------
def _build_recurrent_keras(cell: str, n_features: int, n_timesteps: int,
                            n_classes: int, units: int = 128, dropout: float = 0.3):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    Cell = layers.LSTM if cell == "lstm" else layers.GRU
    inp = keras.Input(shape=(n_timesteps, n_features))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(Cell(units, dropout=dropout))(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name=f"baseline_{cell}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def vanilla_lstm(*args, **kw):
    return _build_recurrent_keras("lstm", *args, **kw)


def gru(*args, **kw):
    return _build_recurrent_keras("gru", *args, **kw)
