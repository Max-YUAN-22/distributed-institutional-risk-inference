"""LSTM-Attention model — TensorFlow 2.10 implementation.

Architecture (matching the manuscript description):
    BiLSTM(128 units) → dropout(0.3)
  → BiLSTM(64 units)  → dropout(0.3)
  → Bahdanau additive attention over T timesteps
  → Dense(64, relu) → Dense(4, softmax)

Total trainable parameters: ≈ 345 K  (checkpoint ≈ 4.7 MB).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# TF is an optional dependency — import lazily so the repo remains importable
# even if only the PyTorch backend is installed.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


def _require_tf():
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for lstm_attention_tf.py.")


def build_model(
    n_timesteps: int = 4,
    n_features: int = 35,
    n_classes: int = 4,
    lstm_units: Tuple[int, int] = (128, 64),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    class_weights: dict | None = None,
) -> "keras.Model":
    _require_tf()

    inputs = keras.Input(shape=(n_timesteps, n_features), name="sequence_input")
    mask_input = keras.Input(shape=(n_timesteps,), dtype=tf.float32, name="mask_input")

    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True, dropout=dropout)
    )(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[1], return_sequences=True, dropout=dropout)
    )(x)
    x = layers.Dropout(dropout)(x)

    # Bahdanau additive attention
    score = layers.Dense(1, activation="tanh", name="attention_score")(x)
    score = tf.squeeze(score, axis=-1)  # (batch, T)
    # apply mask to scores before softmax
    score = score + (1.0 - mask_input) * (-1e9)
    alpha = tf.nn.softmax(score, axis=-1, name="attention_weights")  # (batch, T)
    context = tf.einsum("bt,btf->bf", alpha, x, name="context_vector")

    x = layers.Dense(64, activation="relu")(context)
    outputs = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(
        inputs=[inputs, mask_input], outputs=outputs, name="LSTM_Attention"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    mask_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 4,
    class_weight_map: dict | None = None,
    epochs: int = 60,
    batch_size: int = 256,
    checkpoint_dir: str | Path = "checkpoints",
    learning_rate: float = 1e-3,
) -> "keras.Model":
    _require_tf()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    model = build_model(
        n_timesteps=n_timesteps,
        n_features=n_features,
        n_classes=n_classes,
        learning_rate=learning_rate,
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(ckpt_dir / "lstm_attention_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    model.fit(
        [X_train, mask_train],
        y_train,
        validation_data=([X_val, mask_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_map,
        callbacks=callbacks,
        verbose=1,
    )
    return model
