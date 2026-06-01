"""Smoke tests for fusion modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion import StreamMerger, TensorConstructor


def _toy_source(n_students=4, sem="2023F", source="ams"):
    return pd.DataFrame({
        "student_id": np.arange(n_students),
        "semester":   [sem] * n_students,
        "source":     [source] * n_students,
        "feature_a":  np.arange(n_students, dtype=float),
    })


def test_stream_merger_outer_join():
    merged = StreamMerger(n_partitions=2).merge({
        "ams": _toy_source(source="ams"),
        "lms": _toy_source(source="lms").rename(columns={"feature_a": "feature_b"}),
    })
    assert {"student_id", "semester"}.issubset(merged.columns)
    assert len(merged) == 4


def test_tensor_constructor_shapes():
    df = _toy_source(n_students=6).assign(label=[0, 1, 2, 3, 0, 1])
    df = pd.concat([df, _toy_source(n_students=6, sem="2024S").assign(
        label=[0, 1, 2, 3, 0, 1]
    )], ignore_index=True)
    X, mask, y, ids = TensorConstructor(n_timesteps=2).build(df)
    assert X.shape[0] == 6 and X.shape[1] == 2
    assert mask.shape == (6, 2)
    assert y.shape == (6,)
