"""Temporal tensor constructor — Phase 3 of the distributed pipeline.

Transforms the long-format merged DataFrame into the dense (N, T, F) tensor
expected by the LSTM-Attention model, together with a (N, T) mask tensor
indicating observed timesteps.  Idempotent: re-running on the same input
produces a byte-identical output, which combined with the manifest manager
gives crash-safe Phase-3 execution.
"""
from __future__ import annotations
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


class TensorConstructor:
    """Build (X, mask, y) tensors from a merged long-format DataFrame."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        n_semesters: int = 4,
        label_column: str = "warning_level",
        student_id_col: str = "student_id",
        semester_col: str = "semester",
    ):
        self.feature_columns = list(feature_columns)
        self.n_semesters = n_semesters
        self.label_column = label_column
        self.student_id_col = student_id_col
        self.semester_col = semester_col

    def build(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, mask, y, student_ids).

        X        : (N, T, F)  float32
        mask     : (N, T)     uint8   1 = observed, 0 = missing
        y        : (N,)       int64   latest-semester warning level
        student_ids : (N,)    object  the row-aligned student identifiers
        """
        df = df.sort_values([self.student_id_col, self.semester_col])
        # impute numeric NaNs with cohort-median *per column* so missing
        # semester-slots are filled deterministically.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
            med = df[col].median()
            df[col] = df[col].fillna(med)

        student_ids = df[self.student_id_col].unique()
        N, T, F = len(student_ids), self.n_semesters, len(self.feature_columns)
        X = np.zeros((N, T, F), dtype=np.float32)
        mask = np.zeros((N, T), dtype=np.uint8)
        y = np.zeros(N, dtype=np.int64)

        sid_to_idx = {s: i for i, s in enumerate(student_ids)}
        for (_, row) in df.iterrows():
            i = sid_to_idx[row[self.student_id_col]]
            t = int(row[self.semester_col]) - 1
            if 0 <= t < T:
                X[i, t, :] = row[self.feature_columns].to_numpy(dtype=np.float32)
                mask[i, t] = 1
                if self.label_column in row and not pd.isna(row[self.label_column]):
                    y[i] = int(row[self.label_column])

        print(
            f"  [TENSOR] X={X.shape}  mask={mask.shape}  y={y.shape}"
            f"  observed-cell ratio={mask.mean():.3f}"
        )
        return X, mask, y, student_ids
