"""LMS extractor — Learning Management System (Phase 1).

Columns extracted:
  student_id, semester, login_count, time_on_task_hrs,
  submission_rate, late_submission_rate, online_learning_hrs
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .base import BaseExtractor


class LMSExtractor(BaseExtractor):
    source_name = "lms"

    _COLUMN_MAP = {
        "stu_id":            "student_id",
        "term":              "semester",
        "logins":            "login_count",
        "task_hours":        "time_on_task_hrs",
        "submit_rate":       "submission_rate",
        "late_rate":         "late_submission_rate",
        "online_hrs":        "online_learning_hrs",
    }

    def __init__(
        self,
        csv_fallback: str | Path | None = None,
        output_dir: str | Path = "shards",
        config: Dict[str, Any] | None = None,
    ):
        super().__init__(output_dir, config)
        self.csv_fallback = Path(csv_fallback) if csv_fallback else None

    def _raw_fetch(self) -> pd.DataFrame:
        if self.csv_fallback and self.csv_fallback.exists():
            return pd.read_csv(self.csv_fallback)
        raise FileNotFoundError(
            "LMS extractor requires access to the private institutional LMS."
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={k: v for k, v in self._COLUMN_MAP.items() if k in df})
        # cap submission_rate at [0,1]
        for col in ("submission_rate", "late_submission_rate"):
            if col in df.columns:
                df[col] = df[col].clip(0.0, 1.0)
        df["source"] = self.source_name
        return df
