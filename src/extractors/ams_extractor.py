"""AMS extractor — Academic Management System (Phase 1).

Columns extracted (canonical names):
  student_id, semester, gpa, failed_count, weighted_theory_avg,
  weighted_practice_avg, scholarship_flag, major_code
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .base import BaseExtractor


class AMSExtractor(BaseExtractor):
    source_name = "ams"

    # AMS column mapping:  AMS raw name → canonical name
    _COLUMN_MAP = {
        "stu_id":            "student_id",
        "term":              "semester",
        "cum_gpa":           "gpa",
        "failed_courses":    "failed_count",
        "theory_avg":        "weighted_theory_avg",
        "practice_avg":      "weighted_practice_avg",
        "scholarship":       "scholarship_flag",
        "major":             "major_code",
    }

    def __init__(
        self,
        db_conn_str: str = "",
        csv_fallback: str | Path | None = None,
        output_dir: str | Path = "shards",
        config: Dict[str, Any] | None = None,
    ):
        super().__init__(output_dir, config)
        self.db_conn_str = db_conn_str
        self.csv_fallback = Path(csv_fallback) if csv_fallback else None

    def _raw_fetch(self) -> pd.DataFrame:
        if self.csv_fallback and self.csv_fallback.exists():
            return pd.read_csv(self.csv_fallback)
        # real deployment: SQL query against the AMS Oracle/MySQL database
        raise FileNotFoundError(
            "No csv_fallback provided and db_conn_str not configured. "
            "This extractor requires access to the private institutional AMS."
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self._COLUMN_MAP)
        # encode scholarship as binary flag
        if "scholarship_flag" in df.columns and df["scholarship_flag"].dtype == object:
            df["scholarship_flag"] = (
                df["scholarship_flag"].str.lower().isin({"yes", "1", "true"})
            ).astype(int)
        # clamp GPA to [0, 4.0] and fill missing with cohort median
        if "gpa" in df.columns:
            df["gpa"] = df["gpa"].clip(lower=0.0, upper=4.0)
            df["gpa"] = df["gpa"].fillna(df["gpa"].median())
        df["source"] = self.source_name
        return df[
            ["student_id", "semester", "source", "gpa", "failed_count",
             "weighted_theory_avg", "weighted_practice_avg",
             "scholarship_flag", "major_code"]
            + [c for c in df.columns
               if c not in {"student_id","semester","source","gpa","failed_count",
                             "weighted_theory_avg","weighted_practice_avg",
                             "scholarship_flag","major_code"}]
        ]
