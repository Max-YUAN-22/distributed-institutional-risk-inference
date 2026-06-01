"""SAS extractor — Student Affairs System (Phase 1).

Columns extracted:
  student_id, semester, attendance_rate, absence_count,
  gender, age, hometown_tier, family_income_band
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .base import BaseExtractor


class SASExtractor(BaseExtractor):
    source_name = "sas"

    _COLUMN_MAP = {
        "stu_id":         "student_id",
        "term":           "semester",
        "attend_rate":    "attendance_rate",
        "absent_days":    "absence_count",
        "sex":            "gender",
        "age_enrol":      "age",
        "hometown":       "hometown_tier",
        "income_band":    "family_income_band",
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
            "SAS extractor requires access to the private institutional SAS."
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={k: v for k, v in self._COLUMN_MAP.items() if k in df})
        if "attendance_rate" in df.columns:
            df["attendance_rate"] = df["attendance_rate"].clip(0.0, 1.0)
        if "gender" in df.columns and df["gender"].dtype == object:
            df["gender"] = df["gender"].str.lower().map(
                {"male": 0, "female": 1, "m": 0, "f": 1}
            ).fillna(-1).astype(int)
        df["source"] = self.source_name
        return df
