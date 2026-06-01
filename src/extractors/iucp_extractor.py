"""IUCP extractor — Industry-University Cooperation Platform (Phase 1).

Columns extracted:
  student_id, semester, internship_hours, supervisor_rating,
  certificate_count, competition_count, has_internship
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .base import BaseExtractor


class IUCPExtractor(BaseExtractor):
    source_name = "iucp"

    _COLUMN_MAP = {
        "stu_id":           "student_id",
        "term":             "semester",
        "intern_hours":     "internship_hours",
        "sup_rating":       "supervisor_rating",
        "cert_num":         "certificate_count",
        "comp_num":         "competition_count",
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
            "IUCP extractor requires access to the private institutional IUCP."
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={k: v for k, v in self._COLUMN_MAP.items() if k in df})
        # structural-zero guard (Section 4.8 of manuscript):
        # supervisor_rating is structurally 0 for students without internships
        # — encode as two separate variables to avoid imputation artefacts.
        if "internship_hours" in df.columns and "supervisor_rating" in df.columns:
            df["has_internship"] = (df["internship_hours"] > 0).astype(int)
            df["supervisor_rating"] = df["supervisor_rating"].where(
                df["has_internship"] == 1, other=0.0
            )
        df["source"] = self.source_name
        return df
