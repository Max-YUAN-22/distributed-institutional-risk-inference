"""Smoke tests for source extractors."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src.extractors import AMSExtractor, IUCPExtractor, LMSExtractor, SASExtractor


def _write(tmp: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp / name
    df.to_csv(p, index=False)
    return p


def test_ams_extractor_minimal(tmp_path: Path):
    p = _write(tmp_path, "ams.csv", pd.DataFrame({
        "student_id": [1, 2, 3], "semester": ["2023F"] * 3,
        "cum_gpa": [3.5, 2.8, 4.5], "credits_earned": [15, 12, 18],
        "scholarship": ["none", "merit", "need"],
    }))
    out = AMSExtractor(p).extract()
    assert {"student_id", "semester", "source"}.issubset(out.columns)
    assert (out["gpa"] <= 4.0).all()


def test_iucp_structural_zero(tmp_path: Path):
    p = _write(tmp_path, "iucp.csv", pd.DataFrame({
        "student_id": [1, 2], "semester": ["2023F", "2023F"],
        "has_internship": [0, 1], "supervisor_rating": [80, 75],
        "project_completions": [0, 2],
    }))
    out = IUCPExtractor(p).extract()
    assert out.loc[out["has_internship"] == 0, "supervisor_rating"].isna().all()


def test_lms_submission_rate_clipped(tmp_path: Path):
    p = _write(tmp_path, "lms.csv", pd.DataFrame({
        "student_id": [1], "semester": ["2023F"],
        "submission_rate": [1.4], "login_count": [40], "forum_posts": [3],
    }))
    out = LMSExtractor(p).extract()
    assert 0.0 <= out["submission_rate"].iloc[0] <= 1.0


def test_sas_attendance_clipped(tmp_path: Path):
    p = _write(tmp_path, "sas.csv", pd.DataFrame({
        "student_id": [1], "semester": ["2023F"],
        "attendance_rate": [-0.2], "gender": ["F"], "disciplinary_events": [0],
    }))
    out = SASExtractor(p).extract()
    assert 0.0 <= out["attendance_rate"].iloc[0] <= 1.0
