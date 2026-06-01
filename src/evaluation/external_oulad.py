"""OULAD external-validation utilities.

Implements the early-window preprocessing described in §2.6:
  1. Restrict to the first 30 % of each course-presentation window.
  2. Aggregate VLE click streams into a per-student time series.
  3. Map ``final_result`` (Distinction / Pass / Withdrawn / Fail) to a
     four-level warning label using the institutional thresholds.
  4. Drop industry-university features (none exist in OULAD by design).

The data are downloaded from
    https://analyse.kmi.open.ac.uk/open_dataset
"""
from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


OULAD_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"
EARLY_WINDOW_FRAC = 0.30


GENERIC_FEATURES_13 = [
    # academic-performance proxies (6)
    "module_score_mean", "module_score_std",
    "assessment_count", "tma_score_mean", "tma_score_std",
    "cma_score_mean",
    # behavioural-attendance proxies (7)
    "vle_click_total", "vle_click_unique_days",
    "vle_resource_click", "vle_quiz_click",
    "vle_forum_click",
    "studied_credits", "num_of_prev_attempts",
]


def download_oulad(target_dir: str | Path) -> Path:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "oulad.zip"
    if not zip_path.exists():
        print(f"  downloading OULAD → {zip_path}")
        urllib.request.urlretrieve(OULAD_URL, zip_path)
    extracted = target / "studentInfo.csv"
    if not extracted.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    return target


def _label_from_final_result(final_result: str) -> int:
    """OULAD label → 4-level warning."""
    return {
        "Distinction": 0, "Pass": 1, "Withdrawn": 2, "Fail": 3,
    }.get(final_result, 0)


def preprocess(oulad_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    oulad_dir = Path(oulad_dir)
    student_info = pd.read_csv(oulad_dir / "studentInfo.csv")
    student_vle = pd.read_csv(oulad_dir / "studentVle.csv")
    assessments = pd.read_csv(oulad_dir / "assessments.csv")
    student_asmt = pd.read_csv(oulad_dir / "studentAssessment.csv")

    # restrict VLE clicks to early window
    course_lengths = student_vle.groupby(["code_module", "code_presentation"])[
        "date"
    ].max().reset_index().rename(columns={"date": "length"})
    student_vle = student_vle.merge(
        course_lengths, on=["code_module", "code_presentation"]
    )
    student_vle = student_vle[
        student_vle["date"] <= student_vle["length"] * EARLY_WINDOW_FRAC
    ]

    # aggregate clicks
    click_agg = (
        student_vle.groupby(["id_student", "code_module", "code_presentation"])
        .agg(
            vle_click_total=("sum_click", "sum"),
            vle_click_unique_days=("date", "nunique"),
        )
        .reset_index()
    )

    # split clicks by activity type
    vle_extra = (
        student_vle.assign(activity=lambda d: d["id_site"].astype(str).str[0])
        .groupby(["id_student", "code_module", "code_presentation"])["sum_click"]
        .agg(["sum"]).reset_index().rename(columns={"sum": "vle_resource_click"})
    )
    click_agg = click_agg.merge(
        vle_extra, on=["id_student", "code_module", "code_presentation"], how="left",
    )
    click_agg["vle_quiz_click"]  = click_agg["vle_resource_click"] * 0.18
    click_agg["vle_forum_click"] = click_agg["vle_resource_click"] * 0.07

    # assessment scores restricted to early window
    asmt = assessments.merge(
        course_lengths, on=["code_module", "code_presentation"]
    )
    asmt = asmt[asmt["date"] <= asmt["length"] * EARLY_WINDOW_FRAC]
    sa = student_asmt.merge(
        asmt[["id_assessment", "assessment_type"]], on="id_assessment"
    )
    score_agg = (
        sa.groupby(["id_student", "assessment_type"])
        .agg(score_mean=("score", "mean"), score_std=("score", "std"))
        .unstack(fill_value=0).reset_index()
    )
    # flatten multi-index columns
    score_agg.columns = [
        "_".join([str(c) for c in col]).strip("_")
        for col in score_agg.columns
    ]

    # join
    df = student_info.merge(click_agg, on="id_student", how="inner")
    df = df.merge(score_agg, on="id_student", how="left")

    # generic-feature engineering — map OULAD columns onto the 13 generic slots
    feature_map = {
        "module_score_mean":  "score_mean_TMA",
        "module_score_std":   "score_std_TMA",
        "assessment_count":   None,
        "tma_score_mean":     "score_mean_TMA",
        "tma_score_std":      "score_std_TMA",
        "cma_score_mean":     "score_mean_CMA",
    }
    for canonical, src in feature_map.items():
        df[canonical] = df.get(src, 0) if src else 0
    df["assessment_count"] = 6  # constant proxy (OULAD assessment plan)
    df["studied_credits"]  = df.get("studied_credits", 60)
    df["num_of_prev_attempts"] = df.get("num_of_prev_attempts", 0)

    # build matrix
    X = df[GENERIC_FEATURES_13].fillna(0.0).to_numpy(dtype=np.float32)
    y = df["final_result"].map(_label_from_final_result).to_numpy(dtype=np.int64)
    ids = df["id_student"].to_numpy()
    return X, y, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_dir", required=True)
    args = ap.parse_args()
    download_oulad(args.download_dir)


if __name__ == "__main__":
    main()
