"""Synthetic institutional-data generator.

Produces a 4-source, 4-semester synthetic cohort that follows the same
schema as the private institutional dataset described in the manuscript.
Designed exclusively for reproducibility of the *pipeline*, not the
predictive results — distributions are coarse and uncalibrated.

Usage
-----
    python synthetic_data/generate_synthetic_institutional_data.py \
        --n_students 4000 --output_dir data/synthetic/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEMESTERS = ["2022F", "2023S", "2023F", "2024S"]


def _gen_ams(rng, n_students, sem):
    """Academic Management System: GPA, credits, scholarship."""
    return pd.DataFrame({
        "student_id": np.arange(n_students),
        "semester":   sem,
        "cum_gpa":    np.clip(rng.normal(3.0, 0.5, n_students), 0, 4),
        "credits_earned": rng.poisson(15, n_students),
        "scholarship": rng.choice(["none", "merit", "need"],
                                  n_students, p=[0.7, 0.2, 0.1]),
    })


def _gen_iucp(rng, n_students, sem):
    """Industry-University Cooperation Platform."""
    has_int = rng.binomial(1, 0.55, n_students)
    return pd.DataFrame({
        "student_id": np.arange(n_students),
        "semester":   sem,
        "has_internship":   has_int,
        "supervisor_rating": np.where(
            has_int, rng.normal(75, 12, n_students), np.nan,
        ),
        "project_completions": rng.poisson(0.8, n_students),
    })


def _gen_lms(rng, n_students, sem):
    """Learning Management System: submission rate, login frequency."""
    return pd.DataFrame({
        "student_id": np.arange(n_students),
        "semester":   sem,
        "submission_rate": np.clip(rng.beta(5, 2, n_students), 0, 1),
        "login_count":     rng.poisson(40, n_students),
        "forum_posts":     rng.poisson(3, n_students),
    })


def _gen_sas(rng, n_students, sem):
    """Student-Affairs System: attendance, demographics, behaviour."""
    return pd.DataFrame({
        "student_id": np.arange(n_students),
        "semester":   sem,
        "attendance_rate":  np.clip(rng.beta(8, 2, n_students), 0, 1),
        "gender":           rng.choice(["M", "F"], n_students),
        "disciplinary_events": rng.poisson(0.05, n_students),
    })


def generate(n_students: int, output_dir: str | Path, seed: int = 42) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    sources = {"ams": _gen_ams, "iucp": _gen_iucp,
               "lms": _gen_lms, "sas": _gen_sas}
    for name, fn in sources.items():
        frames = [fn(rng, n_students, sem) for sem in SEMESTERS]
        df = pd.concat(frames, ignore_index=True)
        # introduce 5 % random missingness in non-id columns
        non_id = [c for c in df.columns if c not in ("student_id", "semester")]
        mask = rng.random(df[non_id].shape) < 0.05
        df.loc[:, non_id] = df[non_id].mask(mask)
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"  wrote {name}.csv  ({len(df):>6,} rows)")

    print(f"Done. {n_students} students × {len(SEMESTERS)} semesters × 4 sources.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_students", type=int, default=4000)
    ap.add_argument("--output_dir", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.n_students, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
