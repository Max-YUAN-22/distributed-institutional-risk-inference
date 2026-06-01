"""EdNet-KT1 external-validation utilities.

EdNet-KT1 is a knowledge-tracing dataset of student-question interaction
sequences from a Korean tutoring platform.  We adopt the same
early-window protocol as §2.6: the first half of each student's
interaction sequence is used for feature extraction, and a four-level
risk label is derived from the empirical correctness distribution in
the late-window holdout.

Data source: https://github.com/riiid/ednet
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


EDNET_KT1_URL = "https://github.com/riiid/ednet/releases/download/v1.0/KT1.zip"
EARLY_WINDOW_FRAC = 0.50


GENERIC_FEATURES_13 = [
    # behavioural-attendance proxies (6)
    "interaction_count", "active_days", "session_count",
    "session_length_mean", "elapsed_time_mean",
    "elapsed_time_std",
    # academic-performance proxies (7)
    "correct_rate", "correct_streak_max", "incorrect_streak_max",
    "unique_questions", "unique_tags", "explanation_request_rate",
    "explanation_request_count",
]


def download_ednet(target_dir: str | Path) -> Path:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "KT1.zip"
    if not zip_path.exists():
        print(f"  downloading EdNet-KT1 → {zip_path}")
        urllib.request.urlretrieve(EDNET_KT1_URL, zip_path)
    extracted = target / "KT1"
    if not extracted.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    return target / "KT1"


def _label_from_correctness(rate: float, sample_size: int) -> int:
    """Map late-window correctness rate to a 4-level warning."""
    if sample_size < 5:
        return 0
    if rate >= 0.75:
        return 0  # Normal
    if rate >= 0.55:
        return 1  # Blue
    if rate >= 0.35:
        return 2  # Yellow
    return 3      # Red


def _per_user_features(df: pd.DataFrame) -> dict:
    """Compute the 13 generic features for a single user's KT1 log."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_early = int(n * EARLY_WINDOW_FRAC)
    early = df.iloc[:n_early]
    late = df.iloc[n_early:]

    if len(early) < 5 or len(late) < 5:
        return {}

    # session boundaries: >30-minute gap == new session
    gaps = early["timestamp"].diff().fillna(0)
    session_starts = (gaps > 30 * 60 * 1000).cumsum()
    sess_len = early.groupby(session_starts).size()

    streaks = (early["correct"] == 1).astype(int).values
    correct_streak_max, incorrect_streak_max, cur_c, cur_i = 0, 0, 0, 0
    for s in streaks:
        if s == 1:
            cur_c, cur_i = cur_c + 1, 0
            correct_streak_max = max(correct_streak_max, cur_c)
        else:
            cur_i, cur_c = cur_i + 1, 0
            incorrect_streak_max = max(incorrect_streak_max, cur_i)

    feats = {
        "interaction_count":           int(len(early)),
        "active_days":                 int(early["timestamp"].apply(
            lambda t: t // (86400 * 1000)).nunique()),
        "session_count":               int(session_starts.nunique()),
        "session_length_mean":         float(sess_len.mean()),
        "elapsed_time_mean":           float(early["elapsed_time"].mean()),
        "elapsed_time_std":            float(early["elapsed_time"].std() or 0.0),
        "correct_rate":                float(early["correct"].mean()),
        "correct_streak_max":          int(correct_streak_max),
        "incorrect_streak_max":        int(incorrect_streak_max),
        "unique_questions":            int(early["question_id"].nunique()),
        "unique_tags":                 int(early["tag_id"].nunique()) if
                                       "tag_id" in early.columns else 0,
        "explanation_request_rate":    float(early.get("explanation",
                                                pd.Series([0])).mean()),
        "explanation_request_count":   int(early.get("explanation",
                                                pd.Series([0])).sum()),
    }
    feats["_label"] = _label_from_correctness(
        rate=float(late["correct"].mean()), sample_size=len(late),
    )
    return feats


def preprocess(ednet_dir: str | Path, max_users: int = 12_347
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream-process up to max_users sequence files into (X, y, user_ids)."""
    ednet_dir = Path(ednet_dir)
    user_files = sorted(ednet_dir.glob("u*.csv"))[:max_users]

    rows, labels, ids = [], [], []
    for i, fp in enumerate(user_files):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if "correct" not in df.columns or "timestamp" not in df.columns:
            continue
        feats = _per_user_features(df)
        if not feats:
            continue
        labels.append(feats.pop("_label"))
        rows.append(feats)
        ids.append(fp.stem)

        if (i + 1) % 1000 == 0:
            print(f"    processed {i + 1} users")

    X_df = pd.DataFrame(rows)[GENERIC_FEATURES_13].fillna(0.0)
    X = X_df.to_numpy(dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, np.array(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_dir", required=True)
    args = ap.parse_args()
    download_ednet(args.download_dir)


if __name__ == "__main__":
    main()
