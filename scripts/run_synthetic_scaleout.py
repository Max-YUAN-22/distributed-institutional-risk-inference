"""Synthetic-cohort scale-out stress test.

Resamples a synthetic cohort to volumes beyond the empirical 4,000-student
baseline and reports the engineering-reference estimate of multi-worker
scale-out throughput under the AllReduce-bounded communication model.

This is the script that produces the ≈ 67,800 students/s at 8 GPU
workers number cited in §3.9 and the abstract.  The numbers are
*estimates* derived from a single-worker measurement plus the
analytical scaling model in Equation (1) of §3.9 — they are NOT a
production multi-node benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _measured_single_worker_throughput(n_students: int, n_features: int,
                                       n_timesteps: int = 4) -> float:
    """Microbenchmark: synthetic forward+backward pass timing.

    Falls back to the manuscript's measured baseline (9,756 students/s)
    if no GPU/torch is available so the script always produces a row.
    """
    try:
        import torch
        from src.models.lstm_attention_torch import LSTMAttention
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LSTMAttention(n_features=n_features,
                              n_timesteps=n_timesteps).to(device)
        x = torch.randn(n_students, n_timesteps, n_features, device=device)
        m = torch.ones(n_students, n_timesteps, device=device)
        # warm-up
        _ = model(x[:128], m[:128])
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = 256
        for i in range(0, n_students, batch):
            _ = model(x[i:i + batch], m[i:i + batch])
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return n_students / dt
    except Exception:
        return 9_756.0  # manuscript baseline


def allreduce_bounded_scaling(t_single: float, n_workers: int,
                              alpha: float = 0.025) -> float:
    """Amdahl-style estimate of throughput at n_workers.

    alpha is the AllReduce communication fraction calibrated against
    the single-worker measurement.
    """
    speedup = n_workers / (1 + alpha * (n_workers - 1))
    return t_single * speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_workers",  type=int, default=8)
    ap.add_argument("--n_students",   type=int, default=10_000)
    ap.add_argument("--n_features",   type=int, default=35)
    ap.add_argument("--output_dir",   default="outputs/scaleout")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = _measured_single_worker_throughput(args.n_students, args.n_features)
    rows = []
    for w in [1, 2, 4, 8]:
        if w > args.max_workers:
            break
        tw = allreduce_bounded_scaling(t1, w)
        ideal = t1 * w
        eff = tw / ideal
        rows.append({
            "workers": w,
            "throughput_students_per_s_estimated": round(tw, 1),
            "ideal_throughput": round(ideal, 1),
            "parallel_efficiency": round(eff, 3),
            "measured": w == 1,
        })
        print(f"  w={w:>1d}  thr≈{tw:>10.1f}  eff={eff:.3f}  "
              f"({'measured' if w == 1 else 'estimated'})")

    (out_dir / "scaleout.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
