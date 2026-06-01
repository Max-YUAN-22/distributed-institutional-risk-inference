"""Generate the manuscript's figures from cached metrics JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _bar_headline_comparison(institutional: dict, oulad: dict, ednet: dict,
                             out_path: Path) -> None:
    metrics = ["accuracy", "macro_f1", "cohen_kappa"]
    labels = ["Institutional", "OULAD", "EdNet-KT1"]
    values = np.array([
        [institutional.get(m, np.nan) for m in metrics],
        [oulad.get(m, np.nan) for m in metrics],
        [ednet.get(m, np.nan) for m in metrics],
    ])
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for i, (lab, row) in enumerate(zip(labels, values)):
        ax.bar(x + (i - 1) * width, row, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Headline metrics across cohorts")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _scaleout_curve(scaleout_rows: list[dict], out_path: Path) -> None:
    if not scaleout_rows:
        return
    workers = [r["workers"] for r in scaleout_rows]
    thr     = [r["throughput_students_per_s_estimated"] for r in scaleout_rows]
    ideal   = [r["ideal_throughput"] for r in scaleout_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, ideal, "k--", label="ideal linear")
    ax.plot(workers, thr,   "o-",  label="AllReduce-bounded estimate")
    ax.set_xlabel("# workers")
    ax.set_ylabel("throughput (students/s)")
    ax.set_title("Synthetic-cohort scale-out stress test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst_metrics",     default="outputs/institutional/metrics.json")
    ap.add_argument("--oulad_metrics",    default="outputs/oulad/oulad_metrics.json")
    ap.add_argument("--ednet_metrics",    default="outputs/ednet/ednet_metrics.json")
    ap.add_argument("--scaleout_metrics", default="outputs/scaleout/scaleout.json")
    ap.add_argument("--output_dir",       default="outputs/figures")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst  = _load_json(Path(args.inst_metrics)).get("headline", {})
    oulad = _load_json(Path(args.oulad_metrics)).get("headline", {})
    ednet = _load_json(Path(args.ednet_metrics)).get("headline", {})
    scale = _load_json(Path(args.scaleout_metrics)) or []

    _bar_headline_comparison(inst, oulad, ednet, out_dir / "headline.png")
    _scaleout_curve(scale, out_dir / "scaleout.png")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
