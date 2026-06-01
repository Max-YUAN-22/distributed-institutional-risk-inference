"""End-to-end institutional-pipeline runner.

Runs the five-phase distributed pipeline on synthetic institutional
data (or, if available, the private institutional CSVs) and trains
the LSTM-Attention classifier.

Usage
-----
    python scripts/run_institutional_pipeline.py \
        --data_dir data/synthetic --backend torch \
        --output_dir outputs/institutional
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.extractors import AMSExtractor, IUCPExtractor, LMSExtractor, SASExtractor
from src.fusion import StreamMerger, TensorConstructor, ManifestManager
from src.evaluation.metrics import headline_metrics, per_class_metrics


def _run_extractors(data_dir: Path):
    extractors = {
        "ams":  AMSExtractor(data_dir / "ams.csv"),
        "iucp": IUCPExtractor(data_dir / "iucp.csv"),
        "lms":  LMSExtractor(data_dir / "lms.csv"),
        "sas":  SASExtractor(data_dir / "sas.csv"),
    }
    return {name: ext.extract() for name, ext in extractors.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--output_dir", default="outputs/institutional")
    ap.add_argument("--backend",    choices=["tf", "torch"], default="torch")
    ap.add_argument("--epochs",     type=int, default=60)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestManager(out_dir / "manifest.json")

    print("[1/5] per-source extraction")
    sources = _run_extractors(data_dir)

    print("[2/5] partition-parallel stream merge")
    merged = StreamMerger(n_partitions=8).merge(sources)

    print("[3/5] tensor construction")
    X, mask, y, ids = TensorConstructor().build(merged)
    print(f"      X={X.shape}  mask={mask.shape}  y={y.shape}")

    print("[4/5] manifest-gated artefact writes")
    manifest.write_artefact("X", out_dir / "X.npy",
                            lambda p: np.save(p, X))
    manifest.write_artefact("mask", out_dir / "mask.npy",
                            lambda p: np.save(p, mask))
    manifest.write_artefact("y", out_dir / "y.npy",
                            lambda p: np.save(p, y))
    manifest.write_artefact("ids", out_dir / "ids.npy",
                            lambda p: np.save(p, ids))

    print(f"[5/5] training ({args.backend})")
    # train/val split
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    tr, va = perm[:n_train], perm[n_train:]

    if args.backend == "torch":
        from src.models.lstm_attention_torch import train
    else:
        from src.models.lstm_attention_tf import train  # type: ignore
    model = train(
        X[tr], mask[tr], y[tr], X[va], mask[va], y[va],
        epochs=args.epochs, checkpoint_dir=out_dir / "checkpoints",
    )

    # inference on val
    if args.backend == "torch":
        import torch
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(X[va], dtype=torch.float32),
                torch.tensor(mask[va], dtype=torch.float32),
            )
            y_pred = logits.argmax(-1).cpu().numpy()
            y_prob = torch.softmax(logits, -1).cpu().numpy()
    else:
        prob = model.predict([X[va], mask[va]])
        y_pred = prob.argmax(-1)
        y_prob = prob

    metrics = headline_metrics(y[va], y_pred, y_prob)
    pc = per_class_metrics(y[va], y_pred)
    print("    headline:", metrics)
    (out_dir / "metrics.json").write_text(
        json.dumps({"headline": metrics, "per_class": pc}, indent=2)
    )
    print(f"    metrics → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
