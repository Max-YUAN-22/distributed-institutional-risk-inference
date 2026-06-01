"""OULAD external-validation runner.

Trains the LSTM-Attention model on OULAD's first-30 % window features
(no industry-university features) and reports headline metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.external_oulad import download_oulad, preprocess
from src.evaluation.metrics import headline_metrics, per_class_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--output_dir", default="outputs/oulad")
    ap.add_argument("--backend",    choices=["tf", "torch"], default="torch")
    ap.add_argument("--epochs",     type=int, default=40)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oulad_path = download_oulad(args.data_dir)
    X, y, ids = preprocess(oulad_path)

    # OULAD is not naturally a time series of fixed T; we treat each
    # student as a single timestep of width-1, padded to T=4 to share
    # the architecture with the institutional pipeline.
    T = 4
    Xt = np.repeat(X[:, None, :], T, axis=1).astype(np.float32)
    mask = np.ones((len(X), T), dtype=np.float32)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(Xt))
    n_train = int(0.8 * len(Xt))
    tr, va = perm[:n_train], perm[n_train:]

    if args.backend == "torch":
        from src.models.lstm_attention_torch import train
        import torch
        model = train(
            Xt[tr], mask[tr], y[tr], Xt[va], mask[va], y[va],
            epochs=args.epochs, checkpoint_dir=out_dir / "checkpoints",
        )
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(Xt[va], dtype=torch.float32),
                torch.tensor(mask[va], dtype=torch.float32),
            )
            y_pred = logits.argmax(-1).cpu().numpy()
            y_prob = torch.softmax(logits, -1).cpu().numpy()
    else:
        from src.models.lstm_attention_tf import train  # type: ignore
        model = train(
            Xt[tr], mask[tr], y[tr], Xt[va], mask[va], y[va],
            epochs=args.epochs, checkpoint_dir=out_dir / "checkpoints",
        )
        y_prob = model.predict([Xt[va], mask[va]])
        y_pred = y_prob.argmax(-1)

    metrics = {
        "headline":  headline_metrics(y[va], y_pred, y_prob),
        "per_class": per_class_metrics(y[va], y_pred),
        "n_students": int(len(y)),
    }
    (out_dir / "oulad_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["headline"], indent=2))


if __name__ == "__main__":
    main()
