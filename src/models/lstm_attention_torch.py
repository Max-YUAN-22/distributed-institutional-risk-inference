"""LSTM-Attention model — PyTorch 1.13 implementation.

Identical architecture to the TensorFlow version:
    BiLSTM(128) → BiLSTM(64) → Additive Attention → Dense(64) → Dense(4)
Total parameters: ≈ 345 K.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for lstm_attention_torch.py.")


class LSTMAttention(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(
        self,
        n_features: int = 35,
        n_timesteps: int = 4,
        n_classes: int = 4,
        lstm_units: Tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")
        super().__init__()
        self.n_timesteps = n_timesteps

        # BiLSTM layer 1: input_size=n_features, hidden=lstm_units[0]
        self.bilstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units[0],
            bidirectional=True,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        # BiLSTM layer 2: input_size = 2*lstm_units[0] (bi-directional)
        self.bilstm2 = nn.LSTM(
            input_size=lstm_units[0] * 2,
            hidden_size=lstm_units[1],
            bidirectional=True,
            batch_first=True,
        )
        self.drop2 = nn.Dropout(dropout)

        # Bahdanau attention score
        h2_size = lstm_units[1] * 2  # bidirectional output
        self.attn_score = nn.Linear(h2_size, 1)

        self.fc1 = nn.Linear(h2_size, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """
        x    : (batch, T, F)
        mask : (batch, T)  1=observed, 0=padding
        Returns logits (batch, n_classes).
        """
        out1, _ = self.bilstm1(x)
        out1 = self.drop1(out1)
        out2, _ = self.bilstm2(out1)
        out2 = self.drop2(out2)

        # additive attention
        score = self.attn_score(out2).squeeze(-1)          # (batch, T)
        score = score.masked_fill(mask == 0, float("-inf"))
        alpha = F.softmax(score, dim=-1)                   # (batch, T)
        context = (alpha.unsqueeze(-1) * out2).sum(dim=1)  # (batch, h2_size)

        x2 = F.relu(self.fc1(context))
        return self.fc2(x2)


def train(
    X_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    mask_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 4,
    class_weight_vec: "np.ndarray | None" = None,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "checkpoints",
) -> "LSTMAttention":
    _require_torch()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[2]
    n_timesteps = X_train.shape[1]

    model = LSTMAttention(n_features=n_features, n_timesteps=n_timesteps,
                          n_classes=n_classes).to(device)

    weight_tensor = None
    if class_weight_vec is not None:
        weight_tensor = torch.tensor(class_weight_vec, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-5
    )

    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)

        # validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32).to(device)
            mv = torch.tensor(mask_val, dtype=torch.float32).to(device)
            yv = torch.tensor(y_val, dtype=torch.long).to(device)
            val_loss = criterion(model(xv, mv), yv).item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_dir / "lstm_attention_best.pt")
        print(
            f"  epoch {epoch:03d}/{epochs}  "
            f"train={epoch_loss/len(ds_train):.4f}  val={val_loss:.4f}"
        )

    if best_state:
        model.load_state_dict(best_state)
    return model
