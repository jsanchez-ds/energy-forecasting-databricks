"""
LSTM forecaster (PyTorch) — sequence-to-one load prediction.

Given a fixed-length window of past features (by default 168h = 1 week),
predicts the next hour's load. Runs on CPU, MLflow-logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.forecasting import FEATURE_COLS, TARGET_COL, ForecastMetrics, compute_metrics


@dataclass
class LSTMConfig:
    seq_len: int = 168          # past hours used as input
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 30
    patience: int = 5           # early-stopping patience (epochs without val improvement)
    device: str = "cpu"
    seed: int = 42


class _LSTMRegressor(nn.Module):
    """Sequence-to-one LSTM regressor."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]        # take the last timestep's hidden state
        return self.head(last).squeeze(-1)


def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a (T, F) matrix into (T - seq_len, seq_len, F) sequences."""
    n = X.shape[0] - seq_len
    if n <= 0:
        raise ValueError(f"Not enough rows ({X.shape[0]}) for seq_len={seq_len}")
    seq_X = np.stack([X[i : i + seq_len] for i in range(n)], axis=0)
    seq_y = y[seq_len:]
    return seq_X, seq_y


@dataclass
class LSTMForecaster:
    cfg: LSTMConfig = field(default_factory=LSTMConfig)
    model: _LSTMRegressor | None = None
    scaler_X: StandardScaler | None = None
    scaler_y: StandardScaler | None = None
    history: dict[str, list[float]] = field(default_factory=lambda: {"train": [], "val": []})

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> "LSTMForecaster":
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Build val set from the tail of train if not provided (10%)
        if df_val is None:
            cut = int(len(df_train) * 0.9)
            df_train, df_val = df_train.iloc[:cut], df_train.iloc[cut:]

        X_tr = df_train[FEATURE_COLS].values.astype(np.float32)
        y_tr = df_train[TARGET_COL].values.astype(np.float32)
        X_va = df_val[FEATURE_COLS].values.astype(np.float32)
        y_va = df_val[TARGET_COL].values.astype(np.float32)

        self.scaler_X = StandardScaler().fit(X_tr)
        self.scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr = self.scaler_X.transform(X_tr).astype(np.float32)
        X_va = self.scaler_X.transform(X_va).astype(np.float32)
        y_tr = self.scaler_y.transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32)
        y_va = self.scaler_y.transform(y_va.reshape(-1, 1)).ravel().astype(np.float32)

        Xs_tr, ys_tr = _make_sequences(X_tr, y_tr, self.cfg.seq_len)
        Xs_va, ys_va = _make_sequences(X_va, y_va, self.cfg.seq_len)

        train_ds = TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(ys_tr))
        val_ds = TensorDataset(torch.from_numpy(Xs_va), torch.from_numpy(ys_va))
        train_dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.cfg.batch_size)

        device = torch.device(self.cfg.device)
        self.model = _LSTMRegressor(
            n_features=len(FEATURE_COLS),
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state: dict[str, Any] | None = None
        bad_epochs = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_losses = []
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                train_losses.append(float(loss.item()))

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(float(loss_fn(self.model(xb), yb).item()))

            tr_m = float(np.mean(train_losses))
            va_m = float(np.mean(val_losses))
            self.history["train"].append(tr_m)
            self.history["val"].append(va_m)

            if va_m < best_val - 1e-4:
                best_val = va_m
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.scaler_X is not None and self.scaler_y is not None
        device = torch.device(self.cfg.device)

        X = self.scaler_X.transform(df[FEATURE_COLS].values.astype(np.float32)).astype(np.float32)
        y_dummy = np.zeros(len(X), dtype=np.float32)  # placeholder, not used
        Xs, _ = _make_sequences(X, y_dummy, self.cfg.seq_len)

        self.model.eval()
        preds_scaled: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(Xs), self.cfg.batch_size):
                batch = torch.from_numpy(Xs[i : i + self.cfg.batch_size]).to(device)
                preds_scaled.append(self.model(batch).cpu().numpy())
        y_scaled = np.concatenate(preds_scaled) if preds_scaled else np.array([])
        y = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

        # Pad with NaN for the first `seq_len` rows that have no prediction
        out = np.full(len(df), np.nan)
        out[self.cfg.seq_len :] = y
        return out

    def evaluate(self, df_test: pd.DataFrame) -> ForecastMetrics:
        y_pred_full = self.predict(df_test)
        mask = ~np.isnan(y_pred_full)
        if mask.sum() == 0:
            raise ValueError("All predictions are NaN — test set smaller than seq_len?")
        return compute_metrics(df_test[TARGET_COL].values[mask], y_pred_full[mask])
