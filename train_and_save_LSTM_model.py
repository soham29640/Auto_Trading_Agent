"""
train_and_save_LSTM_model.py
────────────────────────────
Fetches data, computes indicators, trains the LSTM, and saves both the
model weights and the fitted StandardScaler.

Run from the project root:
    python -m src.scripts.train_and_save_LSTM_model
or simply:
    python train_and_save_LSTM_model.py
"""

import os
import sys

# Make the project root importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_loader import fetch_data
from src.utils.indicator_engine import add_indicators, FEATURE_COLS
from src.models.LSTM_model import LSTMModel

# ── Config ────────────────────────────────────────────────────────────────────
TICKER     = "AAPL"
WINDOW     = 78
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
VAL_SPLIT  = 0.1
SAVE_DIR   = "models"
MODEL_PATH  = os.path.join(SAVE_DIR, "LSTM_model.ptl")
SCALER_PATH = os.path.join(SAVE_DIR, "standard_scaler.pkl")

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
print(f"📥  Fetching data for {TICKER} …")
raw = fetch_data(TICKER)           # returns df with Datetime + OHLCV columns
data = add_indicators(raw)         # returns only FEATURE_COLS (9 cols), NaN-free

# add_indicators already returns only FEATURE_COLS – no manual column drop needed
print(f"📊  Dataset shape after indicators: {data.shape}")

# ── Scaling ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
scaled = scaler.fit_transform(data.values)     # use .values → plain ndarray
joblib.dump(scaler, SCALER_PATH)
print(f"💾  Scaler saved → {SCALER_PATH}")

target_idx = FEATURE_COLS.index("Close")       # consistent index = 0

# ── Sliding-window sequences ──────────────────────────────────────────────────
X, y = [], []
for i in range(len(scaled) - WINDOW - 1):
    X.append(scaled[i : i + WINDOW])
    y.append(scaled[i + WINDOW, target_idx])   # scalar target

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y).reshape(-1, 1),  dtype=torch.float32)

# ── Train / val split ─────────────────────────────────────────────────────────
val_size  = max(1, int(len(X) * VAL_SPLIT))
X_train, X_val = X[:-val_size], X[-val_size:]
y_train, y_val = y[:-val_size], y[-val_size:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

print(f"🗂️   Train samples: {len(X_train)} | Val samples: {len(X_val)}")

# ── Training ──────────────────────────────────────────────────────────────────
def train_model() -> None:
    model     = LSTMModel(input_size=len(FEATURE_COLS))
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    # verbose removed in newer PyTorch; use explicit print instead
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── train phase ──
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            # Gradient clipping prevents exploding gradients in LSTMs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── val phase ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb), yb).item()
        val_loss /= len(val_loader)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr  = optimizer.param_groups[0]["lr"]
        lr_tag  = f"  ⬇ lr {prev_lr:.2e}→{new_lr:.2e}" if new_lr != prev_lr else ""

        print(
            f"Epoch {epoch:>3}/{EPOCHS}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}{lr_tag}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best model saved (val={best_val_loss:.6f})")

    print(f"\n🏁  Training complete. Model → {MODEL_PATH}")


if __name__ == "__main__":
    train_model()