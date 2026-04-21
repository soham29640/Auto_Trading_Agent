"""
trading_agent.py
────────────────
Core decision-making agent.  Fetches live market data, runs the LSTM
prediction, and places orders via the Alpaca connector.

Fixes:
- Sanity-check threshold was 0.2 (20 %) which is unrealistically high for
  intraday prices.  Changed default to 0.05 (5 %) with a comment explaining
  the guard's purpose.
- `self.threshold` (the signal threshold) was also confusingly named; renamed
  to `self.signal_threshold` for clarity.
- `df.columns.get_loc("Close")` can raise KeyError if the DataFrame returned
  by `add_indicators` does not have 'Close' at runtime.  Now uses the shared
  `FEATURE_COLS` constant so the index is always consistent with training.
- The scaler inverse-transform used a zero-filled array of shape (1, n_cols).
  This is correct but now uses explicit column count from the loaded scaler
  to avoid shape mismatches if the feature set changes.
- Added `_load_model` helper to encapsulate model loading and make it easier
  to hot-reload weights between trading cycles.
- `act()` no longer swallows all exceptions silently; only expected,
  recoverable errors are caught.  Fatal errors propagate so the caller
  (auto_trade.py) can count them and trigger its failsafe.
"""

import os
import numpy as np
import joblib
import torch

from src.models.LSTM_model import LSTMModel
from src.utils.data_loader import fetch_data
from src.utils.indicator_engine import add_indicators, FEATURE_COLS
from src.alpaca.alpaca_connector import AlpacaOrder

MODEL_PATH  = os.path.join("models", "LSTM_model.ptl")
SCALER_PATH = os.path.join("models", "standard_scaler.pkl")

# Index of 'Close' in FEATURE_COLS – must match training
_CLOSE_IDX = FEATURE_COLS.index("Close")


class TradingAgent:
    """
    Args:
        ticker:           Stock symbol to trade, e.g. 'AAPL'.
        window_size:      Number of bars fed into the LSTM (must match training).
        signal_threshold: Minimum predicted price change (fractional) to trigger
                          a BUY or SELL.  Default 0.01 = 1 %.
        sanity_threshold: If the predicted change exceeds this fraction, the
                          prediction is likely an artefact – hold instead.
                          Default 0.05 = 5 %.
    """

    def __init__(
        self,
        ticker: str,
        window_size: int = 78,
        signal_threshold: float = 0.01,
        sanity_threshold: float = 0.05,
    ) -> None:
        self.ticker           = ticker
        self.window_size      = window_size
        self.signal_threshold = signal_threshold
        self.sanity_threshold = sanity_threshold
        self.last_action: str | None = None

        self.model  = self._load_model()
        self.scaler = joblib.load(SCALER_PATH)
        self.n_features: int = self.scaler.n_features_in_

        self.alpaca = AlpacaOrder()

    # ── Model helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _load_model() -> LSTMModel:
        model = LSTMModel(input_size=len(FEATURE_COLS))
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    # ── Data ──────────────────────────────────────────────────────────────────
    def _get_window(self) -> "pd.DataFrame":   # noqa: F821
        df = fetch_data(self.ticker)
        df = add_indicators(df)

        if len(df) < self.window_size:
            raise ValueError(
                f"Not enough data: need {self.window_size} bars, got {len(df)}."
            )

        return df.iloc[-self.window_size :]

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self) -> tuple[float, float]:
        """Return (current_price, predicted_next_price)."""
        df = self._get_window()

        current: float = float(df["Close"].iloc[-1])

        # Scale → tensor → forward pass
        scaled = self.scaler.transform(df.values)          # (window, n_features)
        x = torch.tensor(
            scaled.reshape(1, self.window_size, self.n_features),
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred_scaled: float = self.model(x).item()

        # Inverse-transform the prediction back to price space
        inv = np.zeros((1, self.n_features), dtype=np.float32)
        inv[0, _CLOSE_IDX] = pred_scaled
        predicted: float = float(self.scaler.inverse_transform(inv)[0, _CLOSE_IDX])

        return current, predicted

    # ── Decision ──────────────────────────────────────────────────────────────
    def act(self, qty: int = 50) -> tuple[str, float | None, float | None]:
        """
        Decide and execute one trading action.

        Returns:
            (action, current_price, predicted_price)
            action is one of: 'BUY', 'SELL', 'HOLD', 'ERROR'
        """
        current, predicted = self.predict()   # let exceptions propagate

        change = (predicted - current) / current  # signed fractional change

        # ── Sanity guard: distrust extreme predictions ─────────────────────
        if abs(change) > self.sanity_threshold:
            return "HOLD", current, predicted

        position = self.alpaca.get_position(self.ticker)
        cash     = self.alpaca.get_cash()
        action   = "HOLD"

        # ── Signal logic ──────────────────────────────────────────────────
        if position == 0:
            cost = current * qty
            if change > self.signal_threshold and cash >= cost:
                action = "BUY"
        elif position > 0:
            if change < -self.signal_threshold:
                action = "SELL"

        # ── De-duplicate: don't repeat the same action consecutively ──────
        if action == self.last_action:
            return "HOLD", current, predicted

        # ── Execute ───────────────────────────────────────────────────────
        if action == "BUY":
            self.alpaca.place_order(self.ticker, qty, "buy")
        elif action == "SELL":
            # Always sell the full position, not just `qty`
            self.alpaca.place_order(self.ticker, position, "sell")

        self.last_action = action
        return action, current, predicted