import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import joblib
from src.models.LSTM_model import LSTMModel
from src.utils.data_loader import fetch_data
from src.utils.indicator_engine import add_indicators
from src.alpaca.alpaca_connector import AlpacaOrder

class TradingAgent:
    def __init__(self, ticker, model_path="models/LSTM_model.ptl", scaler_path="models/standard_scaler.pkl"):
        self.ticker = ticker
        self.model = LSTMModel(input_size=9)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.window_size = 78
        self.threshold = 0.001
        self.alpaca = AlpacaOrder() 

    def fetch_and_prepare_data(self):
        df = fetch_data(self.ticker, interval="5m", period="7d")
        df = add_indicators(df)
        df = df.sort_index()
        df.drop(['Date', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)

        if df.shape[0] < self.window_size:
            raise ValueError(f"Not enough data: only {df.shape[0]} rows found.")

        return df[-self.window_size:].copy()

    def predict(self):
        last_78 = self.fetch_and_prepare_data()
        current = last_78['Close'].iloc[-1]
        scaled = self.scaler.transform(last_78)
        recent_scaled_values = scaled.reshape(1, self.window_size, last_78.shape[1])

        with torch.no_grad():
            input_tensor = torch.tensor(recent_scaled_values, dtype=torch.float32)
            output = self.model(input_tensor)
            predicted_scaled = output.item()

        inverse_input = np.zeros((1, last_78.shape[1]))
        close_index = last_78.columns.get_loc("Close")
        inverse_input[0, close_index] = predicted_scaled
        predicted = self.scaler.inverse_transform(inverse_input)[0, close_index]

        return current, predicted

    def act(self, qty=50):
        current, predicted = self.predict()

        if predicted > current * (1 + self.threshold):
            self.alpaca.place_order(symbol=self.ticker, qty=qty, side="buy")
            return "BUY", current, predicted

        elif predicted < current * (1 - self.threshold):
            self.alpaca.place_order(symbol=self.ticker, qty=qty, side="sell")
            return "SELL", current, predicted

        else:
            return "HOLD", current, predicted
