import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import joblib
from src.models.LSTM_model import LSTMModel
from src.utils.data_loader import fetch_data
from src.utils.indicator_engine import add_indicators

model = LSTMModel(input_size=9)
model.load_state_dict(torch.load("models/LSTM_model.ptl"))
model.eval()

today_weekday = datetime.now().weekday()
today_date = datetime.now().date()
days_to_fetch = 3 if today_weekday in [0, 6] else 2

df = fetch_data("AAPL", days=days_to_fetch, interval="5m")
df = add_indicators(df)
df.drop(['High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)
df['Date'] = df.index.date
unique_dates = sorted(df['Date'].unique())

if today_date in unique_dates:
    today_data = df[df['Date'] == today_date]
else:
    today_data = pd.DataFrame(columns=df.columns)

previous_day = unique_dates[-2] if today_date in unique_dates else unique_dates[-1]
prev_data = df[df['Date'] == previous_day]

if today_data.empty:
    last_78 = prev_data[-78:]
elif today_data.shape[0] >= 78:
    last_78 = today_data[-78:]
else:
    needed = 78 - today_data.shape[0]
    last_78 = pd.concat([prev_data[-needed:], today_data])

if last_78.shape[0] < 78:
    pad_rows = 78 - last_78.shape[0]
    padding = last_78.iloc[[0]].copy()
    padding = pd.concat([padding] * pad_rows)
    last_78 = pd.concat([padding, last_78])

features = last_78.drop(columns=["Date"])
scaler = joblib.load("models/standard_scaler.pkl")
scaled = scaler.transform(features)
recent_scaled_values = scaled.reshape(1, 78, features.shape[1])

with torch.no_grad():
    input_tensor = torch.tensor(recent_scaled_values, dtype=torch.float32)
    prediction = model(input_tensor)
    predicted_value = prediction.item()

inverse_input = np.zeros((1, features.shape[1]))
inverse_input[0, features.columns.get_loc("Close")] = predicted_value
predicted_close_real = scaler.inverse_transform(inverse_input)[0, features.columns.get_loc("Close")]

current_close = features['Close'].iloc[-1]
threshold = 0.001

if predicted_close_real > current_close * (1 + threshold):
    decision = "BUY"
elif predicted_close_real < current_close * (1 - threshold):
    decision = "SELL"
else:
    decision = "HOLD"

print(f"Current Price: {current_close:.2f}")
print(f"Predicted Next Close: {predicted_close_real:.2f}")
print(f"Suggested Action: {decision}")
