import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from src.utils.data_loader import fetch_data
from src.utils.indicator_engine import add_indicators

data = fetch_data("AAPL")
data = add_indicators(data)
data.drop(['Date', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)

save_path = os.path.join("models", "LSTM_model.ptl")

scaler = StandardScaler()
scaled = scaler.fit_transform(data.dropna())
joblib.dump(scaler,"models/standard_scaler.pkl")

target_index = data.columns.get_loc("Close")
window = 78
x, y = [], []
for i in range(len(scaled) - window - 1):
    x.append(scaled[i:i + window])
    y.append(scaled[i + window][target_index])

x = torch.tensor(np.array(x), dtype=torch.float32)
y = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)

from src.models.LSTM_model import LSTMModel

def train_model(x, y):
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(50):
        epoch_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/50 - Loss: {avg_loss:.6f}")


    torch.save(model.state_dict(), save_path)
    print(f"Trained model is saved to {save_path}")

train_model(x, y)
