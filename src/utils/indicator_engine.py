import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def add_indicators(df):
    df = df.copy()
    df['SMA_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator().values.reshape(-1)
    df['SMA_15'] = SMAIndicator(close=df['Close'], window=15).sma_indicator().values.reshape(-1)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi().values.reshape(-1)
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd().values.reshape(-1)
    df['MACD_Signal'] = macd.macd_signal().values.reshape(-1)
    df['MACD_Diff'] = macd.macd_diff().values.reshape(-1)
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband().values.reshape(-1)
    df['BB_Low'] = bb.bollinger_lband().values.reshape(-1)
    df.dropna(inplace=True)
    return df
