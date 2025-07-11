import yfinance as yf
import pandas as pd

def fetch_data(ticker="AAPL", interval="5m", period="30d"):

    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for ticker: {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    dt_col = df['Datetime'] if 'Datetime' in df.columns else df.index.to_series()
    df['Date'] = dt_col.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata') if dt_col.dt.tz is None else dt_col.dt.tz_convert('Asia/Kolkata')
    df.drop(columns=["Datetime"], inplace=True, errors="ignore")

    return df


