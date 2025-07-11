import os
import sys
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.data_loader import fetch_data
from src.agents.trading_agent import TradingAgent
from src.agents.auto_trading_agent import AutoTradingAgent
from src.wallets.wallet import Wallet

wallet = Wallet(100000)
ta = TradingAgent("AAPL", wallet)
ata = AutoTradingAgent(wallet, ta)

st.set_page_config(page_title="AutoTrade AI", layout="wide")
st_autorefresh(interval=60000, key="graph_autorefresh")
st.title("📈 Auto Trade AI - Live Stock Agent")

def fetch_data(ticker="AAPL", interval="1m", period="1d"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df.empty:
        df = yf.download(ticker, interval="5m", period="5d", progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for ticker: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    df = df.reset_index()
    dt_col = df['Datetime'] if 'Datetime' in df.columns else df.index.to_series()
    df['Date'] = dt_col.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata') if dt_col.dt.tz is None else dt_col.dt.tz_convert('Asia/Kolkata')
    return df

def plot_candlestick(df):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'
        )
    ])
    fig.update_layout(
        title="Live Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        yaxis=dict(side="right"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=500
    )
    return fig

if "trader" not in st.session_state:
    st.session_state.trader = Wallet(starting_cash=100000)
trader = st.session_state.trader

if "toggle" not in st.session_state:
    st.session_state.toggle = False

if "last_trade_minute" not in st.session_state:
    st.session_state.last_trade_minute = -5

try:
    df = fetch_data("AAPL")
    if df.empty:
        st.error("⚠️ Not enough data to display chart.")
    else:
        current_price = df['Close'].iloc[-1]
        portfolio = trader.status(current_price)

        st.markdown(f"""
            <style>
                #portfolio-box {{
                    position: fixed;
                    top: 4rem;
                    right: 8rem;
                    width: 190px;
                    padding: 12px;
                    background: rgba(40, 40, 40, 0.95);
                    color: white;
                    border-radius: 12px;
                    font-size: 13px;
                    z-index: 9999;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                }}
                #portfolio-box h4 {{
                    font-size: 14px;
                    margin: 0 0 10px 0;
                    text-align: center;
                }}
                #portfolio-box div {{
                    margin-bottom: 6px;
                }}
            </style>
            <div id="portfolio-box">
                <h4>💼 Portfolio</h4>
                <div>💰 <strong>Cash:</strong> ${portfolio['cash']:.2f}</div>
                <div>📦 <strong>Holdings:</strong> {portfolio['holdings']}</div>
                <div>💵 <strong>Price:</strong> ${current_price:.2f}</div>
                <div>📈 <strong>Value:</strong> ${float(portfolio['portfolio_value']):.2f}</div>
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_candlestick(df), use_container_width=True)
        st.markdown("---")
        st.subheader("📊 Live Trading Panel")

        col1, col2 = st.columns([2, 6])
        with col1:
            if st.session_state.toggle:
                if st.button("🔴 Deactivate Auto Trading Agent"):
                    st.session_state.toggle = False
            else:
                if st.button("🟢 Activate Auto Trading Agent"):
                    st.session_state.toggle = True

        if st.session_state.toggle:
            st.success("🤖 Auto Trading Agent is Active (checks every 5 mins)")
            current_minute = datetime.now().minute
            current_window = current_minute // 5
            last_window = st.session_state.last_trade_minute // 5

            if current_window != last_window:
                st.session_state.last_trade_minute = current_minute
                st.success("🟢 Trading Window Open")
                result = ata.run_once()
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    col1, = st.columns(1)
                    col1.metric("💡 Action", result['action'])

                    col2, col3, col4, col5 = st.columns(4)
                    col2.metric("📉 Current", f"${result['current']}")
                    col3.metric("📈 Predicted", f"${result['predicted']}")
                    col4.metric("💰 Cash", f"${result['cash']}")
                    col5.metric("📦 Holdings", result['holdings'])

                    st.caption(f"⏱️ Last check: {result['time']} | 💼 Portfolio Value: ${result['portfolio_value']}")
            else:
                st.info("⏳ Auto Trading Agent will act in the next 5-minute window.")
        else:
            st.info("⚪ Auto Trading Agent is OFF")

except Exception as e:
    st.error(f"🚫 Error loading data or chart: {e}")

