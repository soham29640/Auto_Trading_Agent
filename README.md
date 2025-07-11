# 📈 AutoTrade AI

**AutoTrade AI** is a real-time paper trading app that uses a rule-based or ML-based trading agent to simulate stock trades. Built with **Streamlit**, **yfinance**, and **Plotly**, it visualizes live stock data and lets users activate an auto-trading agent that evaluates and logs trades every 5 minutes.

---

## 🚀 Features

- 🔄 Real-time 1-minute candlestick chart (Plotly)
- 🤖 Toggleable Auto-Trading Agent (runs every 5 minutes)
- 📈 Displays current & predicted prices, buy/sell/hold action
- 💼 Dynamic portfolio: cash, holdings, value
- 📝 Logs trades to `logs/trades.csv` for analysis

---

## 🗂️ Project Structure

```
Auto_Trade_AI/
├── app.py                  
├── logs/
│   └── trades.csv             
├── models/ 
│   ├── LSTM_model.ptl
│   └── standard_scaler.pkl              
├── src/
│   ├── agents/
│   │   ├── trading_agent.py
│   │   └── auto_trading_agent.py
│   ├── wallets/
│   │   └── wallet.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   └── indicator_engine.py
|   └── models/
|       ├── load_and_predict_LSTM_model.py
|       ├── LSTM_model.py
|       └── train_save_LSTM_model.py


---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/yourusername/Auto_Trade_AI.git
cd Auto_Trade_AI
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Default Settings

- Ticker: `AAPL`
- Interval: `1m`
- Timezone: `Asia/Kolkata`
- Refresh Rate:  
  - Chart: every 1 minute  
  - Trading Agent: every 5 minutes

---

## 🧪 How It Works

1. App fetches latest stock data via `yfinance`
2. Displays candlestick chart using Plotly
3. User activates trading agent from UI
4. Agent checks every 5 minutes to decide Buy/Sell/Hold
5. Trades are recorded in `logs/trades.csv`
6. Portfolio metrics are updated live on the UI

---

## 📬 Author

**Soham Samanta**  
📧 Feel free to raise issues or contribute on GitHub!
