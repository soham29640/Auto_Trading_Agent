# 📈 AutoTrade Agent

**AutoTrade Agent** is an end-to-end AI-powered paper/live trading system. It trains an LSTM neural network on historical stock data, deploys that model inside a real-time decision loop, and executes orders through the **Alpaca** brokerage API. A **Streamlit** dashboard lets you start, stop, and monitor the bot without touching the terminal.

---

## 🚀 Features

| Feature | Details |
|---|---|
| 🤖 LSTM price prediction | Two-layer LSTM trained on 9 technical-indicator features |
| 📊 Technical indicators | SMA 5/15, RSI 14, MACD, Bollinger Bands (via `ta` library) |
| 🔄 Configurable trade loop | Adjustable interval (30 s / 60 s / 120 s) |
| 🏦 Alpaca integration | Supports both **alpaca-py** (new SDK) and **alpaca-trade-api** (legacy) |
| 📄 Paper & Live modes | Switch between paper and real-money trading from the UI |
| 🖥️ Streamlit dashboard | Real-time log terminal, start/stop controls, credential management |
| 📝 Persistent trade log | Decisions written to `trade_log.txt`, auto-trimmed at 1 000 lines |

---

## 🗂️ Project Structure

```
Auto_Trading_Agent/
│
├── app.py                          # Streamlit web dashboard (entry point)
├── auto_trade.py                   # Headless trading loop (can run standalone)
├── train_and_save_LSTM_model.py    # One-shot script: fetch → train → save model
├── requirements.txt                # Python dependencies
├── trade_log.txt                   # Runtime trade log (auto-created)
│
├── models/                         # Persisted artefacts (created by training)
│   ├── LSTM_model.ptl              # Trained LSTM weights (PyTorch state-dict)
│   └── standard_scaler.pkl         # Fitted StandardScaler (joblib)
│
└── src/
    ├── agents/
    │   └── trading_agent.py        # Core decision engine (predict → decide → execute)
    ├── utils/
    │   ├── data_loader.py          # yfinance wrapper – fetches OHLCV data
    │   └── indicator_engine.py     # Computes 9 technical indicators, defines FEATURE_COLS
    ├── models/
    │   └── LSTM_model.py           # PyTorch LSTM architecture definition
    └── alpaca/
        ├── alpaca_connector.py     # REST wrapper – place orders, query cash/positions
        └── alpaca_stream.py        # WebSocket listener for real-time trade-update events
```

---

## 🛠️ Setup Instructions

### 1 — Clone & install

```bash
git clone https://github.com/soham29640/Auto_Trading_Agent.git
cd Auto_Trading_Agent
pip install -r requirements.txt
```

### 2 — Train the LSTM model

Run this **once** before launching the bot. It downloads historical data, computes indicators, trains for 50 epochs, and saves the model and scaler to `models/`.

```bash
python train_and_save_LSTM_model.py
```

### 3 — Launch the dashboard

```bash
streamlit run app.py
```

Open the Streamlit URL, paste your **Alpaca API Key + Secret** in the sidebar, choose Paper or Live mode, pick a ticker/qty/interval, then press **▶️ Start**.

---

## 🔄 Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE  (train_and_save_LSTM_model.py  –  run once offline)      │
│                                                                              │
│  Yahoo Finance (yfinance)                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  data_loader.fetch_data()  →  raw OHLCV DataFrame (7-day, 5-min bars)       │
│       │                                                                      │
│       ▼                                                                      │
│  indicator_engine.add_indicators()  →  9-column feature DataFrame           │
│       │   (Close, SMA_5, SMA_15, RSI, MACD, MACD_Signal,                    │
│       │    MACD_Diff, BB_High, BB_Low)                                       │
│       ▼                                                                      │
│  StandardScaler.fit_transform()  →  scaled numpy array                      │
│       │                                                                      │
│       ├──► scaler  saved to  models/standard_scaler.pkl                     │
│       │                                                                      │
│       ▼                                                                      │
│  Sliding-window sequences  (window = 78 bars)                                │
│       │   X shape: (N, 78, 9)   y shape: (N, 1)  [next Close, scaled]       │
│       ▼                                                                      │
│  LSTMModel.train()  –  50 epochs, Adam + ReduceLROnPlateau, grad clip 1.0   │
│       │                                                                      │
│       └──► best weights  saved to  models/LSTM_model.ptl                    │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  LIVE TRADING PIPELINE  (app.py  →  background thread  →  trading_agent)    │
│                                                                              │
│  User (Streamlit UI)                                                         │
│       │  enters ticker / qty / interval, saves Alpaca credentials            │
│       │  presses ▶️ Start                                                    │
│       ▼                                                                      │
│  app.py  spawns daemon thread  →  run_agent()                                │
│       │                                                                      │
│       ▼  every N seconds (configurable)                                      │
│  TradingAgent.act()                                                          │
│       │                                                                      │
│       ├─ 1. fetch_data(ticker)         Yahoo Finance – last 7 days, 5-min   │
│       │                                                                      │
│       ├─ 2. add_indicators(df)         compute 9-feature matrix              │
│       │                                                                      │
│       ├─ 3. scaler.transform()         normalise using saved scaler          │
│       │                                                                      │
│       ├─ 4. LSTMModel.forward()        predict next scaled Close price       │
│       │                                                                      │
│       ├─ 5. scaler.inverse_transform() convert prediction back to USD        │
│       │                                                                      │
│       ├─ 6. Decision logic                                                   │
│       │       change = (predicted − current) / current                       │
│       │       if |change| > sanity_threshold (5 %)  →  HOLD                 │
│       │       elif no position and change > signal_threshold (1 %)           │
│       │            and cash ≥ cost                  →  BUY                  │
│       │       elif position > 0 and change < −signal_threshold  →  SELL     │
│       │       else                                  →  HOLD                 │
│       │       (consecutive duplicate actions are suppressed)                 │
│       │                                                                      │
│       ├─ 7. AlpacaOrder.place_order()  submit market order to Alpaca REST   │
│       │                                                                      │
│       └─ 8. log()  →  trade_log.txt  +  UI terminal via Queue               │
│                                                                              │
│  app.py  drains Queue every 1 second and updates the live terminal           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Step-by-Step Execution

### Phase 1 — Model Training (`train_and_save_LSTM_model.py`)

| Step | File / Function | What happens |
|------|----------------|--------------|
| 1 | `data_loader.fetch_data("AAPL")` | Downloads 7 days of 5-minute OHLCV bars from Yahoo Finance. Flattens MultiIndex columns, resets the DatetimeIndex, and validates required columns. |
| 2 | `indicator_engine.add_indicators(df)` | Computes SMA 5, SMA 15, RSI 14, MACD, MACD Signal, MACD Diff, Bollinger High and Low. Drops NaN rows and returns a clean `(N, 9)` DataFrame. |
| 3 | `StandardScaler.fit_transform()` | Normalises all 9 features to zero mean and unit variance. The fitted scaler is saved to `models/standard_scaler.pkl` via `joblib`. |
| 4 | Sliding-window loop | Creates overlapping sequences of 78 consecutive bars. The target `y` is the scaled `Close` value of the bar immediately after each window. |
| 5 | `LSTMModel` training | 2-layer LSTM (hidden=64, dropout=0.2) + linear head, trained for 50 epochs. `Adam` optimiser with `ReduceLROnPlateau` scheduler and gradient clipping (`max_norm=1.0`). Best weights (lowest val loss) are saved to `models/LSTM_model.ptl`. |

### Phase 2 — Dashboard Startup (`app.py`)

| Step | What happens |
|------|-------------|
| 1 | `streamlit run app.py` starts the Streamlit server and renders the dashboard. |
| 2 | `_init_state()` initialises all session-state keys (running flag, log list, Queue, credentials). |
| 3 | The sidebar collects Alpaca API Key, Secret, and trading mode (Paper / Live). Pressing **Save Credentials** writes them to environment variables (`APCA_API_KEY_ID`, etc.). |
| 4 | The main area exposes Ticker, Qty, and Interval controls. These are locked while the bot is running. |
| 5 | Pressing **▶️ Start** spawns a daemon `threading.Thread` that runs `run_agent()`. A `threading.Event` is stored in session state for clean shutdown. |
| 6 | The main loop sleeps 1 second then calls `st.rerun()`. Each rerun drains the shared `Queue` and appends new log lines to the live terminal. |

### Phase 3 — Live Trading Loop (background thread)

| Step | File / Function | What happens |
|------|----------------|--------------|
| 1 | `TradingAgent.__init__()` | Loads `LSTMModel` weights and `StandardScaler` from disk. Initialises `AlpacaOrder`, which authenticates with Alpaca and prints available cash. |
| 2 | `TradingAgent._get_window()` | Calls `fetch_data()` → `add_indicators()` and slices the last 78 rows to form the inference window. Raises if fewer than 78 bars are available. |
| 3 | `TradingAgent.predict()` | Scales the 78-bar window, feeds it through the LSTM as a `(1, 78, 9)` tensor, then inverse-transforms the scalar output back to USD. Returns `(current_price, predicted_price)`. |
| 4 | `TradingAgent.act()` | Computes the fractional price change. Applies the sanity guard (rejects predictions with > 5 % swing). Queries Alpaca for current position and cash. Applies signal logic to output BUY / SELL / HOLD. Suppresses consecutive duplicate actions. |
| 5 | `AlpacaOrder.place_order()` | Submits a GTC market order via `alpaca-py` (new SDK) or `alpaca-trade-api` (legacy), transparently switching between the two at import time. |
| 6 | `log()` | Timestamps the decision, writes it to `trade_log.txt` (auto-trimmed at 1 000 lines), and pushes the message onto the shared Queue for the UI. |
| 7 | Interval sleep | Sleeps for the configured interval in 0.25-second increments so that pressing **⏹ Stop** is responsive at any time. |

### Phase 4 — Trade Stream (`src/alpaca/alpaca_stream.py`, optional)

Run `python src/alpaca/alpaca_stream.py` separately to subscribe to Alpaca's WebSocket feed and receive real-time fill / order-update events.

---

## ⚙️ Configuration Reference

| Parameter | Default | Where to change |
|-----------|---------|-----------------|
| Ticker | `AAPL` | Streamlit UI or `auto_trade.py` `main()` |
| Order quantity | `5` | Streamlit UI |
| Trade interval | `60 s` | Streamlit UI (30 / 60 / 120 s) |
| LSTM window size | 78 bars | `trading_agent.py` → `window_size` |
| Signal threshold | 1 % | `trading_agent.py` → `signal_threshold` |
| Sanity threshold | 5 % | `trading_agent.py` → `sanity_threshold` |
| Training epochs | 50 | `train_and_save_LSTM_model.py` → `EPOCHS` |
| Training ticker | `AAPL` | `train_and_save_LSTM_model.py` → `TICKER` |
| Log max lines | 1 000 | `auto_trade.py` → `LOG_MAXLINES` |

---

## 🔑 Alpaca API Keys

1. Sign up at [alpaca.markets](https://alpaca.markets).
2. Open the **Paper Trading** dashboard and generate an API Key + Secret.
3. Paste them into the Streamlit sidebar and press **Save Credentials**.  
   Keys are stored **only in the browser session** and are never persisted to disk.

Alternatively, create a `.env` file in the project root for the headless mode (`python auto_trade.py`):

```env
APCA_API_KEY_ID=PKxxxxxxxxxxxxxxxx
APCA_API_SECRET_KEY=your_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard |
| `yfinance` | Historical market data |
| `pandas` / `numpy` | Data manipulation |
| `ta` | Technical indicators |
| `torch` | LSTM model (PyTorch) |
| `scikit-learn` | StandardScaler |
| `joblib` | Scaler serialisation |
| `alpaca-py` / `alpaca-trade-api` | Brokerage API (new / legacy SDK) |
| `python-dotenv` | `.env` file loading |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📬 Author

**Soham Samanta**  
Feel free to raise issues or contribute on GitHub!
