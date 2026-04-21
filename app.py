"""
app.py  –  Streamlit dashboard for the Trading Bot
────────────────────────────────────────────────────
Run:
    streamlit run app.py

Users paste their Alpaca API key + secret directly in the sidebar.
No .env file required.
"""

import os
import threading
import time
from datetime import datetime
from queue import Empty, Queue

import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_LOG_LINES    = 500
DISPLAY_LINES    = 40
REFRESH_INTERVAL = 1.0  # seconds between auto-refreshes


# ── Session-state initialisation ─────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "running":    False,
        "logs":       [],
        "log_queue":  Queue(),
        "stop_event": None,
        "api_key":    "",
        "api_secret": "",
        "base_url":   "https://paper-api.alpaca.markets",
        "keys_saved": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ── Background thread ─────────────────────────────────────────────────────────
def run_agent(
    ticker: str,
    qty: int,
    interval: int,
    stop_event: threading.Event,
    log_q: Queue,
) -> None:
    """Run inside a daemon thread; pushes log dicts onto *log_q*."""
    from src.agents.trading_agent import TradingAgent

    def push(msg: str) -> None:
        log_q.put({"type": "log", "msg": msg})

    push(f"🚀  Agent started for {ticker} (qty={qty}, interval={interval}s)")

    try:
        agent = TradingAgent(ticker)

        while not stop_event.is_set():
            now = datetime.now().strftime("%H:%M:%S")
            try:
                action, current, predicted = agent.act(qty=qty)
                if current is not None and predicted is not None:
                    msg = f"[{now}] {action} | ${current:.2f} → ${predicted:.2f}"
                else:
                    msg = f"[{now}] {action}"
            except Exception as exc:
                msg = f"[{now}] ❌ {exc}"

            push(msg)

            # Sleep in short increments so stop_event is honoured quickly
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline and not stop_event.is_set():
                time.sleep(0.25)

    except Exception as exc:
        push(f"💥 Fatal thread error: {exc}")
    finally:
        push("🛑  Agent thread exited.")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trading Bot", layout="wide", page_icon="📈")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  –  Alpaca credentials
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔑 Alpaca Credentials")
    st.caption(
        "Get your keys from [alpaca.markets](https://alpaca.markets) → "
        "Paper Trading dashboard."
    )

    api_key = st.text_input(
        "API Key ID",
        value=st.session_state.api_key,
        placeholder="PKXXXXXXXXXXXXXXXX",
        disabled=st.session_state.running,
        type="default",
    )

    api_secret = st.text_input(
        "API Secret Key",
        value=st.session_state.api_secret,
        placeholder="your secret key",
        disabled=st.session_state.running,
        type="password",
    )

    mode = st.radio(
        "Trading Mode",
        ["📄 Paper (safe)", "💰 Live (real money)"],
        disabled=st.session_state.running,
    )
    base_url = (
        "https://paper-api.alpaca.markets"
        if "Paper" in mode
        else "https://api.alpaca.markets"
    )

    if st.button("💾 Save Credentials", disabled=st.session_state.running, use_container_width=True):
        if not api_key.strip() or not api_secret.strip():
            st.error("Both fields are required.")
        else:
            os.environ["APCA_API_KEY_ID"]    = api_key.strip()
            os.environ["APCA_API_SECRET_KEY"] = api_secret.strip()
            os.environ["APCA_API_BASE_URL"]   = base_url

            st.session_state.api_key    = api_key.strip()
            st.session_state.api_secret = api_secret.strip()
            st.session_state.base_url   = base_url
            st.session_state.keys_saved = True
            st.success("✅ Credentials saved!")

    st.divider()

    if st.session_state.keys_saved:
        st.success("🟢 Credentials ready")
        masked = st.session_state.api_key[:4] + "••••••••" + st.session_state.api_key[-4:]
        st.caption(f"Key: `{masked}`")
        st.caption(f"Mode: `{'Paper' if 'paper' in st.session_state.base_url else 'Live'}`")
    else:
        st.warning("🔴 No credentials saved yet")

    st.divider()
    st.caption(
        "💡 Keys are stored **only in this browser session** "
        "and never sent anywhere except Alpaca's API."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📈 Trading Bot Dashboard")

status_label = "🟢 Running" if st.session_state.running else "🔴 Stopped"
st.caption(f"Status: {status_label}")

st.divider()

if not st.session_state.keys_saved:
    st.warning(
        "⬅️ **Paste your Alpaca API Key and Secret in the sidebar first**, "
        "then press **Save Credentials** before starting the bot."
    )

col1, col2, col3 = st.columns(3)
ticker   = col1.text_input("Ticker",        "AAPL",        disabled=st.session_state.running)
qty      = col2.number_input("Qty",         1, 100, 5,     disabled=st.session_state.running)
interval = col3.selectbox("Interval (sec)", [30, 60, 120], disabled=st.session_state.running)

btn_col1, btn_col2 = st.columns(2)

start_disabled = st.session_state.running or not st.session_state.keys_saved
if btn_col1.button("▶️  Start", disabled=start_disabled, use_container_width=True):
    # Re-inject in case env was cleared between reruns
    os.environ["APCA_API_KEY_ID"]    = st.session_state.api_key
    os.environ["APCA_API_SECRET_KEY"] = st.session_state.api_secret
    os.environ["APCA_API_BASE_URL"]   = st.session_state.base_url

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_agent,
        args=(ticker, int(qty), int(interval), stop_event, st.session_state.log_queue),
        daemon=True,
    )
    thread.start()
    st.session_state.running    = True
    st.session_state.stop_event = stop_event
    st.session_state.logs       = []
    st.rerun()

if btn_col2.button("⏹  Stop", disabled=not st.session_state.running, use_container_width=True):
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    st.session_state.running = False
    st.rerun()

st.divider()

# ── Drain queue ───────────────────────────────────────────────────────────────
try:
    while True:
        item = st.session_state.log_queue.get_nowait()
        if item.get("type") == "log":
            st.session_state.logs.append(item["msg"])
except Empty:
    pass

if len(st.session_state.logs) > MAX_LOG_LINES:
    st.session_state.logs = st.session_state.logs[-MAX_LOG_LINES:]

# ── Live terminal ─────────────────────────────────────────────────────────────
st.subheader("🖥  Live Terminal")
if st.session_state.logs:
    st.code("\n".join(st.session_state.logs[-DISPLAY_LINES:]), language="")
else:
    st.info("No logs yet.  Save your credentials and press ▶️ Start to begin trading.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()