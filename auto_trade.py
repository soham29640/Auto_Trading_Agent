import time
from datetime import datetime
from src.agents.trading_agent import TradingAgent

def main():
    ticker = "AAPL"
    agent = TradingAgent(ticker)

    print(f"🔁 Starting Auto Trading Agent for {ticker}")
    
    while True:
        now = datetime.now()
        try:
            action, current, predicted = agent.act(qty=50)
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Action: {action} | Current: ${current:.2f} | Predicted: ${predicted:.2f}")
        except Exception as e:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Error:", e)

        time.sleep(60) 

if __name__ == "__main__":
    main()
