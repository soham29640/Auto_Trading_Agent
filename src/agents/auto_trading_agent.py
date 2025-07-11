import os
import sys
import csv
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.trading_agent import TradingAgent
from src.wallets.wallet import Wallet

wallet = Wallet(starting_cash=100000)
agent = TradingAgent("AAPL",wallet)

os.makedirs("logs", exist_ok=True)

class AutoTradingAgent:
    def __init__(self, wallet, agent, log_path = "logs/trades.csv"):
        self.wallet = wallet
        self.agent = agent
        self.log_path = log_path
        header = ["timestamp", "action", "current_price", "predicted_price", "cash", "holdings", "avg_buy_price", "portfolio_value"]
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_trade(self, action, current, predicted):

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                action,
                round(current, 2),
                round(predicted, 2),
                round(self.wallet.cash, 2),
                self.wallet.holdings,
                round(self.wallet.avg_buy_price, 2),
                round(self.wallet.portfolio_value, 2)
            ])

    def run_once(self):
        try:
            action, current, predicted = self.agent.act()
            self.log_trade(action, current, predicted)
            status = self.wallet.status(current)
            return {
                "time": datetime.now().strftime('%H:%M:%S'),
                "action": action,
                "current": round(current, 2),
                "predicted": round(predicted, 2),
                "cash": round(status["cash"], 2),
                "holdings": status["holdings"],
                "avg_buy_price": round(status["avg_buy_price"], 2),
                "portfolio_value": round(status["portfolio_value"], 2)
            }
        except Exception as e:
            return {"error": str(e)}
