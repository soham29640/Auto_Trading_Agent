import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

class AlpacaOrder:
    def __init__(self):
        self.api = REST(API_KEY, API_SECRET, BASE_URL)
        self.account = self.api.get_account()
        print("Connected to Alpaca")
        print("Account ID:", self.account.id)
        print("Cash:", self.account.cash)
        print("Status:", self.account.status)
        print("Portfolio Value:", self.account.portfolio_value)

    def place_order(self, symbol="AAPL", qty=50, side="buy", type="market", time_in_force="gtc"):
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
        )
        print("Order submitted:", order.id)
        return order

