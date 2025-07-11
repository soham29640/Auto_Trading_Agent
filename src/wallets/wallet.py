import datetime as datetime

class Wallet:
    def __init__(self, starting_cash=100000):
        self.cash = starting_cash
        self.holdings = 0
        self.avg_buy_price = 0.0
        self.trades = []
        self.portfolio_value = starting_cash

    def can_buy(self, price, qty=1):
        return self.cash >= price * qty

    def can_sell(self, qty=1):
        return self.holdings >= qty

    def buy(self, price, qty):
        if self.can_buy(price, qty):
            self.cash -= price * qty
            self.avg_buy_price = (
                (self.avg_buy_price * self.holdings + price * qty) / (self.holdings + qty)
            )
            self.holdings += qty
            self.portfolio_value = self.cash + self.holdings * price
            self._log_trade("BUY", price, qty)
            return True
        return False

    def sell(self, price, qty):
        if self.can_sell(qty):
            self.cash += price * qty
            self.holdings -= qty
            self.portfolio_value = self.cash + self.holdings * price
            if self.holdings == 0:
                self.avg_buy_price = 0.0
            self._log_trade("SELL", price, qty)
            return True
        return False

    def _log_trade(self, action, price, qty):
        self.trades.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "price": price,
            "qty": qty,
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": self.portfolio_value
        })

    def status(self, current_price):
        market_value = self.holdings * current_price
        portfolio_value = self.cash + market_value
        return {
            "cash": self.cash,
            "holdings": self.holdings,
            "avg_buy_price": self.avg_buy_price,
            "market_value": market_value,
            "portfolio_value": portfolio_value
    }


