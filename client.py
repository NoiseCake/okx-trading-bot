import okx.Account as Account
import okx.MarketData as MarketData
import okx.Trade as Trade
from loguru import logger
from config import API_KEY, SECRET_KEY, PASSPHRASE, FLAG


class OKXClient:
    def __init__(self):
        self.account_api = Account.AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
        self.market_api = MarketData.MarketAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
        self.trade_api = Trade.TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
        mode = "PAPER TRADING" if FLAG == "1" else "LIVE TRADING"
        logger.info(f"OKX client initialized — {mode}")

    # ── Market Data ──────────────────────────────────────────────────────────

    def get_ticker(self, inst_id: str) -> dict:
        result = self.market_api.get_ticker(instId=inst_id)
        if result["code"] != "0":
            raise RuntimeError(f"get_ticker failed: {result['msg']}")
        return result["data"][0]

    def get_candlesticks(self, inst_id: str, bar: str = "1H", limit: int = 100) -> list:
        result = self.market_api.get_candlesticks(instId=inst_id, bar=bar, limit=str(limit))
        if result["code"] != "0":
            raise RuntimeError(f"get_candlesticks failed: {result['msg']}")
        return result["data"]

    # ── Account ──────────────────────────────────────────────────────────────

    def get_balance(self, ccy: str = "") -> dict:
        result = self.account_api.get_account_balance(ccy=ccy)
        if result["code"] != "0":
            raise RuntimeError(f"get_balance failed: {result['msg']}")
        return result["data"][0]

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_market_order(self, inst_id: str, side: str, size: str) -> dict:
        """
        side: 'buy' or 'sell'
        size: quantity in base currency (e.g. '0.01' for 0.01 BTC)
        """
        result = self.trade_api.place_order(
            instId=inst_id,
            tdMode="cash",
            side=side,
            ordType="market",
            sz=size,
        )
        if result["code"] != "0":
            raise RuntimeError(f"place_order failed: {result['msg']}")
        order = result["data"][0]
        logger.info(f"Order placed — {side.upper()} {size} {inst_id} | ordId={order['ordId']}")
        return order

    def place_limit_order(self, inst_id: str, side: str, size: str, price: str) -> dict:
        result = self.trade_api.place_order(
            instId=inst_id,
            tdMode="cash",
            side=side,
            ordType="limit",
            sz=size,
            px=price,
        )
        if result["code"] != "0":
            raise RuntimeError(f"place_order failed: {result['msg']}")
        order = result["data"][0]
        logger.info(f"Limit order placed — {side.upper()} {size} {inst_id} @ {price} | ordId={order['ordId']}")
        return order

    def cancel_order(self, inst_id: str, ord_id: str) -> dict:
        result = self.trade_api.cancel_order(instId=inst_id, ordId=ord_id)
        if result["code"] != "0":
            raise RuntimeError(f"cancel_order failed: {result['msg']}")
        logger.info(f"Order cancelled — {ord_id}")
        return result["data"][0]

    def get_order(self, inst_id: str, ord_id: str) -> dict:
        result = self.trade_api.get_order(instId=inst_id, ordId=ord_id)
        if result["code"] != "0":
            raise RuntimeError(f"get_order failed: {result['msg']}")
        return result["data"][0]
