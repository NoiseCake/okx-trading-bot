import okx.Account as Account
import okx.MarketData as MarketData
import okx.PublicData as PublicData
import okx.Trade as Trade
from loguru import logger
from config import API_KEY, SECRET_KEY, PASSPHRASE, FLAG


class OKXClient:
    """Thin wrapper around the three OKX SDK sub-APIs we use: account, market data, and trading."""

    def __init__(self):
        # Each OKX sub-API gets its own authenticated client instance.
        # The False argument disables the built-in SDK logging (we use loguru instead).
        # FLAG="1" → paper trading, FLAG="0" → live trading.
        self.account_api = Account.AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
        self.market_api  = MarketData.MarketAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
        self.trade_api   = Trade.TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
        self.public_api  = PublicData.PublicAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
        mode = "PAPER TRADING" if FLAG == "1" else "LIVE TRADING"
        logger.info(f"OKX client initialized — {mode}")

    # ── Market Data ──────────────────────────────────────────────────────────────

    def get_ticker(self, inst_id: str) -> dict:
        """Return the latest ticker for an instrument (contains 'last' price, bid/ask, etc.)."""
        result = self.market_api.get_ticker(instId=inst_id)
        if result["code"] != "0":
            raise RuntimeError(f"get_ticker failed: {result['msg']}")
        return result["data"][0]

    def get_candlesticks(self, inst_id: str, bar: str = "1H", limit: int = 100) -> list:
        """
        Fetch OHLCV candles. Each candle is a list:
        [timestamp, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
        'confirm'="1" means the candle is closed (not still forming).
        """
        result = self.market_api.get_candlesticks(instId=inst_id, bar=bar, limit=str(limit))
        if result["code"] != "0":
            raise RuntimeError(f"get_candlesticks failed: {result['msg']}")
        return result["data"]

    # ── Account ──────────────────────────────────────────────────────────────────

    def get_balance(self, ccy: str = "") -> dict:
        """Return the full account balance object. Filter by currency symbol if provided."""
        result = self.account_api.get_account_balance(ccy=ccy)
        if result["code"] != "0":
            raise RuntimeError(f"get_balance failed: {result['msg']}")
        return result["data"][0]

    def get_asset_balance(self, ccy: str) -> float:
        """Return the available balance for a single currency (e.g. 'BTC'). Returns 0.0 if not held."""
        balance = self.get_balance(ccy=ccy)
        for detail in balance.get("details", []):
            if detail.get("ccy") == ccy:
                return float(detail.get("availBal") or 0)
        return 0.0

    def get_spot_specs(self, inst_ids: list[str]) -> dict:
        """Return {inst_id: {"lotSz": float, "minSz": float}} for spot instruments.

        Live lot metadata replaces the old hardcoded dict — with 8 instruments a
        stale entry would silently reject or mis-round orders.
        """
        result = self.public_api.get_instruments(instType="SPOT")
        if result["code"] != "0":
            raise RuntimeError(f"get_instruments failed: {result['msg']}")
        want = set(inst_ids)
        return {d["instId"]: {"lotSz": float(d["lotSz"]), "minSz": float(d["minSz"])}
                for d in result["data"] if d["instId"] in want}

    # ── Orders ───────────────────────────────────────────────────────────────────

    def place_market_order(self, inst_id: str, side: str, size: str,
                           tgt_ccy: str = "base_ccy") -> dict:
        """
        Submit a market order that fills immediately at the best available price.
        side: 'buy' or 'sell'
        size: quantity in base currency, or in USDT when tgt_ccy='quote_ccy'
        tdMode='cash' means spot trading with no leverage.

        tgtCcy defaults to 'base_ccy': OKX interprets sz on spot market BUYS as
        the QUOTE amount (USDT) unless told otherwise, so without it a buy of
        '0.03' would purchase 0.03 USDT worth of BTC — not 0.03 BTC — while the
        bot records a 0.03 BTC position. (Sells already default to base_ccy.)
        The xsmom rebalancer passes tgt_ccy='quote_ccy' deliberately: its buys
        are sized as USDT notional.
        """
        result = self.trade_api.place_order(
            instId=inst_id,
            tdMode="cash",
            side=side,
            ordType="market",
            sz=size,
            tgtCcy=tgt_ccy,
        )
        if result["code"] != "0":
            detail = ""
            if result.get("data"):
                d = result["data"][0]
                detail = f" | sCode={d.get('sCode', '')} sMsg={d.get('sMsg', '')}"
            raise RuntimeError(f"place_order failed: {result['msg']}{detail}")
        order = result["data"][0]
        logger.info(f"Order placed — {side.upper()} {size} {inst_id} | ordId={order['ordId']}")
        return order

    def get_order(self, inst_id: str, ord_id: str) -> dict:
        """Fetch the current status of a specific order (filled, partially filled, etc.)."""
        result = self.trade_api.get_order(instId=inst_id, ordId=ord_id)
        if result["code"] != "0":
            raise RuntimeError(f"get_order failed: {result['msg']}")
        return result["data"][0]
