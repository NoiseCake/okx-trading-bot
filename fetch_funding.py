"""
Fetch Binance USDT-perp funding-rate history for the RP2 basket → data/funding/.

Binance keeps full history (BTC/ETH from Sep 2019); OKX only ~3 months, which is
why the proxy is needed. Funding pays every 8h at 00/08/16 UTC. Output CSV:
ts (ms, fundingTime), rate.

Usage:  python3 fetch_funding.py
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "funding")
SYMBOLS = {f"{a}-USDT": f"{a}USDT"
           for a in ("BTC", "ETH", "XRP", "DOGE", "LTC", "LINK", "SOL", "AVAX")}


def fetch(symbol: str) -> pd.DataFrame:
    # startTime=0 is ignored (returns the newest page), so walk forward from a
    # real epoch and stop once the last row is within one funding interval of now.
    rows, start = [], int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    while True:
        for attempt in range(5):
            try:
                r = requests.get(URL, params={"symbol": symbol, "startTime": start,
                                              "limit": 1000}, timeout=20)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2.0 * (attempt + 1))
        if not data:
            break
        rows.extend(data)
        last = int(data[-1]["fundingTime"])
        if last >= (time.time() - 8 * 3600) * 1000:
            break
        start = last + 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df["ts"] = df["fundingTime"].astype("int64")
    df["rate"] = df["fundingRate"].astype(float)
    return df.drop_duplicates("ts").sort_values("ts")[["ts", "rate"]]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for inst, sym in SYMBOLS.items():
        df = fetch(sym)
        df.to_csv(os.path.join(OUT, f"{inst}.csv"), index=False)
        first = pd.Timestamp(df["ts"].iloc[0], unit="ms", tz="UTC")
        last = pd.Timestamp(df["ts"].iloc[-1], unit="ms", tz="UTC")
        # 3 payments/day expected; report coverage so gaps are visible at fetch time
        days = (last - first).days or 1
        print(f"{inst:10} {len(df):6d} rows  {first:%Y-%m-%d} → {last:%Y-%m-%d}"
              f"  ({len(df) / (3 * days):.0%} of 3/day)", flush=True)


if __name__ == "__main__":
    main()
