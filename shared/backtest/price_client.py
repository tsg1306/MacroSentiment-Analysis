import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from config import ASSETS, ENTITY_TICKERS, ALPHA_VANTAGE_KEY, INTRADAY_THRESHOLD_HOURS


class PriceClient:
    """Fetch historical price data via yfinance or Alpha Vantage."""

    def get_prices(self, asset: str, start, end, horizon_hours: float) -> pd.DataFrame:
        """
        Fetch price data for *asset* between *start* and *end*.

        Routing logic:
        - horizon_hours < INTRADAY_THRESHOLD_HOURS  -> Alpha Vantage 1-min bars
        - 1 <= horizon_hours <= 24                  -> yfinance interval='1h'
        - horizon_hours > 24                        -> yfinance interval='1d'

        Returns a DataFrame with a UTC DatetimeIndex and a 'close' column.
        """
        tickers = self._resolve_ticker(asset)

        if horizon_hours < INTRADAY_THRESHOLD_HOURS:
            # Alpha Vantage path
            if not ALPHA_VANTAGE_KEY:
                raise ValueError(
                    "ALPHA_VANTAGE_KEY is required for intraday data "
                    f"(horizon_hours={horizon_hours} < {INTRADAY_THRESHOLD_HOURS})"
                )
            return self._fetch_alpha_vantage(tickers.get("av_symbol") or tickers.get("av"), start, end)

        # yfinance path
        yf_ticker = tickers.get("yf_ticker") or tickers.get("yf")
        if horizon_hours <= 24:
            interval = "1h"
        else:
            interval = "1d"

        return self._fetch_yfinance(yf_ticker, start, end, interval)

    # ------------------------------------------------------------------
    # Ticker resolution
    # ------------------------------------------------------------------

    def _resolve_ticker(self, asset: str) -> dict:
        """Look up ticker symbols in ASSETS first, then ENTITY_TICKERS."""
        if asset in ASSETS:
            entry = ASSETS[asset]
            return {"yf_ticker": entry["yf_ticker"], "av_symbol": entry["av_symbol"]}

        if asset in ENTITY_TICKERS:
            entry = ENTITY_TICKERS[asset]
            return {"yf": entry["yf"], "av": entry["av"]}

        raise KeyError(f"Asset '{asset}' not found in ASSETS or ENTITY_TICKERS")

    # ------------------------------------------------------------------
    # Data providers
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, ticker: str, start, end, interval: str) -> pd.DataFrame:
        import yfinance as yf

        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame(columns=["close"])

        # yfinance may return MultiIndex columns when a single ticker is passed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data[["Close"]].rename(columns={"Close": "close"})

        # Ensure UTC-aware DatetimeIndex
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        return df

    def _fetch_alpha_vantage(self, symbol: str, start, end) -> pd.DataFrame:
        import requests

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "1min",
            "outputsize": "full",
            "apikey": ALPHA_VANTAGE_KEY,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        ts_key = "Time Series (1min)"
        if ts_key not in data:
            return pd.DataFrame(columns=["close"])

        rows = {k: float(v["4. close"]) for k, v in data[ts_key].items()}
        df = pd.DataFrame.from_dict(rows, orient="index", columns=["close"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Localize to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Filter to requested range
        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            df = df[df.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC")
            df = df[df.index <= end_ts]

        return df
