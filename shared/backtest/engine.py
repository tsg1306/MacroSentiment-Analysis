import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from scipy import stats

from shared.backtest.price_client import PriceClient


class BacktestEngine:
    """Run a backtest of sentiment signals against forward price returns."""

    def __init__(self):
        self.price_client = PriceClient()

    def run(self, signals: list, asset: str, horizon_hours: float = 24) -> dict:
        """
        Parameters
        ----------
        signals : list[dict]
            Each dict has 'timestamp' (datetime) and 'signal_value' (float).
        asset : str
            Asset key recognized by PriceClient.
        horizon_hours : float
            Forward-return horizon in hours.

        Returns
        -------
        dict  –  empty if *signals* is empty; error dict if fewer than 5;
                 otherwise full metrics dict.
        """
        if not signals:
            return {}

        n = len(signals)
        if n < 5:
            return {"error": f"Not enough signals ({n} < 5)", "n_signals": n}

        # ── 1. Fetch prices ──────────────────────────────────────────────
        sig_df = pd.DataFrame(signals)
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"], utc=True)
        sig_df = sig_df.sort_values("timestamp").reset_index(drop=True)

        start = sig_df["timestamp"].min()
        end = sig_df["timestamp"].max() + pd.Timedelta(hours=horizon_hours * 2)

        prices = self.price_client.get_prices(asset, start, end, horizon_hours)

        if prices.empty:
            return {"error": "No price data returned", "n_signals": n}

        # ── 2. Compute forward returns ───────────────────────────────────
        horizon_td = pd.Timedelta(hours=horizon_hours)
        fwd_returns = []

        for _, row in sig_df.iterrows():
            t = row["timestamp"]
            t_end = t + horizon_td

            # Find the closest price at t and t+horizon
            p0 = self._closest_price(prices, t)
            p1 = self._closest_price(prices, t_end)

            if p0 is not None and p1 is not None and p0 != 0:
                fwd_returns.append((p1 - p0) / p0)
            else:
                fwd_returns.append(np.nan)

        sig_df["forward_return"] = fwd_returns

        # Drop rows without a valid forward return
        merged = sig_df.dropna(subset=["forward_return"]).reset_index(drop=True)

        if len(merged) < 5:
            return {"error": f"Not enough matched signals ({len(merged)} < 5)", "n_signals": n}

        # ── 3. Metrics ───────────────────────────────────────────────────
        sv = merged["signal_value"].values
        fr = merged["forward_return"].values

        result = {
            "forward_returns_by_bucket": self._bucket_returns(merged),
            "rolling_correlation": self._rolling_correlation(merged, window=20),
            "directional_accuracy": self._directional_accuracy(sv, fr),
            "accuracy_by_quintile": self._accuracy_by_quintile(merged),
            "confusion_matrix": self._confusion_matrix(sv, fr),
            "pearson_r": float(stats.pearsonr(sv, fr)[0]),
            "pearson_pval": float(stats.pearsonr(sv, fr)[1]),
            "spearman_r": float(stats.spearmanr(sv, fr)[0]),
            "n_signals": len(merged),
            "raw_df": merged,
        }
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _closest_price(prices: pd.DataFrame, target_time) -> float | None:
        """Return the 'close' price nearest to *target_time*."""
        if prices.empty:
            return None
        ts = pd.Timestamp(target_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        idx = prices.index.get_indexer([ts], method="nearest")[0]
        if idx < 0:
            return None
        return float(prices["close"].iloc[idx])

    @staticmethod
    def _bucket_returns(df: pd.DataFrame) -> dict:
        """Average forward return per sentiment bucket."""
        def _label(v):
            if v < -0.4:
                return "Very Negative"
            elif v < -0.1:
                return "Negative"
            elif v <= 0.1:
                return "Neutral"
            elif v <= 0.4:
                return "Positive"
            else:
                return "Very Positive"

        df = df.copy()
        df["bucket"] = df["signal_value"].apply(_label)
        grouped = df.groupby("bucket")["forward_return"].mean()
        return grouped.to_dict()

    @staticmethod
    def _rolling_correlation(df: pd.DataFrame, window: int = 20) -> list:
        """Rolling Pearson correlation between signal and forward return."""
        if len(df) < window:
            return []
        rolling = df["signal_value"].rolling(window).corr(df["forward_return"])
        records = []
        for i, val in enumerate(rolling):
            if not np.isnan(val):
                records.append({
                    "timestamp": str(df["timestamp"].iloc[i]),
                    "rolling_corr": float(val),
                })
        return records

    @staticmethod
    def _directional_accuracy(signals: np.ndarray, returns: np.ndarray) -> float:
        """Fraction of observations where sign(signal) == sign(return)."""
        mask = (signals != 0) & (returns != 0)
        if mask.sum() == 0:
            return 0.0
        correct = np.sign(signals[mask]) == np.sign(returns[mask])
        return float(correct.mean())

    @staticmethod
    def _accuracy_by_quintile(df: pd.DataFrame) -> dict:
        """Directional accuracy within each signal-strength quintile (Q1-Q5)."""
        df = df.copy()
        try:
            df["quintile"] = pd.qcut(df["signal_value"].abs(), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        except ValueError:
            # Not enough unique values for 5 bins
            return {}

        result = {}
        for q, grp in df.groupby("quintile", observed=True):
            mask = (grp["signal_value"] != 0) & (grp["forward_return"] != 0)
            if mask.sum() > 0:
                correct = np.sign(grp.loc[mask, "signal_value"].values) == np.sign(grp.loc[mask, "forward_return"].values)
                result[str(q)] = float(correct.mean())
            else:
                result[str(q)] = 0.0
        return result

    @staticmethod
    def _confusion_matrix(signals: np.ndarray, returns: np.ndarray) -> dict:
        """Binary confusion matrix: positive signal/return → P, else N."""
        tp = int(np.sum((signals > 0) & (returns > 0)))
        tn = int(np.sum((signals <= 0) & (returns <= 0)))
        fp = int(np.sum((signals > 0) & (returns <= 0)))
        fn = int(np.sum((signals <= 0) & (returns > 0)))
        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
