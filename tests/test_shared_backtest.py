"""Tests for shared/backtest/price_client.py and shared/backtest/engine.py.

All price data is mocked — no real network calls.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import pytz

from shared.backtest.price_client import PriceClient
from shared.backtest.engine import BacktestEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(start: datetime, periods: int, freq: str = "h",
                   base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic price DataFrame with a UTC DatetimeIndex."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    prices = base + rng.randn(periods).cumsum()
    return pd.DataFrame({"close": prices}, index=idx)


# ---------------------------------------------------------------------------
# PriceClient._resolve_ticker
# ---------------------------------------------------------------------------

class TestResolveTicker:
    def test_resolve_assets_key(self):
        pc = PriceClient()
        result = pc._resolve_ticker("WTI")
        assert result["yf_ticker"] == "CL=F"
        assert result["av_symbol"] == "USOIL"

    def test_resolve_entity_tickers_key(self):
        pc = PriceClient()
        result = pc._resolve_ticker("gold")
        assert result["yf"] == "GC=F"
        assert result["av"] == "XAUUSD"

    def test_resolve_unknown_raises(self):
        pc = PriceClient()
        with pytest.raises(KeyError):
            pc._resolve_ticker("UNKNOWN_ASSET")


# ---------------------------------------------------------------------------
# PriceClient – intraday without key
# ---------------------------------------------------------------------------

class TestPriceClientIntraday:
    @patch("shared.backtest.price_client.ALPHA_VANTAGE_KEY", "")
    def test_raises_without_av_key(self):
        pc = PriceClient()
        with pytest.raises(ValueError, match="ALPHA_VANTAGE_KEY"):
            pc.get_prices("WTI", datetime(2024, 1, 1), datetime(2024, 1, 2),
                          horizon_hours=0.5)


# ---------------------------------------------------------------------------
# BacktestEngine.run – edge cases
# ---------------------------------------------------------------------------

class TestBacktestEdgeCases:
    def test_empty_signals_returns_empty(self):
        engine = BacktestEngine()
        assert engine.run([], "WTI") == {}

    def test_few_signals_returns_error(self):
        engine = BacktestEngine()
        signals = [
            {"timestamp": datetime(2024, 1, 15, tzinfo=pytz.UTC), "signal_value": 0.5}
            for _ in range(3)
        ]
        result = engine.run(signals, "WTI")
        assert "error" in result
        assert result["n_signals"] == 3


# ---------------------------------------------------------------------------
# BacktestEngine.run – full run with mocked prices
# ---------------------------------------------------------------------------

class TestBacktestFullRun:
    @pytest.fixture()
    def engine_and_results(self):
        """Run the backtest with mocked price data and return the results."""
        start = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        price_df = _make_price_df(start, periods=2000, freq="h", seed=42)

        signals = []
        rng = np.random.RandomState(99)
        for i in range(40):
            t = start + timedelta(hours=i * 24)
            sv = rng.uniform(-1, 1)
            signals.append({"timestamp": t, "signal_value": sv})

        engine = BacktestEngine()
        with patch.object(engine.price_client, "get_prices", return_value=price_df):
            results = engine.run(signals, "WTI", horizon_hours=24)

        return results

    def test_all_required_keys(self, engine_and_results):
        required = [
            "forward_returns_by_bucket", "rolling_correlation",
            "directional_accuracy", "accuracy_by_quintile",
            "confusion_matrix", "pearson_r", "pearson_pval",
            "spearman_r", "n_signals", "raw_df",
        ]
        for key in required:
            assert key in engine_and_results, f"Missing key: {key}"

    def test_directional_accuracy_range(self, engine_and_results):
        acc = engine_and_results["directional_accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_confusion_matrix_keys(self, engine_and_results):
        cm = engine_and_results["confusion_matrix"]
        for key in ("TP", "TN", "FP", "FN"):
            assert key in cm, f"Missing CM key: {key}"
            assert isinstance(cm[key], int)

    def test_n_signals(self, engine_and_results):
        assert engine_and_results["n_signals"] == 40

    def test_pearson_r_range(self, engine_and_results):
        r = engine_and_results["pearson_r"]
        assert -1.0 <= r <= 1.0

    def test_spearman_r_range(self, engine_and_results):
        r = engine_and_results["spearman_r"]
        assert -1.0 <= r <= 1.0

    def test_raw_df_is_dataframe(self, engine_and_results):
        assert isinstance(engine_and_results["raw_df"], pd.DataFrame)

    def test_confusion_matrix_sums(self, engine_and_results):
        cm = engine_and_results["confusion_matrix"]
        total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
        assert total == engine_and_results["n_signals"]
