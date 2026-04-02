"""Smoke tests — verify dashboard imports work without errors."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    from shared.nlp.vader_sentiment import VaderSentiment
    from shared.nlp.finbert_sentiment import FinBERTSentiment
    from shared.backtest.engine import BacktestEngine
    from shared.backtest.price_client import PriceClient
    from module1_twitter.twitter.client import TwitterClient
    from module1_twitter.nlp.preprocessor import clean_tweet
    from module1_twitter.signal.extractor import compute_all_windows
    from module2_nlp.ingestion.mock_documents import get_mock_documents
    from module2_nlp.nlp.pipeline import process_document
    from module2_nlp.nlp.chunker import chunk_text
    from module2_nlp.nlp.ner import extract_entities_with_context
    from module2_nlp.signal.aggregator import aggregate_signals
    from shared.db.database import init_db, get_all_entities


def test_config():
    from config import ASSETS, SIGNAL_WINDOWS, ALERT_THRESHOLD, HIGH_AUTHORITY_ACCOUNTS
    assert len(ASSETS) >= 4
    assert len(SIGNAL_WINDOWS) == 3
    assert ALERT_THRESHOLD == 0.4


def test_dashboard_module_importable():
    """Verify dashboard/app.py can be parsed without starting streamlit."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dashboard_app",
        os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'app.py')
    )
    assert spec is not None
