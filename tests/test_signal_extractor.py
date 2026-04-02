from module1_twitter.signal.extractor import compute_weighted_signal, compute_all_windows
from datetime import datetime, timedelta


def test_weighted_signal_basic():
    now = datetime.utcnow()
    tweets = [
        {"author": "realDonaldTrump", "followers_count": 100_000_000,
         "retweet_count": 50000, "sentiment_score": -0.8,
         "asset_tag": "WTI", "created_at": now - timedelta(minutes=30)},
        {"author": "randomuser123", "followers_count": 500,
         "retweet_count": 2, "sentiment_score": 0.9,
         "asset_tag": "WTI", "created_at": now - timedelta(minutes=10)},
        {"author": "zerohedge", "followers_count": 1_200_000,
         "retweet_count": 1500, "sentiment_score": -0.7,
         "asset_tag": "WTI", "created_at": now - timedelta(minutes=50)},
    ]
    result = compute_weighted_signal(tweets, window_minutes=60, asset="WTI")
    assert result["tweet_count"] == 3
    assert result["signal"] < 0  # dominated by Trump + zerohedge bearish
    assert isinstance(result["alert"], bool)


def test_empty():
    result = compute_weighted_signal([], window_minutes=60, asset="WTI")
    assert result == {"signal": 0.0, "tweet_count": 0, "alert": False}


def test_wrong_asset_filtered():
    now = datetime.utcnow()
    tweets = [
        {"author": "test", "followers_count": 1000, "retweet_count": 10,
         "sentiment_score": 0.9, "asset_tag": "GOLD", "created_at": now - timedelta(minutes=5)},
    ]
    result = compute_weighted_signal(tweets, window_minutes=60, asset="WTI")
    assert result["tweet_count"] == 0


def test_window_filter():
    now = datetime.utcnow()
    tweets = [
        {"author": "test", "followers_count": 1000, "retweet_count": 10,
         "sentiment_score": 0.9, "asset_tag": "WTI", "created_at": now - timedelta(minutes=90)},
    ]
    result = compute_weighted_signal(tweets, window_minutes=60, asset="WTI")
    assert result["tweet_count"] == 0


def test_compute_all_windows():
    now = datetime.utcnow()
    tweets = [
        {"author": "test", "followers_count": 1000, "retweet_count": 10,
         "sentiment_score": 0.5, "asset_tag": "WTI", "created_at": now - timedelta(minutes=30)},
    ]
    result = compute_all_windows(tweets, "WTI")
    assert "60min" in result
    assert "240min" in result
    assert "1440min" in result
