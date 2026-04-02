import sys
import os
import math
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import HIGH_AUTHORITY_ACCOUNTS, SIGNAL_WINDOWS, ALERT_THRESHOLD


def compute_weighted_signal(tweets: list, window_minutes: int, asset: str) -> dict:
    """
    Filter tweets by asset_tag == asset AND created_at >= now() - window_minutes.

    For each qualifying tweet:
      weight = log(followers_count + 1) * (1 + log(retweet_count + 1)) * authority_multiplier
      authority_multiplier = 3.0 if author in HIGH_AUTHORITY_ACCOUNTS else 1.0

    signal = sum(weight * sentiment_score) / sum(weight)  (weighted average)
    If no tweets qualify: return {"signal": 0.0, "tweet_count": 0, "alert": False}
    alert = abs(signal) >= ALERT_THRESHOLD

    Returns: {"signal": float, "tweet_count": int, "alert": bool}
    """
    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=window_minutes)

    weighted_sum = 0.0
    weight_total = 0.0
    count = 0

    for tweet in tweets:
        if tweet.get("asset_tag") != asset:
            continue
        created_at = tweet.get("created_at")
        if created_at is None:
            continue
        # Normalize to naive UTC for comparison
        if hasattr(created_at, 'tzinfo') and created_at.tzinfo is not None:
            created_at = created_at.replace(tzinfo=None)
        if created_at < cutoff:
            continue

        followers = tweet.get("followers_count", 0)
        retweets = tweet.get("retweet_count", 0)
        sentiment = tweet.get("sentiment_score", 0.0)
        author = tweet.get("author", "")

        authority_multiplier = 3.0 if author in HIGH_AUTHORITY_ACCOUNTS else 1.0

        weight = math.log(followers + 1) * (1 + math.log(retweets + 1)) * authority_multiplier

        weighted_sum += weight * sentiment
        weight_total += weight
        count += 1

    if count == 0:
        return {"signal": 0.0, "tweet_count": 0, "alert": False}

    signal = weighted_sum / weight_total
    alert = abs(signal) >= ALERT_THRESHOLD

    return {"signal": signal, "tweet_count": count, "alert": alert}


def compute_all_windows(tweets: list, asset: str) -> dict:
    """
    Call compute_weighted_signal for each window in SIGNAL_WINDOWS.
    Returns: {"60min": result_60, "240min": result_240, "1440min": result_1440}
    """
    results = {}
    for window in SIGNAL_WINDOWS:
        key = f"{window}min"
        results[key] = compute_weighted_signal(tweets, window, asset)
    return results
