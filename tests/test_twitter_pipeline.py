"""Tests for Module 1 Twitter pipeline: backends, preprocessor, client."""

import sys
import os
from datetime import datetime, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


REQUIRED_KEYS = [
    "id", "created_at", "author", "followers_count",
    "text", "retweet_count", "like_count",
]


# ── 1. MockBackend returns correct format ──────────────────────────────────

class TestMockBackendFormat:
    def setup_method(self):
        from module1_twitter.twitter.mock_backend import MockBackend
        self.backend = MockBackend()
        self.tweets = self.backend.search(["oil", "gold", "inflation", "Russia"], limit=200)

    def test_has_all_required_keys(self):
        for t in self.tweets:
            for key in REQUIRED_KEYS:
                assert key in t, f"Missing key '{key}' in tweet"

    def test_correct_types(self):
        for t in self.tweets:
            assert isinstance(t["id"], str), "id must be str"
            assert isinstance(t["created_at"], datetime), "created_at must be datetime"
            assert isinstance(t["author"], str), "author must be str"
            assert isinstance(t["followers_count"], int), "followers_count must be int"
            assert isinstance(t["text"], str), "text must be str"
            assert isinstance(t["retweet_count"], int), "retweet_count must be int"
            assert isinstance(t["like_count"], int), "like_count must be int"

    def test_no_none_values(self):
        for t in self.tweets:
            assert t["followers_count"] is not None
            assert t["retweet_count"] is not None
            assert t["like_count"] is not None


# ── 2. search respects limit ───────────────────────────────────────────────

class TestSearchLimit:
    def setup_method(self):
        from module1_twitter.twitter.mock_backend import MockBackend
        self.backend = MockBackend()

    def test_limit_50(self):
        results = self.backend.search(["oil", "crude"], limit=50)
        assert len(results) == 50

    def test_limit_10(self):
        results = self.backend.search(["oil", "crude"], limit=10)
        assert len(results) == 10

    def test_limit_larger_than_matches(self):
        results = self.backend.search(["oil", "crude"], limit=9999)
        assert len(results) <= 9999


# ── 3. get_user_tweets works for authority accounts ────────────────────────

class TestGetUserTweets:
    def setup_method(self):
        from module1_twitter.twitter.mock_backend import MockBackend
        self.backend = MockBackend()

    def test_authority_account_has_tweets(self):
        tweets = self.backend.get_user_tweets("realDonaldTrump", limit=10)
        assert len(tweets) > 0, "realDonaldTrump should have tweets"
        assert all(t["author"] == "realDonaldTrump" for t in tweets)

    def test_authority_account_high_followers(self):
        tweets = self.backend.get_user_tweets("realDonaldTrump", limit=5)
        for t in tweets:
            assert t["followers_count"] > 500_000, (
                "HIGH_AUTHORITY_ACCOUNTS must have followers_count > 500,000"
            )

    def test_limit_respected(self):
        tweets = self.backend.get_user_tweets("realDonaldTrump", limit=2)
        assert len(tweets) <= 2


# ── 4. Preprocessor ────────────────────────────────────────────────────────

class TestPreprocessor:
    def setup_method(self):
        from module1_twitter.nlp.preprocessor import clean_tweet
        self.clean = clean_tweet

    def test_urls_removed(self):
        result = self.clean("Check https://t.co/xyz and http://example.com")
        assert "https" not in result
        assert "http" not in result

    def test_mentions_removed(self):
        result = self.clean("Hello @BBCWorld what's up @Reuters")
        assert "@" not in result

    def test_hashtag_symbol_removed(self):
        result = self.clean("Trending #OPEC and #oil")
        assert "#" not in result
        assert "opec" in result
        assert "oil" in result

    def test_synonyms_applied(self):
        result = self.clean("Crude oil prices and petroleum futures and XAU/USD")
        assert "crude oil" not in result
        assert "petroleum" not in result
        # "oil" should appear (from synonym replacement)
        assert "oil" in result

    def test_lowercase(self):
        result = self.clean("BREAKING NEWS Oil PRICES")
        assert result == result.lower()


# ── 5. Preprocessor is idempotent ──────────────────────────────────────────

class TestPreprocessorIdempotent:
    def setup_method(self):
        from module1_twitter.nlp.preprocessor import clean_tweet
        self.clean = clean_tweet

    def test_idempotent_simple(self):
        raw = "Check out https://t.co/xyz #OPEC @BBCWorld Crude Oil prices CRASH!!!"
        once = self.clean(raw)
        twice = self.clean(once)
        assert once == twice

    def test_idempotent_complex(self):
        raw = "RT @user: Federal Reserve raises rates!!! #inflation $SPX https://link.co/abc"
        once = self.clean(raw)
        twice = self.clean(once)
        assert once == twice

    def test_idempotent_already_clean(self):
        clean_text = "oil prices are stable today"
        assert self.clean(clean_text) == clean_text


# ── 6. ApiBackend raises ValueError with empty token ──────────────────────

class TestApiBackendValidation:
    def test_raises_on_empty_token(self, monkeypatch):
        # Ensure the token is empty
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "")
        # Need to reload config so it picks up the patched env
        import config
        monkeypatch.setattr(config, "TWITTER_BEARER_TOKEN", "")

        # Also patch at the module level if already imported
        import module1_twitter.twitter.api_backend as api_mod
        monkeypatch.setattr(api_mod, "TWITTER_BEARER_TOKEN", "")

        from module1_twitter.twitter.api_backend import ApiBackend
        with pytest.raises(ValueError, match="TWITTER_BEARER_TOKEN"):
            ApiBackend()
