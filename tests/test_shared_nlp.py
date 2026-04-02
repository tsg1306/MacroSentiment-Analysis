import pytest
from shared.nlp.vader_sentiment import VaderSentiment
from shared.nlp.finbert_sentiment import FinBERTSentiment


REQUIRED_KEYS = {"score", "label", "confidence", "model"}


class TestVaderSentiment:
    def setup_method(self):
        self.vader = VaderSentiment()

    def test_returns_correct_keys(self):
        result = self.vader.analyze("The market is up today.")
        assert set(result.keys()) == REQUIRED_KEYS

    def test_score_range(self):
        for text in [
            "Bullish rally surge!",
            "Crash recession bearish panic!",
            "Volumes are stable.",
        ]:
            result = self.vader.analyze(text)
            assert -1.0 <= result["score"] <= 1.0, (
                f"Score {result['score']} out of range for: {text}"
            )

    def test_bullish_text_positive(self):
        result = self.vader.analyze(
            "Markets are extremely bullish, stocks surge on strong earnings rally"
        )
        assert result["label"] == "positive"

    def test_bearish_text_negative(self):
        result = self.vader.analyze(
            "Markets crash amid recession fears, bearish sentiment dominates"
        )
        assert result["label"] == "negative"

    def test_neutral_text(self):
        result = self.vader.analyze("The report was published on Monday")
        assert result["label"] == "neutral"

    def test_confidence_range(self):
        result = self.vader.analyze("Oil prices surge sharply")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_model_field(self):
        result = self.vader.analyze("test")
        assert result["model"] == "vader"

    def test_financial_lexicon_bullish(self):
        result = self.vader.analyze("bullish")
        assert result["label"] == "positive"

    def test_financial_lexicon_crash(self):
        result = self.vader.analyze("crash")
        assert result["label"] == "negative"


class TestFinBERTSentiment:
    def test_singleton(self):
        fb1 = FinBERTSentiment()
        fb2 = FinBERTSentiment()
        assert fb1 is fb2

    def test_returns_correct_keys(self):
        fb = FinBERTSentiment()
        result = fb.analyze("Oil prices surge as OPEC cuts production sharply")
        assert set(result.keys()) == REQUIRED_KEYS

    def test_score_range(self):
        fb = FinBERTSentiment()
        for text in [
            "Bullish rally surge!",
            "Crash recession bearish panic!",
            "Volumes are stable.",
        ]:
            result = fb.analyze(text)
            assert -1.0 <= result["score"] <= 1.0, (
                f"Score {result['score']} out of range for: {text}"
            )

    def test_confidence_range(self):
        fb = FinBERTSentiment()
        result = fb.analyze("Oil prices fall sharply")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_model_field(self):
        fb = FinBERTSentiment()
        result = fb.analyze("test")
        assert result["model"] == "finbert"

    def test_label_values(self):
        fb = FinBERTSentiment()
        result = fb.analyze("Markets rally strongly on positive earnings")
        assert result["label"] in ("positive", "negative", "neutral")

    def test_bullish_text(self):
        fb = FinBERTSentiment()
        result = fb.analyze(
            "Markets are extremely bullish, stocks surge on strong earnings rally"
        )
        assert result["label"] == "positive"

    def test_bearish_text(self):
        fb = FinBERTSentiment()
        result = fb.analyze(
            "Markets crash amid recession fears, bearish sentiment dominates"
        )
        assert result["label"] == "negative"
