from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderSentiment:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        FINANCE_LEXICON = {
            "bullish": 3.0,
            "bearish": -3.0,
            "crash": -3.5,
            "surge": 2.5,
            "ceasefire": 1.5,
            "sanctions": -2.0,
            "tightening": -1.5,
            "easing": 1.5,
            "upgrade": 2.0,
            "downgrade": -2.0,
            "recession": -3.0,
            "rally": 2.5,
        }
        self.analyzer.lexicon.update(FINANCE_LEXICON)

    def analyze(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound > 0.05:
            label = "positive"
        elif compound < -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "score": compound,
            "label": label,
            "confidence": abs(compound),
            "model": "vader",
        }
