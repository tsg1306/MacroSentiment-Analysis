import warnings

_finbert_warning_shown = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class FinBERTSentiment:
    _instance = None
    _model = None
    _tokenizer = None
    _fallback_vader = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if _HAS_TRANSFORMERS:
                cls._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                cls._model = AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert"
                )
                cls._model.eval()
            else:
                global _finbert_warning_shown
                if not _finbert_warning_shown:
                    warnings.warn(
                        "transformers/torch not available. "
                        "FinBERTSentiment will use VADER as fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _finbert_warning_shown = True
                from shared.nlp.vader_sentiment import VaderSentiment
                cls._fallback_vader = VaderSentiment()
        return cls._instance

    def analyze(self, text: str) -> dict:
        if not _HAS_TRANSFORMERS:
            result = self._fallback_vader.analyze(text)
            result["model"] = "finbert"
            return result

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT labels: positive, negative, neutral
        labels = ["positive", "negative", "neutral"]
        scores = probs[0].tolist()

        max_idx = scores.index(max(scores))
        label = labels[max_idx]
        confidence = scores[max_idx]

        # Map to a single score: positive contributes +, negative contributes -
        score = scores[0] - scores[1]  # positive_prob - negative_prob

        return {
            "score": round(score, 4),
            "label": label,
            "confidence": round(confidence, 4),
            "model": "finbert",
        }
