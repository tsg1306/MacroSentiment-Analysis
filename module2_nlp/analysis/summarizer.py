"""
Extractive document summarizer using TF-IDF sentence ranking.
No LLM — pure sklearn + regex.
Priority order: explicit trades > implicit trades > key info sentences.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from module2_nlp.analysis.document_classifier import (
    EXPLICIT_TRADE_MARKERS, IMPLICIT_TRADE_MARKERS, _classify_sentence
)

MIN_SENTENCE_LEN = 25
MAX_SENTENCE_LEN = 350


def _split_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [
        s.strip() for s in raw
        if MIN_SENTENCE_LEN <= len(s.strip()) <= MAX_SENTENCE_LEN
    ]


def _rank_by_tfidf(sentences: list) -> list:
    """Returns sentences sorted by TF-IDF importance (descending)."""
    if len(sentences) < 2:
        return sentences
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=500)
        matrix = vec.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).flatten()
        ranked = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
        return [s for _, s in ranked]
    except Exception:
        return sentences


def generate_summary(text: str, max_bullets: int = 10) -> list:
    """
    Returns list of {text: str, type: "explicit"|"implicit"|"info"} dicts.
    Ordered by: explicit trades first, then implicit, then key info.
    Max length: max_bullets.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    explicit = []
    implicit = []
    info = []

    for sent in sentences:
        kind = _classify_sentence(sent)
        if kind == "explicit":
            explicit.append({"text": sent, "type": "explicit"})
        elif kind == "implicit":
            implicit.append({"text": sent, "type": "implicit"})
        else:
            info.append(sent)

    # Rank info sentences by TF-IDF
    info_ranked = _rank_by_tfidf(info)
    info_bullets = [{"text": s, "type": "info"} for s in info_ranked]

    # Build final list respecting max_bullets
    bullets = explicit + implicit + info_bullets
    return bullets[:max_bullets]
