"""
Computes alignment between tweet sentiment and corpus stance on predefined themes.
Uses mean cosine similarity between tweet embeddings and corpus chunk embeddings.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_embed_model = None

THEMES = {
    "Oil / Energy":      ["oil", "crude", "WTI", "OPEC", "Brent", "petroleum", "Hormuz", "barrel"],
    "Gold / Safe Haven": ["gold", "XAU", "bullion", "safe haven", "precious metal"],
    "Fed / US Rates":    ["Fed", "Federal Reserve", "rates", "inflation", "CPI", "hike", "cut", "hawkish", "dovish"],
    "Geopolitics":       ["Iran", "Russia", "Israel", "sanctions", "war", "ceasefire", "conflict", "Hormuz"],
    "Equities / Risk":   ["S&P", "SPX", "NASDAQ", "equities", "stocks", "recession", "risk"],
    "China / EM":        ["China", "PBOC", "GDP", "yuan", "Beijing", "emerging markets"],
    "ECB / Europe":      ["ECB", "European", "euro", "Germany", "EUR", "eurozone"],
}


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


def compute_theme_alignment(theme_keywords: list, tweet_texts: list, corpus_theme_emb) -> float | None:
    """
    Returns cosine similarity [0,1] between tweet cluster and corpus passages on same theme.
    Returns None if fewer than 3 matching tweets.
    """
    if corpus_theme_emb is None:
        return None

    model = _get_embed_model()
    matching = [t for t in tweet_texts if any(kw.lower() in t.lower() for kw in theme_keywords)]
    if len(matching) < 3:
        return None

    tweet_embs = model.encode(matching[:50], show_progress_bar=False)
    tweet_mean = tweet_embs.mean(axis=0)

    sim = cosine_similarity([tweet_mean], [corpus_theme_emb])[0][0]
    return float(np.clip(sim, 0, 1))


def get_all_theme_alignments(tweet_texts: list) -> dict:
    """
    Returns {theme_name: alignment_score} for all predefined themes.
    Requires corpus to be ingested in ChromaDB.
    """
    from module2_nlp.analysis.corpus_store import get_corpus_theme_embedding

    results = {}
    for theme, keywords in THEMES.items():
        try:
            corpus_emb = get_corpus_theme_embedding(keywords)
            score = compute_theme_alignment(keywords, tweet_texts, corpus_emb)
            if score is not None:
                results[theme] = score
        except Exception:
            pass
    return results
