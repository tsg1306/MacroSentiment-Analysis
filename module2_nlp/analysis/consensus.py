"""
Two approaches for detecting consensus and divergences:
1. Lexical markers (fast, for individual tweets)
2. Inter-source variance on same entity (for aggregated corpus + tweets)
"""
import numpy as np
from collections import defaultdict

CONSENSUS_MARKERS = [
    "confirms", "as expected", "in line with", "consensus",
    "widely expected", "broadly anticipated", "reaffirms",
    "consistent with", "unanimous", "all agree",
]

DIVERGENCE_MARKERS = [
    "contrary to", "surprisingly", "unexpectedly", "despite",
    "however", "risks to the upside", "risks to the downside",
    "at odds with", "challenges", "pushback", "diverges",
    "but wait", "yet markets", "contradicts",
]

WEAK_SIGNAL_MARKERS = [
    "first time since", "unusual", "historic", "unprecedented",
    "quietly", "under the radar", "few noticed", "overlooked",
    "little-noticed", "rarely discussed", "record",
]


def classify_tweet(text: str) -> dict:
    """
    Classifies a single tweet text.
    Returns {label, consensus_score, divergence_score, signal_score}
    label: consensus | divergence | signal_faible | neutral
    """
    text_lower = text.lower()
    c = sum(1 for m in CONSENSUS_MARKERS if m in text_lower)
    d = sum(1 for m in DIVERGENCE_MARKERS if m in text_lower)
    s = sum(1 for m in WEAK_SIGNAL_MARKERS if m in text_lower)

    if s > 0:
        label = "signal_faible"
    elif c > d:
        label = "consensus"
    elif d > c:
        label = "divergence"
    else:
        label = "neutral"

    return {"label": label, "consensus_score": c, "divergence_score": d, "signal_score": s}


def detect_divergences(entity_signals: list, threshold: float = 0.4) -> list:
    """
    entity_signals: [{"entity": str, "source": str, "score": float}, ...]
    Returns divergences where max_score - min_score > threshold, sorted by delta desc.
    Returns: [{entity, doc_a, score_a, doc_b, score_b, delta}, ...]
    """
    by_entity = defaultdict(list)
    for sig in entity_signals:
        by_entity[sig["entity"].lower()].append({
            "source": sig["source"],
            "score":  sig["score"],
        })

    divergences = []
    for entity, items in by_entity.items():
        if len(items) < 2:
            continue
        scores = [i["score"] for i in items]
        delta = max(scores) - min(scores)
        if delta > threshold:
            max_item = max(items, key=lambda x: x["score"])
            min_item = min(items, key=lambda x: x["score"])
            divergences.append({
                "entity":  entity,
                "doc_a":   max_item["source"],
                "score_a": round(max_item["score"], 3),
                "doc_b":   min_item["source"],
                "score_b": round(min_item["score"], 3),
                "delta":   round(delta, 3),
            })

    return sorted(divergences, key=lambda x: x["delta"], reverse=True)


def detect_consensus(entity_signals: list, min_sources: int = 2,
                     max_std: float = 0.20, min_abs_mean: float = 0.25) -> list:
    """
    Returns entities where multiple sources agree (low std, strong signal).
    Returns: [{entity, mean_score, std, n_sources, direction}, ...]
    """
    by_entity = defaultdict(list)
    for sig in entity_signals:
        by_entity[sig["entity"].lower()].append(sig["score"])

    consensus = []
    for entity, scores in by_entity.items():
        if len(scores) < min_sources:
            continue
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std <= max_std and abs(mean) >= min_abs_mean:
            consensus.append({
                "entity":     entity,
                "mean_score": round(mean, 3),
                "std":        round(std, 3),
                "n_sources":  len(scores),
                "direction":  "bullish" if mean > 0 else "bearish",
            })

    return sorted(consensus, key=lambda x: abs(x["mean_score"]), reverse=True)
