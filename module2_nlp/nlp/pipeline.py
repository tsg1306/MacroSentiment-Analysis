import sys
import os
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import ALERT_THRESHOLD
from module2_nlp.nlp.chunker import chunk_text
from module2_nlp.nlp.ner import extract_entities_with_context
from shared.db.database import (
    init_db, save_document, save_entity_sentiment, save_doc_signal
)


def process_document(text: str, file_type: str = "txt", doc_type: str = "news",
                     model: str = "vader", title: str = None, source: str = None,
                     published_at=None) -> dict:
    """
    Full pipeline: parse → chunk → NER → sentiment → save to DB.
    Idempotent on SHA256 doc_id.
    """
    # 1. Sentiment analyzer
    if model == "finbert":
        from shared.nlp.finbert_sentiment import FinBERTSentiment
        analyzer = FinBERTSentiment()
    else:
        from shared.nlp.vader_sentiment import VaderSentiment
        analyzer = VaderSentiment()

    # 2. Generate doc_id
    doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # 3. Init DB
    init_db()

    # 4. Save document
    doc_record = {
        "id": doc_id,
        "title": title or text[:50].strip(),
        "source": source or "unknown",
        "doc_type": doc_type,
        "published_at": published_at,
        "char_count": len(text),
    }
    save_document(doc_record)

    # 5. Chunk text
    chunks = chunk_text(text)

    # 6. NER + sentiment per chunk
    entity_scores = {}  # entity_key -> list of (score, label, confidence, context)

    for chunk in chunks:
        entities = extract_entities_with_context(chunk)
        for ent in entities:
            result = analyzer.analyze(ent["context_text"])

            # Save entity sentiment
            save_entity_sentiment({
                "document_id": doc_id,
                "entity": ent["entity"],
                "entity_type": ent["entity_type"],
                "context_text": ent["context_text"],
                "sentiment_score": result["score"],
                "sentiment_label": result["label"],
                "model_used": result["model"],
                "confidence": result["confidence"],
            })

            key = ent["entity"].lower()
            if key not in entity_scores:
                entity_scores[key] = []
            entity_scores[key].append(result["score"])

    # 7. Aggregate signals per entity
    signals = {}
    for entity_key, scores in entity_scores.items():
        signal_value = sum(scores) / len(scores) if scores else 0.0
        mention_count = len(scores)
        alert = abs(signal_value) >= ALERT_THRESHOLD

        save_doc_signal({
            "document_id": doc_id,
            "entity": entity_key,
            "signal_value": signal_value,
            "mention_count": mention_count,
            "alert": alert,
            "published_at": published_at,
        })

        signals[entity_key] = {
            "signal_value": signal_value,
            "mention_count": mention_count,
            "alert": alert,
        }

    return {
        "document": doc_record,
        "signals": signals,
    }
