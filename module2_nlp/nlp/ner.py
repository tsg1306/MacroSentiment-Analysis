import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TRACKED_ENTITIES, ENTITY_CONTEXT_WINDOW

import spacy

# Load spaCy once at module level
_nlp = spacy.load("en_core_web_sm")

# Flatten tracked entities into a lookup set
_TRACKED_FLAT = {}
for category, entities in TRACKED_ENTITIES.items():
    for e in entities:
        _TRACKED_FLAT[e.lower()] = category


def extract_entities_with_context(text: str) -> list:
    """
    Combine spaCy NER with TRACKED_ENTITIES from config.
    Deduplicate by (entity.lower(), entity_type).
    Returns list of {"entity": str, "entity_type": str, "context_text": str}.
    """
    doc = _nlp(text)
    tokens = [t.text for t in doc]
    seen = set()
    results = []

    # 1. spaCy NER entities
    valid_labels = {"ORG", "PERSON", "GPE", "PRODUCT", "MONEY"}
    for ent in doc.ents:
        if ent.label_ not in valid_labels:
            continue
        key = (ent.text.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        context = _get_context(tokens, ent.start, ent.end)
        results.append({
            "entity": ent.text,
            "entity_type": ent.label_,
            "context_text": context,
        })

    # 2. TRACKED_ENTITIES (keyword matching)
    text_lower = text.lower()
    for entity_str, category in _TRACKED_FLAT.items():
        if entity_str not in text_lower:
            continue
        key = (entity_str, category)
        if key in seen:
            continue
        # Check if already captured by spaCy under a different type
        if any(entity_str == s[0] for s in seen):
            continue
        seen.add(key)
        # Find position in tokens
        idx = _find_token_index(tokens, entity_str)
        context = _get_context(tokens, idx, idx + len(entity_str.split()))
        results.append({
            "entity": entity_str,
            "entity_type": category,
            "context_text": context,
        })

    return results


def _find_token_index(tokens, entity_str):
    """Find approximate token index for an entity string."""
    entity_lower = entity_str.lower()
    entity_parts = entity_lower.split()
    for i in range(len(tokens)):
        if tokens[i].lower().startswith(entity_parts[0][:3]):
            # Check if this looks like a match
            candidate = " ".join(tokens[i:i + len(entity_parts)]).lower()
            if entity_lower in candidate or candidate in entity_lower:
                return i
    # Fallback: search by character position
    text_lower = " ".join(tokens).lower()
    char_pos = text_lower.find(entity_lower)
    if char_pos >= 0:
        prefix = text_lower[:char_pos]
        return len(prefix.split())
    return 0


def _get_context(tokens, start, end):
    """Extract context window around entity position."""
    half = ENTITY_CONTEXT_WINDOW // 2
    ctx_start = max(0, start - half)
    ctx_end = min(len(tokens), end + half)
    return " ".join(tokens[ctx_start:ctx_end])
