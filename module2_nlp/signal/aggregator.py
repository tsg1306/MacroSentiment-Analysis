def aggregate_signals(signals: list, entity: str = None,
                      doc_type_filter: str = None) -> dict:
    """
    Aggregate signal records across multiple documents.
    Weight by mention_count.
    """
    filtered = signals

    if entity:
        entity_lower = entity.lower()
        filtered = [s for s in filtered if s.get("entity", "").lower() == entity_lower]

    if doc_type_filter:
        filtered = [s for s in filtered if s.get("doc_type") == doc_type_filter]

    if not filtered:
        return {"signal": 0.0, "n_docs": 0, "total_mentions": 0}

    total_weighted = sum(
        s.get("signal_value", 0) * s.get("mention_count", 1) for s in filtered
    )
    total_mentions = sum(s.get("mention_count", 1) for s in filtered)
    n_docs = len(filtered)

    signal = total_weighted / total_mentions if total_mentions > 0 else 0.0

    return {
        "signal": signal,
        "n_docs": n_docs,
        "total_mentions": total_mentions,
    }
