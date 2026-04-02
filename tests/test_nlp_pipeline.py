import sys
import os
import pytest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMockDocuments:
    def test_returns_six_docs(self):
        from module2_nlp.ingestion.mock_documents import get_mock_documents
        docs = get_mock_documents()
        assert len(docs) == 6

    def test_required_keys(self):
        from module2_nlp.ingestion.mock_documents import get_mock_documents
        required = ["title", "source", "doc_type", "published_at", "text"]
        for doc in get_mock_documents():
            assert all(k in doc for k in required), f"Missing keys in {doc.get('title')}"

    def test_text_length(self):
        from module2_nlp.ingestion.mock_documents import get_mock_documents
        for doc in get_mock_documents():
            assert len(doc["text"]) >= 300, f"Text too short: {doc.get('title')}"

    def test_valid_doc_types(self):
        from module2_nlp.ingestion.mock_documents import get_mock_documents
        valid = {"news", "hedge_fund_letter", "earnings_call", "fomc_minutes", "research_note"}
        for doc in get_mock_documents():
            assert doc["doc_type"] in valid

    def test_published_at_is_datetime(self):
        from module2_nlp.ingestion.mock_documents import get_mock_documents
        for doc in get_mock_documents():
            assert isinstance(doc["published_at"], datetime)


class TestTxtParser:
    def test_returns_correct_format(self):
        from module2_nlp.ingestion.txt_parser import TxtParser
        parser = TxtParser()
        result = parser.parse("Oil prices fell sharply after the ceasefire.")
        required = ["title", "source", "doc_type", "published_at", "text"]
        assert all(k in result for k in required)
        assert len(result["text"]) > 0


class TestChunker:
    def test_short_text_single_chunk(self):
        from module2_nlp.nlp.chunker import chunk_text
        text = "Oil prices fell. Gold surged."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        from module2_nlp.nlp.chunker import chunk_text
        long_text = " ".join(
            ["Oil prices fell sharply. OPEC cut production. Gold surged on safe haven demand."] * 30
        )
        chunks = chunk_text(long_text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.split()) <= 260, f"Chunk too large: {len(c.split())} words"

    def test_empty_text(self):
        from module2_nlp.nlp.chunker import chunk_text
        assert chunk_text("") == []


class TestNER:
    def test_finds_financial_entities(self):
        from module2_nlp.nlp.ner import extract_entities_with_context
        text = (
            "We remain bullish on crude oil given persistent supply constraints from OPEC. "
            "Gold remains our preferred safe haven. The Federal Reserve is tightening."
        )
        entities = extract_entities_with_context(text)
        names = [e["entity"].lower() for e in entities]
        assert any("oil" in n or "crude" in n for n in names), f"Oil not found: {names}"
        assert any("gold" in n for n in names), f"Gold not found: {names}"

    def test_no_duplicates(self):
        from module2_nlp.nlp.ner import extract_entities_with_context
        text = "Gold prices surged. Gold is a safe haven. Gold rallied strongly."
        entities = extract_entities_with_context(text)
        seen = [(e["entity"].lower(), e["entity_type"]) for e in entities]
        assert len(seen) == len(set(seen)), "Duplicates found"

    def test_context_text_present(self):
        from module2_nlp.nlp.ner import extract_entities_with_context
        text = "Oil prices fell sharply as OPEC cut production significantly."
        entities = extract_entities_with_context(text)
        for e in entities:
            assert "context_text" in e
            assert len(e["context_text"]) > 5


class TestPipeline:
    def test_returns_correct_structure(self):
        from module2_nlp.nlp.pipeline import process_document
        result = process_document(
            "Oil prices surged as OPEC announced major cuts. Gold rallied on safe haven demand.",
            model="vader"
        )
        assert "document" in result
        assert "signals" in result
        assert "id" in result["document"]

    def test_idempotent(self):
        from module2_nlp.nlp.pipeline import process_document
        text = "Gold surged. Oil fell. Markets reacted."
        r1 = process_document(text, model="vader")
        r2 = process_document(text, model="vader")
        assert r1["document"]["id"] == r2["document"]["id"]


class TestAggregator:
    def test_empty_signals(self):
        from module2_nlp.signal.aggregator import aggregate_signals
        result = aggregate_signals([])
        assert result == {"signal": 0.0, "n_docs": 0, "total_mentions": 0}

    def test_filter_by_entity(self):
        from module2_nlp.signal.aggregator import aggregate_signals
        sigs = [
            {"entity": "oil", "signal_value": -0.5, "mention_count": 3, "doc_type": "news"},
            {"entity": "gold", "signal_value": 0.8, "mention_count": 2, "doc_type": "news"},
        ]
        result = aggregate_signals(sigs, entity="oil")
        assert result["n_docs"] == 1
        assert result["signal"] == -0.5

    def test_filter_by_doc_type(self):
        from module2_nlp.signal.aggregator import aggregate_signals
        sigs = [
            {"entity": "oil", "signal_value": -0.5, "mention_count": 3, "doc_type": "news"},
            {"entity": "oil", "signal_value": 0.3, "mention_count": 2, "doc_type": "research_note"},
        ]
        result = aggregate_signals(sigs, entity="oil", doc_type_filter="news")
        assert result["n_docs"] == 1

    def test_weights_by_mention_count(self):
        from module2_nlp.signal.aggregator import aggregate_signals
        sigs = [
            {"entity": "oil", "signal_value": 1.0, "mention_count": 10},
            {"entity": "oil", "signal_value": -1.0, "mention_count": 1},
        ]
        result = aggregate_signals(sigs, entity="oil")
        # Weighted: (1.0*10 + (-1.0)*1) / 11 ≈ 0.818
        assert result["signal"] > 0.5
