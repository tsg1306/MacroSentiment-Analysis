#!/usr/bin/env python3
"""
ingest_corpus.py -- Parse PDFs from data_corpus/, chunk, embed into ChromaDB, and
run NLP pipeline (NER + sentiment) into SQLite.

Usage:
    python scripts/ingest_corpus.py                  # ingest new files only
    python scripts/ingest_corpus.py --force           # re-ingest everything
    python scripts/ingest_corpus.py --model finbert   # use FinBERT instead of VADER
"""

import argparse
import glob
import hashlib
import os
import sys
import time

# ---------------------------------------------------------------------------
# Resolve project root from this script's location and add to sys.path
# so all project imports (config, shared.*, module2_nlp.*) work regardless
# of the working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from shared.db.database import (
    init_db,
    get_ingested_files,
    mark_file_ingested,
    get_session,
)
from shared.db.database import CorpusIngested
from module2_nlp.ingestion.pdf_parser import PdfParser
from module2_nlp.nlp.chunker import chunk_text
from module2_nlp.nlp.pipeline import process_document
from module2_nlp.analysis.corpus_store import ingest_chunks, delete_collection


# ---------------------------------------------------------------------------
# Heuristic to guess doc_type from filename
# ---------------------------------------------------------------------------

def guess_doc_type(filename: str) -> str:
    """
    Derive a doc_type from the PDF filename using simple keyword heuristics.
    Falls back to 'research_note' since the corpus is mostly research PDFs.
    """
    name = filename.lower()

    # Check for news-like patterns first
    if any(kw in name for kw in ("inflation", "war", "gold")):
        return "news"

    # Everything else in our corpus is a research note of some kind
    if any(kw in name for kw in (
        "weekly", "wrap", "week_ahead",
        "sector", "oil", "gas", "energy", "cascade",
        "company_update", "pharmaceuticals",
        "musings", "reflections",
        "macro", "fed", "insights",
        "markets", "market",
        "nuclear", "landscape",
        "spaces", "upcoming",
        "strategy", "desk",
    )):
        return "research_note"

    return "research_note"


# ---------------------------------------------------------------------------
# Force-clear helpers
# ---------------------------------------------------------------------------

def clear_corpus_ingested_table():
    """Delete all rows from the corpus_ingested table."""
    session = get_session()
    try:
        session.query(CorpusIngested).delete()
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def ingest(force: bool = False, model: str = "vader"):
    """
    Walk data_corpus/*.pdf, parse, chunk, embed (ChromaDB), run NLP pipeline
    (NER + sentiment -> SQLite), and mark each file as ingested.

    With --force, the ChromaDB collection and corpus_ingested records are
    wiped first so every file is reprocessed.
    """
    pdf_dir = os.path.join(PROJECT_ROOT, "data_corpus")
    pdf_pattern = os.path.join(pdf_dir, "*.pdf")
    pdf_files = sorted(glob.glob(pdf_pattern))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    # Initialise SQLite tables
    init_db()

    # Handle --force: wipe ChromaDB collection + corpus_ingested table
    if force:
        print("--force: deleting ChromaDB collection and clearing corpus_ingested table...")
        delete_collection()
        clear_corpus_ingested_table()

    already_ingested = get_ingested_files()

    parser = PdfParser()
    total_chunks = 0
    ingested_count = 0
    skipped_count = 0
    error_count = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)

        # Skip if already processed (and not forcing)
        if filename in already_ingested and not force:
            print(f"  Skip (already ingested): {filename}")
            skipped_count += 1
            continue

        try:
            t0 = time.time()

            # 1. Parse PDF
            doc = parser.parse(pdf_path)
            text = doc.get("text", "")
            if not text or len(text.strip()) < 50:
                print(f"  WARN: empty/too-short text from {filename}, skipping")
                error_count += 1
                continue

            # 2. Determine doc_type from filename
            doc_type = guess_doc_type(filename)

            # 3. Generate doc_id = sha256(text)[:16]
            doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

            # 4. Chunk text
            chunks = chunk_text(text)
            if not chunks:
                print(f"  WARN: no chunks produced from {filename}, skipping")
                error_count += 1
                continue

            # 5. Prepare metadata for ChromaDB
            metadatas = [
                {
                    "source": filename,
                    "doc_type": doc_type,
                    "doc_id": doc_id,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]

            # 6. Store in ChromaDB
            ingest_chunks(chunks, metadatas, doc_id)

            # 7. Run NLP pipeline (NER + sentiment -> SQLite)
            process_document(
                text,
                file_type="pdf",
                doc_type=doc_type,
                model=model,
                title=doc.get("title", filename),
                source=filename,
                published_at=doc.get("published_at"),
            )

            # 8. Mark as ingested in SQLite
            mark_file_ingested(filename)

            elapsed = time.time() - t0
            total_chunks += len(chunks)
            ingested_count += 1
            print(f"  Ingested: {filename} ({len(chunks)} chunks, {elapsed:.1f}s)")

        except Exception as exc:
            print(f"  ERROR processing {filename}: {exc}")
            error_count += 1
            continue

    # Summary
    print()
    print("=" * 60)
    print(f"  Total PDF files : {len(pdf_files)}")
    print(f"  Ingested        : {ingested_count}")
    print(f"  Skipped         : {skipped_count}")
    print(f"  Errors          : {error_count}")
    print(f"  Total chunks    : {total_chunks}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDFs from data_corpus/ into ChromaDB + SQLite.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest all files (delete ChromaDB collection + clear corpus_ingested table).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vader",
        choices=["vader", "finbert"],
        help="Sentiment model to use (default: vader).",
    )
    args = parser.parse_args()
    ingest(force=args.force, model=args.model)


if __name__ == "__main__":
    main()
