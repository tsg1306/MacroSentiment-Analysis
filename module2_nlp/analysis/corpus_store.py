"""
ChromaDB wrapper for storing and querying corpus document chunks with embeddings.
Uses sentence-transformers/all-MiniLM-L6-v2 for embedding (singleton pattern).
Persist directory: shared/db/chroma/ relative to project root.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

_embed_model = None
_chroma_client = None
_collection = None

# Resolve persist path relative to project root
_PERSIST_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'db', 'chroma')
)
_COLLECTION_NAME = "corpus_chunks"


def _get_embed_model() -> SentenceTransformer:
    """Singleton for SentenceTransformer('all-MiniLM-L6-v2')."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


def _get_collection():
    """
    Returns the ChromaDB collection (name='corpus_chunks'), creating it if needed.
    Uses PersistentClient so data survives across sessions.
    """
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(_PERSIST_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_PERSIST_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def ingest_chunks(chunks: list[str], metadatas: list[dict], doc_id: str) -> int:
    """
    Embeds chunks with sentence-transformers, stores in ChromaDB with metadata.
    Each chunk gets an id like '{doc_id}_chunk_{i}'.
    Idempotent: uses upsert so re-ingesting the same doc_id overwrites existing chunks.

    Returns the number of chunks ingested.
    """
    if not chunks:
        return 0

    model = _get_embed_model()
    collection = _get_collection()

    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

    # Ensure each metadata dict has required fields
    enriched_metas = []
    for i, meta in enumerate(metadatas):
        m = dict(meta)
        m.setdefault("doc_id", doc_id)
        m.setdefault("chunk_index", i)
        m.setdefault("source", "unknown")
        m.setdefault("doc_type", "unknown")
        enriched_metas.append(m)

    # Upsert in batches to avoid ChromaDB payload limits
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=enriched_metas[start:end],
        )

    return len(chunks)


def semantic_search(query: str, n_results: int = 5, where: dict = None) -> list[dict]:
    """
    Embeds query, queries ChromaDB, returns list of matching chunks.
    Each result: {"text": str, "source": str, "doc_type": str, "similarity": float, "doc_id": str}

    The 'where' parameter is passed to ChromaDB's metadata filter
    (e.g. {"doc_type": "news"}).
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    model = _get_embed_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    # Cap n_results to collection size
    actual_n = min(n_results, collection.count())

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": actual_n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        return []

    output = []
    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # Collection uses cosine space: distance = 1 - cosine_similarity
            similarity = 1.0 - dist
            output.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "doc_type": meta.get("doc_type", "unknown"),
                "similarity": round(float(similarity), 4),
                "doc_id": meta.get("doc_id", "unknown"),
            })

    return output


def get_corpus_theme_embedding(keywords: list[str]) -> np.ndarray | None:
    """
    Queries ChromaDB for chunks matching the given keywords.
    Returns mean embedding vector of matching chunks, or None if no matches.

    Called by cross_source.py for theme alignment computation.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return None

    # Build a combined query from keywords for semantic search
    query_text = " ".join(keywords)

    model = _get_embed_model()
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()

    # Retrieve top chunks matching the theme
    n = min(20, collection.count())
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n,
            include=["embeddings", "distances"],
        )
    except Exception:
        return None

    if not results or not results.get("embeddings") or len(results["embeddings"]) == 0 or len(results["embeddings"][0]) == 0:
        return None

    embeddings = np.array(results["embeddings"][0])
    distances = np.array(results["distances"][0])

    # Filter to reasonably similar chunks (cosine distance < 0.8 => similarity > 0.2)
    mask = distances < 0.8
    if not mask.any():
        return None

    filtered = embeddings[mask]
    return filtered.mean(axis=0)


def get_collection_count() -> int:
    """Returns number of chunks in the ChromaDB collection."""
    collection = _get_collection()
    return collection.count()


def delete_collection():
    """Deletes the entire corpus_chunks collection (for --force re-ingest)."""
    global _chroma_client, _collection
    if _chroma_client is None:
        os.makedirs(_PERSIST_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_PERSIST_PATH)

    try:
        _chroma_client.delete_collection(name=_COLLECTION_NAME)
    except Exception:
        pass

    _collection = None
