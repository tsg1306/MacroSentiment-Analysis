import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into chunks of ~chunk_size words respecting sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_words = []
    current_count = 0

    for sentence in sentences:
        s_words = sentence.split()
        if current_count + len(s_words) > chunk_size and current_count > 0:
            chunks.append(" ".join(current_words))
            # Keep overlap words from end of current chunk
            overlap_words = current_words[-overlap:] if overlap < len(current_words) else current_words
            current_words = list(overlap_words)
            current_count = len(current_words)
        current_words.extend(s_words)
        current_count += len(s_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
