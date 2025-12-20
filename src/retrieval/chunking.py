"""Document chunking utilities for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Word-based sliding-window chunking with overlap.

    `chunk_size` and `overlap` are in whitespace-delimited tokens, which is
    a reasonable proxy for chunk length that avoids pulling in a tokenizer
    dependency just for chunk boundaries.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_document(doc_id: str, source: str, text: str, metadata: dict | None = None, chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Chunk a single document into `Chunk` records carrying source metadata."""
    metadata = metadata or {}
    texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(text=t, doc_id=doc_id, source=source, chunk_index=i, metadata=metadata)
        for i, t in enumerate(texts)
    ]
