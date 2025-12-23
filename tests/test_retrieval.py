"""Tests for chunking, embeddings, and the FAISS vector store."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")

from src.retrieval.chunking import chunk_document, chunk_text  # noqa: E402
from src.retrieval.embeddings import EmbeddingModel  # noqa: E402
from src.retrieval.ingest import generate_fallback_news_docs  # noqa: E402
from src.retrieval.vector_store import FAISSVectorStore  # noqa: E402


class TestChunking:
    def test_chunk_text_respects_size(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_chunk_text_empty_returns_empty(self):
        assert chunk_text("") == []

    def test_chunk_size_must_exceed_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("some text here", chunk_size=10, overlap=10)

    def test_offline_news_corpus_has_at_least_100_scenarios(self):
        docs = generate_fallback_news_docs()
        assert len(docs) >= 100
        assert len({doc["doc_id"] for doc in docs}) == len(docs)
        assert all(doc["text"].startswith("Synthetic scenario:") for doc in docs)

    def test_chunk_document_carries_metadata(self):
        chunks = chunk_document("doc1", "test_source", "a " * 300, metadata={"foo": "bar"})
        assert all(c.doc_id == "doc1" for c in chunks)
        assert all(c.metadata == {"foo": "bar"} for c in chunks)


class TestEmbeddingFallback:
    def test_tfidf_fallback_produces_consistent_dim(self):
        embedder = EmbeddingModel(fallback_dim=16, backend="tfidf-svd")
        corpus = [
            "port congestion delays shipping vessels",
            "supplier reliability certification audit",
            "warehouse capacity throughput logistics",
            "manufacturer downstream production impact",
        ]
        embedder.fit(corpus)
        emb = embedder.encode(corpus)
        assert emb.shape[0] == len(corpus)
        assert emb.shape[1] == embedder.dim


class TestVectorStore:
    def test_add_and_search_returns_nearest(self):
        embedder = EmbeddingModel(fallback_dim=8, backend="tfidf-svd")
        texts = ["apple banana fruit", "car truck vehicle", "banana apple smoothie"]
        embedder.fit(texts)
        embeddings = embedder.encode(texts)

        chunks = chunk_document("d1", "s1", texts[0]) + chunk_document("d2", "s1", texts[1]) + chunk_document("d3", "s1", texts[2])
        # One chunk per short text since they're under the default chunk size.
        store = FAISSVectorStore(dim=embedder.dim)
        store.add(embeddings, chunks)

        query_emb = embedder.encode(["fruit smoothie banana"])[0]
        results = store.search(query_emb, k=2)
        assert len(results) == 2
        # The most similar result should be a fruit-related chunk, not the vehicle one.
        assert "vehicle" not in results[0][0].text

    def test_search_rejects_incompatible_embedding_dimensions(self):
        store = FAISSVectorStore(dim=4)
        store.add(np.ones((1, 4), dtype=np.float32), chunk_document("d1", "test", "sample"))
        with pytest.raises(ValueError, match="does not match store dim"):
            store.search(np.ones(3, dtype=np.float32))

    def test_search_on_empty_store_returns_empty(self):
        store = FAISSVectorStore(dim=4)
        results = store.search(np.zeros(4), k=3)
        assert results == []
