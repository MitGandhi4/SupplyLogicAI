"""High-level retrieval interface used by the LangGraph agent and FastAPI backend."""
from __future__ import annotations

from src.logging_config import get_logger
from src.retrieval.embeddings import DEFAULT_MODEL_DIM, DEFAULT_MODEL_NAME, EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore

logger = get_logger(__name__)


class Retriever:
    """Wraps a fitted embedder + FAISS store for simple `retrieve(query)` calls."""

    def __init__(self):
        self.store = FAISSVectorStore.load()
        backend = self.store.embedding_backend
        if backend is None:
            backend = "sentence-transformers" if self.store.dim == DEFAULT_MODEL_DIM else "tfidf-svd"
        self.embedder = EmbeddingModel(
            model_name=self.store.embedding_model or DEFAULT_MODEL_NAME,
            backend=backend,
        )
        # The TF-IDF fallback embedder must be re-fit on the same corpus used
        # at ingestion time to produce a compatible vector space. We refit on
        # the stored chunk texts, which reproduces the ingestion-time fit
        # deterministically (TfidfVectorizer + TruncatedSVD are deterministic
        # given the same corpus and random_state).
        if self.embedder.backend == "tfidf-svd":
            self.embedder.fit([c.text for c in self.store.chunks])
        if self.embedder.dim != self.store.dim:
            raise ValueError(
                f"Embedding dimension {self.embedder.dim} does not match vector store dimension {self.store.dim}"
            )

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_emb = self.embedder.encode([query])[0]
        results = self.store.search(query_emb, k=k)
        return [
            {
                "text": chunk.text,
                "source": chunk.source,
                "doc_id": chunk.doc_id,
                "score": score,
                "metadata": chunk.metadata,
            }
            for chunk, score in results
        ]
