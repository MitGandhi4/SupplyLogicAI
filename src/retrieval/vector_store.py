"""FAISS-backed vector store for supplier documents and news articles."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.config import settings
from src.logging_config import get_logger
from src.retrieval.chunking import Chunk

logger = get_logger(__name__)

INDEX_PATH = settings.vector_store_dir / "faiss.index"
METADATA_PATH = settings.vector_store_dir / "chunks_metadata.pkl"
CONFIG_PATH = settings.vector_store_dir / "store_config.json"


class FAISSVectorStore:
    """Cosine-similarity FAISS store (inner product over L2-normalized vectors)."""

    def __init__(self, dim: int, embedding_backend: str | None = None, embedding_model: str | None = None):
        self.dim = dim
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vectors / norms

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {embeddings.shape[1]} does not match store dim {self.dim}")
        normalized = self._normalize(embeddings.astype(np.float32))
        self.index.add(normalized)
        self.chunks.extend(chunks)
        logger.info("Added %d chunks to vector store (total=%d)", len(chunks), len(self.chunks))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty; returning no results")
            return []
        if query_embedding.size != self.dim:
            raise ValueError(f"Query embedding dim {query_embedding.size} does not match store dim {self.dim}")
        q = self._normalize(query_embedding.reshape(1, -1).astype(np.float32))
        scores, indices = self.index.search(q, min(k, self.index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH, config_path: Path = CONFIG_PATH) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(config_path, "w") as f:
            json.dump({
                "dim": self.dim,
                "n_chunks": len(self.chunks),
                "embedding_backend": self.embedding_backend,
                "embedding_model": self.embedding_model,
            }, f)
        logger.info("Saved vector store (%d chunks) to %s", len(self.chunks), index_path)

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH, config_path: Path = CONFIG_PATH) -> "FAISSVectorStore":
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {index_path}. Run `python -m src.retrieval.ingest` first."
            )
        with open(config_path) as f:
            config = json.load(f)
        store = cls(
            dim=config["dim"],
            embedding_backend=config.get("embedding_backend"),
            embedding_model=config.get("embedding_model"),
        )
        store.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            store.chunks = pickle.load(f)
        logger.info("Loaded vector store with %d chunks from %s", len(store.chunks), index_path)
        return store
