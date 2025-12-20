"""
Text embedding backend for the RAG pipeline.

Primary backend is `sentence-transformers` (all-MiniLM-L6-v2), which gives
strong semantic retrieval quality. If sentence-transformers / torch is not
installed or fails to load (e.g. on a disk- or memory-constrained machine),
the embedder gracefully degrades to a TF-IDF + Truncated SVD embedding so
the RAG pipeline still runs end-to-end, just with weaker semantic recall.
This fallback is logged loudly so it is never silently masking a real
production issue.
"""
from __future__ import annotations

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_DIM = 384
FALLBACK_DIM = 256


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        fallback_dim: int = FALLBACK_DIM,
        backend: str | None = None,
    ):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.backend: str
        self._st_model = None
        self._vectorizer = None
        self._svd = None
        self._fitted = False
        self.dim: int

        if backend not in (None, "sentence-transformers", "tfidf-svd"):
            raise ValueError(f"Unsupported embedding backend: {backend}")

        if backend != "tfidf-svd":
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(model_name)
                self.dim = self._st_model.get_sentence_embedding_dimension()
                self.backend = "sentence-transformers"
                self._fitted = True
                logger.info("Loaded sentence-transformers embedding model '%s' (dim=%d)", model_name, self.dim)
                return
            except Exception as exc:
                if backend == "sentence-transformers":
                    raise RuntimeError(f"Could not load the configured embedding model '{model_name}'") from exc
                logger.warning(
                    "sentence-transformers unavailable (%s). Falling back to TF-IDF+SVD embeddings. "
                    "Install `sentence-transformers` for higher-quality retrieval.", exc,
                )
        self._init_tfidf()

    def _init_tfidf(self) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        self._svd = TruncatedSVD(n_components=self.fallback_dim, random_state=42)
        self.dim = self.fallback_dim
        self.backend = "tfidf-svd"
        logger.info("Initialized TF-IDF+SVD fallback embedding backend")

    def fit(self, corpus: list[str]) -> None:
        """Only required for the TF-IDF fallback backend; no-op otherwise."""
        if self.backend != "tfidf-svd":
            return
        if len(corpus) < 2:
            corpus = corpus + ["placeholder document for fitting"]
        tfidf = self._vectorizer.fit_transform(corpus)
        n_components = min(self.fallback_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_components = max(n_components, 1)
        if n_components != self._svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)
        self.dim = n_components
        self._fitted = True
        logger.info("Fit TF-IDF+SVD fallback embedder on %d docs (dim=%d)", len(corpus), self.dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.backend == "sentence-transformers":
            emb = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return emb.astype(np.float32)

        if not self._fitted:
            raise RuntimeError("TF-IDF fallback embedder must be `.fit()` on a corpus before `.encode()`")
        tfidf = self._vectorizer.transform(texts)
        emb = self._svd.transform(tfidf)
        return emb.astype(np.float32)
