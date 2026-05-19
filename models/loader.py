from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts and return a float32 array of shape (len(texts), embedding_dim).

        Implementations must:
        - Preserve input order.
        - Handle batching internally.
        - Return a dense np.ndarray (not a framework tensor).
        - Normalize embeddings to unit length.
        """

    def _validate_output(self, texts: list[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(texts):
            raise ValueError(
                f"Row count mismatch: {embeddings.shape[0]} embeddings for {len(texts)} texts"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Dim mismatch: got {embeddings.shape[1]}, expected {self.embedding_dim}"
            )


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_id: str, embedding_dim: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            ) from e

        try:
            self._model = SentenceTransformer(model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}': {e}") from e

        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = embeddings.astype(np.float32)
        self._validate_output(texts, embeddings)
        return embeddings
