from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
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

    def __init__(self, model_id: str, embedding_dim: int, trust_remote_code: bool = False) -> None:

        try:
            self._model = SentenceTransformer(model_id, trust_remote_code=trust_remote_code)
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


class NomicEmbedder(SentenceTransformerEmbedder):
    """Embedder for nomic-ai/nomic-embed-text-v1.5.

    Nomic requires a task-instruction prefix on every input text.
    Use 'search_document:' for corpus docs and 'search_query:' for queries.
    """

    DOC_PREFIX = "search_document:"
    QUERY_PREFIX = "search_query:"

    def __init__(self, model_id: str, embedding_dim: int) -> None:
        super().__init__(model_id, embedding_dim, trust_remote_code=True)

    def embed(self, texts: list[str], batch_size: int = 32, prefix: str = DOC_PREFIX) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        prefixed = [f"{prefix} {t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = embeddings.astype(np.float32)
        self._validate_output(texts, embeddings)
        return embeddings
