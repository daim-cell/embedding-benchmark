from abc import ABC, abstractmethod
import os
from pathlib import Path

import cohere
from dotenv import load_dotenv
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _get_api_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(
            f"API key not found. Set {env_var} in your .env file or environment."
        )
    return key


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
        self._model = SentenceTransformer(model_id, trust_remote_code=trust_remote_code)
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


class OpenAIEmbedder(BaseEmbedder):

    def __init__(self, model_id: str, embedding_dim: int) -> None:
        self._client = openai.OpenAI(api_key=_get_api_key("OPENAI_API_KEY"))
        self._model_id = model_id
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model_id)
            batch_vecs = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            # OpenAI does not return unit-normalized vectors by default.
            norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
            batch_vecs = batch_vecs / np.where(norms == 0, 1.0, norms)
            all_embeddings.append(batch_vecs)

        embeddings = np.vstack(all_embeddings)
        self._validate_output(texts, embeddings)
        return embeddings


class CohereEmbedder(BaseEmbedder):

    DOC_INPUT_TYPE = "search_document"
    QUERY_INPUT_TYPE = "search_query"
    _MAX_BATCH = 96  # Cohere hard limit per request

    def __init__(self, model_id: str, embedding_dim: int) -> None:
        self._client = cohere.Client(api_key=_get_api_key("COHERE_API_KEY"))
        self._model_id = model_id
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        input_type: str = DOC_INPUT_TYPE,
    ) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        effective_batch = min(batch_size, self._MAX_BATCH)
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), effective_batch):
            batch = texts[i : i + effective_batch]
            response = self._client.embed(
                texts=batch,
                model=self._model_id,
                input_type=input_type,
            )
            batch_vecs = np.array(response.embeddings, dtype=np.float32)
            # Cohere v3 embeddings are unit-normalized; normalize defensively.
            norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
            batch_vecs = batch_vecs / np.where(norms == 0, 1.0, norms)
            all_embeddings.append(batch_vecs)

        embeddings = np.vstack(all_embeddings)
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
