from abc import ABC, abstractmethod
import os
import pickle
from pathlib import Path

import cohere
from dotenv import load_dotenv
import numpy as np
import openai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode query texts. Override in subclasses that need query-specific behavior."""
        return self.embed(texts, batch_size)

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

    def __init__(
        self,
        model_id: str,
        embedding_dim: int,
        trust_remote_code: bool = False,
        max_seq_length: int | None = None,
    ) -> None:
        self._model = SentenceTransformer(model_id, trust_remote_code=trust_remote_code)
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
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

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.embed(texts, batch_size, input_type=self.QUERY_INPUT_TYPE)


class NomicEmbedder(SentenceTransformerEmbedder):
    """Embedder for nomic-ai/nomic-embed-text-v1.5.

    Nomic requires a task-instruction prefix on every input text.
    Use 'search_document:' for corpus docs and 'search_query:' for queries.
    """

    DOC_PREFIX = "search_document:"
    QUERY_PREFIX = "search_query:"

    def __init__(self, model_id: str, embedding_dim: int, max_seq_length: int | None = None) -> None:
        super().__init__(model_id, embedding_dim, trust_remote_code=True, max_seq_length=max_seq_length)

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

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.embed(texts, batch_size, prefix=self.QUERY_PREFIX)


class BM25Retriever:
    """Sparse BM25 retrieval baseline. Not an embedder — fits on corpus and scores queries directly."""

    embedding_dim = 0

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []

    def fit(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        tokenized = [t.lower().split() for t in corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus_ids = corpus_ids

    def retrieve(
        self, query_texts: list[str], query_ids: list[str], top_k: int
    ) -> dict[str, dict[str, float]]:
        run: dict[str, dict[str, float]] = {}
        for qid, qtext in zip(query_ids, tqdm(query_texts, desc="BM25 retrieval", leave=False)):
            tokens = qtext.lower().split()
            scores = self._bm25.get_scores(tokens)
            top_indices = np.argsort(-scores)[:top_k]
            run[qid] = {self._corpus_ids[i]: float(scores[i]) for i in top_indices}
        return run

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "corpus_ids": self._corpus_ids}, f)

    @classmethod
    def load_from(cls, path: Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._bm25 = state["bm25"]
        obj._corpus_ids = state["corpus_ids"]
        return obj
