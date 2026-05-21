from models.loader import BaseEmbedder, BM25Retriever, CohereEmbedder, NomicEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder

MODEL_REGISTRY: dict[str, dict] = {
    "bge_m3": {
        "type": "bge",
        "model_id": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768,
        "normalize_embeddings": True,
        "max_seq_length": 8192,
    },
    "nomic_v1_5": {
        "type": "nomic",
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "embedding_dim": 768,
        "normalize_embeddings": True,
        "trust_remote_code": True,
        "doc_prefix": "search_document:",
        "query_prefix": "search_query:",
    },
    "openai_3_small": {
        "type": "openai",
        "model_id": "text-embedding-3-small",
        "embedding_dim": 1536,
    },
    "cohere_en_v3": {
        "type": "cohere",
        "model_id": "embed-english-v3.0",
        "embedding_dim": 1024,
        "doc_input_type": "search_document",
        "query_input_type": "search_query",
    },
    "bm25": {
        "type": "bm25",
        "embedding_dim": 0,
    },
}


def get_model(model_key: str) -> BaseEmbedder | BM25Retriever:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Available keys: {list(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_key]

    if config["type"] == "bm25":
        return BM25Retriever()

    if config["type"] == "bge":
        return SentenceTransformerEmbedder(
            model_id=config["model_id"],
            embedding_dim=config["embedding_dim"],
        )

    if config["type"] == "nomic":
        return NomicEmbedder(
            model_id=config["model_id"],
            embedding_dim=config["embedding_dim"],
        )

    if config["type"] == "openai":
        return OpenAIEmbedder(
            model_id=config["model_id"],
            embedding_dim=config["embedding_dim"],
        )

    if config["type"] == "cohere":
        return CohereEmbedder(
            model_id=config["model_id"],
            embedding_dim=config["embedding_dim"],
        )

    raise ValueError(f"Unsupported loader type '{config['type']}' for model '{model_key}'")
