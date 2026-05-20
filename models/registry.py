from models.loader import BaseEmbedder, NomicEmbedder, SentenceTransformerEmbedder

MODEL_REGISTRY: dict[str, dict] = {
    "bge_m3": {
        "type": "sentence_transformer",
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
}


def get_embedder(model_key: str) -> BaseEmbedder:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Available keys: {list(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_key]

    if config["type"] == "sentence_transformer":
        return SentenceTransformerEmbedder(
            model_id=config["model_id"],
            embedding_dim=config["embedding_dim"],
        )

    if config["type"] == "nomic":
        return NomicEmbedder(
            model_id=config["model_id"],
            embedding_dim=config["embedding_dim"],
        )

    raise ValueError(f"Unsupported loader type '{config['type']}' for model '{model_key}'")
