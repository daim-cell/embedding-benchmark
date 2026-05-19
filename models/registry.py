from models.loader import BaseEmbedder, SentenceTransformerEmbedder

MODEL_REGISTRY: dict[str, dict] = {
    "bge_m3": {
        "type": "sentence_transformer",
        "model_id": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768,
        "normalize_embeddings": True,
        "max_seq_length": 8192,
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

    raise ValueError(f"Unsupported loader type '{config['type']}' for model '{model_key}'")
