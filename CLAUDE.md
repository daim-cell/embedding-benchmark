# CLAUDE.md

This file provides guidance for coding agents working on the Embedding Model Benchmarking Lab.

## Project Goal

Build a reproducible benchmarking lab that compares multiple embedding models on retrieval quality, latency, dimensionality, cost, and embedding-space structure. The project should also include a lightweight contrastive adapter training workflow and before/after retrieval evaluation.

Keep model references generic in documentation, comments, and examples unless the user explicitly asks for concrete model names.

## Core Requirements

The project must support:

1. Loading multiple embedding models through a unified interface.
2. Running the same retrieval benchmark across all models.
3. Computing cosine similarity, MRR@10, and NDCG@10.
4. Measuring total encode time and average latency per embedding.
5. Saving one JSON result file per model.
6. Aggregating results into a comparison CSV.
7. Creating 2D UMAP visualizations of embedding spaces.
8. Training a contrastive adapter on paired domain data.
9. Re-running retrieval benchmarks after adapter training.
10. Producing a final Jupyter notebook report.

## Preferred Repository Structure

```text
benchmarks/
  run_benchmark.py
  compare.py
  visualize_umap.py
models/
  loader.py
  registry.py
scripts/
  download_data.py
training/
  train_adapter.py
  evaluate_adapter.py
notebooks/
  final_report.ipynb
data/
results/
requirements.txt
README.md
CLAUDE.md
```

## Coding Guidelines

- Use Python 3.10+.
- Prefer simple, readable code over complex abstractions.
- Keep the benchmark deterministic where possible.
- Add CLI arguments with `argparse`.
- Use `pathlib.Path` instead of raw string paths.
- Use `tqdm` for long-running loops.
- Use `numpy` arrays for embeddings.
- Normalize embeddings before cosine-similarity search.
- Save intermediate and final outputs under `results/`.
- Do not commit downloaded datasets, model weights, caches, or generated result files unless explicitly requested.

## Model Loader Contract

All model loaders should expose this interface:

```python
def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Return a 2D numpy array with shape `(len(texts), embedding_dim)`."""
```

Expected behavior:

- Accept a list of strings.
- Preserve input order.
- Return a dense 2D `np.ndarray`.
- Use `float32` where possible.
- Avoid returning framework-specific tensors from public loader methods.
- Handle batching internally.
- Raise clear errors when a model cannot be loaded.

## Model Registry Guidelines

Use a registry so benchmark code can select models by string key.

Example pattern:

```python
MODEL_REGISTRY = {
    "model_a": {
        "type": "local_transformer",
        "path": "placeholder-model-path",
        "requires_remote_code": False,
    },
    "model_b": {
        "type": "api_provider",
        "path": "placeholder-api-model",
        "requires_remote_code": False,
    },
}
```

Do not hardcode benchmark logic for a specific model. The benchmark should call the shared loader interface only.


## Benchmarking Logic

`benchmarks/run_benchmark.py` should:

1. Parse CLI arguments.
2. Load dataset files.
3. Load the selected model.
4. Encode corpus documents in batches.
5. Encode queries in batches.
6. L2-normalize corpus and query embeddings.
7. Build a flat vector index.
8. Retrieve top-k corpus IDs for each query.
9. Convert retrieval output into the format expected by the metric library.
10. Compute MRR@10 and NDCG@10.
11. Save a JSON result file.

Result JSON should contain:

```json
{
  "model": "model_a",
  "embedding_dim": 768,
  "mrr_at_10": 0.0,
  "ndcg_at_10": 0.0,
  "avg_latency_per_embedding_ms": 0.0,
  "total_corpus_encode_time_sec": 0.0,
  "total_query_encode_time_sec": 0.0,
  "num_corpus_documents": 0,
  "num_queries": 0,
  "top_k": 10
}
```

## Metric Guidelines

Use ranking metrics consistently:

- MRR@10
- NDCG@10

Make sure query IDs and document IDs match exactly between qrels and retrieval results. Many metric bugs come from mismatched ID types, such as integers in one place and strings in another.

## Vector Search Guidelines

For cosine similarity with a flat inner-product index:

1. Convert embeddings to `float32`.
2. L2-normalize corpus embeddings.
3. L2-normalize query embeddings.
4. Add normalized corpus embeddings to the index.
5. Search with normalized query embeddings.

Keep a stable mapping from index row number to corpus document ID.

## UMAP Guidelines

`benchmarks/visualize_umap.py` should:

- Use the same sampled corpus documents for each model when comparing models.
- Save plots under `results/`.
- Store the sampled document IDs so visualizations are reproducible.
- Avoid interpreting UMAP plots as definitive quality measurements. They are exploratory visualizations.



## CLI Expectations

Every main script should support `--help`.

Recommended smoke-test flow:

```bash
python scripts/download_data.py --dataset <dataset_name> --output-dir data/
python benchmarks/run_benchmark.py --model model_a --dataset data/<dataset_name> --limit-corpus 500 --limit-queries 50
python benchmarks/compare.py --results-dir results/ --output results/comparison.csv
```



## Common Pitfalls

- Forgetting to normalize embeddings before cosine search.
- Mixing integer and string document IDs.
- Comparing UMAP plots generated from different document samples.
- Reporting latency without batch size or hardware context.
- Letting model-specific code leak into the benchmark harness.
- Running a full-size dataset before validating the pipeline on a small subset.

## Agent Instructions

When making changes:

1. Preserve the unified model interface.
2. Keep benchmark logic model-agnostic.
3. Prefer small, testable functions.
4. Add comments only where they clarify non-obvious behavior.
5. Update README or notebook instructions when CLI behavior changes.
6. Avoid adding heavyweight GPU-based dependencies as this is for a local system.
