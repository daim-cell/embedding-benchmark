# CODEX.md

This file provides guidance for Codex when working on the Embedding Model Benchmarking Lab.

## Project Goal

Build a reproducible Python lab for comparing embedding models on retrieval quality, latency, dimensionality, cost, and embedding-space behavior. The project should also support a lightweight contrastive adapter training workflow, then compare retrieval performance before and after adapter training.

Keep examples and docs model-agnostic unless the user explicitly asks for concrete model names.

## Expected Structure

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
CODEX.md
```

Local datasets should live under `data/`. Benchmark outputs, plots, CSVs, and other generated artifacts should live under `results/`. Do not commit downloaded datasets, model weights, caches, or generated results unless the user explicitly requests it.

## Environment

Use Python 3.10 or newer.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Prefer CPU-friendly dependencies for local development. Avoid adding heavyweight GPU-only requirements unless the user asks for them.

## Core Workflow

The project should support:

1. Downloading or preparing a labeled retrieval dataset with `corpus`, `queries`, and `qrels`.
2. Loading embedding models through a unified model interface.
3. Running the same benchmark across all registered models.
4. Computing cosine-similarity retrieval metrics, especially MRR@10 and NDCG@10.
5. Measuring corpus/query encoding time and average latency per embedding.
6. Saving one JSON result file per model.
7. Aggregating benchmark JSON files into a comparison CSV.
8. Creating reproducible 2D UMAP visualizations.
9. Training a contrastive adapter on paired domain data.
10. Re-running and reporting before/after adapter performance in a final notebook.

## Model Loader Contract

All embedding loaders should expose a public method or function equivalent to:

```python
def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Return a 2D numpy array with shape `(len(texts), embedding_dim)`."""
```

Expected behavior:

- Accept a list of strings.
- Preserve input order.
- Return a dense 2D `np.ndarray`.
- Use `float32` where practical.
- Hide framework-specific tensors from public loader methods.
- Handle batching internally.
- Raise clear errors when loading or embedding fails.

## Model Registry

Use `models/registry.py` so benchmark scripts can select models by string key. Keep benchmark code model-agnostic; the benchmark harness should depend only on the shared loader interface.

Example pattern:

```python
MODEL_REGISTRY = {
    "model_a": {
        "type": "local_transformer",
        "path": "placeholder-model-path",
        "requires_remote_code": False,
    },
}
```

## Benchmark Harness

`benchmarks/run_benchmark.py` should:

1. Parse CLI arguments with `argparse`.
2. Load corpus, queries, and qrels from the selected dataset.
3. Load the selected model through the registry/loader.
4. Encode corpus documents and queries in batches.
5. Convert embeddings to `float32`.
6. L2-normalize corpus and query embeddings.
7. Build a flat vector index for cosine similarity, commonly via inner product over normalized vectors.
8. Keep a stable mapping from vector-index row number to corpus document ID.
9. Retrieve top-k documents for each query.
10. Convert retrieval results into the metric library's expected format.
11. Compute MRR@10 and NDCG@10.
12. Save a JSON result file under `results/`.

Result JSON should include:

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

## CLI Expectations

Main scripts should support `--help`.

Useful smoke-test flow:

```bash
python scripts/download_data.py --dataset <dataset_name> --output-dir data/
python benchmarks/run_benchmark.py --model model_a --dataset data/<dataset_name> --limit-corpus 500 --limit-queries 50
python benchmarks/compare.py --results-dir results/ --output results/comparison.csv
```

## UMAP Guidelines

`benchmarks/visualize_umap.py` should:

- Use the same sampled corpus document IDs for every model comparison.
- Persist sampled document IDs so plots are reproducible.
- Save plots under `results/`.
- Treat UMAP as exploratory visualization, not definitive retrieval quality evidence.

## Adapter Training

The contrastive adapter workflow should use paired domain examples, such as query-document or question-answer pairs.

Training intent:

- Positive pairs should be closer in embedding space.
- In-batch negatives should be farther apart.
- Adapter training should improve target-domain retrieval without requiring full model retraining.

## Coding Guidelines

- Prefer simple, readable Python over complex abstractions.
- Use `pathlib.Path` for file paths.
- Use `argparse` for CLIs.
- Use `tqdm` for long-running loops.
- Use `numpy` arrays for embeddings.
- Keep deterministic behavior where practical.
- Keep benchmark logic model-agnostic.
- Add small, testable functions around loading, embedding, searching, metrics, and saving.
- Add comments only for non-obvious behavior.
- Update `README.md` or notebook instructions when CLI behavior changes.

## Common Pitfalls

- Forgetting to normalize embeddings before cosine search.
- Mixing integer and string query or document IDs.
- Losing the mapping between FAISS/vector-index rows and corpus document IDs.
- Comparing UMAP plots generated from different document samples.
- Reporting latency without batch size or hardware context.
- Letting model-specific code leak into the benchmark harness.
- Running full datasets before validating the pipeline on a small subset.

## Codex Working Notes

When returning to this repo:

2. Check `git status --short` and avoid overwriting user changes.
3. Prefer `rg`/`rg --files` for exploration.
4. Keep generated data and benchmark outputs out of version control unless asked.
5. Run focused smoke tests after changing scripts, especially `--help` and small `--limit-*` benchmark runs where available.
