# Embedding Model Benchmarking Lab

A reproducible lab for comparing multiple embedding models on retrieval quality, latency, cost, and embedding-space behavior. The project also includes a lightweight contrastive adapter training loop to measure whether domain-specific fine-tuning can improve retrieval performance.

## Overview

This project benchmarks several embedding models using the same retrieval dataset, evaluation pipeline, and metrics. It is designed to answer practical questions such as:

- Which embedding model performs best on a labeled retrieval task?
- How do latency, embedding dimensionality, and cost compare across models?
- Do different embedding models organize the same corpus into visibly different semantic clusters?
- Can a lightweight contrastive adapter improve retrieval quality on a domain-specific dataset?

The lab includes:

- A unified embedding interface for swapping models by name.
- A benchmark harness for corpus/query encoding, vector search, and metric computation.
- Retrieval metrics including cosine similarity, MRR@10, and NDCG@10.
- A 2D UMAP visualization workflow for comparing embedding spaces.
- A contrastive fine-tuning loop using paired text data.
- Before/after benchmark comparisons for fine-tuned adapters.
- A final notebook report covering retrieval accuracy, latency, cost, and dimensionality tradeoffs.

## Project Structure

```text
embedding-benchmarking-lab/
├── benchmarks/
│   ├── run_benchmark.py        # Main benchmark entry point
│   ├── compare.py              # Aggregate benchmark results
│   └── visualize_umap.py       # Generate 2D embedding visualizations
├── data/
│   └── .gitkeep                # Local datasets are stored here
├── models/
│   ├── loader.py               # Unified model loading and embedding interface
│   └── registry.py             # Model registry configuration
├── training/
│   ├── train_adapter.py        # Contrastive adapter training loop
│   └── evaluate_adapter.py     # Before/after adapter evaluation
├── notebooks/
│   └── final_report.ipynb      # Final benchmark report
├── results/
│   └── .gitkeep                # Benchmark outputs are stored here
├── scripts/
│   └── download_data.py        # Dataset acquisition script
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Environment Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the CPU-only deep learning runtime first to avoid pulling unnecessary GPU binaries. Then install the rest of the project dependencies.

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Suggested dependencies:

```text
numpy
tqdm
einops
python-dotenv
sentence-transformers
openai
cohere
umap-learn
beir
matplotlib
rank-bm25
```

## Dataset Setup

Use a small labeled retrieval dataset for local CPU development. The dataset should provide:

- `corpus`: document IDs mapped to document text
- `queries`: query IDs mapped to query text
- `qrels`: relevance labels mapping queries to relevant documents


Recommended workflow:

1. Start with a small dataset to validate the pipeline quickly.
2. Confirm that corpus, queries, and qrels are present.
3. Run a single model benchmark end-to-end.
4. Scale to additional models only after the first benchmark works.


## Running Benchmarks

Run a benchmark for one model:

```bash
python benchmarks/run_benchmark.py \
  --model model_a \
  --dataset data/<dataset_name> \
  --batch-size 32 \
  --top-k 10 \
  --output results/model_a_results.json
```

The benchmark should:

1. Load corpus, queries, and qrels.
2. Encode the corpus in batches.
3. Encode all queries.
4. L2-normalize embeddings.
5. Build a flat vector index.
6. Retrieve top-k documents per query using cosine similarity.
7. Compute MRR@10 and NDCG@10.
8. Record latency and embedding dimensionality.
9. Save results as JSON.

Example result format:

```json
{
  "model": "model_a",
  "embedding_dim": 768,
  "mrr_at_10": 0.0,
  "ndcg_at_10": 0.0,
  "avg_latency_per_embedding_ms": 0.0,
  "total_corpus_encode_time_sec": 0.0,
  "num_corpus_documents": 0,
  "num_queries": 0
}
```


## UMAP Visualization

Generate 2D projections of corpus embeddings to compare how models organize semantic space.

```bash
python benchmarks/visualize_umap.py \
  --model model_a \
  --dataset data/<dataset_name> \
  --sample-size 1000 \
  --output results/model_a_umap.png
```

Use the same sampled corpus documents across all models so the visualizations are comparable.

Recommended UMAP output:

- One plot per dataset.
- Same document sample for every model.
- Optional labels based on dataset metadata when available.
- Saved images under `results/`.

## Contrastive Adapter Training

The training loop fine-tunes a lightweight adapter on paired domain data of the existing dataset.

Training objective:

- Positive pairs should be close in embedding space.
- In-batch negatives should be pushed farther apart.
- The adapter should improve retrieval on the target domain without requiring full model retraining.


## Development Milestones

### 1. Environment and Dataset

- Create local Python environment.
- Install CPU-only dependencies.
- Download a small retrieval dataset.
- Validate corpus, queries, and qrels.

### 2. Model Loading

- Implement the shared embedding interface.
- Add model registry entries.
- Test each model on a small text batch.
- Confirm embedding shapes and dtype.

### 3. Benchmark Harness

- Encode corpus and queries.
- Build vector index.
- Retrieve top-k results.
- Compute MRR@10 and NDCG@10.
- Save per-model JSON results.

### 4. Comparison and Visualization

- Aggregate all benchmark outputs.
- Save comparison CSV.
- Generate UMAP plots.
- Add results to the final notebook.

### 5. Adapter Training

- Prepare paired domain data.
- Train contrastive adapter.
- Re-run benchmark after fine-tuning.
- Compare before/after metrics.



