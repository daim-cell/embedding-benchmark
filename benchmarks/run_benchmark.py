import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from beir.retrieval.evaluation import EvaluateRetrieval

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.loader import BM25Retriever
from models.registry import get_model


def load_corpus(dataset_dir: Path, limit: int | None = None) -> dict[str, str]:
    corpus: dict[str, str] = {}
    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["_id"])
            title = row.get("title", "")
            text = row.get("text", "")
            corpus[doc_id] = f"{title}\n{text}" if title else text
            if limit is not None and len(corpus) >= limit:
                break
    return corpus


def load_queries(dataset_dir: Path, limit: int | None = None) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(dataset_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
            if limit is not None and len(queries) >= limit:
                break
    return queries


def load_qrels(dataset_dir: Path, split: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    with open(qrels_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            query_id, corpus_id, score = str(parts[0]), str(parts[1]), int(parts[2])
            qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels


def get_cache_path(results_dir: Path, model_key: str, dataset_name: str, limit: int | None) -> Path:
    suffix = f"_limit{limit}" if limit is not None else ""
    filename = f"{model_key}_corpus{suffix}.npz"
    return results_dir / "corpus_cache" / dataset_name / filename


def get_bm25_cache_path(results_dir: Path, dataset_name: str, limit: int | None) -> Path:
    suffix = f"_limit{limit}" if limit is not None else ""
    return results_dir / "corpus_cache" / dataset_name / f"bm25_corpus{suffix}.pkl"


def load_corpus_cache(cache_path: Path) -> tuple[np.ndarray, list[str]] | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    return data["embeddings"], list(data["doc_ids"])


def save_corpus_cache(cache_path: Path, embeddings: np.ndarray, doc_ids: list[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, doc_ids=np.array(doc_ids, dtype=object))


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def retrieve_top_k(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    corpus_ids: list[str],
    query_ids: list[str],
    top_k: int,
) -> dict[str, dict[str, float]]:
    scores = query_embs @ corpus_embs.T  # (n_queries, n_corpus)
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Retrieving", leave=False)):
        row = scores[i]
        top_indices = np.argsort(-row)[:top_k]
        run[qid] = {corpus_ids[j]: float(row[j]) for j in top_indices}
    return run


def compute_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    top_k: int,
) -> dict[str, float]:
    ndcg, *_ = EvaluateRetrieval.evaluate(qrels, run, [top_k])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, run, [top_k], metric="mrr")
    return {
        "mrr_at_10": mrr.get(f"MRR@{top_k}", 0.0),
        "ndcg_at_10": ndcg.get(f"NDCG@{top_k}", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a retrieval benchmark on FIQA or SciFact datasets."
    )
    parser.add_argument("--model", required=True, help="Model key from the registry")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--split", default="test", help="Qrels split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-corpus", type=int, default=None, metavar="N")
    parser.add_argument("--limit-queries", type=int, default=None, metavar="N")
    parser.add_argument("--output", default=None, help="Output JSON path (auto-derived if omitted)")
    parser.add_argument("--no-cache", action="store_true", help="Skip loading and saving corpus cache")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    dataset_name = dataset_dir.name
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = Path(args.output) if args.output else results_dir / "stats" /f"{dataset_name}_{args.model}_{args.split}.json"

    print(f"Loading corpus from {dataset_dir / 'corpus.jsonl'} ...")
    corpus = load_corpus(dataset_dir, args.limit_corpus)
    print(f"  {len(corpus)} documents loaded")

    print(f"Loading queries from {dataset_dir / 'queries.jsonl'} ...")
    all_queries = load_queries(dataset_dir)

    print(f"Loading qrels ({args.split}) ...")
    qrels = load_qrels(dataset_dir, args.split)

    # Only evaluate queries that appear in the chosen qrels split.
    eval_query_ids = [qid for qid in qrels if qid in all_queries]
    if args.limit_queries is not None:
        eval_query_ids = eval_query_ids[: args.limit_queries]
    query_texts = [all_queries[qid] for qid in eval_query_ids]
    print(f"  {len(eval_query_ids)} queries selected for evaluation")

    print(f"Loading model '{args.model}' ...")
    model = get_model(args.model)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[did] for did in corpus_ids]

    corpus_encode_time = 0.0
    query_encode_time = 0.0
    run: dict[str, dict[str, float]]

    if isinstance(model, BM25Retriever):
        # --- BM25 sparse retrieval path ---
        bm25_cache = get_bm25_cache_path(results_dir, dataset_name, args.limit_corpus)
        if not args.no_cache and bm25_cache.exists():
            print(f"  Loading BM25 index from cache: {bm25_cache}")
            model = BM25Retriever.load_from(bm25_cache)
        else:
            print(f"Fitting BM25 on {len(corpus_texts)} documents ...")
            t0 = time.perf_counter()
            model.fit(corpus_ids, corpus_texts)
            corpus_encode_time = time.perf_counter() - t0
            print(f"  Done in {corpus_encode_time:.1f}s")
            if not args.no_cache:
                model.save(bm25_cache)
                print(f"  BM25 index saved to {bm25_cache}")

        print(f"Scoring {len(query_texts)} queries with BM25 ...")
        t0 = time.perf_counter()
        run = model.retrieve(query_texts, eval_query_ids, args.top_k)
        query_encode_time = time.perf_counter() - t0
        print(f"  Done in {query_encode_time:.1f}s")

        avg_latency_ms = query_encode_time / len(eval_query_ids) * 1000 if eval_query_ids else 0.0

    else:
        # --- Dense embedding path ---
        cache_path = get_cache_path(results_dir, args.model, dataset_name, args.limit_corpus)
        cached = None if args.no_cache else load_corpus_cache(cache_path)
        corpus_embs: np.ndarray

        if cached is not None:
            cached_embs, cached_ids = cached
            if cached_ids == corpus_ids:
                print(f"  Loading corpus embeddings from cache: {cache_path}")
                corpus_embs = cached_embs
            else:
                print("  Cache ID mismatch — re-embedding corpus.")
                cached = None

        if cached is None:
            print(f"Embedding {len(corpus_texts)} corpus documents ...")
            t0 = time.perf_counter()
            corpus_embs = model.embed(corpus_texts, batch_size=args.batch_size)
            corpus_encode_time = time.perf_counter() - t0
            print(f"  Done in {corpus_encode_time:.1f}s")
            if not args.no_cache:
                save_corpus_cache(cache_path, corpus_embs, corpus_ids)
                print(f"  Corpus embeddings saved to {cache_path}")

        print(f"Embedding {len(query_texts)} queries ...")
        t0 = time.perf_counter()
        query_embs = model.embed_queries(query_texts, batch_size=args.batch_size)
        query_encode_time = time.perf_counter() - t0
        print(f"  Done in {query_encode_time:.1f}s")

        corpus_embs = _l2_normalize(corpus_embs.astype(np.float32))
        query_embs = _l2_normalize(query_embs.astype(np.float32))

        print(f"Retrieving top-{args.top_k} documents per query ...")
        run = retrieve_top_k(corpus_embs, query_embs, corpus_ids, eval_query_ids, args.top_k)

        texts_encoded = (0 if cached is not None else len(corpus_texts)) + len(query_texts)
        time_encoded = corpus_encode_time + query_encode_time
        avg_latency_ms = (time_encoded / texts_encoded * 1000) if texts_encoded > 0 else 0.0

    print("Computing metrics ...")
    metrics = compute_metrics(qrels, run, args.top_k)
    print(f"  MRR@{args.top_k}:  {metrics['mrr_at_10']:.4f}")
    print(f"  NDCG@{args.top_k}: {metrics['ndcg_at_10']:.4f}")

    result = {
        "dataset": dataset_name,
        "split": args.split,
        "model": args.model,
        "embedding_dim": model.embedding_dim,
        "mrr_at_10": metrics["mrr_at_10"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "avg_latency_per_embedding_ms": avg_latency_ms,
        "total_corpus_encode_time_sec": corpus_encode_time,
        "total_query_encode_time_sec": query_encode_time,
        "num_corpus_documents": len(corpus_texts),
        "num_queries": len(eval_query_ids),
        "top_k": args.top_k
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
