import argparse
import json
import time
from pathlib import Path
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.device("cpu") 
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_benchmark import (
    load_corpus,
    load_queries,
    load_qrels,
    load_corpus_cache,
    save_corpus_cache,
    _l2_normalize,
    retrieve_top_k,
    compute_metrics,
)


def _encode(st_model, texts: list[str], batch_size: int = 32) -> np.ndarray:
    return st_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def train(dataset_dir: Path, ft_model_path: Path) -> None:
    from beir.datasets.data_loader import GenericDataLoader
    from sentence_transformers import InputExample, SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    print("Loading scifact train split via BeIR GenericDataLoader ...")
    corpus_beir, queries_beir, train_qrels = GenericDataLoader(
        data_folder=str(dataset_dir)
    ).load(split="train")

    pairs: list[InputExample] = []
    for qid, docs in train_qrels.items():
        if qid not in queries_beir:
            continue
        query_text = queries_beir[qid]
        for doc_id, score in docs.items():
            if score > 0 and doc_id in corpus_beir:
                title = corpus_beir[doc_id].get("title", "")
                text = corpus_beir[doc_id].get("text", "")
                passage = f"{title}\n{text}" if title else text
                pairs.append(InputExample(texts=[query_text, passage]))
    print(f"Extracted {len(pairs)} training pairs.")
    dataloader = DataLoader(pairs, batch_size=4, shuffle=True)

    print("Loading base model: BAAI/bge-base-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device='cpu')
    print("Model loaded.")

    loss = MultipleNegativesRankingLoss(model)

    warmup_steps = max(1, int(0.1 * len(dataloader)))
    print(f"Fine-tuning for 1 epoch (warmup_steps={warmup_steps}) ...")
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    model.save(str(ft_model_path))
    print(f"Fine-tuned model saved to {ft_model_path}")


def evaluate(
    dataset_dir: Path,
    ft_model_path: Path,
    results_dir: Path,
    args,
) -> None:
    from sentence_transformers import SentenceTransformer

    dataset_name = dataset_dir.name

    print(f"Loading fine-tuned model from {ft_model_path} ...")
    st_model = SentenceTransformer(str(ft_model_path), device='cpu')
    embedding_dim = st_model.get_sentence_embedding_dimension()

    print("Loading corpus, queries, and qrels ...")
    corpus = load_corpus(dataset_dir)
    all_queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir, args.split)

    eval_query_ids = [qid for qid in qrels if qid in all_queries]
    query_texts = [all_queries[qid] for qid in eval_query_ids]
    print(f"  {len(eval_query_ids)} queries selected for evaluation")

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[did] for did in corpus_ids]

    cache_path = results_dir / "corpus_cache" / dataset_name / f"{args.model}_corpus.npz"
    cached = None if args.no_cache else load_corpus_cache(cache_path)

    corpus_encode_time = 0.0
    cache_hit = False

    if cached is not None:
        cached_embs, cached_ids = cached
        if list(cached_ids) == corpus_ids:
            print(f"  Loading corpus embeddings from cache: {cache_path}")
            corpus_embs = cached_embs
            cache_hit = True
        else:
            print("  Cache ID mismatch — re-embedding corpus.")

    if not cache_hit:
        print(f"Embedding {len(corpus_texts)} corpus documents ...")
        t0 = time.perf_counter()
        corpus_embs = _encode(st_model, corpus_texts, batch_size=args.batch_size)
        corpus_encode_time = time.perf_counter() - t0
        print(f"  Done in {corpus_encode_time:.1f}s")
        if not args.no_cache:
            save_corpus_cache(cache_path, corpus_embs, corpus_ids)
            print(f"  Corpus embeddings saved to {cache_path}")

    print(f"Embedding {len(query_texts)} queries ...")
    t0 = time.perf_counter()
    query_embs = _encode(st_model, query_texts, batch_size=args.batch_size)
    query_encode_time = time.perf_counter() - t0
    print(f"  Done in {query_encode_time:.1f}s")

    corpus_embs = _l2_normalize(corpus_embs.astype(np.float32))
    query_embs = _l2_normalize(query_embs.astype(np.float32))

    print(f"Retrieving top-{args.top_k} documents per query ...")
    run = retrieve_top_k(corpus_embs, query_embs, corpus_ids, eval_query_ids, args.top_k)

    print("Computing metrics ...")
    metrics = compute_metrics(qrels, run, args.top_k)
    print(f"  MRR@{args.top_k}:  {metrics['mrr_at_10']:.4f}")
    print(f"  NDCG@{args.top_k}: {metrics['ndcg_at_10']:.4f}")

    texts_encoded = (0 if cache_hit else len(corpus_texts)) + len(eval_query_ids)
    time_encoded = corpus_encode_time + query_encode_time
    avg_latency_ms = (time_encoded / texts_encoded * 1000) if texts_encoded > 0 else 0.0

    result = {
        "dataset": dataset_name,
        "split": args.split,
        "model": args.model,
        "embedding_dim": embedding_dim,
        "mrr_at_10": metrics["mrr_at_10"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "avg_latency_per_embedding_ms": avg_latency_ms,
        "total_corpus_encode_time_sec": corpus_encode_time,
        "total_query_encode_time_sec": query_encode_time,
        "num_corpus_documents": len(corpus_texts),
        "num_queries": len(eval_query_ids),
        "top_k": args.top_k,
    }

    output_path = (
        Path(args.output)
        if args.output
        else results_dir / "stats" / f"{dataset_name}_{args.model}_{args.split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BAAI/bge-base-en-v1.5 on SciFact with contrastive loss, then evaluate."
    )
    parser.add_argument("--dataset", default="data/scifact", help="Path to dataset directory")
    parser.add_argument("--split", default="test", help="Evaluation qrels split (default: test)")
    parser.add_argument(
        "--model",
        default="bge_mnrl_scifact_ft",
        help="Model key used in result JSON (default: bge_base_scifact_ft)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None, help="Output JSON path (auto-derived if omitted)")
    parser.add_argument("--no-cache", action="store_true", help="Skip loading and saving corpus cache")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / args.dataset
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    ft_model_path = project_root / "models" / "bge-base-scifact-ft"

    if ft_model_path.exists():
        print("Fine-tuned model found — skipping training, going straight to evaluation.")
    else:
        train(dataset_dir, ft_model_path)

    evaluate(dataset_dir, ft_model_path, results_dir, args)


if __name__ == "__main__":
    main()
