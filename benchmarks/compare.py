import argparse
import csv
import json
from pathlib import Path


CSV_FIELDS = [
    "cache_file",
    "stats_file",
    "dataset",
    "split",
    "model",
    "embedding_dim",
    "mrr_at_10",
    "ndcg_at_10",
    "avg_latency_per_embedding_ms",
    "total_corpus_encode_time_sec",
    "total_query_encode_time_sec",
    "num_corpus_documents",
    "num_queries",
    "top_k",
]


def infer_model_key(cache_path: Path, dataset: str) -> str:
    stem = cache_path.stem
    legacy_marker = f"_{dataset}_corpus"
    if legacy_marker in stem:
        return stem.split(legacy_marker, maxsplit=1)[0]
    if "_corpus" in stem:
        return stem.split("_corpus", maxsplit=1)[0]
    return stem


def discover_cache_files(cache_root: Path, dataset: str, include_flat_cache: bool) -> list[Path]:
    dataset_cache_dir = cache_root / dataset
    paths = sorted(dataset_cache_dir.glob("*.npz")) if dataset_cache_dir.exists() else []

    if include_flat_cache:
        flat_paths = sorted(cache_root.glob(f"*_{dataset}_corpus*.npz"))
        paths.extend(path for path in flat_paths if path not in paths)

    return paths


def find_stats_path(stats_dir: Path, dataset: str, model_key: str, split: str) -> Path | None:
    expected_path = stats_dir / f"{dataset}_{model_key}_{split}.json"
    if expected_path.exists():
        return expected_path

    candidates = sorted(stats_dir.glob(f"{dataset}_{model_key}_*.json"))
    return candidates[0] if candidates else None


def load_stats_row(
    cache_path: Path,
    stats_path: Path,
    dataset: str,
    split: str,
    model_key: str,
) -> dict[str, object]:
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    row = {field: stats.get(field, "") for field in CSV_FIELDS}
    row["cache_file"] = str(cache_path)
    row["stats_file"] = str(stats_path)
    row["dataset"] = stats.get("dataset", dataset)
    row["split"] = stats.get("split", split)
    row["model"] = stats.get("model", model_key)
    return row


def build_comparison_rows(
    cache_paths: list[Path],
    stats_dir: Path,
    dataset: str,
    split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for cache_path in cache_paths:
        model_key = infer_model_key(cache_path, dataset)
        stats_path = find_stats_path(stats_dir, dataset, model_key, split)
        if stats_path is None:
            print(
                "Warning: no stats JSON found for "
                f"cache {cache_path.name} using model key '{model_key}'"
            )
            continue

        rows.append(load_stats_row(cache_path, stats_path, dataset, split, model_key))

    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a benchmark comparison CSV from stats JSON files. When a dataset "
            "is provided, only models with matching UMAP corpus .npz caches are included."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name matching results/corpus_cache/<dataset>/, such as scifact or fiqa",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Stats split to match when reading results/stats JSON files",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("results/stats"),
        help="Directory containing benchmark stats JSON files",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("results/corpus_cache"),
        help="Root directory containing corpus embedding caches used by UMAP",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to results/csv/<dataset>_<split>_comparison.csv",
    )
    parser.add_argument(
        "--include-flat-cache",
        action="store_true",
        help="Also include legacy flat files like results/corpus_cache/<model>_<dataset>_corpus*.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        args.output
        or Path("results") / "csv" / f"{args.dataset}_{args.split}_comparison.csv"
    )

    cache_paths = discover_cache_files(
        args.cache_root,
        args.dataset,
        include_flat_cache=args.include_flat_cache,
    )
    if not cache_paths:
        raise FileNotFoundError(
            f"No .npz caches found under {args.cache_root / args.dataset}"
        )

    print(f"Found {len(cache_paths)} UMAP cache file(s):")
    for path in cache_paths:
        print(f"  {path}")

    rows = build_comparison_rows(
        cache_paths,
        stats_dir=args.stats_dir,
        dataset=args.dataset,
        split=args.split,
    )
    write_csv(rows, output_path)
    print(f"Comparison CSV saved to {output_path} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
