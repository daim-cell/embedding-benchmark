import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP


@dataclass(frozen=True)
class EmbeddingCache:
    path: Path
    label: str
    doc_ids: list[str]
    embeddings: np.ndarray


def load_cache(path: Path) -> EmbeddingCache:
    data = np.load(path, allow_pickle=True)
    required_keys = {"embeddings", "doc_ids"}
    missing = required_keys.difference(data.files)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required key(s): {missing_text}")

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    doc_ids = [str(doc_id) for doc_id in data["doc_ids"].tolist()]
    if embeddings.ndim != 2:
        raise ValueError(f"{path} embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(doc_ids):
        raise ValueError(
            f"{path} has {embeddings.shape[0]} embeddings but {len(doc_ids)} doc IDs"
        )

    return EmbeddingCache(
        path=path,
        label=path.stem.split("_corpus")[0].replace("_", "-"),
        doc_ids=doc_ids,
        embeddings=embeddings,
    )


def discover_cache_files(cache_root: Path, dataset: str, include_flat_cache: bool) -> list[Path]:
    dataset_cache_dir = cache_root / dataset
    paths = sorted(dataset_cache_dir.glob("*.npz")) if dataset_cache_dir.exists() else []

    if include_flat_cache:
        flat_paths = sorted(cache_root.glob(f"*_{dataset}_corpus*.npz"))
        paths.extend(path for path in flat_paths if path not in paths)

    return paths


def common_doc_ids(caches: list[EmbeddingCache]) -> list[str]:
    common = set(caches[0].doc_ids)
    for cache in caches[1:]:
        common.intersection_update(cache.doc_ids)

    # Preserve the first cache's order so every subplot uses the same document sequence.
    return [doc_id for doc_id in caches[0].doc_ids if doc_id in common]


def sample_doc_ids(doc_ids: list[str], sample_size: int | None, seed: int) -> list[str]:
    if sample_size is None or sample_size >= len(doc_ids):
        return doc_ids

    rng = np.random.default_rng(seed)
    sampled_indices = np.sort(rng.choice(len(doc_ids), size=sample_size, replace=False))
    return [doc_ids[i] for i in sampled_indices]


def align_embeddings(cache: EmbeddingCache, selected_doc_ids: list[str]) -> np.ndarray:
    row_by_doc_id = {doc_id: i for i, doc_id in enumerate(cache.doc_ids)}
    rows = [row_by_doc_id[doc_id] for doc_id in selected_doc_ids]
    return cache.embeddings[rows]


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1.0, norms)


def project_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    if embeddings.shape[0] < 3:
        raise ValueError("UMAP needs at least 3 documents to produce a useful projection")

    effective_neighbors = min(n_neighbors, embeddings.shape[0] - 1)
    reducer = UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(l2_normalize(embeddings))


def plot_projections(
    projections: list[tuple[str, np.ndarray]],
    output_path: Path,
    dataset: str,
    num_documents: int,
    point_size: float,
) -> None:
    num_plots = len(projections)
    cols = min(3, num_plots)
    rows = ceil(num_plots / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.2 * cols, 4.6 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{dataset} corpus embedding spaces ({num_documents} shared documents)")

    for ax in axes.ravel()[num_plots:]:
        ax.axis("off")

    for ax, (label, projection) in zip(axes.ravel(), projections):
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            s=point_size,
            alpha=0.72,
            linewidths=0,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side 2D UMAP plots from corpus embedding caches for one dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name matching results/corpus_cache/<dataset>/, such as scifact or fiqa",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("results/corpus_cache"),
        help="Root directory containing dataset corpus embedding caches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to results/images/<dataset>_umap.png",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Maximum number of shared documents to project. Use 0 for all documents.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument(
        "--include-flat-cache",
        action="store_true",
        help="Also include legacy flat files like results/corpus_cache/<model>_<dataset>_corpus*.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or Path("results") / "images" / f"{args.dataset}_umap.png"
    sample_size = None if args.sample_size == 0 else args.sample_size

    cache_paths = discover_cache_files(
        args.cache_root,
        args.dataset,
        include_flat_cache=args.include_flat_cache,
    )
    if not cache_paths:
        raise FileNotFoundError(
            f"No .npz caches found under {args.cache_root / args.dataset}"
        )

    print(f"Found {len(cache_paths)} cache file(s):")
    for path in cache_paths:
        print(f"  {path}")

    caches = [load_cache(path) for path in cache_paths]
    shared_doc_ids = common_doc_ids(caches)
    if not shared_doc_ids:
        raise ValueError("No shared document IDs found across the selected cache files")

    selected_doc_ids = sample_doc_ids(shared_doc_ids, sample_size, args.seed)
    print(f"Using {len(selected_doc_ids)} shared document(s) for UMAP")

    projections: list[tuple[str, np.ndarray]] = []
    for cache in caches:
        print(f"Projecting {cache.label} ...")
        embeddings = align_embeddings(cache, selected_doc_ids)
        projection = project_umap(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            seed=args.seed,
        )
        projections.append((cache.label, projection))

    plot_projections(
        projections,
        output_path=output_path,
        dataset=args.dataset,
        num_documents=len(selected_doc_ids),
        point_size=args.point_size,
    )
    print(f"UMAP visualization saved to {output_path}")


if __name__ == "__main__":
    main()
