"""
coordinate_solver.py — Distance matrix + PCA/SVD decomposition.

Takes embedding vectors and discovers the empirical coordinate system
of the behavioral response manifold.

Outputs:
    1. Pairwise response distance matrix (cosine distance)
    2. PCA decomposition (principal components + variance ratios)
    3. Projection of each run into the discovered coordinate space

NO classification. NO instability types. NO triggers.
Only empirical geometry.
"""

import json
import os
import sys
import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

# Parent directory for imports
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

logger = logging.getLogger("lattice")


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix.

    D_ij = 1 - cosine_similarity(V_i, V_j)

    Args:
        embeddings: Matrix of shape (n_samples, n_features).

    Returns:
        Distance matrix of shape (n_samples, n_samples).
    """
    logger.info(f"  Computing distance matrix ({embeddings.shape[0]} samples)...")

    sim_matrix = sklearn_cosine_sim(embeddings)
    distance_matrix = 1.0 - sim_matrix

    # Clamp to [0, 1] (numerical stability)
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

    # Zero out diagonal
    np.fill_diagonal(distance_matrix, 0.0)

    logger.info(f"  Distance matrix stats:")
    logger.info(f"    mean: {distance_matrix.mean():.4f}")
    logger.info(f"    std:  {distance_matrix.std():.4f}")
    logger.info(f"    min:  {distance_matrix[distance_matrix > 0].min():.4f}")
    logger.info(f"    max:  {distance_matrix.max():.4f}")

    return distance_matrix


def compute_pca(embeddings: np.ndarray, n_components: int = 8) -> dict:
    """
    Compute PCA decomposition of the embedding matrix.

    Discovers the principal behavioral directions from data.

    Args:
        embeddings: Matrix of shape (n_samples, n_features).
        n_components: Number of principal components to compute.

    Returns:
        dict with:
            - "principal_components": list of {index, variance_ratio, cumulative_variance}
            - "components_matrix": np.ndarray (n_components, n_features) — the PCs
            - "projections": np.ndarray (n_samples, n_components) — projections
            - "explained_variance": np.ndarray — absolute variance per component
            - "explained_variance_ratio": np.ndarray — fraction of total variance
    """
    logger.info(f"  Running PCA (k={n_components}) on {embeddings.shape}...")

    # Determine actual number of components (can't exceed min(n, p))
    max_components = min(embeddings.shape[0], embeddings.shape[1], n_components)
    if max_components < 2:
        logger.warning(f"  Not enough data for PCA (max {max_components} components)")
        # Return trivial decomposition
        return {
            "principal_components": [],
            "components_matrix": np.zeros((1, embeddings.shape[1])),
            "projections": np.zeros((embeddings.shape[0], 1)),
            "explained_variance": np.array([0.0]),
            "explained_variance_ratio": np.array([0.0]),
        }

    pca = PCA(n_components=max_components)
    projections = pca.fit_transform(embeddings)

    # Build principal component info
    pc_info = []
    cumulative = 0.0
    for i in range(max_components):
        ratio = pca.explained_variance_ratio_[i]
        cumulative += ratio
        pc_info.append({
            "index": i,
            "variance_ratio": round(float(ratio), 6),
            "cumulative_variance": round(float(cumulative), 6),
            "eigenvalue": round(float(pca.explained_variance_[i]), 6),
        })

    logger.info(f"  PCA results:")
    for pc in pc_info:
        logger.info(f"    PC{pc['index']}: {pc['variance_ratio']:.4f} variance (cum: {pc['cumulative_variance']:.4f})")

    return {
        "principal_components": pc_info,
        "components_matrix": pca.components_,
        "projections": projections,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def compute_strategy_displacement(
    records: list[dict],
    projections: np.ndarray,
) -> dict:
    """
    Compute the mean displacement per perturbation strategy.

    For each strategy, compute the centroid of all its projections,
    then measure the distance from the global centroid.

    This identifies which perturbation strategies cause the largest
    behavioral displacement — without any labeling or classification.

    Args:
        records: List of run records with "strategy" and "row_index" fields.
        projections: PCA projection matrix (n_samples, n_components).

    Returns:
        dict mapping strategy name to displacement info.
    """
    from collections import defaultdict

    strategy_indices = defaultdict(list)
    for rec in records:
        idx = rec.get("row_index", -1)
        if idx >= 0:
            strategy_indices[rec.get("strategy", "unknown")].append(idx)

    # Global centroid
    global_centroid = projections.mean(axis=0)

    displacements = {}
    for strategy, indices in strategy_indices.items():
        if not indices:
            continue
        strategy_centroid = projections[indices].mean(axis=0)
        displacement = np.linalg.norm(strategy_centroid - global_centroid)
        displacements[strategy] = {
            "n_samples": len(indices),
            "centroid": [round(float(x), 6) for x in strategy_centroid],
            "displacement_from_global": round(float(displacement), 6),
        }

    # Sort by displacement magnitude
    sorted_strategies = sorted(
        displacements.items(), key=lambda x: x[1]["displacement_from_global"], reverse=True
    )

    logger.info(f"  Strategy displacements (from global centroid):")
    for strategy, info in sorted_strategies:
        logger.info(f"    {strategy:20s}: displacement={info['displacement_from_global']:.4f} (n={info['n_samples']})")

    return dict(sorted_strategies)


def solve_behavioral_space(
    jsonl_path: str,
    n_components: int = 8,
    output_path: Optional[str] = None,
) -> str:
    """
    Full pipeline: load records → embed → distance → PCA → output.

    Args:
        jsonl_path: Path to the JSONL run records.
        n_components: Number of PCA components.
        output_path: Path to save behavioral_space.json.

    Returns:
        Path to the output file.
    """
    from lcge_engine.lattice.vector_store import (
        load_run_records,
        build_embedding_matrix,
        save_embeddings,
        save_records_for_solver,
    )

    logger.info("=" * 60)
    logger.info("Coordinate Solver — discovering behavioral basis")
    logger.info("=" * 60)

    # Load records
    logger.info("[1/5] Loading run records...")
    records = load_run_records(jsonl_path)
    logger.info(f"  Loaded {len(records)} records")

    # Build embeddings
    logger.info("[2/5] Building embedding matrix...")
    embeddings, meta = build_embedding_matrix(records)

    # Save embeddings for reproducibility
    output_dir = os.path.dirname(output_path) if output_path else os.path.join(_pkg_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(output_dir, "embeddings.npz")
    save_embeddings(embeddings, [r.get("run_id", "") for r in records], emb_path)

    idx_path = os.path.join(output_dir, "record_index.json")
    save_records_for_solver(records, embeddings, idx_path)

    # Distance matrix
    logger.info("[3/5] Computing distance matrix...")
    distance_matrix = compute_distance_matrix(embeddings)

    # Save distance matrix (numpy format — JSON would be too large)
    dist_path = os.path.join(output_dir, "distance_matrix.npz")
    np.savez_compressed(dist_path, distance_matrix=distance_matrix)
    logger.info(f"  Saved distance matrix to {dist_path}")

    # PCA
    logger.info("[4/5] Computing PCA decomposition...")
    pca_result = compute_pca(embeddings, n_components)

    # Strategy displacement
    logger.info("[5/5] Computing strategy displacements...")
    strategy_displacement = compute_strategy_displacement(records, pca_result["projections"])

    # Build output
    if output_path is None:
        output_path = os.path.join(output_dir, "behavioral_space.json")

    # Projections indexed by run_id
    projections_dict = {}
    for i, rec in enumerate(records):
        run_id = rec.get("run_id", f"run_{i}")
        proj = pca_result["projections"][i]
        projections_dict[run_id] = [round(float(x), 6) for x in proj]

    output = {
        "method": "PCA on TF-IDF response embeddings",
        "parameters": {
            "n_records": meta["n_records"],
            "n_features": meta["n_features"],
            "n_components": len(pca_result["principal_components"]),
            "empty_responses": meta["empty_count"],
        },
        "principal_components": pca_result["principal_components"],
        "distance_matrix_summary": {
            "n_samples": len(records),
            "mean_distance": round(float(distance_matrix.mean()), 6),
            "std_distance": round(float(distance_matrix.std()), 6),
            "min_distance": round(float(distance_matrix[distance_matrix > 0].min()), 6) if distance_matrix[distance_matrix > 0].size > 0 else 0.0,
            "max_distance": round(float(distance_matrix.max()), 6),
        },
        "strategy_displacements": strategy_displacement,
        "projections": projections_dict,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    return output_path


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LCGE v1.3 Coordinate Solver — discover behavioral basis from data",
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to JSONL run records")
    parser.add_argument("--output", "-o", type=str, help="Output path for behavioral_space.json")
    parser.add_argument("--components", "-k", type=int, default=8, help="Number of PCA components")

    args = parser.parse_args()

    path = solve_behavioral_space(
        jsonl_path=args.input,
        n_components=args.components,
        output_path=args.output,
    )

    print(f"\nBehavioral space saved: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
