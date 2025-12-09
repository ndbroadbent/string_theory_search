#!/usr/bin/env python3
"""
Compute ALL speculative shape heuristics for polytopes.
This is the kitchen sink - every metric we can think of.
"""

import json
import struct
import zlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
from scipy.stats import entropy, kurtosis, skew
from sklearn.decomposition import PCA


@dataclass
class PolytopeHeuristics:
    """Every heuristic we can compute for a polytope."""
    polytope_id: int
    h11: int
    h21: int
    vertex_count: int

    # === BASIC GEOMETRY ===
    centroid: list[float] = field(default_factory=list)

    # === π-NESS / CIRCULARITY ===
    sphericity: float = 0.0  # 1 - std(distances)/mean(distances)
    inertia_isotropy: float = 0.0  # min/max eigenvalue of inertia tensor
    inertia_eigenvalues: list[float] = field(default_factory=list)

    # === SPIRALITY ===
    spiral_correlations: dict[str, float] = field(default_factory=dict)  # per axis

    # === CHIRALITY ===
    chirality_optimal: float = 0.0  # Kabsch RMSD (optimal rotation alignment)
    chirality_x: float = 0.0  # Hausdorff-like distance after X reflection (no alignment)
    chirality_y: float = 0.0
    chirality_z: float = 0.0
    chirality_w: float = 0.0
    handedness_det: float = 0.0  # sign of determinant

    # === SYMMETRY ===
    symmetry_x: float = 0.0  # overlap after X reflection
    symmetry_y: float = 0.0
    symmetry_z: float = 0.0
    symmetry_w: float = 0.0

    # === FLATNESS (PCA) ===
    pca_variance_ratios: list[float] = field(default_factory=list)
    flatness_3d: float = 0.0  # variance in top 3 dims
    flatness_2d: float = 0.0
    intrinsic_dim_estimate: float = 0.0

    # === REGULARITY ===
    edge_length_cv: float = 0.0  # coefficient of variation

    # === SPIKINESS ===
    spikiness: float = 0.0  # max/mean distance ratio
    max_exposure: float = 0.0

    # === CONCENTRATION / OUTLIERS ===
    conformity_ratio: float = 0.0
    distance_kurtosis: float = 0.0
    loner_score: float = 0.0

    # === BASIC STATS ===
    coord_mean: float = 0.0
    coord_median: float = 0.0
    coord_mode: float = 0.0
    coord_std: float = 0.0
    coord_iqr: float = 0.0
    coord_range: float = 0.0
    coord_skewness: float = 0.0
    coord_kurtosis: float = 0.0
    mean_median_diff: float = 0.0

    # Per-axis stats
    axis_means: list[float] = field(default_factory=list)
    axis_medians: list[float] = field(default_factory=list)
    axis_stds: list[float] = field(default_factory=list)
    axis_skewness: list[float] = field(default_factory=list)
    axis_kurtosis: list[float] = field(default_factory=list)

    # === ENTROPY / INFORMATION ===
    shannon_entropy: float = 0.0
    axis_entropies: list[float] = field(default_factory=list)
    joint_entropy: float = 0.0

    # === COMPRESSIBILITY ===
    compression_ratio: float = 0.0
    sorted_compression_ratio: float = 0.0
    sort_compression_gain: float = 0.0

    # === AXIS BALANCE ===
    axis_balance: list[float] = field(default_factory=list)  # per axis
    spread_ratio: float = 0.0  # max/min spread across axes

    # Quartiles per axis
    axis_q1: list[float] = field(default_factory=list)
    axis_q2: list[float] = field(default_factory=list)
    axis_q3: list[float] = field(default_factory=list)
    axis_iqr: list[float] = field(default_factory=list)
    axis_quartile_skew: list[float] = field(default_factory=list)

    # === DISTRIBUTION TESTS ===
    normality_pvalue: float = 0.0
    uniform_ks_stat: float = 0.0

    # === CORRELATION ===
    correlations: dict[str, float] = field(default_factory=dict)
    mean_abs_correlation: float = 0.0
    corr_eigenvalues: list[float] = field(default_factory=list)

    # === GOLDEN RATIO ===
    phi_ratio_count: int = 0
    fibonacci_count: int = 0

    # === INTEGER PATTERNS ===
    zero_count: int = 0
    one_count: int = 0
    neg_one_count: int = 0
    prime_count: int = 0

    # === GRAPH PROPERTIES (from edge skeleton) ===
    # (simplified - full graph analysis would need convex hull)

    def to_embedding(self) -> list[float]:
        """Flatten all numeric values into a single embedding vector."""
        embedding = []

        # Scalars
        scalars = [
            self.h11, self.h21, self.vertex_count,
            self.sphericity, self.inertia_isotropy,
            self.chirality_optimal,
            self.chirality_x, self.chirality_y, self.chirality_z, self.chirality_w,
            self.handedness_det,
            self.symmetry_x, self.symmetry_y, self.symmetry_z, self.symmetry_w,
            self.flatness_3d, self.flatness_2d, self.intrinsic_dim_estimate,
            self.edge_length_cv,
            self.spikiness, self.max_exposure,
            self.conformity_ratio, self.distance_kurtosis, self.loner_score,
            self.coord_mean, self.coord_median, self.coord_mode,
            self.coord_std, self.coord_iqr, self.coord_range,
            self.coord_skewness, self.coord_kurtosis, self.mean_median_diff,
            self.shannon_entropy, self.joint_entropy,
            self.compression_ratio, self.sorted_compression_ratio, self.sort_compression_gain,
            self.spread_ratio,
            self.normality_pvalue, self.uniform_ks_stat,
            self.mean_abs_correlation,
            self.phi_ratio_count, self.fibonacci_count,
            self.zero_count, self.one_count, self.neg_one_count, self.prime_count,
        ]
        embedding.extend(scalars)

        # Lists (fixed size)
        embedding.extend(self.centroid[:4] if self.centroid else [0]*4)
        embedding.extend(self.inertia_eigenvalues[:4] if self.inertia_eigenvalues else [0]*4)
        embedding.extend(self.pca_variance_ratios[:4] if self.pca_variance_ratios else [0]*4)
        embedding.extend(self.axis_means[:4] if self.axis_means else [0]*4)
        embedding.extend(self.axis_medians[:4] if self.axis_medians else [0]*4)
        embedding.extend(self.axis_stds[:4] if self.axis_stds else [0]*4)
        embedding.extend(self.axis_skewness[:4] if self.axis_skewness else [0]*4)
        embedding.extend(self.axis_kurtosis[:4] if self.axis_kurtosis else [0]*4)
        embedding.extend(self.axis_entropies[:4] if self.axis_entropies else [0]*4)
        embedding.extend(self.axis_balance[:4] if self.axis_balance else [0]*4)
        embedding.extend(self.axis_q1[:4] if self.axis_q1 else [0]*4)
        embedding.extend(self.axis_q2[:4] if self.axis_q2 else [0]*4)
        embedding.extend(self.axis_q3[:4] if self.axis_q3 else [0]*4)
        embedding.extend(self.axis_iqr[:4] if self.axis_iqr else [0]*4)
        embedding.extend(self.axis_quartile_skew[:4] if self.axis_quartile_skew else [0]*4)
        embedding.extend(self.corr_eigenvalues[:4] if self.corr_eigenvalues else [0]*4)

        # Spiral correlations (4 axes)
        for axis in ['x', 'y', 'z', 'w']:
            embedding.append(self.spiral_correlations.get(axis, 0.0))

        # Correlations (6 pairs)
        for pair in ['xy', 'xz', 'xw', 'yz', 'yw', 'zw']:
            embedding.append(self.correlations.get(pair, 0.0))

        return embedding


def is_prime(n: int) -> bool:
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD after optimal rotation alignment (Kabsch algorithm)."""
    # Center both
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # Compute optimal rotation
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and compute RMSD
    Q_rotated = Q_centered @ R
    rmsd = np.sqrt(np.mean(np.sum((P_centered - Q_rotated)**2, axis=1)))
    return rmsd


def vertex_overlap(v1: np.ndarray, v2: np.ndarray, tolerance: float = 0.1) -> float:
    """Compute fraction of vertices that overlap after transformation."""
    if len(v1) == 0 or len(v2) == 0:
        return 0.0

    # For each vertex in v1, find nearest in v2
    dists = cdist(v1, v2)
    min_dists = dists.min(axis=1)
    overlap_count = np.sum(min_dists < tolerance)
    return overlap_count / len(v1)


def compute_all_heuristics(polytope: dict, polytope_id: int) -> PolytopeHeuristics:
    """Compute every heuristic for a polytope."""

    h11 = polytope["h11"]
    h21 = polytope["h21"]
    vertex_count = polytope["vertex_count"]
    vertices_flat = np.array(polytope["vertices"])

    # Reshape to (n, 4)
    vertices = vertices_flat.reshape(-1, 4)
    coords_flat = vertices.flatten()

    h = PolytopeHeuristics(
        polytope_id=polytope_id,
        h11=h11,
        h21=h21,
        vertex_count=vertex_count,
    )

    # === BASIC GEOMETRY ===
    centroid = np.mean(vertices, axis=0)
    h.centroid = centroid.tolist()

    vertices_centered = vertices - centroid
    distances = np.linalg.norm(vertices_centered, axis=1)

    # === π-NESS / CIRCULARITY ===
    if np.mean(distances) > 0:
        h.sphericity = 1 - np.std(distances) / np.mean(distances)

    # Inertia tensor (simplified: treat vertices as point masses)
    I = np.zeros((4, 4))
    for v in vertices_centered:
        I += np.outer(v, v)
    eigenvalues = np.sort(np.linalg.eigvalsh(I))
    h.inertia_eigenvalues = eigenvalues.tolist()
    if eigenvalues[-1] > 0:
        h.inertia_isotropy = eigenvalues[0] / eigenvalues[-1]

    # === SPIRALITY ===
    # Project onto each axis, measure angle-distance correlation
    for i, axis_name in enumerate(['x', 'y', 'z', 'w']):
        projected = vertices_centered[:, i]
        # Compute angle in the orthogonal 3D subspace
        other_axes = [j for j in range(4) if j != i]
        if len(other_axes) >= 2:
            angles = np.arctan2(vertices_centered[:, other_axes[0]],
                               vertices_centered[:, other_axes[1]])
            if np.std(projected) > 0 and np.std(angles) > 0:
                corr = np.corrcoef(projected, angles)[0, 1]
                h.spiral_correlations[axis_name] = float(corr) if not np.isnan(corr) else 0.0

    # === CHIRALITY ===
    # Optimal (Kabsch RMSD) - same for all axes due to optimal alignment
    mirrored_any = vertices.copy()
    mirrored_any[:, 0] *= -1  # Mirror along any axis for Kabsch
    h.chirality_optimal = kabsch_rmsd(vertices, mirrored_any)

    # Axis-specific chirality (no alignment - measures directional asymmetry)
    # Use mean of min distances from each point to nearest in reflected set
    for i, axis_name in enumerate(['x', 'y', 'z', 'w']):
        mirrored = vertices_centered.copy()
        mirrored[:, i] *= -1
        # Compute pairwise distances between original and mirrored
        dists = cdist(vertices_centered, mirrored)
        # Mean of minimum distance for each point to its nearest in reflected set
        min_dists = np.min(dists, axis=1)
        setattr(h, f'chirality_{axis_name}', float(np.mean(min_dists)))

    # Handedness from determinant (use first 4 vertices if available)
    if len(vertices) >= 4:
        h.handedness_det = float(np.sign(np.linalg.det(vertices[:4, :])))

    # === SYMMETRY ===
    for i, axis_name in enumerate(['x', 'y', 'z', 'w']):
        reflected = vertices.copy()
        reflected[:, i] *= -1
        overlap = vertex_overlap(vertices, reflected)
        setattr(h, f'symmetry_{axis_name}', overlap)

    # === FLATNESS (PCA) ===
    if len(vertices) > 4:
        pca = PCA(n_components=min(4, len(vertices)))
        pca.fit(vertices)
        h.pca_variance_ratios = pca.explained_variance_ratio_.tolist()
        h.flatness_3d = float(sum(h.pca_variance_ratios[:3])) if len(h.pca_variance_ratios) >= 3 else 1.0
        h.flatness_2d = float(sum(h.pca_variance_ratios[:2])) if len(h.pca_variance_ratios) >= 2 else 1.0
        # Estimate intrinsic dimension (how many components for 95% variance)
        cumsum = np.cumsum(h.pca_variance_ratios)
        h.intrinsic_dim_estimate = float(np.searchsorted(cumsum, 0.95) + 1)

    # === SPIKINESS ===
    if np.mean(distances) > 0:
        h.spikiness = float(np.max(distances) / np.mean(distances))
        h.max_exposure = float(np.max(distances))

    # === CONCENTRATION / OUTLIERS ===
    median_dist = np.median(distances)
    core_count = np.sum(distances < median_dist)
    outlier_count = np.sum(distances > 2 * median_dist)
    h.conformity_ratio = float(core_count / (outlier_count + 1))
    h.distance_kurtosis = float(kurtosis(distances)) if len(distances) > 3 else 0.0

    # Loner score
    if len(vertices) > 1:
        pairwise = cdist(vertices, vertices)
        np.fill_diagonal(pairwise, np.inf)
        nn_dists = pairwise.min(axis=1)
        h.loner_score = float(np.max(nn_dists) / np.mean(nn_dists)) if np.mean(nn_dists) > 0 else 0.0

    # === BASIC STATS ===
    h.coord_mean = float(np.mean(coords_flat))
    h.coord_median = float(np.median(coords_flat))
    mode_result = scipy_stats.mode(coords_flat, keepdims=False)
    h.coord_mode = float(mode_result.mode)
    h.coord_std = float(np.std(coords_flat))
    h.coord_iqr = float(np.percentile(coords_flat, 75) - np.percentile(coords_flat, 25))
    h.coord_range = float(np.max(coords_flat) - np.min(coords_flat))
    h.coord_skewness = float(skew(coords_flat))
    h.coord_kurtosis = float(kurtosis(coords_flat))
    h.mean_median_diff = abs(h.coord_mean - h.coord_median)

    # Per-axis stats
    for axis in range(4):
        axis_coords = vertices[:, axis]
        h.axis_means.append(float(np.mean(axis_coords)))
        h.axis_medians.append(float(np.median(axis_coords)))
        h.axis_stds.append(float(np.std(axis_coords)))
        h.axis_skewness.append(float(skew(axis_coords)) if len(axis_coords) > 2 else 0.0)
        h.axis_kurtosis.append(float(kurtosis(axis_coords)) if len(axis_coords) > 3 else 0.0)

        # Quartiles
        q1, q2, q3 = np.percentile(axis_coords, [25, 50, 75])
        h.axis_q1.append(float(q1))
        h.axis_q2.append(float(q2))
        h.axis_q3.append(float(q3))
        h.axis_iqr.append(float(q3 - q1))
        h.axis_quartile_skew.append(float((q3 + q1 - 2*q2) / (q3 - q1 + 1e-10)))

    # === ENTROPY / INFORMATION ===
    hist, _ = np.histogram(coords_flat, bins=20, density=True)
    hist = hist[hist > 0]
    h.shannon_entropy = float(entropy(hist)) if len(hist) > 0 else 0.0

    for axis in range(4):
        hist, _ = np.histogram(vertices[:, axis], bins=10, density=True)
        hist = hist[hist > 0]
        h.axis_entropies.append(float(entropy(hist)) if len(hist) > 0 else 0.0)

    # Joint entropy
    try:
        joint_hist, _ = np.histogramdd(vertices, bins=5)
        joint_flat = joint_hist.flatten()
        joint_flat = joint_flat[joint_flat > 0]
        h.joint_entropy = float(entropy(joint_flat / joint_flat.sum())) if len(joint_flat) > 0 else 0.0
    except:
        h.joint_entropy = 0.0

    # === COMPRESSIBILITY ===
    vertex_bytes = json.dumps(vertices.tolist()).encode()
    compressed = zlib.compress(vertex_bytes, level=9)
    h.compression_ratio = len(compressed) / len(vertex_bytes)

    sorted_coords = np.sort(coords_flat)
    sorted_bytes = json.dumps(sorted_coords.tolist()).encode()
    sorted_compressed = zlib.compress(sorted_bytes, level=9)
    h.sorted_compression_ratio = len(sorted_compressed) / len(sorted_bytes)
    h.sort_compression_gain = len(compressed) / len(sorted_compressed) if len(sorted_compressed) > 0 else 1.0

    # === AXIS BALANCE ===
    for axis in range(4):
        positive = np.sum(vertices[:, axis] > 0)
        negative = np.sum(vertices[:, axis] < 0)
        h.axis_balance.append(float(min(positive, negative) / max(positive, negative, 1)))

    axis_spreads = [np.std(vertices[:, i]) for i in range(4)]
    h.spread_ratio = float(max(axis_spreads) / min(axis_spreads)) if min(axis_spreads) > 0 else 0.0

    # === DISTRIBUTION TESTS ===
    if len(coords_flat) > 8:
        try:
            _, h.normality_pvalue = scipy_stats.normaltest(coords_flat)
        except:
            h.normality_pvalue = 0.0

        try:
            normalized = (coords_flat - coords_flat.min()) / (coords_flat.max() - coords_flat.min() + 1e-10)
            _, h.uniform_ks_stat = scipy_stats.kstest(normalized, 'uniform')
        except:
            h.uniform_ks_stat = 0.0

    # === CORRELATION ===
    if len(vertices) > 2:
        corr_matrix = np.corrcoef(vertices.T)
        pairs = [('xy', 0, 1), ('xz', 0, 2), ('xw', 0, 3), ('yz', 1, 2), ('yw', 1, 3), ('zw', 2, 3)]
        for name, i, j in pairs:
            val = corr_matrix[i, j]
            h.correlations[name] = float(val) if not np.isnan(val) else 0.0

        triu_indices = np.triu_indices(4, k=1)
        corr_vals = corr_matrix[triu_indices]
        h.mean_abs_correlation = float(np.nanmean(np.abs(corr_vals)))
        h.corr_eigenvalues = np.sort(np.linalg.eigvalsh(np.nan_to_num(corr_matrix))).tolist()

    # === GOLDEN RATIO ===
    phi = (1 + np.sqrt(5)) / 2
    phi_count = 0
    for v in vertices:
        for i in range(4):
            for j in range(i+1, 4):
                if abs(v[j]) > 0.01:
                    ratio = abs(v[i] / v[j])
                    if abs(ratio - phi) < 0.1 or abs(ratio - 1/phi) < 0.1:
                        phi_count += 1
    h.phi_ratio_count = phi_count

    # Fibonacci numbers in coordinates
    fib = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
    h.fibonacci_count = int(np.sum([abs(int(c)) in fib for c in coords_flat if c == int(c)]))

    # === INTEGER PATTERNS ===
    h.zero_count = int(np.sum(coords_flat == 0))
    h.one_count = int(np.sum(coords_flat == 1))
    h.neg_one_count = int(np.sum(coords_flat == -1))
    h.prime_count = int(np.sum([is_prime(abs(int(c))) for c in coords_flat if c == int(c)]))

    return h


def load_random_polytopes(jsonl_path: str, idx_path: str, n: int = 3) -> list[tuple[int, dict]]:
    """Load n random polytopes from the database."""

    # Read index
    with open(idx_path, "rb") as f:
        buffer = f.read()

    # Parse binary index: 8 bytes file_len + N*8 bytes u64 offsets
    offsets = []
    view = memoryview(buffer)
    for i in range(8, len(buffer), 8):
        offset = struct.unpack("<Q", view[i:i+8])[0]
        offsets.append(offset)

    print(f"Index has {len(offsets)} polytopes")

    # Random sample
    indices = np.random.choice(len(offsets), size=min(n, len(offsets)), replace=False)

    results = []
    with open(jsonl_path, "r") as f:
        for idx in indices:
            f.seek(offsets[idx])
            line = f.readline()
            data = json.loads(line)
            results.append((int(idx), data))

    return results


def main():
    import chromadb

    jsonl_path = "polytopes_three_gen.jsonl"
    idx_path = "polytopes_three_gen.jsonl.idx"

    print("Loading 3 random polytopes...")
    polytopes = load_random_polytopes(jsonl_path, idx_path, n=3)

    print(f"\nComputing heuristics for {len(polytopes)} polytopes...\n")

    all_heuristics = []

    for polytope_id, data in polytopes:
        print(f"{'='*60}")
        print(f"Polytope #{polytope_id}")
        print(f"{'='*60}")

        h = compute_all_heuristics(data, polytope_id)
        all_heuristics.append(h)

        # Pretty print all heuristics
        d = asdict(h)
        for key, value in d.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
                print(f"  {key}: [{', '.join(f'{v:.4f}' for v in value)}]")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.6f}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        print(f"\n  Embedding vector length: {len(h.to_embedding())}")
        print()

    # Store in ChromaDB
    print("\nStoring in ChromaDB...")
    client = chromadb.PersistentClient(
        path="./chroma_heuristics",
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name="polytope_heuristics",
        metadata={"hnsw:space": "cosine"}
    )

    for h in all_heuristics:
        embedding = h.to_embedding()
        collection.upsert(
            ids=[str(h.polytope_id)],
            embeddings=[embedding],
            metadatas=[{"h11": h.h11, "h21": h.h21, "vertex_count": h.vertex_count}],
        )

    print(f"Stored {len(all_heuristics)} polytopes in ChromaDB")
    print(f"Embedding dimension: {len(all_heuristics[0].to_embedding())}")

    # Save full heuristics as JSON for inspection
    output_path = "heuristics_sample.json"
    with open(output_path, "w") as f:
        json.dump([asdict(h) for h in all_heuristics], f, indent=2)
    print(f"Saved full heuristics to {output_path}")


if __name__ == "__main__":
    main()
