#!/usr/bin/env python3
"""
Hunt for interesting correlations in polytope heuristics.
Finds unexpected relationships between metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

HEURISTICS_PATH = Path("heuristics_sample.json")


def load_heuristics() -> pd.DataFrame:
    """Load heuristics into a DataFrame."""
    with open(HEURISTICS_PATH) as f:
        data = json.load(f)

    # Flatten nested dicts
    rows = []
    for h in data:
        row = {}
        for key, value in h.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    row[f"{key}_{subkey}"] = subval
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    row[f"{key}_{i}"] = v
            else:
                row[key] = value
        rows.append(row)

    return pd.DataFrame(rows)


def find_surprising_correlations(df: pd.DataFrame, min_correlation: float = 0.5) -> list[dict]:
    """
    Find correlations that are:
    1. Strong (above threshold)
    2. Surprising (not between obviously related metrics)
    """
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove obviously related pairs (same prefix)
    def get_prefix(col: str) -> str:
        parts = col.split('_')
        return parts[0] if len(parts) > 1 else col

    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()

    results = []
    for col1, col2 in combinations(numeric_cols, 2):
        corr = corr_matrix.loc[col1, col2]

        if np.isnan(corr):
            continue

        if abs(corr) < min_correlation:
            continue

        # Check if same family (less surprising)
        prefix1, prefix2 = get_prefix(col1), get_prefix(col2)
        same_family = prefix1 == prefix2

        # Check for trivially related names
        trivial = any([
            col1 in col2 or col2 in col1,
            {prefix1, prefix2} <= {'axis', 'coord', 'symmetry', 'chirality'},
        ])

        results.append({
            'col1': col1,
            'col2': col2,
            'correlation': corr,
            'same_family': same_family,
            'trivial': trivial,
            'surprise_score': abs(corr) * (0.5 if same_family else 1.0) * (0.3 if trivial else 1.0)
        })

    return sorted(results, key=lambda x: -x['surprise_score'])


def find_clusters(df: pd.DataFrame, n_clusters: int = 5):
    """Find natural clusters in the data using PCA + KMeans."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(10, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    print(f"\n=== Cluster Analysis ({n_clusters} clusters) ===")
    print(f"PCA explained variance: {sum(pca.explained_variance_ratio_[:3]):.1%} (first 3)")

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_df = df[mask]
        print(f"\nCluster {cluster_id}: {mask.sum()} polytopes")

        # Show distinguishing features
        for col in ['sphericity', 'spikiness', 'h11', 'h21', 'shannon_entropy']:
            if col in df.columns:
                cluster_mean = cluster_df[col].mean()
                overall_mean = df[col].mean()
                if overall_mean != 0:
                    diff = (cluster_mean - overall_mean) / overall_mean * 100
                    if abs(diff) > 20:
                        print(f"  {col}: {cluster_mean:.3f} ({diff:+.0f}% vs avg)")


def find_outlier_dimensions(df: pd.DataFrame):
    """Find dimensions where certain polytopes are extreme outliers."""
    print("\n=== Outlier Analysis ===")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 10:
            continue

        mean, std = values.mean(), values.std()
        if std == 0:
            continue

        z_scores = (values - mean) / std
        outliers = z_scores.abs() > 3

        if outliers.sum() > 0:
            outlier_ids = df.loc[outliers.index[outliers], 'polytope_id'].tolist()
            outlier_vals = values[outliers].tolist()
            print(f"\n{col}: {outliers.sum()} outliers (>3σ)")
            for pid, val in zip(outlier_ids[:3], outlier_vals[:3]):
                print(f"  Polytope {pid}: {val:.4f} (z={z_scores[df[df.polytope_id == pid].index[0]]:.1f})")


def main():
    print("Loading heuristics...")
    df = load_heuristics()
    print(f"Loaded {len(df)} polytopes with {len(df.columns)} features")

    # Find surprising correlations
    print("\n=== Top Surprising Correlations ===")
    correlations = find_surprising_correlations(df, min_correlation=0.4)

    # Filter to truly surprising ones
    surprising = [c for c in correlations if not c['trivial'] and not c['same_family']]

    print("\nMost surprising (different families, non-trivial):")
    for c in surprising[:15]:
        sign = "+" if c['correlation'] > 0 else ""
        print(f"  {c['col1']} ↔ {c['col2']}: {sign}{c['correlation']:.3f}")

    print("\n\nStrong correlations within families:")
    within_family = [c for c in correlations if c['same_family'] and abs(c['correlation']) > 0.7]
    for c in within_family[:10]:
        sign = "+" if c['correlation'] > 0 else ""
        print(f"  {c['col1']} ↔ {c['col2']}: {sign}{c['correlation']:.3f}")

    # Cluster analysis
    if len(df) >= 10:
        find_clusters(df, n_clusters=min(5, len(df) // 10))

    # Outlier analysis
    find_outlier_dimensions(df)

    # Specific interesting questions
    print("\n=== Specific Investigations ===")

    # Does sphericity predict entropy?
    if 'sphericity' in df.columns and 'shannon_entropy' in df.columns:
        corr = df['sphericity'].corr(df['shannon_entropy'])
        print(f"\nSphericity vs Entropy: {corr:.3f}")

    # Does h11-h21 difference correlate with anything?
    if 'h11' in df.columns and 'h21' in df.columns:
        df['h_diff'] = df['h11'] - df['h21']
        interesting = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['h11', 'h21', 'h_diff', 'polytope_id']:
                corr = df['h_diff'].corr(df[col])
                if abs(corr) > 0.3:
                    interesting.append((col, corr))

        if interesting:
            print("\nh11-h21 difference correlates with:")
            for col, corr in sorted(interesting, key=lambda x: -abs(x[1]))[:5]:
                print(f"  {col}: {corr:.3f}")

    # Chirality asymmetry
    chirality_cols = [c for c in df.columns if c.startswith('chirality_') and c != 'chirality_optimal']
    if len(chirality_cols) == 4:
        df['chirality_asymmetry'] = df[chirality_cols].std(axis=1)
        print(f"\nChirality asymmetry (std across axes):")
        print(f"  Mean: {df['chirality_asymmetry'].mean():.4f}")
        print(f"  Max: {df['chirality_asymmetry'].max():.4f} (polytope {df.loc[df['chirality_asymmetry'].idxmax(), 'polytope_id']})")

        # What correlates with chirality asymmetry?
        interesting = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'chirality' not in col and col != 'polytope_id':
                corr = df['chirality_asymmetry'].corr(df[col])
                if abs(corr) > 0.3:
                    interesting.append((col, corr))

        if interesting:
            print("  Correlates with:")
            for col, corr in sorted(interesting, key=lambda x: -abs(x[1]))[:5]:
                print(f"    {col}: {corr:.3f}")


if __name__ == "__main__":
    main()
