#!/usr/bin/env python3
"""
Compute heuristics for polytopes and store in SQLite + JSON + ChromaDB.

Usage:
    # Compute for N random polytopes (adds to existing)
    python scripts/compute_heuristics.py --count 100

    # Recompute all existing polytopes (clears and rebuilds)
    python scripts/compute_heuristics.py --recompute

    # Compute for specific polytope IDs
    python scripts/compute_heuristics.py --ids 12345 67890 11111
"""

import argparse
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_all_heuristics import compute_all_heuristics, PolytopeHeuristics
from dataclasses import asdict


JSONL_PATH = Path("polytopes_three_gen.jsonl")
IDX_PATH = Path("polytopes_three_gen.jsonl.idx")
HEURISTICS_PATH = Path("heuristics_sample.json")
CHROMA_PATH = Path("chroma_heuristics")
DB_PATH = Path("data/string_theory.db")


def load_index() -> list[int]:
    """Load the binary index file, return list of byte offsets."""
    with open(IDX_PATH, "rb") as f:
        buffer = f.read()

    offsets = []
    view = memoryview(buffer)
    for i in range(8, len(buffer), 8):
        offset = struct.unpack("<Q", view[i:i+8])[0]
        offsets.append(offset)

    return offsets


def load_polytope(jsonl_file, offsets: list[int], idx: int) -> dict:
    """Load a single polytope by index."""
    jsonl_file.seek(offsets[idx])
    line = jsonl_file.readline()
    return json.loads(line)


def load_existing_heuristics() -> list[dict]:
    """Load existing heuristics from JSON file."""
    if HEURISTICS_PATH.exists():
        with open(HEURISTICS_PATH) as f:
            return json.load(f)
    return []


def save_heuristics(heuristics: list[dict]):
    """Save heuristics to JSON file."""
    with open(HEURISTICS_PATH, "w") as f:
        json.dump(heuristics, f, indent=2)
    print(f"Saved {len(heuristics)} heuristics to {HEURISTICS_PATH}")


def save_to_sqlite(heuristics: list[dict]):
    """Save heuristics to SQLite database."""
    if not DB_PATH.exists():
        print(f"Warning: Database not found at {DB_PATH}. Run the search binary first to create schema.")
        return

    conn = sqlite3.connect(DB_PATH)

    # Column mapping from JSON to SQLite
    column_mapping = {
        'polytope_id': 'polytope_id',
        'sphericity': 'sphericity',
        'inertia_isotropy': 'inertia_isotropy',
        'chirality_optimal': 'chirality_optimal',
        'chirality_x': 'chirality_x',
        'chirality_y': 'chirality_y',
        'chirality_z': 'chirality_z',
        'chirality_w': 'chirality_w',
        'handedness_det': 'handedness_det',
        'symmetry_x': 'symmetry_x',
        'symmetry_y': 'symmetry_y',
        'symmetry_z': 'symmetry_z',
        'symmetry_w': 'symmetry_w',
        'flatness_3d': 'flatness_3d',
        'flatness_2d': 'flatness_2d',
        'intrinsic_dim_estimate': 'intrinsic_dim_estimate',
        'spikiness': 'spikiness',
        'max_exposure': 'max_exposure',
        'conformity_ratio': 'conformity_ratio',
        'distance_kurtosis': 'distance_kurtosis',
        'loner_score': 'loner_score',
        'coord_mean': 'coord_mean',
        'coord_median': 'coord_median',
        'coord_std': 'coord_std',
        'coord_skewness': 'coord_skewness',
        'coord_kurtosis': 'coord_kurtosis',
        'shannon_entropy': 'shannon_entropy',
        'joint_entropy': 'joint_entropy',
        'compression_ratio': 'compression_ratio',
        'sorted_compression_ratio': 'sorted_compression_ratio',
        'sort_compression_gain': 'sort_compression_gain',
        'phi_ratio_count': 'phi_ratio_count',
        'fibonacci_count': 'fibonacci_count',
        'zero_count': 'zero_count',
        'one_count': 'one_count',
        'prime_count': 'prime_count',
        'outlier_score': 'outlier_score',
        'outlier_max_zscore': 'outlier_max_zscore',
        'outlier_max_dim': 'outlier_max_dim',
        'outlier_count_2sigma': 'outlier_count_2sigma',
        'outlier_count_3sigma': 'outlier_count_3sigma',
    }

    inserted = 0
    for record in heuristics:
        polytope_id = record.get('polytope_id')
        if polytope_id is None:
            continue

        columns = ['polytope_id']
        values = [polytope_id]

        for json_key, db_col in column_mapping.items():
            if json_key == 'polytope_id':
                continue
            if json_key in record:
                columns.append(db_col)
                values.append(record[json_key])

        placeholders = ','.join(['?' for _ in values])
        col_str = ','.join(columns)

        try:
            conn.execute(
                f"INSERT OR REPLACE INTO heuristics ({col_str}) VALUES ({placeholders})",
                values
            )
            inserted += 1
        except sqlite3.Error as e:
            print(f"Error inserting polytope {polytope_id}: {e}")

    conn.commit()
    conn.close()
    print(f"Saved {inserted} heuristics to {DB_PATH}")


def compute_outlier_scores(heuristics: list[dict]) -> list[dict]:
    """
    Compute population-level outlier scores for each polytope.
    Must be run after all individual heuristics are computed.
    """
    if len(heuristics) < 3:
        print("Need at least 3 polytopes to compute outlier scores")
        return heuristics

    print("Computing outlier scores...")

    # Flatten nested structures and get numeric columns
    def flatten(h: dict) -> dict:
        flat = {}
        for key, value in h.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    if isinstance(subval, (int, float)):
                        flat[f"{key}_{subkey}"] = subval
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        flat[f"{key}_{i}"] = v
            elif isinstance(value, (int, float)):
                flat[key] = value
        return flat

    # Build matrix of all numeric values
    flat_data = [flatten(h) for h in heuristics]
    all_keys = set()
    for fd in flat_data:
        all_keys.update(fd.keys())

    # Remove non-metric keys
    exclude_keys = {'polytope_id', 'outlier_score', 'outlier_max_zscore', 'outlier_max_dim'}
    numeric_keys = sorted(all_keys - exclude_keys)

    # Build numpy array
    matrix = np.zeros((len(heuristics), len(numeric_keys)))
    for i, fd in enumerate(flat_data):
        for j, key in enumerate(numeric_keys):
            val = fd.get(key, 0)
            matrix[i, j] = val if not np.isnan(val) else 0

    # Compute z-scores per column
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero

    z_scores = (matrix - means) / stds

    # Compute outlier metrics for each polytope
    for i, h in enumerate(heuristics):
        abs_z = np.abs(z_scores[i])

        # Mean absolute z-score (overall outlierness)
        h['outlier_score'] = float(np.mean(abs_z))

        # Max z-score and which dimension
        max_idx = np.argmax(abs_z)
        h['outlier_max_zscore'] = float(abs_z[max_idx])
        h['outlier_max_dim'] = numeric_keys[max_idx]

        # Count of dimensions where |z| > 2 (moderately unusual)
        h['outlier_count_2sigma'] = int(np.sum(abs_z > 2))

        # Count of dimensions where |z| > 3 (very unusual)
        h['outlier_count_3sigma'] = int(np.sum(abs_z > 3))

    # Report top outliers
    sorted_by_score = sorted(heuristics, key=lambda x: -x['outlier_score'])
    print(f"Top outliers:")
    for h in sorted_by_score[:5]:
        print(f"  Polytope {h['polytope_id']}: score={h['outlier_score']:.3f}, "
              f"max_z={h['outlier_max_zscore']:.1f} ({h['outlier_max_dim']}), "
              f"3σ dims={h['outlier_count_3sigma']}")

    return heuristics


def update_chromadb(heuristics: list[dict]):
    """Update ChromaDB with heuristics."""
    import chromadb

    # Convert dicts back to PolytopeHeuristics for embedding
    def dict_to_heuristics(d: dict) -> PolytopeHeuristics:
        h = PolytopeHeuristics(
            polytope_id=d["polytope_id"],
            h11=d["h11"],
            h21=d["h21"],
            vertex_count=d["vertex_count"],
        )
        for key, value in d.items():
            if hasattr(h, key):
                setattr(h, key, value)
        return h

    print(f"Updating ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    collection_name = "polytope_heuristics"
    sample_embedding = dict_to_heuristics(heuristics[0]).to_embedding()
    expected_dim = len(sample_embedding)

    # Check if collection exists and has correct dimension
    needs_recreate = False
    try:
        collection = client.get_collection(name=collection_name)
        existing = collection.peek(limit=1)
        if existing["embeddings"] is not None and len(existing["embeddings"]) > 0:
            current_dim = len(existing["embeddings"][0])
            if current_dim != expected_dim:
                print(f"Embedding dimension changed ({current_dim} -> {expected_dim}), recreating collection...")
                needs_recreate = True
    except Exception:
        needs_recreate = True

    if needs_recreate:
        # Delete if exists, then create fresh
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # Add in batches
    batch_size = 50
    for i in range(0, len(heuristics), batch_size):
        batch = heuristics[i:i+batch_size]
        objs = [dict_to_heuristics(d) for d in batch]

        ids = [str(h.polytope_id) for h in objs]
        embeddings = [h.to_embedding() for h in objs]
        metadatas = [{"h11": h.h11, "h21": h.h21, "vertex_count": h.vertex_count} for h in objs]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"ChromaDB now has {collection.count()} polytopes")


def compute_random(count: int):
    """Compute heuristics for N random polytopes."""
    print(f"Loading index from {IDX_PATH}...")
    offsets = load_index()
    print(f"Total polytopes available: {len(offsets)}")

    existing = load_existing_heuristics()
    existing_ids = {h["polytope_id"] for h in existing}
    print(f"Existing heuristics: {len(existing)}")

    # Sample new random indices
    available = set(range(len(offsets))) - existing_ids
    if len(available) < count:
        print(f"Warning: only {len(available)} new polytopes available")
        count = len(available)

    new_indices = np.random.choice(list(available), size=count, replace=False)
    print(f"Computing heuristics for {count} new polytopes...")

    new_heuristics = []
    with open(JSONL_PATH, "r") as f:
        for i, idx in enumerate(new_indices):
            if i % 20 == 0:
                print(f"  Processing {i}/{count}...")

            data = load_polytope(f, offsets, idx)
            h = compute_all_heuristics(data, int(idx))
            new_heuristics.append(asdict(h))

    all_heuristics = existing + new_heuristics
    all_heuristics = compute_outlier_scores(all_heuristics)
    save_heuristics(all_heuristics)
    save_to_sqlite(all_heuristics)
    update_chromadb(all_heuristics)


def compute_specific(ids: list[int]):
    """Compute heuristics for specific polytope IDs."""
    print(f"Loading index from {IDX_PATH}...")
    offsets = load_index()

    existing = load_existing_heuristics()
    existing_by_id = {h["polytope_id"]: h for h in existing}

    print(f"Computing heuristics for {len(ids)} specific polytopes...")

    with open(JSONL_PATH, "r") as f:
        for idx in ids:
            if idx >= len(offsets):
                print(f"  Warning: index {idx} out of range, skipping")
                continue

            print(f"  Processing polytope {idx}...")
            data = load_polytope(f, offsets, idx)
            h = compute_all_heuristics(data, idx)
            existing_by_id[idx] = asdict(h)

    all_heuristics = list(existing_by_id.values())
    all_heuristics = compute_outlier_scores(all_heuristics)
    save_heuristics(all_heuristics)
    save_to_sqlite(all_heuristics)
    update_chromadb(all_heuristics)


def recompute_all():
    """Recompute all existing heuristics (additive - updates in place)."""
    existing = load_existing_heuristics()
    if not existing:
        print("No existing heuristics to recompute")
        return

    ids = [h["polytope_id"] for h in existing]
    print(f"Recomputing {len(ids)} polytopes...")

    # compute_specific handles upsert behavior
    compute_specific(ids)


def main():
    parser = argparse.ArgumentParser(description="Compute polytope heuristics")
    parser.add_argument("--count", "-n", type=int, help="Number of random polytopes to add")
    parser.add_argument("--ids", nargs="+", type=int, help="Specific polytope IDs to compute")
    parser.add_argument("--recompute", action="store_true", help="Recompute all existing polytopes")

    args = parser.parse_args()

    if args.recompute:
        recompute_all()
    elif args.ids:
        compute_specific(args.ids)
    elif args.count:
        compute_random(args.count)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
