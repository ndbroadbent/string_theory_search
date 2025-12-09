#!/usr/bin/env python3
"""
Multi-embedding indexer for polytope similarity search.

Creates multiple embedding representations of each polytope,
enabling meta-search to discover which embeddings (or weighted
combinations) best predict fitness similarity.

The embedding weights themselves become the genome of the outer
meta-genetic algorithm.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import chromadb


@dataclass
class PolytopeEmbeddings:
    """All embeddings for a single polytope."""
    polytope_id: int
    h11: int
    h21: int

    # Individual embedding components (each is a list of floats)
    hodge: list[float] = field(default_factory=list)
    vertex_stats: list[float] = field(default_factory=list)
    vertex_coords_flat: list[float] = field(default_factory=list)
    coord_histogram: list[float] = field(default_factory=list)
    f_vector: list[float] = field(default_factory=list)
    geometric: list[float] = field(default_factory=list)
    algebraic: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "polytope_id": self.polytope_id,
            "h11": self.h11,
            "h21": self.h21,
            "hodge": self.hodge,
            "vertex_stats": self.vertex_stats,
            "vertex_coords_flat": self.vertex_coords_flat,
            "coord_histogram": self.coord_histogram,
            "f_vector": self.f_vector,
            "geometric": self.geometric,
            "algebraic": self.algebraic,
        }

    def composite_embedding(self, weights: dict[str, float]) -> list[float]:
        """
        Create weighted combination of embeddings.

        weights: {"hodge": 0.5, "vertex_stats": 1.0, ...}

        This is the key insight: weights are the genome of the meta-GA.
        """
        components = {
            "hodge": self.hodge,
            "vertex_stats": self.vertex_stats,
            "vertex_coords_flat": self.vertex_coords_flat,
            "coord_histogram": self.coord_histogram,
            "f_vector": self.f_vector,
            "geometric": self.geometric,
            "algebraic": self.algebraic,
        }

        result = []
        for name, weight in weights.items():
            if name in components and weight > 0:
                # Normalize each component, then weight it
                comp = np.array(components[name])
                if len(comp) > 0:
                    norm = np.linalg.norm(comp)
                    if norm > 0:
                        comp = comp / norm
                    result.extend((comp * weight).tolist())

        return result


def compute_embeddings(polytope: dict, polytope_id: int) -> PolytopeEmbeddings:
    """Compute all embedding representations for a polytope."""

    h11 = polytope["h11"]
    h21 = polytope["h21"]
    vertex_count = polytope["vertex_count"]
    vertices_flat = polytope["vertices"]  # Flat array of 4D coords

    # Reshape vertices to (n, 4)
    vertices = np.array(vertices_flat).reshape(-1, 4)

    emb = PolytopeEmbeddings(
        polytope_id=polytope_id,
        h11=h11,
        h21=h21,
    )

    # 1. Hodge embedding (simple but important)
    emb.hodge = [
        float(h11),
        float(h21),
        float(h11 - h21),  # Euler/2
        float(h11 + h21),  # Total moduli
        float(h11 * h21),  # Product (interaction term)
    ]

    # 2. Vertex statistics
    emb.vertex_stats = [
        float(vertex_count),
        float(np.mean(vertices)),
        float(np.std(vertices)),
        float(np.min(vertices)),
        float(np.max(vertices)),
        float(np.median(vertices)),
        # Per-dimension stats
        *[float(x) for x in np.mean(vertices, axis=0)],
        *[float(x) for x in np.std(vertices, axis=0)],
        *[float(x) for x in np.min(vertices, axis=0)],
        *[float(x) for x in np.max(vertices, axis=0)],
    ]

    # 3. Flattened vertex coordinates (padded to fixed size)
    MAX_VERTICES = 100  # Pad/truncate to this
    coords_padded = np.zeros(MAX_VERTICES * 4)
    flat_len = min(len(vertices_flat), MAX_VERTICES * 4)
    coords_padded[:flat_len] = vertices_flat[:flat_len]
    emb.vertex_coords_flat = coords_padded.tolist()

    # 4. Coordinate histogram (distribution of values)
    # Bin coordinates into ranges [-5, -4, ..., 4, 5+]
    bins = list(range(-5, 7))  # 12 bins
    hist, _ = np.histogram(vertices_flat, bins=bins)
    emb.coord_histogram = (hist / max(1, len(vertices_flat))).tolist()

    # 5. F-vector (face counts) - approximated from vertices
    # True f-vector requires convex hull computation, approximate here
    emb.f_vector = [
        float(vertex_count),
        float(vertex_count * (vertex_count - 1) / 2),  # Upper bound on edges
        float(vertex_count * 2),  # Rough face estimate
        float(1),  # One 4D cell
    ]

    # 6. Geometric features
    centroid = np.mean(vertices, axis=0)
    distances_from_centroid = np.linalg.norm(vertices - centroid, axis=1)

    # Bounding box
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_extent = bbox_max - bbox_min

    emb.geometric = [
        *centroid.tolist(),
        float(np.mean(distances_from_centroid)),
        float(np.std(distances_from_centroid)),
        float(np.max(distances_from_centroid)),
        *bbox_extent.tolist(),
        float(np.prod(bbox_extent + 1)),  # "Volume" of bounding box
        # Aspect ratios
        float(bbox_extent[0] / max(1, bbox_extent[1])) if len(bbox_extent) > 1 else 1.0,
        float(bbox_extent[0] / max(1, bbox_extent[2])) if len(bbox_extent) > 2 else 1.0,
        float(bbox_extent[0] / max(1, bbox_extent[3])) if len(bbox_extent) > 3 else 1.0,
    ]

    # 7. Algebraic features
    emb.algebraic = [
        float(np.sum(vertices)),
        float(np.sum(vertices ** 2)),
        float(np.sum(np.abs(vertices))),
        float(np.sum(vertices[:, 0] * vertices[:, 1])) if vertices.shape[1] > 1 else 0,
        float(np.sum(vertices[:, 0] * vertices[:, 2])) if vertices.shape[1] > 2 else 0,
        float(np.sum(vertices[:, 0] * vertices[:, 3])) if vertices.shape[1] > 3 else 0,
        # Count special values
        float(np.sum(vertices == 0)),
        float(np.sum(vertices == 1)),
        float(np.sum(vertices == -1)),
        float(np.sum(np.abs(vertices) > 2)),
    ]

    return emb


def stratified_sample(jsonl_path: str, idx_path: str, samples_per_bucket: int = 100) -> list[tuple[int, dict]]:
    """
    Sample polytopes stratified by (h11, h21) to ensure coverage.
    Returns list of (polytope_id, polytope_data) tuples.
    """
    import struct

    print(f"Loading index from {idx_path}...")
    with open(idx_path, "rb") as f:
        # First 8 bytes = file length
        file_len = struct.unpack("<Q", f.read(8))[0]
        # Rest = u64 offsets
        offsets = []
        while True:
            data = f.read(8)
            if not data:
                break
            offsets.append(struct.unpack("<Q", data)[0])

    print(f"Index has {len(offsets)} polytopes")

    # First pass: bucket polytopes by (h11, h21)
    print("Bucketing polytopes by Hodge numbers...")
    buckets: dict[tuple[int, int], list[int]] = {}

    with open(jsonl_path, "r") as f:
        for idx, offset in enumerate(offsets):
            if idx % 500000 == 0:
                print(f"  Scanned {idx}/{len(offsets)}...")

            f.seek(offset)
            line = f.readline()
            try:
                data = json.loads(line)
                h11, h21 = data["h11"], data["h21"]
                key = (h11, h21)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(idx)
            except:
                continue

    print(f"Found {len(buckets)} unique (h11, h21) combinations")

    # Second pass: sample from each bucket
    print(f"Sampling {samples_per_bucket} from each bucket...")
    sampled: list[tuple[int, dict]] = []

    with open(jsonl_path, "r") as f:
        for (h11, h21), ids in buckets.items():
            # Random sample from this bucket
            n_sample = min(samples_per_bucket, len(ids))
            selected = np.random.choice(ids, size=n_sample, replace=False)

            for idx in selected:
                f.seek(offsets[idx])
                line = f.readline()
                try:
                    data = json.loads(line)
                    sampled.append((idx, data))
                except:
                    continue

    print(f"Sampled {len(sampled)} polytopes total")
    return sampled


class PolytopeIndex:
    """ChromaDB-backed multi-embedding index."""

    def __init__(self, persist_dir: str = "./chroma_polytopes"):
        # ChromaDB v1.x uses PersistentClient for disk storage
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.persist_dir = persist_dir

        # One collection per embedding type
        self.collections = {}
        self.embedding_names = [
            "hodge", "vertex_stats", "vertex_coords_flat",
            "coord_histogram", "f_vector", "geometric", "algebraic"
        ]

    def _get_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=f"polytope_{name}",
                metadata={"hnsw:space": "cosine"}
            )
        return self.collections[name]

    def add_polytope(self, emb: PolytopeEmbeddings):
        """Add a polytope's embeddings to all collections."""
        str_id = str(emb.polytope_id)
        metadata = {"h11": emb.h11, "h21": emb.h21}

        for name in self.embedding_names:
            vec = getattr(emb, name)
            if vec and len(vec) > 0:
                coll = self._get_collection(name)
                coll.upsert(
                    ids=[str_id],
                    embeddings=[vec],
                    metadatas=[metadata],
                )

    def query_neighbors(
        self,
        emb: PolytopeEmbeddings,
        weights: dict[str, float],
        k: int = 100
    ) -> list[tuple[int, float]]:
        """
        Query for k nearest neighbors using weighted embedding combination.

        Returns list of (polytope_id, distance) tuples.
        """
        # For weighted queries, we query each collection separately
        # then combine scores
        all_scores: dict[int, float] = {}

        for name, weight in weights.items():
            if weight <= 0 or name not in self.embedding_names:
                continue

            vec = getattr(emb, name)
            if not vec or len(vec) == 0:
                continue

            coll = self._get_collection(name)
            results = coll.query(
                query_embeddings=[vec],
                n_results=k * 2,  # Get more, will filter
            )

            if results and results["ids"] and results["distances"]:
                for id_str, dist in zip(results["ids"][0], results["distances"][0]):
                    pid = int(id_str)
                    # Accumulate weighted scores (lower distance = better)
                    if pid not in all_scores:
                        all_scores[pid] = 0
                    all_scores[pid] += weight * dist

        # Sort by combined score
        sorted_results = sorted(all_scores.items(), key=lambda x: x[1])
        return sorted_results[:k]


def build_index(
    jsonl_path: str = "polytopes_three_gen.jsonl",
    idx_path: str = "polytopes_three_gen.jsonl.idx",
    samples_per_bucket: int = 100,
    output_dir: str = "./chroma_polytopes",
):
    """Build the multi-embedding index with stratified sampling."""

    # Sample polytopes
    samples = stratified_sample(jsonl_path, idx_path, samples_per_bucket)

    # Create index
    index = PolytopeIndex(persist_dir=output_dir)

    # Compute embeddings and add to index
    print(f"Computing embeddings for {len(samples)} polytopes...")
    all_embeddings = []

    for i, (polytope_id, data) in enumerate(samples):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(samples)}...")

        try:
            emb = compute_embeddings(data, polytope_id)
            index.add_polytope(emb)
            all_embeddings.append(emb.to_dict())
        except Exception as e:
            print(f"  Error on polytope {polytope_id}: {e}")
            continue

    # Save embeddings to JSON for inspection
    embeddings_path = Path(output_dir) / "embeddings.json"
    print(f"Saving embeddings to {embeddings_path}...")
    with open(embeddings_path, "w") as f:
        json.dump(all_embeddings, f)

    # ChromaDB v1.x with PersistentClient auto-persists
    print(f"Done! Index saved to {output_dir}")
    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build polytope embedding index")
    parser.add_argument("--jsonl", default="polytopes_three_gen.jsonl", help="Input JSONL file")
    parser.add_argument("--idx", default="polytopes_three_gen.jsonl.idx", help="Index file")
    parser.add_argument("--samples", type=int, default=100, help="Samples per (h11,h21) bucket")
    parser.add_argument("--output", default="./chroma_polytopes", help="Output directory")

    args = parser.parse_args()

    build_index(
        jsonl_path=args.jsonl,
        idx_path=args.idx,
        samples_per_bucket=args.samples,
        output_dir=args.output,
    )
