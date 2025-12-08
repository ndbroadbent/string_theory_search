#!/usr/bin/env python3
"""
Download ALL polytopes from Kreuzer-Skarke database via Hugging Face.
Uses the datasets library for streaming large datasets.

Downloads ~473 million polytopes (15.8 GB)

Usage:
    python download_all_polytopes.py [--output-dir DIR] [--max-polytopes N]
"""

import json
import argparse
from pathlib import Path

# Hugging Face dataset info
HF_REPO = "calabi-yau-data/polytopes-4d"


def download_polytopes(output_file: Path, max_polytopes: int = None, min_vertices: int = None, max_vertices: int = None):
    """Download polytopes using the datasets library."""
    from datasets import load_dataset

    print(f"Loading dataset from {HF_REPO}...")
    print("This may take a while for the full dataset (15.8 GB, 473M polytopes)")

    # Stream the dataset to avoid loading everything into memory
    dataset = load_dataset(HF_REPO, split="full", streaming=True)

    polytopes = []
    total = 0
    skipped = 0

    print("Processing polytopes...")
    for i, row in enumerate(dataset):
        # Filter by vertex count if specified
        vertex_count = row.get("vertex_count", len(row["vertices"]) // 4 if row["vertices"] else 0)

        if min_vertices and vertex_count < min_vertices:
            skipped += 1
            continue
        if max_vertices and vertex_count > max_vertices:
            skipped += 1
            continue

        polytope = {
            "id": total,
            "vertices": row["vertices"],
            "h11": row["h11"],
            "h21": row.get("h12", row.get("h21", 0)),  # h12 in this dataset = h21
            "vertex_count": vertex_count,
        }
        polytopes.append(polytope)
        total += 1

        if total % 100000 == 0:
            print(f"  Processed {total:,} polytopes (skipped {skipped:,})...")

        if max_polytopes and total >= max_polytopes:
            print(f"  Reached max_polytopes limit ({max_polytopes})")
            break

    print(f"\nWriting {total:,} polytopes to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(polytopes, f)

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"Done! File size: {file_size:.1f} MB")
    print(f"Total polytopes: {total:,}")
    if skipped:
        print(f"Skipped (filtered): {skipped:,}")

    return total


def main():
    parser = argparse.ArgumentParser(description="Download Kreuzer-Skarke polytopes")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                       help="Directory to store output file")
    parser.add_argument("--output-file", type=str, default="polytopes_full.json",
                       help="Output filename")
    parser.add_argument("--max-polytopes", type=int, default=None,
                       help="Maximum polytopes to download (for testing)")
    parser.add_argument("--min-vertices", type=int, default=None,
                       help="Minimum vertex count filter")
    parser.add_argument("--max-vertices", type=int, default=None,
                       help="Maximum vertex count filter")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / args.output_file

    print("=" * 60)
    print("Kreuzer-Skarke Polytope Downloader")
    print("=" * 60)
    print(f"Output: {output_file}")
    if args.max_polytopes:
        print(f"Limit: {args.max_polytopes:,} polytopes")
    if args.min_vertices or args.max_vertices:
        print(f"Vertex filter: {args.min_vertices or 'any'} - {args.max_vertices or 'any'}")
    print()

    download_polytopes(
        output_file,
        max_polytopes=args.max_polytopes,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
    )


if __name__ == "__main__":
    main()
