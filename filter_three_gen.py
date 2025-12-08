#!/usr/bin/env python3
"""
Filter polytopes to only those that could give 3 generations.

For 3 generations in string theory:
  Euler characteristic χ = 2(h11 - h21) = ±6
  This means |h11 - h21| = 3

Also prioritize small Hodge numbers since they're preferred for model building.
"""

import json
import argparse
import sys
from pathlib import Path


def filter_polytopes(parquet_dir: Path, output_file: Path, prioritize_small: bool = True):
    """Filter parquet files to extract 3-generation candidates.

    Memory-efficient: processes files in batches, writes results incrementally.
    """
    import pyarrow.parquet as pq

    print(f"Scanning parquet files in {parquet_dir}...", flush=True)

    stats = {
        "total_scanned": 0,
        "three_gen": 0,
        "by_hodge_pair": {},
    }

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files", flush=True)

    # Write results incrementally to avoid memory issues
    temp_file = output_file.with_suffix('.jsonl')

    with open(temp_file, 'w') as out:
        for i, parquet_file in enumerate(parquet_files):
            print(f"\n[{i+1}/{len(parquet_files)}] Processing {parquet_file.name}...", flush=True)

            # Read in batches to limit memory
            parquet = pq.ParquetFile(parquet_file)

            for batch in parquet.iter_batches(batch_size=100000):
                df = batch.to_pandas()

                for _, row in df.iterrows():
                    stats["total_scanned"] += 1

                    h11 = int(row["h11"])
                    # h12 in this dataset is what physicists call h21
                    h21 = int(row.get("h12", row.get("h21", 0)))

                    # Check 3-generation constraint: |h11 - h21| = 3
                    if abs(h11 - h21) == 3:
                        stats["three_gen"] += 1

                        # Track by Hodge pair
                        hodge_key = f"({h11},{h21})"
                        stats["by_hodge_pair"][hodge_key] = stats["by_hodge_pair"].get(hodge_key, 0) + 1

                        polytope = {
                            "vertices": row["vertices"].tolist() if hasattr(row["vertices"], 'tolist') else list(row["vertices"]),
                            "h11": h11,
                            "h21": h21,
                            "vertex_count": int(row.get("vertex_count", len(row["vertices"]) // 4)),
                        }
                        # Write one JSON object per line
                        out.write(json.dumps(polytope) + '\n')

                # Free memory
                del df

            if stats["total_scanned"] % 1000000 == 0:
                print(f"  Scanned {stats['total_scanned']:,}, found {stats['three_gen']:,} three-gen...", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("FILTERING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total polytopes scanned: {stats['total_scanned']:,}", flush=True)
    print(f"Three-generation candidates: {stats['three_gen']:,}", flush=True)
    if stats['total_scanned'] > 0:
        print(f"Reduction: {100 * (1 - stats['three_gen']/stats['total_scanned']):.2f}%", flush=True)

    print(f"\nDistribution by Hodge numbers (h11, h21):", flush=True)
    sorted_pairs = sorted(stats["by_hodge_pair"].items(), key=lambda x: -x[1])
    for hodge_pair, count in sorted_pairs[:20]:
        print(f"  {hodge_pair}: {count:,}", flush=True)
    if len(sorted_pairs) > 20:
        print(f"  ... and {len(sorted_pairs) - 20} more pairs", flush=True)

    # Convert JSONL to sorted JSON array
    print(f"\nConverting to final JSON format...", flush=True)

    # Read back, sort, and write final JSON
    polytopes = []
    with open(temp_file, 'r') as f:
        for line in f:
            polytopes.append(json.loads(line))

    if prioritize_small:
        print("Sorting by Hodge number sum (smaller = simpler)...", flush=True)
        polytopes.sort(key=lambda p: p["h11"] + p["h21"])

    # Add sequential IDs
    for i, p in enumerate(polytopes):
        p["id"] = i

    print(f"Writing {len(polytopes):,} polytopes to {output_file}...", flush=True)
    with open(output_file, 'w') as f:
        json.dump(polytopes, f)

    # Clean up temp file
    temp_file.unlink()

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"Done! File size: {file_size:.1f} MB", flush=True)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter polytopes for 3-generation candidates")
    parser.add_argument("--parquet-dir", type=Path, default=Path("polytope_data"),
                       help="Directory containing parquet files")
    parser.add_argument("--output", type=Path, default=Path("polytopes_three_gen.json"),
                       help="Output JSON file")
    parser.add_argument("--no-sort", action="store_true",
                       help="Don't sort by Hodge number (faster)")
    args = parser.parse_args()

    if not args.parquet_dir.exists():
        print(f"Error: {args.parquet_dir} does not exist")
        return

    filter_polytopes(
        args.parquet_dir,
        args.output,
        prioritize_small=not args.no_sort,
    )


if __name__ == "__main__":
    main()
