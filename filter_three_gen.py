#!/usr/bin/env python3
"""
Filter polytopes to only those that could give 3 generations.

For 3 generations in string theory:
  Euler characteristic χ = 2(h11 - h21) = ±6
  This means |h11 - h21| = 3

Features:
- Streaming output: writes each polytope immediately to JSONL
- Resume capability: tracks completed files, skips already-processed
- Crash-safe: progress persisted after each file
- Idempotent: can run multiple times safely
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_progress(progress_file: Path) -> dict:
    """Load progress from checkpoint file."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        "completed_files": [],
        "total_scanned": 0,
        "three_gen": 0,
        "by_hodge_pair": {},
    }


def save_progress(progress_file: Path, progress: dict):
    """Save progress to checkpoint file (atomic write)."""
    temp = progress_file.with_suffix('.tmp')
    with open(temp, 'w') as f:
        json.dump(progress, f, indent=2)
    temp.rename(progress_file)


def filter_polytopes(parquet_dir: Path, output_file: Path):
    """Filter parquet files to extract 3-generation candidates.

    - Streams output directly to JSONL file (append mode)
    - Checkpoints progress after each parquet file
    - Resumes from where it left off if interrupted
    """
    import pyarrow.parquet as pq

    progress_file = output_file.with_suffix('.progress.json')
    progress = load_progress(progress_file)

    print(f"Scanning parquet files in {parquet_dir}...", flush=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files", flush=True)

    completed = set(progress["completed_files"])
    if completed:
        print(f"Resuming: {len(completed)} files already processed", flush=True)
        print(f"  Already found {progress['three_gen']:,} three-gen candidates", flush=True)

    # Append to output file (creates if doesn't exist)
    with open(output_file, 'a') as out:
        for i, parquet_file in enumerate(parquet_files):
            filename = parquet_file.name

            if filename in completed:
                print(f"[{i+1}/{len(parquet_files)}] Skipping {filename} (already done)", flush=True)
                continue

            print(f"\n[{i+1}/{len(parquet_files)}] Processing {filename}...", flush=True)

            file_found = 0
            file_scanned = 0

            # Read in batches to limit memory
            parquet = pq.ParquetFile(parquet_file)

            for batch in parquet.iter_batches(batch_size=100000):
                df = batch.to_pandas()

                for _, row in df.iterrows():
                    file_scanned += 1

                    h11 = int(row["h11"])
                    h21 = int(row.get("h12", row.get("h21", 0)))

                    # Check 3-generation constraint: |h11 - h21| = 3
                    if abs(h11 - h21) == 3:
                        file_found += 1

                        # Track by Hodge pair
                        hodge_key = f"({h11},{h21})"
                        progress["by_hodge_pair"][hodge_key] = progress["by_hodge_pair"].get(hodge_key, 0) + 1

                        # Handle numpy arrays
                        vertices_raw = row["vertices"]
                        vertices = np.vstack(vertices_raw).flatten().tolist()

                        polytope = {
                            "vertices": vertices,
                            "h11": h11,
                            "h21": h21,
                            "vertex_count": int(row.get("vertex_count", len(vertices) // 4)),
                        }
                        # Write immediately - no buffering risk
                        out.write(json.dumps(polytope) + '\n')
                        out.flush()  # Force to disk

                del df

            # Update progress after completing file
            progress["completed_files"].append(filename)
            progress["total_scanned"] += file_scanned
            progress["three_gen"] += file_found

            print(f"  Found {file_found:,} three-gen in {file_scanned:,} polytopes", flush=True)
            print(f"  Total so far: {progress['three_gen']:,} / {progress['total_scanned']:,}", flush=True)

            # Checkpoint progress to disk
            save_progress(progress_file, progress)

    print(f"\n{'='*60}", flush=True)
    print("FILTERING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total polytopes scanned: {progress['total_scanned']:,}", flush=True)
    print(f"Three-generation candidates: {progress['three_gen']:,}", flush=True)
    if progress['total_scanned'] > 0:
        print(f"Reduction: {100 * (1 - progress['three_gen']/progress['total_scanned']):.2f}%", flush=True)

    print(f"\nDistribution by Hodge numbers (h11, h21):", flush=True)
    sorted_pairs = sorted(progress["by_hodge_pair"].items(), key=lambda x: -x[1])
    for hodge_pair, count in sorted_pairs[:20]:
        print(f"  {hodge_pair}: {count:,}", flush=True)
    if len(sorted_pairs) > 20:
        print(f"  ... and {len(sorted_pairs) - 20} more pairs", flush=True)

    print(f"\nOutput written to: {output_file}", flush=True)
    print(f"Progress saved to: {progress_file}", flush=True)

    return progress


def sort_output(input_file: Path, output_file: Path = None):
    """Sort JSONL file by Hodge number sum (optional post-processing step)."""
    if output_file is None:
        output_file = input_file.with_suffix('.sorted.json')

    print(f"Sorting {input_file} by Hodge number sum...", flush=True)

    # Stream read, sort in memory, write
    polytopes = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                polytopes.append(json.loads(line))

    polytopes.sort(key=lambda p: p["h11"] + p["h21"])

    # Add sequential IDs
    for i, p in enumerate(polytopes):
        p["id"] = i

    print(f"Writing {len(polytopes):,} sorted polytopes to {output_file}...", flush=True)
    with open(output_file, 'w') as f:
        json.dump(polytopes, f)

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"Done! File size: {file_size:.1f} MB", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Filter polytopes for 3-generation candidates")
    parser.add_argument("--parquet-dir", type=Path, default=Path("polytope_data"),
                       help="Directory containing parquet files")
    parser.add_argument("--output", type=Path, default=Path("polytopes_three_gen.jsonl"),
                       help="Output JSONL file (streaming, resumable)")
    parser.add_argument("--sort", action="store_true",
                       help="After filtering, sort output by Hodge number sum")
    parser.add_argument("--sort-only", type=Path, metavar="JSONL_FILE",
                       help="Only sort an existing JSONL file (no filtering)")
    args = parser.parse_args()

    if args.sort_only:
        sort_output(args.sort_only)
        return

    if not args.parquet_dir.exists():
        print(f"Error: {args.parquet_dir} does not exist")
        return

    filter_polytopes(args.parquet_dir, args.output)

    if args.sort:
        sort_output(args.output, args.output.with_suffix('.json'))


if __name__ == "__main__":
    main()
