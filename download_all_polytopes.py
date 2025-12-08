#!/usr/bin/env python3
"""
Download ALL polytopes from Kreuzer-Skarke database via Hugging Face.
Supports resume - can be stopped and restarted.

Downloads ~473 million polytopes organized by vertex count.
Total size: ~50-100 GB estimated

Usage:
    python download_all_polytopes.py [--output-dir DIR] [--min-vertices N] [--max-vertices N]
"""

import os
import json
import argparse
import requests
from pathlib import Path
import time

# Hugging Face dataset structure
HF_BASE = "https://huggingface.co/datasets/calabi-yau/kreuzer-skarke-4d/resolve/main/data"

# Vertex counts available (5-35 for 4D reflexive polytopes)
VERTEX_COUNTS = list(range(5, 36))

def download_with_retry(url: str, output_path: Path, max_retries: int = 5) -> bool:
    """Download a file with retry logic and resume support."""

    # Check if already downloaded
    if output_path.exists():
        # Verify it's valid parquet
        try:
            import pyarrow.parquet as pq
            pq.read_table(output_path)
            print(f"  Already downloaded and valid: {output_path.name}")
            return True
        except:
            print(f"  Corrupt file, re-downloading: {output_path.name}")
            output_path.unlink()

    for attempt in range(max_retries):
        try:
            print(f"  Downloading {output_path.name} (attempt {attempt + 1}/{max_retries})...")

            # Stream download with progress
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded * 100 // total_size
                        print(f"\r    {downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB ({pct}%)", end='', flush=True)

            print()  # newline after progress

            # Rename temp to final
            temp_path.rename(output_path)
            return True

        except Exception as e:
            print(f"  Error: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    return False


def convert_parquet_to_json(parquet_dir: Path, output_file: Path, max_polytopes: int = None):
    """Convert downloaded parquet files to single JSON."""
    import pyarrow.parquet as pq

    print(f"\nConverting parquet files to {output_file}...")

    polytopes = []
    total = 0

    for parquet_file in sorted(parquet_dir.glob("*.parquet")):
        print(f"  Processing {parquet_file.name}...")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()

        for _, row in df.iterrows():
            polytope = {
                "id": int(row.get("polytope_id", total)),
                "vertices": row["vertices"].tolist() if hasattr(row["vertices"], 'tolist') else row["vertices"],
                "h11": int(row["h11"]),
                "h21": int(row["h21"]),
            }
            polytopes.append(polytope)
            total += 1

            if max_polytopes and total >= max_polytopes:
                break

        if max_polytopes and total >= max_polytopes:
            break

        print(f"    Total so far: {total:,}")

    print(f"\nWriting {total:,} polytopes to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(polytopes, f)

    print(f"Done! File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download Kreuzer-Skarke polytopes")
    parser.add_argument("--output-dir", type=Path, default=Path("polytope_data"),
                       help="Directory to store downloaded files")
    parser.add_argument("--min-vertices", type=int, default=5,
                       help="Minimum vertex count to download")
    parser.add_argument("--max-vertices", type=int, default=35,
                       help="Maximum vertex count to download")
    parser.add_argument("--convert", action="store_true",
                       help="Convert to JSON after download")
    parser.add_argument("--max-polytopes", type=int, default=None,
                       help="Maximum polytopes for JSON conversion")
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Kreuzer-Skarke Polytope Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Vertex range: {args.min_vertices} - {args.max_vertices}")
    print()

    # Download each vertex count file
    successful = []
    failed = []

    for n_vertices in range(args.min_vertices, args.max_vertices + 1):
        filename = f"vertices_{n_vertices:02d}.parquet"
        url = f"{HF_BASE}/{filename}"
        output_path = args.output_dir / filename

        print(f"\n[{n_vertices - args.min_vertices + 1}/{args.max_vertices - args.min_vertices + 1}] Vertices = {n_vertices}")

        if download_with_retry(url, output_path):
            successful.append(n_vertices)
        else:
            failed.append(n_vertices)
            print(f"  FAILED to download vertices_{n_vertices}")

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successful: {len(successful)} files")
    print(f"Failed: {len(failed)} files")

    if failed:
        print(f"Failed vertex counts: {failed}")
        print("Re-run the script to retry failed downloads.")

    # Convert if requested
    if args.convert and successful:
        output_json = args.output_dir.parent / "polytopes_full.json"
        convert_parquet_to_json(args.output_dir, output_json, args.max_polytopes)


if __name__ == "__main__":
    main()
