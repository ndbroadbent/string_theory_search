#!/usr/bin/env python3
"""
Fetch Kreuzer-Skarke polytopes for the cyrus-ga multi-polytope landscape
search and write them as JSONL ({"name": ..., "points": [[...], ...]}).

The PFV/racetrack construction wants SMALL h21 (few flux integers, small
mirror), like the published McAllister examples (h21 = 4..7), so we fetch
by h21 slices. Points are the full CYTools-ordered lattice point list, the
same convention as the McAllister points.dat files.

Usage:
  uv run python landscape_smoke/fetch_ga_polytopes.py \
      --h21 4,5,6,7 --per-h21 40 --out /path/to/polytopes.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import fetch_polytopes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h21", default="4,5,6,7")
    ap.add_argument("--per-h21", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w") as out:
        for h21 in (int(x) for x in args.h21.split(",")):
            try:
                pool = list(fetch_polytopes(h21=h21, lattice="N", limit=args.per_h21))
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"[h21={h21}] fetch failed: {e}", flush=True)
                continue
            for idx, poly in enumerate(pool):
                points = np.array(poly.points()).tolist()
                record = {
                    "name": f"h21_{h21}_{idx}",
                    "h11": int(poly.h11(lattice="N")),
                    "h21": int(poly.h21(lattice="N")),
                    "favorable": bool(poly.is_favorable(lattice="N")),
                    "points": points,
                }
                out.write(json.dumps(record) + "\n")
                n_written += 1
            print(f"[h21={h21}] wrote {len(pool)} polytopes", flush=True)
    print(f"total: {n_written} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
