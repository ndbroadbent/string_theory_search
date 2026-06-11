#!/usr/bin/env python3
"""
Landscape smoke test: run CYTools and Cyrus on identical inputs across a
sample of Kreuzer-Skarke polytopes and demand zero silent divergence.

For each polytope:
  - CYTools picks its default triangulation; its heights fix the chamber.
  - Cyrus (landscape_smoke bin) computes the dual polytope, the FRST from
    those same heights, its divisor basis, and exact intersection numbers.
  - We compare: dual not-facet points, simplices, intersection numbers
    (CYTools re-expressed in Cyrus's basis), and h11 (incl. non-favorable
    Batyrev correction). Cyrus's own default-chamber simplices are compared
    to CYTools' default as a separate (non-fatal) chamber-agreement metric.

Outcomes per polytope: MATCH / MISMATCH (silent divergence: fatal finding) /
CYRUS_FAIL (loud failure) / CYTOOLS_FAIL / TIMEOUT.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import fetch_polytopes, config

config.enable_experimental_features()

CYRUS_BIN = Path("/Users/ndbroadbent/code/cyrus/target/release/landscape_smoke")


def compare_one(poly, timeout_s, workdir):
    record = {"h11": None, "h21": None, "favorable": None, "n_points": None}
    t0 = time.time()
    try:
        record["h11"] = int(poly.h11(lattice="N"))
        record["h21"] = int(poly.h21(lattice="N"))
        record["favorable"] = bool(poly.is_favorable(lattice="N"))
        pts = np.array(poly.points())
        record["n_points"] = len(pts)
        tri = poly.triangulate(verbosity=0)
        heights = np.array(tri.heights())
        cy = tri.get_cy()
        dual = poly.dual()
        dual_pts = sorted(
            tuple(int(x) for x in p) for p in dual.points_not_interior_to_facets()
        )
        ct_simplices = sorted(tuple(sorted(int(x) for x in s)) for s in tri.simplices())
    except Exception as e:
        record["outcome"] = "CYTOOLS_FAIL"
        record["detail"] = f"{type(e).__name__}: {str(e)[:200]}"
        record["seconds"] = round(time.time() - t0, 1)
        return record

    pts_path = workdir / "points.csv"
    heights_path = workdir / "heights.csv"
    out_path = workdir / "cyrus.json"
    np.savetxt(pts_path, pts, fmt="%d", delimiter=",")
    heights_path.write_text(",".join(repr(float(h)) for h in heights))
    try:
        proc = subprocess.run(
            [str(CYRUS_BIN), "--points", str(pts_path), "--heights", str(heights_path), "--out", str(out_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        record["outcome"] = "TIMEOUT"
        record["seconds"] = round(time.time() - t0, 1)
        return record
    if proc.returncode != 0:
        record["outcome"] = "CYRUS_FAIL"
        lines = proc.stderr.strip().split("\n")
        panic = [l for l in lines if "panicked" in l or "ERROR" in l]
        idx = lines.index(panic[0]) if panic else len(lines) - 1
        record["detail"] = " | ".join(lines[idx : idx + 2])[:300]
        record["seconds"] = round(time.time() - t0, 1)
        return record
    cyrus = json.loads(out_path.read_text())

    mismatches = []

    if sorted(tuple(p) for p in cyrus["dual_not_facet_points"]) != dual_pts:
        mismatches.append("dual_points")

    cy_simplices = sorted(tuple(s) for s in cyrus["simplices"])
    if cy_simplices != ct_simplices:
        mismatches.append(f"simplices({len(cy_simplices)} vs {len(ct_simplices)})")

    # h11 including the Batyrev correction for non-favorable polytopes
    if cyrus["h11_toric"] + cyrus["h11_extra"] != record["h11"]:
        mismatches.append(
            f"h11({cyrus['h11_toric']}+{cyrus['h11_extra']} vs {record['h11']})"
        )

    # Intersection numbers, CYTools re-expressed in Cyrus's basis
    try:
        cy.set_divisor_basis(list(cyrus["basis"]))
        kappa_ct = cy.intersection_numbers(in_basis=True, format="dok")
        ct_map = {}
        for key, value in kappa_ct.items():
            ct_map[tuple(sorted(int(x) for x in key))] = float(value)
        cyrus_map = {
            tuple(int(x) for x in key.split(",")): float(Fraction(value))
            for key, value in cyrus["kappa_basis"].items()
        }
        keys = set(ct_map) | set(cyrus_map)
        bad = [
            key
            for key in keys
            if abs(ct_map.get(key, 0.0) - cyrus_map.get(key, 0.0)) > 1e-6
        ]
        if bad:
            mismatches.append(f"kappa({len(bad)} of {len(keys)} entries)")
    except Exception as e:
        record["kappa_note"] = f"{type(e).__name__}: {str(e)[:120]}"

    record["chamber_agrees"] = (
        sorted(tuple(s) for s in cyrus["default_chamber_simplices"]) == ct_simplices
    )
    record["outcome"] = "MISMATCH" if mismatches else "MATCH"
    if mismatches:
        record["detail"] = ";".join(mismatches)
    record["seconds"] = round(time.time() - t0, 1)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="3:12,5:12,8:12,12:12,20:12,35:10,60:8,100:6")
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--out", default="landscape_smoke/results.jsonl")
    ap.add_argument("--fetch-factor", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    results = []
    with out_path.open("w") as out:
        for tier in args.tiers.split(","):
            h11, count = (int(x) for x in tier.split(":"))
            try:
                pool = list(
                    fetch_polytopes(h11=h11, lattice="N", limit=count * args.fetch_factor)
                )
            except Exception as e:
                print(f"[tier h11={h11}] fetch failed: {e}", flush=True)
                continue
            rng.shuffle(pool)
            sample = pool[:count]
            timeout_s = 120 + 8 * h11
            print(f"[tier h11={h11}] {len(sample)} polytopes (timeout {timeout_s}s)", flush=True)
            for idx, poly in enumerate(sample):
                with tempfile.TemporaryDirectory() as tmp:
                    record = compare_one(poly, timeout_s, Path(tmp))
                record["tier_h11"] = h11
                record["index"] = idx
                results.append(record)
                out.write(json.dumps(record) + "\n")
                out.flush()
                marker = {"MATCH": ".", "MISMATCH": "X", "CYRUS_FAIL": "F", "CYTOOLS_FAIL": "c", "TIMEOUT": "T"}[record["outcome"]]
                print(f"  [{h11}:{idx}] {record['outcome']} ({record['seconds']}s)" + (f" {record.get('detail','')}" if marker != "." else ""), flush=True)

    outcomes = {}
    chamber = [r.get("chamber_agrees") for r in results if "chamber_agrees" in r]
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print("\n=== SUMMARY ===", flush=True)
    print(f"total={len(results)} outcomes={outcomes}", flush=True)
    if chamber:
        print(f"default-chamber agreement: {sum(chamber)}/{len(chamber)}", flush=True)
    nonfav = [r for r in results if r.get("favorable") is False]
    print(f"non-favorable in sample: {len(nonfav)}", flush=True)


if __name__ == "__main__":
    main()
