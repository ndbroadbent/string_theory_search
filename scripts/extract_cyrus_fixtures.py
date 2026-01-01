#!/usr/bin/env python3
"""
Extract test fixtures for cyrus from McAllister's data.

This script reads McAllister's paper data and CYTools computations, then
dumps JSON fixtures that can be loaded by Rust tests.

Fixtures include:
- polytope.json: primal/dual points
- triangulation.json: heights, simplices
- flux.json: K, M, c_i, basis, kklt_basis
- intersection.json: κ_ijk (computed from CYTools)
- gv_invariants.json: N_q (precomputed GV invariants)
- expected.json: g_s, W0, V_string, V0 (validation targets)

Usage:
    cd string_theory
    uv run python scripts/extract_cyrus_fixtures.py
"""

import json
import sys
from decimal import Decimal
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
OUTPUT_DIR = Path("/Users/ndbroadbent/code/cyrus/crates/cyrus-core/tests/fixtures")

MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4),
    ("5-113-4627-main", 113, 5),
    ("5-113-4627-alternative", 113, 5),
    ("5-81-3213", 81, 5),
    ("7-51-13590", 51, 7),
]


def load_vector(path: Path, dtype=int) -> list:
    """Load comma-separated vector from file."""
    text = path.read_text().strip()
    return [dtype(x) for x in text.split(',')]


def load_matrix(path: Path, dtype=int) -> list:
    """Load comma-separated matrix from file (one row per line)."""
    lines = path.read_text().strip().split('\n')
    return [[dtype(x) for x in line.split(',')] for line in lines]


def load_simplices(path: Path) -> list:
    """Load triangulation simplices."""
    lines = path.read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def compute_kappa_with_cytools(dual_points: list, simplices: list) -> dict:
    """
    Compute intersection numbers using CYTools 2021.

    Returns dict mapping (i,j,k) tuple (canonical: i<=j<=k) to value.
    """
    # Clear cached modules
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_2021))
    from cytools import Polytope
    import numpy as np

    poly = Polytope(np.array(dual_points))
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    divisor_basis = list(cy.divisor_basis())

    kappa_dict = {}
    for row in kappa_sparse:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        val = int(row[3])
        # Store in canonical order (i <= j <= k)
        key = tuple(sorted([i, j, k]))
        kappa_dict[f"{key[0]},{key[1]},{key[2]}"] = val

    sys.path.remove(str(CYTOOLS_2021))

    return {
        "dim": h11,
        "divisor_basis": divisor_basis,
        "entries": kappa_dict
    }


def extract_example(example_name: str, h11_primal: int, h21_primal: int) -> dict:
    """Extract all fixture data for one McAllister example."""
    print(f"Extracting {example_name}...")
    data_dir = DATA_BASE / example_name

    # Polytope data
    primal_points = load_matrix(data_dir / "points.dat")
    dual_points = load_matrix(data_dir / "dual_points.dat")

    # Triangulation
    dual_simplices = load_simplices(data_dir / "dual_simplices.dat")
    primal_heights = load_vector(data_dir / "corrected_heights.dat", float)

    # Flux vectors
    K = load_vector(data_dir / "K_vec.dat")
    M = load_vector(data_dir / "M_vec.dat")
    c_i = load_vector(data_dir / "target_volumes.dat")
    basis = load_vector(data_dir / "basis.dat")
    kklt_basis = load_vector(data_dir / "kklt_basis.dat")

    # McAllister's published physics values (from paper data files - these are INPUTS)
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    V_string = float((data_dir / "cy_vol.dat").read_text().strip())
    c_tau = float((data_dir / "c_tau.dat").read_text().strip())

    # Kähler params (corrected) - McAllister's solved values
    t_corrected = load_vector(data_dir / "corrected_kahler_param.dat", float)

    # Load GV invariants from McAllister's precomputed data
    # These are in "ambient" coordinates
    gv_curves = load_matrix(data_dir / "dual_curves.dat")
    gv_values_raw = (data_dir / "dual_curves_gv.dat").read_text().strip().split(",")
    gv_values = [int(Decimal(x)) for x in gv_values_raw]

    # Compute intersection numbers using CYTools
    print(f"  Computing κ_ijk...")
    kappa_data = compute_kappa_with_cytools(dual_points, dual_simplices)

    h21 = len(K)
    print(f"  h21={h21}, h11_primal={h11_primal}")
    print(f"  g_s = {g_s:.6f}, W0 = {W0:.2e}")
    print(f"  V_string = {V_string:.4f}")

    return {
        "example_name": example_name,
        "h11_primal": h11_primal,
        "h21_primal": h21_primal,
        "polytope": {
            "primal_points": primal_points,
            "dual_points": dual_points,
        },
        "triangulation": {
            "dual_simplices": dual_simplices,
            "primal_heights": primal_heights,
        },
        "flux": {
            "K": K,
            "M": M,
            "c_i": c_i,
            "basis": basis,
            "kklt_basis": kklt_basis,
        },
        "intersection": kappa_data,
        "gv_invariants": {
            "curves": gv_curves,
            "values": gv_values,
        },
        "racetrack": {
            "c_tau": c_tau,
        },
        "kahler_moduli": {
            "t_corrected": t_corrected,
        },
        # McAllister's published results (from paper data files)
        # These are the ground truth we validate against
        "mcallister_values": {
            "g_s": g_s,
            "W0": W0,
            "V_string": V_string,
        }
    }


def convert_numpy_to_python(obj):
    """Recursively convert numpy types to native Python types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(v) for v in obj)
    return obj


def write_fixtures(data: dict, output_dir: Path):
    """Write fixture files for one example."""
    example_dir = output_dir / data["example_name"].replace("-", "_")
    example_dir.mkdir(parents=True, exist_ok=True)

    # Write individual fixture files (INPUTS ONLY - no computed values)
    files = {
        "polytope.json": data["polytope"],
        "triangulation.json": data["triangulation"],
        "flux.json": data["flux"],
        "intersection.json": data["intersection"],
        "gv_invariants.json": data["gv_invariants"],
        "racetrack.json": data["racetrack"],
        "kahler_moduli.json": data["kahler_moduli"],
        # McAllister's published physics values (from paper data files)
        "mcallister_values.json": data["mcallister_values"],
    }

    for filename, content in files.items():
        path = example_dir / filename
        with open(path, 'w') as f:
            json.dump(convert_numpy_to_python(content), f, indent=2)
        print(f"  Wrote {path}")


def main():
    print("=" * 70)
    print("EXTRACTING CYRUS TEST FIXTURES FROM MCALLISTER DATA")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for example_name, h11, h21 in MCALLISTER_EXAMPLES:
        data = extract_example(example_name, h11, h21)
        write_fixtures(data, OUTPUT_DIR)
        print()

    print("=" * 70)
    print(f"Fixtures written to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
