#!/usr/bin/env python3
"""
Compute Gopakumar-Vafa invariants for a Calabi-Yau threefold.

GV invariants N_q count BPS states and appear in:
1. Worldsheet instanton corrections to the prepotential
2. The KKLT target τ formula (eq 5.13)

Uses CYTools' compute_gvs() which computes genus-zero GV invariants
via mirror symmetry.

Validation: Tests against all 5 McAllister examples (arXiv:2107.09064).
Strategy for validation:
- Use CYTools 2021 for geometry (polytope, triangulation, basis)
- Use CYTools latest for GV computation (compute_gvs was added later)

Reference: arXiv:2107.09064 section 5.3
"""

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

# Paths
CYTOOLS_2021 = Path(__file__).parent.parent.parent / "vendor/cytools_mcallister_2107"
CYTOOLS_LATEST = Path(__file__).parent.parent.parent / "vendor/cytools_latest/src"
DATA_BASE = Path(__file__).parent.parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data"

MCALLISTER_EXAMPLES = [
    ("4-214-647", 4, 214),
    ("5-113-4627-main", 5, 113),
    ("5-113-4627-alternative", 5, 113),
    ("5-81-3213", 5, 81),
    ("7-51-13590", 7, 51),
]


def load_dual_points(example_name: str) -> np.ndarray:
    """Load dual polytope points."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_simplices(example_name: str) -> list:
    """Load McAllister's triangulation simplices."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def load_mcallister_gv(example_name: str) -> dict:
    """Load McAllister's pre-computed GV data (ambient coordinates)."""
    data_dir = DATA_BASE / example_name

    curves = []
    with open(data_dir / "dual_curves.dat") as f:
        for line in f:
            row = tuple(int(x) for x in line.strip().split(","))
            curves.append(row)

    with open(data_dir / "dual_curves_gv.dat") as f:
        content = f.read()
        # Use Decimal for exact integer conversion from scientific notation
        gv_values = [int(Decimal(x)) for x in content.strip().split(",")]

    return {c: g for c, g in zip(curves, gv_values)}


def build_cy_2021(example_name: str):
    """Build CY using CYTools 2021 with McAllister's triangulation."""
    # Clear any existing cytools
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    # Use CYTools 2021
    sys.path.insert(0, str(CYTOOLS_2021))
    from cytools import Polytope

    dual_pts = load_dual_points(example_name)
    simplices = load_simplices(example_name)

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    basis = list(cy.divisor_basis())

    # Clean up
    sys.path.remove(str(CYTOOLS_2021))

    return cy, basis, dual_pts, simplices


def get_curve_basis_mat(dual_pts: np.ndarray, simplices: list) -> np.ndarray:
    """Get curve basis matrix from CYTools latest."""
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()
    curve_basis_mat = cy.curve_basis(include_origin=True, as_matrix=True)

    sys.path.remove(str(CYTOOLS_LATEST))
    return curve_basis_mat


def compute_gv_with_latest(dual_pts: np.ndarray, simplices: list, min_points: int = 10000) -> dict:
    """Compute GV invariants using CYTools latest."""
    # Get curve_basis_mat from latest CYTools
    curve_basis_mat = get_curve_basis_mat(dual_pts, simplices)

    # Clear any existing cytools
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    # Use CYTools latest for GV computation only
    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    # Compute GV
    gv_obj = cy.compute_gvs(min_points=min_points)

    # Convert to dict with ambient coordinates using 2021's curve_basis_mat
    # Use Decimal for exact integer conversion from floating point
    gv_ambient = {}
    for q_basis, N_q in gv_obj.dok.items():
        if N_q != 0:
            q_ambient = tuple(int(x) for x in np.array(q_basis) @ curve_basis_mat)
            gv_ambient[q_ambient] = int(Decimal(str(N_q)).to_integral_value())

    # Also return basis coords for later use
    gv_basis = {}
    for q_basis, N_q in gv_obj.dok.items():
        if N_q != 0:
            gv_basis[tuple(q_basis)] = int(Decimal(str(N_q)).to_integral_value())

    # Clean up
    sys.path.remove(str(CYTOOLS_LATEST))

    return gv_ambient, gv_basis, curve_basis_mat


def compute_gv_invariants(cy, min_points: int = 10000) -> dict:
    """
    Compute genus-zero Gopakumar-Vafa invariants.

    Args:
        cy: CYTools CalabiYau object (from latest CYTools)
        min_points: Lattice points to sample (10000 matches McAllister for h11=4)

    Returns:
        Dictionary mapping curve class tuples (in basis coords) to GV invariants
    """
    gv_obj = cy.compute_gvs(min_points=min_points)
    gv_invariants = {}
    for q, N_q in gv_obj.dok.items():
        if N_q != 0:
            # Use Decimal for exact integer conversion from float
            gv_invariants[tuple(q)] = int(Decimal(str(N_q)).to_integral_value())
    return gv_invariants


def test_example(example_name: str, expected_h11: int, expected_h21: int,
                 min_points: int = 10000, verbose: bool = True) -> dict:
    """Test GV computation for one McAllister example."""
    if verbose:
        print("=" * 70)
        print(f"GV TEST - {example_name} (h11={expected_h11}, min_points={min_points})")
        print("=" * 70)

    # Build CY with 2021 to verify geometry
    cy_2021, basis_2021, dual_pts, simplices = build_cy_2021(example_name)

    if verbose:
        print(f"CYTools 2021: h11={cy_2021.h11()}, h21={cy_2021.h21()}")
        print(f"Basis (2021): {basis_2021}")

    h11_match = cy_2021.h11() == expected_h11
    h21_match = cy_2021.h21() == expected_h21

    if not h11_match or not h21_match:
        return {"success": False, "error": f"Hodge mismatch: h11={cy_2021.h11()}, h21={cy_2021.h21()}"}

    # Load McAllister's GV data
    mcallister_gv = load_mcallister_gv(example_name)
    if verbose:
        print(f"McAllister GV data: {len(mcallister_gv)} curves")

    # Compute GV with latest (same geometry, different code)
    gv_ambient, gv_basis, curve_basis_mat = compute_gv_with_latest(dual_pts, simplices, min_points=min_points)

    if verbose:
        print(f"Computed GV: {len(gv_ambient)} curves")

    # Validate against ALL curves in McAllister's data
    matches = 0
    mismatches = 0
    missing = 0

    for q_mcallister, gv_expected in mcallister_gv.items():
        gv_computed = gv_ambient.get(q_mcallister)

        if gv_computed is None:
            missing += 1
            if verbose and missing <= 3:
                print(f"  MISSING: {q_mcallister} (expected GV={gv_expected})")
        elif gv_computed == gv_expected:
            matches += 1
        else:
            mismatches += 1
            if verbose and mismatches <= 3:
                print(f"  MISMATCH: {q_mcallister}: computed={gv_computed}, expected={gv_expected}")

    if verbose:
        print(f"\nValidation: {matches}/{len(mcallister_gv)} match, {mismatches} mismatch, {missing} missing")

    test_passed = (matches == len(mcallister_gv)) and (mismatches == 0) and (missing == 0)

    result = {
        "example_name": example_name,
        "h11": expected_h11,
        "h21": expected_h21,
        "basis_2021": basis_2021,
        "n_gv_mcallister": len(mcallister_gv),
        "n_gv_computed": len(gv_ambient),
        "matches": matches,
        "mismatches": mismatches,
        "missing": missing,
        "gv_basis": gv_basis,  # Return for use in W0 computation
        "test_passed": test_passed,
    }

    if verbose:
        status = "PASS" if test_passed else "FAIL"
        print(f"\n{status}: GV test {'passed' if test_passed else 'failed'}")

    return result


def main():
    """Test GV invariants for all 5 McAllister examples."""
    print("=" * 70)
    print("TESTING ALL 5 MCALLISTER EXAMPLES - GV Invariants")
    print("=" * 70)
    print("\nStrategy: CYTools 2021 for geometry, latest for compute_gvs()")
    print("Settings: min_points=20000 (needed for 5-113-4627), Decimal for exact conversion\n")

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        # 5-113-4627 needs min_points=20000 to find all curves (see CURVE_DISCREPANCY.md)
        # Other examples work fine with 10000 but 20000 doesn't hurt
        result = test_example(name, h11, h21, min_points=20000, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY - GV Invariants")
    print("=" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r["test_passed"] else "FAIL"
        print(f"  {status}: {r['example_name']} - {r['matches']}/{r['n_gv_mcallister']} match")
        all_passed = all_passed and r["test_passed"]

    print()
    if all_passed:
        print("All 5 examples PASSED")
    else:
        print("Some examples FAILED")

    return results


if __name__ == "__main__":
    main()
