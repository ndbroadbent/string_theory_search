#!/usr/bin/env python3
"""
Compute Gopakumar-Vafa invariants for a Calabi-Yau threefold.

GV invariants N_q count BPS states and appear in:
1. Worldsheet instanton corrections to the prepotential
2. The KKLT target tau formula (eq 5.13)

Uses CYTools latest's compute_gvs() (2021 version lacks this method).
min_points=10000 required to match McAllister's 5177 curves for 4-214-647.

Reference: arXiv:2107.09064 section 5.3
"""

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

# Paths
CYTOOLS_LATEST = Path(__file__).parent.parent.parent / "vendor/cytools_latest/src"
DATA_BASE = Path(__file__).parent.parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data"

MCALLISTER_EXAMPLES = [
    ("4-214-647", 4, 214),
    ("5-113-4627-main", 5, 113),
    ("5-113-4627-alternative", 5, 113),
    ("5-81-3213", 5, 81),
    ("7-51-13590", 7, 51),
]


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


def build_cy_latest(example_name: str):
    """Build CY using CYTools latest with McAllister's triangulation."""
    # Clear any existing cytools and use latest
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    dual_pts = load_dual_points(example_name)
    simplices = load_simplices(example_name)

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    # Clean up path
    if str(CYTOOLS_LATEST) in sys.path:
        sys.path.remove(str(CYTOOLS_LATEST))

    return cy, dual_pts


def test_example(example_name: str, expected_h11: int, expected_h21: int,
                 min_points: int = 10000, verbose: bool = True) -> dict:
    """Test GV computation for one McAllister example."""
    if verbose:
        print("=" * 70)
        print(f"GV TEST - {example_name} (h11={expected_h11}, min_points={min_points})")
        print("=" * 70)

    cy, dual_pts = build_cy_latest(example_name)

    if verbose:
        print(f"CY: h11={cy.h11()}, h21={cy.h21()}")

    h11_match = cy.h11() == expected_h11
    h21_match = cy.h21() == expected_h21

    if not h11_match or not h21_match:
        return {"success": False, "error": f"Hodge mismatch: h11={cy.h11()}, h21={cy.h21()}"}

    # Load McAllister's GV data
    mcallister_gv = load_mcallister_gv(example_name)
    if verbose:
        print(f"McAllister GV data: {len(mcallister_gv)} curves")

    # Get curve basis matrix for coordinate conversion
    curve_basis_mat = cy.curve_basis(include_origin=True, as_matrix=True)

    # Compute GV with CYTools latest
    gv_basis = compute_gv_invariants(cy, min_points=min_points)

    # Convert to ambient coords
    gv_ambient = {}
    for q_basis, N_q in gv_basis.items():
        q_ambient = tuple(int(x) for x in np.array(q_basis) @ curve_basis_mat)
        gv_ambient[q_ambient] = N_q

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
    print("\nSettings: min_points=10000, Decimal for exact integer conversion")
    print("Validation: Every computed GV must EXACTLY match McAllister's value\n")

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, h21, min_points=10000, verbose=True)
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
