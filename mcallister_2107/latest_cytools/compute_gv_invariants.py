#!/usr/bin/env python3
"""
Compute Gopakumar-Vafa invariants for a Calabi-Yau threefold.

GV invariants N_q count BPS states and appear in:
1. Worldsheet instanton corrections to the prepotential
2. The KKLT target τ formula (eq 5.13)

Uses CYTools' compute_gvs() which computes genus-zero GV invariants
via mirror symmetry.

Reference: arXiv:2107.09064 section 5.3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope


def compute_gv_invariants(cy, min_points: int = 100) -> dict:
    """
    Compute genus-zero Gopakumar-Vafa invariants.

    Args:
        cy: CYTools CalabiYau object
        min_points: Minimum number of lattice points to compute (controls degree)

    Returns:
        Dictionary mapping curve class tuples to GV invariants {(q1,q2,...): N_q}
    """
    gv_obj = cy.compute_gvs(min_points=min_points)
    gv_invariants = {}
    for q, N_q in gv_obj.dok.items():
        if N_q != 0:
            gv_invariants[tuple(q)] = int(N_q)
    return gv_invariants


# =============================================================================
# VALIDATION - ALL 5 MCALLISTER EXAMPLES
# =============================================================================

DATA_BASE = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data"

MCALLISTER_EXAMPLES = [
    ("4-214-647", 4),
    ("5-113-4627-main", 5),
    ("5-113-4627-alternative", 5),
    ("5-81-3213", 5),
    ("7-51-13590", 7),
]


def load_mcallister_gv(example_name: str) -> dict:
    """
    Load McAllister's GV data for an example.

    Returns:
        Dict mapping ambient coordinate curves to GV values
    """
    data_dir = DATA_BASE / example_name

    # Load curve classes (in ambient space coords: canonical + prime toric divisors)
    curves = []
    with open(data_dir / "dual_curves.dat") as f:
        for line in f:
            row = tuple(int(x) for x in line.strip().split(","))
            curves.append(row)

    # Load GV values
    with open(data_dir / "dual_curves_gv.dat") as f:
        content = f.read()
        gv_values = [int(float(x)) for x in content.strip().split(",")]

    return {c: g for c, g in zip(curves, gv_values)}


def load_simplices(example_name: str) -> list:
    """Load McAllister's triangulation simplices."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def test_example(example_name: str, expected_h11: int, verbose: bool = True) -> dict:
    """
    Test GV invariant computation for one McAllister example.

    Uses coordinate transformation to convert computed (basis) to ambient coords,
    then verifies EXACT match against McAllister's values.

    Args:
        example_name: Folder name in paper_data/
        expected_h11: Expected h11 for the DUAL polytope
        verbose: Print progress

    Returns:
        Dict with test results
    """
    data_dir = DATA_BASE / example_name

    if verbose:
        print("=" * 70)
        print(f"GV TEST - {example_name} (h11={expected_h11})")
        print("=" * 70)

    # Load McAllister's GV data
    mcallister_gv = load_mcallister_gv(example_name)
    if verbose:
        print(f"\nMcAllister GV data: {len(mcallister_gv)} curves")

    # Load dual polytope with McAllister's triangulation
    dual_pts = np.loadtxt(data_dir / "dual_points.dat", delimiter=',').astype(int)
    simplices = load_simplices(example_name)

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    if verbose:
        print(f"CY: h11={cy.h11()}, h21={cy.h21()}")

    h11_match = cy.h11() == expected_h11
    if not h11_match:
        return {"success": False, "error": f"h11 mismatch: {cy.h11()} != {expected_h11}"}

    # Get curve basis matrix for coordinate conversion
    # Rows are basis curves in ambient coords (h11 x (h11+5))
    curve_basis_mat = cy.curve_basis(include_origin=True, as_matrix=True)
    if verbose:
        print(f"Curve basis matrix: {curve_basis_mat.shape}")

    # Compute GV invariants with CYTools (in basis coords)
    gv_computed_basis = compute_gv_invariants(cy, min_points=100)
    if verbose:
        print(f"Computed: {len(gv_computed_basis)} GV invariants")

    # Convert to ambient coords and verify against McAllister
    # q_ambient = q_basis @ curve_basis_mat
    matches = 0
    mismatches = 0
    not_found = 0

    for q_basis, gv_computed in gv_computed_basis.items():
        q_ambient = tuple(int(x) for x in np.array(q_basis) @ curve_basis_mat)
        gv_expected = mcallister_gv.get(q_ambient)

        if gv_expected is None:
            not_found += 1
            if verbose and not_found <= 3:
                print(f"  WARNING: Curve {q_ambient} not in McAllister data")
        elif gv_computed == gv_expected:
            matches += 1
        else:
            mismatches += 1
            if verbose:
                print(f"  MISMATCH: {q_ambient}: computed={gv_computed}, expected={gv_expected}")

    if verbose:
        print(f"\nValidation: {matches} match, {mismatches} mismatch, {not_found} not found")

    # Test passes only if ALL computed curves match exactly
    test_passed = (mismatches == 0) and (not_found == 0) and (matches > 0)

    result = {
        "example_name": example_name,
        "h11": cy.h11(),
        "h11_match": h11_match,
        "n_gv_mcallister": len(mcallister_gv),
        "n_gv_computed": len(gv_computed_basis),
        "matches": matches,
        "mismatches": mismatches,
        "not_found": not_found,
        "test_passed": test_passed,
    }

    if verbose:
        status = "✓" if test_passed else "✗"
        print(f"\n{status} GV test {'PASSED' if test_passed else 'FAILED'}")

    return result


def main():
    """Test GV invariants for all 5 McAllister examples."""
    print("=" * 70)
    print("TESTING ALL 5 MCALLISTER EXAMPLES - GV Invariants")
    print("=" * 70)
    print("\nValidation: Every computed GV must EXACTLY match McAllister's value")
    print("McAllister wrote CYTools - with same triangulation, results must match.\n")

    results = []
    for name, h11 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY - GV Invariants")
    print("=" * 70)
    all_passed = True
    for result in results:
        status = "✓" if result["test_passed"] else "✗"
        print(f"  {status} {result['example_name']}: {result['matches']}/{result['n_gv_computed']} match (McAllister: {result['n_gv_mcallister']})")
        all_passed = all_passed and result["test_passed"]

    print()
    if all_passed:
        print("All 5 examples PASSED")
    else:
        print("Some examples FAILED")

    return results


if __name__ == "__main__":
    main()
