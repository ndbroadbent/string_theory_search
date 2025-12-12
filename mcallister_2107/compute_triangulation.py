#!/usr/bin/env python3
"""
Compute triangulations from polytope points.

This is the foundation - given only lattice points, compute a valid triangulation
and verify it produces sensible CY geometry.

No .dat file shortcuts. CYTools computes everything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope


def compute_triangulation(points: np.ndarray, verbose: bool = True) -> dict:
    """
    Compute triangulation from polytope points.

    Args:
        points: (n_points, 4) array of lattice points
        verbose: Print progress

    Returns:
        Dict with poly, tri, cy objects and sanity check results
    """
    if verbose:
        print(f"Input points: {points.shape}")

    # Create polytope
    poly = Polytope(points)

    if verbose:
        print(f"Polytope created")
        print(f"  Is reflexive: {poly.is_reflexive()}")
        print(f"  Points (including origin): {poly.points().shape[0]}")

    # Compute triangulation - CYTools finds one automatically
    tri = poly.triangulate()

    if verbose:
        print(f"Triangulation computed")
        print(f"  Simplices: {len(tri.simplices())}")
        print(f"  Is regular: {tri.is_regular()}")
        print(f"  Is fine: {tri.is_fine()}")
        print(f"  Is star: {tri.is_star()}")

    # Get CY threefold
    cy = tri.get_cy()

    if verbose:
        print(f"CY threefold")
        print(f"  h11 = {cy.h11()}")
        print(f"  h21 = {cy.h21()}")
        print(f"  Divisor basis: {list(cy.divisor_basis())}")

    # Sanity checks
    checks = {}

    # Check 1: Hodge numbers are positive
    checks["h11_positive"] = cy.h11() > 0
    checks["h21_positive"] = cy.h21() >= 0

    # Check 2: GLSM matrix has correct shape
    glsm = poly.glsm_charge_matrix()
    checks["glsm_rows_eq_h11"] = glsm.shape[0] == cy.h11()
    checks["glsm_has_columns"] = glsm.shape[1] > 0

    # Check 3: Intersection numbers exist
    kappa = cy.intersection_numbers(in_basis=True)
    checks["has_intersection_numbers"] = len(kappa) > 0

    # Check 4: SR ideal exists
    sr = tri.sr_ideal()
    checks["has_sr_ideal"] = len(sr) > 0

    if verbose:
        print(f"Sanity checks:")
        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")

    all_passed = all(checks.values())

    return {
        "poly": poly,
        "tri": tri,
        "cy": cy,
        "checks": checks,
        "all_passed": all_passed,
        "h11": cy.h11(),
        "h21": cy.h21(),
        "glsm_shape": glsm.shape,
        "n_simplices": len(tri.simplices()),
        "n_sr_generators": len(sr),
    }


# =============================================================================
# MCALLISTER TEST
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_points(filename):
    """Load polytope points from .dat file."""
    lines = (DATA_DIR / filename).read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def test_dual():
    """Test triangulation on McAllister's DUAL polytope (h11=4)."""
    print("=" * 70)
    print("TRIANGULATION TEST - McAllister DUAL polytope (h11=4)")
    print("=" * 70)

    points = load_points("dual_points.dat")
    print(f"\n[1] Loaded {points.shape[0]} points")

    print("\n[2] Computing triangulation (CYTools)...")
    result = compute_triangulation(points, verbose=True)

    # Compare with McAllister's triangulation
    print("\n[3] Comparing with McAllister's triangulation...")
    lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
    mcallister_simplices = set(tuple(sorted(int(x) for x in line.split(','))) for line in lines)
    computed_simplices = set(tuple(sorted(s)) for s in result["tri"].simplices())

    hamming = len(mcallister_simplices ^ computed_simplices)
    print(f"    Hamming distance: {hamming}")

    if hamming == 0:
        print("    *** TRIANGULATIONS IDENTICAL ***")
    else:
        print(f"    *** MISMATCH: {hamming} differences ***")

    return result


def test_primal():
    """Test triangulation on McAllister's PRIMAL polytope (h11=214)."""
    print("\n" + "=" * 70)
    print("TRIANGULATION TEST - McAllister PRIMAL polytope (h11=214)")
    print("=" * 70)

    points = load_points("points.dat")
    print(f"\n[1] Loaded {points.shape[0]} points")

    print("\n[2] Computing triangulation (CYTools)...")
    result = compute_triangulation(points, verbose=True)

    # No primal simplices file to compare against
    print("\n[3] No primal simplices.dat for comparison (only dual exists)")

    return result


def main():
    """Test both dual and primal polytopes."""
    result_dual = test_dual()
    result_primal = test_primal()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dual (h11=4):    {'PASS' if result_dual['all_passed'] else 'FAIL'}")
    print(f"Primal (h11=214): {'PASS' if result_primal['all_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
