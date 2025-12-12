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
# MCALLISTER TEST - ALL 5 EXAMPLES
# =============================================================================

DATA_BASE = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data"

# All 5 McAllister examples with expected Hodge numbers
# Format: (name, h11_dual, h21_dual) - dual because that's where we work with fluxes
MCALLISTER_EXAMPLES = [
    ("4-214-647", 4, 214),
    ("5-113-4627-main", 5, 113),
    ("5-113-4627-alternative", 5, 113),
    ("5-81-3213", 5, 81),
    ("7-51-13590", 7, 51),
]


def load_example_points(example_name: str, which: str = "dual") -> np.ndarray:
    """
    Load polytope points from McAllister example.

    Args:
        example_name: One of the MCALLISTER_EXAMPLES names
        which: "dual" or "primal"

    Returns:
        (n_points, 4) array of lattice points
    """
    data_dir = DATA_BASE / example_name
    filename = "dual_points.dat" if which == "dual" else "points.dat"
    lines = (data_dir / filename).read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_example_simplices(example_name: str) -> set:
    """Load McAllister's triangulation simplices (dual only)."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_simplices.dat").read_text().strip().split('\n')
    return set(tuple(sorted(int(x) for x in line.split(','))) for line in lines)


def load_example_model_choices(example_name: str) -> dict:
    """
    Load model choices (Steps 3-4) for a McAllister example.

    Returns:
        Dict with K, M (fluxes), basis_2021 (CYTools 2021 basis), c_i (orientifold)
    """
    data_dir = DATA_BASE / example_name

    # Flux vectors (Step 4)
    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(',')])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(',')])

    # CYTools 2021 basis (needed for flux transformation)
    basis_2021 = [int(x) for x in (data_dir / "basis.dat").read_text().strip().split(',')]

    # Orientifold c_i values (Step 3)
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])

    # Expected physics values (for validation later)
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    V_string = float((data_dir / "cy_vol.dat").read_text().strip())

    return {
        "K": K,
        "M": M,
        "basis_2021": basis_2021,
        "c_i": c_i,
        "g_s": g_s,
        "W0": W0,
        "V_string": V_string,
    }


def test_example(example_name: str, expected_h11: int, expected_h21: int, verbose: bool = True) -> dict:
    """
    Test triangulation and model choice loading for one McAllister example.

    Args:
        example_name: Folder name in paper_data/
        expected_h11: Expected h11 for the DUAL polytope
        expected_h21: Expected h21 for the DUAL polytope
        verbose: Print progress

    Returns:
        Dict with results and pass/fail status
    """
    from compute_basis_transform import load_mcallister_example, test_example as test_basis_transform

    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (dual h11={expected_h11})")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1-2: Load polytope and compute triangulation
    # -------------------------------------------------------------------------
    points = load_example_points(example_name, which="dual")
    if verbose:
        print(f"\n[Step 1] Loaded {points.shape[0]} dual points")

    if verbose:
        print("\n[Step 2] Computing triangulation (CYTools)...")
    result = compute_triangulation(points, verbose=verbose)

    # Verify Hodge numbers match expected
    h11_match = result["h11"] == expected_h11
    h21_match = result["h21"] == expected_h21

    if verbose:
        print(f"\n  Hodge number verification:")
        print(f"    h11: computed={result['h11']}, expected={expected_h11} {'✓' if h11_match else '✗'}")
        print(f"    h21: computed={result['h21']}, expected={expected_h21} {'✓' if h21_match else '✗'}")

    # Compare triangulation with McAllister's
    mcallister_simplices = load_example_simplices(example_name)
    computed_simplices = set(tuple(sorted(s)) for s in result["tri"].simplices())
    hamming = len(mcallister_simplices ^ computed_simplices)

    if verbose:
        print(f"\n  Triangulation comparison:")
        print(f"    McAllister simplices: {len(mcallister_simplices)}")
        print(f"    Computed simplices: {len(computed_simplices)}")
        if hamming == 0:
            print("    *** TRIANGULATIONS IDENTICAL ***")
        else:
            print(f"    Hamming distance: {hamming} (OK - different FRST)")

    # -------------------------------------------------------------------------
    # Step 3-4: Load model choices and transform fluxes (via compute_basis_transform)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 3-4] Loading model choices and transforming fluxes...")

    # Use compute_basis_transform which handles CYTools 2021 basis correctly
    example_data = load_mcallister_example(example_name)
    basis_result = test_basis_transform(example_data, verbose=verbose)

    model = load_example_model_choices(example_name)

    if verbose:
        print(f"\n  c_i: {len(model['c_i'])} values, {np.sum(model['c_i']==6)} O7, {np.sum(model['c_i']==1)} D3")
        print(f"\n  Expected physics (for later validation):")
        print(f"    g_s = {model['g_s']:.6f}")
        print(f"    W₀ = {model['W0']:.2e}")
        print(f"    V_string = {model['V_string']:.2f}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    result["example_name"] = example_name
    result["h11_match"] = h11_match
    result["h21_match"] = h21_match
    result["triangulation_identical"] = hamming == 0
    result["model"] = model
    result["basis_transform_ok"] = basis_result
    result["test_passed"] = result["all_passed"] and h11_match and h21_match and basis_result

    return result


def main():
    """Test all 5 McAllister examples - Steps 1-4."""
    print("=" * 70)
    print("TESTING ALL 5 MCALLISTER EXAMPLES - Steps 1-4")
    print("  Step 1: Polytope loading")
    print("  Step 2: Triangulation")
    print("  Step 3: Orientifold (c_i values)")
    print("  Step 4: Flux vectors K, M (with basis transformation)")
    print("=" * 70)

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, h21, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY - Steps 1-4")
    print("=" * 70)
    all_passed = True
    for result in results:
        status = "✓" if result["test_passed"] else "✗"
        tri_note = "tri=same" if result["triangulation_identical"] else "tri=diff"
        basis_note = "basis ✓" if result["basis_transform_ok"] else "basis ✗"
        print(f"  {status} {result['example_name']}: h11={result['h11']} ({tri_note}, {basis_note})")
        all_passed = all_passed and result["test_passed"]

    print()
    if all_passed:
        print("All 5 examples PASSED Steps 1-4")
    else:
        print("Some examples FAILED")

    return results


if __name__ == "__main__":
    main()
