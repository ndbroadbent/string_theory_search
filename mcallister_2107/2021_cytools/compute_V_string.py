#!/usr/bin/env python3
"""
Compute string frame CY volume V_string using CYTools 2021.

V_string = (1/6) κ_ijk t^i t^j t^k - BBHL_correction

Where:
- κ_ijk: Intersection numbers from CYTools
- t^i: Kähler moduli (from KKLT or loaded from data)
- BBHL = ζ(3) χ(X) / (4 (2π)³)

IMPORTANT: For validation, we use the PRIMAL polytope (points.dat) because
cy_vol.dat contains V_string for the primal CY (h11=214 for 4-214-647).
The dual polytope (dual_points.dat) gives the mirror CY with different h11.

Reference: arXiv:2107.09064 (McAllister et al.)
"""

import sys
from pathlib import Path

# Use CYTools 2021 ONLY
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
sys.path.insert(0, str(CYTOOLS_2021))

import numpy as np
from scipy.special import zeta
from cytools import Polytope

DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

# Examples with (name, h11_primal, h21_primal)
# Note: primal has LARGE h11, dual/mirror has small h11
MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4),
    ("5-113-4627-main", 113, 5),
    ("5-113-4627-alternative", 113, 5),
    ("5-81-3213", 81, 5),
    ("7-51-13590", 51, 7),
]


# =============================================================================
# PURE COMPUTATION FUNCTIONS (work with any CY)
# =============================================================================


def compute_intersection_tensor(cy) -> np.ndarray:
    """
    Compute full symmetric intersection tensor κ_ijk.

    Handles both CYTools 2021 (array format) and latest (dict format).

    Args:
        cy: CYTools CalabiYau object

    Returns:
        (h11, h11, h11) symmetric tensor
    """
    h11 = cy.h11()
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))

    # Handle both dict (latest) and array (2021) formats
    if hasattr(kappa_sparse, 'items'):
        # Dict format: {(i,j,k): val, ...}
        for (i, j, k), val in kappa_sparse.items():
            for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                kappa[perm] = val
    else:
        # Array format: [[i, j, k, val], ...]
        for row in kappa_sparse:
            i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
            for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                kappa[perm] = val

    return kappa


def compute_cy_volume_classical(kappa: np.ndarray, t: np.ndarray) -> float:
    """
    Compute classical CY volume V = (1/6) κ_ijk t^i t^j t^k.

    This is the string frame volume WITHOUT BBHL correction.
    """
    return np.einsum("ijk,i,j,k->", kappa, t, t, t) / 6.0


def compute_bbhl_correction(h11: int, h21: int) -> float:
    """
    Compute BBHL α' correction: ζ(3) χ(X) / (4 (2π)³).

    Where χ(X) = 2(h¹¹ - h²¹) is the Euler characteristic.
    """
    chi = 2 * (h11 - h21)
    return zeta(3) * chi / (4 * (2 * np.pi) ** 3)


def compute_V_string(kappa: np.ndarray, t: np.ndarray, h11: int, h21: int) -> dict:
    """
    Compute string frame volume V_string = V_classical - BBHL.

    Args:
        kappa: Intersection tensor (h11, h11, h11)
        t: Kähler moduli (h11,)
        h11, h21: Hodge numbers

    Returns:
        Dict with V_classical, BBHL, V_string
    """
    V_classical = compute_cy_volume_classical(kappa, t)
    BBHL = compute_bbhl_correction(h11, h21)
    V_string = V_classical - BBHL

    return {
        "V_classical": V_classical,
        "BBHL": BBHL,
        "V_string": V_string,
    }


# =============================================================================
# DATA LOADING FOR TESTS
# =============================================================================


def load_primal_points(example_name: str) -> np.ndarray:
    """Load primal polytope points (points.dat)."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_heights(example_name: str, corrected: bool = True) -> np.ndarray:
    """Load triangulation heights."""
    data_dir = DATA_BASE / example_name
    filename = "corrected_heights.dat" if corrected else "heights.dat"
    text = (data_dir / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_kahler_params(example_name: str, corrected: bool = True) -> np.ndarray:
    """Load Kähler parameters (solved t values)."""
    data_dir = DATA_BASE / example_name
    filename = "corrected_kahler_param.dat" if corrected else "kahler_param.dat"
    text = (data_dir / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_basis(example_name: str) -> list:
    """Load divisor basis indices."""
    data_dir = DATA_BASE / example_name
    text = (data_dir / "basis.dat").read_text().strip()
    return [int(x) for x in text.split(',')]


def load_expected_V_string(example_name: str, corrected: bool = True) -> float:
    """Load expected V_string from McAllister's data."""
    data_dir = DATA_BASE / example_name
    filename = "corrected_cy_vol.dat" if corrected else "cy_vol.dat"
    return float((data_dir / filename).read_text().strip())


# =============================================================================
# VALIDATION AGAINST MCALLISTER DATA
# =============================================================================


def test_example(example_name: str, expected_h11: int, expected_h21: int, verbose: bool = True) -> dict:
    """
    Test V_string computation against one McAllister example.

    Uses the PRIMAL polytope (points.dat) with McAllister's solved Kähler params.

    Args:
        example_name: Name of the example (e.g., "4-214-647")
        expected_h11, expected_h21: Expected Hodge numbers (for primal)
        verbose: Print progress

    Returns:
        Dict with test results
    """
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (primal h11={expected_h11})")
        print("=" * 70)

    # Load PRIMAL polytope points
    points = load_primal_points(example_name)
    if verbose:
        print(f"\nLoaded primal polytope: {points.shape[0]} points")

    # Load triangulation heights
    heights = load_heights(example_name, corrected=True)
    if verbose:
        print(f"Loaded heights: {len(heights)} values")

    # Build CY with McAllister's triangulation
    poly = Polytope(points)
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()
    h11, h21 = cy.h11(), cy.h21()

    if verbose:
        print(f"CY: h11={h11}, h21={h21}")

    # Verify Hodge numbers
    if h11 != expected_h11 or h21 != expected_h21:
        if verbose:
            print(f"  WARNING: Expected h11={expected_h11}, h21={expected_h21}")

    # Set McAllister's divisor basis
    basis = load_basis(example_name)
    cy.set_divisor_basis(basis)
    if verbose:
        print(f"Set divisor basis: {len(basis)} divisors")

    # Get intersection tensor
    kappa = compute_intersection_tensor(cy)
    if verbose:
        nonzero = np.sum(kappa != 0)
        print(f"Intersection tensor: {nonzero} non-zero entries")

    # Load McAllister's SOLVED Kähler moduli
    t = load_kahler_params(example_name, corrected=True)
    if verbose:
        print(f"Loaded Kähler params: {len(t)} values")
        print(f"  t range: [{t.min():.4f}, {t.max():.4f}]")

    # Compute V_string
    result = compute_V_string(kappa, t, h11, h21)
    V_computed = result["V_string"]

    # Load expected
    V_expected = load_expected_V_string(example_name, corrected=True)

    if verbose:
        print(f"\nResults:")
        print(f"  V_classical = {result['V_classical']:.6f}")
        print(f"  BBHL = {result['BBHL']:.6f} (χ = {2*(h11-h21)})")
        print(f"  V_string = {V_computed:.6f}")
        print(f"  V_expected = {V_expected:.6f}")

    # Compute error
    rel_error = abs(V_computed - V_expected) / V_expected
    ratio = V_computed / V_expected

    if verbose:
        print(f"\nValidation:")
        print(f"  Ratio computed/expected = {ratio:.8f}")
        print(f"  Relative error = {rel_error:.2e}")

    # Pass criteria: < 0.2% error (small floating point differences expected)
    passed = rel_error < 2e-3

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"\n{status}: {example_name} (error = {rel_error:.2e})")

    return {
        "example_name": example_name,
        "h11": h11,
        "h21": h21,
        "V_computed": V_computed,
        "V_expected": V_expected,
        "ratio": ratio,
        "rel_error": rel_error,
        "passed": passed,
    }


def main():
    """Test V_string computation against all 5 McAllister examples."""
    print("=" * 70)
    print("V_STRING COMPUTATION - ALL 5 MCALLISTER EXAMPLES (CYTools 2021)")
    print("Using PRIMAL polytope (points.dat) with corrected Kähler params")
    print("=" * 70)

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, h21, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}: {r['example_name']:30s} V={r['V_computed']:10.4f} (expected {r['V_expected']:.4f}, error={r['rel_error']:.2e})")
        all_passed = all_passed and r["passed"]

    print()
    if all_passed:
        print("All 5 examples PASSED")
    else:
        n_passed = sum(1 for r in results if r["passed"])
        print(f"{n_passed}/5 examples passed")

    return results


if __name__ == "__main__":
    main()
