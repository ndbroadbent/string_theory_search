#!/usr/bin/env python3
"""
Compute triangulations from polytope points using CYTools 2021.

This version uses ONLY vendor/cytools_mcallister_2107 - the exact version
McAllister used. NO transformations. NO workarounds.
"""

import sys
from pathlib import Path

# Use CYTools 2021 ONLY
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor/cytools_mcallister_2107"))

import numpy as np
from cytools import Polytope


DATA_BASE = Path(__file__).parent.parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data"

MCALLISTER_EXAMPLES = [
    ("4-214-647", 4, 214),
    ("5-113-4627-main", 5, 113),
    ("5-113-4627-alternative", 5, 113),
    ("5-81-3213", 5, 81),
    ("7-51-13590", 7, 51),
]


def load_example_points(example_name: str, which: str = "dual") -> np.ndarray:
    """Load polytope points from McAllister example."""
    data_dir = DATA_BASE / example_name
    filename = "dual_points.dat" if which == "dual" else "points.dat"
    lines = (data_dir / filename).read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_example_simplices(example_name: str) -> list:
    """Load McAllister's triangulation simplices as list of lists."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def load_example_model_choices(example_name: str) -> dict:
    """
    Load model choices for a McAllister example.

    K and M are in CYTools 2021 basis - use directly, NO transformation needed.
    """
    data_dir = DATA_BASE / example_name

    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(',')])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(',')])
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    V_string = float((data_dir / "cy_vol.dat").read_text().strip())

    return {
        "K": K,
        "M": M,
        "c_i": c_i,
        "g_s": g_s,
        "W0": W0,
        "V_string": V_string,
    }


def build_cy(example_name: str, verbose: bool = True):
    """
    Build CY from McAllister's data using CYTools 2021.

    Uses McAllister's exact triangulation simplices.
    """
    dual_pts = load_example_points(example_name, which="dual")
    simplices = load_example_simplices(example_name)

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    if verbose:
        print(f"CY built: h11={cy.h11()}, h21={cy.h21()}")
        print(f"Divisor basis: {list(cy.divisor_basis())}")

    return poly, tri, cy


def test_example(example_name: str, expected_h11: int, expected_h21: int, verbose: bool = True) -> dict:
    """Test one McAllister example."""
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (h11={expected_h11})")
        print("=" * 70)

    poly, tri, cy = build_cy(example_name, verbose=verbose)
    model = load_example_model_choices(example_name)

    h11_match = cy.h11() == expected_h11
    h21_match = cy.h21() == expected_h21

    if verbose:
        print(f"\nHodge numbers: h11={cy.h11()} (expected {expected_h11}) {'OK' if h11_match else 'FAIL'}")
        print(f"              h21={cy.h21()} (expected {expected_h21}) {'OK' if h21_match else 'FAIL'}")
        print(f"\nK = {model['K']}")
        print(f"M = {model['M']}")
        print(f"g_s = {model['g_s']:.6f}")
        print(f"W0 = {model['W0']:.2e}")
        print(f"V_string = {model['V_string']:.2f}")

    return {
        "example_name": example_name,
        "poly": poly,
        "tri": tri,
        "cy": cy,
        "model": model,
        "h11_match": h11_match,
        "h21_match": h21_match,
        "test_passed": h11_match and h21_match,
    }


def main():
    """Test all 5 McAllister examples."""
    print("=" * 70)
    print("TESTING ALL 5 MCALLISTER EXAMPLES - CYTools 2021")
    print("=" * 70)

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, h21, verbose=True)
        results.append(result)
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r["test_passed"] else "FAIL"
        print(f"  {status}: {r['example_name']} h11={r['cy'].h11()}")
        all_passed = all_passed and r["test_passed"]

    print()
    print("All 5 examples PASSED" if all_passed else "Some examples FAILED")
    return results


if __name__ == "__main__":
    main()
