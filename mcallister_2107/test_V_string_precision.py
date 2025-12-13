#!/usr/bin/env python3
"""
Precision test for V_string computation.

HYPOTHESIS: The data files should match as follows:
- heights.dat + kahler_param.dat → cy_vol.dat
- corrected_heights.dat + corrected_kahler_param.dat → corrected_cy_vol.dat
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
sys.path.insert(0, str(CYTOOLS_2021))

import numpy as np
from scipy.special import zeta
from cytools import Polytope

DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

EXAMPLES = [
    ("4-214-647", 214, 4),
    ("5-113-4627-main", 113, 5),
    ("5-113-4627-alternative", 113, 5),
    ("5-81-3213", 81, 5),
    # ("7-51-13590", 51, 7),  # non-favorable
]


def load_primal_points(example_name: str) -> np.ndarray:
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_heights(example_name: str, corrected: bool) -> np.ndarray:
    data_dir = DATA_BASE / example_name
    filename = "corrected_heights.dat" if corrected else "heights.dat"
    text = (data_dir / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_kahler_params(example_name: str, corrected: bool) -> np.ndarray:
    data_dir = DATA_BASE / example_name
    filename = "corrected_kahler_param.dat" if corrected else "kahler_param.dat"
    text = (data_dir / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_basis(example_name: str) -> list:
    data_dir = DATA_BASE / example_name
    text = (data_dir / "basis.dat").read_text().strip()
    return [int(x) for x in text.split(',')]


def load_expected_V(example_name: str, corrected: bool) -> float:
    data_dir = DATA_BASE / example_name
    filename = "corrected_cy_vol.dat" if corrected else "cy_vol.dat"
    return float((data_dir / filename).read_text().strip())


def compute_V_string(kappa: np.ndarray, t: np.ndarray, h11: int, h21: int) -> dict:
    V_classical = np.einsum("ijk,i,j,k->", kappa, t, t, t) / 6.0
    chi = 2 * (h11 - h21)
    BBHL = zeta(3) * chi / (4 * (2 * np.pi) ** 3)
    V_string = V_classical - BBHL
    return {"V_classical": V_classical, "BBHL": BBHL, "V_string": V_string, "chi": chi}


def test_example(name: str, expected_h11: int, expected_h21: int):
    """Test with MATCHED corrected flag for heights, kahler, and cy_vol."""
    print(f"\n{'='*70}")
    print(f"EXAMPLE: {name} (primal h11={expected_h11})")
    print(f"{'='*70}")

    points = load_primal_points(name)
    basis = load_basis(name)

    results = {}

    for corrected in [False, True]:
        label = "CORRECTED" if corrected else "UNCORRECTED"
        print(f"\n--- {label} (heights + kahler_param + cy_vol all {label.lower()}) ---")

        # Load MATCHED heights
        heights = load_heights(name, corrected=corrected)

        # Build CY
        poly = Polytope(points)
        tri = poly.triangulate(heights=heights)
        cy = tri.get_cy()
        h11, h21 = cy.h11(), cy.h21()
        cy.set_divisor_basis(basis)

        # Get kappa
        kappa_sparse = cy.intersection_numbers(in_basis=True)
        kappa = np.zeros((h11, h11, h11))
        for row in kappa_sparse:
            i, j, k = int(row[0]), int(row[1]), int(row[2])
            val = row[3]
            for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                kappa[perm] = val

        # Load MATCHED kahler params
        t = load_kahler_params(name, corrected=corrected)

        # Load MATCHED expected V
        V_expected = load_expected_V(name, corrected=corrected)

        # Compute
        result = compute_V_string(kappa, t, h11, h21)
        V_computed = result["V_string"]

        print(f"  V_classical: {result['V_classical']:.15f}")
        print(f"  BBHL: {result['BBHL']:.15f}")
        print(f"  V_string (computed): {V_computed:.15f}")
        print(f"  V_string (expected): {V_expected:.15f}")

        diff = V_computed - V_expected
        rel_error = abs(diff) / V_expected

        print(f"  Difference: {diff:.15e}")
        print(f"  Relative error: {rel_error:.15e}")

        if rel_error < 1e-14:
            status = "EXACT MATCH"
        elif rel_error < 1e-10:
            status = "MATCH (float precision)"
        else:
            status = f"MISMATCH"
        print(f"  Status: {status}")

        results[label] = {"computed": V_computed, "expected": V_expected, "error": rel_error}

    return results


def main():
    print("="*70)
    print("V_STRING PRECISION TEST - MATCHED FILES")
    print("="*70)

    all_results = {}
    for name, h11, h21 in EXAMPLES:
        result = test_example(name, h11, h21)
        all_results[name] = result

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, res in all_results.items():
        print(f"\n{name}:")
        for label, data in res.items():
            err = data["error"]
            if err < 1e-14:
                status = "EXACT"
            elif err < 1e-10:
                status = "MATCH"
            else:
                status = "FAIL"
            print(f"  {label}: error={err:.2e} [{status}]")


if __name__ == "__main__":
    main()
