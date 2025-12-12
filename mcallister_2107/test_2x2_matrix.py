#!/usr/bin/env python3
"""
Test 2x2 matrix: (uncorrected/corrected) × (basis/kklt_basis)

Goal: Find which combination reproduces V_string = 4711.83
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_file(filename):
    """Load comma-separated values from file."""
    text = (DATA_DIR / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_int_file(filename):
    """Load comma-separated integers from file."""
    text = (DATA_DIR / filename).read_text().strip()
    return np.array([int(x) for x in text.split(',')])


def load_points():
    """Load primal polytope points."""
    lines = (DATA_DIR / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def compute_volume_direct(kappa, t):
    """V = (1/6) kappa_ijk t^i t^j t^k"""
    return np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0


def build_kappa_tensor(cy, h11):
    """Build full intersection tensor from CYTools sparse format."""
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa


def test_combination(poly, heights_file, kahler_file, basis_file):
    """Test one combination and return computed volume."""
    # Load data
    heights = load_file(heights_file)
    t_values = load_file(kahler_file)
    basis_indices = load_int_file(basis_file)

    # Create triangulation with heights
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()
    h11 = cy.h11()

    # Set basis
    try:
        cy.set_divisor_basis(list(basis_indices))
    except Exception as e:
        return None, f"set_divisor_basis failed: {e}"

    # Check dimensions match
    if len(t_values) != h11:
        return None, f"t_values length {len(t_values)} != h11 {h11}"

    # Build kappa and compute volume
    kappa = build_kappa_tensor(cy, h11)
    V = compute_volume_direct(kappa, t_values)

    return V, None


def main():
    print("=" * 70)
    print("2x2 TEST MATRIX: (uncorrected/corrected) × (basis/kklt_basis)")
    print("=" * 70)

    # Target
    V_target = float((DATA_DIR / "cy_vol.dat").read_text().strip())
    V_corrected_target = float((DATA_DIR / "corrected_cy_vol.dat").read_text().strip())
    print(f"\nTargets:")
    print(f"  cy_vol.dat:           {V_target:.6f}")
    print(f"  corrected_cy_vol.dat: {V_corrected_target:.6f}")

    # Load polytope once
    print("\nLoading polytope...")
    points = load_points()
    poly = Polytope(points)
    print(f"  Points: {points.shape}")

    # Test matrix
    heights_options = [
        ("heights.dat", "uncorrected"),
        ("corrected_heights.dat", "corrected"),
    ]
    kahler_options = [
        ("kahler_param.dat", "uncorrected"),
        ("corrected_kahler_param.dat", "corrected"),
    ]
    basis_options = [
        ("basis.dat", "basis"),
        ("kklt_basis.dat", "kklt_basis"),
    ]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = []

    for heights_file, heights_label in heights_options:
        for kahler_file, kahler_label in kahler_options:
            for basis_file, basis_label in basis_options:
                label = f"{heights_label}_h + {kahler_label}_k + {basis_label}"

                V, error = test_combination(poly, heights_file, kahler_file, basis_file)

                if error:
                    print(f"\n{label}:")
                    print(f"  ERROR: {error}")
                else:
                    ratio = V / V_target
                    error_pct = abs(V - V_target) / V_target * 100
                    print(f"\n{label}:")
                    print(f"  V_computed = {V:.2f}")
                    print(f"  ratio to target = {ratio:.4f}")
                    print(f"  error = {error_pct:.2f}%")

                    results.append((label, V, ratio, error_pct))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by error)")
    print("=" * 70)
    results.sort(key=lambda x: x[3])
    for label, V, ratio, error_pct in results:
        marker = "✓" if error_pct < 1 else "✗"
        print(f"{marker} {error_pct:6.2f}% | V={V:10.2f} | ratio={ratio:.4f} | {label}")


if __name__ == "__main__":
    main()
