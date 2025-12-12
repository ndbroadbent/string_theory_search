#!/usr/bin/env python3
"""
Test for EXACT match of V_string = 4711.829675204889

Using McAllister's CYTools version (2021) from vendor/cytools_mcallister_2107
"""

import sys
from pathlib import Path

# Use McAllister's CYTools version
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))

import numpy as np
from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_file(filename):
    text = (DATA_DIR / filename).read_text().strip()
    return np.array([float(x) for x in text.split(',')])


def load_int_file(filename):
    text = (DATA_DIR / filename).read_text().strip()
    return np.array([int(x) for x in text.split(',')])


def load_points():
    lines = (DATA_DIR / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def compute_volume_direct(kappa, t):
    return np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0


def build_kappa_tensor(cy, h11):
    kappa_sparse = cy.intersection_numbers(in_basis=True)

    # Handle both dict (latest CYTools) and array (2021 CYTools) formats
    if hasattr(kappa_sparse, 'items'):
        # Latest CYTools: dict format
        kappa = np.zeros((h11, h11, h11))
        for (i, j, k), val in kappa_sparse.items():
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[perm] = val
    else:
        # 2021 CYTools: array format [[i, j, k, val], ...]
        kappa = np.zeros((h11, h11, h11))
        for row in kappa_sparse:
            i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[perm] = val
    return kappa


def main():
    print("=" * 70)
    print("EXACT MATCH TEST - Using McAllister's CYTools (2021)")
    print("=" * 70)

    # Targets
    V_target = float((DATA_DIR / "cy_vol.dat").read_text().strip())
    V_corrected = float((DATA_DIR / "corrected_cy_vol.dat").read_text().strip())
    print(f"\nTargets:")
    print(f"  cy_vol.dat:           {V_target}")
    print(f"  corrected_cy_vol.dat: {V_corrected}")

    # Load data
    points = load_points()
    heights = load_file("heights.dat")
    corrected_heights = load_file("corrected_heights.dat")
    t_corrected = load_file("corrected_kahler_param.dat")
    basis = load_int_file("basis.dat")

    print(f"\nData loaded:")
    print(f"  Points: {points.shape}")
    print(f"  Heights: {len(heights)}")
    print(f"  Corrected t: {len(t_corrected)}")
    print(f"  Basis: {len(basis)} indices")

    # Create polytope
    poly = Polytope(points)
    print(f"\nPolytope created")

    # Test with different triangulations
    for heights_name, heights_data in [("default", None), ("heights.dat", heights), ("corrected_heights.dat", corrected_heights)]:
        print(f"\n--- Triangulation: {heights_name} ---")

        try:
            if heights_data is None:
                tri = poly.triangulate()
            else:
                tri = poly.triangulate(heights=heights_data)

            cy = tri.get_cy()
            h11 = cy.h11()
            print(f"  h11={h11}, h21={cy.h21()}")

            # Set basis
            cy.set_divisor_basis(list(basis))
            actual_basis = list(cy.divisor_basis())
            print(f"  Basis set: {actual_basis[:5]}...")

            # Build kappa
            kappa = build_kappa_tensor(cy, h11)

            # Compute volume
            V = compute_volume_direct(kappa, t_corrected)

            diff_target = V - V_target
            diff_corrected = V - V_corrected

            print(f"  V_computed:    {V:.12f}")
            print(f"  vs cy_vol:     {diff_target:+.12f} ({abs(diff_target/V_target)*100:.6f}%)")
            print(f"  vs corrected:  {diff_corrected:+.12f} ({abs(diff_corrected/V_corrected)*100:.6f}%)")

            if abs(diff_target) < 0.001:
                print(f"  *** EXACT MATCH with cy_vol.dat! ***")
            if abs(diff_corrected) < 0.001:
                print(f"  *** EXACT MATCH with corrected_cy_vol.dat! ***")

        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
