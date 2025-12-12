#!/usr/bin/env python3
"""
Verify V_string = 4711.829675204889 EXACTLY.

This script demonstrates the complete, validated computation of V_string.

Key insights:
1. Use corrected_kahler_param.dat (NOT kahler_param.dat) - 3.8× difference!
2. Use heights.dat for correct triangulation
3. Apply BBHL α' correction: V = V_classical - ζ(3)χ/(4(2π)³)

Reference: McAllister et al. arXiv:2107.09064, eq. 4.11
"""

import sys
from pathlib import Path

# Use McAllister's CYTools version for exact match
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))

import numpy as np
from scipy.special import zeta
from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_comma_separated(filename, dtype=float):
    """Load comma-separated values from file."""
    text = (DATA_DIR / filename).read_text().strip()
    return np.array([dtype(x) for x in text.split(',')])


def load_points():
    """Load primal polytope points."""
    lines = (DATA_DIR / "points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def build_kappa_tensor(cy, h11):
    """Build full intersection tensor from CYTools."""
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))

    # Handle both dict (latest) and array (2021) formats
    if hasattr(kappa_sparse, 'items'):
        for (i, j, k), val in kappa_sparse.items():
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[perm] = val
    else:
        for row in kappa_sparse:
            i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
            for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                kappa[perm] = val
    return kappa


def compute_V_string(kappa, t, chi):
    """
    Compute V_string with BBHL correction (eq. 4.11).

    V[0] = (1/6) κ_ijk t^i t^j t^k - ζ(3)χ/(4(2π)³)
    """
    V_classical = np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0
    BBHL = zeta(3) * chi / (4 * (2 * np.pi)**3)
    V_string = V_classical - BBHL
    return V_string, V_classical, BBHL


def main():
    print("=" * 70)
    print("EXACT V_string VERIFICATION")
    print("=" * 70)

    # Load targets
    V_target = float((DATA_DIR / "cy_vol.dat").read_text().strip())
    V_corrected_target = float((DATA_DIR / "corrected_cy_vol.dat").read_text().strip())

    print(f"\nTarget values from McAllister data:")
    print(f"  cy_vol.dat:           {V_target}")
    print(f"  corrected_cy_vol.dat: {V_corrected_target}")

    # Load geometry
    print(f"\n[1] Loading geometry...")
    points = load_points()
    heights = load_comma_separated("heights.dat")

    poly = Polytope(points)
    tri = poly.triangulate(heights=heights)
    cy = tri.get_cy()

    h11, h21 = cy.h11(), cy.h21()
    chi = 2 * (h11 - h21)  # Euler characteristic

    print(f"  h11 = {h11}, h21 = {h21}")
    print(f"  χ = 2(h11 - h21) = {chi}")

    # Load corrected Kähler moduli (CRITICAL!)
    print(f"\n[2] Loading corrected Kähler moduli...")
    t = load_comma_separated("corrected_kahler_param.dat")
    basis = load_comma_separated("basis.dat", dtype=int)
    print(f"  {len(t)} moduli loaded")

    # Set basis and build intersection tensor
    print(f"\n[3] Computing intersection numbers...")
    cy.set_divisor_basis(list(basis))
    kappa = build_kappa_tensor(cy, h11)
    print(f"  Non-zero entries: {np.sum(kappa != 0)}")

    # Compute V_string
    print(f"\n[4] Computing V_string...")
    V_string, V_classical, BBHL = compute_V_string(kappa, t, chi)

    print(f"\n  V_classical = (1/6) κ_ijk t^i t^j t^k = {V_classical:.6f}")
    print(f"  BBHL = ζ(3)χ/(4(2π)³) = {BBHL:.6f}")
    print(f"  V_string = V_classical - BBHL = {V_string:.6f}")

    # Compare
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    diff = V_string - V_target
    error_pct = abs(diff) / V_target * 100

    print(f"\n  V_string (computed): {V_string:.12f}")
    print(f"  V_string (target):   {V_target:.12f}")
    print(f"  Difference:          {diff:+.12f}")
    print(f"  Error:               {error_pct:.6f}%")

    if abs(diff) < 0.01:
        print(f"\n  ✓ EXACT MATCH (within 0.01)")
    elif abs(diff) < 0.1:
        print(f"\n  ~ Close match (within 0.1)")
    else:
        print(f"\n  ✗ Mismatch")

    # Show what happens without BBHL
    print(f"\n" + "-" * 70)
    print("WITHOUT BBHL correction (common mistake):")
    print("-" * 70)
    print(f"  V_classical:  {V_classical:.6f}")
    print(f"  Error:        {(V_classical - V_target):+.2f} ({abs(V_classical - V_target)/V_target*100:.2f}%)")

    # Show what happens with uncorrected t
    print(f"\n" + "-" * 70)
    print("WITH uncorrected kahler_param.dat (common mistake):")
    print("-" * 70)
    t_uncorr = load_comma_separated("kahler_param.dat")
    V_uncorr = np.einsum('ijk,i,j,k->', kappa, t_uncorr, t_uncorr, t_uncorr) / 6.0
    print(f"  V_uncorrected: {V_uncorr:.2f}")
    print(f"  Error:         {(V_uncorr - V_target):+.2f} ({abs(V_uncorr - V_target)/V_target*100:.0f}%)")
    print(f"  Ratio:         {V_uncorr/V_target:.2f}x (should be 1.0)")


if __name__ == "__main__":
    main()
