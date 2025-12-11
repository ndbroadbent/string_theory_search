#!/usr/bin/env python3
"""
Test CYTools basis alignment with McAllister arXiv:2107.09064.

Uses the era-appropriate CYTools version (June 2021, commit bb5b550)
to check if the divisor basis matches McAllister's published data.

The key test: Given fluxes K, M and intersection numbers κ,
compute p = N⁻¹K where N_ab = κ_abc M^c.
If basis aligns, we should get p = (293/110, 163/110, 163/110, 13/22).
"""

import numpy as np
from fractions import Fraction
from pathlib import Path

from cytools import Polytope

# Path to McAllister data
DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_mcallister_data():
    """Load McAllister's flux vectors and expected flat direction."""
    # From eq. 6.55
    K = np.array([-3, -5, 8, 6])
    M = np.array([10, 11, -11, -5])

    # From eq. 6.56 - expected result
    p_expected = np.array([
        float(Fraction(293, 110)),
        float(Fraction(163, 110)),
        float(Fraction(163, 110)),
        float(Fraction(13, 22))
    ])

    # Verify against files
    K_file = np.array([int(x) for x in (DATA_DIR / "K_vec.dat").read_text().strip().split(',')])
    M_file = np.array([int(x) for x in (DATA_DIR / "M_vec.dat").read_text().strip().split(',')])

    assert np.array_equal(K, K_file), f"K mismatch: {K} vs {K_file}"
    assert np.array_equal(M, M_file), f"M mismatch: {M} vs {M_file}"

    print("McAllister data loaded:")
    print(f"  K = {K}")
    print(f"  M = {M}")
    print(f"  p_expected = {p_expected}")

    return K, M, p_expected


def load_dual_polytope():
    """Load dual polytope points from McAllister data."""
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
    points = np.array([[int(x) for x in line.split(',')] for line in lines])
    print(f"\nDual polytope: {len(points)} points, shape {points.shape}")
    return points


def load_mcallister_triangulation():
    """Load McAllister's specific triangulation."""
    lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
    simplices = [[int(x) for x in line.split(',')] for line in lines]
    print(f"McAllister triangulation: {len(simplices)} simplices")
    return simplices


def test_cytools_basis():
    """Main test: check if CYTools basis matches McAllister."""
    print("=" * 60)
    print("CYTools Basis Alignment Test")
    print("=" * 60)

    K, M, p_expected = load_mcallister_data()
    dual_points = load_dual_polytope()
    mcallister_simplices = load_mcallister_triangulation()

    # Create polytope
    print("\nCreating CYTools polytope...")
    poly = Polytope(dual_points)
    print(f"  Reflexive: {poly.is_reflexive()}")
    print(f"  Dimension: {poly.dim()}")

    # Use McAllister's exact triangulation
    print("\nUsing McAllister's triangulation...")
    tri = poly.triangulate(simplices=mcallister_simplices, check_input_simplices=False)

    # Get CY
    print("\nGetting Calabi-Yau...")
    cy = tri.get_cy()
    print(f"  h11 = {cy.h11()}")
    print(f"  h21 = {cy.h21()}")

    if cy.h11() != 4 or cy.h21() != 214:
        print("  ERROR: Wrong Hodge numbers!")
        return False

    # Get divisor basis
    print("\nDivisor basis:")
    basis = cy.divisor_basis()
    print(f"  Indices: {basis}")

    # Get intersection numbers
    print("\nIntersection numbers (in_basis=True):")
    kappa_result = cy.intersection_numbers(in_basis=True)
    print(f"  Type: {type(kappa_result)}")

    h11 = cy.h11()

    # Handle both dict (newer CYTools) and array (older CYTools)
    if isinstance(kappa_result, dict):
        kappa = np.zeros((h11, h11, h11))
        for key, val in kappa_result.items():
            if isinstance(key, tuple) and len(key) == 3:
                i, j, k = key
                for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                    kappa[perm] = val
        print("\n  Non-zero κ_ijk:")
        for (i, j, k), val in sorted(kappa_result.items()):
            print(f"    κ_{i}{j}{k} = {val}")
    else:
        # Older CYTools returns different format
        print(f"  Shape: {kappa_result.shape}")
        print(f"  Data:\n{kappa_result}")

        # This appears to be a sparse representation: each row is [i, j, k, value]
        # Or perhaps [idx, value] pairs. Let's interpret it.
        if kappa_result.shape[1] == 4:
            # Format: [i, j, k, value]
            kappa = np.zeros((h11, h11, h11))
            print("\n  Non-zero κ_ijk:")
            for row in kappa_result:
                i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
                print(f"    κ_{i}{j}{k} = {val}")
                # Symmetrize
                for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                    kappa[perm] = val
        else:
            raise ValueError(f"Unexpected intersection number format: {kappa_result.shape}")

    # Build N_ab = κ_abc M^c
    print("\n" + "=" * 60)
    print("Demirtas Lemma: Computing p = N⁻¹K")
    print("=" * 60)

    N = np.zeros((h11, h11))
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                N[a, b] += kappa[a, b, c] * M[c]

    print(f"\nN_ab = κ_abc M^c:")
    print(N)
    print(f"\ndet(N) = {np.linalg.det(N)}")

    if abs(np.linalg.det(N)) < 1e-10:
        print("ERROR: N is singular!")
        return False

    # Compute p = N⁻¹ K
    p_computed = np.linalg.solve(N, K)

    print(f"\np computed: {p_computed}")
    print(f"p expected: {p_expected}")
    print(f"Difference: {p_computed - p_expected}")

    # Check match
    if np.allclose(p_computed, p_expected, rtol=1e-6):
        print("\n" + "=" * 60)
        print("✓ SUCCESS! Basis alignment confirmed!")
        print("=" * 60)

        # Now compute e^{K₀} from eq. 6.12
        print("\n" + "=" * 60)
        print("Computing e^{K₀} from κ and p (eq. 6.12)")
        print("=" * 60)

        # e^{K₀} = (4/3 κ̃_abc p^a p^b p^c)^{-1}
        kappa_p3 = 0.0
        for a in range(h11):
            for b in range(h11):
                for c in range(h11):
                    kappa_p3 += kappa[a, b, c] * p_computed[a] * p_computed[b] * p_computed[c]

        print(f"  κ_abc p^a p^b p^c = {kappa_p3}")
        eK0 = 1.0 / ((4.0/3.0) * kappa_p3)
        print(f"  e^{{K₀}} = (4/3 × {kappa_p3})⁻¹ = {eK0}")
        print(f"  Expected from paper (back-calculated): ~0.2361")

        # Load other McAllister data to verify full pipeline
        g_s = float((DATA_DIR / "g_s.dat").read_text().strip())
        W0 = float((DATA_DIR / "W_0.dat").read_text().strip())
        V0_string = float((DATA_DIR / "cy_vol.dat").read_text().strip())

        print(f"\n  McAllister published values:")
        print(f"    g_s = {g_s}")
        print(f"    W₀ = {W0}")
        print(f"    V[0] (string frame) = {V0_string}")

        # Compute V₀ using eq. 6.24
        print("\n" + "=" * 60)
        print("Computing V₀ from eq. 6.24")
        print("=" * 60)
        print("  V₀ = -3 × e^{K₀} × (g_s^7 / (4×V[0])²) × W₀²")

        V0 = -3.0 * eK0 * (g_s**7 / (4.0 * V0_string)**2) * W0**2
        print(f"  V₀ = -3 × {eK0:.4f} × ({g_s}^7 / (4×{V0_string})²) × ({W0})²")
        print(f"  V₀ = {V0:.6e}")
        print(f"  Expected: -5.5e-203")

        return True, p_computed, kappa, eK0
    else:
        print("\n" + "=" * 60)
        print("✗ MISMATCH: Basis does not align")
        print("=" * 60)

        # Try to find transformation
        search_transformation(p_computed, p_expected)
        return False


def search_transformation(p_computed, p_expected):
    """Search for simple transformations between bases."""
    from itertools import permutations

    print("\nSearching for permutation + sign transformation...")

    best_error = float('inf')
    best_perm = None
    best_signs = None

    for perm in permutations(range(4)):
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    for s3 in [-1, 1]:
                        signs = np.array([s0, s1, s2, s3])
                        p_test = signs * p_computed[list(perm)]
                        error = np.linalg.norm(p_test - p_expected)
                        if error < best_error:
                            best_error = error
                            best_perm = perm
                            best_signs = signs

    print(f"  Best permutation: {best_perm}")
    print(f"  Best signs: {best_signs}")
    print(f"  Best error: {best_error}")

    if best_error < 0.01:
        p_transformed = best_signs * p_computed[list(best_perm)]
        print(f"  Transformed p: {p_transformed}")


if __name__ == "__main__":
    test_cytools_basis()
