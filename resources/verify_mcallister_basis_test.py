#!/usr/bin/env python3
"""
Test CYTools divisor basis alignment with McAllister paper.

This script loads the 4-214-647 polytope and compares:
1. CYTools' divisor_basis() indices
2. CYTools' intersection numbers in that basis
3. McAllister's reported flat direction p and fluxes K, M

If the bases align, we should be able to verify:
- N_ab = κ̃_abc M^c gives correct N matrix
- p = N⁻¹K gives McAllister's p = (293/110, 163/110, 163/110, 13/22)

This is the critical test to determine if we can use CYTools directly
or need basis transformation.
"""

import numpy as np
from fractions import Fraction
import sys
sys.path.insert(0, 'vendor/cytools_latest')

from cytools import Polytope

# McAllister data for 4-214-647
DATA_DIR = "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"

def load_mcallister_data():
    """Load McAllister's flux vectors and expected results."""
    # From eq. 6.55 in the paper
    K_paper = np.array([-3, -5, 8, 6])
    M_paper = np.array([10, 11, -11, -5])

    # From eq. 6.56 - the flat direction in McAllister's basis
    p_paper = np.array([
        Fraction(293, 110),
        Fraction(163, 110),
        Fraction(163, 110),
        Fraction(13, 22)
    ])

    # Also load from files to verify
    with open(f"{DATA_DIR}/K_vec.dat") as f:
        K_file = np.array([int(x) for x in f.read().strip().split(',')])
    with open(f"{DATA_DIR}/M_vec.dat") as f:
        M_file = np.array([int(x) for x in f.read().strip().split(',')])

    print("McAllister flux vectors:")
    print(f"  K (paper eq. 6.55): {K_paper}")
    print(f"  K (file):           {K_file}")
    print(f"  M (paper eq. 6.55): {M_paper}")
    print(f"  M (file):           {M_file}")
    print(f"  p (paper eq. 6.56): {[float(x) for x in p_paper]}")

    assert np.allclose(K_paper, K_file), "K mismatch!"
    assert np.allclose(M_paper, M_file), "M mismatch!"

    return K_paper, M_paper, p_paper


def load_dual_polytope():
    """Load the dual polytope (h11=4, h21=214) from McAllister data."""
    with open(f"{DATA_DIR}/dual_points.dat") as f:
        lines = f.readlines()

    # Parse points - each line is comma-separated coordinates
    points = []
    for line in lines:
        line = line.strip()
        if line:
            coords = [int(x) for x in line.split(',')]
            points.append(coords)

    points = np.array(points)
    print(f"\nLoaded dual polytope: {len(points)} points")
    print(f"  Points shape: {points.shape}")

    return points


def test_cytools_basis():
    """Test what basis CYTools chooses for the dual polytope."""
    print("\n" + "="*60)
    print("LOADING POLYTOPE INTO CYTOOLS")
    print("="*60)

    # Load dual polytope points
    dual_points = load_dual_polytope()

    # Create CYTools polytope
    poly = Polytope(dual_points)

    print(f"\nPolytope info:")
    print(f"  Is reflexive: {poly.is_reflexive()}")
    print(f"  Dimension: {poly.dim()}")
    print(f"  Points: {poly.points().shape}")

    # Get a triangulation
    print("\nGetting triangulation...")
    tri = poly.triangulate()
    print(f"  Triangulation: {len(tri.simplices())} simplices")

    # Get Calabi-Yau
    print("\nGetting Calabi-Yau...")
    cy = tri.get_cy()

    h11 = cy.h11()
    h21 = cy.h21()
    print(f"  h11 = {h11}")
    print(f"  h21 = {h21}")

    if h11 != 4 or h21 != 214:
        print(f"  WARNING: Expected h11=4, h21=214!")

    # Get the divisor basis CYTools chooses
    print("\n" + "="*60)
    print("CYTOOLS DIVISOR BASIS")
    print("="*60)

    basis = cy.divisor_basis()
    print(f"  Divisor basis indices: {basis}")
    print(f"  (These are ambient toric divisor indices)")

    # Get intersection numbers in this basis
    print("\n" + "="*60)
    print("INTERSECTION NUMBERS (in_basis=True)")
    print("="*60)

    kappa_dict = cy.intersection_numbers(in_basis=True)
    print(f"  Type: {type(kappa_dict)}")
    print(f"  Number of entries: {len(kappa_dict)}")

    # Convert dict to 3D array for easier manipulation
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_dict.items():
        # Symmetrize
        kappa[i, j, k] = val
        kappa[i, k, j] = val
        kappa[j, i, k] = val
        kappa[j, k, i] = val
        kappa[k, i, j] = val
        kappa[k, j, i] = val

    # Print all non-zero intersection numbers
    print("\n  Non-zero κ_ijk (dict format):")
    for (i, j, k), val in sorted(kappa_dict.items()):
        print(f"    κ_{i}{j}{k} = {val}")

    return cy, kappa


def test_demirtas_lemma(cy, kappa):
    """
    Test if we can reproduce McAllister's flat direction using Demirtas lemma.

    The lemma says:
    1. Build N_ab = κ̃_abc M^c
    2. If det(N) ≠ 0, compute p = N⁻¹ K
    3. p is the flat direction

    If CYTools basis matches McAllister, we should get p = (293/110, 163/110, 163/110, 13/22)
    """
    print("\n" + "="*60)
    print("TESTING DEMIRTAS LEMMA (Perturbatively Flat Vacuum)")
    print("="*60)

    K_paper, M_paper, p_paper = load_mcallister_data()

    h11 = cy.h11()

    # Build N_ab = κ̃_abc M^c
    print("\nBuilding N_ab = κ̃_abc M^c...")
    N = np.zeros((h11, h11))
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                N[a, b] += kappa[a, b, c] * M_paper[c]

    print(f"  N matrix:\n{N}")
    print(f"  det(N) = {np.linalg.det(N)}")

    if abs(np.linalg.det(N)) < 1e-10:
        print("  ERROR: N is singular, cannot compute p!")
        return None

    # Compute p = N⁻¹ K
    print("\nComputing p = N⁻¹ K...")
    N_inv = np.linalg.inv(N)
    p_cytools = N_inv @ K_paper

    print(f"  p (CYTools basis): {p_cytools}")
    print(f"  p (McAllister):    {[float(x) for x in p_paper]}")

    # Check if they match
    p_paper_float = np.array([float(x) for x in p_paper])
    if np.allclose(p_cytools, p_paper_float, rtol=1e-6):
        print("\n  ✓ SUCCESS! CYTools basis matches McAllister!")
        return True
    else:
        print("\n  ✗ MISMATCH: CYTools basis differs from McAllister")
        print(f"  Difference: {p_cytools - p_paper_float}")

        # Try to understand the relationship
        print("\n  Attempting to find basis transformation...")

        # Try simple permutations and sign flips
        from itertools import permutations

        best_perm = None
        best_signs = None
        best_error = float('inf')

        for perm in permutations(range(4)):
            for s0 in [-1, 1]:
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        for s3 in [-1, 1]:
                            signs = np.array([s0, s1, s2, s3])
                            # Apply permutation and signs to p_cytools
                            p_test = signs * p_cytools[list(perm)]
                            error = np.linalg.norm(p_test - p_paper_float)
                            if error < best_error:
                                best_error = error
                                best_perm = perm
                                best_signs = signs

        print(f"\n  Best permutation: {best_perm}")
        print(f"  Best signs: {best_signs}")
        print(f"  Best error: {best_error}")

        if best_error < 0.1:
            p_transformed = best_signs * p_cytools[list(best_perm)]
            print(f"  p transformed: {p_transformed}")
            print(f"  p target:      {p_paper_float}")

        # Also check if there's a linear transformation
        # p_paper = T @ p_cytools where T is some matrix
        # We need more equations to solve for T (4x4 = 16 unknowns, only 4 equations)
        # But if we have the fluxes transformed consistently...
        print("\n  Checking if simple matrix transformation works...")

        return False


def test_alternative_triangulations(poly):
    """
    Try different triangulations to see if any give matching basis.
    McAllister might have used a specific triangulation.
    """
    print("\n" + "="*60)
    print("TESTING ALTERNATIVE TRIANGULATIONS")
    print("="*60)

    # Load McAllister's simplices
    with open(f"{DATA_DIR}/dual_simplices.dat") as f:
        lines = f.readlines()

    mcallister_simplices = []
    for line in lines:
        line = line.strip()
        if line:
            simplex = [int(x) for x in line.split(',')]
            mcallister_simplices.append(simplex)

    print(f"McAllister triangulation: {len(mcallister_simplices)} simplices")
    print(f"  First few: {mcallister_simplices[:3]}")

    # Try to use McAllister's triangulation directly
    print("\nAttempting to use McAllister's exact triangulation...")
    try:
        tri = poly.triangulate(simplices=mcallister_simplices)
        cy = tri.get_cy()
        print(f"  Success! h11={cy.h11()}, h21={cy.h21()}")

        basis = cy.divisor_basis()
        print(f"  Divisor basis: {basis}")

        return cy
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def main():
    print("="*60)
    print("CYTools Basis Alignment Test for McAllister 4-214-647")
    print("="*60)

    # First, test with default CYTools triangulation
    cy, kappa = test_cytools_basis()

    # Test Demirtas lemma
    matches = test_demirtas_lemma(cy, kappa)

    if not matches:
        # Try McAllister's specific triangulation
        dual_points = load_dual_polytope()
        poly = Polytope(dual_points)
        cy_mcallister = test_alternative_triangulations(poly)

        if cy_mcallister is not None:
            kappa_mc_dict = cy_mcallister.intersection_numbers(in_basis=True)
            # Convert to array
            h11 = cy_mcallister.h11()
            kappa_mc = np.zeros((h11, h11, h11))
            for (i, j, k), val in kappa_mc_dict.items():
                kappa_mc[i, j, k] = val
                kappa_mc[i, k, j] = val
                kappa_mc[j, i, k] = val
                kappa_mc[j, k, i] = val
                kappa_mc[k, i, j] = val
                kappa_mc[k, j, i] = val
            print("\nRetrying with McAllister triangulation...")
            test_demirtas_lemma(cy_mcallister, kappa_mc)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if matches:
        print("CYTools basis MATCHES McAllister - can proceed with GV derivation!")
    else:
        print("CYTools basis DIFFERS from McAllister - need transformation or 2021 CYTools")


if __name__ == "__main__":
    main()
