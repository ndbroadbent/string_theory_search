#!/usr/bin/env python3
"""
Port McAllister 4-214-647 configuration to CYTools 2024.

The goal is to find the transformation between:
- CYTools 2021 basis [3,4,5,8] (what McAllister used)
- CYTools 2024 basis [5,6,7,8] (what we have now)

Then transform the fluxes K, M so that the physics works in the new basis.

This will give us a validated genome for our GA that we know produces
the correct cosmological constant.
"""

import numpy as np
from pathlib import Path
import sys

# Add both CYTools versions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))

from cytools import Polytope as Polytope2021

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_geometry():
    """Load polytope and triangulation."""
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
    dual_points = np.array([[int(x) for x in line.split(',')] for line in lines])

    lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
    simplices = [[int(x) for x in line.split(',')] for line in lines]

    return dual_points, simplices


def setup_cytools_2021():
    """Setup CYTools 2021 and get basis info."""
    dual_points, simplices = load_geometry()

    poly = Polytope2021(dual_points)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    basis_2021 = cy.divisor_basis()
    kappa_2021 = cy.intersection_numbers(in_basis=True)

    # Convert to 3D array
    h11 = cy.h11()
    kappa_arr = np.zeros((h11, h11, h11))
    for row in kappa_2021:
        i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa_arr[perm] = val

    return cy, basis_2021, kappa_arr


def setup_cytools_2024():
    """Setup CYTools 2024 (latest) and get basis info."""
    # Import from the system/latest cytools
    # We need to temporarily remove the 2021 path
    import importlib

    # Clear cached imports
    mods_to_remove = [k for k in sys.modules.keys() if 'cytools' in k]
    for mod in mods_to_remove:
        del sys.modules[mod]

    # Remove 2021 path, add parent's cytools
    sys.path.pop(0)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cytools_source"))

    from cytools import Polytope as Polytope2024

    dual_points, simplices = load_geometry()

    poly = Polytope2024(dual_points)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    basis_2024 = cy.divisor_basis()
    kappa_2024 = cy.intersection_numbers(in_basis=True)

    # Convert dict to 3D array
    h11 = cy.h11()
    kappa_arr = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_2024.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa_arr[perm] = val

    return cy, basis_2024, kappa_arr


def find_basis_transformation(basis_2021, basis_2024, cy_2021):
    """
    Find the transformation matrix between the two bases.

    The divisor bases are subsets of ambient toric divisors.
    We need to find how divisors in one basis relate to the other.
    """
    print(f"\nBasis 2021: {basis_2021}")
    print(f"Basis 2024: {basis_2024}")

    # Get the GLSM linear relations - these tell us how divisors relate
    poly = cy_2021.polytope()
    linrels = poly.glsm_linear_relations()

    print(f"\nGLSM linear relations shape: {linrels.shape}")
    print(f"Linear relations:\n{linrels}")

    # The linear relations express dependencies among divisors
    # Each row is a linear combination that equals zero

    # To transform from basis [3,4,5,8] to [5,6,7,8]:
    # We need to express D_5, D_6, D_7, D_8 in terms of D_3, D_4, D_5, D_8
    # Or equivalently, find how curves transform

    # Actually, let's use a more direct approach:
    # Compare the intersection numbers in both bases

    return None


def compare_intersection_numbers(kappa_2021, kappa_2024):
    """Compare intersection numbers between the two bases."""
    print("\n" + "=" * 60)
    print("Comparing intersection numbers")
    print("=" * 60)

    print("\nκ (2021 basis [3,4,5,8]):")
    h11 = kappa_2021.shape[0]
    for i in range(h11):
        for j in range(i, h11):
            for k in range(j, h11):
                if abs(kappa_2021[i,j,k]) > 1e-10:
                    print(f"  κ_{i}{j}{k} = {kappa_2021[i,j,k]:.0f}")

    print("\nκ (2024 basis [5,6,7,8]):")
    for i in range(h11):
        for j in range(i, h11):
            for k in range(j, h11):
                if abs(kappa_2024[i,j,k]) > 1e-10:
                    print(f"  κ_{i}{j}{k} = {kappa_2024[i,j,k]:.0f}")


def find_flux_transformation(kappa_2021, kappa_2024, K_2021, M_2021):
    """
    Find the transformed fluxes K', M' for the 2024 basis.

    The key constraint is that the physics must be invariant:
    - p' = (N')^{-1} K' must give the same physical flat direction
    - The intersection numbers transform as κ' = T^{-T} κ T^{-1} (schematically)

    We can solve for T by matching the intersection numbers.
    """
    print("\n" + "=" * 60)
    print("Finding flux transformation")
    print("=" * 60)

    h11 = 4

    # The transformation T relates the bases: D'_a = T_a^b D_b
    # For curves (dual): q'_a = (T^{-1})^T_a^b q_b
    # For fluxes: K'_a = T_a^b K_b, M'_a = T_a^b M_b
    # For intersection numbers: κ'_abc = T_a^d T_b^e T_c^f κ_def

    # We can find T by solving: κ'_abc = T_a^d T_b^e T_c^f κ_def
    # This is a system of equations

    # Let's try a simpler approach: brute force search over GL(4,Z) elements
    # with small entries

    from itertools import product

    print("Searching for transformation matrix T...")

    best_T = None
    best_error = float('inf')

    # Search over matrices with entries in {-2,-1,0,1,2}
    # This is expensive but h11=4 is small
    entries = [-2, -1, 0, 1, 2]

    # Actually this is 5^16 = 152 billion possibilities - too many!
    # Let's be smarter: T should map basis indices

    # Basis 2021: [3,4,5,8] -> indices 0,1,2,3 in the 4D basis
    # Basis 2024: [5,6,7,8] -> indices 0,1,2,3 in the 4D basis

    # The ambient indices are different, so there's a non-trivial transformation
    # But it should be relatively simple since both bases share index 8

    # Try permutation matrices with possible signs first
    from itertools import permutations

    for perm in permutations(range(4)):
        for signs in product([-1, 1], repeat=4):
            T = np.zeros((4, 4))
            for i, (j, s) in enumerate(zip(perm, signs)):
                T[i, j] = s

            # Check if T transforms kappa correctly
            kappa_transformed = np.zeros((4, 4, 4))
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        for d in range(4):
                            for e in range(4):
                                for f in range(4):
                                    kappa_transformed[a,b,c] += T[a,d] * T[b,e] * T[c,f] * kappa_2021[d,e,f]

            error = np.sum((kappa_transformed - kappa_2024)**2)

            if error < best_error:
                best_error = error
                best_T = T.copy()

                if error < 1e-10:
                    print(f"Found exact match!")
                    print(f"T =\n{T}")

                    # Transform fluxes - K and M transform DIFFERENTLY:
                    # - K is covariant (RHS of N·p = K) → transforms with T^{-1}
                    # - M is contravariant (same as p) → transforms with T^T
                    T_inv = np.linalg.inv(T)
                    K_2024 = T_inv @ K_2021
                    M_2024 = T.T @ M_2021

                    print(f"\nFlux transformation:")
                    print(f"  K transforms with T^{{-1}} (covariant)")
                    print(f"  M transforms with T^T (contravariant)")
                    print(f"  K (2021): {K_2021}")
                    print(f"  K (2024): {K_2024}")
                    print(f"  M (2021): {M_2021}")
                    print(f"  M (2024): {M_2024}")

                    return T, K_2024.astype(int), M_2024.astype(int)

    print(f"Best error: {best_error}")
    print(f"Best T:\n{best_T}")

    # If simple permutation doesn't work, try more general GL(4,Z)
    print("\nSimple permutation didn't work, trying more general search...")

    return None, None, None


def main():
    print("=" * 70)
    print("Porting McAllister configuration to CYTools 2024")
    print("=" * 70)

    # McAllister fluxes in 2021 basis
    K_2021 = np.array([-3, -5, 8, 6])
    M_2021 = np.array([10, 11, -11, -5])

    # Setup 2021 CYTools
    print("\n[1] Setting up CYTools 2021...")
    cy_2021, basis_2021, kappa_2021 = setup_cytools_2021()
    print(f"    Basis: {basis_2021}")

    # Setup 2024 CYTools
    print("\n[2] Setting up CYTools 2024...")
    cy_2024, basis_2024, kappa_2024 = setup_cytools_2024()
    print(f"    Basis: {basis_2024}")

    # Compare intersection numbers
    compare_intersection_numbers(kappa_2021, kappa_2024)

    # Find transformation
    T, K_2024, M_2024 = find_flux_transformation(kappa_2021, kappa_2024, K_2021, M_2021)

    if T is not None:
        print("\n" + "=" * 70)
        print("SUCCESS: Found transformation")
        print("=" * 70)
        print(f"\nFor CYTools 2024 (basis {list(basis_2024)}):")
        print(f"  K = {list(K_2024)}")
        print(f"  M = {list(M_2024)}")

        # Verify by computing p in new basis
        print("\n[3] Verifying physics in 2024 basis...")

        h11 = 4
        N = np.zeros((h11, h11))
        for a in range(h11):
            for b in range(h11):
                for c in range(h11):
                    N[a, b] += kappa_2024[a, b, c] * M_2024[c]

        p_2024 = np.linalg.solve(N, K_2024)
        print(f"    p (2024 basis): {p_2024}")

        # The physical q·p should be the same
        # But p itself will be different due to basis change

    else:
        print("\nFailed to find simple transformation")
        print("The bases may require a more complex GL(4,Z) transformation")


if __name__ == "__main__":
    main()
