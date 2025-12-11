#!/usr/bin/env python3
"""
Transform McAllister K, M fluxes from CYTools 2021 basis to latest CYTools basis.

CYTools 2021 basis: [3, 4, 5, 8]
CYTools latest basis: [5, 6, 7, 8]

The transformation matrix T (derived from GLSM linear relations):
    D_old = T @ D_new  (how old basis divisors relate to new basis)

Fluxes transform DIFFERENTLY based on their index type:
- K is covariant (RHS of N·p = K) → K_new = T^{-1} @ K_old
- M is contravariant (same as p) → M_new = T^T @ M_old

This preserves the F-term equation N·p = K and keeps e^{K0} invariant.
"""

import numpy as np


# Known transformation matrix (from GLSM linear relations)
# Maps old basis [3,4,5,8] to new basis [5,6,7,8]
T = np.array([
    [-1,  1,  0,  0],  # D3 = -D5 + D6
    [ 1, -1,  1,  0],  # D4 = D5 - D6 + D7
    [ 1,  0,  0,  0],  # D5 = D5
    [ 0,  0,  0,  1],  # D8 = D8
])

T_inv = np.linalg.inv(T)

# Intersection numbers for verification
KAPPA_OLD_SPARSE = {
    (0,0,0): 1, (0,0,1): -1, (0,0,2): -1, (0,1,1): 1, (0,1,2): 1, (0,2,2): 1,
    (1,1,1): -1, (1,1,2): -1, (1,2,2): -1, (1,2,3): 1, (1,3,3): -2,
    (2,2,2): -1, (2,3,3): -2, (3,3,3): 8
}

KAPPA_NEW_SPARSE = {
    (0,0,0): -1, (0,2,3): 1, (0,3,3): -2,
    (1,2,3): 1, (1,3,3): -2, (2,3,3): -2, (3,3,3): 8
}


def sparse_to_dense(sparse, h11=4):
    """Convert sparse dict to dense symmetric 3D array."""
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa


def transform_fluxes(K_old, M_old):
    """
    Transform K, M from old basis [3,4,5,8] to new basis [5,6,7,8].

    K is covariant → K_new = T^{-1} @ K_old
    M is contravariant → M_new = T^T @ M_old
    """
    K_new = T_inv @ K_old
    M_new = T.T @ M_old
    return K_new.astype(int), M_new.astype(int)


def compute_physics(kappa, K, M):
    """Compute p and e^{K0} from intersection numbers and fluxes."""
    N = np.einsum('abc,c->ab', kappa, M)
    p = np.linalg.solve(N, K)
    kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
    eK0 = 1.0 / ((4.0/3.0) * kappa_p3)
    return p, eK0


def main():
    print("=" * 70)
    print("Transform McAllister K, M to latest CYTools basis")
    print("=" * 70)

    # McAllister's fluxes in 2021 basis [3,4,5,8]
    K_old = np.array([-3, -5, 8, 6])
    M_old = np.array([10, 11, -11, -5])

    print(f"\nOld basis [3,4,5,8]:")
    print(f"  K_old = {K_old}")
    print(f"  M_old = {M_old}")

    # Transform to new basis
    K_new, M_new = transform_fluxes(K_old, M_old)

    print(f"\nNew basis [5,6,7,8]:")
    print(f"  K_new = {K_new}")
    print(f"  M_new = {M_new}")

    print(f"\nTransformation matrix T:")
    print(T)
    print(f"det(T) = {np.linalg.det(T):.0f}")

    print(f"\nTransformation rules:")
    print(f"  K_new = T^{{-1}} @ K_old  (covariant)")
    print(f"  M_new = T^T @ M_old      (contravariant)")

    # Verify physics is preserved
    print("\n" + "=" * 70)
    print("Verifying physics invariance")
    print("=" * 70)

    kappa_old = sparse_to_dense(KAPPA_OLD_SPARSE)
    kappa_new = sparse_to_dense(KAPPA_NEW_SPARSE)

    p_old, eK0_old = compute_physics(kappa_old, K_old, M_old)
    p_new, eK0_new = compute_physics(kappa_new, K_new, M_new)

    print(f"\np (old basis): {p_old}")
    print(f"p (new basis): {p_new}")

    # Verify p transforms correctly
    p_new_from_old = T.T @ p_old
    print(f"p transformed via T^T: {p_new_from_old}")
    print(f"p transformation MATCH: {np.allclose(p_new, p_new_from_old)}")

    print(f"\ne^{{K0}} (old basis): {eK0_old:.6f}")
    print(f"e^{{K0}} (new basis): {eK0_new:.6f}")
    print(f"e^{{K0}} MATCH: {np.isclose(eK0_old, eK0_new)}")

    if np.isclose(eK0_old, eK0_new):
        print("\n" + "=" * 70)
        print("SUCCESS: Physics preserved under basis transformation")
        print("=" * 70)
        print(f"\nUse these values with latest CYTools (basis [5,6,7,8]):")
        print(f"  K = {list(K_new)}")
        print(f"  M = {list(M_new)}")
    else:
        print("\nERROR: e^{K0} mismatch - transformation failed!")


if __name__ == "__main__":
    main()
