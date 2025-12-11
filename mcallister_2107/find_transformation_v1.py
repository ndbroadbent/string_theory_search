#!/usr/bin/env python3
"""
Find the GL(4,Z) transformation between CYTools 2021 and latest bases.

2021 basis: [3,4,5,8]
Latest basis: [5,6,7,8]

We need T such that κ'_abc = T_a^d T_b^e T_c^f κ_def
"""
import numpy as np
from itertools import permutations, product

# Intersection numbers from CYTools 2021 (basis [3,4,5,8])
kappa_2021_sparse = {
    (0,0,0): 1, (0,0,1): -1, (0,0,2): -1, (0,1,1): 1, (0,1,2): 1, (0,2,2): 1,
    (1,1,1): -1, (1,1,2): -1, (1,2,2): -1, (1,2,3): 1, (1,3,3): -2,
    (2,2,2): -1, (2,3,3): -2, (3,3,3): 8
}

# Intersection numbers from CYTools latest (basis [5,6,7,8])
kappa_latest_sparse = {
    (0,0,0): -1, (0,2,3): 1, (0,3,3): -2,
    (1,2,3): 1, (1,3,3): -2, (2,3,3): -2, (3,3,3): 8
}

def sparse_to_dense(sparse, h11=4):
    """Convert sparse dict to dense symmetric 3D array."""
    kappa = np.zeros((h11, h11, h11))
    for (i,j,k), val in sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa

kappa_2021 = sparse_to_dense(kappa_2021_sparse)
kappa_latest = sparse_to_dense(kappa_latest_sparse)

print("Searching for transformation T...")
print("κ'_abc = T_a^d T_b^e T_c^f κ_def")

# McAllister fluxes in 2021 basis
K_2021 = np.array([-3, -5, 8, 6])
M_2021 = np.array([10, 11, -11, -5])

def transform_kappa(T, kappa):
    """Transform intersection numbers: κ' = T κ T^T (tensor contraction)."""
    h11 = 4
    kappa_new = np.zeros((h11, h11, h11))
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                for d in range(h11):
                    for e in range(h11):
                        for f in range(h11):
                            kappa_new[a,b,c] += T[a,d] * T[b,e] * T[c,f] * kappa[d,e,f]
    return kappa_new

def search_gl4z():
    """Search for T in GL(4,Z) with small entries."""
    best_T = None
    best_error = float('inf')

    # First try permutation matrices with signs
    print("\nTrying signed permutation matrices...")
    for perm in permutations(range(4)):
        for signs in product([-1, 1], repeat=4):
            T = np.zeros((4, 4))
            for i, (j, s) in enumerate(zip(perm, signs)):
                T[i, j] = s

            kappa_trans = transform_kappa(T, kappa_2021)
            error = np.sum((kappa_trans - kappa_latest)**2)

            if error < best_error:
                best_error = error
                best_T = T.copy()
                if error < 1e-10:
                    return T, error

    print(f"Best signed permutation error: {best_error}")

    # Try more general GL(4,Z) with entries in {-1,0,1}
    print("\nTrying general GL(4,Z) with entries in {-1,0,1}...")
    entries = [-1, 0, 1]

    count = 0
    for t in product(entries, repeat=16):
        T = np.array(t).reshape(4, 4)

        # Skip singular matrices
        det = np.linalg.det(T)
        if abs(det) < 0.5:  # det should be ±1 for GL(4,Z)
            continue

        count += 1
        kappa_trans = transform_kappa(T, kappa_2021)
        error = np.sum((kappa_trans - kappa_latest)**2)

        if error < best_error:
            best_error = error
            best_T = T.copy()
            print(f"  New best: error={error:.2f}, det={det:.0f}")
            if error < 1e-10:
                return T, error

    print(f"Checked {count} matrices with det=±1")
    print(f"Best error: {best_error}")

    return best_T, best_error

T, error = search_gl4z()

print("\n" + "="*60)
if error < 1e-10:
    print("FOUND EXACT TRANSFORMATION!")
else:
    print(f"Best transformation (error={error}):")
print("="*60)
print(f"\nT =\n{T.astype(int)}")
print(f"det(T) = {np.linalg.det(T):.0f}")

# Transform fluxes
# K is covariant (sits on RHS of N·p = K), transforms with T^{-1}
# M is contravariant (same as p), transforms with T^T
T_inv = np.linalg.inv(T)
K_latest = (T_inv @ K_2021).astype(int)
M_latest = (T.T @ M_2021).astype(int)

print(f"\nFlux transformation:")
print(f"  K transforms with T^{{-1}} (covariant)")
print(f"  M transforms with T^T (contravariant)")
print(f"  K (2021 basis [3,4,5,8]): {K_2021}")
print(f"  K (latest basis [5,6,7,8]): {K_latest}")
print(f"  M (2021 basis): {M_2021}")
print(f"  M (latest basis): {M_latest}")

# Verify by computing p in both bases
print("\n" + "="*60)
print("Verifying physics...")
print("="*60)

def compute_p(kappa, M, K):
    h11 = 4
    N = np.zeros((h11, h11))
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                N[a, b] += kappa[a, b, c] * M[c]
    return np.linalg.solve(N, K)

p_2021 = compute_p(kappa_2021, M_2021, K_2021)
print(f"\np in 2021 basis: {p_2021}")

if error < 1e-10:
    p_latest = compute_p(kappa_latest, M_latest, K_latest)
    print(f"p in latest basis: {p_latest}")

    # p is contravariant, transforms with T^T
    p_transformed = T.T @ p_2021
    print(f"p_2021 transformed by T^T: {p_transformed}")
    print(f"Match: {np.allclose(p_latest, p_transformed)}")

    # Compute e^{K0} in both bases - should be identical (scalar invariant)
    def compute_eK0(kappa, p):
        kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
        return 1.0 / ((4.0/3.0) * kappa_p3)

    eK0_2021 = compute_eK0(kappa_2021, p_2021)
    eK0_latest = compute_eK0(kappa_latest, p_latest)
    print(f"\ne^{{K0}} (2021 basis): {eK0_2021:.6f}")
    print(f"e^{{K0}} (latest basis): {eK0_latest:.6f}")
    print(f"e^{{K0}} MATCH: {np.isclose(eK0_2021, eK0_latest)}")
