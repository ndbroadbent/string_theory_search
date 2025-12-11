#!/usr/bin/env python3
"""
Solve for the basis transformation T using intersection number constraints.

Given κ_2021 and κ_latest, find T such that:
  κ_latest[a,b,c] = sum_{d,e,f} T[a,d] T[b,e] T[c,f] κ_2021[d,e,f]

This is a system of polynomial equations in T. Since T ∈ GL(4,Z),
we can solve it by setting up linear constraints.
"""
import numpy as np
from scipy.optimize import minimize
import sympy as sp

# Intersection numbers
kappa_2021_sparse = {
    (0,0,0): 1, (0,0,1): -1, (0,0,2): -1, (0,1,1): 1, (0,1,2): 1, (0,2,2): 1,
    (1,1,1): -1, (1,1,2): -1, (1,2,2): -1, (1,2,3): 1, (1,3,3): -2,
    (2,2,2): -1, (2,3,3): -2, (3,3,3): 8
}

kappa_latest_sparse = {
    (0,0,0): -1, (0,2,3): 1, (0,3,3): -2,
    (1,2,3): 1, (1,3,3): -2, (2,3,3): -2, (3,3,3): 8
}

def sparse_to_dense(sparse, h11=4):
    kappa = np.zeros((h11, h11, h11))
    for (i,j,k), val in sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa

kappa_2021 = sparse_to_dense(kappa_2021_sparse)
kappa_latest = sparse_to_dense(kappa_latest_sparse)

def transform_kappa(T, kappa):
    h11 = 4
    kappa_new = np.einsum('ad,be,cf,def->abc', T, T, T, kappa)
    return kappa_new

def objective(T_flat):
    T = T_flat.reshape(4, 4)
    kappa_trans = transform_kappa(T, kappa_2021)
    return np.sum((kappa_trans - kappa_latest)**2)

print("Solving for transformation T...")
print("Using numerical optimization to find T ∈ GL(4,R) first")

# Try multiple random starting points
best_T = None
best_error = float('inf')

np.random.seed(42)
for trial in range(100):
    # Random starting point near identity
    T0 = np.eye(4) + 0.5 * np.random.randn(4, 4)

    result = minimize(objective, T0.flatten(), method='L-BFGS-B')

    if result.fun < best_error:
        best_error = result.fun
        best_T = result.x.reshape(4, 4)

    if result.fun < 1e-10:
        print(f"  Trial {trial}: Found exact solution!")
        break

print(f"\nBest error: {best_error}")
print(f"Best T (continuous):\n{best_T}")
print(f"det(T) = {np.linalg.det(best_T)}")

# Round to nearest integers
T_int = np.round(best_T).astype(int)
print(f"\nRounded to integers:\n{T_int}")
print(f"det(T_int) = {np.linalg.det(T_int)}")

# Check if integer solution works
kappa_check = transform_kappa(T_int.astype(float), kappa_2021)
error_int = np.sum((kappa_check - kappa_latest)**2)
print(f"Error with integer T: {error_int}")

if error_int < 1e-10:
    print("\n" + "="*60)
    print("SUCCESS! Found integer transformation")
    print("="*60)

    K_2021 = np.array([-3, -5, 8, 6])
    M_2021 = np.array([10, 11, -11, -5])

    # K and M transform DIFFERENTLY:
    # - K is covariant (RHS of N·p = K) → transforms with T^{-1}
    # - M is contravariant (same as p) → transforms with T^T
    T_inv = np.linalg.inv(T_int.astype(float))
    K_latest = (T_inv @ K_2021)
    M_latest = (T_int.T @ M_2021)

    print(f"\nTransformation T:")
    print(T_int)

    print(f"\nFlux transformation:")
    print(f"  K transforms with T^{{-1}} (covariant)")
    print(f"  M transforms with T^T (contravariant)")
    print(f"  K (2021): {K_2021}  →  K (latest): {K_latest.astype(int)}")
    print(f"  M (2021): {M_2021}  →  M (latest): {M_latest.astype(int)}")

    # Verify physics
    def compute_p(kappa, M, K):
        h11 = 4
        N = np.zeros((h11, h11))
        for a in range(h11):
            for b in range(h11):
                for c in range(h11):
                    N[a, b] += kappa[a, b, c] * M[c]
        return np.linalg.solve(N, K)

    p_2021 = compute_p(kappa_2021, M_2021, K_2021)
    p_latest = compute_p(kappa_latest, M_latest, K_latest)

    print(f"\nFlat direction:")
    print(f"  p (2021 basis): {p_2021}")
    print(f"  p (latest basis): {p_latest}")

    # p is contravariant (same as M), transforms with T^T
    p_transformed = T_int.T @ p_2021
    print(f"  p_2021 transformed by T^T: {p_transformed}")
    print(f"  Match: {np.allclose(p_latest, p_transformed)}")

    # Compute e^{K0} in both bases - should be identical (scalar)
    def compute_eK0(kappa, p):
        kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
        return 1.0 / ((4.0/3.0) * kappa_p3)

    eK0_2021 = compute_eK0(kappa_2021, p_2021)
    eK0_latest = compute_eK0(kappa_latest, p_latest)

    print(f"\ne^{{K0}}:")
    print(f"  2021 basis: {eK0_2021}")
    print(f"  latest basis: {eK0_latest}")
    print(f"  Match: {np.isclose(eK0_2021, eK0_latest)}")
