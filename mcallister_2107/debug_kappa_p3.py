#!/usr/bin/env python3
"""
Debug κ̃_abc p^a p^b p^c computation.

The formula is e^{K₀} = (4/3) × (κ̃_abc p^a p^b p^c)^{-1}
McAllister's value: e^{K₀} ≈ 0.2362 → κ_p3 ≈ 5.6459

We're getting κ_p3 ≈ 3.20, which is 1.77x too small.

Check if the issue is:
1. Basis mismatch between κ and p
2. Different intersection number conventions
3. Using wrong CYTools version
"""

import sys
from pathlib import Path
import numpy as np
from fractions import Fraction

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
CYTOOLS_LATEST = ROOT_DIR / "vendor/cytools_latest/src"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"


def load_dual_points(example_name: str) -> np.ndarray:
    lines = (DATA_BASE / example_name / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_simplices(example_name: str) -> list:
    lines = (DATA_BASE / example_name / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def load_model(example_name: str) -> dict:
    data_dir = DATA_BASE / example_name
    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(",")])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(",")])
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    return {"K": K, "M": M, "g_s": g_s}


def get_2021_kappa_and_p(dual_pts, simplices, model):
    """Get kappa tensor and p using CYTools 2021."""
    # Clear cytools modules
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_2021))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    # Get kappa tensor
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = np.zeros((h11, h11, h11))
    for row in kappa_sparse:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        val = row[3]
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            kappa[perm] = val

    divisor_basis = list(cy.divisor_basis())

    # Compute p from N @ p = K where N_ab = κ_abc M^c
    N = np.einsum('abc,c->ab', kappa, model['M'])
    p = np.linalg.solve(N, model['K'])

    sys.path.remove(str(CYTOOLS_2021))

    return kappa, p, divisor_basis, N


def get_latest_kappa_and_p(dual_pts, simplices, model):
    """Get kappa tensor and p using CYTools Latest (with basis transformation)."""
    # Clear cytools modules
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    # Get kappa tensor
    kappa_dict = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_dict.items():
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            kappa[perm] = val

    divisor_basis = list(cy.divisor_basis())

    sys.path.remove(str(CYTOOLS_LATEST))

    return kappa, divisor_basis


def main():
    example = "4-214-647"
    print("=" * 70)
    print(f"DEBUGGING κ̃_abc p^a p^b p^c for {example}")
    print("=" * 70)

    dual_pts = load_dual_points(example)
    simplices = load_simplices(example)
    model = load_model(example)

    print(f"\nMcAllister's flux vectors:")
    print(f"  K = {model['K']}")
    print(f"  M = {model['M']}")

    # McAllister's expected p (eq. 6.56)
    p_expected = np.array([293/110, 163/110, 163/110, 13/22])
    print(f"\nMcAllister's p (eq. 6.56):")
    print(f"  p = (293/110, 163/110, 163/110, 13/22)")
    print(f"    = {p_expected}")

    # Compute with CYTools 2021
    print("\n" + "=" * 50)
    print("Using CYTools 2021 (McAllister's version)")
    print("=" * 50)

    kappa_2021, p_2021, basis_2021, N_2021 = get_2021_kappa_and_p(dual_pts, simplices, model)

    print(f"\nDivisor basis: {basis_2021}")
    print(f"Computed p: {p_2021}")
    print(f"p error vs expected: {np.max(np.abs(p_2021 - p_expected)):.2e}")

    # Compute κ_abc p^a p^b p^c
    kappa_p3_2021 = np.einsum('abc,a,b,c->', kappa_2021, p_2021, p_2021, p_2021)
    eK0_2021 = (4/3) / kappa_p3_2021

    print(f"\nκ_abc p^a p^b p^c = {kappa_p3_2021:.6f}")
    print(f"e^{{K₀}} = (4/3) / κ_p3 = {eK0_2021:.6f}")
    print(f"Expected e^{{K₀}} ≈ 0.2362")

    # Show some κ values
    print(f"\nSample κ values (2021):")
    for i in range(4):
        for j in range(i, 4):
            for k in range(j, 4):
                val = kappa_2021[i, j, k]
                if val != 0:
                    print(f"  κ_{{{i}{j}{k}}} = {int(val)}")

    # Now compute with CYTools Latest
    print("\n" + "=" * 50)
    print("Using CYTools Latest")
    print("=" * 50)

    kappa_latest, basis_latest = get_latest_kappa_and_p(dual_pts, simplices, model)

    print(f"\nDivisor basis: {basis_latest}")

    # Need to transform K and M to latest basis
    # T transforms 2021 → latest, so:
    # K_new = T⁻¹ @ K_old  (covariant)
    # M_new = T.T @ M_old  (contravariant)

    # The transformation matrix from LATEST_CYTOOLS_CONVERSION_RESULT.md
    T = np.array([
        [-1,  1,  0,  0],  # D3 = -D5 + D6
        [ 1, -1,  1,  0],  # D4 = D5 - D6 + D7
        [ 1,  0,  0,  0],  # D5 = D5
        [ 0,  0,  0,  1],  # D8 = D8
    ])

    K_latest = np.linalg.solve(T, model['K'])  # K_new = T⁻¹ @ K_old
    M_latest = T.T @ model['M']  # M_new = T.T @ M_old

    print(f"Transformed K: {K_latest}")
    print(f"Transformed M: {M_latest}")

    # Compute p in latest basis
    N_latest = np.einsum('abc,c->ab', kappa_latest, M_latest)
    p_latest = np.linalg.solve(N_latest, K_latest)

    print(f"Computed p (latest): {p_latest}")

    # κ_abc p^a p^b p^c in latest basis
    kappa_p3_latest = np.einsum('abc,a,b,c->', kappa_latest, p_latest, p_latest, p_latest)
    eK0_latest = (4/3) / kappa_p3_latest

    print(f"\nκ_abc p^a p^b p^c = {kappa_p3_latest:.6f}")
    print(f"e^{{K₀}} = (4/3) / κ_p3 = {eK0_latest:.6f}")

    # Show some κ values for latest
    print(f"\nSample κ values (latest):")
    for i in range(4):
        for j in range(i, 4):
            for k in range(j, 4):
                val = kappa_latest[i, j, k]
                if val != 0:
                    print(f"  κ_{{{i}{j}{k}}} = {int(val)}")

    # Verify κ transformation
    print("\n" + "=" * 50)
    print("Verifying κ tensor transformation")
    print("=" * 50)

    # Under basis change D_new = T @ D_old, κ transforms as:
    # κ_new[i,j,k] = T^{-1}[i,a] T^{-1}[j,b] T^{-1}[k,c] κ_old[a,b,c]
    T_inv = np.linalg.inv(T)
    kappa_2021_transformed = np.einsum('ia,jb,kc,abc->ijk', T_inv, T_inv, T_inv, kappa_2021)

    print(f"\nTransformed κ from 2021 to latest:")
    for i in range(4):
        for j in range(i, 4):
            for k in range(j, 4):
                val = kappa_2021_transformed[i, j, k]
                if abs(val) > 1e-6:
                    print(f"  κ_{{{i}{j}{k}}} = {val:.1f}")

    print(f"\nDifference kappa_latest - kappa_2021_transformed:")
    diff = np.abs(kappa_latest - kappa_2021_transformed)
    print(f"  Max difference: {np.max(diff):.6f}")

    # The scalar κ_p3 should be basis-independent
    # But we need to use the transformed p
    p_transformed = T.T @ p_2021
    print(f"\np transformed via T.T: {p_transformed}")
    print(f"Direct latest p: {p_latest}")
    print(f"Difference: {np.max(np.abs(p_transformed - p_latest)):.6e}")

    # Final verification
    kappa_p3_check = np.einsum('abc,a,b,c->', kappa_latest, p_transformed, p_transformed, p_transformed)
    print(f"\nκ_p3 using latest κ and transformed p: {kappa_p3_check:.6f}")
    print(f"κ_p3 using 2021 κ and 2021 p: {kappa_p3_2021:.6f}")

    # What about the EXPECTED value?
    print("\n" + "=" * 50)
    print("Analysis of expected value")
    print("=" * 50)

    eK0_expected = 0.2362
    kappa_p3_expected = (4/3) / eK0_expected

    print(f"\nExpected e^{{K₀}} ≈ {eK0_expected}")
    print(f"Required κ_p3 = {kappa_p3_expected:.4f}")
    print(f"Computed κ_p3 = {kappa_p3_2021:.4f}")
    print(f"Ratio expected/computed = {kappa_p3_expected/kappa_p3_2021:.4f}")

    # What if there's a factor of 2 somewhere?
    print(f"\nPossible correction factors:")
    ratio = kappa_p3_expected / kappa_p3_2021
    print(f"  Ratio = {ratio:.4f}")
    print(f"  1.5 × ratio = {1.5 * ratio:.4f}")
    print(f"  ratio / 1.5 = {ratio / 1.5:.4f}")
    print(f"  sqrt(ratio) = {np.sqrt(ratio):.4f}")


if __name__ == "__main__":
    main()
