#!/usr/bin/env python3
"""
Extract dual intersection numbers from CYTools for McAllister 4-214-647.

This provides ground truth for testing the Rust implementation at each stage.
"""

import numpy as np
from pathlib import Path
import json

from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def main():
    # Load dual polytope points
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
    dual_points = np.array([[int(x) for x in line.split(',')] for line in lines])

    # Load triangulation
    lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
    simplices = [[int(x) for x in line.split(',')] for line in lines]

    print(f"Dual points shape: {dual_points.shape}")
    print(f"Number of simplices: {len(simplices)}")

    # Create CYTools geometry
    poly = Polytope(dual_points)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    h11 = cy.h11()  # This is h21 of primal = 4
    print(f"h11 (dual) = {h11}")

    # Get divisor basis
    basis = list(cy.divisor_basis())
    print(f"Divisor basis: {basis}")

    # Get GLSM charge matrix
    glsm = cy.glsm_charge_matrix()
    print(f"GLSM shape: {glsm.shape}")
    print("GLSM rows:")
    for i, row in enumerate(glsm):
        print(f"  {i}: {list(row)}")

    # Get intersection numbers (NOT in basis)
    kappa_full = cy.intersection_numbers(in_basis=False)
    print(f"\nFull intersection numbers (point-indexed):")
    print(f"  Type: {type(kappa_full)}")

    # Handle dict format
    kappa_full_list = []
    for key, val in kappa_full.items():
        i, j, k = key
        kappa_full_list.append([i, j, k, val])
        if len(kappa_full_list) <= 20:
            print(f"  κ[{i},{j},{k}] = {val}")
    if len(kappa_full_list) > 20:
        print(f"  ... ({len(kappa_full_list) - 20} more)")

    # Get intersection numbers IN BASIS
    kappa_basis = cy.intersection_numbers(in_basis=True)
    print(f"\nBasis intersection numbers (basis-indexed):")
    print(f"  Type: {type(kappa_basis)}")

    kappa_basis_list = []
    for key, val in kappa_basis.items():
        i, j, k = key
        kappa_basis_list.append([i, j, k, val])
        print(f"  κ[{i},{j},{k}] = {val}")

    # Load flux vectors
    K = np.array([int(x) for x in (DATA_DIR / "K_vec.dat").read_text().strip().split(',')])
    M = np.array([int(x) for x in (DATA_DIR / "M_vec.dat").read_text().strip().split(',')])
    print(f"\nK = {K}")
    print(f"M = {M}")

    # Compute N matrix and flat direction
    kappa_tensor = np.zeros((h11, h11, h11))
    for row in kappa_basis_list:
        i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa_tensor[perm] = val

    N = np.zeros((h11, h11))
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                N[a, b] += kappa_tensor[a, b, c] * M[c]

    print(f"\nN matrix:")
    print(N)
    print(f"det(N) = {np.linalg.det(N)}")

    p = np.linalg.solve(N, K)
    print(f"\np = {p}")

    # Compute κ_abc p^a p^b p^c
    kappa_ppp = 0.0
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                kappa_ppp += kappa_tensor[a, b, c] * p[a] * p[b] * p[c]

    print(f"\nκ_abc p^a p^b p^c = {kappa_ppp}")

    eK0 = 1.0 / ((4.0/3.0) * kappa_ppp)
    print(f"e^{{K₀}} = {eK0}")
    print(f"Expected: 0.234393")

    # Now test with McAllister's basis [3,4,5,8]
    print("\n" + "="*60)
    print("With McAllister's basis [3, 4, 5, 8]:")
    print("="*60)

    mcallister_basis = [3, 4, 5, 8]
    cy.set_divisor_basis(mcallister_basis)
    kappa_mcallister = cy.intersection_numbers(in_basis=True)
    print(f"\nMcAllister basis intersection numbers:")
    kappa_mcallister_list = []
    for key, val in kappa_mcallister.items():
        i, j, k = key
        kappa_mcallister_list.append([i, j, k, val])
        print(f"  κ[{i},{j},{k}] = {val}")

    # Compute with McAllister's basis
    kappa_m_tensor = np.zeros((4, 4, 4))
    for row in kappa_mcallister_list:
        i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa_m_tensor[perm] = val

    N_m = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                N_m[a, b] += kappa_m_tensor[a, b, c] * M[c]

    print(f"\nN matrix (McAllister basis):")
    print(N_m)
    print(f"det(N) = {np.linalg.det(N_m)}")

    p_m = np.linalg.solve(N_m, K)
    print(f"\np = {p_m}")

    kappa_m_ppp = 0.0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                kappa_m_ppp += kappa_m_tensor[a, b, c] * p_m[a] * p_m[b] * p_m[c]

    print(f"\nκ_abc p^a p^b p^c = {kappa_m_ppp}")

    eK0_m = 1.0 / ((4.0/3.0) * kappa_m_ppp)
    print(f"e^{{K₀}} = {eK0_m}")
    print(f"Expected: 0.234393")

    # Export for Rust tests
    output = {
        "latest_cytools": {
            "basis": [int(b) for b in basis],
            "kappa_full": [[int(row[0]), int(row[1]), int(row[2]), int(row[3])] for row in kappa_full_list],
            "kappa_basis": [[int(row[0]), int(row[1]), int(row[2]), int(row[3])] for row in kappa_basis_list],
            "N_matrix": [[float(x) for x in row] for row in N.tolist()],
            "p": [float(x) for x in p.tolist()],
            "kappa_ppp": float(kappa_ppp),
            "eK0": float(eK0),
        },
        "mcallister_basis": {
            "basis": mcallister_basis,
            "kappa_basis": [[int(row[0]), int(row[1]), int(row[2]), int(row[3])] for row in kappa_mcallister_list],
            "N_matrix": [[float(x) for x in row] for row in N_m.tolist()],
            "p": [float(x) for x in p_m.tolist()],
            "kappa_ppp": float(kappa_m_ppp),
            "eK0": float(eK0_m),
        },
    }

    output_path = Path(__file__).parent / "dual_intersection_ground_truth.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
