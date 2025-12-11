#!/usr/bin/env python3
"""
Compute the exact basis transformation using GLSM linear relations.

The GLSM linear relations tell us how ambient toric divisors are related.
From this we can derive the exact transformation between any two bases.
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))
from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"

# Load geometry
lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
dual_points = np.array([[int(x) for x in line.split(',')] for line in lines])

lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
simplices = [[int(x) for x in line.split(',')] for line in lines]

# Setup CYTools
poly = Polytope(dual_points)
tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
cy = tri.get_cy()

print("Polytope points:")
pts = poly.points()
print(f"  {pts.shape[0]} points in {pts.shape[1]}D")

print("\nGLSM linear relations:")
linrels = poly.glsm_linear_relations()
print(f"  Shape: {linrels.shape}")
print(f"  Relations:\n{linrels}")

print("\nGLSM charge matrix:")
charges = poly.glsm_charge_matrix()
print(f"  Shape: {charges.shape}")
print(f"  Charges:\n{charges}")

# The linear relations tell us: sum_i L_ij D_i = 0 for each row j
# This means we can express any divisor as a linear combination of basis divisors

basis_2021 = [3, 4, 5, 8]  # What CYTools 2021 chose
basis_latest = [5, 6, 7, 8]  # What CYTools latest chooses

print(f"\nBasis 2021: {basis_2021}")
print(f"Basis latest: {basis_latest}")

# To find how D_5, D_6, D_7, D_8 relate to D_3, D_4, D_5, D_8:
# We need to express D_6, D_7 in terms of D_3, D_4, D_5, D_8 using linear relations

# The linear relations matrix L satisfies L @ D = 0
# If we have basis B, we can express any D_i = sum_j c_ij D_{B_j}

# Let's extract the relevant submatrices
n_pts = linrels.shape[1]  # Number of divisors (polytope points)
n_rels = linrels.shape[0]  # Number of linear relations

print(f"\n{n_pts} divisors, {n_rels} linear relations")
print(f"h11 = {n_pts - n_rels - 1} (should be 4)")

# For each non-basis divisor, solve for its expression in terms of basis
def express_in_basis(div_idx, basis, linrels):
    """
    Express divisor D_{div_idx} as linear combination of basis divisors.
    Uses the GLSM linear relations.
    """
    n_pts = linrels.shape[1]

    if div_idx in basis:
        # It's already a basis element
        result = np.zeros(len(basis))
        result[basis.index(div_idx)] = 1
        return result

    # The linear relations give us: L @ D = 0
    # Rearrange to express D_{div_idx} in terms of others

    # Extract column for div_idx
    col = linrels[:, div_idx]

    # Find a relation where this divisor appears
    for rel_idx, coeff in enumerate(col):
        if abs(coeff) > 1e-10:
            # This relation involves D_{div_idx}
            # Relation: sum_i L[rel_idx, i] * D_i = 0
            # So: coeff * D_{div_idx} = -sum_{i != div_idx} L[rel_idx, i] * D_i

            relation = linrels[rel_idx]
            # Express D_{div_idx} = (-1/coeff) * sum_{i != div_idx} L[rel_idx, i] * D_i

            # Now we need to express this in terms of basis
            result = np.zeros(len(basis))
            for i, c in enumerate(relation):
                if i == div_idx:
                    continue
                if i in basis:
                    result[basis.index(i)] -= c / coeff
                # If i is not in basis, we'd need to recurse...

            return result

    return None

print("\n" + "="*60)
print("Expressing latest basis divisors in terms of 2021 basis")
print("="*60)

# Build transformation matrix T
# T[i,j] = coefficient of D_{basis_2021[j]} in expression for D_{basis_latest[i]}
T = np.zeros((4, 4))

for i, div_latest in enumerate(basis_latest):
    expr = express_in_basis(div_latest, basis_2021, linrels)
    if expr is not None:
        T[i, :] = expr
        print(f"D_{div_latest} = {' + '.join(f'{c:.2f}*D_{basis_2021[j]}' for j, c in enumerate(expr) if abs(c) > 1e-10)}")
    else:
        print(f"D_{div_latest}: could not express in basis")

print(f"\nTransformation matrix T (latest = T @ 2021):")
print(T)
print(f"det(T) = {np.linalg.det(T)}")
