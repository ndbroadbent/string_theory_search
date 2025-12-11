#!/usr/bin/env python3
"""Get intersection numbers from CYTools 2021."""
import numpy as np
from pathlib import Path
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

print(f"basis_2021 = {list(cy.divisor_basis())}")

# Get kappa
kappa_result = cy.intersection_numbers(in_basis=True)
h11 = cy.h11()
kappa = np.zeros((h11, h11, h11))
for row in kappa_result:
    i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
    for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
        kappa[perm] = val

print("kappa_2021 = {")
for i in range(h11):
    for j in range(i, h11):
        for k in range(j, h11):
            if abs(kappa[i,j,k]) > 1e-10:
                print(f"    ({i},{j},{k}): {int(kappa[i,j,k])},")
print("}")
