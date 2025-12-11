#!/usr/bin/env python3
"""Get intersection numbers from CYTools latest."""
import numpy as np
from pathlib import Path
import sys

# Use latest CYTools from vendor
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"

# Load geometry
lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
dual_points = np.array([[int(x) for x in line.split(',')] for line in lines])

lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
simplices = [[int(x) for x in line.split(',')] for line in lines]

# Setup CYTools
poly = Polytope(dual_points)
tri = poly.triangulate(simplices=simplices)
cy = tri.get_cy()

print(f"basis_latest = {list(cy.divisor_basis())}")
print(f"h11 = {cy.h11()}, h21 = {cy.h21()}")

# Get kappa
kappa_dict = cy.intersection_numbers(in_basis=True)
print(f"kappa type: {type(kappa_dict)}")

print("kappa_latest = {")
if isinstance(kappa_dict, dict):
    for (i,j,k), val in sorted(kappa_dict.items()):
        print(f"    ({i},{j},{k}): {int(val)},")
else:
    # Array format from older versions
    h11 = cy.h11()
    for row in kappa_dict:
        i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
        print(f"    ({i},{j},{k}): {int(val)},")
print("}")
