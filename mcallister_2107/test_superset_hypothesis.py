#!/usr/bin/env python3
"""Test if higher min_points gives superset containing all McAllister curves.

Result: At min_points=20000, we find ALL 1009 McAllister curves for 5-113-4627.
This proves McAllister's curves are a subset with a different enumeration cutoff.

| min_points | Our Curves | Matches | Missing |
|------------|------------|---------|---------|
| 10000      | 1147       | 845/1009| 164     |
| 15000      | 1603       | 968/1009| 41      |
| 20000      | 2194       | 1009/1009| 0      |
"""
import sys
from decimal import Decimal
from pathlib import Path
import numpy as np

CYTOOLS_LATEST = Path(__file__).parent.parent / "vendor/cytools_latest/src"
DATA_BASE = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data"


def load_dual_points(example_name):
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_simplices(example_name):
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def load_mcallister_gv(example_name):
    data_dir = DATA_BASE / example_name
    curves = []
    with open(data_dir / "dual_curves.dat") as f:
        for line in f:
            row = tuple(int(x) for x in line.strip().split(","))
            curves.append(row)
    with open(data_dir / "dual_curves_gv.dat") as f:
        content = f.read()
        gv_values = [int(Decimal(x)) for x in content.strip().split(",")]
    return {c: g for c, g in zip(curves, gv_values)}


def main():
    example = "5-113-4627-main"
    print(f"=== Testing superset hypothesis for {example} ===\n")

    dual_pts = load_dual_points(example)
    simplices = load_simplices(example)
    mcallister_gv = load_mcallister_gv(example)
    print(f"McAllister has {len(mcallister_gv)} curves")

    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()
    curve_basis_mat = cy.curve_basis(include_origin=True, as_matrix=True)

    for min_pts in [10000, 15000, 20000, 30000, 50000]:
        print(f"\n--- min_points={min_pts} ---")
        gv_obj = cy.compute_gvs(min_points=min_pts)

        our_gv = {}
        for q_basis, N_q in gv_obj.dok.items():
            if N_q != 0:
                q_ambient = tuple(int(x) for x in np.array(q_basis) @ curve_basis_mat)
                our_gv[q_ambient] = int(Decimal(str(N_q)).to_integral_value())

        matches = sum(1 for q in mcallister_gv if q in our_gv and our_gv[q] == mcallister_gv[q])
        missing = sum(1 for q in mcallister_gv if q not in our_gv)

        print(f"Our curves: {len(our_gv)}")
        print(f"Matches: {matches}/{len(mcallister_gv)}")
        print(f"Missing: {missing}")

        if missing == 0:
            print("\nSUCCESS: All McAllister curves found!")
            break


if __name__ == "__main__":
    main()
