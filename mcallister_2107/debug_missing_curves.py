#!/usr/bin/env python3
"""Debug missing curves in 5-113-4627."""
import sys
import numpy as np
from pathlib import Path
from decimal import Decimal

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
    # Focus on 5-113-4627-main
    example = "5-113-4627-main"
    print(f"=== Analyzing {example} ===\n")

    dual_pts = load_dual_points(example)
    simplices = load_simplices(example)
    mcallister_gv = load_mcallister_gv(example)

    # CYTools latest
    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    curve_basis_mat = cy.curve_basis(include_origin=True, as_matrix=True)
    print(f"curve_basis_mat:\n{curve_basis_mat}\n")

    # Compute GV
    gv_obj = cy.compute_gvs(min_points=10000)

    # Convert and compare
    our_gv = {}
    for q_basis, N_q in gv_obj.dok.items():
        if N_q != 0:
            q_ambient = tuple(int(x) for x in np.array(q_basis) @ curve_basis_mat)
            our_gv[q_ambient] = int(Decimal(str(N_q)).to_integral_value())

    # Find missing curves
    missing = []
    for q_mcallister, gv_expected in mcallister_gv.items():
        if q_mcallister not in our_gv:
            missing.append((q_mcallister, gv_expected))

    print(f"Missing curves: {len(missing)}")
    print("\nFirst 5 missing curves:")
    for q, gv in missing[:5]:
        print(f"  {q} -> GV={gv}")

    # Check if any of our curves are close to the missing ones
    print("\n\nLooking for similar curves in our data...")
    for q_missing, gv_missing in missing[:3]:
        print(f"\nMissing: {q_missing}")
        # Look for curves with same GV value
        matches_gv = [q for q, gv in our_gv.items() if gv == gv_missing]
        if matches_gv:
            print(f"  Curves with same GV ({gv_missing}):")
            for m in matches_gv[:3]:
                diff = tuple(a - b for a, b in zip(m, q_missing))
                print(f"    {m}  diff={diff}")

    # The key insight: curve_basis_mat differs between CYTools versions!
    # McAllister used 2021's curve_basis_mat, we're using latest's.
    # The GV computation gives us curves in basis coords (h11-dim),
    # then we multiply by curve_basis_mat to get ambient coords.
    #
    # If curve_basis_mat differs, the SAME basis-coords curve maps to
    # DIFFERENT ambient coords!
    #
    # Let's convert McAllister's ambient coords back to basis coords
    # and compare there.

    print("\n\n=== Trying to invert curve_basis_mat ===")
    print(f"curve_basis_mat shape: {curve_basis_mat.shape}")

    # curve_basis_mat is (h11, n_divisors) = (5, 10)
    # It's not square, so we need pseudo-inverse
    # q_ambient = q_basis @ curve_basis_mat
    # q_basis = q_ambient @ pinv(curve_basis_mat)

    pinv = np.linalg.pinv(curve_basis_mat)
    print(f"pinv shape: {pinv.shape}")

    # Convert McAllister's curves to basis coords
    mcallister_basis = {}
    for q_ambient, gv in mcallister_gv.items():
        q_basis = tuple(int(round(x)) for x in np.array(q_ambient) @ pinv)
        mcallister_basis[q_basis] = gv

    # Also get our basis coords
    our_basis = {}
    for q_basis, N_q in gv_obj.dok.items():
        if N_q != 0:
            our_basis[tuple(q_basis)] = int(Decimal(str(N_q)).to_integral_value())

    # Compare in basis coords
    matches_basis = 0
    missing_basis = 0
    for q_mcallister_basis, gv_expected in mcallister_basis.items():
        if q_mcallister_basis in our_basis:
            if our_basis[q_mcallister_basis] == gv_expected:
                matches_basis += 1
        else:
            missing_basis += 1

    print(f"\n=== Comparison in BASIS coords ===")
    print(f"McAllister curves (converted to basis): {len(mcallister_basis)}")
    print(f"Our curves (in basis): {len(our_basis)}")
    print(f"Matches: {matches_basis}")
    print(f"Missing: {missing_basis}")

    # Show some examples of missing in basis coords
    print("\nFirst 5 missing (in basis coords):")
    count = 0
    for q_basis, gv in mcallister_basis.items():
        if q_basis not in our_basis:
            print(f"  {q_basis} -> GV={gv}")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
