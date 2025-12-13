#!/usr/bin/env python3
"""
Debug the curve basis mismatch and verify the transformation fixes it.

The problem: q·p should give exponents like 32/110, but we get ~0.009.

Root cause: p is in 2021's divisor basis [3,4,5,8], but curves from
CYTools latest are in latest's curve basis [5,6,7,8].

Solution: Transform p from 2021's basis to latest's basis using T.T,
where T is the basis transformation matrix from LATEST_CYTOOLS_CONVERSION_RESULT.md.
"""

import sys
from pathlib import Path
from decimal import Decimal
import numpy as np

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
CYTOOLS_LATEST = ROOT_DIR / "vendor/cytools_latest/src"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

# Transformation matrix from 2021 basis [3,4,5,8] to latest basis [5,6,7,8]
# From LATEST_CYTOOLS_CONVERSION_RESULT.md
T_2021_TO_LATEST = np.array([
    [-1,  1,  0,  0],  # D3 = -D5 + D6
    [ 1, -1,  1,  0],  # D4 = D5 - D6 + D7
    [ 1,  0,  0,  0],  # D5 = D5
    [ 0,  0,  0,  1],  # D8 = D8
])


def load_dual_points(example_name: str) -> np.ndarray:
    lines = (DATA_BASE / example_name / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_simplices(example_name: str) -> list:
    lines = (DATA_BASE / example_name / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def load_mcallister_curves(example_name: str) -> tuple:
    """Load McAllister's curves and GVs (ambient coordinates)."""
    data_dir = DATA_BASE / example_name

    curves = []
    with open(data_dir / "dual_curves.dat") as f:
        for line in f:
            curves.append(np.array([int(x) for x in line.strip().split(",")]))

    with open(data_dir / "dual_curves_gv.dat") as f:
        gv_values = [int(Decimal(x)) for x in f.read().strip().split(",")]

    return curves, gv_values


def load_model(example_name: str) -> dict:
    data_dir = DATA_BASE / example_name
    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(",")])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(",")])
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    return {"K": K, "M": M, "g_s": g_s}


def get_2021_info(dual_pts, simplices):
    """Get kappa from CYTools 2021."""
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
    sys.path.remove(str(CYTOOLS_2021))

    return kappa, divisor_basis


def get_latest_curve_basis(dual_pts, simplices):
    """Get curve basis matrix from CYTools latest."""
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    divisor_basis = list(cy.divisor_basis())
    curve_basis_mat = cy.curve_basis(include_origin=True, as_matrix=True)

    sys.path.remove(str(CYTOOLS_LATEST))

    return divisor_basis, curve_basis_mat


def main():
    example = "4-214-647"
    print("=" * 70)
    print(f"VERIFYING CURVE BASIS TRANSFORMATION - {example}")
    print("=" * 70)

    dual_pts = load_dual_points(example)
    simplices = load_simplices(example)
    model = load_model(example)

    print(f"\nK = {model['K']} (in 2021 basis)")
    print(f"M = {model['M']} (in 2021 basis)")
    print(f"g_s = {model['g_s']}")

    # Get 2021 info
    print("\n1. CYTools 2021:")
    kappa, basis_2021 = get_2021_info(dual_pts, simplices)
    print(f"   Divisor basis: {basis_2021}")

    # Compute p in 2021 basis
    N = np.einsum('abc,c->ab', kappa, model['M'])
    p_2021 = np.linalg.solve(N, model['K'])
    print(f"   p (2021 basis) = {p_2021}")
    print(f"   Expected: (293/110, 163/110, 163/110, 13/22) = (2.664, 1.482, 1.482, 0.591)")

    # Get latest curve basis
    print("\n2. CYTools Latest:")
    basis_latest, curve_basis_mat = get_latest_curve_basis(dual_pts, simplices)
    print(f"   Divisor basis: {basis_latest}")
    print(f"   Curve basis mat shape: {curve_basis_mat.shape}")

    # Use the known transformation matrix
    print("\n3. Transformation T (from LATEST_CYTOOLS_CONVERSION_RESULT.md):")
    T = T_2021_TO_LATEST
    print(f"   T =")
    for row in T:
        print(f"     {row}")
    print(f"   det(T) = {int(round(np.linalg.det(T)))}")

    # Transform p to latest's basis
    # p is contravariant: p_new = T.T @ p_old
    p_latest = T.T @ p_2021
    print(f"\n   p (2021 basis) = {p_2021}")
    print(f"   p (latest basis) = T.T @ p_2021 = {p_latest}")

    # Also transform M for M·q
    # M is contravariant: M_new = T.T @ M_old
    M_latest = T.T @ model['M']
    print(f"\n   M (2021 basis) = {model['M']}")
    print(f"   M (latest basis) = T.T @ M = {M_latest}")

    # Verify against LATEST_CYTOOLS_CONVERSION_RESULT.md values
    print("\n   Verification against LATEST_CYTOOLS_CONVERSION_RESULT.md:")
    print(f"   Expected M_latest = [-10, -1, 11, -5]")
    M_expected = np.array([-10, -1, 11, -5])
    if np.allclose(M_latest, M_expected):
        print(f"   ✓ M transformation matches!")
    else:
        print(f"   ✗ M transformation MISMATCH!")

    # Load McAllister's curves (ambient coords)
    print("\n4. Loading McAllister's curves (ambient coords)...")
    curves_ambient, gv_values = load_mcallister_curves(example)
    print(f"   {len(curves_ambient)} curves")

    # Transform curves to latest's internal basis
    curve_basis_pinv = np.linalg.pinv(curve_basis_mat)

    # Compute q·p for each curve using p in latest's basis
    print(f"\n5. Computing q·p with p in latest's basis:")
    print(f"   (Target: leading terms should have q·p ≈ 32/110 = 0.2909)")

    qp_list = []
    for q_ambient, gv in zip(curves_ambient, gv_values):
        # Transform curve from ambient to latest's internal basis
        q_latest = q_ambient @ curve_basis_pinv
        q_latest_int = np.round(q_latest).astype(int)

        # Compute q·p using both in latest's basis
        qp = float(np.dot(q_latest_int, p_latest))

        # Compute M·q using M in latest's basis
        Mq = int(np.dot(M_latest, q_latest_int))

        if qp > 0:
            qp_list.append((qp, tuple(q_latest_int), gv, Mq))

    qp_list.sort(key=lambda x: x[0])

    print(f"\n   Smallest 15 positive q·p values:")
    for qp, q, gv, Mq in qp_list[:15]:
        print(f"     q·p = {qp:.6f} ({qp * 110:.1f}/110), q = {q}, N_q = {gv}, M·q = {Mq}")

    # Check for curves with expected exponents
    print(f"\n6. Looking for curves with q·p ≈ 32/110 = 0.2909 and 33/110 = 0.3000:")
    for target, name in [(32/110, "32/110"), (33/110, "33/110")]:
        close = [(qp, q, gv, Mq) for qp, q, gv, Mq in qp_list if abs(qp - target) < 0.005]
        if close:
            print(f"\n   Matches for {name}:")
            for qp, q, gv, Mq in close[:5]:
                print(f"     q·p = {qp:.6f}, q = {q}, N_q = {gv}, M·q = {Mq}")
        else:
            print(f"\n   No matches for {name}")

    # Verify the racetrack structure
    print("\n7. Racetrack analysis:")
    from collections import defaultdict
    exponent_groups = defaultdict(list)
    for qp, q, gv, Mq in qp_list:
        key = round(qp * 110)  # Convert to integer n where q·p = n/110
        exponent_groups[key].append((q, gv, Mq))

    # Show leading exponents
    sorted_keys = sorted(exponent_groups.keys())
    print(f"   Leading exponents (n/110 where q·p = n/110):")
    for n in sorted_keys[:15]:
        terms = exponent_groups[n]
        # Sum (M·q) * N_q for this exponent
        total_coeff = sum(Mq * gv for _, gv, Mq in terms)
        print(f"     n={n:3d}: {len(terms):3d} curves, Σ(M·q)N_q = {total_coeff}")

    # Check if 32 and 33 have opposite-sign coefficients (racetrack structure)
    if 32 in exponent_groups and 33 in exponent_groups:
        coeff_32 = sum(Mq * gv for _, gv, Mq in exponent_groups[32])
        coeff_33 = sum(Mq * gv for _, gv, Mq in exponent_groups[33])
        print(f"\n   Racetrack: n=32 coeff = {coeff_32}, n=33 coeff = {coeff_33}")
        if coeff_32 * coeff_33 < 0:
            print(f"   ✓ Opposite signs - racetrack structure confirmed!")
        else:
            print(f"   ✗ Same signs - not a typical racetrack")


if __name__ == "__main__":
    main()
