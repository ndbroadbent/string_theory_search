#!/usr/bin/env python3
"""
Debug racetrack term computation for failing examples.

Question: McAllister solved ALL 5 examples. Why does our solver fail on 3?

Hypothesis: Our curve basis transformation or coefficient grouping is wrong.
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

sys.path.insert(0, str(CYTOOLS_2021))
from cytools import Polytope


def load_example(name: str) -> dict:
    """Load all data for an example."""
    d = DATA_BASE / name

    # Polytope
    lines = (d / "dual_points.dat").read_text().strip().split('\n')
    dual_points = np.array([[int(x) for x in line.split(',')] for line in lines])

    # Simplices
    lines = (d / "dual_simplices.dat").read_text().strip().split('\n')
    simplices = [[int(x) for x in line.split(',')] for line in lines]

    # Fluxes
    K = np.array([int(x) for x in (d / "K_vec.dat").read_text().strip().split(',')])
    M = np.array([int(x) for x in (d / "M_vec.dat").read_text().strip().split(',')])

    # Curves and GV
    lines = (d / "dual_curves.dat").read_text().strip().split('\n')
    curves = np.array([[int(x) for x in line.split(',')] for line in lines])
    gv = np.array([int(x) for x in (d / "dual_curves_gv.dat").read_text().strip().split(',')])

    # Expected values
    g_s = float((d / "g_s.dat").read_text().strip())
    W0 = float((d / "W_0.dat").read_text().strip())

    return {
        "name": name,
        "dual_points": dual_points,
        "simplices": simplices,
        "K": K,
        "M": M,
        "curves": curves,
        "gv": gv,
        "g_s_expected": g_s,
        "W0_expected": W0,
    }


def analyze_racetrack(data: dict):
    """Analyze racetrack structure for one example."""
    name = data["name"]
    print("=" * 70)
    print(f"ANALYZING: {name}")
    print("=" * 70)

    # Setup geometry
    poly = Polytope(data["dual_points"])
    tri = poly.triangulate(simplices=data["simplices"], check_input_simplices=False)
    cy = tri.get_cy()

    h11 = cy.h11()
    basis = list(cy.divisor_basis())

    print(f"\nh11={h11}, basis={basis}")
    print(f"K={data['K']}, M={data['M']}")
    print(f"Expected: g_s={data['g_s_expected']:.6f}, W₀={data['W0_expected']:.2e}")

    # Get intersection numbers
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))
    for row in kappa_sparse:
        i, j, k, val = int(row[0]), int(row[1]), int(row[2]), row[3]
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val

    # Compute p
    K, M = data["K"], data["M"]
    N = np.einsum('abc,c->ab', kappa, M)
    det_N = np.linalg.det(N)
    print(f"\ndet(N) = {det_N:.2f}")

    if abs(det_N) < 1e-10:
        print("ERROR: N is singular!")
        return

    p = np.linalg.solve(N, K)
    print(f"p = {p}")

    # Project curves to basis
    curves = data["curves"]
    gv = data["gv"]

    # Curves are in ambient (h11+5 components), project to basis
    curves_basis = curves[:, basis]

    print(f"\nCurves: {len(curves)} total, ambient dim={curves.shape[1]}, basis dim={curves_basis.shape[1]}")

    # Compute q·p and M·q for all curves
    q_dot_p = curves_basis @ p
    M_dot_q = curves_basis @ M

    # Filter for positive q·p < 1
    mask = (q_dot_p > 0) & (q_dot_p < 1)
    n_valid = mask.sum()
    print(f"Curves with 0 < q·p < 1: {n_valid}")

    # Group by q·p value
    groups = defaultdict(lambda: {"coeff": 0, "count": 0, "gv_sum": 0})
    for i in np.where(mask)[0]:
        key = round(q_dot_p[i], 6)
        groups[key]["coeff"] += M_dot_q[i] * gv[i]
        groups[key]["count"] += 1
        groups[key]["gv_sum"] += gv[i]

    # Sort by exponent
    sorted_groups = sorted(groups.items(), key=lambda x: x[0])

    print(f"\nRacetrack terms (first 10):")
    print(f"{'q·p':>12} {'coeff':>12} {'count':>8} {'sign':>6}")
    print("-" * 42)
    for exp, g in sorted_groups[:10]:
        sign = "+" if g["coeff"] > 0 else "-"
        print(f"{exp:>12.6f} {g['coeff']:>12d} {g['count']:>8d} {sign:>6}")

    # Find first pair with OPPOSITE signs (the actual racetrack structure)
    print(f"\nSearching for opposite-sign pair...")

    found_pair = False
    for i in range(len(sorted_groups)):
        exp1, g1 = sorted_groups[i]
        c1 = g1["coeff"]
        if c1 == 0:
            continue

        for j in range(i+1, len(sorted_groups)):
            exp2, g2 = sorted_groups[j]
            c2 = g2["coeff"]
            if c2 == 0:
                continue

            if c1 * c2 < 0:
                # Found opposite-sign pair!
                print(f"\nFound racetrack pair at indices {i}, {j}:")
                print(f"  α = {exp1:.6f}, A = {c1}")
                print(f"  β = {exp2:.6f}, B = {c2}")

                ratio = -c1 * exp1 / (c2 * exp2)
                if ratio > 0:
                    delta = exp2 - exp1
                    Im_tau = np.log(ratio) / (2 * np.pi * delta)
                    g_s = 1 / Im_tau
                    print(f"\n  Racetrack solution:")
                    print(f"    Im(τ) = {Im_tau:.4f}")
                    print(f"    g_s = {g_s:.6f} (expected: {data['g_s_expected']:.6f})")
                    print(f"    ratio = {g_s / data['g_s_expected']:.4f}")
                    found_pair = True
                    break
                else:
                    print(f"  ratio = {ratio} (negative, trying next pair)")

        if found_pair:
            break

    if not found_pair:
        print("\n  NO opposite-sign pair found in racetrack terms!")
        print("  This is a fundamental problem with the curve/flux combination.")

    return sorted_groups


def main():
    examples = [
        "4-214-647",          # PASSES
        "5-113-4627-main",    # PASSES
        "5-81-3213",          # FAILS
        "7-51-13590",         # FAILS
    ]

    for name in examples:
        data = load_example(name)
        analyze_racetrack(data)
        print("\n")


if __name__ == "__main__":
    main()
