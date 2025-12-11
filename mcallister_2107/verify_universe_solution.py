#!/usr/bin/env python3
"""Verify the universe CC solution found by GA."""
import sys
sys.path.insert(0, str(__file__).replace('/mcallister_2107/verify_universe_solution.py', '/vendor/cytools_latest/src'))

import numpy as np
from pathlib import Path


def main():
    # The solution found
    K = np.array([16, 12, 11, 13])
    M = np.array([-20, -18, 21, 15])

    # Target
    UNIVERSE_LAMBDA = 2.846e-122
    TARGET_LOG = np.log10(UNIVERSE_LAMBDA)

    print("=" * 60)
    print("VERIFICATION OF UNIVERSE CC SOLUTION")
    print("=" * 60)
    print(f"\nK = {K.tolist()}")
    print(f"M = {M.tolist()}")
    print(f"\nTarget: Λ = {UNIVERSE_LAMBDA:.3e}")
    print(f"Target log10(|V₀|) = {TARGET_LOG:.4f}")

    # Load geometry
    DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
    dual_points = np.loadtxt(DATA_DIR / "dual_points.dat", delimiter=',').astype(int)
    with open(DATA_DIR / "dual_simplices.dat") as f:
        simplices = [[int(x) for x in line.strip().split(',')] for line in f]

    print("\n--- Loading CYTools geometry ---")
    from cytools import Polytope
    poly = Polytope(dual_points)
    tri = poly.triangulate(simplices=simplices)
    cy = tri.get_cy()

    h11 = cy.h11()
    print(f"h11 = {h11}")
    print(f"Divisor basis = {list(cy.divisor_basis())}")

    # Get intersection numbers
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val

    print(f"\n--- Step 1: Compute N = κ_abc M^c ---")
    N = np.einsum('abc,c->ab', kappa, M)
    print(f"N matrix:\n{N}")
    det_N = np.linalg.det(N)
    print(f"det(N) = {det_N:.6f}")

    print(f"\n--- Step 2: Solve N·p = K ---")
    p = np.linalg.solve(N, K)
    print(f"p = {p}")
    print(f"All p > 0? {np.all(p > 0)}")

    print(f"\n--- Step 3: Compute e^K0 ---")
    kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
    print(f"κ_abc p^a p^b p^c = {kappa_p3:.6f}")
    eK0 = 1.0 / ((4.0/3.0) * kappa_p3)
    print(f"e^K0 = {eK0:.6f}")

    print(f"\n--- Step 4: Compute GV invariants and racetrack ---")
    gv_obj = cy.compute_gvs(min_points=100)
    gv_invariants = {}
    for q, N_q in gv_obj.dok.items():
        if N_q != 0:
            gv_invariants[tuple(q)] = int(N_q)
    print(f"Non-zero GV invariants: {len(gv_invariants)}")

    # Racetrack computation
    curves_by_action = {}
    for q, N_q in gv_invariants.items():
        q_arr = np.array(q)
        action = np.dot(q_arr, p)
        if action > 0:
            M_dot_q = np.dot(M, q_arr)
            if M_dot_q != 0:
                key = round(action, 6)
                if key not in curves_by_action:
                    curves_by_action[key] = []
                curves_by_action[key].append({
                    'q': q, 'N_q': N_q, 'M_dot_q': M_dot_q, 'action': action,
                })

    print(f"Actions with curves: {len(curves_by_action)}")
    sorted_actions = sorted(curves_by_action.keys())[:5]
    print(f"First 5 actions: {sorted_actions}")

    action1, action2 = sorted_actions[0], sorted_actions[1]
    print(f"\nUsing actions: {action1:.6f}, {action2:.6f}")

    def sum_coeff(action):
        return sum(c['M_dot_q'] * c['N_q'] * c['action'] for c in curves_by_action[action])

    c1, c2 = sum_coeff(action1), sum_coeff(action2)
    print(f"c1 = {c1:.6f}, c2 = {c2:.6f}")

    delta_action = action2 - action1
    ratio = -c2 / c1
    print(f"delta_action = {delta_action:.6f}")
    print(f"ratio = -c2/c1 = {ratio:.6f}")

    g_s = 2 * np.pi * delta_action / np.log(ratio)
    print(f"\ng_s = 2π × {delta_action:.6f} / ln({ratio:.6f})")
    print(f"g_s = {g_s:.6f}")

    exponent = -2 * np.pi * action1 / g_s
    print(f"\nW_0 exponent = -2π × {action1:.6f} / {g_s:.6f} = {exponent:.2f}")
    W_0 = abs(c1) * np.exp(exponent) if exponent > -500 else 0
    print(f"W_0 = |c1| × exp({exponent:.2f}) = {W_0:.6e}")

    print(f"\n--- Step 5: Compute V_0 ---")
    V_CY = 4711.83  # From McAllister data
    print(f"V_CY (string frame) = {V_CY}")

    # V_0 = -3 × e^K0 × (g_s^7 / (4 V_CY)^2) × W_0^2
    factor1 = g_s**7 / (4 * V_CY)**2
    V_0 = -3 * eK0 * factor1 * W_0**2
    print(f"\nV_0 = -3 × {eK0:.6f} × ({g_s:.6f}^7 / (4×{V_CY})^2) × ({W_0:.6e})^2")
    print(f"V_0 = {V_0:.6e}")

    # Log form for precision
    if W_0 > 0:
        log_V0 = (np.log10(3) + np.log10(eK0) + 7*np.log10(g_s)
                  - 2*np.log10(4*V_CY) + 2*np.log10(W_0))
        print(f"\nlog10(|V_0|) = {log_V0:.4f}")
    else:
        log_V0 = -np.inf
        print(f"\nW_0 = 0, so V_0 = 0")

    print(f"\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Target:   log10(|V₀|) = {TARGET_LOG:.4f}")
    print(f"Computed: log10(|V₀|) = {log_V0:.4f}")
    print(f"Distance: {abs(log_V0 - TARGET_LOG):.4f}")

    ratio_to_universe = 10**(log_V0 - TARGET_LOG)
    print(f"\n|V₀|/Λ_universe = {ratio_to_universe:.4f}")

    if abs(log_V0 - TARGET_LOG) < 0.1:
        print(f"\n✅ VERIFIED: Solution matches universe's Λ within 0.1 dex!")
    else:
        print(f"\n❌ MISMATCH: Distance = {abs(log_V0 - TARGET_LOG):.2f}")


if __name__ == '__main__':
    main()
