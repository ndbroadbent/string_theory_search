#!/usr/bin/env python3
"""
Search for (K, M) pairs that reproduce McAllister's full physics in the latest CYTools basis.

Target values:
- e^K0 = 0.2344
- g_s = 0.00911134
- W_0 = 2.30012e-90
- V_0 = -5.5e-203

This is an MVP test of the inner loop sampling strategy.
"""
import sys
sys.path.insert(0, str(__file__).replace('/mcallister_2107/search_km.py', '/vendor/cytools_latest/src'))

import numpy as np
from pathlib import Path
from scipy.special import spence


def Li2(x):
    """Polylogarithm Li_2(x)."""
    return spence(1 - x)


# Target values from McAllister
TARGET_EK0 = 0.2344
TARGET_GS = 0.00911134
TARGET_W0 = 2.30012e-90
TARGET_V0 = -5.5e-203
TARGET_CY_VOL = 4711.83


def sparse_to_dense(sparse, h11):
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa


def compute_racetrack(p, M, gv_invariants):
    """Compute g_s and W_0 from the racetrack mechanism."""
    curves_by_action = {}
    for q, N_q in gv_invariants.items():
        q = np.array(q)
        action = np.dot(q, p)
        if action > 0:
            M_dot_q = np.dot(M, q)
            if M_dot_q != 0:
                key = round(action, 6)
                if key not in curves_by_action:
                    curves_by_action[key] = []
                curves_by_action[key].append({
                    'q': q, 'N_q': N_q, 'M_dot_q': M_dot_q, 'action': action,
                })

    if len(curves_by_action) < 2:
        return None

    sorted_actions = sorted(curves_by_action.keys())
    if len(sorted_actions) < 2:
        return None

    action1, action2 = sorted_actions[0], sorted_actions[1]

    def sum_coeff(action):
        return sum(c['M_dot_q'] * c['N_q'] * c['action'] for c in curves_by_action[action])

    c1, c2 = sum_coeff(action1), sum_coeff(action2)
    if c1 == 0 or c2 == 0:
        return None

    delta_action = action2 - action1
    if delta_action <= 0:
        return None

    ratio = -c2 / c1
    if ratio <= 0:
        return None

    g_s = 2 * np.pi * delta_action / np.log(ratio)
    if g_s <= 0 or g_s > 1:
        return None

    exponent = -2 * np.pi * action1 / g_s
    W_0 = 0.0 if exponent < -500 else abs(c1) * np.exp(exponent)

    return {'g_s': g_s, 'W_0': W_0, 'action1': action1, 'action2': action2, 'c1': c1, 'c2': c2}


def compute_V0(eK0, g_s, W_0, V_CY):
    return -3 * eK0 * (g_s**7 / (4 * V_CY)**2) * W_0**2


def check_N_invertible(kappa, M):
    N = np.einsum('abc,c->ab', kappa, M)
    return abs(np.linalg.det(N)) > 1e-10, N


def compute_p(N, K):
    return np.linalg.solve(N, K)


def check_p_positive(p):
    return np.all(p > 0)


def check_tadpole(K, M, Q_D3=500):
    return -0.5 * np.dot(M, K) <= Q_D3


def compute_eK0(kappa, p):
    kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
    if abs(kappa_p3) < 1e-10:
        return None
    return 1.0 / ((4.0/3.0) * kappa_p3)


def main():
    # Load McAllister's polytope
    DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
    dual_points = np.loadtxt(DATA_DIR / "dual_points.dat", delimiter=',').astype(int)
    with open(DATA_DIR / "dual_simplices.dat") as f:
        simplices = [[int(x) for x in line.strip().split(',')] for line in f]

    # Compute GV invariants using cygv
    print("Computing GV invariants via cygv...")
    from cytools import Polytope
    poly = Polytope(dual_points)
    tri = poly.triangulate(simplices=simplices)
    cy = tri.get_cy()

    gv_obj = cy.compute_gvs(min_points=100)
    gv_invariants = {}
    for q, N_q in gv_obj.dok.items():
        if N_q != 0:
            gv_invariants[tuple(q)] = int(N_q)
    print(f"Computed {len(gv_invariants)} non-zero GV invariants")

    if len(gv_invariants) == 0:
        raise RuntimeError("No GV invariants computed - cannot proceed")

    # Get intersection numbers
    basis = list(cy.divisor_basis())
    print(f"Using basis: {basis}")
    print(f"h11 = {cy.h11()}")

    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = sparse_to_dense(kappa_sparse, h11)
    print(f"Intersection numbers loaded: {len(kappa_sparse)} non-zero entries")

    # Search
    print("\n" + "="*60)
    print("Searching for (K, M) pairs...")
    print("="*60)

    np.random.seed(42)
    K_range = range(-15, 16)
    M_range = range(-15, 16)

    stats = {
        'total': 0, 'N_invertible': 0, 'p_positive': 0,
        'tadpole_ok': 0, 'valid': 0, 'has_racetrack': 0, 'close_to_target': 0,
    }
    best_results = []

    N_SAMPLES = 10_000_000  # 10 million samples
    print(f"Sampling {N_SAMPLES} random (K, M) pairs...")

    for i in range(N_SAMPLES):
        K = np.array([np.random.choice(list(K_range)) for _ in range(h11)])
        M = np.array([np.random.choice(list(M_range)) for _ in range(h11)])

        stats['total'] += 1

        invertible, N = check_N_invertible(kappa, M)
        if not invertible:
            continue
        stats['N_invertible'] += 1

        p = compute_p(N, K)
        if not check_p_positive(p):
            continue
        stats['p_positive'] += 1

        if not check_tadpole(K, M):
            continue
        stats['tadpole_ok'] += 1

        eK0 = compute_eK0(kappa, p)
        if eK0 is None or eK0 < 0:
            continue
        stats['valid'] += 1

        racetrack = compute_racetrack(p, M, gv_invariants)

        result = {
            'K': K.tolist(), 'M': M.tolist(), 'p': p.tolist(),
            'eK0': eK0, 'eK0_error': abs(eK0 - TARGET_EK0) / TARGET_EK0,
        }

        if racetrack:
            g_s, W_0 = racetrack['g_s'], racetrack['W_0']
            V_0 = compute_V0(eK0, g_s, W_0, TARGET_CY_VOL)

            result['g_s'] = g_s
            result['W_0'] = W_0
            result['V_0'] = V_0
            result['g_s_error'] = abs(g_s - TARGET_GS) / TARGET_GS
            result['W_0_log_error'] = abs(np.log10(W_0 + 1e-300) - np.log10(TARGET_W0))
            result['V_0_log_error'] = abs(np.log10(abs(V_0) + 1e-300) - np.log10(abs(TARGET_V0)))
            result['total_error'] = (
                result['eK0_error'] + result['g_s_error'] +
                result['W_0_log_error'] / 100 + result['V_0_log_error'] / 100
            )
        else:
            result['g_s'] = None
            result['total_error'] = float('inf')

        if result['g_s'] is not None:
            stats['has_racetrack'] += 1
            best_results.append(result)
            if result['eK0_error'] < 0.05:
                stats['close_to_target'] += 1

    # Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total samples:     {stats['total']:,}")
    print(f"N invertible:      {stats['N_invertible']:,} ({100*stats['N_invertible']/stats['total']:.1f}%)")
    print(f"p positive:        {stats['p_positive']:,} ({100*stats['p_positive']/stats['total']:.1f}%)")
    print(f"Tadpole OK:        {stats['tadpole_ok']:,} ({100*stats['tadpole_ok']/stats['total']:.1f}%)")
    print(f"Valid (eK0 > 0):   {stats['valid']:,} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"Has racetrack:     {stats['has_racetrack']:,} ({100*stats['has_racetrack']/max(1,stats['valid']):.1f}% of valid)")
    print(f"Close to target:   {stats['close_to_target']:,}")

    print("\n" + "="*60)
    print("TARGET VALUES (McAllister)")
    print("="*60)
    print(f"e^K0 = {TARGET_EK0}")
    print(f"g_s  = {TARGET_GS}")
    print(f"W_0  = {TARGET_W0:.2e}")
    print(f"V_0  = {TARGET_V0:.2e}")

    if best_results:
        best_results.sort(key=lambda x: x['total_error'])

        print("\n" + "="*60)
        print("TOP 10 BEST MATCHES (by combined error)")
        print("="*60)

        for i, r in enumerate(best_results[:10]):
            print(f"\n#{i+1}: total_error = {r['total_error']:.4f}")
            print(f"    K = {r['K']}")
            print(f"    M = {r['M']}")
            print(f"    p = [{', '.join(f'{x:.4f}' for x in r['p'])}]")
            print(f"    e^K0 = {r['eK0']:.6f} (target: {TARGET_EK0}, err: {r['eK0_error']*100:.2f}%)")
            if r['g_s'] is not None:
                print(f"    g_s  = {r['g_s']:.6f} (target: {TARGET_GS}, err: {r['g_s_error']*100:.2f}%)")
                print(f"    W_0  = {r['W_0']:.2e} (target: {TARGET_W0:.2e}, log_err: {r['W_0_log_error']:.1f})")
                print(f"    V_0  = {r['V_0']:.2e} (target: {TARGET_V0:.2e}, log_err: {r['V_0_log_error']:.1f})")
    else:
        print("\nNo results with valid racetrack found.")


if __name__ == '__main__':
    main()
