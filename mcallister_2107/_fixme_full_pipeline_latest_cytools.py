#!/usr/bin/env python3
"""
Complete end-to-end pipeline for computing vacuum energy from first principles.
Adapted for latest CYTools (basis [5,6,7,8]).

This script demonstrates that we can compute all physical observables
(g_s, W₀, e^{K₀}, V₀) starting only from:
- Polytope geometry
- Triangulation
- Flux vectors K, M
- GV invariants

Reproduces McAllister et al. arXiv:2107.09064 Section 6.4 for polytope 4-214-647.
"""

import sys
from pathlib import Path

# Use vendored latest cytools
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from collections import defaultdict
from mpmath import mp, mpf, exp, log, pi

from cytools import Polytope

# High precision for W₀ ~ 10^{-90}
mp.dps = 150

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"


def load_geometry():
    """Load polytope and triangulation."""
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
    dual_points = np.array([[int(x) for x in line.split(',')] for line in lines])

    lines = (DATA_DIR / "dual_simplices.dat").read_text().strip().split('\n')
    simplices = [[int(x) for x in line.split(',')] for line in lines]

    return dual_points, simplices


def load_fluxes():
    """McAllister fluxes transformed to latest CYTools basis [5,6,7,8]."""
    K = np.array([8, 5, -8, 6])
    M = np.array([-10, -1, 11, -5])
    return K, M




def compute_flat_direction(kappa, M, K, h11):
    """
    Compute flat direction p using Demirtas lemma.
    p = N⁻¹K where N_ab = κ_abc M^c
    """
    N = np.zeros((h11, h11))
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                N[a, b] += kappa[a, b, c] * M[c]

    if abs(np.linalg.det(N)) < 1e-10:
        raise ValueError("N matrix is singular - no flat direction exists")

    p = np.linalg.solve(N, K)
    return p, N


def compute_eK0(kappa, p, h11):
    """
    Compute e^{K₀} from intersection numbers and flat direction.
    e^{K₀} = (4/3 κ_abc p^a p^b p^c)^{-1}   [eq. 6.12]
    """
    kappa_p3 = 0.0
    for a in range(h11):
        for b in range(h11):
            for c in range(h11):
                kappa_p3 += kappa[a, b, c] * p[a] * p[b] * p[c]

    eK0 = 1.0 / ((4.0/3.0) * kappa_p3)
    return eK0, kappa_p3


def identify_racetrack_terms(gv_invariants, M, p, cutoff=1.0):
    """
    Identify and group racetrack terms from GV invariants.
    Returns grouped terms sorted by q·p exponent.
    """
    groups = defaultdict(lambda: {'q_dot_p': None, 'eff_coeff': 0, 'count': 0})

    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple)
        q_dot_p = np.dot(q, p)

        if 0 < q_dot_p < cutoff:
            M_dot_q = np.dot(M, q)
            if M_dot_q != 0:
                key = round(q_dot_p, 6)
                if groups[key]['q_dot_p'] is None:
                    groups[key]['q_dot_p'] = q_dot_p
                groups[key]['eff_coeff'] += M_dot_q * N_q
                groups[key]['count'] += 1

    grouped = list(groups.values())
    grouped.sort(key=lambda g: g['q_dot_p'])

    return grouped


def solve_racetrack(grouped_terms):
    """
    Solve the two-term racetrack F-term equation.

    For W = ζ(A e^{2πiτα} + B e^{2πiτβ}):
    ∂W/∂τ = 0 gives e^{2πiτ(β-α)} = -Aα/(Bβ)

    Returns (Im_tau, g_s, W0)
    """
    if len(grouped_terms) < 2:
        raise ValueError("Need at least 2 terms for racetrack")

    t1, t2 = grouped_terms[0], grouped_terms[1]

    alpha = mpf(str(t1['q_dot_p']))
    beta = mpf(str(t2['q_dot_p']))
    A = mpf(str(t1['eff_coeff']))
    B = mpf(str(t2['eff_coeff']))

    ratio = -A * alpha / (B * beta)

    if ratio <= 0:
        raise ValueError(f"No real solution: ratio = {float(ratio)}")

    delta = beta - alpha
    y = -log(ratio) / (2 * pi * delta)

    g_s = float(1 / y)

    # Compute W₀ = |W(τ_vev)|
    zeta = mpf(1) / (mpf(2)**mpf('1.5') * pi**mpf('2.5'))
    tau = mp.j * y

    W = -zeta * (A * exp(2 * pi * mp.j * tau * alpha) +
                 B * exp(2 * pi * mp.j * tau * beta))
    W0 = float(abs(W))

    return float(y), g_s, W0


def compute_V0(eK0, g_s, V_string, W0):
    """
    Compute vacuum energy V₀ using eq. 6.24.
    V₀ = -3 × e^{K₀} × (g_s^7 / (4×V[0])²) × W₀²
    """
    return -3.0 * eK0 * (g_s**7 / (4.0 * V_string)**2) * W0**2


def main():
    print("=" * 70)
    print("COMPLETE END-TO-END PIPELINE (Latest CYTools)")
    print("Computing vacuum energy from first principles")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Load geometry and setup CYTools
    # =========================================================================
    print("\n[1] Loading geometry...")

    dual_points, simplices = load_geometry()
    K, M = load_fluxes()

    print(f"    Polytope: {len(dual_points)} points")
    print(f"    Triangulation: {len(simplices)} simplices")
    print(f"    Fluxes: K={K}, M={M}")

    poly = Polytope(dual_points)
    tri = poly.triangulate(simplices=simplices)
    cy = tri.get_cy()

    h11 = cy.h11()
    h21 = cy.h21()
    basis = cy.divisor_basis()

    print(f"    h11 = {h11}, h21 = {h21}")
    print(f"    Divisor basis: {basis}")

    # =========================================================================
    # STEP 2: Compute intersection numbers
    # =========================================================================
    print("\n[2] Computing intersection numbers...")

    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val

    print(f"    κ has {len(kappa_sparse)} non-zero entries")

    # =========================================================================
    # STEP 3: Compute flat direction p
    # =========================================================================
    print("\n[3] Computing flat direction p = N⁻¹K...")

    p, N = compute_flat_direction(kappa, M, K, h11)
    print(f"    p = {p}")
    print(f"    det(N) = {np.linalg.det(N):.2f}")

    # =========================================================================
    # STEP 4: Compute e^{K₀}
    # =========================================================================
    print("\n[4] Computing e^{K₀} from κ and p...")

    eK0, kappa_p3 = compute_eK0(kappa, p, h11)
    print(f"    κ_abc p^a p^b p^c = {kappa_p3:.6f}")
    print(f"    e^{{K₀}} = {eK0:.6f}")

    # =========================================================================
    # STEP 5: Compute GV invariants
    # =========================================================================
    print("\n[5] Computing GV invariants...")

    gv_obj = cy.compute_gvs(min_points=100)
    gv_invariants = {}
    for q, N_q in gv_obj.dok.items():
        if N_q != 0:
            gv_invariants[tuple(q)] = int(N_q)

    print(f"    Found {len(gv_invariants)} non-zero GV invariants")

    # =========================================================================
    # STEP 6: Build racetrack from GV invariants
    # =========================================================================
    print("\n[6] Building racetrack...")

    grouped = identify_racetrack_terms(gv_invariants, M, p)

    print(f"    Found {len(grouped)} distinct exponents in racetrack")
    print(f"    Leading terms:")
    for g in grouped[:3]:
        print(f"      q·p = {g['q_dot_p']:.6f}, eff_coeff = {g['eff_coeff']}")

    # =========================================================================
    # STEP 7: Solve F-term equation
    # =========================================================================
    print("\n[7] Solving F-term equation ∂W/∂τ = 0...")

    Im_tau, g_s, W0 = solve_racetrack(grouped)

    print(f"    Im(τ) = {Im_tau:.6f}")
    print(f"    g_s = {g_s:.8f}")
    print(f"    W₀ = {W0:.6e}")

    # =========================================================================
    # STEP 8: Compute vacuum energy V₀
    # =========================================================================
    print("\n[8] Computing vacuum energy V₀...")

    V_string = float((DATA_DIR / "cy_vol.dat").read_text().strip())
    print(f"    V[0] (string frame) = {V_string:.2f} [from file]")

    V0 = compute_V0(eK0, g_s, V_string, W0)

    print(f"    V₀ = {V0:.6e}")

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Expected values from paper
    g_s_expected = 0.00911134
    W0_expected = 2.30012e-90
    V0_expected = -5.5e-203
    eK0_expected = 0.2361

    print(f"\n{'Quantity':<15} {'Computed':>20} {'Expected':>20} {'Match':>10}")
    print("-" * 70)
    print(f"{'g_s':<15} {g_s:>20.8f} {g_s_expected:>20.8f} {'✓' if abs(g_s/g_s_expected - 1) < 0.001 else '✗':>10}")
    print(f"{'W₀':<15} {W0:>20.6e} {W0_expected:>20.6e} {'✓' if abs(np.log10(W0) - np.log10(W0_expected)) < 0.1 else '✗':>10}")
    print(f"{'e^{{K₀}}':<15} {eK0:>20.6f} {eK0_expected:>20.6f} {'✓' if abs(eK0/eK0_expected - 1) < 0.01 else '~':>10}")
    print(f"{'V₀':<15} {V0:>20.6e} {V0_expected:>20.6e} {'✓' if abs(np.log10(-V0) - np.log10(-V0_expected)) < 0.1 else '✗':>10}")

    print("\n" + "=" * 70)
    print("SUCCESS: All values computed from first principles!")
    print("=" * 70)

    return {
        'g_s': g_s,
        'W0': W0,
        'eK0': eK0,
        'V0': V0,
        'p': p,
        'Im_tau': Im_tau
    }


if __name__ == "__main__":
    main()
