#!/usr/bin/env python3
"""
Compute KKLT moduli stabilization using McAllister's iterative algorithm.

From arXiv:2107.09064 Section 5.2:

The key insight: log(W0) and g_s enter only as overall factors, so we can solve
    (1/2) κ_ijk t^j t^k = c_i
first (independent of fluxes), then scale by the W0-dependent factor.

Algorithm (equations 5.8-5.11):
1. Start from random point t_init in Kähler cone
2. Target: τ* = (c_1, c_2, ..., c_h11)
3. Interpolate: τ_α = (1-α)τ_init + α×τ*
4. At each step, solve LINEAR system: κ_ijk t^j ε^k = τ_{m+1} - τ_m
5. Scale final result

This avoids solving 214 coupled quadratic equations - instead we follow a path
through Kähler moduli space using linear steps.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from cytools import Polytope

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
RESOURCES_DIR = Path(__file__).parent.parent / "resources"


def load_polytope(use_dual: bool = False):
    """Load McAllister's polytope."""
    filename = "dual_points.dat" if use_dual else "points.dat"
    lines = (DATA_DIR / filename).read_text().strip().split('\n')
    points = np.array([[int(x) for x in line.split(',')] for line in lines])
    poly = Polytope(points)
    tri = poly.triangulate()
    cy = tri.get_cy()
    return poly, tri, cy


def get_intersection_tensor(cy):
    """Get full intersection tensor κ_ijk."""
    h11 = cy.h11()
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_sparse.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val
    return kappa


def compute_tau(kappa, t):
    """Compute divisor volumes τ_i = (1/2) κ_ijk t^j t^k."""
    return 0.5 * np.einsum('ijk,j,k->i', kappa, t, t)


def compute_V(kappa, t):
    """Compute CY volume V = (1/6) κ_ijk t^i t^j t^k."""
    return np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0


def compute_jacobian(kappa, t):
    """
    Compute Jacobian matrix J_ik = κ_ijk t^j for the linear step.

    From eq. 5.11: κ_ijk t^j ε^k = Δτ^i
    This is a linear system J @ ε = Δτ where J_ik = κ_ijk t^j
    """
    return np.einsum('ijk,j->ik', kappa, t)


def iterative_solve(kappa, target_tau, n_steps=1000, t_init=None, verbose=True):
    """
    McAllister's iterative algorithm (Section 5.2, eqs 5.8-5.11).

    Solve: τ_i = (1/2) κ_ijk t^j t^k = target_tau_i

    By following a path from t_init to the solution.
    """
    h11 = len(target_tau)

    # Initialize
    if t_init is None:
        t_init = np.ones(h11) * 5.0  # Start with uniform guess

    t = t_init.copy()
    tau_init = compute_tau(kappa, t)

    if verbose:
        print(f"Initial τ: {tau_init[:5]}..." if h11 > 5 else f"Initial τ: {tau_init}")
        print(f"Target τ: {target_tau[:5]}..." if h11 > 5 else f"Target τ: {target_tau}")

    # Path following: τ_α = (1-α)τ_init + α×τ*
    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_target_step = (1 - alpha) * tau_init + alpha * target_tau

        tau_current = compute_tau(kappa, t)
        delta_tau = tau_target_step - tau_current

        # Solve linear system: J @ ε = Δτ where J_ik = κ_ijk t^j
        J = compute_jacobian(kappa, t)

        try:
            epsilon = np.linalg.solve(J, delta_tau)
        except np.linalg.LinAlgError:
            # Singular matrix - try pseudoinverse
            epsilon = np.linalg.lstsq(J, delta_tau, rcond=None)[0]

        # Update t
        t = t + epsilon

        # Keep t positive (Kähler cone constraint)
        t = np.maximum(t, 1e-6)

        if verbose and (m + 1) % (n_steps // 10) == 0:
            tau_achieved = compute_tau(kappa, t)
            error = np.sqrt(np.mean((tau_achieved - target_tau)**2))
            print(f"  Step {m+1}/{n_steps}: RMS error = {error:.6f}")

    tau_final = compute_tau(kappa, t)
    return t, tau_final


def mcallister_kklt_solve(kappa, c_i, W0, c_tau=3.34109, verbose=True):
    """
    Full McAllister KKLT solution.

    1. Solve τ_i = c_i (unit-normalized, W0-independent)
    2. Scale t by sqrt(ln(W0^-1) / (2π)) since τ ~ t²

    The target is: τ_i = (c_i / 2π) × ln(W0^-1)
    We first solve: τ_i = c_i
    Then scale: τ^target / τ^unit = ln(W0^-1) / (2π)
    Since τ ~ t²: t_scale = sqrt(ln(W0^-1) / (2π))

    Args:
        kappa: Intersection tensor
        c_i: Dual Coxeter numbers
        W0: Flux superpotential
        c_tau: From eq 2.29, relates g_s to W0 (not used in basic scaling)
    """
    h11 = len(c_i)

    if verbose:
        print("=" * 70)
        print("McAllister KKLT Iterative Algorithm")
        print("=" * 70)
        print(f"h11 = {h11}")
        print(f"W0 = {W0:.2e}")

    # Step 1: Solve τ_i = c_i (unit-normalized)
    if verbose:
        print(f"\n[1] Solving τ_i = c_i (unit-normalized)...")

    t_unit, tau_unit = iterative_solve(kappa, c_i, n_steps=500, verbose=verbose)

    if verbose:
        print(f"\n    Unit solution t: {t_unit[:5]}..." if h11 > 5 else f"    Unit solution t: {t_unit}")
        print(f"    Unit τ achieved: {tau_unit[:5]}..." if h11 > 5 else f"    Unit τ achieved: {tau_unit}")

    # Step 2: Scale by W0-dependent factor
    # Target: τ_i = (c_i / 2π) × ln(W0^-1)
    # Unit: τ_i = c_i
    # Ratio: τ^target / τ^unit = ln(W0^-1) / (2π)
    # Since τ ~ t²: t_scale = sqrt(ln(W0^-1) / (2π))
    ln_W0_inv = np.log(1.0 / np.abs(W0))
    tau_scale = ln_W0_inv / (2 * np.pi)
    t_scale = np.sqrt(tau_scale)

    if verbose:
        print(f"\n[2] Scaling by W0-dependent factor...")
        print(f"    ln(W0^-1) = {ln_W0_inv:.2f}")
        print(f"    τ scale factor = {tau_scale:.4f}")
        print(f"    t scale factor = {t_scale:.4f}")

    t_final = t_unit * t_scale
    tau_final = compute_tau(kappa, t_final)
    V_string = compute_V(kappa, t_final)

    if verbose:
        print(f"\n[3] Final result:")
        print(f"    t: {t_final[:5]}..." if h11 > 5 else f"    t: {t_final}")
        print(f"    τ: {tau_final[:5]}..." if h11 > 5 else f"    τ: {tau_final}")
        print(f"    V_string = {V_string:.2f}")

    return {
        "t": t_final,
        "tau": tau_final,
        "V_string": V_string,
        "t_unit": t_unit,
        "tau_unit": tau_unit,
        "t_scale": t_scale,
    }


def test_dual():
    """Test on dual polytope (h11=4) - should give V_string ≈ 4695."""
    print("\n" + "#" * 70)
    print("# TEST: DUAL POLYTOPE (h11=4)")
    print("#" * 70)

    poly, tri, cy = load_polytope(use_dual=True)
    h11 = cy.h11()
    print(f"h11={h11}, h21={cy.h21()}")

    kappa = get_intersection_tensor(cy)

    # For dual, all 4 basis divisors are O7-planes: c_i = 6
    c_i = np.array([6.0, 6.0, 6.0, 6.0])
    W0 = 2.30012e-90

    result = mcallister_kklt_solve(kappa, c_i, W0, verbose=True)

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"V_string = {result['V_string']:.2f}")
    print(f"Expected ≈ 4695 (dual) or 4711 (primal)")

    return result


def test_primal():
    """Test on primal polytope (h11=214)."""
    print("\n" + "#" * 70)
    print("# TEST: PRIMAL POLYTOPE (h11=214)")
    print("#" * 70)

    poly, tri, cy = load_polytope(use_dual=False)
    h11 = cy.h11()
    print(f"h11={h11}, h21={cy.h21()}")

    kappa = get_intersection_tensor(cy)
    print(f"Intersection tensor computed: {int(np.sum(kappa != 0))} non-zero entries")

    # Load orientifold c_i values
    with open(RESOURCES_DIR / "mcallister_4-214-647_orientifold.json") as f:
        orientifold = json.load(f)

    basis = list(cy.divisor_basis())
    kklt_basis = orientifold['kklt_basis']
    c_values = orientifold['c_i_values']
    point_to_c = {idx: c_values[i] for i, idx in enumerate(kklt_basis)}
    c_i = np.array([float(point_to_c.get(idx, 1.0)) for idx in basis])

    print(f"c_i: {int(np.sum(c_i==6))} O7-planes, {int(np.sum(c_i==1))} D3-instantons")

    W0 = 2.30012e-90

    result = mcallister_kklt_solve(kappa, c_i, W0, verbose=True)

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"V_string = {result['V_string']:.2f}")
    print(f"Expected = 4711.83")
    error = abs(result['V_string'] - 4711.83) / 4711.83 * 100
    print(f"Error: {error:.2f}%")

    return result


if __name__ == "__main__":
    # Test dual first (fast)
    test_dual()

    print("\n\n")

    # Test primal (slower but should now work)
    test_primal()
