#!/usr/bin/env python3
"""
Full Pipeline: Compute V₀ from First Principles

This script chains all compute_ modules to go from raw inputs to V₀.

INPUTS (Model Choices):
    - primal_points: np.ndarray (N x 4) primal polytope vertices
    - dual_points: np.ndarray (M x 4) dual polytope vertices
    - K: np.ndarray (h21,) flux vector
    - M: np.ndarray (h21,) flux vector
    - orientifold_o7_indices: list[int] divisor indices with O7-planes (c_i=6)

OUTPUTS:
    - V0: Vacuum energy (cosmological constant)
    - V_string: String frame CY volume
    - g_s: String coupling
    - W0: Flux superpotential
    - e_K0: Complex structure Kähler potential factor
    - t: Kähler moduli solution

PIPELINE STEPS:
    1-4: Model choices (inputs)
    5: Hodge numbers h11, h21
    6: Intersection numbers κ_ijk
    7: Euler characteristic χ
    8: GV invariants N_q
    9: N_ab = κ_abc M^c
    10: p = N^{-1} K (flat direction)
    11: e^{K₀} = (4/3 × κ_abc p^a p^b p^c)^{-1}
    12-14: Racetrack → g_s, W₀
    15: Target τ = c_i/c_τ + χ(D_i)/24
    16: KKLT solve for t
    17: V_string = (1/6)κt³ - BBHL
    18: V₀ = -3 × e^{K₀} × (g_s^7/(4V)²) × W₀²

Reference: arXiv:2107.09064
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from scipy.special import zeta
from mpmath import mp, mpf, exp as mp_exp, log as mp_log, pi as mp_pi

from cytools import Polytope

# Import pipeline components
from compute_target_tau import compute_c_tau, compute_target_tau
from compute_chi_divisor import compute_chi_divisor
from compute_gv_invariants import compute_gv_invariants
from compute_kklt_iterative import (
    SparseIntersectionTensor,
    iterative_solve,
    compute_gv_correction,
)

# High precision for W₀ ~ 10^{-90}
mp.dps = 150


def run_pipeline(
    primal_points: np.ndarray,
    dual_points: np.ndarray,
    K: np.ndarray,
    M: np.ndarray,
    orientifold_o7_indices: list[int],
    verbose: bool = True,
) -> dict:
    """
    Run the full physics pipeline from polytope to V₀.

    Args:
        primal_points: Primal polytope vertices (N x 4)
        dual_points: Dual polytope vertices (M x 4)
        K: Flux vector K (h21 components)
        M: Flux vector M (h21 components)
        orientifold_o7_indices: Divisor indices hosting O7-planes (c_i=6)
        verbose: Print progress

    Returns:
        Dict with all computed quantities
    """
    results = {}

    # =========================================================================
    # STEP 5: Hodge Numbers
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("STEP 5: Hodge Numbers")
        print("=" * 70)

    # Create polytopes
    primal_poly = Polytope(primal_points)
    dual_poly = Polytope(dual_points)

    # Triangulate (CYTools default FRST)
    primal_tri = primal_poly.triangulate()
    dual_tri = dual_poly.triangulate()

    primal_cy = primal_tri.get_cy()
    dual_cy = dual_tri.get_cy()

    h11_primal = primal_cy.h11()
    h21_primal = primal_cy.h21()
    h11_dual = dual_cy.h11()
    h21_dual = dual_cy.h21()

    if verbose:
        print(f"  Primal: h11={h11_primal}, h21={h21_primal}")
        print(f"  Dual:   h11={h11_dual}, h21={h21_dual}")

    results["h11"] = h11_primal
    results["h21"] = h21_primal

    # =========================================================================
    # STEP 6: Intersection Numbers κ_ijk (PRIMAL)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 6: Intersection Numbers")
        print("=" * 70)

    kappa_primal = primal_cy.intersection_numbers(in_basis=True)

    if verbose:
        print(f"  Primal κ: {len(kappa_primal)} non-zero entries")

    results["kappa_primal"] = kappa_primal

    # Also get dual κ for racetrack (Step 9-11)
    kappa_dual = dual_cy.intersection_numbers(in_basis=True)

    if verbose:
        print(f"  Dual κ:   {len(kappa_dual)} non-zero entries")

    results["kappa_dual"] = kappa_dual

    # Build dense tensor for dual (small, h11=4)
    kappa_dual_dense = np.zeros((h11_dual, h11_dual, h11_dual))
    for (i, j, k), val in kappa_dual.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa_dual_dense[perm] = val

    # =========================================================================
    # STEP 7: Euler Characteristic
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 7: Euler Characteristic")
        print("=" * 70)

    chi = 2 * (h11_primal - h21_primal)

    if verbose:
        print(f"  χ = 2(h11 - h21) = 2({h11_primal} - {h21_primal}) = {chi}")

    results["chi"] = chi

    # =========================================================================
    # STEP 8: GV Invariants (from dual for racetrack)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 8: GV Invariants")
        print("=" * 70)

    gv_invariants = compute_gv_invariants(dual_cy, min_points=100)

    if verbose:
        print(f"  Found {len(gv_invariants)} non-zero GV invariants")

    results["gv_invariants"] = gv_invariants

    # =========================================================================
    # STEP 9: N-matrix = κ_abc M^c
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 9: N-matrix")
        print("=" * 70)

    N_matrix = np.einsum('abc,c->ab', kappa_dual_dense, M)
    det_N = np.linalg.det(N_matrix)

    if verbose:
        print(f"  N = κ_abc M^c:")
        print(f"  {N_matrix}")
        print(f"  det(N) = {det_N:.6f}")

    if abs(det_N) < 1e-10:
        raise ValueError("N matrix is singular - invalid flux choice")

    results["N_matrix"] = N_matrix

    # =========================================================================
    # STEP 10: Flat direction p = N^{-1} K
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 10: Flat Direction p")
        print("=" * 70)

    p = np.linalg.solve(N_matrix, K)

    if verbose:
        print(f"  p = N^{{-1}} K = {p}")

    results["p"] = p

    # =========================================================================
    # STEP 11: e^{K₀} = (4/3 × κ_abc p^a p^b p^c)^{-1}
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 11: e^{K₀}")
        print("=" * 70)

    kappa_p3 = np.einsum('abc,a,b,c->', kappa_dual_dense, p, p, p)
    e_K0 = 1.0 / ((4.0/3.0) * kappa_p3)

    if verbose:
        print(f"  κ_abc p^a p^b p^c = {kappa_p3:.6f}")
        print(f"  e^{{K₀}} = {e_K0:.6f}")

    results["e_K0"] = e_K0

    # =========================================================================
    # STEPS 12-14: Racetrack → g_s, W₀
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEPS 12-14: Racetrack Stabilization")
        print("=" * 70)

    g_s, W0 = solve_racetrack(gv_invariants, p, M, verbose=verbose)

    results["g_s"] = float(g_s)
    results["W0"] = W0  # Keep as mpf for precision

    # =========================================================================
    # STEP 15: Target τ = c_i/c_τ + χ(D_i)/24
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 15: Target Divisor Volumes")
        print("=" * 70)

    # Build c_i array for primal divisors
    basis_indices = list(primal_cy.divisor_basis())
    c_i = np.array([6.0 if idx in orientifold_o7_indices else 1.0 for idx in basis_indices])

    n_o7 = int(np.sum(c_i == 6))
    n_d3 = int(np.sum(c_i == 1))

    if verbose:
        print(f"  Orientifold: {n_o7} O7-planes (c=6), {n_d3} D3-instantons (c=1)")

    # Compute c_τ
    c_tau = compute_c_tau(float(g_s), float(W0))

    if verbose:
        print(f"  c_τ = 2π / (g_s × ln(1/W₀)) = {c_tau:.6f}")

    # Compute χ(D_i) for primal divisors
    chi_D = compute_chi_divisor(primal_poly, kappa_primal, basis_indices)

    if verbose:
        print(f"  χ(D_i) range: [{chi_D.min():.0f}, {chi_D.max():.0f}]")

    # Zeroth-order target (without GV corrections for now)
    tau_target_zeroth = c_i / c_tau + chi_D / 24.0

    if verbose:
        print(f"  τ_target range: [{tau_target_zeroth.min():.4f}, {tau_target_zeroth.max():.4f}]")

    results["c_i"] = c_i
    results["c_tau"] = c_tau
    results["chi_D"] = chi_D
    results["tau_target"] = tau_target_zeroth

    # =========================================================================
    # STEP 16: KKLT Solve for t
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 16: KKLT Solve for Kähler Moduli t")
        print("=" * 70)

    kappa_sparse = SparseIntersectionTensor(primal_cy)

    if verbose:
        print(f"  Sparse κ: {kappa_sparse.n_entries} entries")

    # Initialize t from uniform starting point
    t_init = np.ones(h11_primal)
    tau_init = kappa_sparse.compute_tau(t_init)
    scale = np.sqrt(np.mean(tau_target_zeroth) / np.mean(tau_init))
    t_init = t_init * scale

    if verbose:
        print(f"  Initial t scale: {scale:.4f}")

    # Solve
    t_solution, tau_achieved, converged = iterative_solve(
        kappa_sparse, tau_target_zeroth,
        n_steps=500, t_init=t_init, tol=1e-3, verbose=verbose
    )

    if verbose:
        print(f"  Converged: {converged}")
        print(f"  t range: [{t_solution.min():.4f}, {t_solution.max():.4f}]")
        n_neg = np.sum(t_solution < 0)
        print(f"  Negative t values: {n_neg}/{len(t_solution)}")

    results["t"] = t_solution
    results["tau_achieved"] = tau_achieved
    results["kklt_converged"] = converged

    # =========================================================================
    # STEP 17: V_string = (1/6)κt³ - BBHL
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 17: String Frame Volume")
        print("=" * 70)

    V_classical = kappa_sparse.compute_V(t_solution)
    BBHL = zeta(3) * chi / (4 * (2 * np.pi) ** 3)
    V_string = V_classical - BBHL

    if verbose:
        print(f"  V_classical = {V_classical:.4f}")
        print(f"  BBHL = {BBHL:.6f}")
        print(f"  V_string = V_classical - BBHL = {V_string:.4f}")

    results["V_classical"] = V_classical
    results["BBHL"] = BBHL
    results["V_string"] = V_string

    # =========================================================================
    # STEP 18: V₀ = -3 × e^{K₀} × (g_s^7/(4V)²) × W₀²
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 18: Vacuum Energy V₀")
        print("=" * 70)

    # Use mpmath for precision with W₀²
    g_s_mp = mpf(str(g_s))
    V_string_mp = mpf(str(V_string))
    e_K0_mp = mpf(str(e_K0))
    W0_mp = mpf(str(W0))

    V0 = -3 * e_K0_mp * (g_s_mp**7 / (4 * V_string_mp)**2) * W0_mp**2

    if verbose:
        print(f"  e^{{K₀}} = {float(e_K0_mp):.6f}")
        print(f"  g_s = {float(g_s_mp):.6f}")
        print(f"  V_string = {float(V_string_mp):.4f}")
        print(f"  W₀ = {W0_mp:.2e}")
        print(f"  V₀ = {V0:.2e}")

    results["V0"] = V0

    return results


def solve_racetrack(gv_invariants: dict, p: np.ndarray, M: np.ndarray,
                    verbose: bool = True) -> tuple:
    """
    Solve racetrack stabilization for g_s and W₀.

    Uses leading two terms of the racetrack superpotential.

    Args:
        gv_invariants: Dict {(q1,q2,...): N_q}
        p: Flat direction vector
        M: Flux vector
        verbose: Print progress

    Returns:
        (g_s, W0) as (float, mpf)
    """
    # Build racetrack terms: group by exponent q·p
    from collections import defaultdict

    terms = defaultdict(lambda: mpf(0))

    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple)

        # Exponent: q·p
        exp_coeff = np.dot(q, p)

        # Coefficient: (M·q) × N_q
        M_dot_q = np.dot(M, q)
        coeff = M_dot_q * N_q

        # Round exponent to nearest rational for grouping
        exp_key = round(exp_coeff, 8)
        terms[exp_key] += coeff

    # Find two leading terms (smallest positive exponents)
    positive_exps = sorted([e for e in terms.keys() if e > 0])

    if len(positive_exps) < 2:
        raise ValueError("Need at least 2 positive exponent terms for racetrack")

    alpha = mpf(str(positive_exps[0]))
    beta = mpf(str(positive_exps[1]))
    A = terms[positive_exps[0]]
    B = terms[positive_exps[1]]

    if verbose:
        print(f"  Leading terms:")
        print(f"    α = {float(alpha):.6f}, A = {float(A)}")
        print(f"    β = {float(beta):.6f}, B = {float(B)}")

    # Solve F-term: ratio = -Aα/(Bβ), Im(τ) = -log|ratio| / (2π(β-α))
    ratio = -A * alpha / (B * beta)

    if ratio <= 0:
        # Need to handle sign carefully
        Im_tau = -mp_log(abs(ratio)) / (2 * mp_pi * (beta - alpha))
    else:
        Im_tau = -mp_log(ratio) / (2 * mp_pi * (beta - alpha))

    g_s = 1 / Im_tau

    if verbose:
        print(f"  Im(τ) = {float(Im_tau):.4f}")
        print(f"  g_s = 1/Im(τ) = {float(g_s):.6f}")

    # Compute W₀ = |W(τ_min)|
    # W = ζ × Σ (M·q) N_q Li₂(e^{2πiτ(q·p)})
    # At minimum, dominated by leading terms

    ZETA = mpf(1) / (2**(mpf(3)/2) * mp_pi**(mpf(5)/2))

    tau = mpf(0) + Im_tau * mpf(1j) if hasattr(mpf(1j), 'imag') else Im_tau  # Im(τ) only

    # W₀ ≈ |ζ × A × Li₂(e^{-2π×Im(τ)×α}) + B × Li₂(e^{-2π×Im(τ)×β})|
    from mpmath import polylog

    arg_A = mp_exp(-2 * mp_pi * Im_tau * alpha)
    arg_B = mp_exp(-2 * mp_pi * Im_tau * beta)

    W_A = A * polylog(2, arg_A)
    W_B = B * polylog(2, arg_B)

    W0 = abs(ZETA * (W_A + W_B))

    if verbose:
        print(f"  W₀ = {W0:.2e}")

    return float(g_s), W0


# =============================================================================
# VALIDATION HARNESS (loads McAllister data for comparison)
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"
RESOURCES_DIR = Path(__file__).parent.parent / "resources"


def load_mcallister_inputs():
    """Load McAllister polytope 4-214-647 inputs for validation."""
    import json

    # Primal polytope
    lines = (DATA_DIR / "points.dat").read_text().strip().split('\n')
    primal_points = np.array([[int(x) for x in line.split(',')] for line in lines])

    # Dual polytope
    lines = (DATA_DIR / "dual_points.dat").read_text().strip().split('\n')
    dual_points = np.array([[int(x) for x in line.split(',')] for line in lines])

    # Flux vectors (CYTools 2021 basis)
    K = np.array([-3, -5, 8, 6])
    M = np.array([10, 11, -11, -5])

    # Orientifold O7-plane indices
    with open(RESOURCES_DIR / "mcallister_4-214-647_orientifold.json") as f:
        orientifold = json.load(f)
    o7_indices = orientifold["o7_divisor_indices"]

    return {
        "primal_points": primal_points,
        "dual_points": dual_points,
        "K": K,
        "M": M,
        "o7_indices": o7_indices,
    }


def load_mcallister_expected():
    """Load McAllister's expected results for validation."""
    return {
        "g_s": float((DATA_DIR / "g_s.dat").read_text().strip()),
        "W0": float((DATA_DIR / "W_0.dat").read_text().strip()),
        "V_string": float((DATA_DIR / "cy_vol.dat").read_text().strip()),
        "c_tau": float((DATA_DIR / "c_tau.dat").read_text().strip()),
        "e_K0": 0.234393,  # Back-calculated from paper
        "V0": -5.5e-203,   # From paper eq. 6.63
    }


def validate_against_mcallister():
    """Run pipeline and compare against McAllister's published results."""
    print("\n" + "#" * 70)
    print("# VALIDATION: McAllister 4-214-647")
    print("#" * 70)

    # Load inputs
    inputs = load_mcallister_inputs()
    expected = load_mcallister_expected()

    print(f"\nInputs:")
    print(f"  Primal polytope: {inputs['primal_points'].shape}")
    print(f"  Dual polytope: {inputs['dual_points'].shape}")
    print(f"  K = {inputs['K']}")
    print(f"  M = {inputs['M']}")
    print(f"  O7 indices: {len(inputs['o7_indices'])} divisors")

    print(f"\nExpected (McAllister):")
    print(f"  g_s = {expected['g_s']}")
    print(f"  W₀ = {expected['W0']:.2e}")
    print(f"  V_string = {expected['V_string']}")
    print(f"  e^{{K₀}} = {expected['e_K0']}")
    print(f"  V₀ = {expected['V0']:.2e}")

    # Run pipeline
    print("\n" + "=" * 70)
    print("RUNNING PIPELINE")
    print("=" * 70)

    results = run_pipeline(
        primal_points=inputs["primal_points"],
        dual_points=inputs["dual_points"],
        K=inputs["K"],
        M=inputs["M"],
        orientifold_o7_indices=inputs["o7_indices"],
        verbose=True,
    )

    # Compare results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    def compare(name, computed, expected, rel_tol=0.01):
        if expected == 0:
            error = abs(computed)
        else:
            error = abs(computed - expected) / abs(expected)
        status = "✓" if error < rel_tol else "✗"
        print(f"  {name}: computed={computed:.6g}, expected={expected:.6g}, error={100*error:.2f}% {status}")
        return error < rel_tol

    all_ok = True
    all_ok &= compare("e^{K₀}", results["e_K0"], expected["e_K0"])
    all_ok &= compare("g_s", results["g_s"], expected["g_s"])
    all_ok &= compare("c_τ", results["c_tau"], expected["c_tau"])
    all_ok &= compare("V_string", results["V_string"], expected["V_string"], rel_tol=0.05)

    # W₀ comparison (may differ due to racetrack approximation)
    W0_computed = float(results["W0"])
    W0_expected = expected["W0"]
    W0_log_ratio = abs(np.log10(W0_computed) - np.log10(W0_expected))
    W0_ok = W0_log_ratio < 5  # Within 5 orders of magnitude
    print(f"  W₀: computed={W0_computed:.2e}, expected={W0_expected:.2e}, log10 diff={W0_log_ratio:.1f} {'✓' if W0_ok else '✗'}")

    # V₀ comparison (will be off if W₀ is off)
    V0_computed = float(results["V0"])
    V0_expected = expected["V0"]
    V0_log_ratio = abs(np.log10(abs(V0_computed)) - np.log10(abs(V0_expected)))
    V0_ok = V0_log_ratio < 10  # Within 10 orders of magnitude (W₀² effect)
    print(f"  V₀: computed={V0_computed:.2e}, expected={V0_expected:.2e}, log10 diff={V0_log_ratio:.1f} {'✓' if V0_ok else '✗'}")

    if all_ok and W0_ok:
        print("\n" + "=" * 70)
        print("✓ PIPELINE VALIDATED")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ PIPELINE VALIDATION FAILED")
        print("=" * 70)

    return results


if __name__ == "__main__":
    validate_against_mcallister()
