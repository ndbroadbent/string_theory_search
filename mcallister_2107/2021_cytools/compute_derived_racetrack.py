#!/usr/bin/env python3
"""
Compute racetrack stabilization: g_s and W₀ from flux and GV invariants.

Steps 12-14 of the pipeline:
- Step 12: Build racetrack from GV invariants
- Step 13: Solve F-term for g_s = 1/Im(τ)
- Step 14: Compute W₀ = |W(τ_min)|

CRITICAL: Basis transformation between CYTools versions!
- p is computed from CYTools 2021's kappa (in 2021's divisor basis)
- GV curves from compute_gvs() are in latest's curve basis
- Must transform p and M to latest's basis before computing q·p and M·q

Reference: arXiv:2107.09064 section 5
"""

import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
from mpmath import mp, mpf, polylog, pi as mp_pi, exp as mp_exp
from scipy.optimize import minimize_scalar

# Import basis transformation utilities - DON'T REIMPLEMENT
from compute_basis_transform import compute_T_from_glsm, transform_fluxes

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"
CYTOOLS_LATEST = ROOT_DIR / "vendor/cytools_latest/src"
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

MCALLISTER_EXAMPLES = [
    ("4-214-647", 4, 214),
    ("5-113-4627-main", 5, 113),
    ("5-113-4627-alternative", 5, 113),
    ("5-81-3213", 5, 81),
    ("7-51-13590", 7, 51),
]

# High precision for W₀ ~ 10^{-90}
mp.dps = 150

# Constant from eq. 2.22: ζ = 1/(2^{3/2} π^{5/2})
ZETA = mpf(1) / (mpf(2) ** mpf('1.5') * mp_pi ** mpf('2.5'))


def load_dual_points(example_name: str) -> np.ndarray:
    """Load dual polytope points from McAllister's data."""
    lines = (DATA_BASE / example_name / "dual_points.dat").read_text().strip().split('\n')
    return np.array([[int(x) for x in line.split(',')] for line in lines])


def load_simplices(example_name: str) -> list:
    """Load triangulation simplices from McAllister's data."""
    lines = (DATA_BASE / example_name / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]


def load_mcallister_curves(example_name: str) -> tuple:
    """
    Load McAllister's pre-computed curves and GV invariants.

    These are in AMBIENT coordinates (h21+5 components), which are
    version-independent. We'll transform to internal basis using
    the curve_basis_mat from CYTools latest.
    """
    data_dir = DATA_BASE / example_name

    curves = []
    with open(data_dir / "dual_curves.dat") as f:
        for line in f:
            curves.append(np.array([int(x) for x in line.strip().split(",")]))

    with open(data_dir / "dual_curves_gv.dat") as f:
        gv_values = [int(Decimal(x)) for x in f.read().strip().split(",")]

    return curves, gv_values


def load_model_choices(example_name: str) -> dict:
    """Load K, M, g_s, W0 from McAllister's data."""
    data_dir = DATA_BASE / example_name
    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(",")])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(",")])
    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    return {"K": K, "M": M, "g_s": g_s, "W0": W0}


def get_2021_kappa_and_basis(dual_pts: np.ndarray, simplices: list) -> tuple:
    """
    Get kappa tensor and divisor basis from CYTools 2021.

    Returns:
        (kappa, divisor_basis) where kappa is h11 x h11 x h11 tensor
    """
    # Clear cached modules
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


def get_latest_curve_basis_and_cy(dual_pts: np.ndarray, simplices: list) -> tuple:
    """
    Get curve basis matrix and CY object from CYTools latest.

    Returns:
        (curve_basis_mat, divisor_basis, cy)
    Note: Does NOT remove CYTools from sys.path (caller may need it for compute_T_from_glsm)
    """
    # Clear cached modules
    mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for m in mods:
        del sys.modules[m]

    sys.path.insert(0, str(CYTOOLS_LATEST))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    curve_basis_mat = cy.curve_basis(include_origin=True, as_matrix=True)
    divisor_basis = list(cy.divisor_basis())

    # Note: Don't remove CYTOOLS_LATEST from path yet - we may need it for compute_T_from_glsm
    return curve_basis_mat, divisor_basis, cy


# compute_basis_transformation removed - use compute_T_from_glsm from compute_basis_transform


def compute_p_vector(kappa: np.ndarray, K: np.ndarray, M: np.ndarray) -> tuple:
    """
    Compute flat direction vector p = N^{-1} K where N = κ_abc M^c.

    Returns:
        (p, N, det_N)
    """
    N = np.einsum('abc,c->ab', kappa, M)
    det_N = np.linalg.det(N)

    if abs(det_N) < 1e-10:
        raise ValueError(f"N matrix is singular (det={det_N})")

    p = np.linalg.solve(N, K)
    return p, N, det_N


def compute_racetrack_terms(
    curves_ambient: list,
    gv_values: list,
    curve_basis_pinv: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
) -> list:
    """
    Compute racetrack terms from GV invariants.

    For each curve, computes:
    - q·p (exponent in e^{2πiτ(q·p)})
    - M·q (coefficient factor)
    - N_q (GV invariant)

    Returns list of (q_dot_p, coefficient) tuples, grouped by exponent.
    """
    terms_dict = defaultdict(lambda: 0)

    for q_ambient, N_q in zip(curves_ambient, gv_values):
        # Transform curve from ambient to internal basis
        q = q_ambient @ curve_basis_pinv
        q_int = np.round(q).astype(int)

        # Compute products
        qp = float(np.dot(q_int, p))
        Mq = int(np.dot(M, q_int))
        coeff = Mq * N_q

        if abs(coeff) > 0 and qp > 0:
            exp_key = round(qp, 8)
            terms_dict[exp_key] += coeff

    # Convert to sorted list of (exponent, coefficient) pairs
    terms = sorted([(e, c) for e, c in terms_dict.items() if abs(c) > 0], key=lambda x: x[0])
    return terms


def compute_W_at_Im_tau(terms: list, Im_tau: float) -> mpf:
    """
    Compute |W(τ)| at given Im(τ), assuming Re(τ) = 0.

    W = -ζ Σ_q (M·q) N_q Li₂(e^{2πiτ(q·p)})

    For τ = i*Im(τ): e^{2πiτ(q·p)} = e^{-2π*Im(τ)*(q·p)}
    """
    W_sum = mpf(0)
    Im_tau_mp = mpf(str(Im_tau))

    for exp_coeff, coeff in terms:
        arg = mp_exp(-2 * mp_pi * Im_tau_mp * mpf(str(exp_coeff)))
        W_sum += mpf(str(float(coeff))) * polylog(2, arg)

    return abs(-ZETA * W_sum)


def compute_W_at_complex_tau(terms: list, Re_tau: float, Im_tau: float) -> mpf:
    """
    Compute |W(τ)| at given complex τ = Re_tau + i*Im_tau.

    W = -ζ Σ_q (M·q) N_q Li₂(e^{2πiτ(q·p)})

    For τ = x + iy:
        e^{2πiτ α} = e^{2πi(x+iy)α} = e^{-2παy} × e^{2πiαx}

    The argument to Li₂ is complex when Re(τ) ≠ 0.
    """
    from mpmath import mpc, exp as mp_exp_c

    W_sum = mpc(0)
    Re_tau_mp = mpf(str(Re_tau))
    Im_tau_mp = mpf(str(Im_tau))

    for alpha, coeff in terms:
        alpha_mp = mpf(str(alpha))
        # e^{2πiτα} = e^{-2παy} × e^{2πiαx}
        magnitude = mp_exp(-2 * mp_pi * Im_tau_mp * alpha_mp)
        phase = 2 * mp_pi * alpha_mp * Re_tau_mp
        arg = magnitude * mp_exp_c(mpc(0, phase))
        W_sum += mpf(str(float(coeff))) * polylog(2, arg)

    return abs(-ZETA * W_sum)


def compute_W_derivative_at_Im_tau(terms: list, Im_tau: float) -> mpf:
    """
    Compute dW/d(Im_tau) at given Im(τ).

    For W = -ζ Σ (M·q) N_q Li₂(e^{-2π Im(τ) (q·p)}):
    dW/d(Im_tau) = -ζ Σ (M·q) N_q × dLi₂/dx × dx/d(Im_tau)

    where x = e^{-2π Im(τ) (q·p)} and:
    - dLi₂/dx = Li₁(x) / x = -ln(1-x) / x
    - dx/d(Im_tau) = -2π (q·p) × e^{-2π Im(τ) (q·p)}

    So: dW/d(Im_tau) = ζ Σ (M·q) N_q × 2π (q·p) × ln(1-x)
    """
    dW_sum = mpf(0)
    Im_tau_mp = mpf(str(Im_tau))

    for exp_coeff, coeff in terms:
        alpha = mpf(str(exp_coeff))
        x = mp_exp(-2 * mp_pi * Im_tau_mp * alpha)
        # dW/d(Im_tau) term: 2π α × coeff × ln(1-x)
        from mpmath import log
        if x < mpf('1e-300'):
            # For very small x, ln(1-x) ≈ -x, so term ≈ 2π α × coeff × (-x)
            dW_sum += -2 * mp_pi * alpha * mpf(str(float(coeff))) * x
        else:
            dW_sum += 2 * mp_pi * alpha * mpf(str(float(coeff))) * log(1 - x)

    return ZETA * dW_sum


def solve_racetrack(terms: list, verbose: bool = True) -> dict:
    """
    Solve racetrack for g_s and W₀ using F-term stabilization.

    From McAllister eq. 2.22, the flux superpotential is:
        W = -ζ Σ (M·q) N_q Li₂(e^{2πiτ(q·p)})

    The F-term ∂W/∂τ = 0 gives (for the two leading terms q₁, q₂):
        (M·q₁)(p·q₁)N_{q₁} × e^{2πiτα} + (M·q₂)(p·q₂)N_{q₂} × e^{2πiτβ} = 0

    This defines δ (eq. 2.25):
        δ = -[(M·q₁)(p·q₁)N_{q₁}] / [(M·q₂)(p·q₂)N_{q₂}]

    The solution (eq. 2.26) is:
        ⟨e^{2πiτ}⟩ ≈ δ^{1/ε}  where ε = β - α

    For opposite-sign W coefficients: δ > 0, solution is purely imaginary τ
    For same-sign W coefficients: δ < 0, solution has Re(τ) = 1/(2ε)

    In both cases:
        Im(τ) = ln(1/|δ|) / (2πε)
        g_s = 1/Im(τ)

    Returns dict with g_s, W0, and diagnostic info.
    """
    if len(terms) < 2:
        return {
            "success": False,
            "error": f"Need at least 2 terms, found {len(terms)}",
        }

    # Get the two leading terms
    # terms contains (exponent, superpotential_coefficient) where coeff = (M·q)×N_q
    alpha, coeff_alpha = terms[0]
    beta, coeff_beta = terms[1]

    if verbose:
        print(f"  Racetrack has {len(terms)} unique exponents")
        print(f"  Leading terms (α=q·p, coeff=(M·q)×N_q):")
        for e, c in terms[:5]:
            print(f"    α={e:.6f}, coeff={int(c)}")

    # Compute ε = β - α (exponent gap)
    epsilon = beta - alpha

    if epsilon <= 0:
        return {
            "success": False,
            "error": f"Invalid exponent ordering: α={alpha}, β={beta}",
        }

    # Compute δ from eq. 2.25:
    # δ = -[(M·q₁)(p·q₁)N_{q₁}] / [(M·q₂)(p·q₂)N_{q₂}]
    #   = -(coeff_alpha × α) / (coeff_beta × β)
    # Note: coeff = (M·q)×N_q, so coeff×α = (M·q)×(p·q)×N_q
    derivative_coeff_1 = coeff_alpha * alpha  # (M·q₁)(p·q₁)N_{q₁}
    derivative_coeff_2 = coeff_beta * beta    # (M·q₂)(p·q₂)N_{q₂}

    if abs(derivative_coeff_2) < 1e-10:
        return {
            "success": False,
            "error": f"Second term has zero derivative coefficient",
        }

    delta = -derivative_coeff_1 / derivative_coeff_2

    if verbose:
        print(f"\n  Racetrack parameters:")
        print(f"    α = {alpha:.6f}, β = {beta:.6f}")
        print(f"    ε = β - α = {epsilon:.6f}")
        print(f"    derivative coeff₁ = {derivative_coeff_1:.2f}")
        print(f"    derivative coeff₂ = {derivative_coeff_2:.2f}")
        print(f"    δ = {delta:.6f}")

    # Check |δ| < 1 (eq. 2.25 requirement)
    if abs(delta) >= 1:
        if verbose:
            print(f"  WARNING: |δ| = {abs(delta):.4f} >= 1, racetrack may not work well")

    # From eq. 2.26: ⟨e^{2πiτ}⟩ ≈ δ^{1/ε}
    # Taking log: 2πiτ = (1/ε) × ln(δ)
    # For δ < 0: ln(δ) = ln|δ| + iπ
    # So: τ = [ln|δ| + iπ] / (2πiε) = 1/(2ε) - i×ln|δ|/(2πε)
    # Therefore: Im(τ) = -ln|δ|/(2πε) = ln(1/|δ|)/(2πε)

    abs_delta = abs(delta)
    if abs_delta <= 0:
        return {
            "success": False,
            "error": f"δ = 0, cannot solve racetrack",
        }

    Im_tau = np.log(1.0 / abs_delta) / (2 * np.pi * epsilon)

    if verbose:
        if delta < 0:
            print(f"  Same-sign coefficients: δ < 0")
            print(f"    Re(τ) = 1/(2ε) = {1/(2*epsilon):.4f}")
        else:
            print(f"  Opposite-sign coefficients: δ > 0")
            print(f"    Re(τ) = 0")
        print(f"  Im(τ) = ln(1/|δ|)/(2πε) = {Im_tau:.4f}")

    if Im_tau <= 0:
        return {
            "success": False,
            "error": f"Im(τ) = {Im_tau:.4f} <= 0, no valid stabilization",
        }

    g_s = 1.0 / Im_tau

    if verbose:
        print(f"  g_s = 1/Im(τ) = {g_s:.6f}")

    # Compute W₀ at the stabilization point
    # For δ < 0 (same-sign coefficients), τ has a non-zero real part
    if delta < 0:
        Re_tau = 1.0 / (2 * epsilon)
        W0 = abs(compute_W_at_complex_tau(terms, Re_tau, Im_tau))
        if verbose:
            print(f"  Computing W₀ at complex τ = {Re_tau:.4f} + {Im_tau:.4f}i")
    else:
        Re_tau = 0.0
        W0 = abs(compute_W_at_Im_tau(terms, Im_tau))
        if verbose:
            print(f"  Computing W₀ at τ = {Im_tau:.4f}i")

    if verbose:
        print(f"  W₀ = |W(τ)| = {float(W0):.2e}")

    return {
        "success": True,
        "g_s": g_s,
        "W0": W0,
        "Re_tau": Re_tau,
        "Im_tau": Im_tau,
        "delta": delta,
        "epsilon": epsilon,
        "n_terms": len(terms),
        "leading_exponents": [e for e, c in terms[:3]],
    }


def test_example(example_name: str, expected_h11: int, expected_h21: int,
                 verbose: bool = True) -> dict:
    """
    Test racetrack computation for one McAllister example.

    Computes g_s and W0 from scratch, compares to McAllister's values.
    """
    if verbose:
        print("=" * 70)
        print(f"RACETRACK TEST - {example_name} (h11={expected_h11})")
        print("=" * 70)

    # Load data
    dual_pts = load_dual_points(example_name)
    simplices = load_simplices(example_name)
    model = load_model_choices(example_name)
    curves_ambient, gv_values = load_mcallister_curves(example_name)

    if verbose:
        print(f"\nModel inputs:")
        print(f"  K = {model['K']}")
        print(f"  M = {model['M']}")
        print(f"  Curves: {len(curves_ambient)}")

    # Get kappa from CYTools 2021
    kappa, basis_2021 = get_2021_kappa_and_basis(dual_pts, simplices)

    # Get curve basis and CY object from CYTools latest
    curve_basis_mat, basis_latest, cy_latest = get_latest_curve_basis_and_cy(dual_pts, simplices)
    curve_basis_pinv = np.linalg.pinv(curve_basis_mat)

    if verbose:
        print(f"\nBases:")
        print(f"  2021: {basis_2021}")
        print(f"  Latest: {basis_latest}")

    # Compute p in 2021's basis
    p_2021, N, det_N = compute_p_vector(kappa, model['K'], model['M'])

    if verbose:
        print(f"\np (2021 basis) = {p_2021}")

    # Compute basis transformation if needed
    if basis_2021 != basis_latest:
        T = compute_T_from_glsm(cy_latest, basis_2021, basis_latest)

        # Transform p and M to latest's basis (contravariant: use T.T)
        p_latest = T.T @ p_2021
        M_latest = T.T @ model['M']

        if verbose:
            print(f"p (latest basis) = {p_latest}")
            print(f"M (latest basis) = {M_latest}")
    else:
        p_latest = p_2021
        M_latest = model['M']

    # Build racetrack terms
    if verbose:
        print("\nBuilding racetrack...")

    terms = compute_racetrack_terms(
        curves_ambient, gv_values, curve_basis_pinv, p_latest, M_latest
    )

    # Solve racetrack
    if verbose:
        print("\nSolving racetrack...")

    result = solve_racetrack(terms, verbose=verbose)

    if not result["success"]:
        return result

    # Compare to expected values
    g_s_expected = model["g_s"]
    W0_expected = model["W0"]

    if verbose:
        print(f"\nExpected (from McAllister):")
        print(f"  g_s = {g_s_expected:.6f}")
        print(f"  W₀ = {W0_expected:.2e}")

    # Compute ratios
    g_s_ratio = result["g_s"] / g_s_expected
    W0_ratio = float(result["W0"]) / abs(W0_expected)

    if verbose:
        print(f"\nComparison:")
        print(f"  g_s ratio (computed/expected): {g_s_ratio:.4f}")
        print(f"  W₀ ratio (computed/expected): {W0_ratio:.4f}")

    # Pass criteria: within 10% for g_s, order of magnitude for W0
    g_s_ok = abs(g_s_ratio - 1.0) < 0.1
    W0_ok = 0.1 < W0_ratio < 10.0  # Within order of magnitude
    test_passed = g_s_ok and W0_ok

    result.update({
        "example_name": example_name,
        "g_s_expected": g_s_expected,
        "W0_expected": W0_expected,
        "g_s_ratio": g_s_ratio,
        "W0_ratio": W0_ratio,
        "test_passed": test_passed,
    })

    if verbose:
        status = "PASS" if test_passed else "FAIL"
        print(f"\n{status}: Racetrack test {'passed' if test_passed else 'failed'}")

    return result


def main():
    """Test racetrack for all 5 McAllister examples."""
    print("=" * 70)
    print("TESTING ALL 5 MCALLISTER EXAMPLES - Steps 12-14: Racetrack")
    print("=" * 70)
    print("\nStrategy: Use McAllister's ambient curves, transform to consistent basis")
    print("Settings: mpmath dps=150\n")

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, h21, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY - Steps 12-14: Racetrack")
    print("=" * 70)
    all_passed = True
    for r in results:
        if r.get("success", False):
            status = "PASS" if r.get("test_passed", False) else "FAIL"
            g_s_str = f"g_s={r['g_s']:.6f}"
            W0_exp = np.log10(float(r['W0'])) if float(r['W0']) > 0 else float('-inf')
            W0_str = f"W₀~10^{W0_exp:.0f}"
            ratio_str = f"(ratio: g_s={r['g_s_ratio']:.3f}, W0={r['W0_ratio']:.3f})"
            print(f"  {status}: {r['example_name']}: {g_s_str}, {W0_str} {ratio_str}")
        else:
            print(f"  FAIL: {r.get('example_name', '?')}: {r.get('error', 'unknown')}")
        all_passed = all_passed and r.get("test_passed", False)

    print()
    if all_passed:
        print("All 5 examples PASSED Steps 12-14")
    else:
        print("Some examples FAILED")

    return results


if __name__ == "__main__":
    main()
