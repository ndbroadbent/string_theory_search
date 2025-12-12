#!/usr/bin/env python3
"""
Compute racetrack stabilization: g_s and W₀ from flux and GV invariants.

Steps 12-14 of the pipeline:
- Step 12: Build racetrack from GV invariants
- Step 13: Solve F-term for g_s = 1/Im(τ)
- Step 14: Compute W₀ = |W(τ_min)|

Reference: arXiv:2107.09064 section 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_latest/src"))

import numpy as np
from collections import defaultdict
from mpmath import mp, mpf, polylog, pi as mp_pi, log as mp_log, exp as mp_exp

from cytools import Polytope

from compute_triangulation import (
    MCALLISTER_EXAMPLES,
    DATA_BASE,
    load_example_points,
    load_example_model_choices,
    load_example_simplices,
)
from compute_basis_transform import load_mcallister_example
from compute_gv_invariants import compute_gv_invariants


def load_simplices_list(example_name: str) -> list:
    """Load McAllister's triangulation simplices as a list of lists."""
    data_dir = DATA_BASE / example_name
    lines = (data_dir / "dual_simplices.dat").read_text().strip().split('\n')
    return [[int(x) for x in line.split(',')] for line in lines]

# High precision for W₀ ~ 10^{-90}
mp.dps = 150

# Constant from eq. 2.22
ZETA = mpf(1) / (mpf(2) ** mpf('1.5') * mp_pi ** mpf('2.5'))


def compute_W0_from_gv(
    gv_invariants: dict,
    p: np.ndarray,
    M: np.ndarray,
    Im_tau: float,
    verbose: bool = True,
) -> mpf:
    """
    Compute W₀ = |W(τ)| from GV invariants at given Im(τ).

    The superpotential (eq. 5.4):
    W = -ζ Σ_q (M·q) N_q Li₂(e^{2πiτ(q·p)})

    For τ = i*Im(τ): e^{2πiτ(q·p)} = e^{-2π*Im(τ)*(q·p)}

    Args:
        gv_invariants: Dict {(q1,q2,...): N_q} from CYTools
        p: Flat direction vector (h11 components)
        M: Flux vector M (h11 components)
        Im_tau: Imaginary part of τ (= 1/g_s)
        verbose: Print progress

    Returns:
        W₀ = |W(τ)| as mpf
    """
    W_sum = mpf(0)
    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple)
        exp_coeff = float(np.dot(q, p))  # q·p
        M_dot_q = int(np.dot(M, q))      # M·q
        coeff = M_dot_q * N_q            # (M·q) * N_q

        if abs(coeff) > 0 and exp_coeff > 0:
            arg = mp_exp(-2 * mp_pi * mpf(str(Im_tau)) * mpf(str(exp_coeff)))
            W_sum += mpf(str(float(coeff))) * polylog(2, arg)

    return abs(-ZETA * W_sum)


def compute_racetrack(
    gv_invariants: dict,
    p: np.ndarray,
    M: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Build racetrack and solve for g_s and W₀ by numerical minimization.

    The racetrack superpotential (eq. 5.4):
    W = -ζ Σ_q (M·q) N_q Li₂(e^{2πiτ(q·p)})

    We find Im(τ) that minimizes |W(τ)| using all GV terms.

    Args:
        gv_invariants: Dict {(q1,q2,...): N_q} from CYTools
        p: Flat direction vector (h11 components)
        M: Flux vector M (h11 components)
        verbose: Print progress

    Returns:
        Dict with g_s, W0, success, and intermediate values
    """
    from scipy.optimize import minimize_scalar

    # Build racetrack terms: list of (exponent, coefficient) pairs
    # Group by exponent q·p
    terms_dict = defaultdict(lambda: 0)

    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple)
        exp_coeff = float(np.dot(q, p))  # q·p
        M_dot_q = int(np.dot(M, q))      # M·q
        coeff = M_dot_q * N_q            # (M·q) * N_q

        if abs(coeff) > 0 and exp_coeff > 0:
            exp_key = round(exp_coeff, 8)
            terms_dict[exp_key] += coeff

    # Convert to sorted list of (exponent, coefficient) pairs
    terms = sorted([(e, c) for e, c in terms_dict.items() if abs(c) > 0], key=lambda x: x[0])

    if len(terms) < 2:
        return {
            "success": False,
            "error": f"Need at least 2 positive exponent terms, found {len(terms)}",
        }

    if verbose:
        print(f"  Racetrack has {len(terms)} terms")
        print(f"  Leading terms (α, coeff):")
        for e, c in terms[:5]:
            print(f"    α={e:.6f}, coeff={int(c)}")

    # Define |W(Im_tau)| for minimization
    def W_magnitude(Im_tau_val):
        if Im_tau_val <= 0:
            return float('inf')
        Im_tau = mpf(str(Im_tau_val))
        W_sum = mpf(0)
        for exp_coeff, coeff in terms:
            arg = mp_exp(-2 * mp_pi * Im_tau * mpf(str(exp_coeff)))
            W_sum += mpf(str(float(coeff))) * polylog(2, arg)
        return float(abs(W_sum))

    # Search for minimum in reasonable range
    # g_s typically 0.001 to 0.1, so Im_tau = 1/g_s in range 10 to 1000
    result = minimize_scalar(W_magnitude, bounds=(10, 500), method='bounded')

    if not result.success:
        return {"success": False, "error": "Minimization failed"}

    Im_tau_min = mpf(str(result.x))
    g_s = 1 / Im_tau_min

    if verbose:
        print(f"  Minimization found Im(τ) = {float(Im_tau_min):.4f}")
        print(f"  g_s = 1/Im(τ) = {float(g_s):.6f}")

    # Compute W₀ at the minimum
    W0 = compute_W0_from_gv(gv_invariants, p, M, float(Im_tau_min), verbose=False)

    if verbose:
        print(f"  W₀ = |W(τ)| = {float(W0):.2e}")

    return {
        "success": True,
        "g_s": float(g_s),
        "W0": W0,  # Keep as mpf for precision
        "Im_tau": float(Im_tau_min),
        "n_terms": len(terms),
        "leading_exponents": [e for e, c in terms[:3]],
    }


def test_example(example_name: str, expected_h11: int, simplices: list = None, verbose: bool = True) -> dict:
    """
    Test racetrack computation for one McAllister example.

    Args:
        example_name: Folder name in paper_data/
        expected_h11: Expected h11 for the DUAL polytope
        simplices: Optional triangulation simplices (if None, CYTools picks)
        verbose: Print progress

    Returns:
        Dict with computed and expected g_s, W0
    """
    data_dir = DATA_BASE / example_name

    if verbose:
        print("=" * 70)
        print(f"RACETRACK TEST - {example_name} (h11={expected_h11})")
        print("=" * 70)

    # Load polytope and get CY
    dual_pts = load_example_points(example_name, which="dual")

    # Get old basis from CYTools 2021 (for flux transformation)
    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/cytools_mcallister_2107"))
    mods_to_remove = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for mod in mods_to_remove:
        del sys.modules[mod]

    from cytools import Polytope as Polytope2021
    poly_2021 = Polytope2021(dual_pts)
    tri_2021 = poly_2021.triangulate()
    cy_2021 = tri_2021.get_cy()
    old_basis = list(cy_2021.divisor_basis())

    # Restore latest CYTools
    mods_to_remove = [k for k in list(sys.modules.keys()) if 'cytools' in k]
    for mod in mods_to_remove:
        del sys.modules[mod]
    sys.path.pop(0)

    from cytools import Polytope as PolytopeNew
    poly_new = PolytopeNew(dual_pts)

    # Use provided triangulation or let CYTools pick
    if simplices is not None:
        tri_new = poly_new.triangulate(simplices=simplices, check_input_simplices=False)
        if verbose:
            print(f"\nUsing provided triangulation ({len(simplices)} simplices)")
    else:
        tri_new = poly_new.triangulate()
        if verbose:
            print(f"\nUsing CYTools default triangulation")

    cy_new = tri_new.get_cy()
    new_basis = list(cy_new.divisor_basis())

    if verbose:
        print(f"CY: h11={cy_new.h11()}, h21={cy_new.h21()}")
        print(f"Old basis (2021): {old_basis}")
        print(f"New basis (latest): {new_basis}")

    # Load example data and transform fluxes
    example_data = load_mcallister_example(example_name)
    from compute_basis_transform import compute_T_from_glsm, transform_fluxes

    K_old = example_data["K"]
    M_old = example_data["M"]

    if old_basis == new_basis:
        K_new, M_new = K_old, M_old
    else:
        T = compute_T_from_glsm(cy_new, old_basis, new_basis)
        K_new, M_new = transform_fluxes(K_old, M_old, T)

    if verbose:
        print(f"K (new basis): {K_new}")
        print(f"M (new basis): {M_new}")

    # Compute p = N^{-1} K
    kappa_dict = cy_new.intersection_numbers(in_basis=True)
    h11 = cy_new.h11()
    kappa = np.zeros((h11, h11, h11))
    for (i, j, k), val in kappa_dict.items():
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val

    N = np.einsum('abc,c->ab', kappa, M_new)
    det_N = np.linalg.det(N)

    if abs(det_N) < 1e-10:
        return {"success": False, "error": "N matrix is singular"}

    p = np.linalg.solve(N, K_new)

    if verbose:
        print(f"p = {p}")

    # Compute GV invariants
    gv = compute_gv_invariants(cy_new, min_points=100)

    if verbose:
        print(f"GV invariants: {len(gv)} non-zero")

    # Run racetrack
    if verbose:
        print("\nSolving racetrack...")

    result = compute_racetrack(gv, p, M_new, verbose=verbose)

    if not result["success"]:
        return result

    # Load expected values
    model = load_example_model_choices(example_name)
    g_s_expected = model["g_s"]
    W0_expected = model["W0"]

    if verbose:
        print(f"\nExpected:")
        print(f"  g_s = {g_s_expected:.6f}")
        print(f"  W₀ = {W0_expected:.2e}")

    # Compare
    g_s_ratio = result["g_s"] / g_s_expected if g_s_expected != 0 else float('inf')
    W0_log_diff = abs(np.log10(float(result["W0"])) - np.log10(abs(W0_expected)))

    if verbose:
        print(f"\nComparison:")
        print(f"  g_s ratio: {g_s_ratio:.4f}")
        print(f"  log10(W₀) diff: {W0_log_diff:.1f}")

    # Pass criteria: match to input precision (6 significant figures)
    g_s_ok = abs(result["g_s"] - g_s_expected) / g_s_expected < 1e-6
    W0_ok = abs(float(result["W0"]) - W0_expected) / W0_expected < 1e-6

    result["g_s_expected"] = g_s_expected
    result["W0_expected"] = W0_expected
    result["g_s_ratio"] = g_s_ratio
    result["W0_log_diff"] = W0_log_diff
    result["test_passed"] = g_s_ok and W0_ok

    if verbose:
        status = "✓" if result["test_passed"] else "✗"
        print(f"\n{status} Racetrack test {'PASSED' if result['test_passed'] else 'FAILED'}")

    return result


def main():
    """Test racetrack for all 5 McAllister examples."""
    print("=" * 70)
    print("TESTING ALL 5 MCALLISTER EXAMPLES - Steps 12-14: Racetrack")
    print("=" * 70)

    results = []
    for name, h11, _ in MCALLISTER_EXAMPLES:
        # Load McAllister's triangulation simplices for validation
        simplices = load_simplices_list(name)
        result = test_example(name, h11, simplices=simplices, verbose=True)
        results.append(result)
        result["example_name"] = name
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY - Steps 12-14: Racetrack")
    print("=" * 70)
    all_passed = True
    for result in results:
        if result.get("success", False):
            status = "✓" if result.get("test_passed", False) else "✗"
            g_s_str = f"g_s={result['g_s']:.4f}"
            W0_str = f"W₀~10^{np.log10(float(result['W0'])):.0f}"
            print(f"  {status} {result['example_name']}: {g_s_str}, {W0_str}")
        else:
            print(f"  ✗ {result['example_name']}: FAILED - {result.get('error', 'unknown')}")
        all_passed = all_passed and result.get("test_passed", False)

    print()
    if all_passed:
        print("All 5 examples PASSED Steps 12-14")
    else:
        print("Some examples FAILED")

    return results


if __name__ == "__main__":
    main()
