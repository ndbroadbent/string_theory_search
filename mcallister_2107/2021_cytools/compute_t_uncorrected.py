#!/usr/bin/env python3
"""
Compute "uncorrected" Kähler moduli t where τ = c_i FROM SCRATCH.

This is Phase 1 of the two-phase KKLT solver:
1. Solve τ = c_i (this script) → t_uncorrected
2. Path-follow from t_uncorrected to target τ (compute_kahler_param.py)

ALGORITHM: Multi-start path-following with V > 0 filter
- Try many random t_init
- Path-follow each to τ = c_i
- Keep solutions with V > 0 (correct branch)
- Return the one with highest V (most robust)

This is slow (~50-100 attempts needed) but there's no known faster algorithm.
See KKLT_SOLVER_RESEARCH.md for why other approaches fail.
"""

import sys
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"
CYTOOLS_2021 = ROOT_DIR / "vendor/cytools_mcallister_2107"

sys.path.insert(0, str(CYTOOLS_2021))

from cytools import Polytope

from compute_kahler_param import SparseKappa, _path_follow

MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4, True),
    ("5-113-4627-main", 113, 5, True),
    ("5-113-4627-alternative", 113, 5, True),
    ("5-81-3213", 81, 5, True),
    ("7-51-13590", 51, 7, False),
]


# =============================================================================
# CORE SOLVER
# =============================================================================


def solve_t_uncorrected(kappa: SparseKappa, c_i: np.ndarray,
                        n_attempts: int = 100, n_steps: int = 150,
                        verbose: bool = False) -> dict:
    """
    Solve for t where τ(t) = c_i (the dual Coxeter numbers).

    Uses multi-start path-following with V > 0 filter.

    Args:
        kappa: SparseKappa object with intersection numbers
        c_i: Dual Coxeter numbers (1 for D3, 6 for O7)
        n_attempts: Number of random initializations to try
        n_steps: Path-following steps per attempt
        verbose: Print progress

    Returns:
        dict with:
            success: bool
            t: solution array (or None)
            tau: achieved τ values
            V: classical volume
            n_converged: how many attempts converged
            n_positive_V: how many had V > 0
    """
    h11 = kappa.h11
    tau_target = c_i.astype(float)

    # Build kappa tensor for V computation
    kappa_tensor = np.zeros((h11, h11, h11))
    for i, j, k, val in kappa.kappa_basis_entries:
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            kappa_tensor[perm] = val

    def compute_V(t):
        return np.einsum('ijk,i,j,k->', kappa_tensor, t, t, t) / 6.0

    # Track results
    solutions = []  # (V, t, tau)
    n_converged = 0

    # Strategy 1: Scaled uniform initializations
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        t_init = np.ones(h11) * scale

        # Scale to match target τ magnitude
        tau_init = kappa.compute_tau_kklt(t_init)
        if np.mean(tau_init) > 0:
            adjust = np.sqrt(np.mean(tau_target) / np.mean(tau_init))
            if not np.isnan(adjust) and 0.01 < adjust < 100:
                t_init = t_init * adjust

        t_result, tau_result, conv = _path_follow(kappa, tau_target, t_init, n_steps)

        if conv and t_result is not None:
            n_converged += 1
            V = compute_V(t_result)
            solutions.append((V, t_result.copy(), tau_result.copy()))
            if verbose:
                print(f"  Uniform scale={scale:.1f}: V={V:.2f}")

    # Strategy 2: Random positive initializations
    np.random.seed(42)
    for attempt in range(n_attempts):
        # Random positive t with varying scale
        scale = 10 ** np.random.uniform(-0.5, 1.5)  # 0.3 to 30
        t_init = np.abs(np.random.randn(h11)) * scale + 0.1

        # Scale to match target τ magnitude
        tau_init = kappa.compute_tau_kklt(t_init)
        if np.mean(tau_init) > 0:
            adjust = np.sqrt(np.mean(tau_target) / np.mean(tau_init))
            if not np.isnan(adjust) and 0.01 < adjust < 100:
                t_init = t_init * adjust

        t_result, tau_result, conv = _path_follow(kappa, tau_target, t_init, n_steps)

        if conv and t_result is not None:
            n_converged += 1
            V = compute_V(t_result)
            solutions.append((V, t_result.copy(), tau_result.copy()))

            if verbose and attempt % 20 == 0:
                print(f"  Attempt {attempt}: V={V:.2f}")

    # Filter for V > 0
    positive_solutions = [(V, t, tau) for V, t, tau in solutions if V > 0]
    n_positive_V = len(positive_solutions)

    if verbose:
        print(f"  Converged: {n_converged}, V > 0: {n_positive_V}")

    if not positive_solutions:
        return {
            "success": False,
            "error": f"No V > 0 solutions found ({n_converged} converged)",
            "t": None,
            "tau": None,
            "V": None,
            "n_converged": n_converged,
            "n_positive_V": 0,
        }

    # Return solution with highest V (most robust)
    best = max(positive_solutions, key=lambda x: x[0])
    V_best, t_best, tau_best = best

    # Check convergence quality
    tau_error = np.sqrt(np.mean((tau_best - tau_target) ** 2))
    rel_error = tau_error / np.mean(tau_target)

    return {
        "success": rel_error < 0.01,
        "t": t_best,
        "tau": tau_best,
        "V": V_best,
        "tau_rel_error": rel_error,
        "n_converged": n_converged,
        "n_positive_V": n_positive_V,
    }


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data(example_name: str) -> dict:
    """Load all required data for an example."""
    data_dir = DATA_BASE / example_name

    points = np.array([[int(x) for x in line.split(',')]
                       for line in (data_dir / "points.dat").read_text().strip().split('\n')])
    heights = np.array([float(x) for x in (data_dir / "corrected_heights.dat").read_text().strip().split(',')])
    basis = [int(x) for x in (data_dir / "basis.dat").read_text().strip().split(',')]
    kklt_basis = [int(x) for x in (data_dir / "kklt_basis.dat").read_text().strip().split(',')]
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])

    # Load expected for validation (but NOT used in computation!)
    t_expected = np.array([float(x) for x in (data_dir / "kahler_param.dat").read_text().strip().split(',')])

    return {
        "points": points,
        "heights": heights,
        "basis": basis,
        "kklt_basis": kklt_basis,
        "c_i": c_i,
        "t_expected": t_expected,  # Only for validation comparison
    }


def get_cy(points: np.ndarray, heights: np.ndarray, basis: list, is_favorable: bool):
    """Build CY, handling favorable vs non-favorable."""
    if is_favorable:
        poly = Polytope(points)
        tri = poly.triangulate(heights=heights)
        cy = tri.get_cy()
        cy.set_divisor_basis(basis)
        return cy
    else:
        # Use latest CYTools for non-favorable
        CYTOOLS_LATEST = ROOT_DIR / "vendor/cytools_latest/src"
        mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
        for m in mods:
            del sys.modules[m]

        sys.path.insert(0, str(CYTOOLS_LATEST))
        from cytools import Polytope as PL
        from cytools import config
        config.enable_experimental_features()

        poly = PL(points)
        tri = poly.triangulate(heights=heights)
        cy = tri.get_cy()
        cy.set_divisor_basis(basis)

        # Restore 2021
        mods = [k for k in list(sys.modules.keys()) if 'cytools' in k]
        for m in mods:
            del sys.modules[m]
        sys.path.remove(str(CYTOOLS_LATEST))
        sys.path.insert(0, str(CYTOOLS_2021))

        return cy


# =============================================================================
# VALIDATION TEST
# =============================================================================


def test_example(example_name: str, h11_expected: int, h21_expected: int,
                 is_favorable: bool, verbose: bool = True) -> dict:
    """
    Test t_uncorrected computation FROM SCRATCH.

    Computes t where τ = c_i using multi-start, then compares to McAllister's
    kahler_param.dat to validate we found the same solution.
    """
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (h11={h11_expected})")
        print("=" * 70)

    # Load data
    data = load_data(example_name)
    h11 = len(data["basis"])

    # Build CY
    if verbose:
        print("  Building CY...")
    cy = get_cy(data["points"], data["heights"], data["basis"], is_favorable)

    kappa_basis = cy.intersection_numbers(in_basis=True)
    kappa_all = cy.intersection_numbers(in_basis=False)

    # Build sparse kappa
    kappa = SparseKappa(kappa_basis, kappa_all, data["basis"], data["kklt_basis"])

    if verbose:
        print(f"  c_i: {sum(data['c_i'] == 1)} D3s, {sum(data['c_i'] == 6)} O7s")
        print("  Solving τ = c_i from scratch...")

    # SOLVE FROM SCRATCH - no cheating!
    result = solve_t_uncorrected(kappa, data["c_i"], n_attempts=100, n_steps=150, verbose=verbose)

    if not result["success"]:
        if verbose:
            print(f"\nFAIL: {example_name} - {result.get('error', 'unknown')}")
        return {
            "example_name": example_name,
            "passed": False,
            "error": result.get("error"),
        }

    t_computed = result["t"]
    V_computed = result["V"]

    # Compare to expected (for validation only - we computed t from scratch!)
    t_expected = data["t_expected"]

    # Build tensor for expected V
    kappa_tensor = np.zeros((h11, h11, h11))
    if hasattr(kappa_basis, 'items'):
        for (i, j, k), val in kappa_basis.items():
            if val != 0:
                for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                    kappa_tensor[perm] = val
    else:
        for row in kappa_basis:
            i, j, k = int(row[0]), int(row[1]), int(row[2])
            val = row[3]
            for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
                kappa_tensor[perm] = val

    V_expected = np.einsum('ijk,i,j,k->', kappa_tensor, t_expected, t_expected, t_expected) / 6.0

    # Metrics
    t_corr = np.corrcoef(t_computed, t_expected)[0, 1]
    V_ratio = V_computed / V_expected if V_expected != 0 else 0

    if verbose:
        print(f"\n  Results (computed from scratch):")
        print(f"    t range: [{t_computed.min():.4f}, {t_computed.max():.4f}]")
        print(f"    t expected: [{t_expected.min():.4f}, {t_expected.max():.4f}]")
        print(f"    t correlation: {t_corr:.4f}")
        print(f"    V computed: {V_computed:.2f}")
        print(f"    V expected: {V_expected:.2f}")
        print(f"    V ratio: {V_ratio:.4f}")
        print(f"    τ rel_error: {result['tau_rel_error']:.6f}")

    # Pass criteria:
    # 1. V > 0 (correct branch)
    # 2. V ratio close to expected (found same solution)
    # 3. t correlation high (same branch)
    passed = V_computed > 0 and abs(V_ratio - 1.0) < 0.1 and t_corr > 0.9

    if verbose:
        print(f"\n{'PASS' if passed else 'FAIL'}: {example_name}")

    return {
        "example_name": example_name,
        "passed": passed,
        "t_correlation": t_corr,
        "V_computed": V_computed,
        "V_expected": V_expected,
        "V_ratio": V_ratio,
        "n_converged": result["n_converged"],
        "n_positive_V": result["n_positive_V"],
    }


def main():
    """Test all 5 McAllister examples."""
    print("=" * 70)
    print("COMPUTE T_UNCORRECTED - Solve τ = c_i FROM SCRATCH")
    print("=" * 70)
    print()
    print("Multi-start path-following with V > 0 filter.")
    print("NO CHEATING - t is computed, not loaded from .dat files.")
    print()

    results = []
    for name, h11, h21, is_favorable in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, h21, is_favorable, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for r in results:
        if r["passed"]:
            print(f"  PASS: {r['example_name']:30s} "
                  f"V_ratio={r['V_ratio']:.4f} t_corr={r['t_correlation']:.4f} "
                  f"({r['n_positive_V']}/{r['n_converged']} V>0)")
        else:
            print(f"  FAIL: {r['example_name']:30s} {r.get('error', '')}")
        all_passed = all_passed and r["passed"]

    print()
    if all_passed:
        print("All 5 examples PASSED")
        print("  - t_uncorrected computed from scratch for all examples")
        print("  - All match McAllister's solution (same branch)")
    else:
        print("Some examples FAILED")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
