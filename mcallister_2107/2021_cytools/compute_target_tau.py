#!/usr/bin/env python3
"""
Step 15: Compute target divisor volumes for KKLT stabilization.

From McAllister eq 2.29 and 5.12:
    c_τ = 2π / (g_s × ln(1/W₀))
    τ_target = c_i / c_τ

Inputs (all from upstream steps):
    - c_i: Dual Coxeter numbers from orientifold (1 for D3, 6 for O7)
    - g_s: String coupling from racetrack (Step 13)
    - W₀: Flux superpotential from racetrack (Step 14)

Output:
    - c_τ: Characteristic scale
    - τ_target: Zeroth-order target divisor volumes for KKLT solver

Reference: arXiv:2107.09064 eq 2.29, 5.12
"""

import sys
from pathlib import Path
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

MCALLISTER_EXAMPLES = [
    "4-214-647",
    "5-113-4627-main",
    "5-113-4627-alternative",
    "5-81-3213",
    "7-51-13590",
]


# =============================================================================
# GENERAL PURPOSE FUNCTIONS
# =============================================================================

def compute_c_tau(g_s: float, W0: float) -> float:
    """
    Compute c_τ from g_s and W₀.

    Formula (eq 2.29):
        c_τ = 2π / (g_s × ln(1/W₀))

    Args:
        g_s: String coupling (from racetrack stabilization)
        W0: Flux superpotential magnitude (from racetrack)

    Returns:
        c_τ value
    """
    ln_W0_inv = np.log(1.0 / abs(W0))
    c_tau = 2 * np.pi / (g_s * ln_W0_inv)
    return c_tau


def compute_target_tau(c_i: np.ndarray, c_tau: float) -> np.ndarray:
    """
    Compute zeroth-order target divisor volumes for KKLT.

    Formula (eq 5.12):
        τ_target = c_i / c_τ

    Args:
        c_i: Dual Coxeter numbers (1 for D3, 6 for O7)
        c_tau: Characteristic scale from compute_c_tau()

    Returns:
        τ_target array (same shape as c_i)
    """
    return c_i / c_tau


# =============================================================================
# DATA LOADING
# =============================================================================

def load_example_data(example_name: str) -> dict:
    """Load all relevant data for a McAllister example."""
    data_dir = DATA_BASE / example_name

    g_s = float((data_dir / "g_s.dat").read_text().strip())
    W0 = float((data_dir / "W_0.dat").read_text().strip())
    c_tau_expected = float((data_dir / "c_tau.dat").read_text().strip())
    c_i = np.array([int(x) for x in (data_dir / "target_volumes.dat").read_text().strip().split(',')])

    return {
        "g_s": g_s,
        "W0": W0,
        "c_tau_expected": c_tau_expected,
        "c_i": c_i,
    }


# =============================================================================
# VALIDATION
# =============================================================================

def test_example(example_name: str, verbose: bool = True) -> dict:
    """Test c_τ computation against McAllister data for one example."""
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name}")
        print("=" * 70)

    data = load_example_data(example_name)

    if verbose:
        print(f"\nInputs:")
        print(f"  g_s = {data['g_s']:.6f}")
        print(f"  W₀ = {data['W0']:.2e}")
        print(f"  ln(1/W₀) = {np.log(1/data['W0']):.2f}")
        print(f"  c_i has {len(data['c_i'])} divisors ({sum(data['c_i'] == 6)} O7, {sum(data['c_i'] == 1)} D3)")

    # Compute c_τ
    c_tau_computed = compute_c_tau(data['g_s'], data['W0'])
    c_tau_expected = data['c_tau_expected']
    c_tau_error = abs(c_tau_computed - c_tau_expected)
    c_tau_ratio = c_tau_computed / c_tau_expected

    if verbose:
        print(f"\nc_τ:")
        print(f"  Computed: {c_tau_computed:.6f}")
        print(f"  Expected: {c_tau_expected:.6f}")
        print(f"  Ratio: {c_tau_ratio:.6f}")
        print(f"  Error: {c_tau_error:.2e}")

    # Compute target τ values
    tau_target = compute_target_tau(data['c_i'], c_tau_computed)

    if verbose:
        # Show a few sample target values
        print(f"\nSample τ_target values:")
        for i in range(min(5, len(tau_target))):
            print(f"  τ[{i}] = {tau_target[i]:.6f} (c_i = {data['c_i'][i]})")

    # Check pass/fail (tolerance for floating point storage precision)
    passed = c_tau_error < 1e-4

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"\n{status}: c_τ error = {c_tau_error:.2e}")

    return {
        "example_name": example_name,
        "g_s": data['g_s'],
        "W0": data['W0'],
        "c_tau_computed": c_tau_computed,
        "c_tau_expected": c_tau_expected,
        "c_tau_ratio": c_tau_ratio,
        "tau_target": tau_target,
        "passed": passed,
    }


def main():
    """Test all 5 McAllister examples."""
    print("=" * 70)
    print("STEP 15: COMPUTE c_τ AND TARGET τ - ALL 5 EXAMPLES")
    print("=" * 70)

    results = []
    for example in MCALLISTER_EXAMPLES:
        result = test_example(example, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}: {r['example_name']} c_τ ratio = {r['c_tau_ratio']:.6f}")
        all_passed = all_passed and r["passed"]

    print()
    if all_passed:
        print("All 5 examples PASSED")
    else:
        print("Some examples FAILED")

    return results


if __name__ == "__main__":
    main()
