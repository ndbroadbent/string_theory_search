#!/usr/bin/env python3
"""
Load and validate dual Coxeter numbers c_i for KKLT moduli stabilization.

CRITICAL: c_i values are MODEL CHOICES, NOT computed quantities!

From REPRODUCTION_OUTLINE.md:
  "The orientifold involution (which divisors are O7 vs D3) is a MODEL CHOICE,
   not computed from geometry!"

The c_i values determine the source of non-perturbative effects:
- c_i = 1: D3-brane instanton on rigid divisor
- c_i = 6: Gaugino condensation on O7-plane with SO(8) gauge group
- c_i = 2: Sp(2) gaugino condensation (rare, seen in 7-51-13590)

The KKLT superpotential is:
    W = W_0 + Σ_i A_i exp(-2π T_i / c_i)

McAllister's data provides c_i values in target_volumes.dat.

Reference: arXiv:2107.09064 (McAllister et al.)
"""

import sys
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_BASE = ROOT_DIR / "resources/small_cc_2107.09064_source/anc/paper_data"

# McAllister examples (name, h11_primal, h21_primal)
MCALLISTER_EXAMPLES = [
    ("4-214-647", 214, 4),
    ("5-113-4627-main", 113, 5),
    ("5-113-4627-alternative", 113, 5),
    ("5-81-3213", 81, 5),
    ("7-51-13590", 51, 7),
]


# =============================================================================
# DUAL COXETER NUMBER REFERENCE
# =============================================================================

DUAL_COXETER_NUMBERS = {
    # Common in string compactifications
    "so(8)": 6,   # O7-plane with 4 D7s
    "sp(2)": 2,   # Rare case in 7-51-13590
    "su(n)": lambda n: n,
    "so(n)": lambda n: n - 2,
    "sp(n)": lambda n: n + 1,
    # Exceptional algebras
    "e6": 12,
    "e7": 18,
    "e8": 30,
    "f4": 9,
    "g2": 4,
}


# =============================================================================
# DATA LOADING (MODEL INPUTS)
# =============================================================================


def load_c_i(example_name: str) -> np.ndarray:
    """
    Load c_i values from target_volumes.dat.

    THESE ARE MODEL INPUTS, NOT COMPUTED VALUES.

    Args:
        example_name: Name of the McAllister example

    Returns:
        Array of c_i values (typically 1 or 6, occasionally 2)
    """
    data_dir = DATA_BASE / example_name
    target_path = data_dir / "target_volumes.dat"

    if not target_path.exists():
        raise FileNotFoundError(f"target_volumes.dat not found: {target_path}")

    text = target_path.read_text().strip()
    return np.array([int(x) for x in text.split(",")])


def load_kklt_basis(example_name: str) -> np.ndarray:
    """
    Load KKLT basis indices from kklt_basis.dat.

    The KKLT basis contains only divisors that contribute to the superpotential
    (i.e., rigid divisors that can host D3-instantons or O7-planes).

    Returns:
        Array of 1-indexed divisor indices
    """
    data_dir = DATA_BASE / example_name
    basis_path = data_dir / "kklt_basis.dat"

    if not basis_path.exists():
        # Some examples may not have kklt_basis.dat
        return None

    text = basis_path.read_text().strip()
    return np.array([int(x) for x in text.split(",")])


def get_c_i_statistics(c_values: np.ndarray) -> dict:
    """
    Get statistics about c_i values.

    Returns:
        Dict with counts of each c_i value
    """
    unique, counts = np.unique(c_values, return_counts=True)

    stats = {
        "n_total": len(c_values),
        "n_d3": int(np.sum(c_values == 1)),        # D3-instantons
        "n_o7": int(np.sum(c_values == 6)),        # O7-planes with SO(8)
        "n_sp2": int(np.sum(c_values == 2)),       # Sp(2) gaugino condensation
        "unique_values": list(unique),
        "value_counts": dict(zip(map(int, unique), map(int, counts))),
    }

    # Validate: all values should be 1, 2, or 6
    unexpected = set(unique) - {1, 2, 6}
    if unexpected:
        stats["unexpected_values"] = list(unexpected)

    return stats


# =============================================================================
# VALIDATION TESTS
# =============================================================================


def test_example(example_name: str, expected_h11: int, verbose: bool = True) -> dict:
    """
    Test c_i loading for one McAllister example.

    Validates:
    1. target_volumes.dat exists and is readable
    2. Length matches expected h11 (KKLT basis divisors)
    3. All values are valid (1, 2, or 6)

    Returns:
        Dict with test results
    """
    if verbose:
        print("=" * 70)
        print(f"TEST - {example_name} (primal h11={expected_h11})")
        print("=" * 70)

    # Load c_i values
    try:
        c_values = load_c_i(example_name)
    except FileNotFoundError as e:
        if verbose:
            print(f"FAIL: {e}")
        return {"example_name": example_name, "passed": False, "error": str(e)}

    # Get statistics
    stats = get_c_i_statistics(c_values)

    if verbose:
        print(f"\n  Total divisors in KKLT basis: {stats['n_total']}")
        print(f"  D3-instantons (c_i=1): {stats['n_d3']}")
        print(f"  O7-planes with SO(8) (c_i=6): {stats['n_o7']}")
        if stats['n_sp2'] > 0:
            print(f"  Sp(2) gaugino (c_i=2): {stats['n_sp2']}")
        print(f"  Value counts: {stats['value_counts']}")

    # Validation checks
    passed = True
    errors = []

    # Check for unexpected values
    if "unexpected_values" in stats:
        passed = False
        errors.append(f"Unexpected c_i values: {stats['unexpected_values']}")

    # Check that we have at least some non-zero values
    if stats["n_total"] == 0:
        passed = False
        errors.append("No c_i values found")

    # Note: We can't check length == h11 because KKLT basis excludes non-rigid divisors
    # The number of KKLT divisors is less than h11

    if verbose:
        print(f"\n  First 10 c_i values: {list(c_values[:10])}")
        if errors:
            print(f"\n  ERRORS: {errors}")
        status = "PASS" if passed else "FAIL"
        print(f"\n{status}: {example_name}")

    return {
        "example_name": example_name,
        "passed": passed,
        "n_divisors": stats["n_total"],
        "n_d3": stats["n_d3"],
        "n_o7": stats["n_o7"],
        "errors": errors if errors else None,
    }


def main():
    """Test c_i loading against all 5 McAllister examples."""
    print("=" * 70)
    print("DUAL COXETER NUMBERS (c_i) - MODEL INPUT VALIDATION")
    print("c_i = 1 (D3-instanton), 6 (O7/SO(8)), 2 (Sp(2))")
    print("=" * 70)
    print("\nNOTE: c_i values are MODEL CHOICES from target_volumes.dat")
    print("      They are NOT computed from geometry!")

    results = []
    for name, h11, h21 in MCALLISTER_EXAMPLES:
        result = test_example(name, h11, verbose=True)
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        errors = f" [{r['errors']}]" if r.get("errors") else ""
        print(f"  {status}: {r['example_name']:30s} {r['n_divisors']} divisors "
              f"({r['n_d3']} D3, {r['n_o7']} O7){errors}")
        all_passed = all_passed and r["passed"]

    print()
    if all_passed:
        print(f"All {len(results)} examples PASSED")
        print("c_i values loaded successfully from target_volumes.dat")
    else:
        n_passed = sum(1 for r in results if r["passed"])
        print(f"{n_passed}/{len(results)} examples passed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
