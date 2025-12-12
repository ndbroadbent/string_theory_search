#!/usr/bin/env python3
"""
Step 15: Compute target divisor volumes for KKLT stabilization.

From McAllister eq 5.12, the zeroth-order target is:
    τ_target = c_i / c_τ

Where:
    c_τ = 2π / (g_s × ln(1/W₀))

Inputs (all from upstream steps):
    - c_i: Dual Coxeter numbers from orientifold (Step 3)
    - g_s: String coupling from racetrack (Step 13)
    - W₀: Flux superpotential from racetrack (Step 14)

Output:
    - τ_target: Zeroth-order target for KKLT solver (Step 16)

The χ(D_i)/24 and GV corrections are applied iteratively inside the KKLT solver.

Reference: arXiv:2107.09064 eq 2.29, 5.12
"""

import numpy as np
from pathlib import Path


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


def compute_target_tau(c_i: np.ndarray, g_s: float, W0: float) -> np.ndarray:
    """
    Compute zeroth-order target divisor volumes for KKLT.

    Formula (eq 5.12):
        τ_target = c_i / c_τ

    Args:
        c_i: Dual Coxeter numbers (1 for D3, 6 for O7)
        g_s: String coupling
        W0: Flux superpotential

    Returns:
        τ_target array (same shape as c_i)
    """
    c_tau = compute_c_tau(g_s, W0)
    tau_target = c_i / c_tau
    return tau_target


# =============================================================================
# VALIDATION
# =============================================================================

def main():
    """Validate c_τ against McAllister 4-214-647 data."""
    DATA_DIR = Path(__file__).parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647"

    print("=" * 70)
    print("STEP 15: Compute c_τ and zeroth-order target")
    print("=" * 70)

    # Load inputs
    g_s = float((DATA_DIR / "g_s.dat").read_text().strip())
    W0 = float((DATA_DIR / "W_0.dat").read_text().strip())
    c_tau_expected = float((DATA_DIR / "c_tau.dat").read_text().strip())

    print(f"\nInputs:")
    print(f"  g_s = {g_s}")
    print(f"  W₀ = {W0:.2e}")
    print(f"  ln(1/W₀) = {np.log(1/W0):.2f}")

    # Compute and verify c_τ
    # NOTE: McAllister's data files store values at 6 significant figures.
    # In the actual pipeline, g_s and W0 come from derive_racetrack.py at
    # full float64 precision, so c_tau will be computed more precisely.
    # This validation confirms our formula matches McAllister's to their
    # stored precision.
    c_tau = compute_c_tau(g_s, W0)
    error = abs(c_tau - c_tau_expected)

    print(f"\nc_τ:")
    print(f"  Computed: {c_tau:.6f}")
    print(f"  Expected: {c_tau_expected:.6f}")
    print(f"  Error: {error:.2e} (limited by 6 sig fig storage)")

    assert error < 1e-5, f"c_τ mismatch: {c_tau} vs {c_tau_expected}"
    print(f"\n✓ c_τ VALIDATED")


if __name__ == "__main__":
    main()
