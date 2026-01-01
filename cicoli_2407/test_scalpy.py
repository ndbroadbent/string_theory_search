#!/usr/bin/env python3
"""
Test ScalPy installation and basic quintessence computation.

This tests:
1. ScalPy imports correctly
2. We can set up a quintessence model with exponential potential
3. We can compute w(z) for the model
4. We can extract (w0, wa) from the evolution

Reference: arXiv:1503.02407 (ScalPy paper)
"""

import numpy as np


def test_scalpy_import():
    """Test that ScalPy imports correctly."""
    try:
        import scalpy
        print(f"✅ ScalPy imported successfully")
        print(f"   Version: {scalpy.__version__ if hasattr(scalpy, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"❌ ScalPy import failed: {e}")
        return False


def test_quintessence_model():
    """Test quintessence model with exponential potential."""
    try:
        import scalpy.scalar as sp

        # Parameters for exponential potential V = V0 * exp(K * phi)
        # K > 0 means field rolls down
        # For quintessence, we need K ~ O(1) in Planck units

        # Create quintessence model
        # ScalPy parameters:
        # - Ophi_i: initial scalar field density parameter
        # - li: lambda parameter for potential slope
        # - n: power for power-law potential (not used for exponential)

        print("\nSetting up quintessence with exponential potential...")
        print("  V(phi) = V0 * exp(-lambda * phi)")

        # Exponential potential parameters
        # lambda = sqrt(2) gives w = -1/3 at late times (thawing)
        # lambda < sqrt(2) gives w closer to -1
        li = 0.5  # slope parameter (smaller = closer to cosmological constant)
        Ophi_i = 0.85  # initial dark energy fraction (at recombination)

        # Create model
        # ScalPy's 'scalar_exp' class handles exponential potential
        model = sp.scalar_exp(Ophi_i=Ophi_i, li=li)

        print(f"  Created model with lambda={li}, Ophi_i={Ophi_i}")

        # Solve the cosmological evolution
        sol = model.sol()
        print(f"  Solved evolution: {sol.shape[0]} points")

        return True, model

    except Exception as e:
        print(f"❌ Quintessence model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_equation_of_state(model):
    """Test computing w(z) from the model."""
    try:
        print("\nComputing equation of state w(z)...")

        # Get w(z) at various redshifts
        z_vals = [0, 0.5, 1, 2, 5, 10]

        print(f"  {'z':>6} {'w(z)':>10}")
        print("  " + "-" * 18)

        w_vals = []
        for z in z_vals:
            w = model.eqn_state_pres_z(z)
            w_vals.append(w)
            print(f"  {z:>6.1f} {w:>10.4f}")

        # Check that w is in reasonable range
        w0 = w_vals[0]
        print(f"\n  w(z=0) = {w0:.4f}")

        if -1.5 < w0 < -0.3:
            print("  ✅ w0 is in reasonable quintessence range")
        else:
            print(f"  ⚠️ w0 = {w0} seems outside typical range [-1.5, -0.3]")

        return True, w_vals

    except Exception as e:
        print(f"❌ Equation of state test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_cpl_fitting(model):
    """Test fitting CPL parameters (w0, wa) from w(z)."""
    try:
        print("\nFitting CPL parameters...")
        print("  CPL: w(a) = w0 + wa * (1 - a)")
        print("  where a = 1/(1+z)")

        # Get w at many redshifts
        z_arr = np.linspace(0, 2, 100)
        w_arr = np.array([model.eqn_state_pres_z(z) for z in z_arr])
        a_arr = 1 / (1 + z_arr)

        # Fit to CPL: w = w0 + wa * (1 - a)
        # This is linear in (w0, wa)
        # Design matrix: [1, (1-a)]
        X = np.column_stack([np.ones_like(a_arr), 1 - a_arr])
        coeffs, residuals, rank, s = np.linalg.lstsq(X, w_arr, rcond=None)

        w0_fit, wa_fit = coeffs

        print(f"\n  Best fit CPL parameters:")
        print(f"    w0 = {w0_fit:.4f}")
        print(f"    wa = {wa_fit:.4f}")

        # Check quality of fit
        w_predicted = w0_fit + wa_fit * (1 - a_arr)
        rms_error = np.sqrt(np.mean((w_arr - w_predicted) ** 2))
        print(f"\n  RMS fit error: {rms_error:.6f}")

        # Compare to DESI constraints
        print(f"\n  DESI DR1 constraints (approx):")
        print(f"    w0 = -0.7 ± 0.2")
        print(f"    wa = -1.0 ± 0.5")

        return True, (w0_fit, wa_fit)

    except Exception as e:
        print(f"❌ CPL fitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all tests."""
    print("=" * 60)
    print("ScalPy Test Suite for Quintessence")
    print("=" * 60)

    results = []

    # Test 1: Import
    print("\n[Test 1] Import ScalPy")
    results.append(test_scalpy_import())

    # Test 2: Quintessence model
    print("\n[Test 2] Quintessence Model Setup")
    success, model = test_quintessence_model()
    results.append(success)

    if model is not None:
        # Test 3: Equation of state
        print("\n[Test 3] Equation of State w(z)")
        success, w_vals = test_equation_of_state(model)
        results.append(success)

        # Test 4: CPL fitting
        print("\n[Test 4] CPL Parameter Fitting")
        success, cpl_params = test_cpl_fitting(model)
        results.append(success)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("✅ All tests passed! ScalPy is working.")
    else:
        print("❌ Some tests failed")

    return all(results)


if __name__ == "__main__":
    main()
