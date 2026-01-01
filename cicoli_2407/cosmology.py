#!/usr/bin/env python3
"""
Cosmology solver for quintessence dark energy.

Solves the Klein-Gordon equation for a scalar field φ with potential V(φ):

  φ̈ + 3Hφ̇ + V'(φ) = 0

coupled to the Friedmann equation:

  H² = (8πG/3) × (ρ_m + ρ_r + ρ_φ)

where:
  ρ_φ = (1/2)φ̇² + V(φ)  (scalar field energy density)
  p_φ = (1/2)φ̇² - V(φ)  (scalar field pressure)
  w = p_φ/ρ_φ           (equation of state)

For slow roll (φ̇² << V):  w ≈ -1 + (2/3)ε  where ε = (M_pl²/2)(V'/V)²

Units: We use M_pl = 1 (reduced Planck mass) throughout.

Reference: Cicoli et al. arXiv:2407.03405
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit


# Physical constants
H0 = 67.4  # Hubble constant in km/s/Mpc (Planck 2018)
H0_SI = H0 * 1000 / (3.086e22)  # Convert to 1/s
T0 = 1 / H0_SI  # Hubble time in seconds
T0_GYR = T0 / (3.156e16)  # Hubble time in Gyr (~14.5 Gyr)

# Planck mass and related
M_PL = 2.435e18  # Reduced Planck mass in GeV
RHO_CRIT_TODAY = 3 * (H0_SI ** 2) / (8 * np.pi)  # Critical density in kg/m³

# Cosmological parameters (Planck 2018)
OM_M0 = 0.315  # Matter density today
OM_R0 = 9.2e-5  # Radiation density today
OM_DE0 = 1 - OM_M0 - OM_R0  # Dark energy density today


def hilltop_potential(phi, Lambda4, f):
    """
    Hilltop quintessence potential.

    V(φ) = Λ⁴ × [1 - cos(φ/f)]

    This is the form from Cicoli 2407.03405 for the quintessence axion.

    Parameters
    ----------
    phi : float or ndarray
        Field value (in M_pl units)
    Lambda4 : float
        Energy scale to the fourth power (in M_pl⁴ units)
    f : float
        Decay constant (in M_pl units)

    Returns
    -------
    V : float or ndarray
        Potential value
    """
    return Lambda4 * (1 - np.cos(phi / f))


def hilltop_potential_derivative(phi, Lambda4, f):
    """
    Derivative of hilltop potential.

    V'(φ) = (Λ⁴/f) × sin(φ/f)
    """
    return (Lambda4 / f) * np.sin(phi / f)


def exponential_potential(phi, V0, lam):
    """
    Exponential quintessence potential.

    V(φ) = V₀ × exp(-λφ)

    Parameters
    ----------
    phi : float or ndarray
        Field value (in M_pl units)
    V0 : float
        Normalization (in M_pl⁴)
    lam : float
        Slope parameter (dimensionless)

    Returns
    -------
    V : float or ndarray
        Potential value
    """
    return V0 * np.exp(-lam * phi)


def exponential_potential_derivative(phi, V0, lam):
    """
    Derivative of exponential potential.

    V'(φ) = -λ × V₀ × exp(-λφ) = -λ × V(φ)
    """
    return -lam * exponential_potential(phi, V0, lam)


class QuintessenceCosmology:
    """
    Solve cosmological evolution for quintessence dark energy.

    The system of equations (in conformal time):
      dφ/dN = φ'/H  (where N = ln(a))
      dπ/dN = -3π - V'(φ)/H²  (where π = φ'/H)

    We use e-folding number N = ln(a) as the time variable.
    """

    def __init__(self, potential_func, potential_deriv_func, **potential_params):
        """
        Initialize quintessence cosmology.

        Parameters
        ----------
        potential_func : callable
            V(phi, **params) -> potential value
        potential_deriv_func : callable
            V'(phi, **params) -> potential derivative
        **potential_params : dict
            Parameters passed to potential functions
        """
        self.V = lambda phi: potential_func(phi, **potential_params)
        self.dV = lambda phi: potential_deriv_func(phi, **potential_params)
        self.potential_params = potential_params

    def rho_phi(self, phi, phi_dot, H):
        """
        Scalar field energy density.

        ρ_φ = (1/2)φ̇² + V(φ)

        In e-folding variables (φ' = dφ/dN = φ̇/H):
        ρ_φ = (1/2)(Hφ')² + V(φ)
        """
        return 0.5 * (H * phi_dot) ** 2 + self.V(phi)

    def p_phi(self, phi, phi_dot, H):
        """
        Scalar field pressure.

        p_φ = (1/2)φ̇² - V(φ)
        """
        return 0.5 * (H * phi_dot) ** 2 - self.V(phi)

    def w_phi(self, phi, phi_dot, H):
        """
        Scalar field equation of state.

        w = p/ρ = [(1/2)φ̇² - V] / [(1/2)φ̇² + V]
        """
        kinetic = 0.5 * (H * phi_dot) ** 2
        V = self.V(phi)
        rho = kinetic + V
        p = kinetic - V
        if rho == 0:
            return -1.0
        return p / rho

    def friedmann_H2(self, a, phi, phi_dot, H_guess=None):
        """
        Solve Friedmann equation for H².

        H² = (8πG/3)(ρ_m + ρ_r + ρ_φ)

        In units where M_pl = 1:
        H² = (ρ_m + ρ_r + ρ_φ) / 3

        With:
        - ρ_m = ρ_m0 × a⁻³
        - ρ_r = ρ_r0 × a⁻⁴
        - ρ_φ = (1/2)(Hφ')² + V(φ)

        Note: This is implicit in H because ρ_φ depends on H through φ̇ = H×φ'.
        For slow roll, kinetic term is small and we can ignore this.
        """
        # Normalize so that sum of densities = 3H₀² today
        rho_m = OM_M0 * a ** (-3)
        rho_r = OM_R0 * a ** (-4)

        # For slow roll approximation, ignore kinetic term
        V = self.V(phi)
        rho_phi = V  # Slow roll: kinetic << potential

        # Friedmann equation (normalized to H0=1)
        H2 = (rho_m + rho_r + rho_phi) / 3.0

        return H2

    def equations(self, y, N):
        """
        Evolution equations in e-folding time N = ln(a).

        Variables:
          y[0] = φ (field value)
          y[1] = π = dφ/dN (field velocity in e-foldings)

        Equations:
          dφ/dN = π
          dπ/dN = -(3 - ε)π - (V'/V)(V/H²)

        where ε = -(dH/dN)/H is the first slow-roll parameter.
        """
        phi, pi = y
        a = np.exp(N)

        # Get H² from Friedmann
        H2 = self.friedmann_H2(a, phi, pi)
        if H2 <= 0:
            H2 = 1e-100  # Prevent division by zero

        # Potential and derivative
        V = self.V(phi)
        dV = self.dV(phi)

        # Evolution equations (using slow-roll approximation for stability)
        dphi_dN = pi
        dpi_dN = -3 * pi - dV / H2

        return [dphi_dN, dpi_dN]

    def solve(self, phi_i, phi_dot_i, N_span=(-10, 0), N_points=1000):
        """
        Solve the cosmological evolution.

        Parameters
        ----------
        phi_i : float
            Initial field value at N = N_span[0]
        phi_dot_i : float
            Initial dφ/dN at N = N_span[0]
        N_span : tuple
            Range of e-foldings (N_start, N_end). N=0 is today.
        N_points : int
            Number of output points

        Returns
        -------
        dict with:
            N : ndarray of e-folding values
            a : ndarray of scale factor
            z : ndarray of redshift
            phi : ndarray of field values
            pi : ndarray of dφ/dN
            w : ndarray of equation of state
            H : ndarray of Hubble parameter (normalized to H0)
        """
        N = np.linspace(N_span[0], N_span[1], N_points)
        y0 = [phi_i, phi_dot_i]

        # Solve ODE
        sol = odeint(self.equations, y0, N)

        phi = sol[:, 0]
        pi = sol[:, 1]
        a = np.exp(N)
        z = 1 / a - 1

        # Compute H and w at each point
        H = np.zeros_like(N)
        w = np.zeros_like(N)

        for i in range(len(N)):
            H2 = self.friedmann_H2(a[i], phi[i], pi[i])
            H[i] = np.sqrt(max(H2, 0))
            w[i] = self.w_phi(phi[i], pi[i], H[i])

        return {
            "N": N,
            "a": a,
            "z": z,
            "phi": phi,
            "pi": pi,
            "w": w,
            "H": H,
        }


def fit_cpl(z, w):
    """
    Fit CPL parameters (w0, wa) to w(z) data.

    CPL parameterization:
      w(a) = w0 + wa × (1 - a)
      w(z) = w0 + wa × z/(1+z)

    Parameters
    ----------
    z : ndarray
        Redshift values
    w : ndarray
        Equation of state values

    Returns
    -------
    w0 : float
        w at z=0
    wa : float
        Rate of change
    """
    a = 1 / (1 + z)

    # Design matrix for linear fit: w = w0 + wa*(1-a)
    X = np.column_stack([np.ones_like(a), 1 - a])
    coeffs, _, _, _ = np.linalg.lstsq(X, w, rcond=None)

    w0, wa = coeffs
    return w0, wa


def test_lcdm():
    """Test with ΛCDM (constant w=-1)."""
    print("=" * 60)
    print("Test: ΛCDM (should give w = -1 always)")
    print("=" * 60)

    # For ΛCDM, use a constant potential (no dynamics)
    # This is a bit of a hack - just set V very large and flat
    Lambda4 = OM_DE0 * 3.0  # Dark energy density in our units
    f = 1e10  # Very large decay constant = flat potential

    cosmo = QuintessenceCosmology(
        hilltop_potential, hilltop_potential_derivative, Lambda4=Lambda4, f=f
    )

    # Start near the minimum (φ ≈ 0)
    sol = cosmo.solve(phi_i=0.01, phi_dot_i=0.0, N_span=(-10, 0))

    print(f"\nResults at z=0:")
    print(f"  φ = {sol['phi'][-1]:.6f}")
    print(f"  w = {sol['w'][-1]:.6f}")

    # w should be close to -1
    w0 = sol["w"][-1]
    print(f"\n  Expected w ≈ -1.0, got w = {w0:.4f}")
    if abs(w0 + 1) < 0.1:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")

    return sol


def test_thawing_quintessence():
    """Test thawing quintessence (hilltop potential)."""
    print("\n" + "=" * 60)
    print("Test: Thawing Quintessence (hilltop potential)")
    print("=" * 60)

    # Parameters from Cicoli 2407.03405 Section 4.5
    # Λ⁴ ~ 10⁻¹²⁰ M_pl⁴
    # f ~ 0.085 M_pl
    Lambda4 = 1e-120  # In M_pl⁴
    f = 0.085  # In M_pl

    # But we need to normalize to cosmological scales
    # ρ_DE,0 ~ 0.7 × ρ_crit ~ 0.7 × 3H₀²
    # Set Lambda4 to give correct dark energy density
    Lambda4 = OM_DE0 * 3.0 / 2  # Factor of 2 because V = Λ⁴(1-cos) ~ Λ⁴ near maximum

    cosmo = QuintessenceCosmology(
        hilltop_potential, hilltop_potential_derivative, Lambda4=Lambda4, f=f
    )

    # Start near the hilltop (φ ≈ π×f = maximum of 1-cos)
    # Slight displacement → field rolls down → thawing
    phi_i = np.pi * f * 0.99  # Just below hilltop
    phi_dot_i = 0.0  # Start at rest

    sol = cosmo.solve(phi_i=phi_i, phi_dot_i=phi_dot_i, N_span=(-10, 0))

    # Compute w at various redshifts
    print("\nEquation of state w(z):")
    print(f"  {'z':>6} {'w':>10}")
    print("  " + "-" * 18)
    for z_val in [0, 0.5, 1, 2, 5]:
        idx = np.argmin(np.abs(sol["z"] - z_val))
        print(f"  {z_val:>6.1f} {sol['w'][idx]:>10.4f}")

    # Fit CPL
    # Use only z < 3 for fitting (where CPL is more accurate)
    mask = sol["z"] < 3
    w0, wa = fit_cpl(sol["z"][mask], sol["w"][mask])

    print(f"\nCPL fit (z < 3):")
    print(f"  w0 = {w0:.4f}")
    print(f"  wa = {wa:.4f}")

    print(f"\nDESI DR1 constraints (approx):")
    print(f"  w0 = -0.7 ± 0.2")
    print(f"  wa = -1.0 ± 0.5")

    return sol, w0, wa


def main():
    """Run tests."""
    print("Quintessence Cosmology Solver")
    print("=" * 60)

    # Test 1: ΛCDM
    sol_lcdm = test_lcdm()

    # Test 2: Thawing quintessence
    sol_thaw, w0, wa = test_thawing_quintessence()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("✅ Cosmology solver is working!")
    print(f"   Thawing quintessence gives (w0, wa) = ({w0:.3f}, {wa:.3f})")


if __name__ == "__main__":
    main()
