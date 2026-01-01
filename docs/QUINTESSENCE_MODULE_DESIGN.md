# Quintessence Module Design

## Overview

This module extends the existing KKLT pipeline to support dynamical dark energy (quintessence)
as suggested by DESI 2024/2025 results. Instead of finding a static vacuum energy V₀, we
compute the time-evolution of a rolling modulus and extract (ρ_DE, w₀, wₐ).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Quintessence Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   CYTools    │───▶│   Partial    │───▶│  Scalar Potential    │  │
│  │  (geometry)  │    │    KKLT      │    │      V(T)            │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         │                   │                       ▼               │
│         │                   │            ┌──────────────────────┐  │
│         │                   │            │  Cosmology Solver    │  │
│         │                   │            │  (ODE integration)   │  │
│         │                   │            └──────────────────────┘  │
│         │                   │                       │               │
│         │                   │                       ▼               │
│         │                   │            ┌──────────────────────┐  │
│         │                   │            │   CPL Extraction     │  │
│         │                   │            │   (w₀, wₐ fitting)   │  │
│         │                   │            └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Fitness Evaluation                        │   │
│  │         (ρ_DE, w₀, wₐ, α_em, α_s, sin²θ_W, N_gen)           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
string_theory/
├── quintessence/
│   ├── __init__.py
│   ├── partial_kklt.py      # Partial moduli stabilization
│   ├── potential.py         # V(T) computation
│   ├── cosmology.py         # ODE solver for field evolution
│   ├── cpl_fitting.py       # Extract (w₀, wₐ) from w(t)
│   └── constants.py         # Physical constants, DESI targets
├── physics_bridge.py        # Updated to support quintessence mode
└── tests/
    └── test_quintessence.py
```

## Module Specifications

### 1. partial_kklt.py

```python
"""
Partial KKLT stabilization - stabilize all moduli EXCEPT the rolling one.
"""

def solve_partial_kklt(
    kappa: np.ndarray,           # Intersection numbers (h11, h11, h11)
    c_i: np.ndarray,             # Dual Coxeter numbers (h11,)
    chi_D: np.ndarray,           # Divisor Euler characteristics (h11,)
    rolling_idx: int,            # Index of rolling modulus (0 to h11-1)
    T_rolling: float,            # Fixed value for rolling modulus
    t_init: Optional[np.ndarray] = None,
    n_steps: int = 200,
) -> Tuple[np.ndarray, bool]:
    """
    Solve KKLT equations with one modulus held fixed.

    The constraint is:
        τᵢ = (cᵢ/2π) ln(W₀⁻¹) + χ(Dᵢ)/24   for i ≠ rolling_idx
        T[rolling_idx] = T_rolling           (fixed)

    Returns:
        t: Kähler moduli array (h11,)
        converged: bool
    """
    pass
```

### 2. potential.py

```python
"""
Compute scalar potential V(T) as function of rolling modulus.
"""

from dataclasses import dataclass
from typing import Callable

@dataclass
class ScalarPotential:
    """Encapsulates V(T) and its derivatives."""
    V: Callable[[float], float]      # V(T)
    dV: Callable[[float], float]     # dV/dT
    d2V: Callable[[float], float]    # d²V/dT²


def compute_scalar_potential(
    kappa: np.ndarray,           # Intersection numbers
    t_stabilized: np.ndarray,    # Stabilized moduli values
    rolling_idx: int,            # Which modulus rolls
    W0: float,                   # Flux superpotential
    g_s: float,                  # String coupling
    A_i: np.ndarray,             # Pfaffian prefactors
    c_i: np.ndarray,             # Dual Coxeter numbers
    exp_K0: float,               # Complex structure contribution
) -> ScalarPotential:
    """
    Construct V(T) for the rolling modulus.

    The full scalar potential is (eq. 6.24 of McAllister):
        V = eᴷ (Kⁱʲ̄ Dᵢ W D̄ⱼ̄ W̄ - 3|W|²)

    For KKLT at the minimum: V₀ = -3 eᴷ |W|²

    For quintessence, we don't minimize - we evaluate V(T) where T = T_rolling.

    Returns:
        ScalarPotential with V(T), dV/dT, d²V/dT²
    """
    pass


def compute_volume_as_function(
    kappa: np.ndarray,
    t_stabilized: np.ndarray,
    rolling_idx: int,
    chi: int,  # Euler characteristic for BBHL
) -> Callable[[float], float]:
    """
    Return V_string(T) where T is the rolling modulus.

    V_string = (1/6) κᵢⱼₖ tⁱ tʲ tᵏ - ζ(3)χ/(4(2π)³)

    With t[rolling_idx] = T (variable), others fixed.
    """
    pass
```

### 3. cosmology.py

```python
"""
Solve cosmological evolution of quintessence field.
"""

from dataclasses import dataclass
from scipy.integrate import solve_ivp

# Physical constants
M_pl = 1.0  # Planck mass (we work in Planck units)
H0 = 2.195e-18  # Hubble constant in s⁻¹ (67.4 km/s/Mpc)
Omega_m0 = 0.315  # Matter density today
Omega_r0 = 9.0e-5  # Radiation density today

@dataclass
class CosmologyResult:
    """Result of cosmological evolution."""
    a: np.ndarray          # Scale factor history
    t: np.ndarray          # Time in seconds
    T: np.ndarray          # Field value history
    T_dot: np.ndarray      # Field velocity history
    H: np.ndarray          # Hubble parameter history
    w: np.ndarray          # Equation of state history
    rho_DE: np.ndarray     # Dark energy density history


def evolve_quintessence(
    potential: ScalarPotential,
    T_initial: float,
    T_dot_initial: float = 0.0,
    a_initial: float = 1e-10,  # Start deep in radiation era
    a_final: float = 1.0,      # End today
    n_points: int = 1000,
) -> CosmologyResult:
    """
    Solve the coupled quintessence + Friedmann equations.

    Field equation:
        T̈ + 3H Ṫ + dV/dT = 0

    Friedmann equation:
        H² = (ρ_m + ρ_r + ρ_φ) / (3 M_pl²)

    where:
        ρ_φ = ½ Ṫ² + V(T)  (quintessence energy density)
        ρ_m = ρ_m0 / a³     (matter)
        ρ_r = ρ_r0 / a⁴     (radiation)

    Equation of state:
        w = (½ Ṫ² - V) / (½ Ṫ² + V)

    Returns:
        CosmologyResult with full evolution history
    """

    def equations(ln_a, y):
        """
        y = [T, T_dot_over_H]
        We use ln(a) as time variable for numerical stability.
        """
        T, T_dot_over_H = y
        a = np.exp(ln_a)

        # Energy densities (in Planck units)
        rho_m = Omega_m0 * 3 * H0**2 / a**3
        rho_r = Omega_r0 * 3 * H0**2 / a**4
        V = potential.V(T)

        # Hubble parameter: H² = (ρ_m + ρ_r + ½Ṫ² + V) / 3
        # This is implicit in T_dot, so we iterate or use approximation
        # For slow-roll: Ṫ² << V, so H² ≈ (ρ_m + ρ_r + V) / 3
        H2 = (rho_m + rho_r + V) / 3  # Slow-roll approximation
        H = np.sqrt(max(H2, 1e-100))

        T_dot = T_dot_over_H * H

        # Field equation: T̈ + 3H Ṫ + V' = 0
        # => d(T_dot)/d(ln a) = -3 T_dot - V'/H²
        dV = potential.dV(T)
        dT_dot_over_H_dlna = -3 * T_dot_over_H - dV / H**2

        # dT/d(ln a) = T_dot / H
        dT_dlna = T_dot / H if H > 0 else 0

        return [dT_dlna, dT_dot_over_H_dlna]

    # Initial conditions
    y0 = [T_initial, T_dot_initial / H0]  # Normalize by H0
    ln_a_span = [np.log(a_initial), np.log(a_final)]
    ln_a_eval = np.linspace(ln_a_span[0], ln_a_span[1], n_points)

    # Solve
    sol = solve_ivp(
        equations,
        ln_a_span,
        y0,
        t_eval=ln_a_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )

    # Extract results
    a = np.exp(sol.t)
    T = sol.y[0]
    T_dot_over_H = sol.y[1]

    # Compute derived quantities
    V_arr = np.array([potential.V(Ti) for Ti in T])
    rho_m = Omega_m0 * 3 * H0**2 / a**3
    rho_r = Omega_r0 * 3 * H0**2 / a**4
    H = np.sqrt((rho_m + rho_r + V_arr) / 3)
    T_dot = T_dot_over_H * H

    rho_DE = 0.5 * T_dot**2 + V_arr
    w = (0.5 * T_dot**2 - V_arr) / (0.5 * T_dot**2 + V_arr)

    # Convert ln(a) to cosmic time (approximate)
    # dt = da / (a H)
    t = np.cumsum(np.diff(a, prepend=a[0]) / (a * H))

    return CosmologyResult(
        a=a, t=t, T=T, T_dot=T_dot, H=H, w=w, rho_DE=rho_DE
    )
```

### 4. cpl_fitting.py

```python
"""
Fit w(a) to CPL parameterization and extract (w₀, wₐ).
"""

from scipy.optimize import curve_fit

def cpl_model(a: np.ndarray, w0: float, wa: float) -> np.ndarray:
    """
    CPL parameterization: w(a) = w0 + wa * (1 - a)
    """
    return w0 + wa * (1 - a)


def fit_cpl(
    a: np.ndarray,
    w: np.ndarray,
    a_min: float = 0.3,  # Only fit recent universe (z < 2.3)
) -> Tuple[float, float, float, float]:
    """
    Fit w(a) data to CPL model.

    Args:
        a: Scale factor array
        w: Equation of state array
        a_min: Minimum scale factor to include in fit

    Returns:
        w0: Equation of state today
        wa: Rate of change
        w0_err: Uncertainty on w0
        wa_err: Uncertainty on wa
    """
    # Filter to recent universe
    mask = a >= a_min
    a_fit = a[mask]
    w_fit = w[mask]

    # Fit
    try:
        popt, pcov = curve_fit(
            cpl_model,
            a_fit,
            w_fit,
            p0=[-1.0, 0.0],  # Start from ΛCDM
            bounds=([-2.0, -5.0], [0.0, 5.0]),
        )
        w0, wa = popt
        w0_err, wa_err = np.sqrt(np.diag(pcov))
    except RuntimeError:
        # Fit failed, return NaN
        w0, wa = np.nan, np.nan
        w0_err, wa_err = np.nan, np.nan

    return w0, wa, w0_err, wa_err


def compute_w0_wa_direct(
    potential: ScalarPotential,
    T_today: float,
    T_dot_today: float,
) -> Tuple[float, float]:
    """
    Compute (w₀, wₐ) directly from potential at today's field value.

    This is a faster approximation that doesn't require full evolution.

    w₀ = (½Ṫ² - V) / (½Ṫ² + V)

    wₐ requires the second derivative and is more complex.
    For slow-roll: wₐ ≈ -2 * (1 + w₀) * (ε - η/2)
    where ε = ½(V'/V)² and η = V''/V
    """
    V = potential.V(T_today)
    dV = potential.dV(T_today)
    d2V = potential.d2V(T_today)

    kinetic = 0.5 * T_dot_today**2
    w0 = (kinetic - V) / (kinetic + V)

    # Slow-roll parameters
    epsilon = 0.5 * (dV / V)**2 if V != 0 else 0
    eta = d2V / V if V != 0 else 0

    # Approximate wa (valid for slow-roll)
    wa = -2 * (1 + w0) * (epsilon - 0.5 * eta)

    return w0, wa
```

### 5. constants.py

```python
"""
Physical constants and observational targets.
"""

import numpy as np

# =============================================================================
# Fundamental Constants (Planck units: ℏ = c = G = 1)
# =============================================================================

M_PLANCK = 1.0  # Planck mass
L_PLANCK = 1.0  # Planck length
T_PLANCK = 1.0  # Planck time

# In SI units for reference
M_PLANCK_KG = 2.176e-8      # kg
L_PLANCK_M = 1.616e-35      # m
T_PLANCK_S = 5.391e-44      # s

# =============================================================================
# Cosmological Parameters (Planck 2018 + DESI)
# =============================================================================

# Hubble constant
H0_SI = 67.4  # km/s/Mpc
H0_PLANCK = 2.195e-18 / T_PLANCK_S  # In Planck units (1/t_Pl)

# Matter densities (as fractions of critical density)
OMEGA_M0 = 0.315      # Total matter
OMEGA_B0 = 0.0493     # Baryons
OMEGA_CDM0 = 0.265    # Cold dark matter
OMEGA_R0 = 9.0e-5     # Radiation (photons + neutrinos)
OMEGA_DE0 = 0.685     # Dark energy

# Critical density today
RHO_CRIT_0 = 3 * H0_PLANCK**2  # In Planck units (M_Pl^4)

# Dark energy density today
RHO_DE_0 = OMEGA_DE0 * RHO_CRIT_0  # ≈ 2.888e-122 M_Pl^4

# Age of universe
T_UNIVERSE_S = 13.8e9 * 3.156e7  # seconds
T_UNIVERSE_PLANCK = T_UNIVERSE_S / T_PLANCK_S  # In Planck times

# =============================================================================
# DESI Targets (2024/2025)
# =============================================================================

# CPL parameters from DESI DR2 + Planck + SNe
# (Values vary by dataset combination - these are representative)

class DESITargets:
    """DESI dark energy constraints."""

    # Planck + DESI BAO only
    W0_DESI_BAO = -0.45
    W0_DESI_BAO_ERR = 0.21
    WA_DESI_BAO = -1.79
    WA_DESI_BAO_ERR = 0.65

    # Planck + DESI + Pantheon+
    W0_PANTHEON = -0.868
    W0_PANTHEON_ERR = 0.052
    WA_PANTHEON = -0.65  # approximate
    WA_PANTHEON_ERR = 0.30

    # Planck + DESI + Union3
    W0_UNION3 = -0.681
    W0_UNION3_ERR = 0.089
    WA_UNION3 = -1.25  # approximate
    WA_UNION3_ERR = 0.40

    # Default targets (conservative)
    W0_TARGET = -0.7
    W0_SIGMA = 0.2
    WA_TARGET = -1.0
    WA_SIGMA = 0.5


class LambdaCDMTargets:
    """Standard ΛCDM targets (for comparison)."""

    W0 = -1.0
    WA = 0.0

    # Cosmological constant
    LAMBDA_PLANCK = 2.888e-122  # In Planck units (M_Pl^4)
    LAMBDA_SIGMA_LOG10 = 1.0   # Allow 1 order of magnitude


# =============================================================================
# Standard Model Targets (unchanged from before)
# =============================================================================

class StandardModelTargets:
    """Standard Model physics targets."""

    ALPHA_EM = 7.297e-3        # Fine structure constant
    ALPHA_EM_SIGMA = 1e-5

    ALPHA_S = 0.118            # Strong coupling at M_Z
    ALPHA_S_SIGMA = 0.001

    SIN2_THETA_W = 0.231       # Weinberg angle
    SIN2_THETA_W_SIGMA = 0.001

    N_GEN = 3                  # Fermion generations
    N_GEN_SIGMA = 0.1          # Must be exactly 3
```

## Updated Genome

```python
@dataclass
class QuintessenceGenome:
    """Extended genome for quintessence search."""

    # Geometry
    polytope_id: int
    triangulation_id: int

    # Fluxes
    K: List[int]  # Length h21
    M: List[int]  # Length h21

    # Orientifold
    orientifold_mask: List[bool]

    # KKLT parameters
    t_init: List[float]  # Length h11, for branch selection

    # Quintessence-specific
    rolling_modulus_idx: int     # Which modulus rolls (0 to h11-1)
    T_initial: float             # Initial field value
    T_dot_initial: float = 0.0   # Initial velocity (usually 0)

    # Mode selection
    mode: str = "quintessence"   # "kklt" or "quintessence"
```

## Updated Fitness Function

```python
def compute_fitness_quintessence(
    genome: QuintessenceGenome,
    cy_data: CYData,
) -> float:
    """
    Compute fitness for quintessence mode.

    Targets:
    1. ρ_DE(today) ≈ 2.888e-122 M_Pl^4  (the tiny number)
    2. w₀ ≈ -0.7 (DESI)
    3. wₐ ≈ -1.0 (DESI)
    4. Standard Model physics (α_em, α_s, sin²θ_W, N_gen)
    """

    # 1. Partial KKLT stabilization
    t_stabilized, converged = solve_partial_kklt(
        cy_data.kappa,
        cy_data.c_i,
        cy_data.chi_D,
        rolling_idx=genome.rolling_modulus_idx,
        T_rolling=genome.T_initial,
        t_init=genome.t_init,
    )

    if not converged:
        return -np.inf

    # 2. Construct potential
    potential = compute_scalar_potential(
        cy_data.kappa,
        t_stabilized,
        genome.rolling_modulus_idx,
        cy_data.W0,
        cy_data.g_s,
        cy_data.A_i,
        cy_data.c_i,
        cy_data.exp_K0,
    )

    # 3. Evolve cosmology
    result = evolve_quintessence(
        potential,
        T_initial=genome.T_initial,
        T_dot_initial=genome.T_dot_initial,
    )

    # 4. Extract observables
    rho_DE_today = result.rho_DE[-1]
    w0, wa, _, _ = fit_cpl(result.a, result.w)

    # 5. Compute fitness components

    # Dark energy density (log scale because of 10^-122)
    log_rho_target = np.log10(RHO_DE_0)
    log_rho_computed = np.log10(rho_DE_today) if rho_DE_today > 0 else -200
    rho_score = -((log_rho_computed - log_rho_target) / 1.0)**2  # 1 dex tolerance

    # w0 score
    w0_score = -((w0 - DESITargets.W0_TARGET) / DESITargets.W0_SIGMA)**2

    # wa score
    wa_score = -((wa - DESITargets.WA_TARGET) / DESITargets.WA_SIGMA)**2

    # Standard Model (placeholder - same as before)
    sm_score = compute_sm_fitness(cy_data, t_stabilized)

    # Total fitness (can weight components differently)
    fitness = rho_score + w0_score + wa_score + sm_score

    return fitness
```

## Testing Strategy

### 1. Unit Tests for Individual Components

```python
def test_partial_kklt():
    """Test that partial stabilization works."""
    # Use McAllister 4-214-647 data
    # Fix one modulus, verify others still stabilize
    pass

def test_potential_derivatives():
    """Test V(T), dV/dT, d²V/dT² consistency."""
    # Numerical derivatives should match analytical
    pass

def test_cosmology_lambda():
    """Test that Λ limit gives w = -1."""
    # Constant potential should give w(t) = -1
    pass

def test_cpl_fitting():
    """Test CPL extraction on known function."""
    # Generate w(a) = w0 + wa(1-a), verify recovery
    pass
```

### 2. Integration Tests

```python
def test_known_quintessence_model():
    """Test against analytically known quintessence."""
    # Exponential potential V = V0 * exp(-λφ) has known solutions
    pass

def test_mcallister_as_limit():
    """Verify KKLT limit recovers McAllister results."""
    # With no rolling (T_dot = 0), should get static V0
    pass
```

### 3. DESI Validation

```python
def test_desi_targets():
    """Verify we can hit DESI targets with toy potential."""
    # Construct potential that should give (w0, wa) ≈ (-0.7, -1.0)
    # Verify the pipeline recovers these values
    pass
```

## Implementation Order

1. **constants.py** - Define all constants and targets
2. **potential.py** - V(T) computation (core physics)
3. **cosmology.py** - ODE solver for evolution
4. **cpl_fitting.py** - Extract (w0, wa)
5. **partial_kklt.py** - Modify existing KKLT solver
6. **Integration** - Connect to physics_bridge.py
7. **Tests** - Validate each component
8. **Genome update** - Add rolling modulus to GA

## Notes

- The cosmology solver uses `ln(a)` as time variable for numerical stability
- Slow-roll approximation is used where possible for speed
- Full evolution is ~seconds per configuration (not a bottleneck)
- The 10⁻¹²² problem remains - we still need tiny V(T_today)
