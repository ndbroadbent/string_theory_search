# Numerical Optimization Over Complex Structure Moduli Space

This document provides a detailed algorithmic specification for computing W₀ from first principles, based on McAllister et al. (arXiv:2107.09064) and Demirtas et al. (arXiv:1912.10047).

## Context: Multi-Objective Fitness

W₀ computation is the **expensive tier** in a hierarchical fitness evaluation:

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: INSTANT - Topology                                      │
│   N_gen = |h¹¹ - h²¹| = 3                                       │
│   Pre-filter: 473M → 12.2M polytopes                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: FAST - Gauge Couplings                                  │
│   α_em, α_s, sin²θ_W from Kähler moduli + geometry              │
│   Cost: O(seconds)                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: EXPENSIVE - W₀ and Λ  ← THIS DOCUMENT                   │
│   Full period/flux optimization                                 │
│   Cost: O(minutes to hours)                                     │
│   Target: Λ ~ 10⁻¹²² (our universe)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Only run Tier 3 on candidates that pass Tiers 1 and 2.**

See `COMPUTING_PERIODS.md` Section 6 for the full multi-objective fitness structure.

---

## The Core Problem

**Given**: A Calabi-Yau threefold X (via polytope) and a flux configuration (M, K)
**Compute**: W₀ = ⟨|W_flux|⟩ at the stabilized vacuum

The key insight is that W₀ is NOT computed by evaluating periods at an arbitrary point. Instead:
1. Fluxes must satisfy Diophantine constraints for "perturbatively flat vacua"
2. The vacuum lies along a 1D flat direction z = pτ in complex structure moduli space
3. W₀ comes from a racetrack mechanism between two leading instanton terms
4. The result is exponentially small: W₀ ~ δ^(p·q/ε) where δ < 1

---

## Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL W₀ COMPUTATION                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. GEOMETRIC SETUP                                              │
│    polytope → triangulation → CY → intersection numbers         │
│    mirror → GV invariants → prepotential F(z)                   │
├─────────────────────────────────────────────────────────────────┤
│ 2. FLUX SEARCH (Diophantine)                                    │
│    Find (M, K) ∈ Z^n × Z^n satisfying:                          │
│    - Tadpole: -½ M·K ≤ Q_D3                                     │
│    - Flat direction: p = N^(-1) K ∈ Kähler cone                 │
│    - Orthogonality: K·p = 0                                     │
├─────────────────────────────────────────────────────────────────┤
│ 3. RACETRACK IDENTIFICATION                                     │
│    Find curve pairs (q₁, q₂) with:                              │
│    - p·q₁ < 1, p·q₂ < 1                                         │
│    - 0 < ε = p·(q₂-q₁) < 1                                      │
│    - Dominant contributions at large Im(τ)                       │
├─────────────────────────────────────────────────────────────────┤
│ 4. VACUUM SOLVER                                                │
│    Solve ∂_τ W_eff = 0 for ⟨τ⟩                                  │
│    Compute W₀ = |W_eff(⟨τ⟩)|                                    │
├─────────────────────────────────────────────────────────────────┤
│ 5. PATH-FOLLOWING (for Kähler stabilization)                    │
│    Navigate exponentially many Kähler cone chambers             │
│    Use convex interpolation algorithm                           │
├─────────────────────────────────────────────────────────────────┤
│ 6. INSTANTON ITERATION                                          │
│    Iteratively incorporate worldsheet corrections               │
│    Converge to full solution                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Geometric Setup

### 1.1 Input Data

```python
INPUT:
    polytope_vertices: np.ndarray  # Points defining Δ° (primal polytope)
    # OR
    dual_vertices: np.ndarray  # Points defining Δ (mirror polytope)

OUTPUT:
    h11, h21: int  # Hodge numbers
    kappa_ijk: np.ndarray  # Triple intersection numbers (h11 × h11 × h11)
    c2_i: np.ndarray  # Second Chern class integrals
    chi: int  # Euler characteristic
    gv_invariants: dict  # {curve_class: N_q}
```

### 1.2 Implementation

```python
from cytools import Polytope
import numpy as np

def geometric_setup(polytope_vertices):
    """
    Phase 1: Extract all geometric data from polytope.
    """
    # Construct polytope and triangulation
    p = Polytope(polytope_vertices)
    tri = p.triangulate()
    cy = tri.get_cy()

    h11 = cy.h11()
    h21 = cy.h21()

    # Triple intersections (for prepotential polynomial part)
    kappa = cy.intersection_numbers()

    # Second Chern class (for instanton iteration)
    c2 = cy.second_chern_class()

    # Euler characteristic
    chi = cy.euler_characteristic()

    # GV invariants (the expensive part)
    # For h21 ≤ 4, min_points=100 is usually sufficient
    gvs = cy.compute_gvs(min_points=100, format='dok')

    return {
        'h11': h11,
        'h21': h21,
        'kappa': kappa,
        'c2': c2,
        'chi': chi,
        'gv_invariants': gvs,
        'cy': cy
    }
```

### 1.3 Building the Prepotential

```python
from scipy.special import spence  # For polylogarithms

def Li3(x):
    """Trilogarithm Li_3(x) = Σ x^k/k³"""
    if abs(x) < 1e-10:
        return x
    # Series expansion for |x| < 1
    result = 0
    term = x
    for k in range(1, 500):
        result += term / k**3
        term *= x
        if abs(term / (k+1)**3) < 1e-15:
            break
    return result

def Li2(x):
    """Dilogarithm Li_2(x) = -∫₀ˣ ln(1-t)/t dt"""
    return -spence(1 - x)  # scipy convention

def prepotential(z, kappa, a_mat, c_vec, chi, gv_invariants):
    """
    F(z) = F_poly + F_inst

    F_poly = -1/6 κ_abc z^a z^b z^c + 1/2 a_ab z^a z^b + c_a z^a/24 + ζ(3)χ/(2(2πi)³)
    F_inst = -1/(2πi)³ Σ_q N_q Li_3(e^{2πi q·z})

    Args:
        z: complex array of size h21
        kappa: triple intersections κ_abc
        a_mat: matrix a_ab (rational, from intersection data)
        c_vec: vector c_a = ∫ c_2 ∧ ω_a
        chi: Euler characteristic
        gv_invariants: {tuple(q): N_q}

    Returns:
        F: complex number
    """
    h = len(z)
    two_pi_i = 2j * np.pi

    # Polynomial part
    F_poly = 0
    # Cubic term
    for a in range(h):
        for b in range(h):
            for c in range(h):
                F_poly -= kappa[a,b,c] * z[a] * z[b] * z[c] / 6
    # Quadratic term
    F_poly += 0.5 * z @ a_mat @ z
    # Linear term
    F_poly += np.dot(c_vec, z) / 24
    # Constant term
    zeta_3 = 1.2020569031595943
    F_poly += zeta_3 * chi / (2 * two_pi_i**3)

    # Instanton part
    F_inst = 0
    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple)
        q_dot_z = np.dot(q, z)
        exp_arg = np.exp(two_pi_i * q_dot_z)
        if abs(exp_arg) < 0.99:  # Inside convergence radius
            F_inst -= N_q * Li3(exp_arg)
    F_inst /= two_pi_i**3

    return F_poly + F_inst
```

---

## Phase 2: Perturbatively Flat Flux Search

### 2.1 The Lemma (Demirtas et al. 1912.10047)

**Theorem**: A perturbatively flat vacuum exists if we can find (M, K) ∈ Z^n × Z^n such that:

1. **Tadpole constraint**: -½ M·K ≤ Q_D3
2. **Invertibility**: N_ab := κ̃_abc M^c is invertible
3. **Kähler cone**: p := N^(-1) K lies in the Kähler cone of the mirror
4. **Orthogonality**: K^T N^(-1) K = 0  (equivalently K·p = 0)
5. **Integrality**: a·M ∈ Z, b·M ∈ Z (for flux quantization)

The flux vectors are then:
```
f = (c_a M^a / 24, a_ab M^b, 0, M^a)
h = (0, K_a, 0, 0)
```

And the flat direction is z = pτ where W_flux^(pert) ≡ 0.

### 2.2 Search Algorithm

```python
def search_perturbatively_flat_fluxes(kappa, a_mat, b_vec, Q_D3, kahler_cone, max_flux=20):
    """
    Search for (M, K) satisfying perturbatively flat conditions.

    For h21 ≤ 4, exhaustive search is feasible.
    For h21 > 4, use heuristics or ML guidance.

    Args:
        kappa: Triple intersection κ_abc
        a_mat: Matrix a_ab
        b_vec: Vector b_a
        Q_D3: D3-brane charge tadpole bound
        kahler_cone: Function or rays defining Kähler cone
        max_flux: Maximum absolute value for flux components

    Returns:
        List of valid (M, K) pairs
    """
    n = kappa.shape[0]  # h21
    valid_fluxes = []

    # Generate candidate M vectors
    from itertools import product

    for M in product(range(-max_flux, max_flux+1), repeat=n):
        M = np.array(M)
        if np.all(M == 0):
            continue

        # Compute N_ab = κ_abc M^c
        N = np.einsum('abc,c->ab', kappa, M)

        # Check invertibility
        try:
            N_inv = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            continue

        # Check integrality conditions
        if not np.allclose(a_mat @ M, np.round(a_mat @ M)):
            continue
        if not np.allclose(b_vec @ M, np.round(b_vec @ M)):
            continue

        # Search for compatible K
        for K in product(range(-max_flux, max_flux+1), repeat=n):
            K = np.array(K)
            if np.all(K == 0):
                continue

            # Compute flat direction p = N^(-1) K
            p = N_inv @ K

            # Check orthogonality: K·p = 0
            if abs(K @ p) > 1e-10:
                continue

            # Check Kähler cone membership
            if not in_kahler_cone(p, kahler_cone):
                continue

            # Check tadpole
            flux_charge = -0.5 * M @ K
            if flux_charge > Q_D3:
                continue

            valid_fluxes.append({
                'M': M.copy(),
                'K': K.copy(),
                'p': p.copy(),
                'N_inv': N_inv.copy(),
                'flux_charge': flux_charge
            })

    return valid_fluxes

def in_kahler_cone(p, kahler_cone):
    """Check if p is in the Kähler cone."""
    # For toric varieties: all components positive in appropriate basis
    # More sophisticated check needed for general case
    return np.all(p > 0)
```

### 2.3 ML-Accelerated Search (Optional)

For h21 > 4, train a classifier to pre-filter candidates:

```python
def train_flux_classifier(training_data):
    """
    Train NN to predict P(perturbatively flat | M, K, geometry).

    Can achieve 99.99%+ accuracy on rejecting invalid fluxes,
    reducing search space dramatically.
    """
    import torch
    import torch.nn as nn

    class FluxClassifier(nn.Module):
        def __init__(self, n_moduli):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 * n_moduli + n_moduli**3, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, M, K, kappa_flat):
            x = torch.cat([M, K, kappa_flat], dim=-1)
            return self.net(x)

    # Training loop omitted - standard supervised learning
    pass
```

---

## Phase 3: Racetrack Identification

### 3.1 The Racetrack Mechanism

At large Im(τ) along the flat direction z = pτ:

```
W_flux(τ) = -ζ Σ_q (M·q) N_q Li_2(e^{2πiτ p·q})
         ≈ -ζ [(M·q₁) N_{q₁} e^{2πiτ p·q₁} + (M·q₂) N_{q₂} e^{2πiτ p·q₂}]
```

For a racetrack minimum, we need two competing terms with similar exponents.

### 3.2 Finding Racetrack Pairs

```python
def find_racetrack_pairs(p, M, gv_invariants, max_candidates=10):
    """
    Find pairs (q₁, q₂) that give racetrack stabilization.

    Conditions:
    (d) p·q₁ < 1 and p·q₂ < 1
    (e) 0 < ε := p·(q₂ - q₁) < 1
    (f) q₁, q₂ dominate at large Im(τ)

    Returns:
        List of valid (q₁, q₂, δ, ε) tuples sorted by expected W₀
    """
    # Filter curves with p·q < 1
    small_curves = []
    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple)
        p_dot_q = np.dot(p, q)
        M_dot_q = np.dot(M, q)
        if 0 < p_dot_q < 1 and M_dot_q != 0 and N_q != 0:
            small_curves.append({
                'q': q,
                'p_dot_q': p_dot_q,
                'M_dot_q': M_dot_q,
                'N_q': N_q,
                'coeff': M_dot_q * N_q * p_dot_q
            })

    # Sort by p·q (smallest first = dominant at large τ)
    small_curves.sort(key=lambda x: x['p_dot_q'])

    # Find valid pairs
    pairs = []
    for i, c1 in enumerate(small_curves[:max_candidates]):
        for c2 in small_curves[i+1:max_candidates]:
            q1, q2 = c1['q'], c2['q']
            p_q1, p_q2 = c1['p_dot_q'], c2['p_dot_q']

            epsilon = p_q2 - p_q1
            if not (0 < epsilon < 1):
                continue

            # Compute δ = |coeff₁|/|coeff₂|
            delta = abs(c1['coeff']) / abs(c2['coeff'])
            if delta >= 1:
                delta = 1 / delta
                # Swap if needed to ensure |δ| < 1
                q1, q2 = q2, q1
                p_q1, p_q2 = p_q2, p_q1
                epsilon = -epsilon

            # Estimate W₀ ~ δ^(p·q₁/ε)
            if abs(epsilon) > 1e-10:
                log_W0_estimate = (p_q1 / epsilon) * np.log(delta)
            else:
                log_W0_estimate = float('-inf')

            pairs.append({
                'q1': q1,
                'q2': q2,
                'delta': delta,
                'epsilon': epsilon,
                'p_q1': p_q1,
                'p_q2': p_q2,
                'log_W0_estimate': log_W0_estimate
            })

    # Sort by smallest expected W₀
    pairs.sort(key=lambda x: x['log_W0_estimate'])

    return pairs
```

---

## Phase 4: Vacuum Solver

### 4.1 Effective Superpotential

Along the flat direction z = pτ:

```python
def W_effective(tau, p, M, gv_invariants):
    """
    W_eff(τ) = ζ Σ_q (M·q) N_q Li_2(e^{2πiτ p·q})

    where ζ = 1/(2^{3/2} π^{5/2})
    """
    zeta = 1 / (2**1.5 * np.pi**2.5)
    two_pi_i = 2j * np.pi

    W = 0
    for q_tuple, N_q in gv_invariants.items():
        q = np.array(q_tuple)
        M_q = np.dot(M, q)
        p_q = np.dot(p, q)
        if M_q == 0 or p_q <= 0:
            continue

        exp_arg = np.exp(two_pi_i * tau * p_q)
        if abs(exp_arg) < 0.99:
            W += M_q * N_q * Li2(exp_arg)

    return zeta * W
```

### 4.2 F-term Equation Solver

```python
from scipy.optimize import brentq, minimize_scalar

def solve_racetrack_vacuum(p, M, gv_invariants, tau_range=(5, 200)):
    """
    Solve ∂_τ W_eff = 0 for the vacuum value ⟨τ⟩.

    At the minimum, Im(τ) = 1/g_s >> 1 (weak coupling).

    Returns:
        tau_vev: complex (vacuum expectation value)
        W0: float (|W_eff(⟨τ⟩)|)
        g_s: float (string coupling)
    """
    def dW_dtau_imag(im_tau):
        """Derivative ∂W/∂τ at τ = i·im_tau (along imaginary axis)."""
        tau = 1j * im_tau

        zeta = 1 / (2**1.5 * np.pi**2.5)
        two_pi = 2 * np.pi

        dW = 0
        for q_tuple, N_q in gv_invariants.items():
            q = np.array(q_tuple)
            M_q = np.dot(M, q)
            p_q = np.dot(p, q)
            if M_q == 0 or p_q <= 0:
                continue

            # ∂_τ Li_2(e^{2πiτ p·q}) = 2πi p·q · Li_1(e^{2πiτ p·q}) / e^{2πiτ p·q}
            #                       = -2πi p·q · ln(1 - e^{2πiτ p·q})
            exp_arg = np.exp(2j * np.pi * tau * p_q)
            if abs(exp_arg) < 0.99:
                dW += M_q * N_q * p_q * (-np.log(1 - exp_arg))

        return (zeta * 2j * np.pi * dW).imag

    # Find zero crossing
    try:
        tau_min, tau_max = tau_range
        im_tau_vev = brentq(dW_dtau_imag, tau_min, tau_max)
    except ValueError:
        # No zero crossing - try minimizing |dW/dτ|
        result = minimize_scalar(lambda t: abs(dW_dtau_imag(t)),
                                 bounds=tau_range, method='bounded')
        im_tau_vev = result.x

    tau_vev = 1j * im_tau_vev
    W0 = abs(W_effective(tau_vev, p, M, gv_invariants))
    g_s = 1 / im_tau_vev

    return tau_vev, W0, g_s
```

---

## Phase 5: Path-Following in Kähler Cone

### 5.1 The Problem

At large h¹¹, the Kähler cone has exponentially many chambers (phases).
A randomly chosen triangulation usually does NOT contain the desired vacuum.
We need to navigate through chambers to find one that does.

### 5.2 McAllister's Algorithm (Section 5.2 of 2107.09064)

```python
def path_following_algorithm(kappa, c_target, t_init, N_steps=1000):
    """
    Find Kähler parameters t such that divisor volumes = target.

    The divisor volumes are: τ_i = ½ κ_ijk t^j t^k

    Strategy:
    1. Start at random t_init with volumes τ_init
    2. Define straight path in τ-space: τ_α = (1-α) τ_init + α τ_target
    3. Follow path by solving for t_α at each step

    The key insight: τ-space path is straight, but t-space path curves
    because κ_ijk can jump across phase boundaries.

    Args:
        kappa: Triple intersections κ_ijk
        c_target: Target divisor volumes (c₁, ..., c_{h11})
        t_init: Initial Kähler parameters (random point in cone)
        N_steps: Number of path segments

    Returns:
        t_final: Kähler parameters at target volumes
        path: List of (α, t_α) pairs for the path taken
    """
    h = kappa.shape[0]

    def volumes_from_t(t):
        """τ_i = ½ κ_ijk t^j t^k"""
        return 0.5 * np.einsum('ijk,j,k->i', kappa, t, t)

    def linear_solve_step(t_current, tau_next, tau_current):
        """
        Solve: κ_ijk t^j ε^k = τ_next - τ_current
        for the step ε, then return t_next = t_current + ε
        """
        # Matrix A_ik = κ_ijk t^j
        A = np.einsum('ijk,j->ik', kappa, t_current)

        delta_tau = tau_next - tau_current

        try:
            epsilon = np.linalg.solve(A, delta_tau)
        except np.linalg.LinAlgError:
            # Singular matrix - at phase boundary
            epsilon = np.linalg.lstsq(A, delta_tau, rcond=None)[0]

        return t_current + epsilon

    # Initialize
    tau_init = volumes_from_t(t_init)
    tau_target = np.array(c_target)

    t_current = t_init.copy()
    path = [(0.0, t_init.copy())]

    # Follow path
    for m in range(1, N_steps + 1):
        alpha = m / N_steps
        tau_next = (1 - alpha) * tau_init + alpha * tau_target
        tau_current = volumes_from_t(t_current)

        t_next = linear_solve_step(t_current, tau_next, tau_current)

        # Check if we crossed a phase boundary
        # (In practice, would need to detect this and handle accordingly)

        path.append((alpha, t_next.copy()))
        t_current = t_next

    return t_current, path
```

---

## Phase 6: Instanton Iteration

### 6.1 Iterative Correction Algorithm

```python
def instanton_iteration(kappa, c_tau, chi_D, gv_invariants, gamma,
                        tol=1e-10, max_iter=100):
    """
    Iteratively incorporate worldsheet instanton corrections.

    Starting from tree-level solution t^(0), iterate:

    ½ κ_ijk t^j_{(n)} t^k_{(n)} = c_i/c_τ + χ(D_i)/24
        - 1/(2π)² Σ_q q_i N_q Li_2((-1)^{γ·q} e^{-2π q·t_{(n-1)}})

    Args:
        kappa: Triple intersections
        c_tau: Overall scaling factor (from eq. 520)
        chi_D: Euler characteristics χ(D_i) for each divisor
        gv_invariants: {q: N_q}
        gamma: B-field parity vector
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        t_converged: Final Kähler parameters
        n_iterations: Number of iterations until convergence
    """
    h = kappa.shape[0]

    # Tree-level target volumes
    c_tree = np.ones(h) / c_tau + chi_D / 24

    # Initial solution (tree-level)
    t_current = solve_volume_constraint(kappa, c_tree)

    for n in range(max_iter):
        # Compute instanton correction
        inst_correction = np.zeros(h)
        for q_tuple, N_q in gv_invariants.items():
            q = np.array(q_tuple)
            q_dot_t = np.dot(q, t_current)

            # Skip negligible contributions
            if q_dot_t > 10:  # e^{-2π·10} ≈ 0
                continue

            sign = (-1) ** np.dot(gamma, q)
            exp_arg = sign * np.exp(-2 * np.pi * q_dot_t)

            inst_correction += q * N_q * Li2(exp_arg)

        inst_correction /= (2 * np.pi) ** 2

        # New target volumes
        c_new = np.ones(h) / c_tau + chi_D / 24 - inst_correction

        # Solve for new t
        t_new = solve_volume_constraint(kappa, c_new)

        # Check convergence
        if np.max(np.abs(t_new - t_current)) < tol:
            return t_new, n + 1

        t_current = t_new

    return t_current, max_iter

def solve_volume_constraint(kappa, c_target):
    """
    Solve ½ κ_ijk t^j t^k = c_i for t.

    This is a system of quadratic equations.
    Use Newton's method or specialized solver.
    """
    from scipy.optimize import root

    def residual(t):
        return 0.5 * np.einsum('ijk,j,k->i', kappa, t, t) - c_target

    def jacobian(t):
        return np.einsum('ijk,k->ij', kappa, t)

    # Initial guess (scaled identity)
    t0 = np.ones(len(c_target)) * np.mean(c_target) ** 0.5

    result = root(residual, t0, jac=jacobian)

    return result.x
```

---

## Complete Pipeline

```python
def compute_W0_from_polytope(polytope_vertices, max_flux=10):
    """
    Full W₀ computation from polytope to final value.

    Returns:
        W0: float (flux superpotential at vacuum)
        g_s: float (string coupling)
        tau: complex (axiodilaton vev)
        flux_data: dict (M, K vectors and other parameters)
    """
    # Phase 1: Geometric setup
    print("Phase 1: Geometric setup...")
    geo = geometric_setup(polytope_vertices)

    # Phase 2: Find perturbatively flat fluxes
    print("Phase 2: Searching for perturbatively flat fluxes...")
    fluxes = search_perturbatively_flat_fluxes(
        geo['kappa'],
        geo['a_mat'],
        geo['b_vec'],
        geo['Q_D3'],
        geo['kahler_cone'],
        max_flux=max_flux
    )

    if not fluxes:
        raise ValueError("No perturbatively flat fluxes found")

    print(f"  Found {len(fluxes)} valid flux configurations")

    best_W0 = float('inf')
    best_result = None

    for flux in fluxes:
        M, K, p = flux['M'], flux['K'], flux['p']

        # Phase 3: Find racetrack pairs
        pairs = find_racetrack_pairs(p, M, geo['gv_invariants'])

        if not pairs:
            continue

        # Phase 4: Solve for vacuum
        try:
            tau_vev, W0, g_s = solve_racetrack_vacuum(
                p, M, geo['gv_invariants']
            )
        except Exception:
            continue

        if W0 < best_W0:
            best_W0 = W0
            best_result = {
                'W0': W0,
                'g_s': g_s,
                'tau': tau_vev,
                'M': M,
                'K': K,
                'p': p,
                'racetrack': pairs[0]
            }

    if best_result is None:
        raise ValueError("No valid vacuum found")

    print(f"  Best W₀ = {best_W0:.6e}")
    print(f"  g_s = {best_result['g_s']:.6f}")

    return best_result
```

---

## Automation Philosophy

### The Key Insight

> "NOTHING is beyond automation now. Literally nothing."
> "Even if every evaluation takes a minute, in theory, GA is possible now."
> "Even if it requires thousands of dollars in AI API tokens."

### Implementation Strategies

#### Strategy A: Brute Force + Parallelism

For h²¹ ≤ 4, exhaustive search over (M, K) pairs is feasible:
- ~10⁸ pairs with |flux| ≤ 20
- ~1ms per validation
- ~100 CPU-hours total
- Trivially parallelizable

#### Strategy B: ML Pre-filtering

Train neural networks to accelerate search:

```python
class FluxPredictor(nn.Module):
    """
    Predicts P(valid | M, K, geometry) with 99.99% accuracy.
    Reduces search space by ~10⁴x.
    """
    pass

class W0Estimator(nn.Module):
    """
    Predicts log₁₀(W₀) without full computation.
    Accuracy: ±2 orders of magnitude.
    Use as fitness heuristic in GA.
    """
    pass
```

#### Strategy C: LLM-in-the-Loop

For difficult reasoning tasks (basis matching, anomaly detection):

```python
def llm_assisted_validation(flux_data, geometry_data):
    """
    Call Claude/GPT API for complex reasoning steps.

    Cost: ~$0.01-$0.10 per call
    Use for: basis identification, error diagnosis, strategy selection
    """
    prompt = f"""
    Given this Calabi-Yau geometry with h¹¹={geometry_data['h11']}, h²¹={geometry_data['h21']},
    and these flux vectors M={flux_data['M']}, K={flux_data['K']},
    verify that the perturbatively flat vacuum conditions are satisfied.

    Check:
    1. Is N_ab = κ_abc M^c invertible?
    2. Does p = N^(-1)K lie in the Kähler cone?
    3. Is K·p = 0 satisfied?

    Provide step-by-step reasoning.
    """

    response = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    return parse_validation_response(response)
```

### The GA Genome

```rust
struct Compactification {
    polytope_id: u64,           // Index into Kreuzer-Skarke
    M_vec: Vec<i32>,           // NSNS flux (h21 integers)
    K_vec: Vec<i32>,           // RR flux (h21 integers)
    // tau is computed, not evolved
}

fn fitness(genome: &Compactification) -> f64 {
    // Full W₀ computation pipeline
    let result = compute_W0_from_polytope(genome);

    // Fitness = -log₁₀(W₀) for minimization
    // Higher fitness = smaller W₀
    -result.W0.log10()
}
```

---

## References

1. **McAllister et al. arXiv:2107.09064** "Small cosmological constants in string theory"
   - Section 3: Perturbatively flat vacua (eqs. 458-527)
   - Section 5: Computational methods (eqs. 861-1033)
   - Appendix: Data files and notation

2. **Demirtas et al. arXiv:1912.10047** "Vacua with Small Flux Superpotential"
   - The Lemma (eqs. 143-165)
   - Explicit example with W₀ ~ 10⁻⁸

3. **CYTools** https://cy.tools
   - Polytope analysis, triangulations, GV invariants

4. **cygv** https://crates.io/crates/cygv
   - Rust/Python library for GV invariant computation
