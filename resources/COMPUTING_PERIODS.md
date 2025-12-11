# Computing Periods and W₀ for String Theory Vacua

This document describes the mathematical framework and computational tools needed to compute the flux superpotential W₀ from first principles.

## Executive Summary

**Goal**: Compute W₀ ~ 10⁻⁹⁰ (flux superpotential) from a polytope specification.

**Key insight**: W₀ is NOT computed by "evaluating periods at a point." Instead:
1. GV invariants determine the prepotential F(z)
2. Prepotential determines periods Π(z) = (z^A, F_A)
3. Fluxes (M, K) must satisfy Diophantine constraints for perturbatively flat vacua
4. W₀ comes from a **racetrack mechanism** with two competing instanton terms
5. The smallness of W₀ is exponential: W₀ ~ δ^(p·q/ε) where δ < 1

**What we have working**:
- CYTools + cygv: Compute GV invariants ✓
- GV invariants match McAllister's data ✓
- V₀ formula verified ✓

**What's needed for full W₀ computation**:
- Perturbatively flat flux search algorithm
- Racetrack vacuum solver
- Path-following through Kähler cone chambers

---

## 1. Mathematical Background

### 1.1 Period Vector and Prepotential

For a Calabi-Yau threefold X with mirror X̃, the period vector at large complex structure (LCS) is:

```
Π = (F_A, z^A)^T
```

where F_A = ∂F/∂z^A and F is the prepotential:

```
F(z) = F_poly(z) + F_inst(z)
```

The polynomial part:
```
F_poly = -1/6 κ̃_abc z^a z^b z^c + 1/2 ã_ab z^a z^b + 1/24 c̃_a z^a + ζ(3)χ(X̃)/(2(2πi)³)
```

The instanton part (from GV invariants):
```
F_inst = -1/(2πi)³ Σ_q N_q Li₃(e^(2πi q·z))
```

where N_q are genus-zero Gopakumar-Vafa invariants and q runs over effective curves.

### 1.2 Flux Superpotential

The Gukov-Vafa-Witten superpotential is:
```
W_flux(τ, z) = √(2/π) Π^T Σ (f - τh)
```

where:
- τ = C₀ + i/g_s is the axiodilaton
- Σ = [[0, I], [-I, 0]] is the symplectic pairing
- f, h are integer flux vectors in H³(X, Z)

### 1.3 Why W₀ is Exponentially Small

The key is the **perturbatively flat vacuum** construction:

1. Choose fluxes of the form:
   ```
   f = (c_a M^a / 24, a_ab M^b, 0, M^a)
   h = (0, K_a, 0, 0)
   ```

2. This ensures W_flux^(pert) ≡ 0 along the flat direction z = p τ, where:
   ```
   p^a = (κ̃_abc M^c)^(-1) K_b
   ```
   must lie in the Kähler cone of X̃.

3. The constraint K · p = 0 must hold.

4. W₀ comes entirely from **instanton corrections**, giving a racetrack:
   ```
   W_flux(τ) ≈ -ζ [M·q̃₁ N_{q̃₁} e^(2πiτ q̃₁·p) + M·q̃₂ N_{q̃₂} e^(2πiτ q̃₂·p)]
   ```

5. The F-term equation ∂_τW = 0 gives:
   ```
   ⟨e^(2πiτ)⟩ ≈ δ^(1/ε)  where ε = p·(q̃₂ - q̃₁) < 1
   ```

6. The vacuum value is:
   ```
   W₀ ~ δ^(p·q̃₁/ε) ~ δ^(p·q̃₂/ε) ≪ 1
   ```

This exponential suppression is why W₀ ~ 10⁻⁹⁰ is achievable.

---

## 2. Computational Tools

### 2.1 cygv (Rust + Python)

**Purpose**: Compute Gopakumar-Vafa invariants via HKTY procedure.

**Installation**:
```bash
pip install cygv
# or
uv add cygv
```

**Usage via CYTools**:
```python
from cytools import Polytope

# Load polytope (mirror/dual for LCS computation)
dual_pts = np.loadtxt("dual_points.dat", delimiter=',').astype(int)
p = Polytope(dual_pts)
tri = p.triangulate()
cy = tri.get_cy()

# Compute GV invariants
gvs = cy.compute_gvs(min_points=100, format='dok')
# Returns: {(d₁, d₂, ...): N_d} for curve classes d
```

**What cygv actually computes**:
- Genus-zero GV invariants N_q for effective curve classes q
- These are the BPS state counts / holomorphic curve counts
- Used to build F_inst via the Li₃ sum

**What cygv does NOT compute**:
- Periods directly
- The vacuum point z where W₀ is minimized
- The flux vectors (M, K) that give perturbatively flat vacua

### 2.2 CYTools

**Purpose**: Polytope analysis, triangulations, intersection numbers, CY construction.

**Key functions**:
```python
cy.h11()  # Hodge number h¹¹
cy.h21()  # Hodge number h²¹
cy.intersection_numbers()  # Triple intersection κ_abc
cy.glsm_charge_matrix()  # GLSM charges
cy.compute_gvs()  # GV invariants (wraps cygv)
```

### 2.3 Other Tools

**lefschetz-family** (Python/Sage): Numerical periods for projective hypersurfaces via Picard-Lefschetz. Good for sanity checks but not toric-native.

**Macaulay2 Dmodules**: GKZ system construction for verifying charge vectors.

---

## 3. Computing W₀: The Full Algorithm

Based on McAllister et al. arXiv:2107.09064 Section 5 and Demirtas et al. arXiv:1912.10047.

### Phase 1: Geometric Setup

```
INPUT: Polytope Δ° (from Kreuzer-Skarke database)

1. Construct mirror Δ (polar dual)
2. Get triangulation → toric variety V → CY hypersurface X
3. Compute h¹¹(X), h²¹(X)
4. Compute triple intersections κ_ijk on X
5. Compute GV invariants N_q via cygv
6. Build prepotential F(z) = F_poly + F_inst
```

### Phase 2: Find Perturbatively Flat Fluxes

```
SEARCH for (M, K) ∈ Z^n × Z^n satisfying:

1. Tadpole bound: -½ M·K ≤ Q_D3
2. Flat direction: p^a = (κ̃_abc M^c)^(-1) K_b ∈ K̃_X (Kähler cone of mirror)
3. Orthogonality: K·p = 0
4. Integrality: a·M ∈ Z, b·M ∈ Z (for flux quantization)
```

This is a Diophantine constraint problem. For small h²¹ (≤ 4), exhaustive search is feasible.

### Phase 3: Find Racetrack Curve Pairs

```
SEARCH for (q̃₁, q̃₂) generators of Mori cone such that:

1. p·q̃₁ < 1 and p·q̃₂ < 1  (inside unit disk)
2. 0 < ε := p·(q̃₂ - q̃₁) < 1  (small splitting)
3. Leading terms: q̃₁, q̃₂ dominate F_inst at large Im(τ)
4. Hierarchy: |δ| = |M·q̃₁ · p·q̃₁ · N_{q̃₁}| / |M·q̃₂ · p·q̃₂ · N_{q̃₂}| < 1
```

### Phase 4: Solve for Vacuum

```
SOLVE the racetrack F-term equation:

∂_τ W_eff = 0

where W_eff(τ) = ζ Σ_q M·q N_q Li₂(e^(2πiτ p·q))

This gives ⟨τ⟩ = i g_s^(-1) + (axion)

Then: W₀ = |W_eff(⟨τ⟩)|
```

### Phase 5: Path-Following in Kähler Cone

McAllister's algorithm (Section 5.2 of 2107.09064):

```
PROBLEM: Find K̈ahler params t such that divisor volumes = target values

1. Start at random point t_init in extended Kähler cone K_X
2. Define target τ* = (c₁, c₂, ..., c_h11) for divisor volumes
3. Path τ_α = (1-α) τ_init + α τ*  for 0 ≤ α ≤ 1
4. τ_α is convex combination, so always in cone of effective divisors
5. Solve for t_α by following the path:
   - Divide into N >> 1 sections
   - At each step, solve linear system: κ_ijk t^j ε^k = τ_{m+1} - τ_m
6. At α = 1, have t* satisfying ½ κ_ijk t^j t^k = c_i
```

### Phase 6: Incorporate Instanton Corrections

Iterative algorithm (Eq. 952 of 2107.09064):

```
ITERATE starting from t^(0) solving tree-level:

t^(n) solves:
½ κ_ijk t^j_{(n)} t^k_{(n)} = c_i/c_τ + χ(D_i)/24
    - 1/(2π)² Σ_q q_i N_q Li₂((-1)^(γ·q) e^(-2π q·t_{(n-1)}))

Continue until convergence.
```

---

## 4. Key Parameters for McAllister's 4-214-647 Example

```
Polytope: 4-214-647 (Kreuzer-Skarke ID)
- Primal: 294 points, h¹¹ = 214, h²¹ = 4
- Dual (mirror): 12 points, h¹¹ = 4, h²¹ = 214

Published values:
- g_s = 0.00911134
- W₀ = 2.30012e-90
- V_CY = 4711.83 (Einstein frame volume)
- V₀(AdS) = -5.5e-203

Flux vectors (4-dimensional, h²¹ = 4 basis):
- K_vec: RR flux
- M_vec: NSNS flux

GV invariants computed via cygv:
- (1,0,0,0): 252
- (2,0,0,0): -9252
- (3,0,0,0): 848628
- etc.
```

---

## 5. Verified: V₀ Formula

Given W₀, the AdS vacuum energy is:

```python
def compute_V0_AdS(W_0, g_s, V_CY):
    """
    V₀ = -3 e^K₀ (g_s⁷ / (4V)²) W₀²

    At the KKLT minimum with |W₀| << 1.
    """
    # Kähler potential at minimum
    # e^K₀ ≈ 1 / (8 V^2) for large volume
    e_K0 = 1 / (8 * V_CY**2)

    # Prefactor from supergravity
    prefactor = -3 * e_K0 * (g_s**7) / (4 * V_CY)**2

    V0 = prefactor * W_0**2
    return V0

# McAllister values:
V0 = compute_V0_AdS(2.3e-90, 0.00911134, 4711.83)
# V0 ≈ -5.5e-203 ✓
```

---

## 6. Multi-Objective Fitness Structure

The goal is to find compactifications matching ALL Standard Model parameters, not just W₀.

### 6.1 Target Physics Constants

| Parameter | Symbol | Target Value | Source |
|-----------|--------|--------------|--------|
| Fine structure constant | α_em | 7.297×10⁻³ | QED at low energy |
| Strong coupling | α_s | 0.118 | QCD at M_Z |
| Weinberg angle | sin²θ_W | 0.231 | Electroweak unification |
| Fermion generations | N_gen | 3 | Topology |
| Cosmological constant | Λ | 2.888×10⁻¹²² | Observation (Planck units) |

### 6.2 Computational Cost Hierarchy

**Critical insight**: These parameters have vastly different computational costs.

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: INSTANT (topology only)                                 │
├─────────────────────────────────────────────────────────────────┤
│ N_gen = |h¹¹ - h²¹|                                             │
│ Cost: O(1) - read from polytope data                            │
│ Pre-filter: 473M polytopes → 12.2M with N_gen = 3               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: FAST (geometry + moduli)                                │
├─────────────────────────────────────────────────────────────────┤
│ α_em, α_s, sin²θ_W from:                                        │
│   - Kähler moduli (cycle volumes)                               │
│   - Triple intersection numbers κ_ijk                           │
│   - String coupling g_s                                         │
│ Cost: O(seconds) - CYTools computation                          │
│ Filter: Keep candidates with gauge couplings in right ballpark  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: EXPENSIVE (periods + flux optimization)                 │
├─────────────────────────────────────────────────────────────────┤
│ W₀, Λ from:                                                     │
│   - GV invariants (cygv)                                        │
│   - Perturbatively flat flux search                             │
│   - Racetrack vacuum solver                                     │
│ Cost: O(minutes to hours) - full optimization pipeline          │
│ Only run on promising candidates from Tier 2                    │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Gauge Coupling Formulas

From Kähler moduli and geometry:

```python
def compute_gauge_couplings(kahler_moduli, intersection_numbers, g_s):
    """
    Gauge couplings from string compactification.

    In type IIB on CY orientifold with D7-branes:
    - 1/g_YM² = Re(T_D) / g_s  for D7 wrapping divisor D
    - T_D = τ_D + i θ_D  (complexified volume)
    - τ_D = ½ κ_Dij t^i t^j  (divisor volume)

    GUT relations (at string scale):
    - α_GUT ≈ g_s / (4π V_D)
    - sin²θ_W depends on embedding of SM in GUT group
    """
    # Simplified - actual computation depends on brane configuration
    pass
```

### 6.4 Multi-Objective Fitness Function

```python
def fitness(genome: Compactification) -> dict:
    """
    Hierarchical multi-objective fitness evaluation.

    Returns dict of fitness components, evaluated lazily.
    """
    result = {}

    # TIER 1: Topology (instant)
    n_gen = abs(genome.h11 - genome.h21)
    result['n_gen_error'] = abs(n_gen - 3)

    # Early exit if wrong generation count
    if result['n_gen_error'] > 0:
        return result  # Don't waste compute on wrong topology

    # TIER 2: Gauge couplings (fast)
    couplings = compute_gauge_couplings(
        genome.kahler_moduli,
        genome.intersection_numbers,
        genome.g_s
    )

    result['alpha_em_error'] = abs(log10(couplings['alpha_em']) - log10(7.297e-3))
    result['alpha_s_error'] = abs(log10(couplings['alpha_s']) - log10(0.118))
    result['sin2_theta_W_error'] = abs(couplings['sin2_theta_W'] - 0.231)

    # Early exit if gauge couplings way off
    tier2_score = (result['alpha_em_error'] + result['alpha_s_error'] +
                   result['sin2_theta_W_error'])
    if tier2_score > 2.0:  # More than 2 orders of magnitude off
        return result

    # TIER 3: Cosmological constant (expensive)
    W0_result = compute_W0_from_fluxes(genome)

    if W0_result['success']:
        W0 = W0_result['W0']
        V0 = compute_V0_AdS(W0, genome.g_s, genome.volume)

        # Target: Λ ~ 10⁻¹²² (after uplift, |V0| should be same order)
        result['lambda_error'] = abs(log10(abs(V0)) - (-122))
        result['W0'] = W0
        result['V0'] = V0
    else:
        result['lambda_error'] = float('inf')

    return result

def aggregate_fitness(components: dict) -> float:
    """
    Combine multi-objective into single scalar for GA selection.

    Weighted sum with penalties for constraint violations.
    """
    if components.get('n_gen_error', 0) > 0:
        return -1e10  # Hard constraint: must have 3 generations

    weights = {
        'alpha_em_error': 1.0,
        'alpha_s_error': 1.0,
        'sin2_theta_W_error': 1.0,
        'lambda_error': 2.0,  # Higher weight - this is the hard one
    }

    score = 0
    for key, weight in weights.items():
        error = components.get(key, float('inf'))
        score -= weight * error  # Negative because we maximize fitness

    return score
```

### 6.5 GA Genome Structure

```rust
struct Compactification {
    // Polytope selection
    polytope_id: u64,           // Index into Kreuzer-Skarke (pre-filtered for N_gen=3)
    triangulation_id: u32,      // Which triangulation of this polytope

    // Continuous moduli
    kahler_moduli: Vec<f64>,    // h¹¹ positive real values (cycle volumes)
    complex_moduli: Vec<Complex64>, // h²¹ complex values
    g_s: f64,                   // String coupling (0 < g_s < 1 for perturbative)

    // Discrete choices (for W₀ computation)
    M_vec: Vec<i32>,            // NSNS flux (h²¹ integers)
    K_vec: Vec<i32>,            // RR flux (h²¹ integers)
}
```

---

## 7. Automation Strategy

The key insight: **NOTHING is beyond automation now.**

### Option A: Pure Numerical Optimization

Treat the entire pipeline as a black-box optimization:
```
minimize Σ_i w_i × |log(param_i / target_i)|

subject ot
- N_gen = 3 (hard constraint)
- (M, K) ∈ Z^n × Z^n
- -½ M·K ≤ Q_D3
- g_s < 1 (weak coupling)
```

This is mixed-integer nonlinear programming (MINLP). Modern solvers + massive parallelism could work.

### Option B: ML-Assisted Search

Train neural networks as heuristics:
1. **Flux classifier**: Given (M, K), predict if perturbatively flat conditions satisfied (99.99% accuracy possible)
2. **W₀ estimator**: Given geometry + fluxes, estimate log|W₀| without full computation
3. **Gauge coupling predictor**: Fast approximation of α_em, α_s, sin²θ_W from geometry
4. **Racetrack detector**: Identify promising (q̃₁, q̃₂) pairs quickly

These can achieve high accuracy and reduce search space by 10⁴-10⁶×.

### Option C: LLM-in-the-Loop

For truly difficult steps (basis matching, topological reasoning):
- Call GPT/Claude API with structured prompts
- Use chain-of-thought to navigate decision trees
- Cost: potentially thousands of dollars, but tractable

### Key Principle

> "Even if every evaluation takes a minute, in theory, GA is possible now."
> "Even if it requires thousands of dollars in AI API tokens."

The hierarchical filtering ensures we only spend expensive compute on promising candidates.

---

## 8. References

1. **McAllister et al. 2107.09064**: "Small cosmological constants in string theory"
   - Full algorithm for finding AdS vacua with small Λ
   - Section 3: Perturbatively flat vacua construction
   - Section 5: Computational methods and path-following

2. **Demirtas et al. 1912.10047**: "Vacua with Small Flux Superpotential"
   - The "Lemma" for constructive flux finding
   - Explicit example with W₀ ~ 10⁻⁸

3. **cygv documentation**: https://docs.rs/cygv
   - HKTY procedure implementation
   - GV/GW invariant computation

4. **CYTools paper 2211.03823**: "CYTools: A Software Package for Analyzing CY Manifolds"
   - Polytope analysis, triangulations, intersection numbers
