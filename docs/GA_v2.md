# GA v2: Redesigned Search Architecture

This document describes the redesigned genetic algorithm architecture based on insights from reproducing McAllister et al. (arXiv:2107.09064).

## McAllister Ground Truth

We have validated data from McAllister et al. (arXiv:2107.09064) for polytope 4-214-647:

```python
# Polytope - PRIMAL has h11=214 (the one McAllister actually uses)
primal_points = "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/points.dat"
# 294 points, h11=214, h21=4

# Polytope - DUAL has h11=4 (mirror of primal)
dual_points = "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/dual_points.dat"
# 12 points, h11=4, h21=214

# Triangulation of dual (15 simplices - FRST)
simplices = "resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/dual_simplices.dat"

# Fluxes (for h11=4 dual)
K = [-3, -5, 8, 6]
M = [10, 11, -11, -5]

# Dual Coxeter numbers (214 values for primal, from target_volumes.dat)
# c_i = 1 for D3-instantons (165 divisors)
# c_i = 6 for O7-planes with so(8) (49 divisors)

# Results
g_s = 0.00911134
W_0 = 2.30012e-90
V_string = 4711.83
V_0 = -5.5e-203
```

### Dual vs Primal Polytopes

**Key insight:** McAllister uses the PRIMAL polytope (h11=214) for Kähler moduli stabilization, but the flux superpotential W₀ is computed on the DUAL (h11=4).

- **Primal (h11=214):** 294 lattice points, 214 Kähler moduli, 4 complex structure moduli
- **Dual (h11=4):** 12 lattice points, 4 Kähler moduli, 214 complex structure moduli
- **Mirror symmetry:** The physics is equivalent; they're computing on both sides

For our GA, we can work with either:
- Working with h11=4 dual is computationally cheaper (smaller matrices)
- Working with h11=214 primal gives access to the full divisor structure

---

## Orientifold Involution: The Missing Genome Parameter

**Key discovery:** The orientifold involution determines which divisors host O7-planes (c_i = 6) vs D3-instantons (c_i = 1). This is a **model choice** that must be part of the GA genome.

### What We Can Compute From First Principles

1. **Divisor rigidity** - Computable combinatorially from the polytope (no cohomCalg needed)
   - See `mcallister_2107/compute_rigidity_combinatorial.py`
   - Validated 214/214 match with McAllister's data

2. **Which rigid divisors are O7-planes** - Requires the orientifold involution choice

### The Involution

From McAllister eq. (2.18), the involution negates a subset of homogeneous coordinates:
```
I : x_{I_α} → -x_{I_α},  α = 1, ..., k
```

Each negated coordinate creates an O7-plane on the divisor {x_i = 0}.

For 4-214-647: 49 coordinates are negated → 49 O7-planes with c_i = 6.

### Constraints on Valid Involutions

1. **h¹'¹₋(X) = h²'¹₊(X) = 0** - No geometric moduli projected out
2. **Tadpole cancellation** - D3-brane charge must balance

### GA Genome Update

```python
genome = {
    "polytope_id": int,           # Which polytope
    "triangulation_id": int,      # Which triangulation
    "K": [int] * h11,             # Flux vector K
    "M": [int] * h11,             # Flux vector M
    "orientifold_mask": [bool],   # NEW: Which coordinates to negate
}
```

See [docs/ORIENTIFOLD_INVOLUTION.md](ORIENTIFOLD_INVOLUTION.md) for full details.

---

## Validated: Random (K, M) Search Works

We validated the entire pipeline by running `mcallister_2107/search_km.py` - a random search over (K, M) pairs for McAllister's polytope.

**Search space:**
- K, M each have h11=4 integers in range [-15, 15]
- Total combinations: 31⁴ × 31⁴ = **853 billion** pairs
- We sampled: **10 million** (0.001% of space)

**Filter pass rates:**
- N invertible: 93.3%
- p positive (in Kähler cone): 11.8%
- Valid e^K0: 9.9%
- Has valid racetrack: 30.6% of valid

**Best result found:**
```python
K = [14, 5, 4, -14]
M = [-8, 7, 4, -9]

# Results (vs McAllister targets):
g_s  = 0.007554   # target: 0.00911 (17% error)
W_0  = 1.65e-90   # target: 2.30e-90 (0.1 log error - almost exact!)
V_0  = -7.27e-204 # target: -5.50e-203 (within 1 order of magnitude!)
```

**This proves:**
1. The racetrack mechanism produces exponentially small W_0
2. Random sampling finds valid vacua (~3% of samples)
3. We can reproduce McAllister-level cosmological constants (10^-203)
4. The pipeline works end-to-end with latest CYTools

---

## Key Insight: K, M Fluxes Replace Continuous Moduli

The original GA searched over continuous Kähler moduli, which was fundamentally flawed:

**Old (broken) approach:**
- Genome: `(polytope_id, kahler_moduli[], complex_moduli[], flux_f[], flux_h[], g_s)`
- Problem: We don't compute periods, so `flux_f`, `flux_h` → W₀ pipeline was fake
- Problem: Continuous moduli search has no physics guidance

**New (correct) approach:**
- Genome: `(polytope_id, triangulation_id, K[], M[])`
- K, M are integer vectors of length h11 (typically 4-10 integers each)
- All physics (g_s, W₀, V₀, Kähler moduli) is **deterministically computed** from (K, M)
- The Demirtas lemma gives: `p = N⁻¹K` where `N_ab = κ_abc M^c`
- p determines the flat direction → Kähler moduli are fixed, not searched

This is how McAllister actually found their vacua.

---

## Two-Level Architecture

### Outer GA: Evolves Polytope Selection Strategies

The outer GA evolves **how to select polytopes** and **how hard to search each one**.

**Genome (MetaAlgorithm):**
```rust
struct MetaAlgorithm {
    // Feature weights for polytope similarity
    feature_weights: HashMap<String, f64>,  // sphericity, chirality, entropy, etc.

    // Search strategy parameters
    similarity_radius: f64,
    interpolation_weight: f64,

    // Inner loop control
    samples_per_polytope: i32,  // How many (K, M) pairs to try per polytope

    // GA hyperparameters
    population_size: i32,
    mutation_rate: f64,
    // ... etc
}
```

**Fitness:** Rate of improvement in V₀ over time. We're looking for strategies that consistently find slightly better polytopes - extracting signal from noise.

**What it learns:**
- Which geometric features (h11, sphericity, chirality, etc.) correlate with good vacua
- How many (K, M) samples to try before moving to next polytope
- Optimal balance between exploration (new polytopes) and exploitation (more samples)

### Inner Loop: Constrained Random Sampling of (K, M)

For each polytope selected by the outer GA, the inner loop samples (K, M) pairs.

**This is NOT a GA.** It's constraint-guided random sampling with fast filtering.

```
for each polytope selected by outer GA:
    for i in 1..samples_per_polytope:
        K, M = random_integer_vectors(h11, range=[-10, 10])

        # Fast filters (microseconds each)
        if not check_N_invertible(κ, M): continue
        if not check_p_in_kahler_cone(κ, K, M): continue
        if not check_orthogonality(K, p): continue
        if not check_tadpole_bound(K, M): continue

        # Expensive evaluation (seconds to minutes)
        result = compute_racetrack_vacuum(polytope, K, M)

        if result.success:
            save_to_database(result)
```

---

## Why GA Doesn't Work for (K, M)

Consider the genome `(polytope_id, triangulation_id, K, M)`:

1. **polytope_id**: Discrete, 12M options. Can't meaningfully mutate or crossover.
2. **triangulation_id**: Discrete, finite per polytope. Can't meaningfully mutate.
3. **K, M**: Integer vectors. Mutation = random perturbation. Crossover = mixing integers.

The problem: **there's no gradient**. Changing K from `[-3, -5, 8, 6]` to `[-3, -5, 8, 7]` might flip the result from "valid vacuum with V₀ = 10⁻²⁰³" to "singular N matrix, no solution."

The fitness landscape is essentially random at the (K, M) level. GA would be no better than random search.

**However:** The fitness landscape at the *polytope* level may have structure. Certain geometric features might correlate with having more valid (K, M) pairs. The outer GA can learn this.

---

## The (K, M) Constraint Filters

From McAllister/Demirtas, valid (K, M) must satisfy:

### 1. N Matrix Invertibility
```python
N_ab = κ_abc * M^c  # Contract intersection numbers with M
if det(N) == 0: reject  # No flat direction exists
```
**Cost:** O(h11³) ~ microseconds

### 2. Flat Direction in Kähler Cone
```python
p = N_inverse @ K  # Solve N·p = K
if not all(p > 0): reject  # p must be in positive orthant
if not in_kahler_cone(p): reject  # p must satisfy cone inequalities
```
**Cost:** O(h11³) for solve + O(h11²) for cone check ~ microseconds

### 3. Orthogonality Constraint
```python
if abs(K @ p) > epsilon: reject  # K·p must vanish
```
**Cost:** O(h11) ~ nanoseconds

### 4. Tadpole Bound
```python
if -0.5 * (M @ K) > Q_D3: reject  # D3 charge constraint
```
**Cost:** O(h11) ~ nanoseconds

**Expected filter rate:** >99.9% of random (K, M) pairs fail these filters. Only ~0.1% proceed to expensive evaluation.

---

## Expensive Evaluation: Racetrack Computation

For (K, M) pairs that pass all filters:

### Step 1: Compute e^{K₀}
```python
p = solve(N, K)  # Already computed
kappa_p3 = einsum('abc,a,b,c->', kappa, p, p, p)
e_K0 = 1.0 / ((4.0/3.0) * kappa_p3)
```

### Step 2: Find Racetrack Curve Pairs
```python
# Load GV invariants for this polytope (precomputed)
gv_invariants = load_gv(polytope_id)

# Group curves by q·p (determines instanton action)
curves_by_action = group_curves(gv_invariants, p)

# Find pairs (q̃₁, q̃₂) with small action difference ε = p·(q̃₂ - q̃₁)
racetrack_pairs = find_racetrack_pairs(curves_by_action)
```

### Step 3: Solve F-term Equation
```python
# W_eff(τ) = ζ Σ_q (M·q) N_q Li₂(exp(2πiτ q·p))
# Solve ∂_τ W_eff = 0 for τ

tau = solve_f_term(M, p, gv_invariants, racetrack_pairs)
g_s = 1 / tau.imag
```

### Step 4: Compute W₀ and V₀
```python
W_0 = evaluate_superpotential(tau, M, p, gv_invariants)
V_0 = -3 * e_K0 * (g_s**7 / (4 * V_CY)**2) * W_0**2
```

**Cost:** Seconds to minutes depending on GV invariant computation.

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTER GA                                      │
│  Evolves: feature_weights, samples_per_polytope, search_params          │
│  Fitness: best V₀ found during run                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      POLYTOPE SELECTION                                 │
│  Use feature_weights to select next polytope from 12M candidates        │
│  Similarity search, clustering, interpolation based on evolved params   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    INNER LOOP (per polytope)                            │
│  for i in 1..samples_per_polytope:                                      │
│      K, M = random_integers(h11)                                        │
│      if passes_all_filters(K, M):                                       │
│          result = compute_racetrack(K, M)                               │
│          save(result)                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATABASE                                       │
│  All valid evaluations saved: (polytope, K, M, g_s, W₀, V₀, ...)        │
│  Outer GA sees: rate of improvement → which strategies find signal      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Outer GA Learns Search Efficiency

The key innovation: **samples_per_polytope is evolved, not fixed.**

- Algorithm A: samples_per_polytope = 100, explores many polytopes shallowly
- Algorithm B: samples_per_polytope = 10000, explores few polytopes deeply

Which is better? Depends on the fitness landscape structure. The outer GA learns this automatically.

If most polytopes have zero valid (K, M) pairs but a few have many → deep search is better.
If valid (K, M) pairs are spread evenly across polytopes → broad search is better.

The outer GA will discover the optimal balance through evolution.

---

## Implementation Phases

### Phase 1: Validate Pipeline
- [ ] Port McAllister reproduction to latest CYTools (basis transformation)
- [ ] Verify we can compute (g_s, W₀, V₀) from (polytope, K, M)
- [ ] Create unit test with known-good McAllister values

### Phase 2: Build Inner Loop
- [ ] Implement fast (K, M) filters
- [ ] Implement racetrack computation
- [ ] Benchmark: filters per second, evaluations per minute

### Phase 3: Integrate with Outer GA
- [ ] Add `samples_per_polytope` to MetaAlgorithm
- [ ] Replace old Compactification genome with new (K, M) approach
- [ ] Update fitness to use racetrack-computed V₀

### Phase 4: Scale Up
- [ ] Precompute GV invariants for all 12M polytopes (expensive, one-time)
- [ ] Parallelize inner loop across workers
- [ ] Run outer GA with real evaluations

---

## Comparison: Old vs New

| Aspect | Old GA | New GA v2 |
|--------|--------|-----------|
| Inner genome | Continuous (kahler_moduli, g_s, fluxes) | Discrete (K, M integers) |
| Inner search | GA with mutation/crossover | Constrained random sampling |
| Physics | Fake (no period computation) | Real (racetrack from GV invariants) |
| Moduli | Searched (wrong) | Computed from K, M (correct) |
| Outer GA | Evolves feature weights + GA params | Same + samples_per_polytope |
| Validation | None (garbage results) | McAllister reproduction |

---

## References

1. **McAllister et al. arXiv:2107.09064** - "Small cosmological constants in string theory"
   - Section 2.3: Flux vacua and Demirtas lemma
   - Section 5: Computational methods
   - Section 6.4: 4-214-647 example with V₀ = -5.5×10⁻²⁰³

2. **Demirtas et al. arXiv:1912.10047** - "Vacua with Small Flux Superpotential"
   - The lemma: p = N⁻¹K construction
   - Constraint structure for valid (K, M)

3. **CLAUDE.md** - Project documentation
   - KKLT moduli stabilization formulas
   - V₀ = -3 × e^{K₀} × (g_s⁷ / (4V[0])²) × W₀²
