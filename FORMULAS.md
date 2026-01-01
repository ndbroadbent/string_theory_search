# String Theory Compactification Formulas

Complete reference of formulas needed to compute the cosmological constant from a Calabi-Yau compactification.

## The Goal

Compute the vacuum energy (cosmological constant):
```
V₀ = -3 eᴷ |W|²    (in Planck units, Mpl⁴)
```

This requires computing:
1. **K** - the Kähler potential
2. **W** - the superpotential (flux + non-perturbative)

---

## Part 1: Geometry (CYTools can compute these)

### 1.1 Classical Calabi-Yau Volume
```
V_classical = (1/6) κᵢⱼₖ tⁱ tʲ tᵏ
```
Where:
- κᵢⱼₖ = triple intersection numbers of divisors
- tⁱ = Kähler moduli (2-cycle volumes)

**WARNING:** This is the CLASSICAL volume. The physical volume includes instanton corrections (see Part 1A below).

### 1.2 Classical Divisor Volumes
```
τᵢ = (1/2) κᵢⱼₖ tʲ tᵏ
```

**WARNING:** This is the CLASSICAL divisor volume. See Part 1A for corrections.

### 1.3 Hodge Numbers
- h¹¹ = number of Kähler moduli
- h²¹ = number of complex structure moduli
- χ = 2(h¹¹ - h²¹) = Euler characteristic

### 1.4 Frame Conversions
```
V_Einstein = V_string / g_s^(3/2)
```

---

## Part 1A: Worldsheet Instanton Corrections (CRITICAL!)

**These corrections are ESSENTIAL for reproducing McAllister's results.**

The classical formulas (1.1, 1.2) give V ≈ 17900 for polytope 4-214-647.
The instanton-corrected formulas give V ≈ 4711 (the correct answer).
**This is a 3.8× difference - not a small correction!**

### 1A.1 Corrected CY Volume (McAllister eq. 4.11)
```
V[0] = (1/6) κᵢⱼₖ tⁱ tʲ tᵏ                    [Classical term]
       - ζ(3)χ(X) / (4(2π)³)                  [BBHL α' correction - CRITICAL!]
       + (instanton sum with GV invariants)   [Usually tiny, ~0.001]
```

**VERIFIED for polytope 4-214-647:**
```
V_classical (with corrected t):  4712.338
BBHL correction:                 -0.509  (= ζ(3)×420 / (4(2π)³))
Instanton sum:                   ~0.001
─────────────────────────────────────────
V[0] (computed):                 4711.831
V[0] (cy_vol.dat):               4711.830  ✓ EXACT MATCH
```

Where:
- κᵢⱼₖ = intersection numbers
- tⁱ = Kähler moduli (2-cycle volumes) - **USE corrected_kahler_param.dat!**
- χ(X) = Euler characteristic = 2(h¹¹ - h²¹)
- ζ(3) = 1.202056903... (Riemann zeta function)
- N_q = genus-zero Gopakumar-Vafa invariants
- q = curve class in Mori cone M(X)

### 1A.1a BBHL Correction (α' correction at string tree level)
```
BBHL = ζ(3)χ(X) / (4(2π)³)
```

For polytope 4-214-647 with h¹¹=214, h²¹=4:
- χ = 2(214 - 4) = 420
- BBHL = 1.202 × 420 / 992.2 = **0.508832**

**This correction is NOT optional!** Without it, V is wrong by ~0.5 (0.01% error).

### 1A.2 Corrected Divisor Volumes (McAllister eq. 4.12)
```
Tᵢ = (1/2) κᵢⱼₖ tʲ tᵏ
     - χ(Dᵢ)/24
     + (1/(2π)²) Σ_q qᵢ N_q Li₂((-1)^{γ·q} e^{-2πq·t})
```

### 1A.2a Divisor Euler Characteristic χ(D_i)

The χ(D_i) in eq. 4.12 is the **topological** Euler characteristic of the divisor:
```
χ(D) = 12 × χ(O_D) - D³
```
Where:
- χ(O_D) = h⁰ - h¹ + h² is the holomorphic Euler characteristic
- D³ = κ_DDD is the triple self-intersection

**For toric CY hypersurface divisors**, χ(O_D) is computed COMBINATORIALLY using
Braun et al. arXiv:1712.04946 eq (2.7):

| Point location (µ) | h^•(O_D)     | χ(O_D)  | g definition |
|--------------------|--------------|---------|--------------|
| Vertex (µ=0)       | (1, 0, g)    | 1 + g   | interior pts in dual facet |
| Edge interior (µ=1)| (1, g, 0)    | 1 - g   | interior pts in dual edge |
| 2-face interior    | (1+g, 0, 0)  | 1 + g   | g = 0 (dual is vertex) |

**Rigid divisors** have g = 0, so χ(O_D) = 1 and χ(D) = 12 - D³.

**Non-rigid divisors** (g > 0) can have much larger χ(D) values:
- Vertex with g=2, D³=-9: χ(D) = 12×3 - (-9) = 45

**Implementation:** `mcallister_2107/compute_chi_divisor.py`
**Validated:** 2.4% RMS error against McAllister (error is GV corrections only)

### 1A.3 The Iterative Solution Problem

Given KKLT target divisor volumes τᵢ = (cᵢ/2π) ln(W₀⁻¹), we need to solve:
```
Tᵢ(t) = τᵢ   (implicit equation for t)
```

This requires **iterative solution** because Tᵢ depends nonlinearly on t through both the classical term and the instanton sum.

### 1A.4 McAllister's Data Files

For polytope 4-214-647:

| File | Contains | V_classical result |
|------|----------|-------------------|
| `kahler_param.dat` | UNCORRECTED t (no instantons) | 17901 ❌ |
| `corrected_kahler_param.dat` | CORRECTED t (with instantons) | 4712.34 ✓ |

After applying BBHL correction to corrected result: **4711.83 ✓ EXACT**

Other data files:
- `heights.dat` = triangulation heights (MUST use for correct triangulation)
- `small_curves_gv.dat`, `dual_curves_gv.dat` = Gopakumar-Vafa invariants N_q
- `small_curves.dat`, `dual_curves.dat` = curve classes q

### 1A.5 Complete V_string Computation Recipe

```python
# 1. Load corrected Kähler moduli (NOT kahler_param.dat!)
t = load("corrected_kahler_param.dat")

# 2. Use correct triangulation
tri = poly.triangulate(heights=load("heights.dat"))

# 3. Compute classical volume
V_classical = (1/6) * einsum('ijk,i,j,k', kappa, t, t, t)

# 4. Apply BBHL correction
chi = 2 * (h11 - h21)
BBHL = zeta(3) * chi / (4 * (2*pi)**3)
V_string = V_classical - BBHL

# Result: V_string ≈ 4711.83 ✓
```

---

## Part 2: Kähler Potential

### 2.1 Full Kähler Potential (McAllister eq. 2.13)
```
K = K_Kähler + K_dilaton + K_complex_structure

K = -2 ln(√2 V_E) - ln(-i(τ - τ̄)) - ln(-i ∫_X Ω ∧ Ω̄)
```
Where:
- V_E = Einstein frame volume
- τ = axio-dilaton = C₀ + i/g_s
- Ω = holomorphic 3-form
- ∫Ω∧Ω̄ = **REQUIRES PERIODS**

### 2.2 Simplified Form at Large Volume
At large volume and large complex structure:
```
eᴷ ≈ eᴷ⁰ × g_s / (2 V²)
```
Where eᴷ⁰ depends on complex structure moduli.

### 2.3 Complex Structure Kähler Potential
```
K_cs = -ln(-i ∫_X Ω ∧ Ω̄) = -ln(Π† · Σ · Π)
```
Where:
- Π = period vector
- Σ = symplectic matrix

### 2.4 eᴷ⁰ Formula (McAllister eq. 6.12)
```
eᴷ⁰ = (4/3) × (κ̃_abc p^a p^b p^c)⁻¹
```
Where:
- κ̃_abc = mirror (dual) intersection numbers
- p = direction in complex structure moduli space (perturbatively flat direction)

---

## Part 3: Superpotential

### 3.1 Total Superpotential (KKLT form, eq. 1.1)
```
W = W_flux + W_np

W = W₀ + Σᵢ Aᵢ exp(-2π Tᵢ / cᵢ)
```
Where:
- W₀ = flux superpotential (from fluxes + periods)
- Aᵢ = Pfaffian prefactors (one-loop determinants)
- Tᵢ = holomorphic Kähler moduli
- cᵢ = dual Coxeter numbers

### 3.2 Gukov-Vafa-Witten Flux Superpotential
```
W_flux = ∫_X G₃ ∧ Ω = (F - τH) · Π
```
Where:
- G₃ = F₃ - τ H₃ (complexified 3-form flux)
- F₃ = RR 3-form flux (integer quantized)
- H₃ = NS-NS 3-form flux (integer quantized)
- τ = axio-dilaton
- Π = period vector
- Ω = holomorphic 3-form

**THIS IS THE KEY FORMULA - requires periods!**

### 3.3 Period Vector
```
Π = ( ∫_{A^I} Ω,  ∫_{B_I} Ω )ᵀ = (z^I, F_I)ᵀ
```
Where:
- A^I, B_I = symplectic basis of 3-cycles
- z^I = complex structure coordinates
- F_I = ∂F/∂z^I (derivatives of prepotential)
- I = 0, 1, ..., h²¹

### 3.4 Prepotential
```
F(z) = F_poly(z) + F_inst(z)
```

Polynomial part (classical):
```
F_poly = -(1/6) κ̃_abc z^a z^b z^c + (1/2) a_ab z^a z^b + b_a z^a + (c/2) + ...
```
Where:
- κ̃_abc = classical triple intersection (on mirror)
- a_ab, b_a, c = topological data

Instanton part:
```
F_inst = Σ_q N_q Li₃(e^(2πi q·z))
```
Where:
- N_q = genus-0 Gopakumar-Vafa invariants
- q = curve class in H₂(X̃, Z)
- Li₃ = polylogarithm

### 3.5 Gopakumar-Vafa Invariants
These count BPS states / holomorphic curves. Computed via:
1. Mirror symmetry (from prepotential expansion)
2. Localization on moduli space of stable maps
3. Topological string partition function

---

## Part 4: Moduli Stabilization

### 4.1 F-flatness Conditions (eq. 5.1)
At supersymmetric minimum, Dᵢ W = 0:
```
Dᵢ W = ∂ᵢ W + (∂ᵢ K) W = 0
```

### 4.2 Kähler Moduli at Minimum (eq. 5.7)
```
Re(Tᵢ) ≈ (cᵢ / 2π) × ln(W₀⁻¹)
```
Small W₀ → large volumes → small V₀.

### 4.3 String Coupling
```
g_s = 2π / (c_τ × ln(W₀⁻¹))
```
Where c_τ is model-dependent (eq. 2.29).

---

## Part 5: Vacuum Energy

### 5.1 AdS Vacuum Energy (eq. 6.24, 6.63)
```
V₀ = -3 eᴷ |W|²
```

At the KKLT minimum with |W| ≈ |W₀|:
```
V₀ = -3 × eᴷ⁰ × (g_s⁷ / (4V[0])²) × W₀²
```

### 5.2 McAllister 4-214-647 Values
- W₀ = 2.3 × 10⁻⁹⁰
- g_s = 0.00911
- V[0] = 4711 (string frame)
- eᴷ⁰ ≈ 0.236 (back-calculated)
- **V₀ = -5.5 × 10⁻²⁰³ Mpl⁴**

### 5.3 Uplift (for de Sitter)
To get positive CC, add anti-D3 branes:
```
V_uplift = D / V^(4/3)
```
Where D depends on warped throat geometry.

---

## Part 6: Tadpole Constraint

### 6.1 D3-brane Tadpole
```
N_flux + N_D3 + N_O3 = χ(X) / 24
```
Where:
```
N_flux = (1/2) ∫_X F₃ ∧ H₃
```

---

## WHAT WE CAN COMPUTE (CYTools)

| Quantity | Method | Status |
|----------|--------|--------|
| Intersection numbers κᵢⱼₖ | `cy.intersection_numbers()` | ✓ Working |
| Hodge numbers h¹¹, h²¹ | `cy.h11()`, `cy.h21()` | ✓ Working |
| Euler characteristic | `cy.chi()` | ✓ Working |
| CY volume | `cy.compute_cy_volume(t)` | ✓ Working |
| Divisor volumes | `cy.compute_divisor_volumes(t)` | ✓ Working |
| Kähler cone | `cy.kahler_cone()` | ✓ Working |

---

## WHAT WE CAN COMPUTE (Now Working!)

| Quantity | Method | Status |
|----------|--------|--------|
| **GV invariants N_q** | `cy.compute_gvs()` via cygv | ✓ Validated |
| **χ(D_i)** | Braun eq (2.7) combinatorially | ✓ Validated (2.4% = GV only) |
| **Divisor rigidity** | Dual face interior points | ✓ Working |

## WHAT WE CANNOT COMPUTE (Still Missing)

| Quantity | Why Missing | Impact |
|----------|-------------|--------|
| **Periods Π** | Requires solving Picard-Fuchs equations | W₀ is garbage |
| **Prepotential F(z)** | Requires periods | Can't compute periods |
| **eᴷ⁰** | Requires periods for ∫Ω∧Ω̄ | Full K is wrong |
| **Pfaffians Aᵢ** | One-loop determinants on divisors | W_np prefactors unknown |

---

## HOW TO COMPUTE PERIODS

### Method 1: Picard-Fuchs Equations

The periods satisfy a system of differential equations:
```
L Π(z) = 0
```
Where L is the Picard-Fuchs operator.

For a CY hypersurface in weighted projective space:
```
L = Π (θ - αᵢ) - z Π (θ + βⱼ)
```
Where θ = z d/dz and αᵢ, βⱼ come from the weights.

**Algorithm:**
1. Construct Picard-Fuchs operator from CY data
2. Find Frobenius basis of solutions at large complex structure (z → 0)
3. Analytic continuation to point of interest
4. Extract period vector Π(z)

### Method 2: Direct Integration (Numerical)

Use numerical integration:
```
∫_γ Ω
```
Over a basis of 3-cycles γ.

**Problem:** Finding explicit cycles is hard for general CY.

### Method 3: Mirror Symmetry

If X̃ is the mirror of X:
- Periods of X at point z ↔ volumes on X̃ at mirror point
- Use GLSM / toric methods to compute

---

## Part 7: Toric Geometry & GLSM

### 7.1 Fayet-Iliopoulos Parameters = Kähler Moduli

From Bouchard arXiv:0901.3695 eq. (108):
```
∫_C^a ω = t^a
```

Where:
- C^a = basis of resolving 2-cycles (a = 1, ..., h¹¹)
- ω = Kähler form
- t^a = Fayet-Iliopoulos (FI) parameters in the GLSM

**Key insight:** "So the FI parameters in the GLSM really map to the 'Kähler volumes' of the resolving cycles" (eq. 107-108)

This connects:
- **GLSM** (physics): FI parameters t^a appear in D-term constraints
- **Geometry**: t^a are volumes of 2-cycles = Kähler moduli

### 7.2 Secondary Fan and Triangulations

The **secondary fan** is the fan over the set of all regular triangulations of a polytope.

- Each **cone** in the secondary fan corresponds to a triangulation
- **Heights** h ∈ ℝ^n parametrize the secondary fan (one height per lattice point)
- A triangulation is **regular** iff it's induced by some height function

**Triangulation from heights:**
```python
# Heights lift points to (n+1) dimensions, triangulation = lower convex hull
tri = poly.triangulate(heights=h)
```

### 7.3 Kähler Cone as Projection of Secondary Cone

From CYTools paper arXiv:2211.03823 §5.4.1:

> "The Kähler cone can be thought of as a **projection** of the secondary cone that removes the linear subspaces."

More precisely:
- Secondary cone ⊂ ℝ^n (n = number of lattice points)
- Kähler cone ⊂ ℝ^h¹¹ (h¹¹ ≤ n - 4 for 4d polytopes)
- Projection removes: origin point + linear dependencies from GLSM relations

### 7.4 GLSM Charge Matrix

The GLSM charge matrix Q relates divisors to points:
```
Q : ℝ^{n_pts} → ℝ^{h¹¹}
```

Properties:
- Shape: (h¹¹, n_pts)
- Rows span kernel of point matrix M
- Encodes D-term constraints: Σᵢ Qᵢₐ |φᵢ|² = t^a

**NOT a simple projection:** Q @ h does not directly give t because:
1. Heights live in extended space (include origin, non-basis points)
2. Linear subspaces must be quotiented out

### 7.5 The Unsolved Problem: Heights → Kähler Moduli

**Given:** Heights h ∈ ℝ^n defining a triangulation
**Want:** Kähler moduli t ∈ ℝ^h¹¹ inside the Kähler cone

McAllister's algorithm (Section 5 of arXiv:2107.09064):
1. Pick random h_init in secondary fan
2. This gives triangulation T and "associated" t_init
3. Solve KKLT iteratively: t_init → τ_init → τ_target → t_final

**The missing piece:** Step 2 says heights are "naturally associated" to t but doesn't specify the map.

For h¹¹ ≤ 12, CYTools can compute `tip_of_stretched_cone()` to get a starting point.
For h¹¹ = 214, this is computationally intractable.

### 7.6 McAllister's Path-Following Algorithm (Section 5.2)

**THIS IS THE CORRECT ALGORITHM - USE THIS, NOT SCIPY OPTIMIZATION**

The problem: Find t such that τ(t) = τ_target where τ_i = (1/2) κ_ijk t^j t^k.

**Key insight from McAllister Section 5.2:**
> "Following the path is then reduced to moving from t_m to t_{m+1}."

At each step, we solve a **LINEAR system** (not nonlinear optimization):

```
κᵢⱼₖ tʲ εᵏ = Δτᵢ   (linear in ε!)
```

Where:
- ε = t_{m+1} - t_m (small step)
- Δτ = τ_{m+1} - τ_m (target change)

**The Algorithm:**
```python
def solve_kklt_path_following(kappa, tau_target, n_steps=200):
    # 1. Start with uniform t
    t = np.ones(h11) * 0.5
    tau_init = compute_tau(kappa, t)

    # 2. Scale t so initial τ is right magnitude
    scale = sqrt(mean(tau_target) / mean(tau_init))
    t = t * scale
    tau_init = compute_tau(kappa, t)

    # 3. Interpolate from τ_init to τ_target in N steps
    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_step = (1 - alpha) * tau_init + alpha * tau_target

        # 4. At each step, solve LINEAR system for ε
        tau_current = compute_tau(kappa, t)
        delta_tau = tau_step - tau_current

        # J_ik = κ_ijk t^j  (Jacobian, computed from sparse kappa)
        J = compute_jacobian(kappa, t)

        # Solve: J @ ε = delta_tau  (LINEAR!)
        epsilon = np.linalg.lstsq(J, delta_tau, rcond=1e-8)[0]

        # 5. Update t
        t = t + epsilon

    return t
```

**Why this is fast:**
- Each step is ONE matrix solve: O(h11³) ≈ 10M ops for h11=214
- Total: O(n_steps × h11³) ≈ 2 billion ops
- With sparse kappa: much faster since κ has only ~2000 nonzero entries

**Why nonlinear optimization (scipy) is WRONG:**
- Treats this as unconstrained optimization over h11=214 variables
- Each iteration evaluates residual + Jacobian many times
- No path structure → can diverge wildly
- Orders of magnitude slower

### 7.7 Why Full Nonlinear Newton Fails

**For h¹¹=214:** The Jacobian J_ik = κ_ijk t^j has rank ~65 (not 214)
- 149-dimensional nullspace
- Infinitely many solutions t for given τ
- Full Newton on the nonlinear system diverges without path structure

**McAllister's path-following works because:**
- It interpolates smoothly from τ_init to τ_target
- Each LINEAR step stays near the previous solution
- The path stays inside the Kähler cone throughout

### 7.8 TWO-PHASE KKLT ALGORITHM (CRITICAL!)

**This is the key to computing t from scratch without cheating.**

McAllister's data provides TWO files for Kähler parameters:
- `kahler_param.dat`: Solution to **τ = c_i** (simpler equation, "uncorrected")
- `corrected_kahler_param.dat`: Solution to **τ = c_i/c_τ + χ/24** (full KKLT, "corrected")

**The algorithm has TWO phases:**

**PHASE 1: Solve τ = c_i (uncorrected KKLT)**
```
Find t such that: (1/2) κ_ijk t^j t^k = c_i
```
- c_i are the dual Coxeter numbers (1 for D3-instanton, 6 for O7-plane)
- This equation is INDEPENDENT of g_s, W₀ (no flux dependence!)
- This is what McAllister calls finding "a point in the extended Kähler cone"
- McAllister describes this in Section 5.2 of arXiv:2107.09064

**PHASE 2: Path-follow to target τ**
```
Path-follow from t_uncorrected → t_corrected where:
τ_target = c_i/c_τ + χ(D_i)/24  (with instanton corrections via eq. 4.12)
```

**Why two phases?**
- The equation τ(t) = τ_target has MULTIPLE solution branches
- Different branches give different V_string values (some positive, some negative)
- Starting from random t and solving directly often lands on WRONG branch (V < 0)
- Phase 1 finds the CORRECT branch (the one where t_uncorrected lives)
- Phase 2 stays on that branch while moving to the target τ

**Verified for 4-214-647:**
```
Phase 1: t_uncorrected satisfies τ = c_i (ratio = 1.0000)
         V_classical at t_uncorrected = 17901

Phase 2: Path-follow t_uncorrected → τ_target
         V_classical at t_corrected = 4712 ✓ (matches expected)
```

**The Phase 1 initialization (McAllister Section 5.2):**
> "We start by picking a random point h_init in the subset of the secondary fan of FRSTs...
> Such a point is naturally associated to a point in the extended Kähler cone, t_init"

**CRITICAL: Why random t fails but random heights works:**

The paper says "random point h_init in the secondary fan" - this means random HEIGHTS,
not random t! Heights parametrize valid triangulations (points in the Kähler cone).
Random t may not correspond to ANY valid triangulation.

In CYTools terms:
1. `poly.random_triangulations_fast(N=k)` - generates k random FRSTs with valid heights
2. Heights → t conversion: The triangulation heights encode a point in the Kähler cone
3. **Problem**: CYTools 2021 doesn't have `heights_to_kahler()` function

**What we tested (December 2024):**
- Random heights (wrong size 294): Error - heights are 219-dim, not num_points
- Random heights (correct size 219): "Triangulation is non-fine or non-star"
- `random_triangulations_fast()`: Works! But gives DIFFERENT κ per triangulation
- Random t with fixed triangulation: Mostly doesn't converge, or gives wrong V

**The key insight:**
McAllister uses the SAME triangulation throughout (from heights.dat), and their
t_init comes from that triangulation. We're using their triangulation but random t,
which doesn't work because:
- Random t is outside the Kähler cone for that triangulation
- Path-following from outside the cone diverges or finds wrong branch

**Possible solutions:**
1. Understand heights → t mapping (how CYTools does it internally)
2. Multi-start with V > 0 filter (brute force, slow but works)
3. Use Kähler cone generators as starting directions

Then path-follow from t_init to τ = c_i using the linear system:
```
κ_ijk t^j ε^k = Δτ_i
```

**Implementation in compute_kahler_param.py:**
```python
def solve_kklt_path_following(kappa, tau_target, c_i, n_steps=200):
    # PHASE 1: Solve τ = c_i (finds correct branch)
    t_uncorrected, _, _ = _path_follow(kappa, c_i.astype(float), t_init, n_steps)

    # PHASE 2: Path-follow to target τ (stays on correct branch)
    t_corrected, tau_achieved, converged = _path_follow(kappa, tau_target, t_uncorrected, n_steps)

    return t_corrected, tau_achieved, converged
```

### 7.9 Reference: Witten's "Phases of N=2"

Witten, "Phases of N=2 theories in two dimensions," Nucl. Phys. B 403 (1993) 159

Key concepts:
- Different **phases** of the GLSM correspond to different triangulations
- Phase boundaries are walls in FI parameter space
- Each phase has a geometric interpretation (orbifold, flop, etc.)

### 7.8 Further Reading

See **docs/TORIC_GEOMETRY.md** for comprehensive documentation on:
- Heights and triangulations
- Secondary fan structure
- GLSM charge matrix
- The projection from heights to Kähler moduli
- Open problems

---

## REFERENCES

### Primary
- McAllister et al., arXiv:2107.09064 - "Small cosmological constants in string theory"
- KKLT, hep-th/0301240 - Original moduli stabilization
- Gukov-Vafa-Witten, hep-th/9906070 - Flux superpotential

### Period Computation
- [Hosono-Klemm-Theisen-Yau](https://arxiv.org/abs/hep-th/9308122) - Mirror symmetry and periods
- [Picard-Fuchs for CY](https://arxiv.org/abs/0910.4215) - Picard-Fuchs equations
- [Computational Mirror Symmetry](https://arxiv.org/abs/2303.00757) - Recent algorithms

### Toric Geometry & GLSM
- [Witten - Phases of N=2](https://arxiv.org/abs/hep-th/9301042) - GLSM, FI parameters, phase structure
- [Bouchard - Toric intro](https://arxiv.org/abs/0901.3695) - FI params = Kähler volumes (eq 108)
- [CYTools paper](https://arxiv.org/abs/2211.03823) - Kähler cone as projection of secondary cone
- [MacFadden - Efficient CY algorithm](https://arxiv.org/abs/2309.10855) - Secondary cones and heights
- [Segal - GKZ guide](https://arxiv.org/abs/2412.14748) - GKZ theory foundations
- [Bouchard - CY toric](https://arxiv.org/abs/hep-th/0702063) - Symplectic quotients, GLSM

### Tools
- Klemm-Kreuzer "Instanton" code: http://hep.itp.tuwien.ac.at/~kreuzer/CY/
- CYTools: Polytopes, intersection numbers, volumes
- cymyc: Numerical CY metrics, Yukawa couplings

---

## Part 8: Quintessence & Time-Varying Dark Energy (NEW - DESI 2024/2025)

**CRITICAL UPDATE:** Recent DESI results suggest dark energy may NOT be a cosmological constant.
See `research/DESI_DARK_ENERGY_IMPLICATIONS.md` for full analysis.

### 8.1 The DESI Result

DESI 2024/2025 BAO data combined with CMB and supernovae suggest:
- **w₀ ≈ -0.7 to -0.8** (not -1)
- **wₐ ≈ -0.8 to -1.3** (not 0)
- ~2.5-3σ preference for dynamical dark energy

The CPL (Chevallier-Polarski-Linder) parameterization:
```
w(a) = w₀ + wₐ(1 - a)
```
Where a = scale factor (a=1 today).

For cosmological constant: w₀ = -1, wₐ = 0 (constant w = -1).

### 8.2 Quintessence Basics

For a canonical scalar field φ with potential V(φ):
```
w_φ = (φ̇²/2 - V) / (φ̇²/2 + V)
```

For slow-roll (φ̇² << V):
```
w ≈ -1 + (2/3)ε
```
Where ε = (M_pl²/2)(V'/V)² is the slow-roll parameter.

**Key constraint:** For canonical kinetic terms, -1 ≤ w ≤ 1 (no phantom!).

### 8.3 The Phantom Crossing Problem

DESI data suggests w crossed -1 from below (phantom) to above (quintessence).
But canonical quintessence CANNOT cross w = -1.

**Two explanations:**
1. **True phantom:** Requires exotic physics (ghost instabilities)
2. **Apparent phantom:** Multi-field effects create illusion of phantom crossing

### 8.4 The KMIX Model (String-Motivated Solution!)

From arXiv:2511.23463 (MIT, Nov 2025):

**Kinetically Mixed Axion-Dilaton Quintessence:**
```
L = -½(∂φ)² - ½e^(αφ)(∂θ)² - V(φ,θ)
```

Where:
- φ = dilaton-like modulus field
- θ = axion-like field (periodic potential)
- α = kinetic coupling (from string Kähler potential)

**Key insight:** The exponential kinetic coupling e^(αφ) is EXACTLY what appears
in string compactifications from the Kähler potential!

**Result:** KMIX can appear to have w < -1 in CPL fits while remaining completely
stable (no ghosts). The "phantom crossing" is an artifact of the parameterization.

### 8.5 Quintessence from String Moduli

From Cicoli et al. arXiv:2112.10779:

**NO-GO THEOREM:** Slow-roll quintessence is NOT possible in any parametrically
controlled regime of moduli space (large volume, weak coupling limits).

**Implication:** Must work in the INTERIOR of moduli space where numerical
(but not parametric) control is possible.

### 8.6 Connecting to Our CY Search

**Old target (static Λ):**
```
V₀ = -3 eᴷ |W|² ≈ 2.888 × 10⁻¹²² M_pl⁴
```

**New target (dynamical w):**
Instead of matching a single number Λ, we need configurations where:

1. **Complex structure moduli:** Fully stabilized (as before)
2. **Most Kähler moduli:** Stabilized (as in KKLT)
3. **One or more moduli/axions:** Rolling slowly → quintessence
4. **The rolling gives:** Correct (w₀, wₐ) values

### 8.7 Computing w from a Rolling Modulus

For a Kähler modulus T with potential V(T):
```
w₀ = -1 + (2/3) × (∂V/∂T / V)² |_{T=T₀}

wₐ = (rate of change of slow-roll parameter ε)
```

The full computation requires solving the cosmological evolution:
```
T̈ + 3HṪ + ∂V/∂T = 0
H² = (Ṫ²/2 + V) / (3M_pl²)
```

### 8.8 What Changes in the Pipeline

**Current fitness function:**
```python
fitness = Σ (physics_target - physics_computed)² / σ²
# with targets: α_em, α_s, sin²θ_W, N_gen, Λ
```

**Updated fitness function (for DESI):**
```python
fitness = Σ (physics_target - physics_computed)² / σ²
# with targets: α_em, α_s, sin²θ_W, N_gen, w₀, wₐ
# plus: stability_check (other moduli stabilized)
```

**New observables to compute:**
- w₀ from potential shape at "today"
- wₐ from potential curvature / field velocity
- Stability eigenvalues for non-rolling moduli

### 8.9 DESI Target Values

From DESI DR2 + Planck + SNe (varies by dataset combination):
```
w₀ = -0.7 ± 0.15  (approximate, depends on analysis)
wₐ = -1.0 ± 0.4   (approximate, depends on analysis)
```

Compare to Λ:
```
w₀ = -1.0
wₐ = 0.0
```

### 8.10 Key Papers

**Quintessence in String Theory:**
- arXiv:2112.10779 - Cicoli et al., "Quintessence and the Swampland" (no-go in asymptotic regions)
- arXiv:2206.10649 - Brinkmann et al., "Stringy Quintessence Constructions"
- arXiv:1808.02877 - Heisenberg et al., DESI predictions and swampland

**DESI Results & KMIX:**
- arXiv:2511.23463 - Toomey et al., "KMIX: Kinetically Mixed Axion-Dilaton Quintessence"
- DESI Collaboration DR1/DR2 cosmological results (2024-2025)

---

## BOTTOM LINE

**To compute V₀ from first principles, we need periods.**

Periods require either:
1. Solving Picard-Fuchs equations (analytical, works for simple cases)
2. Numerical integration over cycles (hard for general CY)
3. Mirror symmetry + GV invariants (what McAllister uses)

Without periods, our W₀ is fake → our V₀ is fake → our entire GA is optimizing garbage.

**Next step:** Find or implement Picard-Fuchs solver for toric CY hypersurfaces.
