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

## WHAT WE CANNOT COMPUTE (Missing!)

| Quantity | Why Missing | Impact |
|----------|-------------|--------|
| **Periods Π** | Requires solving Picard-Fuchs equations | W₀ is garbage |
| **Prepotential F(z)** | Requires periods + GV invariants | Can't compute periods |
| **GV invariants N_q** | Need special algorithms at high h¹¹ | Can't compute F_inst |
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

## REFERENCES

### Primary
- McAllister et al., arXiv:2107.09064 - "Small cosmological constants in string theory"
- KKLT, hep-th/0301240 - Original moduli stabilization
- Gukov-Vafa-Witten, hep-th/9906070 - Flux superpotential

### Period Computation
- [Hosono-Klemm-Theisen-Yau](https://arxiv.org/abs/hep-th/9308122) - Mirror symmetry and periods
- [Picard-Fuchs for CY](https://arxiv.org/abs/0910.4215) - Picard-Fuchs equations
- [Computational Mirror Symmetry](https://arxiv.org/abs/2303.00757) - Recent algorithms

### Tools
- Klemm-Kreuzer "Instanton" code: http://hep.itp.tuwien.ac.at/~kreuzer/CY/
- CYTools: Polytopes, intersection numbers, volumes
- cymyc: Numerical CY metrics, Yukawa couplings

---

## BOTTOM LINE

**To compute V₀ from first principles, we need periods.**

Periods require either:
1. Solving Picard-Fuchs equations (analytical, works for simple cases)
2. Numerical integration over cycles (hard for general CY)
3. Mirror symmetry + GV invariants (what McAllister uses)

Without periods, our W₀ is fake → our V₀ is fake → our entire GA is optimizing garbage.

**Next step:** Find or implement Picard-Fuchs solver for toric CY hypersurfaces.
