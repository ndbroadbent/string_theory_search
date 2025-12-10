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

### 1.1 Calabi-Yau Volume
```
V = (1/6) κᵢⱼₖ tⁱ tʲ tᵏ
```
Where:
- κᵢⱼₖ = triple intersection numbers of divisors
- tⁱ = Kähler moduli (2-cycle volumes)

### 1.2 Divisor Volumes
```
τᵢ = (1/2) κᵢⱼₖ tʲ tᵏ
```

### 1.3 Hodge Numbers
- h¹¹ = number of Kähler moduli
- h²¹ = number of complex structure moduli
- χ = 2(h¹¹ - h²¹) = Euler characteristic

### 1.4 Frame Conversions
```
V_Einstein = V_string / g_s^(3/2)
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
