# Toric Geometry Reference

This document collects key insights about toric geometry as it relates to Calabi-Yau compactifications and the KKLT moduli stabilization problem.

## Table of Contents
1. [Core Objects and Their Relationships](#core-objects-and-their-relationships)
2. [Heights and Triangulations](#heights-and-triangulations)
3. [Secondary Fan and Secondary Cone](#secondary-fan-and-secondary-cone)
4. [GLSM and FI Parameters](#glsm-and-fi-parameters)
5. [The Projection: Heights → Kähler Moduli](#the-projection-heights--kähler-moduli)
6. [The Unsolved Problem](#the-unsolved-problem)
7. [Key Equations](#key-equations)
8. [References](#references)

---

## Core Objects and Their Relationships

### Dimensional Summary

For a reflexive polytope with n lattice points and h¹¹ Kähler moduli:

| Object | Dimension | Lives In |
|--------|-----------|----------|
| Heights h | n | ℝⁿ (height space) |
| Secondary cone | n | ℝⁿ |
| Raw Mori cone | n | ℝⁿ |
| GLSM charge matrix Q | h¹¹ × n | - |
| Kähler moduli t | h¹¹ | ℝ^h¹¹ |
| Kähler cone | h¹¹ | ℝ^h¹¹ |

### Key Relationship Chain

```
Heights h ∈ ℝⁿ
    ↓ (lifting construction)
Regular Triangulation T
    ↓ (secondary cone)
Secondary Cone ⊂ ℝⁿ
    ↓ (cone duality)
Raw Mori Cone ⊂ ℝⁿ
    ↓ (projection to divisor basis)
Mori Cone ⊂ ℝ^h¹¹
    ↓ (cone duality)
Kähler Cone ⊂ ℝ^h¹¹
    ↓ (pick point inside)
Kähler Moduli t ∈ ℝ^h¹¹
```

---

## Heights and Triangulations

### Definition: Regular Triangulation

A triangulation of a point set A = {p₁, ..., pₙ} ⊂ ℝᵈ is **regular** if it can be obtained by:

1. **Lifting**: Assign heights hᵢ to each point: p̃ᵢ = (pᵢ, hᵢ) ∈ ℝᵈ⁺¹
2. **Convex hull**: Compute conv({p̃₁, ..., p̃ₙ})
3. **Project lower faces**: The triangulation consists of projections of "lower faces" (faces whose outward normal has negative last component)

Reference: arXiv:2309.10855 §3.1

### Height Vector

The **height vector** h = (h₁, ..., hₙ) ∈ ℝⁿ parametrizes the lift. Different height vectors can produce the same triangulation.

### CYTools Usage

```python
# Create triangulation from heights
tri = poly.triangulate(heights=h)

# Get heights that define this triangulation
h = tri.heights()
```

---

## Secondary Fan and Secondary Cone

### Definition: Secondary Cone

The **secondary cone** of a triangulation T is the set of all height vectors that produce T:

```
SecCone(T) = { h ∈ ℝⁿ : h generates triangulation T }
```

This forms the interior of a polyhedral cone. The cone is defined by linear inequalities (hyperplanes).

### Key Properties

1. **Interior points** of SecCone(T) all generate triangulation T
2. **Boundary points** (on walls) may generate different triangulations
3. **Regularity**: T is regular ⟺ SecCone(T) is full-dimensional (solid)

### Secondary Fan

The **secondary fan** is the complete fan whose cones are all the secondary cones of all regular triangulations. It tiles ℝⁿ (modulo some linear subspace).

### Algorithm: Computing Secondary Cone

For a 2D triangulation with adjacent simplices {pₙ₁, pₛ₁, pₛ₂} and {pₙ₂, pₛ₁, pₛ₂}:

The secondary cone inequality is:
```
hₛ₁ + hₛ₂ ≤ hₙ₁ + hₙ₂   (or opposite sign depending on orientation)
```

This generalizes to higher dimensions via null-space calculations.

Reference: arXiv:2309.10855 Algorithm 1

---

## GLSM and FI Parameters

### The Gauged Linear Sigma Model (GLSM)

A toric variety can be realized as a **symplectic quotient**:

```
M = μ⁻¹(Re(t)) / G
```

Where:
- μₐ : ℂᵏ → ℂ are moment maps: Σᵢ Qᵢₐ |zᵢ|² = Re(tₐ)
- G = U(1)^(k-3) acts as: zⱼ → exp(i Qⱼₐ αₐ) zⱼ
- Qᵢₐ is the **GLSM charge matrix**
- tₐ are the **complexified Kähler parameters**

Reference: arXiv:hep-th/0702063 §4.3

### Key Insight: FI Parameters = Kähler Moduli

From Bouchard arXiv:0901.3695 eq. (108):

```
∫_{C^a} ω = t^a
```

Where:
- C^a = basis of resolving 2-cycles
- ω = Kähler form
- t^a = Fayet-Iliopoulos (FI) parameters

**"The FI parameters in the GLSM really map to the 'Kähler volumes' of the resolving cycles."**

### GLSM Charge Matrix

The charge matrix Q has shape (h¹¹, n_pts) where:
- Rows span the kernel of the point matrix
- Encodes D-term constraints
- Satisfies Calabi-Yau condition: Σᵢ Qᵢₐ = 0 for all a

---

## The Projection: Heights → Kähler Moduli

### The Core Problem

Heights h ∈ ℝⁿ live in a much higher-dimensional space than Kähler moduli t ∈ ℝ^h¹¹.

For the McAllister polytope 4-214-647:
- n = 219 lattice points
- h¹¹ = 214 Kähler moduli

### How CYTools Does It

1. **Secondary cone** lives in ℝⁿ (height space)
2. **Raw Mori cone** = dual(secondary cone) - also in ℝⁿ
3. **Projection**: Pick columns corresponding to divisor basis
4. **Kähler cone** = dual(projected Mori cone) - in ℝ^h¹¹

```python
# In CYTools (toricvariety.py line 816):
mori_raw = self.triangulation().secondary_cone().dual()

# Then project to divisor basis (line 831):
new_rays = rays[:, basis]  # Pick columns for basis divisors
```

### The Projection Matrix

The projection from ℝⁿ → ℝ^h¹¹ uses the **divisor basis**:
- Divisor basis = indices of h¹¹ divisors that form a basis
- Projection = selecting those columns from the raw cone rays

### What This Means

**The Kähler cone is a projection of the secondary cone** (after duality).

From CYTools paper arXiv:2211.03823 §5.4.1:
> "The Kähler cone can be thought of as a projection of the secondary cone that removes the linear subspaces."

---

## The Height → Kähler Mapping (SOLVED!)

### CYTools Implementation

CYTools provides the mapping in `cytools/utils.py`:

```python
from cytools.utils import heights_to_kahler, project_heights_to_kahler, kahler_to_heights
```

### Algorithm: `project_heights_to_kahler`

Given heights h ∈ ℝⁿ⁺¹ (including origin), compute Kähler parameters:

```python
def project_heights_to_kahler(poly, heights_in, prime_divisors=None):
    # 1. Get GLSM basis indices
    basis = [i-1 for i in poly.glsm_basis(include_origin=True)]

    # 2. Get effective cone rays for non-basis divisors
    if prime_divisors is None:
        prime_divisors = [r for i,r in enumerate(effective_cone.rays()) if i not in basis]

    # 3. Subtract origin height from all others
    origin_height = heights_in[0]
    kahler_parameters = heights_in[1:] - origin_height

    # 4. Apply corrections using linear relations from effective cone
    for e, ee in enumerate(prime_divisors):
        prime_ind = extra_divs[e]
        prime_height = kahler_parameters[prime_ind]
        # Linear relation: basis coefficients = ee, prime coeff = -1
        lin_rel[basis] = ee
        lin_rel[prime_ind] = -1
        # Apply correction
        kahler_parameters += prime_height * lin_rel

    return kahler_parameters
```

### The Key Insight

The mapping from heights to Kähler parameters is:

1. **Translate by origin**: `t_raw = h[1:] - h[0]`
2. **Project using linear relations**: Apply corrections from the effective cone to ensure non-basis divisors are properly related to basis divisors

This projection uses the **effective cone** (dual of Mori cone) to enforce linear relations between divisors.

### Inverse: `kahler_to_heights`

```python
def kahler_to_heights(poly, kahler_in):
    # Set basis coordinates from Kähler params, non-basis to 0
    basis = poly.glsm_basis(include_origin=True)
    return [t_i if i in basis else 0 for i in range(h11+5)]
```

### Usage Example

```python
import numpy as np
from cytools import Polytope
from cytools.utils import heights_to_kahler, kahler_to_heights

# Load polytope and triangulation with specific heights
poly = Polytope(points)
heights = np.loadtxt("heights.dat")
tri = poly.triangulate(heights=heights)

# Convert heights to Kähler moduli
t = heights_to_kahler(poly, heights)  # Returns h11-dimensional vector

# Inverse: Kähler to heights (note: loses information for non-basis)
h_recovered = kahler_to_heights(poly, t)
```

---

## KKLT Solver Implementation (VALIDATED)

The KKLT solver in `mcallister_2107/compute_kklt_iterative.py` has been validated against
McAllister's data with **0.0003% error** on V_string = 4711.83.

### Key Findings

1. **heights_to_kahler() Doesn't Work for McAllister 4-214-647**
   - `heights_to_kahler(poly, heights)` gives t with **negative correlation** (-0.61) to the solution
   - This is specific to this polytope/triangulation combination
   - For validation, use `kahler_param.dat` (uncorrected) scaled to match τ_target

2. **Extended Kähler Cone is Essential**
   - The solution has 19/214 negative t values
   - The solver MUST allow negative t (fixed backtracking bug)
   - What matters is positive divisor volumes τ, not positive t

3. **Starting Point Determines Solution**
   - The map τ(t) → t has multiple solutions
   - Starting from scaled `kahler_param.dat` converges to correct solution
   - Starting from uniform t converges to different (wrong) solution

### Validated Results

```
V_string (computed) = 4711.85
V_string (expected) = 4711.83
V error = 0.0003%
||t_ours - t_corrected|| / ||t_corrected|| = 0.000061
```

### For New Polytopes

For arbitrary polytopes (not McAllister validation), the challenge remains:
finding a good starting point t_init. Options:

1. **Random sampling** in extended Kähler cone and take best solution by V
2. **Tip of stretched cone** (if computable for small h11)
3. **Heights-based** (may work for some polytopes, test correlation first)

---

## Important: Basis Mismatch (See CLAUDE.md)

**CYTools versions use different divisor bases!**

- **CYTools 2021** (McAllister's paper): `vendor/cytools_mcallister_2107`
- **CYTools 2025** (latest): `vendor/cytools_latest`

Use `mcallister_2107/transform_km_to_new_cytools_basis.py` to convert between them.

See CLAUDE.md "CYTools (Two Versions!)" section for details.

---

## Key Equations

### 1. D-term / Moment Map (GLSM)
```
Σᵢ Qᵢₐ |zᵢ|² = Re(tₐ)    for a = 1, ..., h¹¹
```

### 2. FI Parameters = Kähler Volumes
```
∫_{C^a} ω = t^a
```

### 3. Kähler Class from Symplectic Quotient
```
[ω] = Σᵢ rᵢ eᵢ    (Duistermaat-Heckman theorem)
```
The Kähler class varies linearly with the FI parameters rᵢ.

### 4. Secondary Cone Inequality (2D example)
```
h_{s1} + h_{s2} ≤ h_{n1} + h_{n2}
```
For adjacent simplices sharing edge (s1, s2) with opposite vertices n1, n2.

### 5. Cone Duality
```
Mori cone = dual(Secondary cone)
Kähler cone = dual(Mori cone)
```

### 6. Divisor Volume (Classical)
```
τᵢ = (1/2) κᵢⱼₖ tʲ tᵏ
```

### 7. Jacobian of τ(t)
```
∂τᵢ/∂tʲ = κᵢⱼₖ tᵏ
```
For h¹¹=214, this Jacobian has rank ~65 (not 214), meaning ~149-dimensional nullspace.

---

## Witten's Phases

From Witten "Phases of N=2 theories" (hep-th/9301042):

### Phase Structure

Different values of FI parameters rₐ correspond to different **phases**:
- Phase I: Calabi-Yau sigma model (rₐ > 0)
- Phase II/III: Hybrid Landau-Ginzburg models
- Phase boundaries: Some rₐ = 0

### Key Quote (around line 3125)
> "The symplectic quotient ℂᴺ//U(1)ᵈ depends on d parameters r₁, ..., rᵈ, the constant terms that can be added to the D functions... the Kähler class of Z = ℂᴺ//U(1)ᵈ is [ω] = Σᵢ rᵢ eᵢ."

This confirms: **FI parameters directly control the Kähler class**.

---

## References

### Primary Sources

1. **Witten**, "Phases of N=2 theories in two dimensions" (hep-th/9301042)
   - Original GLSM paper
   - FI parameters and phase structure

2. **Bouchard**, "Toric geometry introduction" (arXiv:0901.3695)
   - Eq (108): FI params = Kähler volumes
   - Accessible physics-oriented introduction

3. **CYTools paper** (arXiv:2211.03823)
   - §5.4.1: Kähler cone as projection of secondary cone
   - Computational methods

4. **MacFadden**, "Efficient Algorithm for Generating Homotopy Inequivalent Calabi-Yaus" (arXiv:2309.10855)
   - Secondary cones and height vectors
   - Algorithm for finding heights in intersection of 2-face cones

5. **Segal**, "A Short Guide to GKZ" (arXiv:2412.14748)
   - GKZ discriminants and secondary polytope
   - Mathematical foundations

6. **Bouchard**, "Complex geometry, CY manifolds and toric geometry" (hep-th/0702063)
   - §4.3: Toric manifolds as symplectic quotients
   - GLSM charge matrix and moment maps

### Textbooks

- Cox, Little, Schenck: "Toric Varieties" (comprehensive mathematical treatment)
- Gelfand, Kapranov, Zelevinsky: "Discriminants, Resultants, and Multidimensional Determinants" (GKZ theory)

---

## TODO / Open Questions

1. **Explicit height → t formula**: How exactly do heights map to Kähler moduli?

2. **MOSEK/OSQP role**: CYTools uses these for `tip_of_stretched_cone()`. Can we use similar LP methods for large h¹¹?

3. **McAllister's actual algorithm**: Their code must have a way to get t from heights. Need to find it.

4. **Rank deficiency**: Why is the Jacobian ∂τ/∂t rank-deficient? Is this physical (gauge redundancy) or numerical?

5. **Alternative starting points**: Can we use random sampling in the Kähler cone instead of computing the tip?
