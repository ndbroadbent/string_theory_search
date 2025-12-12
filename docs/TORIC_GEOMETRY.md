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

## The Unsolved Problem

### Given: Heights h defining a triangulation
### Want: Kähler moduli t inside the Kähler cone

### What We Know

1. Heights h define a triangulation T
2. T has a secondary cone SecCone(T) ⊂ ℝⁿ
3. The Kähler cone K ⊂ ℝ^h¹¹ is derived from SecCone(T)
4. We need a point t inside K

### What McAllister Does

From arXiv:2107.09064 Section 5:

1. Pick random h_init in secondary fan
2. This gives triangulation T and "associated" t_init
3. Solve KKLT iteratively: t_init → τ_init → τ_target → t_final

**The paper says heights are "naturally associated" to t but doesn't specify the explicit map!**

### Why This Is Hard for Large h¹¹

For h¹¹ ≤ 12, CYTools can compute `tip_of_stretched_cone()` to get a valid starting point.

For h¹¹ = 214:
- `tip_of_stretched_cone()` is computationally intractable
- Warning: "This operation might take a while for d > ~12 and is likely impossible for d > ~18"
- The cone has 214 dimensions with complex geometry

### Attempted Solutions That Failed

1. **Newton iteration for unit τ**: Jacobian is rank-deficient (rank 65 vs 214)
2. **L-BFGS-B optimization**: Found spurious local minimum (V=-4478 vs +4712)
3. **Direct GLSM projection Q @ h**: Gives negative correlation with actual t

### What We Still Need

The explicit algorithm/formula to:
1. Take heights h ∈ ℝⁿ
2. Produce valid Kähler moduli t ∈ ℝ^h¹¹ inside the Kähler cone

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
