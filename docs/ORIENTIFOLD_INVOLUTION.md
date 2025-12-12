# Orientifold Involutions and O7-Planes

This document explains how orientifold involutions determine O7-plane locations in type IIB string compactifications, based on McAllister et al. (arXiv:2107.09064).

## Overview

In type IIB O3/O7 orientifold compactifications, the orientifold involution I : X -> X determines:
1. Which divisors host O7-planes (fixed loci of the involution)
2. The D3-brane tadpole (from the Euler characteristic of the fixed locus)
3. Which moduli survive the orientifold projection

**Key insight:** The involution is a **model choice**, not determined by the polytope alone.

---

## The Orientifold Involution

### Definition (McAllister eq. 2.18)

For a Calabi-Yau hypersurface X in a toric variety V, the involution is defined by restricting an ambient involution I_hat : V -> V to X.

A general Z_2 conjugacy class of Aut^0(V, C) can be represented by **negating a subset of homogeneous coordinates**:

```
I_hat : x_{I_alpha} -> -x_{I_alpha},  alpha = 1, ..., k
```

where {x_{I_1}, ..., x_{I_k}} is a subset of {x_1, ..., x_n} - the coordinates to negate.

### O7-Plane Locations

Each negated coordinate x_i creates an **O7-plane on the divisor {x_i = 0}**.

The O7-plane is the fixed locus of the involution: if x_i -> -x_i, then the locus {x_i = 0} is pointwise fixed.

### Example: McAllister 4-214-647

For the h11=214 primal polytope:
- Total rigid divisors: 214 (all 214 prime toric divisors are rigid)
- O7-planes: 49 (from target_volumes.dat with c_i = 6)
- D3-instantons: 165 (from target_volumes.dat with c_i = 1)

This means their involution negates exactly 49 coordinates.

---

## Constraints on Valid Involutions

### 1. Hodge Number Constraint

From McAllister Section 2.2:

> "For simplicity we restrict to involutions for which h^{1,1}_-(X) = h^{2,1}_+(X) = 0, a very large class of which can be found systematically [31]."

**Physical meaning:** No geometric moduli are projected out by the orientifold.

- h^{1,1}_- = 0 means all Kahler moduli survive
- h^{2,1}_+ = 0 means all complex structure moduli survive

**Implementation note:** Reference [31] is "M. Kim, L. McAllister and J. Moritz, work in progress" - the systematic algorithm for finding valid involutions was not published.

### 2. D3-Brane Tadpole Cancellation

From McAllister eq. (2.10):

```
N_D3 + (1/2) * integral_X H_3 ^ F_3 = chi_f / 4
```

where:
- N_D3 = number of mobile D3-branes
- H_3, F_3 = three-form fluxes
- chi_f = Euler characteristic of the fixed locus of I in X

For 4-214-647: chi_f / 4 = 110 (the D3-brane tadpole)

### 3. D7-Brane Tadpole Cancellation

From McAllister Section 2.1:

> "We choose to cancel the D7-brane charge tadpole of the O7-planes locally, by placing four D7-branes on top of each O7-plane."

This gives so(8) gauge stacks on each O7-plane, with dual Coxeter number c_2(so(8)) = 6.

---

## Dual Coxeter Numbers (c_i)

The superpotential has the form (McAllister eq. 1.1):

```
W = W_flux(z, tau) + sum_D A_D(z, tau) exp(-2*pi*T_D / c_D)
```

where c_D is the dual Coxeter number:

| Source | c_i | Description |
|--------|-----|-------------|
| D3-brane instanton | 1 | Euclidean D3-brane on rigid divisor |
| so(8) gaugino condensation | 6 | O7-plane with 4 D7-branes (local tadpole cancellation) |

### KKLT Moduli Stabilization

At the KKLT minimum (McAllister eq. 5.7):

```
Re(T_i) ~ (c_i / 2*pi) * ln(W_0^{-1})
```

So:
- D3-instanton divisors (c_i = 1): tau_i ~ ln(W_0^{-1}) / (2*pi) ~ 32.85
- O7-plane divisors (c_i = 6): tau_i ~ 6 * ln(W_0^{-1}) / (2*pi) ~ 197.10

---

## Computing Divisor Rigidity

### The Problem

A divisor D contributes to the superpotential iff it is **rigid**: h^i(D, O_D) = (1, 0, 0).

This requires computing divisor cohomology, traditionally done with cohomCalg.

**Limitation:** cohomCalg has a 63-vertex limit, making it unusable for h11=214 polytopes.

### The Solution: Combinatorial Rigidity

From Braun, Long, McAllister, Stillman, Sung (arXiv:1712.04946):

For prime toric divisors on CY threefold hypersurfaces, rigidity is determined **combinatorially** from the polytope structure:

1. **Points interior to 2-faces of Delta^o** -> always rigid
2. **Points interior to 1-faces of Delta^o** -> always rigid (in Delta-favorable models)
3. **Vertices of Delta^o** -> rigid iff the dual facet in Delta has **no interior points**

### CYTools Implementation

```python
from cytools import Polytope

poly = Polytope(points)

# For each vertex, check if dual facet has interior points
for face in poly.faces(0):  # 0-faces = vertices
    dual_facet = face.dual_face()
    interior_pts = dual_facet.interior_points()
    is_rigid = (len(interior_pts) == 0)
```

### Validation

We validated this combinatorial method against McAllister's target_volumes.dat:
- **214/214 exact match** for the h11=214 primal polytope
- See `mcallister_2107/compute_rigidity_combinatorial.py`

---

## GA Genome: Orientifold as a Parameter

### Current Understanding

The genome for our GA search should include:

```python
genome = {
    # Polytope selection
    "polytope_id": int,           # Index into Kreuzer-Skarke database
    "triangulation_id": int,      # Which triangulation (FRST, etc.)

    # Flux vectors (determines W_0, g_s)
    "K": [int] * h21,             # Flux vector K (length = h21 of dual)
    "M": [int] * h21,             # Flux vector M (length = h21 of dual)

    # Orientifold choice (determines c_i values)
    "orientifold_mask": [bool] * n_coords,  # Which coordinates to negate
}
```

### Constraints on orientifold_mask

Not all binary masks are valid. The mask must satisfy:
1. h^{1,1}_-(X) = 0 (no Kahler moduli projected out)
2. h^{2,1}_+(X) = 0 (no complex structure moduli projected out)
3. Tadpole cancellation with chosen fluxes

### Search Strategy Options

1. **Enumerate valid involutions**: For small polytopes, enumerate all valid masks
2. **Random sampling with rejection**: Sample masks, reject invalid ones
3. **Learn from McAllister**: Use their 49/214 ratio (~23%) as a prior

---

## What Remains Unknown

1. **Systematic algorithm for finding valid involutions** - Referenced as [31] but unpublished
2. **How h^{1,1}_- = 0 constrains the mask** - Requires understanding how involution acts on cohomology
3. **Optimal O7-plane count** - Is 23% special, or just McAllister's choice?

---

## References

### Primary Sources

1. **McAllister et al. (arXiv:2107.09064)** - "Vacua with Small Flux Superpotential"
   - Section 2.1: O7-planes and D7-brane tadpole cancellation
   - Section 2.2: Orientifold involutions, eq. (2.18)
   - Section 2.3: D3-brane tadpole, eq. (2.10)
   - Section 6.4: 4-214-647 example with 49 O7-planes

2. **Braun, Long, McAllister, Stillman, Sung (arXiv:1712.04946)** - "The Hodge Numbers of Divisors of Calabi-Yau Threefold Hypersurfaces"
   - Combinatorial formula for divisor rigidity
   - No need for cohomCalg

3. **Demirtas et al. (arXiv:1912.10047)** - "Vacua with Small Flux Superpotential"
   - The p = N^{-1}K construction
   - Flux constraints for valid vacua

### CYTools Methods

- `Polytope.faces(dim)` - Get faces of given dimension
- `PolytopeFace.dual_face()` - Get the dual face in the polar polytope
- `PolytopeFace.interior_points()` - Get points interior to a face

### Data Files (4-214-647)

```
resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/
    points.dat              # Primal polytope (294 points, h11=214)
    dual_points.dat         # Dual polytope (12 points, h11=4)
    target_volumes.dat      # c_i values: 1 (D3) or 6 (O7) for 214 divisors
    kklt_basis.dat          # KKLT divisor basis indices
    K_vec.dat               # Flux vector K = [-3, -5, 8, 6]
    M_vec.dat               # Flux vector M = [10, 11, -11, -5]
    g_s.dat                 # String coupling = 0.00911134
    W_0.dat                 # Flux superpotential = 2.30012e-90
    cy_vol.dat              # CY volume = 4711.83
```

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Combinatorial rigidity | Validated | `mcallister_2107/compute_rigidity_combinatorial.py` |
| V_string computation | Validated | `mcallister_2107/compute_V_string.py` |
| Orientifold mask in GA | Not implemented | Needs integration into genome |
| Valid involution enumeration | Not implemented | Algorithm unknown |
