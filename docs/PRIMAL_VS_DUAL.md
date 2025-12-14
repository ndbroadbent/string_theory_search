# Primal vs Dual Polytopes in String Theory Compactifications

## Overview

In toric geometry for Calabi-Yau compactifications, every reflexive polytope Δ° has a **polar dual** polytope Δ. These are related by:

```
Δ = { y ∈ ℝ⁴ : ⟨x, y⟩ ≥ -1 for all x ∈ Δ° }
```

The CY threefold X from Δ° and the CY threefold X̃ from Δ are **mirror pairs**:
- h¹¹(X) = h²¹(X̃)
- h²¹(X) = h¹¹(X̃)

## Notation in McAllister (arXiv:2107.09064)

| Symbol | Name | Description |
|--------|------|-------------|
| Δ° | Primal polytope | The "original" polytope from Kreuzer-Skarke |
| Δ | Dual polytope | Polar dual of Δ° |
| X | CY from Δ° | Calabi-Yau hypersurface in toric variety from Δ° |
| X̃ | Mirror CY from Δ | Calabi-Yau hypersurface in toric variety from Δ |
| V | Toric variety from Δ° | Ambient space for X |
| Ṽ | Toric variety from Δ | Ambient space for X̃ |

## Hodge Numbers and Physics

The key relationship is:
```
h¹¹(X) = h²¹(X̃)    (Kähler moduli of X = complex structure moduli of X̃)
h²¹(X) = h¹¹(X̃)    (complex structure moduli of X = Kähler moduli of X̃)
```

In the McAllister pipeline:
- **X** (primal CY): h¹¹ = large (51-214), h²¹ = small (4-7)
- **X̃** (mirror CY): h¹¹ = small (4-7), h²¹ = large (51-214)

## What's Computed on Which Manifold

### On the Primal X (from Δ°):
- Kähler moduli t^i (h¹¹ of them)
- Divisor volumes τ_i
- CY volume V = (1/6)κ_{ijk}t^i t^j t^k
- Intersection numbers κ_{ijk}
- KKLT stabilization (non-perturbative superpotential)
- Orientifold structure (O7-planes, D3-instantons)

### On the Mirror X̃ (from Δ):
- Complex structure moduli z^a (h²¹ of them, which = h¹¹(X̃))
- Flux superpotential W_flux(z, τ)
- Prepotential F(z)
- GV invariants (via mirror map)
- The vector **p** from flux equations: p^a = (κ̃_{abc} M^c)^{-1} K_b

### The Key Formula

The flux vacuum condition (eq. 2.24 in McAllister):
```
p^a := (κ̃_{abc} M^c)^{-1} K_b ∈ K_{X̃}
```

Here:
- κ̃_{abc} are intersection numbers of the **mirror** X̃ (computed from Δ)
- K, M are flux vectors (h²¹ dimensional)
- p must lie in the Kähler cone of X̃

## CYTools Data Files

McAllister's data provides both:

| File | Polytope | Description |
|------|----------|-------------|
| `points.dat` | Δ° (primal) | Lattice points, # points ≈ h¹¹ + 4 |
| `dual_points.dat` | Δ (dual) | Lattice points, # points ≈ h²¹ + 4 |
| `dual_simplices.dat` | Δ (dual) | Triangulation simplices |
| `basis.dat` | For primal CY | Divisor basis indices |
| `kahler_param.dat` | For primal CY | Kähler parameters t^i |
| `dual_curves.dat` | For mirror CY | Curves in Mori cone of X̃ |
| `K_vec.dat`, `M_vec.dat` | Fluxes | h²¹-dimensional vectors |

## Favorability

A polytope can be **favorable** or **non-favorable**:

### Δ°-favorable (N-favorable in CYTools)
- Every 2-face of Δ° with interior points is dual to a 1-face of Δ without interior points
- Result: h¹¹(X) = h¹¹(V) (all divisors inherited from ambient space)
- Basis size = h¹¹

### Δ-favorable (M-favorable in CYTools)
- Every 2-face of Δ with interior points is dual to a 1-face of Δ° without interior points
- Result: h²¹(X) = h¹¹(Ṽ) = h¹¹(X̃)

### The 7-51-13590 Case

| Property | Primal (Δ°) | Dual (Δ) |
|----------|-------------|----------|
| Points | 65 | 13 |
| N-favorable | **NO** | YES |
| M-favorable | YES | NO |
| CY Hodge | h¹¹=51, h²¹=7 | h¹¹=7, h²¹=51 |
| Basis size | **48** (not 51!) | 7 |

The primal is NOT N-favorable, so only 48 of 51 divisors can be expressed in terms of the prime toric divisor basis. McAllister handled this by working with the reduced 48-dimensional basis.

## Practical Implications

### For CYTools 2021

```python
# Check favorability
poly.is_favorable(lattice='N')  # Δ°-favorable
poly.is_favorable(lattice='M')  # Δ-favorable

# For 7-51-13590:
# - Primal: N-favorable=False, M-favorable=True
# - Dual: N-favorable=True, M-favorable=False
```

### When Primal is Non-Favorable

Options:
1. **Use the dual polytope** for computations where h¹¹ is needed
2. **Use reduced basis** (48 instead of 51) as McAllister did
3. **Enable experimental features** in latest CYTools: `config.enable_experimental_features()`

### Mirror Symmetry Relations

The prepotential has cubic term:
```
F_poly = -(1/6) κ̃_{abc} z^a z^b z^c + ...
```

At large complex structure, z^a → p^a × τ (perturbatively flat vacuum), so:
```
e^{K₀} = (4/3 × κ̃_{abc} p^a p^b p^c)^{-1}
```

This uses the **mirror** intersection numbers κ̃_{abc}.

## Summary: Which Polytope for What

| Computation | Use Polytope | Why |
|-------------|--------------|-----|
| Kähler moduli stabilization | Primal (Δ°) | t^i live on X |
| Volume V_string | Primal (Δ°) | V = (1/6)κ_{ijk}t^i t^j t^k |
| Flux superpotential W₀ | Dual (Δ) | W_flux depends on z^a |
| Vector p, e^{K₀} | Dual (Δ) | κ̃_{abc} from mirror X̃ |
| GV invariants | Dual (Δ) | Curves on mirror |
| Racetrack stabilization | Dual (Δ) | Curves q, exponents q·p |

## Code Pattern

```python
# Load both polytopes
primal_points = load("points.dat")
dual_points = load("dual_points.dat")

# For Kähler moduli (h11 dimensional):
poly_primal = Polytope(primal_points)
cy_primal = poly_primal.triangulate().get_cy()  # h11=51 for 7-51
t = load("kahler_param.dat")  # 48 values (reduced basis)
V_string = compute_volume(cy_primal, t)

# For flux computations (h21 dimensional):
poly_dual = Polytope(dual_points)
tri_dual = poly_dual.triangulate(simplices=load("dual_simplices.dat"))
cy_dual = tri_dual.get_cy()  # h11=7 (which is h21 of primal)
kappa_tilde = cy_dual.intersection_numbers()  # κ̃_{abc}
p = solve_for_p(kappa_tilde, K, M)
eK0 = compute_eK0(kappa_tilde, p)
```

## References

- McAllister et al. arXiv:2107.09064, Section 2.4 (Orientifolds of CY hypersurfaces)
- Kreuzer-Skarke database: all 473M reflexive 4-polytopes
- CYTools documentation on favorability
