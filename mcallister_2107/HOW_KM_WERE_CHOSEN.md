# How McAllister's K, M Fluxes Were Chosen

## Summary

McAllister et al. did **NOT** randomly sample K, M. They solved a set of **Diophantine constraints** that guarantee a perturbatively flat vacuum with exponentially small W₀.

## The Constraints (from arXiv:2107.09064, Section 2.3)

For flux vectors of the form:
```
f⃗ = (0, Mᵃ, 0, 0)
h⃗ = (0, Kₐ, 0, 0)
```

The following constraints must be satisfied:

### (a) Tadpole Constraint
```
0 ≤ -½ M·K ≤ χf/4
```
The D3-brane charge from fluxes must not exceed the orientifold fixed-point contribution.

### (b) p in Kähler Cone
```
pᵃ = (κ̃ₐbᶜ Mᶜ)⁻¹ Kb  ∈  K_X̃
```
The flat direction p (computed via the Demirtas lemma) must lie inside the Kähler cone of the mirror threefold.

### (c) Orthogonality
```
K · p = 0
```
The flux K must be orthogonal to the flat direction p.

### Additional Racetrack Conditions (d, e, f)

To get a racetrack superpotential with exponentially small W₀:

**(d)** Find curve pair (q̃₁, q̃₂) generating the effective cone with:
- q̃₁ · p < q̃₂ · p  (different actions)
- (M · q̃₁)(M · q̃₂) < 0  (opposite signs for racetrack)

**(e)** GV invariants N_q̃₁ and N_q̃₂ must be nonzero

**(f)** The pair dominates - other curves have larger q·p (more suppressed)

## Why This is Hard

The paper states:
> "The conditions for a perturbatively flat vacuum in (2.3) are **Diophantine in nature**, and so are difficult to solve in general. Nevertheless, in practice we have been able to find solutions to the constraints when h²¹ is relatively small."

Diophantine = finding integer solutions to polynomial constraints. This is computationally expensive, especially as h²¹ grows.

## For Our GA

### Option 1: Brute Force (what search_km.py does)
- Sample random integer K, M
- Check constraints (a), (b), (c)
- Compute racetrack and W₀
- Keep configurations with small |V₀|

This works for small h²¹ (like 4) but becomes expensive at larger h²¹.

### Option 2: Guided Sampling
1. **Start with constraint (b)**: Choose M such that N = κ_abc M^c is invertible
2. **Impose (c)**: K must satisfy K·p = 0, which constrains the space
3. **Check (a)**: Filter by tadpole bound
4. **Compute physics**: Only for configurations passing all filters

### Option 3: Solve Diophantine System
Directly solve the constraint system - but this requires sophisticated number theory algorithms.

## McAllister's Specific K, M for 4-214-647

In basis [3, 4, 5, 8]:
```python
K = [-3, -5, 8, 6]
M = [10, 11, -11, -5]
```

These give:
- p = [293/110, 163/110, 163/110, 13/22] (all positive → in Kähler cone)
- K·p = 0 ✓
- -½ M·K = 110 ≤ χf/4 ✓

## What Our Random Search Found

From 10M samples (search_km.py), we found configurations achieving:
- W₀ ~ 10⁻⁹⁰ (same order as McAllister)
- V₀ ~ 10⁻²⁰⁴ (within 1 order of magnitude of McAllister's -5.5×10⁻²⁰³)

This shows that for h²¹ = 4, random sampling with constraint filtering is viable. The search space has ~10¹² valid configurations, so with 10⁷ samples we explored ~0.001% and still found excellent solutions.

## Implications for GA Design

1. **Constraint filtering is essential** - most random (K, M) violate constraints
2. **Filter pipeline**: N invertible → p positive → tadpole OK → has racetrack
3. **For small h²¹**: Random sampling works
4. **For large h²¹**: May need smarter search (e.g., lattice algorithms, constraint propagation)

## References

- arXiv:2107.09064, Section 2.3: "Flux vacua"
- arXiv:1912.10047: Earlier work on perturbatively flat vacua
- Demirtas lemma: p = N⁻¹K where N_ab = κ_abc M^c
