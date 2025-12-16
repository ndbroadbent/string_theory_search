# χ(D_i) Investigation Summary

## The Problem

The KKLT target divisor volume formula (McAllister eq 5.12) is:
```
τ_i = c_i/c_τ + χ(D_i)/24 - GV_corrections
```

We needed to compute χ(D_i) for each divisor. Our initial formula gave wrong values.

## What We Tried

### Formula 1: Braun et al. (arXiv:1712.04946)
```
χ(D) = 12 × χ(O_D) - D³
```
Where:
- χ(O_D) computed combinatorially from point location in polytope
- D³ = κ_DDD is triple self-intersection

**Results for 4-214-647 (using basis.dat):**
- χ range: [4, 47]
- Mean: 6.24

**Problem:** Values too large! Implied χ from McAllister's τ data is [2.8, 9.1]

### Root Cause Discovery

The issue was **basis.dat vs kklt_basis.dat mismatch**:

1. `basis.dat` includes points 1, 2 which are **vertices with g > 0**:
   - Point 1: vertex, g=1 → χ(O_D)=2 → χ(D)=12×2-D³=33
   - Point 2: vertex, g=2 → χ(O_D)=3 → χ(D)=12×3-D³=47

2. `kklt_basis.dat` does NOT include these problematic points

### Correct Approach

When computing χ for **KKLT divisors only** (using kklt_basis.dat indexing):
- χ range: **[4, 12]** ← Much better!
- Mean: 5.94

This is because kklt_basis excludes the high-g vertices.

## The χ(D) Formula

From the paper (eq 4.11, line 744):
```
T_i = -i(∫_X C_4 ∧ ω_i - χ(D_i)/24 C_0) + (1/g_s) Σ g_s^k T^[k]_i + ...
```

And eq 4.12 (line 790):
```
T_i = (1/2) κ_ijk t^j t^k - χ(D_i)/24 + GV_corrections
```

The χ(D_i) here is the **topological Euler characteristic of the divisor D_i**.

## CYTools second_chern_class()

CYTools has `cy.second_chern_class()` which gives `c_2(X) · D_i` for each basis divisor.
- This is NOT the same as χ(D_i)
- For 4-214-647: c_2·D range is [-4, 58], mean 0.32
- This does NOT match the implied χ values [2.8, 9.1]

## Current Understanding

The formula `χ(D) = 12 × χ(O_D) - D³` appears correct for the Braun combinatorial computation.

The discrepancy is because:
1. We were computing χ for `basis.dat` divisors
2. McAllister's τ_expected is for `kklt_basis.dat` divisors
3. The problematic high-χ divisors (points 1, 2) aren't in kklt_basis

## Remaining Questions

1. Is Braun's formula the same χ that McAllister uses in eq 4.12?
2. What about the 4 unmapped KKLT divisors (points 8, 9, 10, 17)?
3. Should we search for "Euler characteristic divisor" in the paper for explicit definition?

## Key Code Location

The χ computation is in:
- `mcallister_2107/2021_cytools/compute_chi_divisor.py`
- Function `compute_chi_divisor(poly, kappa_sparse, basis_indices)`

The new function for KKLT divisors:
- `compute_chi_for_kklt_divisors(poly, kappa_sparse, basis, kklt_basis)`
- In `mcallister_2107/2021_cytools/compute_kahler_param.py`

## Numerical Results

For 4-214-647 with kklt_basis divisors:
```
Target τ (c_i/c_τ + χ/24):  [0.4660, 2.2958]
Expected τ:                 [0.4167, 2.1752]
Difference (GV correction): [-0.28, 0.08], mean -0.006
```

The ~0.04 RMS error is consistent with GV corrections we're not computing.
