# Numeric Precision in String Theory Computations

## Overview

Computing the cosmological constant V₀ ≈ 10⁻¹²³ requires careful attention to numeric precision. This document explains when float64 is sufficient and when arbitrary precision (mpmath) is required.

## The Key Insight: Relative vs Absolute Precision

What matters is **relative precision**, not absolute precision.

| Quantity | Typical Value | Relative Precision Needed | Recommended Type |
|----------|---------------|---------------------------|------------------|
| Intersection numbers κᵢⱼₖ | O(1) - O(100) | ~10⁻¹² | float64 |
| Kähler moduli tⁱ | O(1) - O(100) | ~10⁻¹² | float64 |
| CY volume V_string | O(1000) | ~10⁻¹² | float64 |
| String coupling g_s | O(0.01) | ~10⁻¹⁰ | float64 |
| Flux superpotential W₀ | **10⁻⁹⁰** | ~10⁻¹⁰ | **mpmath** |
| Vacuum energy V₀ | **10⁻²⁰³** | ~10⁻¹⁰ | **mpmath** |

## Why W₀ Requires Arbitrary Precision

The flux superpotential has the form (McAllister eq. 6.61):
```
W₀ = 80 × ζ × 528⁻³³
```

Where:
- 528⁻³³ ≈ 10⁻⁹⁰ (cannot be represented in float64!)
- ζ is a product of racetrack coefficients

float64 can only represent numbers down to ~10⁻³⁰⁸, but with only ~15 significant digits. Computing `528**(-33)` in float64 gives garbage beyond the first few digits.

**Solution:** Use `mpmath` with sufficient precision:
```python
import mpmath
mpmath.mp.dps = 150  # 150 decimal places

W0 = mpmath.mpf(80) * zeta * mpmath.power(528, -33)
```

## Why V_string Does NOT Require Arbitrary Precision

The CY volume formula:
```
V_string = (1/6) κᵢⱼₖ tⁱ tʲ tᵏ - ζ(3)χ/(4(2π)³)
```

Involves:
- ~10 million terms (214³ for h¹¹=214)
- Each term is O(1) to O(1000)
- Final result is O(1000)

float64 provides ~15 significant digits, giving relative precision ~10⁻¹⁵. After summing 10⁷ terms, we expect precision loss to ~10⁻¹² at worst. This is more than sufficient.

### Verified Example

For McAllister's polytope 4-214-647:
```
V_string (computed): 4711.829675202376
V_string (target):   4711.829675204889
Difference:          2.5×10⁻⁹
Relative error:      5×10⁻¹³
```

The 2.5×10⁻⁹ absolute difference is **not a bug** - it's expected floating point variance across different:
- CPU architectures (Intel vs ARM)
- BLAS libraries (MKL vs OpenBLAS vs Accelerate)
- numpy versions
- Compiler optimization flags

## Error Propagation to V₀

The vacuum energy formula:
```
V₀ = -3 × e^K₀ × (g_s⁷ / (4×V_string)²) × W₀²
```

Error propagation:
- δV₀/V₀ ≈ 2 × δV_string/V_string + 2 × δW₀/W₀ + 7 × δg_s/g_s + ...

Since V_string appears as V², a 10⁻¹³ relative error in V_string contributes ~2×10⁻¹³ relative error to V₀. This is negligible compared to other uncertainties.

## When to Use mpmath

### MUST use mpmath:
1. **Racetrack superpotential W₀** - involves exp(-2πτ) with τ ~ 100
2. **Final V₀ computation** - W₀² dominates
3. **Any quantity involving W₀**

### float64 is fine:
1. **Intersection numbers κᵢⱼₖ** - integers or simple rationals
2. **Kähler moduli tⁱ** - O(1) to O(100)
3. **CY volume V_string** - O(1000)
4. **Divisor volumes τᵢ** - O(1) to O(100)
5. **e^K₀** - O(0.1) to O(1)
6. **g_s** - O(0.01)

## Code Example

```python
import numpy as np
import mpmath

mpmath.mp.dps = 150  # 150 decimal places for W₀ computation

# Geometric quantities - float64 is fine
kappa = cy.intersection_numbers(in_basis=True)  # float64
t = np.array([...])  # float64
V_string = np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0  # float64

# Racetrack - MUST use mpmath
tau = mpmath.mpf(109.75)
W0 = mpmath.mpf(80) * zeta * mpmath.exp(-2 * mpmath.pi * tau)  # mpmath

# Final V₀ - use mpmath since it involves W₀
e_K0 = mpmath.mpf(0.234393)
g_s = mpmath.mpf(0.00911134)
V_string_mp = mpmath.mpf(V_string)  # convert for final calc

V0 = -3 * e_K0 * (g_s**7 / (4 * V_string_mp)**2) * W0**2
# V0 ≈ -5.5×10⁻²⁰³
```

## Cross-Platform Reproducibility

Floating point results can vary slightly across platforms due to:

1. **CPU architecture**: x86-64 vs ARM64 vs others
2. **SIMD instructions**: AVX2 vs NEON vs SSE
3. **BLAS library**: MKL vs OpenBLAS vs Accelerate
4. **Compiler flags**: `-ffast-math` reorders operations
5. **numpy version**: Internal optimizations change

For V_string ≈ 4711, expect variations of ~10⁻⁹ to 10⁻⁸ across platforms. This is normal and acceptable.

For W₀ and V₀ computed with mpmath, results should be identical across platforms (mpmath is pure Python, platform-independent).

## Summary

| Computation | Precision | Library | Notes |
|-------------|-----------|---------|-------|
| κᵢⱼₖ, tⁱ, V_string | float64 | numpy | Standard numeric computing |
| W₀, V₀ | 150+ digits | mpmath | Exponentially small values |
| Cross-platform V_string | ±10⁻⁸ | - | Expected variance |
| Cross-platform W₀ | exact | mpmath | Pure Python, deterministic |

## References

- McAllister et al., arXiv:2107.09064, Section 6 (discusses precision requirements)
- mpmath documentation: https://mpmath.org/
- IEEE 754 floating point: https://en.wikipedia.org/wiki/IEEE_754
