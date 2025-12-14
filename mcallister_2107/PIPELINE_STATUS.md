# McAllister Pipeline Status - 2024-12-14

## Overview

This document tracks the validation status of each pipeline step against McAllister's 5 examples (arXiv:2107.09064).

**Goal:** Compute V₀ = -5.5 × 10⁻²⁰³ for polytope 4-214-647, then generalize to GA search.

---

## Script Status Summary

| Script | Status | Tests Passing | Key Formula | Notes |
|--------|--------|---------------|-------------|-------|
| `compute_triangulation.py` | ✅ | 5/5 | - | Loads polytope, heights, simplices |
| `compute_V_string.py` | ✅ | 4/5 | eq 4.11 | 7-51 excluded (non-favorable) |
| `compute_c_i.py` | ✅ | 5/5 | - | MODEL INPUT, not computed |
| `compute_rigidity_combinatorial.py` | ✅ | 4/5 | Braun eq 2.7 | 7-51 excluded |
| `compute_target_tau.py` | ✅ | 5/5 | eq 2.29, 5.12 | |
| `compute_chi_divisor.py` | ✅ | 4/5 | 12×χ(O_D)-D³ | ~3% error = GV corrections |
| `compute_gv_invariants.py` | ✅ | 5/5 | - | min_points=20000 |
| `compute_derived_racetrack.py` | ✅ | 5/5 | eq 2.22-2.26 | **FIXED 2024-12-14**: same-sign coefficient case |
| `compute_kklt_iterative.py` | ✅ | 4/5 | eq 4.12 | V_string validated; solver exists but main() only validates |
| `compute_divisor_cohomology.py` | ✅ | 5/5 | Koszul | Dual polytopes only (primal too large) |

---

## Key Formulas (from .tex source)

### 1. V_string (eq 4.11)
```
V_string = (1/6) κ_ijk t^i t^j t^k - ζ(3)χ/(4(2π)³)
```
- BBHL correction: `ζ(3) × χ / (4(2π)³)` where `χ = 2(h11 - h21)`
- **Validated:** 5-81-3213 matches to 1e-13 (exact), others 6-9 sig figs
- **Note:** Compare against `cy_vol.dat` (classical), NOT `corrected_cy_vol.dat` (with worldsheet instantons)

### 2. e^{K₀} (eq 6.12)
```
e^{K₀} = (4/3 × κ̃_abc p^a p^b p^c)^{-1} = (3/4) / κ̃p³
```
- **CRITICAL:** The inverse applies to the ENTIRE expression including 4/3
- See `EK0_FORMULA_DISCREPANCY_RESOLUTION.md` for full explanation
- **Validated:** All 5 examples match when using correct formula

### 3. c_τ (eq 2.29)
```
c_τ = 2π / (g_s × ln(1/W₀))
```
- **Validated:** All 5 examples match to < 1e-6 error

### 4. Target τ (eq 5.12-5.13)
```
τ_target = c_i/c_τ + χ(D_i)/24 - GV_corrections
```
- First two terms validated, ~3% residual = GV corrections (not yet implemented)

### 5. T_i with instanton corrections (eq 4.12)
```
T_i(t) = (1/2) κ_ijk t^j t^k - χ(D_i)/24 + (GV sum)
```
- Classical term validated
- GV corrections NOT yet integrated into solver

### 6. Racetrack Stabilization (eq 2.22-2.26)
```
W_flux(τ) = -ζ Σ_q (M·q) N_q Li₂(e^{2πiτ(q·p)})

δ = -[(M·q₁)(p·q₁)N_{q₁}] / [(M·q₂)(p·q₂)N_{q₂}]
ε = (p·q₂) - (p·q₁)

⟨e^{2πiτ}⟩ ≈ δ^{1/ε}

For δ > 0 (opposite-sign): τ = i × ln(1/δ)/(2πε)
For δ < 0 (same-sign):     τ = 1/(2ε) + i × ln(1/|δ|)/(2πε)

g_s = 1/Im(τ)
W₀ = |W_flux(τ)|
```
- **Validated:** All 5 examples match g_s exactly, W₀ within 2.2%
- **Key insight:** Same-sign coefficient case requires complex τ with Re(τ) ≠ 0

### 7. Final V₀ (eq 6.24)
```
V₀ = -3 × e^{K₀} × (g_s⁷/(4×V_string)²) × W₀²
```
- **Requires:** e^{K₀}, g_s, V_string, W₀
- W₀ ~ 10⁻⁹⁰ requires mpmath (150+ digits)

---

## Remaining Issues

### 1. ✅ RESOLVED: compute_derived_racetrack.py - NOW 5/5 pass

**Fixed 2024-12-14.** All 5 examples now pass.

**Root cause was:** The original solver only handled opposite-sign coefficient racetracks.
For 3 examples, the two leading terms have **same-sign coefficients** (δ < 0).

**The fix:**
1. Use correct δ formula from eq. 2.25: `δ = -[(M·q₁)(p·q₁)N_{q₁}] / [(M·q₂)(p·q₂)N_{q₂}]`
2. For same-sign case (δ < 0), τ has non-zero real part: `Re(τ) = 1/(2ε)`
3. Compute W₀ using `compute_W_at_complex_tau()` which handles complex τ

**Results:**
- ✅ 4-214-647: g_s=0.009111, W₀~10⁻⁹⁰ (ratio=1.000)
- ✅ 5-113-4627-main: g_s=0.011119, W₀~10⁻⁶¹ (ratio=1.000)
- ✅ 5-113-4627-alternative: g_s=0.003589, W₀~10⁻⁹⁵ (ratio=1.000)
- ✅ 5-81-3213: g_s=0.050414, W₀~10⁻²³ (ratio=1.000)
- ✅ 7-51-13590: g_s=0.040331, W₀~10⁻²⁰ (ratio=1.022)

### 2. GV corrections in KKLT solver
The τ_target formula has ~3% residual error because we haven't added GV corrections to the iterative solver.

**Required:**
- Integrate GV invariants into `compute_kklt_iterative.py`
- Use `corrected_kahler_param.dat` (not `kahler_param.dat`) for validation

### 3. 7-51-13590 non-favorable
The primal polytope for 7-51-13590 is non-favorable in CYTools 2021, so:
- `compute_V_string.py` - skipped
- `compute_rigidity_combinatorial.py` - skipped
- `compute_chi_divisor.py` - skipped

This is NOT a bug - the polytope genuinely doesn't have a favorable triangulation in 2021.

---

## Verified Understanding

### e^{K₀} Formula (RESOLVED)
- Initial confusion: thought formula was `(4/3) × (κp³)^{-1}`
- **Correct:** `((4/3) × κp³)^{-1} = (3/4)/(κp³)`
- The 16/9 discrepancy was from misreading LaTeX parentheses
- See `EK0_FORMULA_DISCREPANCY_RESOLUTION.md`

### V_string vs corrected_V_string
- `cy_vol.dat` = classical V + BBHL (what we compute)
- `corrected_cy_vol.dat` = classical V + BBHL + worldsheet instantons
- Our formula matches `cy_vol.dat`, not `corrected_cy_vol.dat`

### c_i values (MODEL CHOICE)
- c_i = 1 (D3-instanton), 6 (O7/SO(8)), or 2 (Sp(2))
- These are MODEL INPUTS from orientifold choice
- Loaded from `target_volumes.dat`, NOT computed from geometry

---

## Next Steps

1. ~~**Fix racetrack script**~~ ✅ DONE (2024-12-14)
2. **Test KKLT iterative solver** - verify against `corrected_kahler_param.dat`
3. **Add GV corrections** - to τ_target and T_i solver
4. **End-to-end test** - compute V₀ = -5.5e-203 for 4-214-647
5. **Document remaining formulas** - add .tex equation references

---

## Test Commands

```bash
# Run all validated scripts (all should PASS)
uv run python mcallister_2107/2021_cytools/compute_V_string.py
uv run python mcallister_2107/2021_cytools/compute_c_i.py
uv run python mcallister_2107/2021_cytools/compute_rigidity_combinatorial.py
uv run python mcallister_2107/2021_cytools/compute_target_tau.py
uv run python mcallister_2107/2021_cytools/compute_chi_divisor.py
uv run python mcallister_2107/2021_cytools/compute_gv_invariants.py  # slow, ~2min
uv run python mcallister_2107/2021_cytools/compute_derived_racetrack.py  # 5/5 PASS
```
