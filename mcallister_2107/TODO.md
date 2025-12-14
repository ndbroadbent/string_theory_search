# McAllister Pipeline TODO

**Goal:** Reproduce V₀ for ALL 5 McAllister examples, then generalize to 12M polytopes.

---

## ✅ RESOLVED: Racetrack solver now works for all 5 examples

**Status:** 5/5 PASS (all examples work!)

**Root cause:** The original solver only handled "opposite-sign coefficient" racetracks.
For 3 examples, the two leading superpotential terms have **same-sign** coefficients.

**The fix (2024-12-14):**

1. **Correct δ formula (eq. 2.25):** The racetrack hierarchy parameter is:
   ```
   δ = -[(M·q₁)(p·q₁)N_{q₁}] / [(M·q₂)(p·q₂)N_{q₂}]
   ```
   The derivative coefficients include the `(p·q)` factor, not just `(M·q)×N_q`.

2. **Same-sign case (δ < 0):** When both superpotential coefficients have the same sign,
   δ is negative and the solution has non-zero Re(τ):
   ```
   τ = 1/(2ε) + i × ln(1/|δ|)/(2πε)
   ```
   where ε = β - α (exponent gap).

3. **W₀ computation:** Must use `compute_W_at_complex_tau()` for same-sign cases,
   evaluating the superpotential at the complex τ with non-zero real part.

**Validated results:**
- 4-214-647: g_s=0.009111, W₀~10⁻⁹⁰ ✓
- 5-113-4627-main: g_s=0.011119, W₀~10⁻⁶¹ ✓
- 5-113-4627-alternative: g_s=0.003589, W₀~10⁻⁹⁵ ✓
- 5-81-3213: g_s=0.050414, W₀~10⁻²³ ✓
- 7-51-13590: g_s=0.040331, W₀~10⁻²⁰ ✓

---

## Script Status

| Script | Status | Notes |
|--------|--------|-------|
| `compute_V_string.py` | ✅ 4/5 | 7-51 non-favorable |
| `compute_c_i.py` | ✅ 5/5 | MODEL INPUT |
| `compute_rigidity_combinatorial.py` | ✅ 4/5 | 7-51 non-favorable |
| `compute_target_tau.py` | ✅ 5/5 | |
| `compute_chi_divisor.py` | ✅ 4/5 | ~3% = GV corrections |
| `compute_gv_invariants.py` | ✅ 5/5 | |
| `compute_kklt_iterative.py` | ⚠️ 4/5 | Needs t solver + GV corrections |
| `compute_derived_racetrack.py` | ✅ 5/5 | **FIXED 2024-12-14** |
| `compute_divisor_cohomology.py` | ✅ 5/5 | Dual polytopes (primal too large for cohomCalg) |

---

## Verified Formulas

1. **e^{K₀} = (4/3 × κ̃p³)⁻¹** - RESOLVED (LaTeX parsing confusion)
2. **V_string = (1/6)κt³ - BBHL** - compare to `cy_vol.dat` not `corrected_cy_vol.dat`
3. **c_τ = 2π/(g_s × ln(1/W₀))** - all 5 match
4. **c_i values** - MODEL INPUTS from `target_volumes.dat`

---

## Next Steps

1. **Debug racetrack for failing examples**
   - Print leading terms for 5-81-3213 vs 4-214-647
   - Check if curve basis transformation is correct
   - Verify M·q and q·p products match paper's structure

2. **Once racetrack works for all 5:**
   - End-to-end V₀ computation
   - Validate against paper values

3. **Then generalize:**
   - Remove hardcoded example paths
   - Test on arbitrary polytopes from 12M filtered set

---

## Test Commands

```bash
# Working scripts
uv run python mcallister_2107/2021_cytools/compute_V_string.py
uv run python mcallister_2107/2021_cytools/compute_gv_invariants.py

# Broken (the blocker)
uv run python mcallister_2107/2021_cytools/compute_derived_racetrack.py
```
