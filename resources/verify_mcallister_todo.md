# McAllister Verification Pipeline - TODO

**Goal**: Build a fully automated, end-to-end pipeline that can compute W₀, g_s, V₀ for any polytope + flux combination without hand-fed paper data.

**Test Case**: Polytope 4-214-647 from McAllister et al. arXiv:2107.09064

---

## Completed ✓

### 1. Reproduce McAllister from Paper Formulas (v6)
- [x] Load correct CY geometry (h¹¹=4, h²¹=214) from dual polytope
- [x] Use `intersection_numbers(in_basis=True)` for proper h¹¹ basis
- [x] Implement analytic racetrack solution (eq. 6.59-6.61)
- [x] Compute g_s = 2π/(110 log 528) = 0.00911134 ✓
- [x] Compute W₀ = 80ζ × 528⁻³³ = 2.30012e-90 ✓
- [x] Implement V₀ formula (eq. 6.24) with e^{K₀} = 0.2361
- [x] Compute V₀ = -5.5e-203 ✓
- [x] Create regression test: `verify_mcallister_from_paper_v6.py`

**Result**: All three values match McAllister's published data.

**Limitation**: Uses hand-fed constants from paper (racetrack coefficients, e^{K₀}).

---

## TODO - Stage-by-Stage Components

### 2. Re-derive Two-Term Racetrack from GV Inputs
**Status**: Not started

**Goal**: Build W_flux(τ) from geometry + flux + GV invariants, show it reduces to the 2-term racetrack.

**Inputs available**:
- `dual_curves.dat` - 5177 curves in 9D ambient basis
- `dual_curves_gv.dat` - 5177 GV invariants
- Fluxes M = [10, 11, -11, -5], K = [-3, -5, 8, 6]
- p from paper = (293/110, 163/110, 163/110, 13/22)

**Steps**:
- [ ] Load GV invariants N_q̃ for curve classes q̃
- [ ] Handle 9D → 4D basis projection (or work in 9D consistently)
- [ ] Compute q̃·p for each curve
- [ ] Compute M·q̃ for each curve
- [ ] Build effective coefficients: (M·q̃) × N_q̃
- [ ] Sort by q̃·p to find leading terms
- [ ] Verify the two smallest q̃·p values are 32/110 and 33/110
- [ ] Verify effective coefficients match -1 and +512 from eq. 6.59
- [ ] Build truncated W_flux(τ) = ζ Σ (M·q̃) N_q̃ Li₂(e^{2πiτ(q̃·p)})

**Output**: `verify_mcallister_recompute_derived_racetrack.py`

---

### 3. Numeric F-term Solver
**Status**: Not started

**Goal**: Solve ∂W/∂τ = 0 numerically instead of using analytic "528" solution.

**Steps**:
- [ ] Implement W_flux(τ) evaluation using mpmath polylog
- [ ] Implement ∂W/∂τ derivative
- [ ] Use mpmath.findroot to solve dW/dτ = 0 for Im(τ)
- [ ] Extract g_s = 1/Im(τ) and W₀ = |W_flux(τ_vev)|
- [ ] Verify matches analytic solution (Im τ ≈ 109.75, g_s ≈ 0.00911)

**Output**: `verify_mcallister_numeric_fterm.py`

---

### 4. Compute e^{K₀} from κ̃ and p
**Status**: Not started

**Goal**: Replace hardcoded e^{K₀} = 0.2361 with computed value.

**Challenge**: Basis mismatch between CYTools and McAllister.

**Options**:
- [ ] Option A: Find GL(4,Z) basis transformation between CYTools and paper
- [ ] Option B: Work entirely in paper's basis using their intersection numbers
- [ ] Option C: Verify e^{K₀} = (4/3 κ̃_abc p^a p^b p^c)⁻¹ ≈ 0.2361 in some consistent basis

**Output**: `verify_mcallister_compute_eK0.py`

---

### 5. Compute V[0] from Kähler Moduli
**Status**: Not started

**Goal**: Derive V[0] = 4711.83 from Kähler moduli stabilization instead of using cy_vol.dat.

**Steps**:
- [ ] Understand relationship: V[0] = V_cytools × g_s^{-3/2}
- [ ] Load kahler_param.dat (214 parameters for primal)
- [ ] Compute CY volume from intersection numbers and Kähler moduli
- [ ] Verify matches cy_vol.dat value

**Note**: This may require working with the primal (h¹¹=214) side.

**Output**: `verify_mcallister_compute_volume.py`

---

### 6. Perturbatively Flat Flux Search (Demirtas Lemma)
**Status**: Not started

**Goal**: Given arbitrary fluxes, check if they admit a perturbatively flat vacuum.

**Steps**:
- [ ] Build N_ab = κ̃_abc M^c
- [ ] Check det(N) ≠ 0
- [ ] Compute p = N⁻¹K
- [ ] Check p is in Kähler cone
- [ ] Check K·p ≈ 0
- [ ] Check integrality constraints

**Output**: `verify_mcallister_flux_flatness.py`

---

### 7. End-to-End Pipeline (No Paper Inputs)
**Status**: Not started

**Goal**: Combine all stages into single evaluation function.

```python
def evaluate_compactification(polytope, triangulation, flux_M, flux_K):
    """
    Returns (g_s, W0, V0) computed entirely from geometry + flux.
    No hand-fed paper data.
    """
    # 1. Load CY, get κ̃_abc
    # 2. Check flux flatness, get p
    # 3. Get GV invariants
    # 4. Build W_flux(τ)
    # 5. Solve F-term numerically
    # 6. Compute e^{K₀} from κ̃, p
    # 7. Compute V[0] from Kähler moduli
    # 8. Compute V₀
    return g_s, W0, V0
```

**Validation**: Run on 4-214-647, verify matches v6 results.

**Output**: `verify_mcallister_e2e.py`

---

## Data Files Reference

### McAllister Ancillary Data (4-214-647)
```
resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/
├── dual_points.dat          # 12 vertices of dual polytope
├── dual_simplices.dat       # 15 simplices (triangulation)
├── dual_curves.dat          # 5177 × 9 curve classes (ambient basis)
├── dual_curves_gv.dat       # 5177 GV invariants
├── K_vec.dat                # [-3, -5, 8, 6]
├── M_vec.dat                # [10, 11, -11, -5]
├── g_s.dat                  # 0.00911134
├── W_0.dat                  # 2.30012e-90
├── cy_vol.dat               # 4711.83
├── kahler_param.dat         # 214 Kähler parameters (primal)
├── basis.dat                # 214 divisor indices (primal)
├── kklt_basis.dat           # 216 KKLT basis indices (primal)
└── ...
```

### Key Equations
| Equation | Description |
|----------|-------------|
| eq. 2.18 | N_ab = κ̃_abc M^c |
| eq. 2.19 | p = N⁻¹K (flat direction) |
| eq. 2.22 | ζ = 1/(2^{3/2} π^{5/2}) |
| eq. 2.23 | W_flux = ζ Σ (M·q̃) N_q̃ Li₂(e^{2πiτ(q̃·p)}) |
| eq. 6.12 | e^{K₀} = (4/3 κ̃_abc p^a p^b p^c)⁻¹ |
| eq. 6.24 | V₀ = -3 e^{K₀} g_s^7/(4V[0])² W₀² |
| eq. 6.55 | Fluxes for 4-214-647 |
| eq. 6.56 | p = (293/110, 163/110, 163/110, 13/22) |
| eq. 6.59 | 2-term racetrack with coefficients -1, +512 |
| eq. 6.60 | e^{2π Im(τ)/110} = 528 |
| eq. 6.61 | W₀ = 80ζ × 528⁻³³ |
| eq. 6.63 | V₀ ≈ -5.5 × 10⁻²⁰³ |

---

## Notes

- **Basis alignment is the hardest problem**: CYTools uses different divisor basis than McAllister. This affects stages 2, 4, and 6.
- **9D vs 4D curves**: dual_curves.dat has 9 components (ambient), CYTools gives 4 (h¹¹ basis). Need GLSM projection.
- **Primal vs Dual**: Some data (kahler_param, basis.dat) is for primal (h¹¹=214), not dual (h¹¹=4).
