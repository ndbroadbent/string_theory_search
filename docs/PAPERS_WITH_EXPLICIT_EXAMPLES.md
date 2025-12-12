# Papers with Explicit Flux Vacuum Examples

These papers contain explicit numerical data (polytope, fluxes, W₀, g_s, V_AdS) that can be used as test cases for our pipeline.

---

## 1. McAllister et al. 2107.09064 (PRIMARY - Already Using)

**Title:** Small Cosmological Constants in String Theory
**Authors:** Demirtas, Long, McAllister, Stillman
**arXiv:** [2107.09064](https://arxiv.org/abs/2107.09064)
**PDF:** `resources/small_cc_2107.09064.pdf`
**Ancillary Data:** `resources/small_cc_2107.09064_source/anc/paper_data/`

### Explicit Example: Polytope 4-214-647
| Parameter | Value |
|-----------|-------|
| h¹¹ | 214 |
| h²¹ | 4 |
| K (flux, 2021 basis) | [-3, -5, 8, 6] |
| M (flux, 2021 basis) | [10, 11, -11, -5] |
| g_s | 0.00911134 |
| W₀ | 2.30 × 10⁻⁹⁰ |
| V_string | 4711.83 |
| e^{K₀} | 0.234393 |
| V₀ | -5.5 × 10⁻²⁰³ Mpl⁴ |

**Data Files:**
- `points.dat` - Primal polytope (294 points)
- `dual_points.dat` - Dual polytope (12 points)
- `heights.dat` - Triangulation heights
- `K_vec.dat`, `M_vec.dat` - Flux vectors
- `g_s.dat`, `W_0.dat`, `cy_vol.dat` - Physics results
- `corrected_kahler_param.dat` - Kähler moduli solution

**Status:** ✅ Data downloaded, pipeline validation in progress

---

## 2. Demirtas et al. 1912.10047

**Title:** Vacua with Small Flux Superpotential
**Authors:** Demirtas, Kim, McAllister, Moritz
**arXiv:** [1912.10047](https://arxiv.org/abs/1912.10047)
**Published:** Phys. Rev. Lett. 124 (2020) 211603
**PDF:** `resources/demirtas_small_W0_1912.10047.pdf` (TO DOWNLOAD)

### Explicit Example
| Parameter | Value |
|-----------|-------|
| h¹¹ | 2 |
| h²¹ | 272 |
| W₀ | ≈ 2 × 10⁻⁸ |
| Regime | Large complex structure, weak coupling |

**Key Method:** Perturbatively flat flux vacua (PFFV) - find fluxes where W₀ vanishes at leading order, then compute corrections.

**Ancillary Data:** No ancillary folder (only .tex and .bbl in source)

**Status:** ⚠️ Need to extract flux vectors and full data from PDF

**Why Important:** Same authors as McAllister - validates method at different h¹¹

---

## 3. CICY Flat Flux Paper 2201.10581

**Title:** Systematics of perturbatively flat flux vacua for CICYs
**Authors:** Marchesano, Prieto, Quirant
**arXiv:** [2201.10581](https://arxiv.org/abs/2201.10581)
**Published:** JHEP 08 (2022) 297
**PDF:** `resources/cicy_flat_flux_2201.10581.pdf` (TO DOWNLOAD)

### Explicit Examples
- Studies ALL 36 pCICYs (projective Complete Intersection CYs) with h¹¹=2
- Found examples with W₀ ≈ 2 × 10⁻²⁷
- K3-fibered vs non-K3-fibered classification

| Parameter | Value |
|-----------|-------|
| h¹¹ | 2 |
| W₀ | ≈ 2.0769 × 10⁻²⁷ (best case) |
| Geometry | Complete Intersection CY (not KS polytope) |

**Status:** ⚠️ Need to download and extract specific examples

**Why Important:** Different geometry class (CICY vs toric hypersurface), tests generality

---

## 4. All Flux Vacua Explicit 1212.4530

**Title:** Finding all flux vacua in an explicit example
**Authors:** Martinez-Pedrera, Mehta, Rummel, Westphal
**arXiv:** [1212.4530](https://arxiv.org/abs/1212.4530)
**Published:** JHEP 06 (2013) 110
**PDF:** `resources/all_flux_vacua_explicit_1212.4530.pdf`

### Explicit Geometry
**Manifold:** CP⁴₁₁₁₆₉ - degree 18 hypersurface in weighted projective space P⁴(1,1,1,6,9)
Also known as the "Swiss cheese" manifold, standard example for LVS.

| Parameter | Value |
|-----------|-------|
| h¹¹ | 2 |
| h²¹ | 272 |
| Fluxes | 6 three-cycles (2 complex structure moduli preserved by discrete symmetry) |

### Key Feature
- **Complete enumeration** of ALL supersymmetric flux vacua for this CY
- Uses polynomial homotopy continuation method
- Found 1,374 explicit vacua with flux data
- Studies N_vac as function of D3 tadpole L: N_vac ≃ 0.02 L^1.83

**Note:** Same geometry as arXiv:1912.10047 (Demirtas et al.)

**Status:** ✅ PDF downloaded, geometry identified

**Why Important:** Complete enumeration - can test pipeline against ALL vacua, not just one

---

## 5. Large Volume Scenario 0805.1029

**Title:** General Analysis of LARGE Volume Scenarios with String Loop Moduli Stabilisation
**Authors:** Cicoli, Conlon, Quevedo
**arXiv:** [0805.1029](https://arxiv.org/abs/0805.1029)
**Published:** JHEP 10 (2008) 105
**PDF:** `resources/large_volume_scenario_0805.1029.pdf` (TO DOWNLOAD)

### Key Feature
- Different stabilization mechanism (LVS vs KKLT)
- Non-supersymmetric AdS minimum at exponentially large volume
- Studies K3 fibrations and other CY classes

**Status:** ⚠️ Need to check for explicit numerical examples

**Why Important:** Tests pipeline with LVS scenario (different physics)

---

## 6. Coexisting Flux Vacua 2507.00615 (VERY RECENT - July 2025)

**Title:** Coexisting Flux String Vacua from Numerical Kähler Moduli Stabilisation
**Authors:** (Multiple)
**arXiv:** [2507.00615](https://arxiv.org/abs/2507.00615)
**PDF:** `resources/coexisting_flux_vacua_2507.00615.pdf` (TO DOWNLOAD)

### Key Features
- Studies >80,000 Calabi-Yau threefolds with h¹¹ ≤ 6
- Scans over g_s values
- Finds explicit realizations of KKLT, Kähler uplift, and other scenarios
- Uses Kreuzer-Skarke polytopes
- Uses database from Crino et al. for O3/O7 orientifolds

### EXPLICIT EXAMPLE (eq. 52 in paper)
**Polytope vertices (4D, 7 vertices as columns):**
```
[[ 1, -3, -2, -2,  0,  0,  1],
 [ 0, -1, -1,  0,  0,  1,  1],
 [ 0, -1,  0, -1,  1,  0,  1],
 [ 0, -2,  0,  0,  0,  0,  2]]
```
**Height vector:** (-1, 0, 3, 3, 0, 0, 0, 0)

| Parameter | Value |
|-----------|-------|
| h¹¹ | 3 |
| χ | -112 |
| A_i | 1 (all) |
| a_i | 2π/22 |

**Scanned ranges:**
- g_s ∈ [10⁻²·⁵, 10⁰·⁵] (50 values)
- |W₀| ∈ [10⁻¹⁰, 10¹] (100 values)

**Source files:** `resources/coexisting_flux_2507_source/`

**Status:** ✅ Explicit polytope extracted, ready for testing

**Why Important:** Large-scale systematic study with modern methods, explicit polytope given

---

## Download Commands

```bash
cd /Users/ndbroadbent/code/string_theory/resources

# Paper 2: Demirtas small W0
curl -L "https://arxiv.org/pdf/1912.10047.pdf" -o demirtas_small_W0_1912.10047.pdf
pdftotext demirtas_small_W0_1912.10047.pdf demirtas_small_W0_1912.10047.txt

# Paper 3: CICY flat flux
curl -L "https://arxiv.org/pdf/2201.10581.pdf" -o cicy_flat_flux_2201.10581.pdf
pdftotext cicy_flat_flux_2201.10581.pdf cicy_flat_flux_2201.10581.txt

# Paper 4: All flux vacua
curl -L "https://arxiv.org/pdf/1212.4530.pdf" -o all_flux_vacua_explicit_1212.4530.pdf
pdftotext all_flux_vacua_explicit_1212.4530.pdf all_flux_vacua_explicit_1212.4530.txt

# Paper 5: Large Volume Scenario
curl -L "https://arxiv.org/pdf/0805.1029.pdf" -o large_volume_scenario_0805.1029.pdf
pdftotext large_volume_scenario_0805.1029.pdf large_volume_scenario_0805.1029.txt

# Paper 6: Coexisting flux vacua (2025)
curl -L "https://arxiv.org/pdf/2507.00615.pdf" -o coexisting_flux_vacua_2507.00615.pdf
pdftotext coexisting_flux_vacua_2507.00615.pdf coexisting_flux_vacua_2507.00615.txt

# Check for ancillary data
for id in 1912.10047 2201.10581 1212.4530 2507.00615; do
  echo "=== $id ==="
  curl -sL "https://arxiv.org/e-print/$id" -o /tmp/$id.tar.gz
  tar -tzf /tmp/$id.tar.gz 2>/dev/null | grep -E "anc|data|\.dat|\.csv" || echo "No ancillary data"
done
```

---

## Priority for Pipeline Validation

1. **2107.09064** (McAllister) - Already have data, highest h¹¹ ✅
2. **1912.10047** (Demirtas) - Same authors, h¹¹=2, simpler geometry
3. **2507.00615** (2025) - Most recent, large-scale, likely has code/data
4. **1212.4530** (All vacua) - Complete enumeration for one CY
5. **2201.10581** (CICY) - Different geometry class

---

## What We Need From Each Paper

For each test case, extract:
1. **Polytope/Geometry specification** (KS ID, CICY matrix, or vertex list)
2. **Hodge numbers** (h¹¹, h²¹)
3. **Triangulation** (heights or simplices if non-default)
4. **Orientifold data** (O7/D3 divisor indices, or c_i values)
5. **Flux vectors** (K, M integers)
6. **Physics results** (g_s, W₀, V_string, V₀)

This gives us ground truth to validate the pipeline produces correct physics.
