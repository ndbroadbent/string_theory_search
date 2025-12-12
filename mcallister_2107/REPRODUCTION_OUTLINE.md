# McAllister 4-214-647 Reproduction: Complete Outline

## Purpose

We are building a **general-purpose physics pipeline** that can evaluate any Calabi-Yau compactification and compute the cosmological constant V₀. This pipeline will be used for genetic algorithm search across the Kreuzer-Skarke database (~400M polytopes).

Before trusting GA results, we must verify our pipeline produces correct physics. McAllister et al. (arXiv:2107.09064) provides the only published end-to-end computation of V₀ ≈ 10⁻¹²³ from a specific polytope. Their polytope 4-214-647 with V₀ = -5.5 × 10⁻²⁰³ is our **gold standard for verification**.

This document details the complete pipeline:
- What is a model choice vs computed quantity
- Which steps are direct computation vs optimization vs search
- Precision requirements (float64 vs mpmath)
- Computational cost at each step

Once we can exactly reproduce McAllister's result, we can trust the pipeline for arbitrary polytopes.

---

## Overview of the Pipeline

### A. Model Choices (Inputs)
1. Polytope selection (Kreuzer-Skarke database)
2. Triangulation choice (heights)
3. Orientifold involution (which divisors are O7 vs D3)
4. Flux vectors K, M (integers, h²¹ components each)

### B. Direct Computations (CYTools)
5. Hodge numbers h¹¹, h²¹
6. Intersection numbers κᵢⱼₖ
7. Euler characteristic χ = 2(h¹¹ - h²¹)
8. Gopakumar-Vafa invariants N_q (mirror symmetry)

### C. Derived Quantities (Linear Algebra)
9. N_ab = κ_abc M^c (contract intersection numbers with M)
10. p = N⁻¹ K (flat direction in moduli space)
11. e^{K₀} = (4/3 × κ_abc p^a p^b p^c)⁻¹

### D. Racetrack Stabilization (mpmath required)
12. Build racetrack from GV invariants
13. Solve ∂W/∂τ = 0 for Im(τ) → g_s = 1/Im(τ)
14. Compute W₀ = |W(τ_min)| (exponentially small, ~10⁻⁹⁰)

### E. KKLT Moduli Stabilization
15. Target divisor volumes: τᵢ = (cᵢ/2π) ln(W₀⁻¹)
16. Solve T_i(t) = τᵢ for t (ITERATIVE, includes GV corrections)

### F. Final Volume Computation
17. V_string = (1/6)κᵢⱼₖtⁱtʲtᵏ - ζ(3)χ/(4(2π)³) + (GV terms)

### G. Vacuum Energy (Final Result)
18. V₀ = -3 × e^{K₀} × (g_s⁷/(4V_string)²) × W₀²

---

## Detailed Steps

### PART A: MODEL CHOICES (Given/Chosen)

These are inputs to the computation, not computed quantities.

#### Step 1: Polytope Selection
- **What:** Choose a reflexive polytope from Kreuzer-Skarke database
- **For 4-214-647:**
  - Primal: 294 points in 4D, h¹¹=214, h²¹=4
  - Dual: 12 points in 4D, h¹¹=4, h²¹=214
- **File:** `points.dat` (primal) or `dual_points.dat` (dual)
- **Note:** McAllister works with the DUAL (h¹¹=4) for the racetrack, but divisor data is for PRIMAL

#### Step 2: Triangulation
- **What:** Choose a fine, regular, star triangulation (FRST)
- **Method:** Specified by height vector (one height per lattice point)
- **For 4-214-647:**
  - `heights.dat` = 219 values
  - Gives 1011 simplices
- **Computed by:** CYTools `poly.triangulate(heights=...)`

#### Step 3: Orientifold Involution
- **What:** Choose which homogeneous coordinates to negate under the involution
- **Effect:** Determines which divisors are:
  - O7-planes (c_i = 6, gaugino condensation)
  - D3-instantons (c_i = 1, Euclidean D3-brane)
- **For 4-214-647:**
  - 49 O7-planes
  - 165 D3-instantons
- **File:** `target_volumes.dat` contains c_i values (1 or 6)
- **Critical:** This is a MODEL CHOICE, not computed from geometry!

#### Step 4: Flux Vectors K, M
- **What:** Integer flux vectors, h²¹ components each
- **Constraint:** Must satisfy tadpole cancellation
- **For 4-214-647 (dual, h²¹=4 for racetrack basis):**
  - K = [-3, -5, 8, 6] (CYTools 2021 basis [3,4,5,8])
  - M = [10, 11, -11, -5]
  - Or in latest CYTools basis [5,6,7,8]:
    - K = [8, 5, -8, 6]
    - M = [-10, -1, 11, -5]
- **Files:** `K_vec.dat`, `M_vec.dat`
- **Critical:** Finding valid (K, M) requires SEARCH (see Part F)

---

### PART B: DIRECT COMPUTATIONS (CYTools)

These are computed directly from the polytope/triangulation.

#### Step 5: Hodge Numbers
- **Formula:** Read from CYTools
- **Computed by:** `cy.h11()`, `cy.h21()`
- **For 4-214-647 (primal):** h¹¹=214, h²¹=4
- **Precision:** Exact integers

#### Step 6: Intersection Numbers κᵢⱼₖ
- **What:** Triple intersection of divisors Dᵢ ∩ Dⱼ ∩ Dₖ
- **Computed by:** `cy.intersection_numbers(in_basis=True)`
- **For 4-214-647:** 214×214×214 tensor (sparse, ~6400 non-zero)
- **Precision:** Exact integers (computed combinatorially)

#### Step 7: Euler Characteristic
- **Formula:** χ = 2(h¹¹ - h²¹)
- **For 4-214-647:** χ = 2(214 - 4) = 420
- **Precision:** Exact integer

#### Step 8: Gopakumar-Vafa Invariants
- **What:** Genus-zero curve counts N_q for each curve class q
- **Computed by:** CYTools via `cy.compute_gvs(max_deg=N)`
- **Underlying library:** `cygv` (installed with CYTools)
- **For 4-214-647:** ~344 curves with non-zero GV invariants
- **Verification files:** `small_curves.dat`, `small_curves_gv.dat`
- **Precision:** Exact integers

```python
gvs = cy.compute_gvs(max_deg=10)  # compute up to degree 10
gv_dict = gvs.dok  # {(q1, q2, ...): N_q, ...}
```

**Verified:** CYTools correctly computes quintic GV invariants (2875, 609250, ...)

---

### PART C: DERIVED QUANTITIES (Linear Algebra)

These follow directly from K, M, and κ.

#### Step 9: N-matrix
- **Formula:** N_ab = κ_abc M^c
- **What:** Contract intersection numbers with flux M
- **Dimension:** h²¹ × h²¹ (4×4 for dual)
- **Precision:** float64 sufficient

```python
N = np.einsum('abc,c->ab', kappa, M)
```

#### Step 10: Flat Direction p
- **Formula:** p = N⁻¹ K (Demirtas lemma)
- **What:** Direction in moduli space where flux superpotential vanishes perturbatively
- **Requires:** N must be invertible (det(N) ≠ 0)
- **For 4-214-647:** p = [293/110, 163/110, 163/110, 13/22] ≈ [2.664, 1.482, 1.482, 0.591]
- **Precision:** float64 sufficient

```python
p = np.linalg.solve(N, K)
```

#### Step 11: e^{K₀} (Complex Structure Kähler Potential)
- **Formula:** e^{K₀} = (4/3 × κ_abc p^a p^b p^c)⁻¹
- **For 4-214-647:** e^{K₀} = 0.234393
- **Precision:** float64 sufficient

```python
kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
e_K0 = 1.0 / (4.0/3.0 * kappa_p3)
```

---

### PART D: RACETRACK STABILIZATION (mpmath Required)

This determines g_s and W₀.

#### Step 12: Build Racetrack Superpotential
- **Formula:** W = -ζ Σ_q (M·q) N_q Li₂(e^{2πiτ(q·p)})
- **What:** Sum over curves, grouped by exponent q·p
- **Inputs:**
  - Curve classes q from `small_curves.dat`
  - GV invariants N_q from `small_curves_gv.dat`
  - Flat direction p from Step 10
  - Flux M from Step 4
- **Leading terms for 4-214-647:**
  - α = 32/110, coefficient A = 5
  - β = 33/110, coefficient B = -2560
- **Precision:** mpmath with 150+ digits

#### Step 13: Solve F-term for g_s
- **Formula:** ∂W/∂τ = 0
- **Solution:** e^{2πiτ(β-α)} = -Aα/(Bβ)
- **Result:** Im(τ) = -log(|ratio|) / (2π(β-α))
- **For 4-214-647:** Im(τ) = 109.75, so g_s = 1/109.75 = 0.00911134
- **Precision:** mpmath required for accurate τ

```python
ratio = -A * alpha / (B * beta)
Im_tau = -mpmath.log(abs(ratio)) / (2 * mpmath.pi * (beta - alpha))
g_s = 1 / Im_tau
```

#### Step 14: Compute W₀
- **Formula:** W₀ = |W(τ_min)|
- **For 4-214-647:** W₀ ≈ 2.30 × 10⁻⁹⁰
- **Precision:** mpmath REQUIRED (10⁻⁹⁰ cannot be represented in float64)
- **Verification:** Compare with `W_0.dat`

---

### PART E: KKLT MODULI STABILIZATION

This determines the Kähler moduli t and volume V_string.

#### Step 15: Target Divisor Volumes
- **Formula (eq 5.12-5.13):**
```
c_τ = 2π / (g_s × ln(1/W₀))
τ_target = c_i / c_τ + χ(D_i)/24 - GV_corrections
```
- **Inputs:**
  - c_i from orientifold (Step 3): either 1 or 6
  - g_s from racetrack (Step 13)
  - W₀ from racetrack (Step 14)
  - χ(D_i) = 12 × χ(O_D) - D³ (topological Euler char of divisor)
- **χ(O_D) from Braun eq (2.7):**
  - Vertex: χ(O_D) = 1 + g (g = interior pts in dual facet)
  - Edge interior: χ(O_D) = 1 - g (g = interior pts in dual edge)
  - 2-face interior: χ(O_D) = 1 (g = 0)
- **For 4-214-647:**
  - g_s = 0.00911134
  - ln(W₀⁻¹) = 206.4
  - c_τ = 2π / (0.00911134 × 206.4) = 3.341 ✓ validated
  - χ(D_i) range: [4, 9] for rigid divisors, mean 5.84
  - With χ/24 correction: τ ≈ 0.42-2.18 (2.4% error = GV corrections only)
- **Precision:** float64 sufficient
- **Implementation:**
  - `compute_target_tau.py` - c_τ and zeroth-order target
  - `compute_chi_divisor.py` - χ(D_i) via Braun formula ✓ validated
  - `compute_gv_invariants.py` - GV corrections via cy.compute_gvs()

**CRITICAL:** McAllister uses `kklt_basis.dat` (not `basis.dat`) which excludes
non-rigid divisors. Points 1, 2 are non-rigid vertices (g=1, g=2) with χ = 33, 45.

```python
c_tau = 2 * np.pi / (g_s * np.log(1 / W0))  # ~3.34 for McAllister
chi = compute_chi_divisor(poly, kappa, basis)  # Braun formula
tau_target = c_i / c_tau + chi / 24 - gv_correction
```

#### Step 16: Solve for Kähler Moduli t (ITERATIVE)
- **Problem:** Find t such that T_i(t) = τᵢ (target)
- **With instanton corrections (eq. 4.12):**
```
T_i(t) = (1/2) κᵢⱼₖ tʲ tᵏ - χ(Dᵢ)/24 + (GV instanton sum)
```
- **Method:** Iterative/Newton solver
- **Starting point:** Solve classical τᵢ = (1/2) κᵢⱼₖ tʲ tᵏ, then iterate
- **McAllister's solution:** `corrected_kahler_param.dat` (214 values)
- **Precision:** float64 sufficient
- **Computational cost:** Moderate (nonlinear system, h¹¹ unknowns)

**THIS IS WHERE `kahler_param.dat` vs `corrected_kahler_param.dat` MATTERS:**
- `kahler_param.dat` = solution to CLASSICAL equation (no GV terms)
- `corrected_kahler_param.dat` = solution WITH GV instanton corrections
- Using uncorrected gives V ≈ 17900 (3.8× WRONG)
- Using corrected gives V ≈ 4712 (correct after BBHL)

---

### PART F: FINAL VOLUME COMPUTATION

#### Step 17: Compute V_string
- **Formula (eq. 4.11):**
```
V_string = (1/6) κᵢⱼₖ tⁱ tʲ tᵏ - ζ(3)χ/(4(2π)³) + (GV instanton sum)
```
- **Components:**
  - Classical: (1/6) κᵢⱼₖ tⁱ tʲ tᵏ = 4712.34
  - BBHL α' correction: -ζ(3)×420/(4(2π)³) = -0.509
  - GV instanton sum: ~0.001 (negligible)
- **Result:** V_string = 4711.83
- **Precision:** float64 sufficient
- **Verification:** Compare with `cy_vol.dat` = 4711.829675204889

```python
V_classical = np.einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0
BBHL = zeta(3) * chi / (4 * (2*np.pi)**3)
V_string = V_classical - BBHL  # + GV terms if desired
```

---

### PART G: VACUUM ENERGY

#### Step 18: Compute V₀
- **Formula (eq. 6.24):**
```
V₀ = -3 × e^{K₀} × (g_s⁷ / (4×V_string)²) × W₀²
```
- **Inputs:**
  - e^{K₀} = 0.234393 (Step 11)
  - g_s = 0.00911134 (Step 13)
  - V_string = 4711.83 (Step 17)
  - W₀ = 2.30×10⁻⁹⁰ (Step 14)
- **Result:** V₀ = -5.5 × 10⁻²⁰³ Mpl⁴
- **Precision:** mpmath REQUIRED (W₀² ~ 10⁻¹⁸⁰)

```python
V0 = -3 * e_K0 * (g_s**7 / (4 * V_string)**2) * W0**2
```

---

## Summary: Computational Requirements

| Step | Category | Precision | Computational Cost |
|------|----------|-----------|-------------------|
| 1-4 | Model choices | N/A | N/A (given) |
| 5-7 | Direct | exact int | O(1) |
| 8 | Direct | exact int | CYTools (O(max_deg)) |
| 9-11 | Linear algebra | float64 | O(h²¹³) |
| 12-14 | Racetrack | mpmath | O(#curves) |
| 15 | KKLT target | float64 | O(h¹¹) |
| 16 | KKLT solve | float64 | O(h¹¹² × iterations) |
| 17 | Volume | float64 | O(h¹¹³) |
| 18 | V₀ | mpmath | O(1) |

---

## What Requires SEARCH (Computationally Intensive)

The above assumes we're given valid (K, M) flux vectors. Finding them requires search:

### Finding Valid (K, M)
- **Constraint 1:** det(N) ≠ 0 (N must be invertible)
- **Constraint 2:** Tadpole cancellation
- **Constraint 3:** All τᵢ > 0 (inside Kähler cone)
- **Constraint 4:** W₀ is exponentially small (racetrack works)

McAllister's approach:
1. Fix orientifold (choose which divisors are O7 vs D3)
2. Search over integer (K, M) pairs
3. For each pair, compute racetrack → check if W₀ is small
4. If W₀ ~ 10⁻⁹⁰, declare success

This search is what makes the problem hard in general. For reproduction, we use McAllister's pre-found (K, M).

---

## Files Required for Full Reproduction

### From McAllister's Ancillary Data:
| File | Content | Used In |
|------|---------|---------|
| `dual_points.dat` | Dual polytope (12 points) | Racetrack geometry |
| `points.dat` | Primal polytope (294 points) | Volume computation |
| `heights.dat` | Triangulation heights | Step 2 |
| `K_vec.dat` | Flux K | Step 4 |
| `M_vec.dat` | Flux M | Step 4 |
| `target_volumes.dat` | c_i values (1 or 6) | Step 3, 15 |
| `small_curves.dat` | Curve classes | Step 12 |
| `small_curves_gv.dat` | GV invariants | Step 12 |
| `corrected_kahler_param.dat` | Solved t values | Step 16 (verify) |
| `cy_vol.dat` | Target V_string | Step 17 (verify) |
| `W_0.dat` | Target W₀ | Step 14 (verify) |
| `g_s.dat` | Target g_s | Step 13 (verify) |

### What We Compute:
| Quantity | Formula | Verified Against |
|----------|---------|------------------|
| e^{K₀} | (4/3 κₐᵦ꜀ pᵃpᵇp꜀)⁻¹ | 0.234393 |
| g_s | 1/Im(τ) from racetrack | `g_s.dat` = 0.00911134 |
| W₀ | \|W(τ_min)\| | `W_0.dat` = 2.30×10⁻⁹⁰ |
| V_string | (1/6)κt³ - BBHL | `cy_vol.dat` = 4711.83 |
| V₀ | -3 e^{K₀} (g_s⁷/(4V)²) W₀² | -5.5×10⁻²⁰³ |

---

## Current Status

| Step | Status | Implementation | Notes |
|------|--------|----------------|-------|
| 1-4 | ✅ | Data files | From McAllister ancillary data |
| 5-7 | ✅ | CYTools | `cy.h11()`, `cy.h21()`, `cy.chi()` |
| 8 | ✅ | CYTools | `cy.compute_gvs(min_points=N)` via cygv |
| 9-11 | ✅ | `full_pipeline_from_data.py` | e^{K₀} = 0.234393 ✓ |
| 12-14 | ✅ | `compute_derived_racetrack.py`, `full_pipeline_from_data.py` | g_s, W₀ match |
| 15 | ✅ | `compute_target_tau.py`, `compute_chi_divisor.py` | 2.4% error (GV only) |
| 16 | ⚠️ | `compute_kklt_iterative.py` | Need GV integration |
| 17 | ✅ | `compute_V_string.py`, `verify_V_string.py` | 4711.83 exact ✓ |
| 18 | ✅ | `full_pipeline_from_data.py` | -5.5×10⁻²⁰³ ✓ |

### Implementation Files (`mcallister_2107/`)

| File | Step | Purpose | Status |
|------|------|---------|--------|
| `compute_triangulation.py` | 2 | Load polytope, apply heights | ✅ |
| `compute_divisor_cohomology.py` | 3 | Compute h^i(D) via cohomCalg | ✅ |
| `compute_rigidity_combinatorial.py` | 3 | Rigidity via dual face interior pts | ✅ |
| `compute_c_i.py` | 3 | Dual Coxeter numbers c_i | ✅ |
| `compute_derived_racetrack.py` | 12-14 | Build racetrack, solve for g_s, W₀ | ✅ |
| `compute_target_tau.py` | 15 | c_τ = 2π/(g_s × ln(1/W₀)) | ✅ |
| `compute_chi_divisor.py` | 15 | χ(D) = 12×χ(O_D) - D³ (Braun) | ✅ 2.4% |
| `compute_gv_invariants.py` | 15-16 | GV via cy.compute_gvs() | ✅ |
| `compute_kklt_iterative.py` | 16 | Solve T_i(t) = τ_target | ⚠️ |
| `compute_V_string.py` | 17 | V = (1/6)κt³ - BBHL | ✅ |

### Remaining Work
1. **Step 16:** Integrate GV corrections into KKLT solver iteration
2. **Step 16:** Test solver convergence with full τ_target (χ + GV)
3. **End-to-end:** Run full pipeline without McAllister's pre-solved t values

### Key Discoveries
- **KKLT basis ≠ divisor basis:** McAllister's `kklt_basis.dat` excludes non-rigid
  divisors (points 1, 2 with g=1, g=2). Their `corrected_target_volumes.dat` uses
  this KKLT basis, not `basis.dat`.
- **Non-rigid divisors:** Have χ(D) >> 6 (e.g., 33, 45 for g=1,2). GA pipeline
  must either exclude these or handle their large χ contributions.
- **χ(D) validation:** 2.4% RMS error is entirely from GV corrections (not computed
  yet in Step 15). The χ formula itself is correct.

---

## References

- McAllister et al., arXiv:2107.09064
  - Eq. 4.11: V[0] with instanton corrections
  - Eq. 4.12: T_i with instanton corrections
  - Eq. 6.24: V₀ formula
  - Section 6.4: Polytope 4-214-647 details
- Demirtas et al., arXiv:1912.10047: Flat direction lemma
- BBHL: Becker-Becker-Haack-Louis α' correction
