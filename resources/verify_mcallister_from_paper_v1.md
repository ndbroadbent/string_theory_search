# Full Pipeline: Verify McAllister W₀ Computation

**Goal**: Compute W₀ = 2.3×10⁻⁹⁰ from first principles using McAllister's polytope 4-214-647.

**Reference**: McAllister et al. arXiv:2107.09064, Demirtas et al. arXiv:1912.10047

---

## What We Know Works

Based on prior research (see `COMPUTING_PERIODS.md`, `MCALLISTER_REPRODUCTION_v2.md`):

| Component | Status | Notes |
|-----------|--------|-------|
| GV invariants | ✓ Verified | CYTools/cygv matches McAllister (252, -9252, 848628...) |
| V₀ formula | ✓ Verified | V₀ = -3 e^K W₀² gives -5.5e-203 with McAllister's W₀ |
| Curve basis | ✓ Understood | 9D→4D is linear via GLSM, GVs match exactly |
| Intersection numbers | ✓ Available | CYTools computes κ_ijk for h¹¹=4 dual |

## What We Can Compute (but haven't implemented yet)

| Component | Status | How |
|-----------|--------|-----|
| Prepotential F(z) | ✓ Possible | F = F_poly + F_inst from GV invariants via Li₃ sum |
| Periods Π(z) | ✓ Possible | Π = (z^A, ∂F/∂z^A) - derivatives of prepotential |
| W₀ | ✓ Possible | Full racetrack algorithm (Phases 3-6 below) |

**Tools available**:
- **cygv**: Computes GV invariants → used to build F_inst
- **lefschetz-family**: Numerical periods via Picard-Lefschetz (sanity check)
- **mpmath**: Arbitrary precision for W₀ ~ 10⁻⁹⁰

**What's NOT implemented yet**:
- The full pipeline in Python (this pseudocode IS that implementation spec)

**This document IS the implementation spec** for computing W₀ from first principles.

---

## Inputs (from McAllister's data files)

```
DATA_DIR = resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/

Geometry (DUAL polytope - the h¹¹=4 side):
  - dual_points.dat      → Polytope vertices (12 points)
  - dual_simplices.dat   → Triangulation (15 simplices)
  - dual_curves.dat      → Curve classes (5177 × 9 components, ambient basis)
  - dual_curves_gv.dat   → GV invariants (5177 values)

Fluxes (4-dimensional, h²¹=4 on mirror):
  - K_vec.dat            → RR flux: [-3, -5, 8, 6]
  - M_vec.dat            → NSNS flux: [10, 11, -11, -5]

Published results:
  - W_0.dat              → 2.30012e-90
  - g_s.dat              → 0.00911134
  - c_tau.dat            → 3.34109 (1/g_s × log(1/W₀) / 2π)
  - cy_vol.dat           → 4711.83 (Einstein frame)

Primal polytope data (h¹¹=214 side - NOT directly usable with dual):
  - points.dat           → 294 lattice points
  - basis.dat            → 214 divisor indices (for PRIMAL, not dual!)
  - kahler_param.dat     → 214 ambient coordinates (for PRIMAL)
```

**CRITICAL**: `basis.dat` and `kahler_param.dat` are for the PRIMAL (h¹¹=214) polytope.
They CANNOT be used with the dual (h¹¹=4) CY in CYTools.
See `MCALLISTER_BASIS_DAT.md` for full explanation.

---

## Algorithm Pseudocode

### Phase 1: Load Geometry

```
FUNCTION load_geometry(dual_points_file, dual_simplices_file):
    """
    Load the DUAL polytope (h¹¹=4) with McAllister's exact triangulation.
    """
    vertices = read_csv(dual_points_file)         # 12 points
    simplices = read_csv(dual_simplices_file)     # 15 simplices

    # Use EXACT triangulation from McAllister
    polytope = Polytope(vertices)
    triangulation = polytope.triangulate(simplices=simplices)
    cy = triangulation.get_cy()

    # Verify
    ASSERT cy.h11() == 4
    ASSERT cy.h21() == 214

    # Get intersection numbers (4×4×4 tensor)
    kappa = cy.intersection_numbers()

    RETURN {cy, kappa}
```

### Phase 2: Load GV Invariants (Use McAllister's Precomputed)

```
FUNCTION load_gv_invariants(curves_file, gv_file):
    """
    Load McAllister's precomputed GV invariants.

    NOTE: The 9-component curve classes need to be projected to 4D.
    However, for computing W₀ we can use CYTools' 4D GVs directly
    since they are mathematically equivalent.
    """
    # Option A: Use McAllister's data directly
    curves_9d = read_matrix(curves_file)   # (5177, 9)
    gv_values = read_vector(gv_file)       # 5177 values

    gv_dict_9d = {}
    FOR i in range(len(gv_values)):
        gv_dict_9d[tuple(curves_9d[i])] = gv_values[i]

    RETURN gv_dict_9d

    # Option B: Compute via CYTools (equivalent, in 4D basis)
    # gvs_4d = cy.compute_gvs(min_points=100, format='dok')
    # RETURN gvs_4d
```

### Phase 3: Verify Perturbatively Flat Conditions

The fluxes (M, K) must satisfy specific Diophantine constraints for a perturbatively flat vacuum to exist.

```
FUNCTION verify_flat_conditions(M, K, kappa):
    """
    Check that (M, K) satisfy perturbatively flat vacuum conditions.

    From Demirtas 1912.10047 "Lemma":
    1. N_ab = κ_abc M^c must be invertible
    2. p = N^{-1} K must lie in Kähler cone (all p^a > 0)
    3. K · p = 0

    Args:
        M: NSNS flux vector [10, 11, -11, -5]
        K: RR flux vector [-3, -5, 8, 6]
        kappa: Triple intersection numbers κ_abc (4×4×4)
    """
    n = len(M)  # h²¹ = 4

    # Compute N_ab = Σ_c κ_abc M^c
    N = np.einsum('abc,c->ab', kappa, M)

    # Check invertibility
    det_N = np.linalg.det(N)
    IF abs(det_N) < 1e-10:
        RETURN FAIL("N is singular - fluxes don't satisfy condition 1")

    N_inv = np.linalg.inv(N)

    # Compute flat direction p = N^{-1} K
    p = N_inv @ K

    # Check Kähler cone (simplified: all positive)
    IF np.any(p <= 0):
        RETURN FAIL(f"p = {p} not in Kähler cone")

    # Check orthogonality K · p = 0
    K_dot_p = K @ p
    IF abs(K_dot_p) > 1e-10:
        RETURN FAIL(f"K·p = {K_dot_p} ≠ 0")

    RETURN SUCCESS(p=p, N_inv=N_inv)
```

### Phase 4: Identify Racetrack Curve Pairs

```
FUNCTION find_racetrack_pairs(p, M, gv_dict):
    """
    Find curve pairs (q₁, q₂) that give racetrack stabilization.

    The effective superpotential along z = pτ is:
    W_eff(τ) = ζ Σ_q (M·q) N_q Li₂(e^{2πiτ p·q})

    At large Im(τ), dominated by terms with smallest p·q.
    Need two competing terms for racetrack minimum.

    Conditions (McAllister eq. 478-484):
    (d) p·q₁ < 1 and p·q₂ < 1
    (e) 0 < ε = p·(q₂ - q₁) < 1
    (f) These terms dominate
    """
    # Filter curves with 0 < p·q < 1 and M·q ≠ 0
    candidates = []
    FOR (q_tuple, N_q) in gv_dict.items():
        IF N_q == 0:
            CONTINUE

        q = np.array(q_tuple[:4])  # Use first 4 components if 9D
        p_dot_q = p @ q
        M_dot_q = M @ q

        IF 0 < p_dot_q < 1 AND M_dot_q != 0:
            candidates.append({
                'q': q, 'p_dot_q': p_dot_q,
                'M_dot_q': M_dot_q, 'N_q': N_q,
                'coeff': M_dot_q * N_q * p_dot_q
            })

    # Sort by p·q (smallest = most dominant)
    candidates.sort(key=lambda x: x['p_dot_q'])

    # Find valid pairs with small ε = p·(q₂ - q₁)
    pairs = []
    FOR i, c1 in enumerate(candidates[:20]):
        FOR c2 in candidates[i+1:20]:
            epsilon = c2['p_dot_q'] - c1['p_dot_q']
            IF 0 < epsilon < 1:
                delta = abs(c1['coeff']) / abs(c2['coeff'])
                IF delta > 1:
                    delta = 1/delta
                    c1, c2 = c2, c1

                pairs.append({
                    'q1': c1['q'], 'q2': c2['q'],
                    'epsilon': epsilon, 'delta': delta,
                    'p_q1': c1['p_dot_q'], 'p_q2': c2['p_dot_q']
                })

    RETURN pairs
```

### Phase 5: Build Effective Superpotential (High Precision)

```
FUNCTION W_effective(tau, p, M, gv_dict):
    """
    W_eff(τ) = ζ Σ_q (M·q) N_q Li₂(e^{2πiτ p·q})

    where ζ = 1/(2^{3/2} π^{5/2}) ≈ 0.0179

    CRITICAL: Use arbitrary precision (mpmath) since W₀ ~ 10^{-90}
    Standard float64 cannot represent this.
    """
    from mpmath import mp, mpf, mpc, polylog, pi, exp
    mp.dps = 150  # 150 decimal places

    zeta = mpf(1) / (mpf(2)**mpf('1.5') * pi**mpf('2.5'))
    two_pi_i = mpc(0, 2) * pi

    W = mpc(0)
    FOR (q_tuple, N_q) in gv_dict.items():
        IF N_q == 0:
            CONTINUE

        q = [mpf(x) for x in q_tuple[:4]]
        M_dot_q = sum(mpf(M[i]) * q[i] for i in range(4))
        p_dot_q = sum(mpf(p[i]) * q[i] for i in range(4))

        IF M_dot_q == 0 OR p_dot_q <= 0:
            CONTINUE

        exp_arg = exp(two_pi_i * tau * p_dot_q)
        IF abs(exp_arg) < mpf('0.99'):
            W += M_dot_q * N_q * polylog(2, exp_arg)

    RETURN zeta * W
```

### Phase 6: Solve F-term Equation

```
FUNCTION solve_vacuum(p, M, gv_dict, tau_range=(10, 500)):
    """
    Solve ∂W_eff/∂τ = 0 for vacuum expectation value ⟨τ⟩.

    At minimum: Im(τ) = 1/g_s >> 1 (weak coupling)

    For McAllister: g_s = 0.00911 → Im(τ) ≈ 110
    """
    from mpmath import mp, mpc, log, pi, exp
    mp.dps = 150

    FUNCTION dW_dtau(im_tau):
        """∂W/∂τ at τ = i × im_tau"""
        tau = mpc(0, im_tau)

        dW = mpc(0)
        FOR (q_tuple, N_q) in gv_dict.items():
            # ... (similar loop structure)
            # ∂/∂τ Li₂(e^{2πiτ p·q}) = 2πi (p·q) × Li₁ = -2πi (p·q) × ln(1-e^{...})
            dW += M_dot_q * N_q * p_dot_q * (-log(1 - exp_arg))

        RETURN dW

    # Binary search for zero crossing on imaginary axis
    lo, hi = tau_range
    WHILE hi - lo > 1e-10:
        mid = (lo + hi) / 2
        val = dW_dtau(mid).imag
        IF val > 0:
            lo = mid
        ELSE:
            hi = mid

    im_tau_vev = (lo + hi) / 2
    tau_vev = mpc(0, im_tau_vev)

    W0 = abs(W_effective(tau_vev, p, M, gv_dict))
    g_s = 1 / im_tau_vev

    RETURN {tau: tau_vev, W0: W0, g_s: g_s}
```

### Phase 7: Compute V₀(AdS)

```
FUNCTION compute_V0_AdS(W0, g_s, V_CY):
    """
    V₀ = -3 e^K |W₀|²

    At KKLT minimum with large volume:
    e^K ≈ g_s / (8 V²)

    From McAllister eq. 682-686.
    """
    from mpmath import mpf

    e_K = mpf(g_s) / (8 * mpf(V_CY)**2)
    V0 = -3 * e_K * mpf(W0)**2

    RETURN V0
```

---

## Main Pipeline

```
FUNCTION main():
    print("McAllister W₀ Full Pipeline Verification")

    # Phase 1: Load geometry with EXACT triangulation
    geo = load_geometry(
        DATA_DIR / "dual_points.dat",
        DATA_DIR / "dual_simplices.dat"
    )
    ASSERT geo.cy.h11() == 4

    # Phase 2: Load fluxes
    K = [-3, -5, 8, 6]   # from K_vec.dat
    M = [10, 11, -11, -5] # from M_vec.dat

    # Phase 3: Load GV invariants (use CYTools 4D basis)
    gv_dict = geo.cy.compute_gvs(min_points=100, format='dok')

    # Phase 4: Verify perturbatively flat conditions
    result = verify_flat_conditions(M, K, geo.kappa)
    IF result.failed:
        print(f"FAIL: {result.message}")
        RETURN
    p = result.p

    # Phase 5: Find racetrack pairs
    pairs = find_racetrack_pairs(p, M, gv_dict)

    # Phase 6: Solve for vacuum (HIGH PRECISION)
    vacuum = solve_vacuum(p, M, gv_dict)

    # Phase 7: Compute V₀
    V_CY = 4711.83  # from cy_vol.dat
    V0 = compute_V0_AdS(vacuum.W0, vacuum.g_s, V_CY)

    # Compare with published values
    print(f"Computed W₀: {vacuum.W0}")
    print(f"Expected W₀: 2.30012e-90")
    print(f"Computed g_s: {vacuum.g_s}")
    print(f"Expected g_s: 0.00911134")
```

---

## Critical Requirements

### 1. Arbitrary Precision Arithmetic

W₀ ~ 10⁻⁹⁰ requires 100+ decimal places. Use `mpmath`:

```python
from mpmath import mp
mp.dps = 150  # 150 decimal places
```

### 2. Correct Geometry

Must use DUAL polytope (`dual_points.dat`) with McAllister's exact triangulation (`dual_simplices.dat`).

**DO NOT** use `basis.dat` or `kahler_param.dat` with the dual CY - these are for the PRIMAL.

### 3. GV Invariant Basis

CYTools gives 4D curve classes. McAllister uses 9D ambient. The GV values are identical - just different representations of the same curves.

### 4. Frame Conversion

CYTools volumes are STRING frame. McAllister reports EINSTEIN frame.
```
V_Einstein = V_String × g_s^{-3/2}
```

---

## Dependencies

```
mpmath          # Arbitrary precision (pip install mpmath)
numpy           # Arrays
cytools         # CY geometry (vendor/cytools_latest)
```

Note: cygv is only needed if computing GV invariants from scratch. We can use McAllister's precomputed values or CYTools' `compute_gvs()`.

---

## Expected Output

```
McAllister W₀ Full Pipeline Verification

Phase 1: Geometry loaded (h¹¹=4, h²¹=214)
Phase 2: Fluxes K=[-3,-5,8,6], M=[10,11,-11,-5]
Phase 3: 5177 GV invariants loaded
Phase 4: Perturbatively flat conditions satisfied
         p = [0.xxx, 0.xxx, 0.xxx, 0.xxx]
         K·p = 0.0
Phase 5: Found 3 racetrack pairs
         Best: ε = 0.xxx, δ = 1.23e-45
Phase 6: Vacuum solved
         ⟨τ⟩ = 109.74i
         g_s = 0.00911134
         W₀ = 2.30e-90
Phase 7: V₀(AdS) = -5.5e-203

Comparison:
  W₀:  computed=2.30e-90, expected=2.30e-90 ✓
  g_s: computed=0.00911,  expected=0.00911  ✓
  V₀:  computed=-5.5e-203

SUCCESS: Full pipeline reproduces McAllister!
```

---

## References

- `COMPUTING_PERIODS.md` - Full algorithm and multi-objective fitness
- `NUMERICAL_OPTIMIZATION_OVER_COMPLEX_STRUCTURE.md` - Detailed phase implementations
- `MCALLISTER_REPRODUCTION_v2.md` - Why optimization approaches fail
- `MCALLISTER_BASIS_DAT.md` - basis.dat is for PRIMAL, not dual
- McAllister et al. arXiv:2107.09064 Section 5
- Demirtas et al. arXiv:1912.10047 (The "Lemma")
