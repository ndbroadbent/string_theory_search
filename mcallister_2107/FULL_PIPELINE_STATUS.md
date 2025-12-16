# full_pipeline.py Status

## Goal

End-to-end computation of V₀ (cosmological constant) from polytope + (K, M, orientifold), with NO .dat files except model inputs.

## Pipeline Steps

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| 1. Polytope & triangulation | (inline) | ✅ | Uses points.dat, heights.dat |
| 2. Intersection numbers κ | CYTools | ✅ | cy.intersection_numbers() |
| 3. GV invariants | compute_gv_invariants.py | ✅ | Uses min_points=20000 |
| 4. N_ab matrix | (inline) | ✅ | N_ab = κ_abc M^c |
| 5. Flat direction p | (inline) | ✅ | p = N⁻¹ K |
| 6. e^K₀ | (inline) | ✅ | (4/3) × (κ̃_abc p^a p^b p^c)⁻¹ |
| 7. Racetrack g_s, W₀ | compute_derived_racetrack.py | ✅ | Matches 1.0000 |
| 8. c_i values | compute_c_i.py | ✅ | 1 for D3, 6 for O7 |
| 9. χ(D_i) | compute_chi_divisor.py | ⚠️ | Formula works but basis mismatch |
| 10. Target τ | compute_target_tau.py | ✅ | τ = c_i/c_τ + χ/24 |
| 11. Solve for t | compute_kahler_param.py | ❌ | Solver diverging |
| 12. V_string | compute_V_string.py | ✅ | With BBHL correction |
| 13. V₀ | (inline) | ⏳ | Blocked by step 11 |

## Current Blocker

**Step 11: compute_kahler_param.py** is failing because:
1. Iterative Newton solver diverges
2. Jacobian may be ill-conditioned
3. basis.dat vs kklt_basis.dat mismatch causes confusion

## What's Validated

- Racetrack (g_s, W₀): **PASS** with ratio = 1.0000
- e^K₀: Computed value matches expected ~0.234
- GV invariants: All 5 examples pass
- V_string formula: Works when given correct t values

## Expected Final Result

For 4-214-647:
- V₀ = -5.5 × 10⁻²⁰³ Mpl⁴
- V_string ≈ 4711.83
- g_s ≈ 0.00911134
- W₀ ≈ 2.3 × 10⁻⁹⁰

## File Location

`mcallister_2107/2021_cytools/full_pipeline.py`

## Key Formula

```
V₀ = -3 × e^K₀ × (g_s⁷ / (4×V_string)²) × W₀²
```

## Recent Fixes Applied

1. **mpmath complex handling**: Use `mpc` not `mpf(1j)`
2. **Basis transformation**: Transform K, M from 2021 to latest basis when needed
3. **BBHL correction**: Included in V_string computation

## Test Command

```bash
cd /Users/ndbroadbent/code/string_theory
uv run python mcallister_2107/2021_cytools/full_pipeline.py
```

## Three-Layer Architecture

See `mcallister_2107/CLAUDE.md` for details:
- Layer 1: Pure compute_X() functions
- Layer 2: Individual tests (can load upstream .dat)
- Layer 3: full_pipeline.py (no .dat except model inputs)

The current focus is getting Layer 2 (compute_kahler_param.py) working before integrating into Layer 3.
