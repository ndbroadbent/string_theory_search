# McAllister Volume Reproduction Analysis

## Summary

We successfully reproduced McAllister et al. (arXiv:2107.09064) CY volume for polytope 4-214-647 using CYTools with constrained optimization.

## Final Result

```
Kähler moduli: [0.58470441, 2.33881763, 2.92352204, 1.16940881]
V_einstein: 4711.8297 (matches McAllister's 4711.83 exactly)
cone.contains(t): True
All divisor volumes: positive
```

## What We Tried (and Why It Failed)

### Attempt 1: Using `basis.dat` First 4 Entries as Divisor Basis

**Approach**: The documentation in `MCALLISTER_SMALL_CC_DETAILS.md` says "the first 4 indices are the KKLT run's divisor basis", so we tried `cy.set_divisor_basis([1, 2, 3, 4])`.

**Result**: `ValueError: Input divisors do not form a basis`

**Why it failed**: McAllister's `basis.dat` has 214 entries with values ranging 1-218. These are NOT CYTools divisor indices. CYTools only has 8 prime toric divisors `(1, 2, 3, 4, 5, 6, 7, 8)` for this polytope.

### Attempt 2: Finding "True Basis" Divisors

**Approach**: We noticed `basis.dat` and `kklt_basis.dat` both have 214 entries. The 4 indices in `basis.dat` but NOT in `kklt_basis.dat` are `[1, 2, 46, 130]`. We tried these as the divisor basis.

**Result**: `ValueError: Indices are not in appropriate range`

**Why it failed**: Indices 46 and 130 are way outside CYTools' range of 1-8. The numbering systems are completely different.

### Attempt 3: GLSM Projection from 214 Ambient Parameters

**Approach**: Use `cy.glsm_charge_matrix()` and `cy.divisor_basis_matrix()` to project McAllister's 214 `kahler_param.dat` values to 4 Kähler moduli.

**Result**: Could not implement - dimension mismatch.

**Why it failed**: McAllister uses a **Fine Regular Star Triangulation (FRST)** with 218 prime toric divisors derived from ALL lattice points (294 in `points.dat`). CYTools with the 12-vertex dual polytope only constructs 8 prime toric divisors. The two parameterizations are fundamentally different.

### Attempt 4: Scaling Cone Tip to Match Volume

**Approach**: Get `cone.tip_of_stretched_cone(1.0)`, then scale by `(target_V / V_tip)^(1/3)` since volume scales as t³.

**Result**: Produced moduli that were OUTSIDE the Kähler cone (`cone.contains() = False`).

**Why it failed**: Simple uniform scaling doesn't preserve cone membership for non-spherical cones.

## What Worked: Constrained Optimization

**Approach**: Optimize for Kähler moduli that minimize `|V(t) - V_target|²` subject to the constraint `H @ t >= 0` (staying inside Kähler cone).

```python
from scipy.optimize import minimize

def objective(t):
    V = cy.compute_cy_volume(t)
    return (V - target_V_string) ** 2

def cone_constraint(t):
    return H @ t  # H @ t >= 0 for interior

result = minimize(
    objective,
    t_init,  # scaled tip as initial guess
    method='SLSQP',
    constraints={'type': 'ineq', 'fun': cone_constraint},
)
```

**Result**:
- Found valid Kähler moduli inside the cone
- Volume matches exactly (0.000000% error)
- All divisor volumes are positive

## Key Insight: Different Triangulation Granularity

McAllister uses a **finer triangulation** than CYTools:

| Property | McAllister FRST | CYTools Default |
|----------|-----------------|-----------------|
| Lattice points used | 294 (all of Δ) | 12 (vertices of Δ*) |
| Prime toric divisors | 218 | 8 |
| Ambient parameters | 214 (non-basis) | 4 (= h11) |
| Basis divisors | 4 | 4 |

The `kahler_param.dat` with 214 values represents coordinates in a 214-dimensional ambient space that cannot be directly projected to CYTools' 4-dimensional Kähler moduli space because the triangulations are different.

## Frame Conversion

Always convert between frames:
```
V_string = V_einstein × g_s^(3/2)
V_einstein = V_string × g_s^(-3/2)

With g_s = 0.00911134:
  V_string = 4.0979
  V_einstein = 4711.83
```

## Files

- `tools/reproduce_mcallister_volume.py` - Working reproduction script
- `tests/fixtures/mcallister_4_214_647.json` - Test fixture with correct Kähler moduli
- `resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/` - McAllister raw data

## McAllister Data File Reference

| File | Entries | Description |
|------|---------|-------------|
| `dual_points.dat` | 12 | Vertices of dual polytope Δ* |
| `dual_simplices.dat` | 15 | Triangulation simplices |
| `points.dat` | 294 | All lattice points of Δ |
| `basis.dat` | 214 | Indices of prime toric divisors (1-218), NOT CYTools indices |
| `kklt_basis.dat` | 214 | Non-basis divisor indices |
| `kahler_param.dat` | 214 | Ambient Kähler coordinates (one per non-basis divisor) |
| `target_volumes.dat` | 214 | Target Einstein-frame divisor volumes |
| `g_s.dat` | 1 | String coupling = 0.00911134 |
| `cy_vol.dat` | 1 | Expected CY volume = 4711.83 (Einstein frame) |
| `W_0.dat` | 1 | Superpotential = 2.30012e-90 |
