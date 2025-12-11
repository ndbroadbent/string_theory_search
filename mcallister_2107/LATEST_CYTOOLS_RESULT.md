# McAllister 4-214-647 Configuration for Latest CYTools

## Summary

We have successfully ported the McAllister et al. (arXiv:2107.09064) vacuum configuration from CYTools 2021 to the latest CYTools. This gives us a **validated ground truth** for our GA pipeline.

## The Key Insight

K and M fluxes transform **differently** under a basis change:

| Quantity | Index Type | Transformation |
|----------|------------|----------------|
| **K** (flux) | covariant | `K_new = T⁻¹ @ K_old` |
| **M** (flux) | contravariant | `M_new = T.T @ M_old` |
| **p** (moduli) | contravariant | `p_new = T.T @ p_old` |

This is because K and M sit on opposite sides of the F-term equation `N·p = K`.

## Transformation Matrix

From CYTools 2021 basis `[3,4,5,8]` to latest basis `[5,6,7,8]`:

```python
T = np.array([
    [-1,  1,  0,  0],  # D3 = -D5 + D6
    [ 1, -1,  1,  0],  # D4 = D5 - D6 + D7
    [ 1,  0,  0,  0],  # D5 = D5
    [ 0,  0,  0,  1],  # D8 = D8
])
# det(T) = 1
```

## Validated Configuration

### For Latest CYTools (basis [5,6,7,8])

```python
K = [8, 5, -8, 6]
M = [-10, -1, 11, -5]
```

### Original McAllister (basis [3,4,5,8])

```python
K = [-3, -5, 8, 6]
M = [10, 11, -11, -5]
```

## Physics (Invariant Under Transformation)

| Quantity | Value |
|----------|-------|
| e^{K₀} | 0.234393 |
| g_s | 0.00911134 |
| W₀ | 2.30 × 10⁻⁹⁰ |
| V_CY (string frame) | 4711.83 |
| V₀ (cosmological constant) | -5.5 × 10⁻²⁰³ Mpl⁴ |

## Verification Script

Run `mcallister_2107/transform_km_to_new_cytools_basis.py` to verify:

```bash
cd /path/to/string_theory
uv run python mcallister_2107/transform_km_to_new_cytools_basis.py
```

Output confirms:
- p transforms correctly via T.T
- e^{K₀} is identical in both bases (0.234393)
- Physics is preserved

## What This Means for the GA

1. **Ground Truth**: We now have a validated (K, M) pair that reproduces McAllister's result in latest CYTools
2. **Pipeline Validation**: Any physics pipeline we build can be tested against this known-good configuration
3. **Basis Independence**: The GA can work in any basis as long as we transform fluxes correctly

## Files

- `transform_km_to_new_cytools_basis.py` - Clean script using known T, verifies physics
- `LATEST_CYTOOLS_CONVERSION_CORRECTED.md` - Detailed derivation of correct transformation rules
- `find_transformation.py` - Brute-force search (slow, but found T)
- `find_transformation_v3.py` - Port script with search (fixed bug)
- `search_km.py` - Random (K, M) search that also found valid configurations

## The Bug That Was Fixed

The original scripts had:
```python
# WRONG - treats both as same index type
K_new = T @ K_old
M_new = T @ M_old
```

Fixed to:
```python
# CORRECT - respects covariant/contravariant distinction
K_new = T_inv @ K_old  # covariant
M_new = T.T @ M_old    # contravariant
```

This single fix made e^{K₀} match between bases (0.234 vs the wrong 1.027).
