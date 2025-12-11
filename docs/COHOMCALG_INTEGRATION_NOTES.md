# cohomCalg Integration Notes

## Status
- cohomCalg compiles and runs on macOS (vendor/cohomCalg/)
- Works for simple examples (dP1, Quintic)
- **Fails for h11=4 Altman polytopes** - "infinite domain" error in PolyLib

## Key Learnings

### 1. CYTools Point Ordering
CYTools `poly.points()` returns points in a different order than Altman database:
- Index 0 is always the **origin** (0,0,0,0)
- Remaining indices don't match Altman's D1, D2, ... ordering
- Must create explicit mapping between orderings

```python
# Example for POLYID=1001
altman_to_cytools = {0: 7, 1: 1, 2: 2, 3: 5, 4: 3, 5: 6, 6: 4, 7: 0}
```

### 2. GLSM Matrix Issues
CYTools `poly.glsm_linear_relations()` returns shape `(h11+1, n_points)`:
- Includes origin as a column (column 0)
- Has h11+1 rows (one redundant relation)
- For cohomCalg, need exactly h11 rows and exclude origin column

```python
glsm_full = poly.glsm_linear_relations()  # shape (5, 9) for h11=4
glsm = glsm_full[:, 1:]  # Remove origin column -> (5, 8)
# Then reduce to 4 independent rows using QR decomposition
```

### 3. Altman Database Structure
Located in `data/toriccy/h11_X/`:

| File | Key Fields |
|------|------------|
| `*.poly.json` | POLYID, DVERTS (vertices), NVERTS |
| `*.triang.json` | POLYID, TRIANGN, SRIDEAL, DIVCOHOM |
| `*.invol.json` | INVOLDIVCOHOM (only involution divisors) |
| `*.geom.json` | KAHLERMAT, MORIMAT |

**DIVCOHOM format**: `"{{h0,h1,h2,h11},{h0,h1,h2,h11},...}"`

**SRIDEAL format**: `"{D1*D8,D2*D3,D4*D5,D6*D7}"` (1-indexed)

### 4. cohomCalg Input Format
```
vertex u1 = ( x, y, z, w ) | GLSM: ( q1, q2, q3, q4 );
...
srideal [u1*u2, u3*u4*u5];
monomialfile off;
ambientcohom O( d1, d2, d3, d4 );
```

- Ambient dimension = n_coords - n_glsm_charges
- For 4D ambient with 8 coords, need exactly 4 GLSM charges

### 5. The "infinite domain" Error
```
count_points: ? infinite domain
ERROR: INTERNAL: Invalid number of rational functions computed.
```

This occurs when PolyLib's lattice point counting encounters an unbounded polytope. Possible causes:
1. GLSM charges don't define a proper fan/cone structure
2. Line bundle degree gives unbounded solution space
3. Mismatch between ambient dimension and GLSM rank

### 6. What Works vs What Doesn't

**Works:**
- dP1 (2D ambient, 4 coords, 2 GLSM charges)
- Quintic ambient P^4 (4D ambient, 5 coords, 1 GLSM charge)
- Altman database lookup for precomputed DIVCOHOM

**Doesn't work yet:**
- h11=4 CY3 with 8 vertices and 4 GLSM charges
- Need to debug GLSM charge extraction from CYTools

## Next Steps

1. **Debug GLSM extraction**: The reduced 4x8 GLSM might not define a valid fan
2. **Check CYTools methods**: May need `glsm_charge_matrix()` or similar instead of `glsm_linear_relations()`
3. **Verify with known working example**: Find a CY3 that cohomCalg handles correctly
4. **Consider alternative**: For h11 ≤ 6, just use Altman database (validated, fast)

## Code Location
- cohomCalg wrapper: `mcallister_2107/compute_divisor_cohomology.py`
- Binary: `vendor/cohomCalg/bin/cohomcalg`
- Altman data: `data/toriccy/h11_4/`

## References
- cohomCalg paper: arXiv:1003.5217
- Altman database paper: arXiv:2111.03078
- CYTools paper: arXiv:2211.03823
