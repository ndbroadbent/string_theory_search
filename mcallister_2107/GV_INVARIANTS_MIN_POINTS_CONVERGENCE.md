# GV Invariants: min_points Convergence Analysis

## Summary

For polytope 4-214-647 (h11=4, h21=214), we tested CYTools' `compute_gvs(min_points=N)` at various values to determine the minimum setting needed to match McAllister's published curve data.

## Final Result (with Decimal fix)

```
min_points=10000: 5177 curves, 3.58s
Validation: 5177/5177 match, 0 mismatch, 0 missing
SUCCESS: All GV invariants match McAllister's data!
```

## Convergence Testing Results

Initial testing (before precision fix) showed apparent mismatches:

| min_points | Curves Computed | Time (s) | Match | Mismatch | Missing |
|------------|-----------------|----------|-------|----------|---------|
| 5000       | 2765            | 1.24     | 2361  | 404      | 2412    |
| **10000**  | **5177**        | **3.49** | 3781  | 1396     | 0       |
| 15000      | 8932            | 9.70     | 3781  | 1396     | 0       |
| 20000      | 10556           | 13.90    | 3781  | 1396     | 0       |

## Key Finding

**min_points=10000 produces exactly 5177 curves** - the same count as McAllister's `dual_curves.dat`.

The 1396 "mismatches" were precision errors, not computational differences. After applying the Decimal fix: **100% match**.

## Precision Issue (SOLVED)

The mismatches were due to floating-point precision loss when converting large GV invariants:

```
computed: 38512679141944848024
expected: 38512679141944844288
difference: 3736 (on 10^19 magnitude = ~10^-16 relative error)
```

This is exactly the precision limit of 64-bit floating point. The fix is to use `Decimal` for exact integer conversion:

```python
from decimal import Decimal

# Loading McAllister's data (stored as scientific notation)
gv_values = [int(Decimal(x)) for x in content.strip().split(",")]

# Converting CYTools output
gv_value = int(Decimal(str(N_q)).to_integral_value())
```

## Recommendation

For the GA pipeline:
- Use **min_points=10000** as the default
- Use `Decimal` for all GV integer conversions (already implemented in benchmark script)

## Physics Note

Higher-degree curves (computed at larger min_points) are exponentially suppressed in W₀:

```
contribution ~ exp(-2π Im(τ) (q·p))
```

For Im(τ) ~ 110 (g_s ~ 0.009), curves with q·p > 1 contribute negligibly. The 5177 curves from min_points=10000 capture all physically relevant contributions.
