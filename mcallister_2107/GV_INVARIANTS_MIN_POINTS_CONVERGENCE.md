# GV Invariants: min_points Convergence Analysis

## Summary

CYTools' `compute_gvs()` has two parameters for controlling curve enumeration:

| Parameter | Description | Use Case |
|-----------|-------------|----------|
| `min_points` | Sample N lattice points, find curves organically | **Recommended** - fast, matches McAllister |
| `max_deg` | Exhaustively enumerate ALL curves up to degree D | Very slow, not what McAllister used |

**Key insight:** McAllister's "max degree 280" for 4-214-647 was the highest degree curve *found by sampling*, not an explicit cutoff. They used `min_points`-style sampling, which is why `max_deg=300` is orders of magnitude slower.

## Final Results (All 5 Examples)

With `min_points=20000` and Decimal precision fix:

| Example | McAllister Curves | Our Curves | Match | Time |
|---------|-------------------|------------|-------|------|
| 4-214-647 | 5177 | 10556 | 5177/5177 ✓ | ~16s |
| 5-113-4627-main | 1009 | 2194 | 1009/1009 ✓ | ~8s |
| 5-113-4627-alt | 1009 | 2194 | 1009/1009 ✓ | ~8s |
| 5-81-3213 | 557 | 1033 | 557/557 ✓ | ~5s |
| 7-51-13590 | 211 | 311 | 211/211 ✓ | ~3s |

**All 5 examples PASS.**

## Why 5-113-4627 Needed Higher min_points

Initially with `min_points=10000`, 5-113-4627 showed 845/1009 match. Investigation revealed:

| min_points | Our Curves | Matches | Missing |
|------------|------------|---------|---------|
| 10000 | 1147 | 845/1009 | 164 |
| 15000 | 1603 | 968/1009 | 41 |
| **20000** | **2194** | **1009/1009** | **0** |

McAllister's 1009 curves are a **subset** of what we compute at higher min_points. They simply used a different sampling depth. See `CURVE_DISCREPANCY.md` for full analysis.

## min_points vs max_deg: Critical Difference

```python
# FAST - sample N points, find curves organically
# This is what McAllister used
cy.compute_gvs(min_points=20000)  # ~16s for 4-214-647

# INTRACTABLE - exhaustively enumerate ALL curves up to degree D
# NOT what McAllister used - combinatorial explosion
cy.compute_gvs(max_deg=300)  # killed after 4+ min with no result
```

McAllister's data shows "max degree 280" for 4-214-647, but this is just the highest degree curve that *happened to be sampled*. They did NOT enumerate all 280-degree curves exhaustively.

**Always use `min_points`, never `max_deg`** (unless you have a specific reason and lots of time).

## Precision Issue (SOLVED)

Large GV invariants (~10^19) lose precision in float64. Use `Decimal`:

```python
from decimal import Decimal

# Loading McAllister's data (stored as scientific notation)
gv_values = [int(Decimal(x)) for x in content.strip().split(",")]

# Converting CYTools output
gv_value = int(Decimal(str(N_q)).to_integral_value())
```

## Recommendation

For the GA pipeline:
- Use **min_points=20000** as the default (needed for 5-113-4627)
- Use `Decimal` for all GV integer conversions
- Never use `max_deg` - it's exhaustive enumeration, not sampling

## Physics Note

Higher-degree curves are exponentially suppressed in W₀:

```
contribution ~ exp(-2π Im(τ) (q·p))
```

For Im(τ) ~ 110 (g_s ~ 0.009), curves with q·p > 2 contribute negligibly. The sampling approach naturally captures the physically relevant curves without exhaustive enumeration.
