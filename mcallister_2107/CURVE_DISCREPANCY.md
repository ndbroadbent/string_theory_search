# GV Invariant Curve Discrepancy: 5-113-4627

## Summary

When validating GV invariant computation against McAllister's 5 examples (arXiv:2107.09064), **3/5 pass perfectly** but **5-113-4627 shows a discrepancy**:

| Example | Result | Our Curves | McAllister Curves | Match |
|---------|--------|------------|-------------------|-------|
| 4-214-647 | **PASS** | 5177 | 5177 | 5177/5177 |
| 5-113-4627-main | FAIL | 1147 | 1009 | 845/1009 |
| 5-113-4627-alternative | FAIL | 1147 | 1009 | 845/1009 |
| 5-81-3213 | **PASS** | 557 | 557 | 557/557 |
| 7-51-13590 | **PASS** | 211 | 211 | 211/211 |

## The Discrepancy

For 5-113-4627:
- We compute **1147 curves** with `min_points=10000`
- McAllister has **1009 curves** in `dual_curves.dat`
- **845 curves match exactly** (both coordinates and GV values)
- **164 McAllister curves are "missing"** from our computation
- **302 of our curves** aren't in McAllister's data

## Key Insight: 845 Matches Proves Basis is Correct

The 845 exact matches prove:
1. Our `curve_basis_mat` transformation is correct
2. The GV computation itself is working
3. The Decimal precision handling is correct

If the basis transformation were wrong, we'd have **0 matches**, not 845.

## Analysis of Missing Curves

### In Ambient Coordinates (10-dimensional)

First 5 missing curves:
```
(-15, 7, 2, 1, 0, 0, 4, -1, 1, 1) -> GV=6561
(-15, 6, -1, 4, 0, 2, 1, -2, 2, 3) -> GV=-36000
(-15, 6, -2, 5, 0, 3, 0, -2, 2, 3) -> GV=-164
(-12, 6, 1, 4, 0, 1, 0, -3, 3, 0) -> GV=23337
(-15, 7, 0, 3, 0, 2, 2, -1, 1, 1) -> GV=189918
```

### In Basis Coordinates (5-dimensional)

Same curves converted to h11=5 basis:
```
(7, 2, 1, -1, 1) -> GV=6561
(6, -1, 4, -2, 2) -> GV=-36000
(6, -2, 5, -2, 2) -> GV=-164
(6, 1, 4, -3, 3) -> GV=23337
(7, 0, 3, -1, 1) -> GV=189918
```

### Curves with Same GV Exist at Different Coordinates

Some missing curves have matching GV values in our data, but at different coordinates:
```
Missing: (-15, 6, -1, 4, 0, 2, 1, -2, 2, 3)  GV=-36000
Ours:    (-12, 6, 1, 3, 0, 1, 1, -2, 2, 0)   GV=-36000
Diff:    (3, 0, 2, -1, 0, -1, 0, 0, 0, -3)
```

The differences are **NOT** in the span of GLSM linear relations, so these are genuinely different curve classes, not just different representations.

## Possible Explanations

### 1. Different GV Enumeration Algorithm (Most Likely)

The `cygv` library (used by latest CYTools) may enumerate curves differently than whatever McAllister used in 2021. The algorithm samples lattice points and computes GV invariants - different sampling or cutoff criteria could produce different sets.

### 2. Different Degree Cutoff

We compute 1147 curves, they have 1009. Perhaps:
- McAllister had a maximum degree cutoff we don't know about
- Different `min_points` settings produce different curve sets

### 3. Bug in McAllister's Data (Unlikely)

- McAllister is a top string theorist; this is his expertise
- CYTools was written by their group
- Paper has 50+ citations - others have used this data
- 3/5 examples pass perfectly - systematic bug would affect all

## Resolution: Superset Hypothesis CONFIRMED

**Test:** Compute with increasing `min_points` and check if we find all McAllister curves.

| min_points | Our Curves | Matches | Missing |
|------------|------------|---------|---------|
| 10000 | 1147 | 845/1009 | 164 |
| 15000 | 1603 | 968/1009 | 41 |
| **20000** | **2194** | **1009/1009** | **0** |

**Result:** At `min_points=20000`, we find ALL 1009 McAllister curves plus 1185 additional higher-degree curves.

**Conclusion:** McAllister computed GV invariants with some cutoff (likely `min_points` equivalent around 10000-15000). Their 1009 curves are a proper subset of what `cygv` computes. There is **no bug** - just different enumeration depth.

## Physics Impact

The key question: **Do the 164 missing curves affect W₀?**

High-degree curves are exponentially suppressed by:
```
exp(-2π Im(τ) q·p)
```

For Im(τ) ~ 110 (g_s ~ 0.009), curves with large q·p contribute negligibly. If the missing curves are high-degree, they may not affect the physics at all.

## Next Steps

1. Test superset hypothesis: compute with `min_points=20000` or higher
2. Compute W₀ using only the 845 matching curves
3. Compare to McAllister's expected W₀ value
4. If W₀ matches, the missing curves don't matter for physics

## Files

- `debug_missing_curves.py`: Analysis script for this discrepancy
- `2021_cytools/compute_gv_invariants.py`: Main GV computation with validation
- `2021_cytools/benchmark_gv_invariant_computation.py`: Single-example benchmark
