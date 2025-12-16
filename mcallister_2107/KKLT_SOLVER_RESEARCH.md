# KKLT Solver Research Notes

## The Problem

We need to solve for Kähler moduli `t` from the KKLT stabilization equation:
```
τ_i = (1/2) κ_ijk t^j t^k
```
where τ_target = c_i/c_τ + χ(D_i)/24.

**The challenge**: Multiple solution branches exist (τ→t is many-to-one). Different branches give different V_string values - some positive (correct), some negative (wrong).

## Key Discoveries

### 1. McAllister's Two-File System

McAllister provides TWO sets of Kähler parameters:
- `kahler_param.dat` (uncorrected): Satisfies **τ = c_i exactly**
- `corrected_kahler_param.dat`: Satisfies τ = c_i/c_τ + χ/24 (target τ)

**Verified for 4-214-647**:
```
t_uncorrected → τ = c_i (ratio = 1.0000, std = 0.0012)
t_corrected → τ = c_i/c_τ + χ/24 (our target)
```

### 2. The Correct Branch

Starting from `t_uncorrected` and path-following to target τ gives:
- **V_classical = 4712.4** (CORRECT! matches expected 4712.3)
- **t correlation = 1.0000** with expected

Starting from uniform t gives:
- **V_classical = 5458** or negative (WRONG branch)
- **t has negative components** (t range: [-19, 203])

### 3. V at Different Points

| Starting point | τ achieved | V_classical |
|----------------|------------|-------------|
| t_uncorrected | c_i | 17901 |
| t_corrected (McAllister) | target τ | 4712 |
| Uniform t → target τ | target τ | 5458 (wrong branch) |
| t_uncorrected → target τ | target τ | 4712 (correct!) |

### 4. The Solution Space Structure

The equation τ(t) = τ_target has multiple solutions because:
- τ_i = (1/2) κ_ijk t^j t^k is quadratic in t
- κ has rank ~65 (not full rank 214), creating ~149-dim nullspace
- Different branches correspond to different "basins" in t-space

### 5. Initialization Strategies Tested

| Strategy | Converges? | V_classical | Notes |
|----------|------------|-------------|-------|
| Uniform t=10 | Yes | -25405 | **WRONG BRANCH** |
| t ∝ c_i pattern | No | - | Diverges |
| t_uncorrected * 0.1 | Yes | 17901 | **CORRECT BRANCH** |
| max(t_uncorrected, 1) | Yes | 17934 | **CORRECT BRANCH** |
| Random positive | No | - | Diverges |
| Sorted like t_uncorrected | No | - | Diverges |

### 6. Key Insight: What Makes t_uncorrected Special?

McAllister's `t_uncorrected` is the solution to the SIMPLER equation:
```
τ = c_i (just the dual Coxeter numbers)
```
NOT τ = c_i/c_τ + χ/24.

This simpler equation has the SAME correct branch structure. Once on the correct branch, path-following to the full target τ stays on that branch.

### 7. The Jacobian Rank Problem

At uniform t=10:
- Jacobian rank: 65/214
- Condition number: 5.4e+11

At t_uncorrected:
- Jacobian rank: 214/214
- Condition number: 3.1e+03

**The correct branch has a well-conditioned Jacobian!**

## Current Status

### What Works
- `compute_kahler_param.py` passes 5/5 when given McAllister's τ_expected
- Path-following from t_uncorrected to target τ gives correct V

### What Doesn't Work
- `full_pipeline.py` fails because we can't find the correct branch without t_uncorrected
- Starting from uniform t leads to wrong branch

## Open Questions

1. **How to find the correct branch without cheating?**
   - Can we use the Kähler cone generators?
   - Is there a geometric criterion to identify the correct branch?
   - Can we detect we're on the wrong branch (negative V) and try again?

2. **What's special about McAllister's t_uncorrected geometrically?**
   - Is it related to the tip of the Kähler cone?
   - Is there a physical interpretation?

3. **Can we compute t_uncorrected from scratch?**
   - It satisfies τ = c_i (simpler equation)
   - But we still need to find the right branch for THAT equation

## McAllister's Algorithm (Section 5.2)

**CRITICAL FINDING**: McAllister's algorithm is in the paper Section 5.2!

### The Algorithm

1. **Pick random h_init in secondary fan** (NOT random t!)
   - Use `poly.random_triangulations_fast()` to get valid FRSTs
   - Heights naturally correspond to points in the Kähler cone

2. **Convert heights → t_init**
   - Heights give a valid triangulation
   - The triangulation defines κ_ijk
   - t_init comes from the triangulation point

3. **Path-follow from τ(t_init) → τ_target**
   - Linear system at each step: `κ_ijk t^j ε^k = Δτ_i`
   - Stays in Kähler cone throughout

4. **Check V > 0**
   - If V < 0, wrong branch - try another random h_init
   - If V > 0, found correct branch

### Key Quote from Paper

> "We start by picking a random point h_init in the subset of the secondary fan of FRSTs,
> which we denote by G. Such a point is naturally associated to a point in the extended
> Kähler cone, t_init, with basis divisor volumes τ_init."

### Why Random t Doesn't Work

- Random t may not correspond to ANY valid triangulation
- The Kähler cone has complex geometry
- Starting outside it leads to wrong branches

### What We Tested

1. **Random heights (wrong size)**: Heights are 219-dim, not 294-dim ❌
2. **Random heights (correct size)**: "Triangulation is non-fine or non-star" ❌
3. **`random_triangulations_fast()`**: Works! But gives DIFFERENT κ per triangulation
4. **Random t with fixed triangulation**: Mostly doesn't converge, or wrong V

### The Remaining Problem

Even with McAllister's triangulation (from `corrected_heights.dat`), starting from
uniform t doesn't find the correct branch. The issue is:

- McAllister's heights define the triangulation AND the initial t
- We're using their heights for triangulation but random t for initialization
- Need to understand how heights → t conversion works

### CYTools Functions

- `poly.random_triangulations_fast(N=k)`: Generate k random FRSTs
- `poly.triangulate(heights=h)`: Triangulate with specific heights
- Heights are 219-dim for 4-214-647 (not num_points=294)
- 219 = h11 + 5 = 214 + 5 (why +5 not +4?)

## Cross-Reference: docs/TORIC_GEOMETRY.md

**IMPORTANT**: See `docs/TORIC_GEOMETRY.md` section "KKLT Solver Implementation (VALIDATED)"

Key findings documented there:
1. **`heights_to_kahler()` doesn't work for 4-214-647** - gives negative correlation (-0.61)!
2. **Extended Kähler cone allows negative t** - 19/214 t values are negative in solution
3. **Starting from scaled `kahler_param.dat` converges; uniform t doesn't**

For new polytopes, documented options:
1. Random sampling in extended Kähler cone, take best by V
2. Tip of stretched cone (expensive for large h11)
3. Heights-based (test correlation first - may not work)

## Conclusion: The Practical Reality

**For validation against McAllister:**
- Use `kahler_param.dat` as t_init (this is NOT cheating - it's the correct branch)
- Path-follow to target τ
- This validates our path-following algorithm works

**For new polytopes in the GA:**
- Multi-start with V > 0 filter is the only option
- Try ~100 random t_init, keep the one with V > 0
- This is slow but necessary

**Why other approaches fail:**
- `heights_to_kahler()` - negative correlation for 4-214-647
- `tip_of_stretched_cone()` - O(2^h11) complexity, unusable for h11 > 20
- Random heights - don't give valid FRSTs
- `random_triangulations_fast()` - gives different κ per triangulation

**The fundamental issue:**
The τ(t) = τ_target equation has MANY solutions. Finding the physical one (V > 0)
requires either:
1. A good starting point (McAllister's data)
2. Brute force search (slow but works)

There is no known efficient algorithm to find the correct branch from scratch.

## Files

- `compute_kahler_param.py`: Main solver (path-following + Newton-Raphson fallback)
- `full_pipeline.py`: End-to-end V₀ computation (currently failing)
- `debug_full_pipeline.py`, `debug_full_pipeline_v2.py`: Debug scripts

## Formula Reference

```
τ_i = (1/2) κ_ijk t^j t^k                    (divisor volumes)
c_τ = 2π / (g_s × ln(1/W₀))                  (KKLT constant)
τ_target = c_i/c_τ + χ(D_i)/24               (target τ with corrections)
V_string = (1/6) κ_ijk t^i t^j t^k - BBHL    (CY volume)
V₀ = -3 e^{K₀} (g_s⁷/(4V)²) W₀²             (cosmological constant)
```

## Data Files

For 4-214-647:
- h11 = 214, h21 = 4
- c_i: 165 ones (D3), 49 sixes (O7)
- c_τ = 3.3411
- V_string expected = 4712
