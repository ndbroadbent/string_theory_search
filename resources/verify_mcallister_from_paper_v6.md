# Verify McAllister Full Pipeline v6 - Basis Alignment Issue

**Status**: BLOCKED - CYTools divisor basis does not match McAllister's flux basis

**Reference Implementation**: `verify_mcallister_full_pipeline_v5.py`

---

## The Problem

When running `verify_mcallister_full_pipeline_v5.py`, Phase 1 fails:

```
======================================================================
McAllister W₀ Reproduction (v5)
Polytope: 4-214-647 (dual with h¹¹=4, h²¹=214)
======================================================================

[Phase 1] Loading geometry and verifying basis alignment...
Simplex indices: min=0, max=8
Note: Simplices already 0-indexed
✓ Loaded CY: h¹¹=4, h²¹=214
det(N) = 0.000000

ValueError: N is singular - fluxes don't satisfy invertibility condition
```

The root cause: **CYTools returns intersection numbers in a 9-divisor ambient toric basis (indices 0-8), while McAllister's fluxes K and M are in a 4-dimensional h¹¹ basis.**

---

## Detailed Diagnosis

### CYTools Intersection Numbers (Raw)

When using `cy.intersection_numbers()` (default, NOT `in_basis=True`):
- Returns 83 entries with indices ranging 0-7 (8 divisors)
- These are ambient toric divisors, not the 4D CY basis

```python
# Sample entries:
κ_{0,0,0} = -1512
κ_{0,0,1} = 504
κ_{0,0,2} = 756
...
```

When contracted with M = [10, 11, -11, -5] (a 4D vector):
```
N matrix (wrong basis):
[[-18072.   6024.   9036.    396.]
 [  6024.  -2008.  -3012.   -132.]
 [  9036.  -3012.  -4518.   -198.]
 [   396.   -132.   -198.     66.]]

det(N) = 0.0  ← SINGULAR
```

### CYTools Intersection Numbers (in_basis=True)

When using `cy.intersection_numbers(in_basis=True)`:
- Returns 7 entries in the reduced h¹¹=4 basis
- CYTools chooses divisor basis [5, 6, 7, 8] (ambient indices)

```python
# All 7 entries:
κ_(0, 0, 0) = -1
κ_(0, 2, 3) = 1
κ_(0, 3, 3) = -2
κ_(1, 2, 3) = 1
κ_(1, 3, 3) = -2
κ_(2, 3, 3) = -2
κ_(3, 3, 3) = 8
```

When contracted with M = [10, 11, -11, -5]:
```
N matrix (CYTools basis):
[[-10.   0.  -5.  -1.]
 [  0.   0.  -5.  -1.]
 [ -5.  -5.   0.  31.]
 [ -1.  -1.  31. -60.]]

det(N) = -18100  ← Non-singular, good!

p = N⁻¹ K = [-0.20, 0.88, 0.93, 0.37]
Expected p = [2.66, 1.48, 1.48, 0.59]  ← MISMATCH
```

**Conclusion**: CYTools uses a different divisor basis than McAllister. The fluxes K and M must be transformed to match.

---

## McAllister's Data Files

### dual_points.dat (12 vertices of dual polytope)
```
0,0,0,0
-1,2,-1,-1
1,-1,0,0
-1,-1,1,1
-1,-1,1,2
-1,-1,2,1
-1,-1,2,3
-1,-1,3,2
-1,-1,2,2
-1,0,1,1
-1,1,0,0
0,-1,1,1
```

### dual_simplices.dat (15 simplices - McAllister's triangulation)
```
0,1,2,3,4
0,1,2,3,5
0,1,2,4,6
0,1,2,5,7
0,1,2,6,7
0,1,3,4,5
0,1,4,5,8
0,1,4,6,8
0,1,5,7,8
0,1,6,7,8
0,2,3,4,5
0,2,4,5,8
0,2,4,6,8
0,2,5,7,8
0,2,6,7,8
```

Note: Simplices use indices 0-8 (only 9 of the 12 points).

### K_vec.dat (flux vector K)
```
-3,-5,8,6
```

### M_vec.dat (flux vector M)
```
10,11,-11,-5
```

### g_s.dat
```
0.00911134
```

### W_0.dat
```
2.30012e-90
```

### cy_vol.dat
```
4711.829675204889
```

---

## What We Need to Determine

### Option A: Find the basis transformation

McAllister's basis and CYTools' basis [5,6,7,8] are related by some GL(4,Z) transformation T:

```
K_cytools = T @ K_mcallister
M_cytools = T @ M_mcallister
```

To find T, we need to:
1. Identify what basis McAllister uses (ambient indices? which ones?)
2. Compute the transformation matrix between bases

### Option B: Use McAllister's intersection numbers directly

If we can extract the intersection tensor κ̃_abc in McAllister's basis from:
- Their supplementary code
- Computing from the paper's explicit formulas
- Reverse-engineering from the known p vector

Then we bypass CYTools for intersection numbers entirely.

### Option C: Verify the expected p vector computes correctly

From eq. 6.56, McAllister gives:
```
p = (293/110, 163/110, 163/110, 13/22) = (2.6636, 1.4818, 1.4818, 0.5909)
```

If we can find intersection numbers κ̃_abc such that with K=[-3,-5,8,6] and M=[10,11,-11,-5]:
```
N_ab = κ̃_abc M^c  (invertible)
p = N⁻¹ K = (293/110, 163/110, 163/110, 13/22)
```

This would give us the correct κ̃_abc tensor in McAllister's basis.

---

## Questions for Research

1. **What divisor basis does McAllister use?**
   - The dual polytope has 12 points → 12 potential toric divisors
   - The triangulation uses points 0-8 → 9 toric divisors
   - h¹¹=4, so 4 independent divisors (5 linear relations)
   - Which 4 ambient divisors correspond to indices 0,1,2,3 in K_vec and M_vec?

2. **Is there a kklt_basis.dat for the DUAL?**
   - The file `kklt_basis.dat` exists but has 216 entries → for primal (h¹¹=214)
   - We need the basis for the dual (h¹¹=4)

3. **Can we compute κ̃_abc from the constraint that p is known?**
   - We have 4 equations from p = N⁻¹K
   - We have 4 equations from K·p = 0
   - κ̃_abc is symmetric with at most 20 independent components
   - This is underdetermined, but combined with CYTools' topological constraints...

4. **Does McAllister's supplementary code reveal the basis?**
   - Check if their code explicitly constructs the intersection tensor
   - Look for basis transformation matrices

---

## Next Steps

1. Search McAllister's arXiv supplementary material for:
   - How they construct intersection numbers
   - What divisor basis they use for the dual polytope
   - Any explicit basis transformation code

2. Try different CYTools divisor basis choices:
   - `cy.set_divisor_basis(...)` if available
   - See if any combination gives det(N) ≠ 0 with p matching

3. Algebraically derive κ̃_abc:
   - Use p_expected and the constraint p = N⁻¹K
   - Combined with κ symmetry and CYTools topology

4. Contact the authors or check their GitHub for clarification
