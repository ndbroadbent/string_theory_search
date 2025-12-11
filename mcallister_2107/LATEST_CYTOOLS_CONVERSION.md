# Converting McAllister Configuration to Latest CYTools

**Goal**: Find the polytope, Kähler moduli, and flux parameters that work with the **latest CYTools** (2025) to reproduce McAllister's results, so we can use this as a validated genome in our GA.

**Status**: IN PROGRESS - basis transformation found but physics validation failing

---

## The Problem

CYTools 2021 (commit `bb5b550`, June 30, 2021) and CYTools latest (December 2025) choose **different divisor bases** for the same polytope:

| Version | Divisor Basis |
|---------|---------------|
| CYTools 2021 | `[3, 4, 5, 8]` |
| CYTools Latest | `[5, 6, 7, 8]` |

This means the fluxes K, M and flat direction p are expressed in different coordinate systems. We need to find the transformation to port McAllister's configuration.

---

## What We Know

### McAllister's Configuration (2021 basis)

```python
# Fluxes in 2021 basis [3,4,5,8]
K_2021 = [-3, -5, 8, 6]
M_2021 = [10, 11, -11, -5]

# Flat direction (computed from p = N⁻¹K)
p_2021 = [2.664, 1.482, 1.482, 0.591]  # = [293/110, 163/110, 163/110, 13/22]

# Results
g_s = 0.00911134
W₀ = 2.30012e-90
e^{K₀} = 0.2344
V₀ = -5.46e-203
```

### Intersection Numbers

**CYTools 2021 (basis [3,4,5,8]):**
```python
kappa_2021 = {
    (0,0,0): 1,  (0,0,1): -1, (0,0,2): -1,
    (0,1,1): 1,  (0,1,2): 1,  (0,2,2): 1,
    (1,1,1): -1, (1,1,2): -1, (1,2,2): -1,
    (1,2,3): 1,  (1,3,3): -2,
    (2,2,2): -1, (2,3,3): -2, (3,3,3): 8
}
```

**CYTools Latest (basis [5,6,7,8]):**
```python
kappa_latest = {
    (0,0,0): -1,
    (0,2,3): 1,  (0,3,3): -2,
    (1,2,3): 1,  (1,3,3): -2,
    (2,3,3): -2, (3,3,3): 8
}
```

Note: The latest basis has far fewer non-zero entries (7 vs 14).

---

## Approaches Tried

### Approach 1: GLSM Linear Relations (compute_transformation.py)

Tried to use the GLSM linear relations to express divisors in one basis in terms of the other.

```python
linrels = poly.glsm_linear_relations()
# Shape: (5, 9)
# [[ 1  1  1  1  1  1  1  1  1]
#  [ 0 -1  0  0  0  2  2  4  2]
#  [ 0  0 -1  0  0  3  3  6  3]
#  [ 0  0  0 -1  0 -1  1  0  0]
#  [ 0  0  0  0 -1  1 -1  1  0]]
```

**Problem**: D_6 and D_7 gave the same expression, resulting in singular transformation matrix (det=0).

### Approach 2: Numerical Optimization (solve_transformation.py)

Searched for T ∈ GL(4,ℤ) such that κ_latest = T⊗T⊗T · κ_2021

```python
def transform_kappa(T, kappa):
    return np.einsum('ad,be,cf,def->abc', T, T, T, kappa)

# Numerical optimization found:
T = [[0, 1, 0, 0],
     [1, 1, 0, 0],
     [1, 0, 1, 0],
     [0, 0, 0, 1]]

det(T) = -1  # Valid GL(4,Z) element
error = 0.0  # Perfectly transforms kappa!
```

**Result**: Found a T that transforms intersection numbers exactly.

### Approach 3: Flux Transformation

Tried different transformation rules for fluxes:

**Attempt A**: Direct transformation K' = T·K
```python
K_latest = T @ K_2021  # = [-5, -8, 5, 6]
M_latest = T @ M_2021  # = [11, 21, -1, -5]
# Result: e^{K₀} = 0.019 (wrong, should be 0.234)
```

**Attempt B**: Dual transformation K' = (T⁻¹)ᵀ·K
```python
T_inv_T = np.linalg.inv(T).T
# = [[-1,  1,  1,  0],
#    [ 1,  0, -1,  0],
#    [ 0,  0,  1,  0],
#    [ 0,  0,  0,  1]]

K_latest = T_inv_T @ K_2021  # = [6, -11, 8, 6]
M_latest = T_inv_T @ M_2021  # = [-10, 21, -11, -5]
# Result: e^{K₀} = 0.221 (closer but still wrong)
```

**Problem**: Neither transformation rule gives matching physics.

---

## Current Status

### What Works
- ✅ Found T that transforms κ_2021 → κ_latest exactly
- ✅ T is in GL(4,ℤ) with det(T) = -1

### What Doesn't Work
- ❌ Transformed fluxes don't give same e^{K₀}
- ❌ p vectors don't match under any transformation tried
- ❌ Physics (g_s, W₀, V₀) not validated in latest basis

---

## Possible Issues

1. **Multiple valid transformations**: There may be other T matrices that also transform κ correctly but have different flux transformation properties.

2. **Basis normalization**: The bases might have different normalizations that affect how fluxes transform.

3. **Curve vs divisor duality**: The transformation for curves (which appear in the GV sum) might be different from the transformation for fluxes.

4. **Non-linear transformation**: The relationship might not be a simple linear basis change.

---

## Next Steps to Try

### Option A: Brute Force Search
Search all GL(4,ℤ) matrices with small entries that:
1. Transform κ correctly
2. Give matching e^{K₀} when fluxes are transformed

### Option B: Direct Verification
Instead of transforming, directly solve for fluxes K', M' in the latest basis that give the same physics:
1. Find K', M' such that p' = (N')⁻¹K' gives e^{K₀} = 0.234
2. Verify that GV invariants + transformed curves give same racetrack

### Option C: Use Both CYTools Versions
Keep using CYTools 2021 for McAllister reproduction and validation, use latest CYTools for new searches. Accept that we can't directly compare genomes between versions.

### Option D: Investigate CYTools Internals
Look at exactly how CYTools computes `divisor_basis()` and understand the mathematical relationship between different basis choices.

---

## Code Files

| File | Purpose |
|------|---------|
| `get_kappa_2021.py` | Extract κ from CYTools 2021 |
| `get_kappa_latest.py` | Extract κ from CYTools latest |
| `compute_transformation.py` | GLSM-based approach (failed) |
| `solve_transformation.py` | Numerical optimization (partial success) |
| `find_transformation.py` | Brute force search (too slow) |

---

## The Ultimate Goal

We want a validated GA genome that looks like:

```python
genome = {
    'polytope_id': '4-214-647',
    'dual_points': [...],  # 12 points
    'simplices': [...],    # 15 simplices
    'K': [...],            # 4 integers (flux)
    'M': [...],            # 4 integers (flux)
    # Optional: explicit Kähler moduli if not derived from p
}
```

That, when evaluated with **latest CYTools** in our GA, produces:
- g_s ≈ 0.00911
- W₀ ≈ 2.3e-90
- V₀ ≈ -5.5e-203

This would serve as our ground truth validation case for the entire physics pipeline.


------------

> can you explain again why we had to resort to search and couldn't directly determine the corresponding K/M values for this other divisor basis?

because I thought you said the physics were identical if you match the numbers

⏺ You're right to push on this. Let me explain the confusion:

  The physics IS identical - same polytope, same triangulation, just different coordinate system for the divisors.

  The transformation should be deterministic:
  - Old basis [3,4,5,8]: K = [-3, -5, 8, 6], M = [10, 11, -11, -5]
  - New basis [5,6,7,8]: K = ???, M = ???

  The divisors are related by GLSM linear relations. If D_3 = -D_5 + D_6 (for example), then fluxes should transform accordingly.

  What went wrong earlier:

  I tried several transformation matrices (T, T⁻¹, Tᵀ, etc.) but none gave matching e^K0. This was surprising and I gave up too quickly.

  The likely issue:

  Either:
  1. I made an error in computing T from the GLSM relations
  2. The flux transformation has a subtlety I missed (maybe it's not just linear)
  3. The GV invariants are also in a basis and need to transform

  Let me try again properly - this should be computable exactly, not searched:

⏺ Bash(uv run python3 -c "
      import sys
      sys.path.insert(0, 'vendor/cytools_latest/src')
      from cytools import Polytope
      import numpy as np

      # Load polytope
      dual_points = np.loadtxt('resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/dual_points.dat', delimiter=',').astype(int)
      with open('resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/dual_simplices.dat') as f:
          simplices = [[int(x) for x in line.strip().split(',')] for line in f]

      poly = Polytope(dual_points)
      tri = poly.triangulate(simplices=simplices)
      cy = tri.get_cy()

      # Get the basis change matrix directly from CYTools
      # This should give us exactly how to transform between bases

      old_basis = [3, 4, 5, 8]
      new_basis = [5, 6, 7, 8]

      # Set old basis and get the divisor basis matrix
      cy.set_divisor_basis(old_basis)
      # The basis matrix tells us how basis divisors relate to all divisors
      print('Divisor basis (old):', cy.divisor_basis())

      # Now let's compute kappa and p in old basis
      def sparse_to_dense(sparse, h11=4):
          kappa = np.zeros((h11, h11, h11))
          for (i,j,k), val in sparse.items():
              for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
                  kappa[perm] = val
          return kappa

      kappa_old = sparse_to_dense(cy.intersection_numbers(in_basis=True))
      K_old = np.array([-3, -5, 8, 6])
      M_old = np.array([10, 11, -11, -5])

      N_old = np.einsum('abc,c->ab', kappa_old, M_old)
      p_old = np.linalg.solve(N_old, K_old)
      kappa_p3_old = np.einsum('abc,a,b,c->', kappa_old, p_old, p_old, p_old)
      eK0_old = 1.0 / ((4.0/3.0) * kappa_p3_old)

      print(f'\\nOLD BASIS [3,4,5,8]:')
      print(f'  p = {p_old}')
      print(f'  e^K0 = {eK0_old}')

      # Now switch to new basis
      cy.set_divisor_basis(new_basis)
      kappa_new = sparse_to_dense(cy.intersection_numbers(in_basis=True))

      # The KEY insight: p is a point in the Kähler moduli space
      # It transforms as a VECTOR (contravariant), not as flux (covariant)
      #
      # If we know p_old in old basis, we need to find p_new in new basis
      # such that they represent the SAME physical point
      #
      # The divisor volumes τ_i = (1/2) κ_ijk t^j t^k are physical
      # Let's compute them in old basis and match in new basis

      # Divisor volumes in old basis (for the 4 basis divisors)
      def divisor_volumes(kappa, t):
          h11 = len(t)
          vols = np.zeros(h11)
          for i in range(h11):
              vols[i] = 0.5 * np.einsum('jk,j,k->', kappa[i], t, t)
          return vols

      tau_old = divisor_volumes(kappa_old, p_old)
      print(f'  Divisor volumes τ = {tau_old}')

      # Now we need to find p_new such that the PHYSICAL divisor volumes match
      # But wait - the basis divisors are DIFFERENT, so the volumes won't directly match
      # We need to transform the volumes too

      # Actually, let's think about this differently.
      # The Kähler form J = t^i D_i (sum over basis divisors)
      # Physical volume V = (1/6) κ_ijk t^i t^j t^k is basis-independent
      #
      # Let's verify: compute V in old basis
      V_old = (1/6) * np.einsum('ijk,i,j,k->', kappa_old, p_old, p_old, p_old)
      print(f'  CY volume V = {V_old}')

      # The claim: there exists p_new such that V_new = V_old
      # AND all the physics (eK0, etc) matches

      # Let's try to SOLVE for p_new directly
      # We need: (4/3) κ_new_abc p^a p^b p^c = (4/3) κ_old_abc p^a p^b p^c
      # This is a cubic equation in p_new

      # Actually simpler: K and M transform, then p_new = N_new^{-1} K_new
      # The question is: how do K and M transform?

      # K and M are flux vectors. They pair with divisors: K · D
      # If D_old = T · D_new, then for K·D to be invariant: K_new = T^T · K_old

      # Let's find T from GLSM relations
      print('\\n--- Finding transformation T ---')
      linrels = poly.glsm_linear_relations()
      print(f'GLSM relations shape: {linrels.shape}')
      print(linrels)

      # From GLSM: D_3, D_4 can be expressed in terms of other divisors
      # Row 3: D_3 + D_5 - D_6 = 0  =>  D_3 = -D_5 + D_6
      # Row 4: D_4 - D_5 + D_6 - D_7 = 0  =>  D_4 = D_5 - D_6 + D_7

      # So in terms of new basis [D_5, D_6, D_7, D_8]:
      # D_3 = -1*D_5 + 1*D_6 + 0*D_7 + 0*D_8
      # D_4 = 1*D_5 - 1*D_6 + 1*D_7 + 0*D_8
      # D_5 = 1*D_5 + 0*D_6 + 0*D_7 + 0*D_8
      # D_8 = 0*D_5 + 0*D_6 + 0*D_7 + 1*D_8

      T = np.array([
          [-1,  1,  0,  0],  # D_3 in terms of [D_5, D_6, D_7, D_8]
          [ 1, -1,  1,  0],  # D_4
          [ 1,  0,  0,  0],  # D_5
          [ 0,  0,  0,  1],  # D_8
      ])
      print(f'\\nT (old basis in terms of new):')
      print(T)
      print(f'det(T) = {np.linalg.det(T)}')

      # If J = t_old · D_old = t_old · (T · D_new) = (T^T · t_old) · D_new
      # So t_new = T^T · t_old (Kähler moduli transform with T^T)

      # For fluxes K, M that pair with divisors K · D:
      # K_old · D_old = K_old · (T · D_new)
      # For this to equal K_new · D_new, we need K_new = T^T · K_old

      K_new = T.T @ K_old
      M_new = T.T @ M_old
      print(f'\\nTransformed fluxes:')
      print(f'  K_new = T^T @ K_old = {K_new}')
      print(f'  M_new = T^T @ M_old = {M_new}')

      # Now compute physics in new basis
      N_new = np.einsum('abc,c->ab', kappa_new, M_new)
      print(f'\\nN_new = κ_new · M_new:')
      print(N_new)
      print(f'det(N_new) = {np.linalg.det(N_new)}')

      if abs(np.linalg.det(N_new)) > 1e-10:
          p_new = np.linalg.solve(N_new, K_new)
          kappa_p3_new = np.einsum('abc,a,b,c->', kappa_new, p_new, p_new, p_new)
          eK0_new = 1.0 / ((4.0/3.0) * kappa_p3_new)
          V_new = (1/6) * np.einsum('ijk,i,j,k->', kappa_new, p_new, p_new, p_new)

          print(f'\\nNEW BASIS [5,6,7,8]:')
          print(f'  p = {p_new}')
          print(f'  e^K0 = {eK0_new}')
          print(f'  CY volume V = {V_new}')

          print(f'\\n--- COMPARISON ---')
          print(f'  e^K0 old: {eK0_old:.6f}')
          print(f'  e^K0 new: {eK0_new:.6f}')
          print(f'  Match: {np.isclose(eK0_old, eK0_new)}')
      else:
          print('N_new is singular!')
      "
      )
  ⎿  Divisor basis (old): [3 4 5 8]

     OLD BASIS [3,4,5,8]:
       p = [2.66363636 1.48181818 1.48181818 0.59090909]
       e^K0 = 0.23439299551782825
       Divisor volumes τ = [0.045      0.48144628 0.48144628 0.09      ]
       CY volume V = 0.5332923866766843

     --- Finding transformation T ---
     GLSM relations shape: (5, 9)
     [[ 1  0  0  0  0  6  6 12  6]
      [ 0  1  0  0  0 -2 -2 -4 -2]
      [ 0  0  1  0  0 -3 -3 -6 -3]
      [ 0  0  0  1  0  1 -1  0  0]
      [ 0  0  0  0  1 -1  1 -1  0]]

     T (old basis in terms of new):
     [[-1  1  0  0]
      [ 1 -1  1  0]
      [ 1  0  0  0]
      [ 0  0  0  1]]
     det(T) = 1.0

     Transformed fluxes:
       K_new = T^T @ K_old = [ 6  2 -5  6]
       M_new = T^T @ M_old = [-10  -1  11  -5]

     N_new = κ_new · M_new:
     [[ 10.   0.  -5.  21.]
      [  0.   0.  -5.  21.]
      [ -5.  -5.   0.  -1.]
      [ 21.  21.  -1. -40.]]
     det(N_new) = 12100.00000000001

     NEW BASIS [5,6,7,8]:
       p = [0.4        0.53636364 0.93636364 0.31818182]
       e^K0 = 1.027422638394576
       CY volume V = 0.12166366140746301

     --- COMPARISON ---
       e^K0 old: 0.234393
       e^K0 new: 1.027423
       Match: False
