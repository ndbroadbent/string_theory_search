# e^K₀ Formula - RESOLVED

## Problem Statement (RESOLVED)

~~When computing e^K₀ using CYTools intersection numbers and the formula from McAllister et al. (arXiv:2107.09064) eq 6.12, there is a systematic factor of 16/9 discrepancy.~~

**RESOLUTION: LaTeX parsing error!**

The original LaTeX (from AdS4_v3.tex line 1171):
```latex
e^{\mathcal{K}_0}:=\left(\frac{4}{3}\tilde{\kappa}_{abc}p^ap^bp^c\right)^{-1}
```

**Correct interpretation:**
```
e^K₀ = (4/3 × κ̃_abc p^a p^b p^c)^{-1}
     = 1 / (4/3 × κ_p3)
     = (3/4) / κ_p3
     = (3/4) × (κ̃_abc p^a p^b p^c)^{-1}
```

The `^{-1}` applies to the ENTIRE expression including the 4/3 factor, so:
- **Correct formula: e^K₀ = (3/4) / κ_p3** ✓
- NOT: e^K₀ = (4/3) / κ_p3 ✗

There was never a physics discrepancy - just a misreading of the LaTeX parentheses!

## What e^K₀ Represents

From the paper, e^K₀ is the exponential of the tree-level Kähler potential for complex structure moduli. It appears in the vacuum energy formula (eq 6.24):

```
V₀ = -3 × e^K₀ × (g_s⁷/(4V[0])²) × W₀²
```

The paper derives (around eq 6.11-6.12):
```
K_eff(Im τ) = -log(Im τ) - log(-i ∫_X Ω ∧ Ω̄)
            = -4 log(Im τ) + K₀ + O(Im τ)^{-3}

with constant e^K₀ := (4/3) × (κ̃_abc p^a p^b p^c)^{-1}
```

## Notation

- **κ̃_abc**: Intersection numbers of the mirror threefold X̃ (the dual CY)
- **κ_ijk**: Intersection numbers of the primal threefold X
- **p^a**: Components of the flat direction vector satisfying z = p·τ
- The paper uses the map κ̃_abc → κ_ijk when working on the mirror (eq 4.10)

## Verified Data

### 5-113-4627-main Example

**Paper's explicit κ̃ values (eq 6.5-6.6):**
```
κ̃_111 = 89,  κ̃_113 = 16,  κ̃_114 = 12,  κ̃_115 = 7
κ̃_134 = 3,   κ̃_145 = 3,   κ̃_155 = -3
κ̃_222 = 8,   κ̃_223 = -2,  κ̃_224 = -2,  κ̃_225 = -2
κ̃_234 = 1,   κ̃_245 = 1
κ̃_555 = -1
```

**CYTools output (0-indexed, so κ̃_111 → κ_000):**
```
κ_{000} = 89
κ_{002} = 16
κ_{003} = 12
κ_{004} = 7
κ_{023} = 3
κ_{034} = 3
κ_{044} = -3
κ_{111} = 8
κ_{112} = -2
κ_{113} = -2
κ_{114} = -2
κ_{123} = 1
κ_{134} = 1
κ_{444} = -1
```

**✓ MATCH: All κ values agree (accounting for 0 vs 1 indexing)**

**Paper's explicit p (eq 6.8):**
```
p = (7/58, 15/58, 101/116, 151/58, -13/116)
  = (0.12069, 0.25862, 0.87069, 2.60345, -0.11207)
```

**CYTools computed p:**
```
p = [0.12068966, 0.25862069, 0.87068966, 2.60344828, -0.11206897]
```

**✓ MATCH: p values agree to 6+ decimal places**

**Paper's explicit e^K₀ (eq 6.12):**
```
e^K₀ = 1170672/12843563 = 0.091149
```

## The Computation

### How we compute κ_p3

```python
import numpy as np

# Load κ from CYTools (sparse format: [[i, j, k, value], ...])
kappa_sparse = cy.intersection_numbers(in_basis=True)

# Build full symmetric tensor
h11 = cy.h11()  # = 5 for this example
kappa = np.zeros((h11, h11, h11))
for row in kappa_sparse:
    i, j, k = int(row[0]), int(row[1]), int(row[2])
    val = row[3]
    # Fill all 6 permutations (κ is symmetric)
    for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
        kappa[perm] = val

# Compute p from flux vectors K, M
# N_ab = κ_abc M^c, then solve N @ p = K
N = np.einsum('abc,c->ab', kappa, M)
p = np.linalg.solve(N, K)

# Compute κ_p3 = κ_abc p^a p^b p^c
kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)
```

### Numerical Results for 5-113-4627-main

```
κ_p3 (einsum over all a,b,c) = 8.228327

Paper's e^K₀ = 0.091149
Implied κ_p3 from (4/3)/e^K₀ = 14.628137

Our e^K₀ = (4/3)/κ_p3 = 0.162042
Ratio our/paper = 1.777778 = 16/9

BUT: e^K₀ = (3/4)/κ_p3 = 0.091149 ← MATCHES EXACTLY
```

### Verification of Contraction

To verify the einsum is correct, we also computed term-by-term:

```python
kappa_p3_explicit = 0
for i in range(h11):
    for j in range(i, h11):
        for k in range(j, h11):
            val = kappa[i, j, k]
            if val != 0:
                # Multiplicity: 1 for i=j=k, 3 for two equal, 6 for all different
                if i == j == k:
                    mult = 1
                elif i == j or j == k or i == k:
                    mult = 3
                else:
                    mult = 6
                kappa_p3_explicit += val * mult * p[i] * p[j] * p[k]
```

Result: `kappa_p3_explicit = 8.228327` (identical to einsum)

The largest contributing terms:
```
κ_{023} × 6 × p^0 × p^2 × p^3 = 3 × 6 × 0.1207 × 0.8707 × 2.6034 = 4.9244
κ_{123} × 6 × p^1 × p^2 × p^3 = 1 × 6 × 0.2586 × 0.8707 × 2.6034 = 3.5174
κ_{003} × 3 × p^0 × p^0 × p^3 = 12 × 3 × 0.1207 × 0.1207 × 2.6034 = 1.3652
```

## Results for All Examples (UPDATED)

Using the **correct formula e^K₀ = (3/4) / κ_p3**:

| Example | κ_p3 | e^K₀ = (3/4)/κ_p3 | Paper's e^K₀ | Match? |
|---------|------|-------------------|--------------|--------|
| 5-113-4627-main | 8.228 | 0.0911 | 0.0911 | ✓ EXACT |
| 4-214-647 | 3.200 | 0.2344 | ~0.236* | ✓ MATCH |
| 7-51-13590 | 6.485 | 0.1157 | 0.2719 | ✗ NO |

*4-214-647's e^K₀ back-calculated from V₀ = -5.5e-203

**Note: 7-51-13590 still doesn't match.** This example may have additional complications (different flat direction structure, different κ normalization, etc.).

## Possible Explanations

### 1. Different Definition of κ_abc p^a p^b p^c

The paper might use a non-standard contraction. Possibilities:
- Only sum over sorted indices (a ≤ b ≤ c) without multiplicity factors
- Different symmetrization convention
- Factor absorbed into κ definition

We tested summing only sorted indices without multiplicity:
```
κ_p3 (sorted, no mult) = 1.714
(4/3)/1.714 = 0.778 ← Does NOT match 0.0911
```

### 2. Prepotential Normalization

The prepotential is typically written as:
```
F = -(1/6) κ_abc z^a z^b z^c + ...
```

The 1/6 is a symmetry factor. If there's a mismatch in how this propagates to e^K₀, it could explain the discrepancy.

### 3. Mirror Map Subtlety

The paper uses κ̃_abc (mirror intersection numbers) in eq 6.12. We're computing κ on the dual polytope, which SHOULD be κ̃. But there might be a normalization difference in the mirror map.

### 4. Kähler Potential Derivation

From the special geometry of CY moduli spaces:
```
K_cs = -log(-i ∫ Ω ∧ Ω̄) = -log(i(z̄^a F_a - z^a F̄_ā))
```

At large complex structure with F = -(1/6)κz³:
```
F_a = -(1/2) κ_abc z^b z^c
z̄^a F_a = -(1/2) κ_abc z̄^a z^b z^c
```

The factor of 4/3 in the paper might come from this derivation. But we haven't traced it through.

## Code to Reproduce

```python
#!/usr/bin/env python3
"""Reproduce the e^K₀ discrepancy."""

import sys
import numpy as np
from pathlib import Path
from fractions import Fraction

CYTOOLS_2021 = Path("vendor/cytools_mcallister_2107")
DATA_BASE = Path("resources/small_cc_2107.09064_source/anc/paper_data")

def main():
    example = "5-113-4627-main"

    # Load data
    data_dir = DATA_BASE / example
    K = np.array([int(x) for x in (data_dir / "K_vec.dat").read_text().strip().split(",")])
    M = np.array([int(x) for x in (data_dir / "M_vec.dat").read_text().strip().split(",")])

    dual_pts_lines = (data_dir / "dual_points.dat").read_text().strip().split('\n')
    dual_pts = np.array([[int(x) for x in line.split(',')] for line in dual_pts_lines])

    simplices_lines = (data_dir / "dual_simplices.dat").read_text().strip().split('\n')
    simplices = [[int(x) for x in line.split(',')] for line in simplices_lines]

    # Use CYTools 2021
    sys.path.insert(0, str(CYTOOLS_2021))
    from cytools import Polytope

    poly = Polytope(dual_pts)
    tri = poly.triangulate(simplices=simplices, check_input_simplices=False)
    cy = tri.get_cy()

    # Get κ tensor
    kappa_sparse = cy.intersection_numbers(in_basis=True)
    h11 = cy.h11()
    kappa = np.zeros((h11, h11, h11))
    for row in kappa_sparse:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        val = row[3]
        for perm in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]:
            kappa[perm] = val

    # Compute p
    N = np.einsum('abc,c->ab', kappa, M)
    p = np.linalg.solve(N, K)

    # Compute κ_p3
    kappa_p3 = np.einsum('abc,a,b,c->', kappa, p, p, p)

    # Compare
    eK0_paper = Fraction(1170672, 12843563)
    eK0_43 = (4/3) / kappa_p3
    eK0_34 = (3/4) / kappa_p3

    print(f"κ_p3 = {kappa_p3:.6f}")
    print(f"Paper's e^K₀ = {float(eK0_paper):.6f}")
    print(f"e^K₀ via (4/3)/κ_p3 = {eK0_43:.6f}, ratio = {eK0_43/float(eK0_paper):.6f}")
    print(f"e^K₀ via (3/4)/κ_p3 = {eK0_34:.6f}, ratio = {eK0_34/float(eK0_paper):.6f}")

if __name__ == "__main__":
    main()
```

## Open Questions

1. **Why is the factor 3/4 instead of 4/3?** The paper explicitly states (4/3). Where does the 16/9 come from?

2. **Why doesn't 7-51-13590 match?** Even with the 3/4 factor, the ratio is 0.426, not 1.0. Is there something special about this example?

3. **Is there a different convention for the contraction κ_abc p^a p^b p^c?** We use standard Einstein summation over all indices with a symmetric tensor.

4. **Could this be related to the (1/6) in the prepotential or volume formula?** The CY volume is V = (1/6)κ_ijk t^i t^j t^k. How does this propagate to e^K₀?

5. **Is there a typo in eq 6.12?** Unlikely for such a key formula, but worth considering.

## References

- McAllister et al., arXiv:2107.09064, equations 6.5-6.12, 6.24
- CYTools: https://cy.tools/
- Vendor version: `vendor/cytools_mcallister_2107` (matches CYTools version from 2021)
