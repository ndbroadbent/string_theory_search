# Understanding basis.dat vs kklt_basis.dat

## Summary

McAllister's data files use TWO different divisor indexing schemes:

1. **`basis.dat`** (214 entries): The CYTools 2021 default divisor basis
2. **`kklt_basis.dat`** (214 entries): The divisors that contribute to the KKLT superpotential

**These are NOT the same!** They differ by 4 points:
- In `basis.dat` but NOT in `kklt_basis.dat`: **[1, 2, 46, 130]**
- In `kklt_basis.dat` but NOT in `basis.dat`: **[8, 9, 10, 17]**

## Which files use which indexing?

| File | Indexing | Notes |
|------|----------|-------|
| `corrected_kahler_param.dat` | `basis.dat` | 214 t values for Kähler moduli |
| `corrected_target_volumes.dat` | `kklt_basis.dat` | 214 τ values for KKLT divisor volumes |
| `target_volumes.dat` (c_i) | `kklt_basis.dat` | 214 dual Coxeter numbers (1 or 6) |
| `c_tau.dat` | N/A | Single scalar |
| `g_s.dat`, `W_0.dat` | N/A | Single scalars |

## The Key Relationship

The KKLT equation is:
```
τ_i = (1/2) κ_ijk t^j t^k
```

Where:
- `t` is indexed by `basis.dat` (h11 = 214 values)
- `τ` for KKLT divisors is indexed by `kklt_basis.dat`
- `κ_ijk` is computed in `basis.dat` coordinates

**For the 210 shared divisors** (points in both bases):
- McAllister's `t_expected` gives `τ_computed` that matches `τ_expected` with **RMS = 0.000117** (essentially exact)

**For the 4 unmapped KKLT divisors** (points 8, 9, 10, 17):
- These are in `kklt_basis.dat` but NOT in `basis.dat`
- Their τ values cannot be computed via κ_ijk t^j t^k using the `basis.dat` coordinates
- McAllister provides their expected τ values anyway:
  - kklt[5] = point 8: τ = 0.5716
  - kklt[6] = point 9: τ = 0.5742
  - kklt[7] = point 10: τ = 2.0775
  - kklt[14] = point 17: τ = 0.5218

## Open Question

**How are τ values for points 8, 9, 10, 17 computed?**

Possibilities:
1. McAllister used a DIFFERENT divisor basis internally (one that includes these points)
2. These divisors have τ computed via some other formula (not κ_ijk t^j t^k)
3. These are "autochthonous" divisors mentioned in the paper (Section 4.3)
4. There's a basis transformation we're missing

## Implications for Our Pipeline

### Option A: Ignore the 4 unmapped divisors
- Use `basis.dat` as the divisor basis
- Solve for t to match τ for the 210 shared divisors only
- Accept that 4 KKLT divisors won't be validated
- **Risk**: May miss important physics

### Option B: Use kklt_basis.dat as the basis
- Set `cy.set_divisor_basis(kklt_basis_dat)`
- All 214 KKLT divisors can then have τ computed
- But then `corrected_kahler_param.dat` (t values) won't match!
- Would need to transform t values between bases

### Option C: Find the basis transformation
- Points 1, 2, 46, 130 in `basis.dat` become points 8, 9, 10, 17 in `kklt_basis.dat`
- There may be a linear transformation T such that `t_kklt = T @ t_basis`
- Need to derive this from the GLSM or intersection numbers

## Current State (as of this writing)

The `compute_kahler_param.py` script was being modified to:
1. Use `basis.dat` for κ_ijk computation
2. Only constrain the 210 shared divisors
3. Leave the 4 non-KKLT basis divisors (1, 2, 46, 130) unconstrained

This approach was hitting numerical issues in the iterative solver (divergence/overflow).

## Verification Commands

```python
# Check if basis.dat matches CYTools 2021 default
cy = tri.get_cy()
default_basis = list(cy.divisor_basis())
# Result: default_basis == basis_dat  → True

# Verify τ computation for shared divisors
# Using basis.dat and t_expected:
# RMS error = 0.000117 for 210 shared divisors
```

## References

- McAllister paper Section 5.2: Iterative algorithm for solving KKLT
- McAllister paper Section 4.3: Autochthonous divisors
- `mcallister_2107/CLAUDE.md`: Three-layer architecture explanation
- `resources/small_cc_2107.09064_source/anc/paper_data/readme.txt`: Data file descriptions
