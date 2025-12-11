# Re-derive Two-Term Racetrack from GV Inputs (v1)

**Goal**: Build W_flux(τ) from geometry + flux + GV invariants, and show it reduces to McAllister's 2-term racetrack (eq. 6.59).

**Validates**: That we can construct the superpotential from first principles without hand-fed racetrack coefficients.

**Status**: PLAN - has open questions that must be resolved before implementation.

---

## Open Questions (Must Resolve Before Implementation)

### Q1: How to project 9D ambient curves to 4D h¹¹ basis?

The dual_curves.dat contains curves in 9D ambient toric basis. We need them in 4D to compute q̃·p.

**Possible approaches**:
- A) Use CYTools GLSM charge matrix to project 9D → 4D
- B) Find if McAllister provides a projection matrix in ancillary files
- C) Work entirely in 9D with an extended p vector

**Needs**: Investigation of CYTools API for curve basis projection.

### Q2: What basis is McAllister's p = (293/110, 163/110, 163/110, 13/22) in?

We know CYTools uses divisor basis [5,6,7,8] (ambient indices), but McAllister's p is in some other 4D basis.

**Possible approaches**:
- A) Find basis transformation GL(4,Z) between CYTools and McAllister
- B) Find McAllister's basis definition in paper or ancillary code
- C) Back-compute: if we know q̃·p = 32/110 for certain curves, solve for the transformation

**Needs**: Either paper reference or algebraic solution.

### Q3: How to compute M·q̃ in consistent basis?

M = [10, 11, -11, -5] is 4D in McAllister's basis. Curves are 9D.

**Same problem as Q1** - need basis alignment.

### Q4: Where does the factor of 5 in eq. 6.59 come from?

Eq. 6.59 has `5ζ` not just `ζ`. Is this:
- A) A combinatorial factor from the GV sum?
- B) Related to counting curve degeneracies?
- C) An overall normalization convention?

**Needs**: Careful reading of eq. 2.22-2.23 derivation.

### Q5: Why do small_curves_vols.dat values not match 32/110 and 33/110 exactly?

Closest values are 0.2888 and 0.2996, not 0.2909 and 0.3000.

**Possible explanations**:
- A) These are volumes in a different normalization
- B) Numerical precision issues
- C) These are not q̃·p values but something else (e.g., curve volumes in string units)

**Needs**: Understanding of what small_curves_vols.dat actually contains.

---

## What We're Trying to Reproduce

From eq. 6.59:
```
W_flux(τ) = 5ζ [-e^{2πiτ·(32/110)} + 512·e^{2πiτ·(33/110)}] + O(e^{2πiτ·(13/22)})
```

The two leading terms have:
- q̃₁·p = 32/110 with effective coefficient -1
- q̃₂·p = 33/110 with effective coefficient +512

From eq. 6.58, the raw GV invariants for the leading curves are:
```
N_q̃ = (1, -2, 252, -2)
```

---

## The General Formula

From eq. 2.22:
```
W_flux(τ) = -ζ Σ_{q̃ ∈ M(X̃)} (M·q̃) N_q̃ Li₂(e^{2πiτ(q̃·p)})
```

Where:
- ζ = 1/(2^{3/2} π^{5/2}) ≈ 0.0179
- M = [10, 11, -11, -5] is the flux vector
- N_q̃ is the GV invariant for curve class q̃
- p = (293/110, 163/110, 163/110, 13/22) is the flat direction
- Li₂(x) → x for small x (leading order)

At large Im(τ), Li₂(e^{2πiτ(q̃·p)}) ≈ e^{2πiτ(q̃·p)}, so:
```
W_flux(τ) ≈ -ζ Σ_{q̃} (M·q̃) N_q̃ e^{2πiτ(q̃·p)}
```

---

## Available Data

### small_curves.dat (344 curves × 219 components)
Each row is a curve class in the **219-dimensional ambient/primal basis** (h¹¹=214 + extra).
These are sparse vectors with mostly zeros and a few ±1 entries.

### small_curves_gv.dat (344 GV invariants)
One GV invariant per curve.
- **Unique values**: {-2: 29 curves, 1: 315 curves}
- **Note**: The value 252 from eq. 6.58 is NOT in this dataset

### small_curves_vols.dat (344 curve volumes)
Single line with 344 comma-separated floats.
- Range: -0.033 to 0.841
- **Negative values exist** (29 curves), which is surprising for q̃·p
- Closest to 32/110 = 0.2909: idx=196, vol=0.2888, GV=1
- Closest to 33/110 = 0.3000: idx=213, vol=0.2996, GV=1

### small_curves_cutoff.dat
Contains: `1.0` - likely the cutoff for "small" curves (q̃·p < 1.0)

### dual_curves.dat (5177 curves × 9 components)
Curves in 9D ambient toric basis for the dual polytope.

### dual_curves_gv.dat (5177 GV invariants)
GV invariants for dual curves. This is the larger dataset that may contain 252.

---

## Key Finding: Leading Curves in dual_curves.dat

The first few curves in `dual_curves.dat` have GV values matching eq. 6.58:

| idx | q̃ (9D) | GV |
|-----|--------|-----|
| 0 | [0,0,0,0,0,1,1,0,-2] | -2 |
| 1 | [-6,2,3,-1,1,1,0,0,0] | 252 |
| 2 | [0,0,0,0,1,0,0,1,-2] | -2 |
| 3 | [0,0,0,1,-1,-1,0,0,1] | 1 |

The GV = (1, -2, 252, -2) from eq. 6.58 are present! We need to:
1. Project these 9D curves to 4D using CYTools' GLSM charge matrix
2. Compute q̃·p in that 4D basis
3. Find which give q̃·p = 32/110 and 33/110

---

## The Basis Problem (Again)

**Issue**: The curve data is in different bases than McAllister's p vector.

- `small_curves.dat`: 219D primal ambient basis
- `dual_curves.dat`: 9D dual ambient basis
- McAllister's p: 4D moduli basis for the dual (h¹¹=4)

To compute q̃·p, we need curves and p in the **same basis**.

### Options:

**Option A**: Work with dual_curves.dat (9D) and project to 4D
- Need GLSM charge matrix to project 9D → 4D
- Then compute q̃·p in that 4D basis
- Challenge: Our p = (293/110, ...) is in McAllister's moduli basis, not CYTools' basis

**Option B**: Use small_curves.dat with primal p
- The primal has h¹¹=214, so p would be 214-dimensional
- But McAllister's data is for the dual (h¹¹=4)

**Option C**: Trust McAllister's curve volumes
- `small_curves_vols.dat` might contain q̃·p values directly
- If so, we can bypass the basis problem entirely

---

## Investigation: small_curves_vols.dat

```bash
head -10 small_curves_vols.dat
wc -l small_curves_vols.dat
```

If this contains the curve volumes (which should relate to q̃·p), we can:
1. Sort curves by volume
2. Find the two smallest non-zero volumes
3. Check if they match 32/110 = 0.2909 and 33/110 = 0.3

---

## Algorithm (Assuming we can get q̃·p)

```python
def build_racetrack_from_gv():
    """
    Build W_flux(τ) from GV invariants and identify the leading terms.
    """
    # Load data
    curves = load_curves()        # Curve classes q̃
    gv_invariants = load_gv()     # N_q̃ for each curve

    # Flux vector M (from paper eq. 6.55)
    M = np.array([...])  # In the right basis!

    # Flat direction p (from paper eq. 6.56)
    p = np.array([293/110, 163/110, 163/110, 13/22])

    # Compute q̃·p and M·q̃ for each curve
    terms = []
    for i, (q, N_q) in enumerate(zip(curves, gv_invariants)):
        q_dot_p = compute_q_dot_p(q, p)  # Needs basis alignment!
        M_dot_q = compute_M_dot_q(M, q)  # Needs basis alignment!

        if q_dot_p > 0 and q_dot_p < 1:  # Small enough to matter
            effective_coeff = M_dot_q * N_q
            terms.append({
                'q_dot_p': q_dot_p,
                'effective_coeff': effective_coeff,
                'M_dot_q': M_dot_q,
                'N_q': N_q
            })

    # Sort by q̃·p (smallest first = leading terms)
    terms.sort(key=lambda t: t['q_dot_p'])

    # The two leading terms should be:
    # q̃₁·p = 32/110 = 0.2909..., effective_coeff = -1
    # q̃₂·p = 33/110 = 0.3000..., effective_coeff = +512

    return terms[:10]  # Return top 10 for inspection
```

---

## Expected Results

If successful, we should find:

| Rank | q̃·p | M·q̃ | N_q̃ | Effective Coeff |
|------|------|------|------|-----------------|
| 1 | 32/110 = 0.2909 | ? | ? | -1 |
| 2 | 33/110 = 0.3000 | ? | ? | +512 |
| 3 | 13/22 = 0.5909 | ? | ? | (subleading) |

The factor of 5 in eq. 6.59 (`5ζ`) comes from some combinatorial factor in the sum.

---

## Key Questions to Resolve

1. **What basis is small_curves_vols.dat in?**
   - If it contains q̃·p directly, we're golden
   - Need to check if values near 0.29 and 0.30 appear

2. **How to compute M·q̃?**
   - M is 4D in McAllister's basis
   - Curves are 219D or 9D in ambient basis
   - Need the projection or a different approach

3. **Where does the factor of 5 come from?**
   - Eq. 6.59 has 5ζ, not just ζ
   - Is it from GV counting conventions or Li₂ expansion?

4. **Are the GV invariants (1, -2, 252, -2) visible in small_curves_gv.dat?**
   - The file shows mostly 1s and -2s
   - Where is the 252?

---

## Implementation Plan

**BLOCKED**: Implementation depends on resolving Q1-Q3 (basis alignment).

### Phase 0: Resolve Open Questions (REQUIRED FIRST)
- [ ] Q1: Determine how to project 9D curves to 4D
- [ ] Q2: Identify McAllister's divisor basis
- [ ] Q3: Establish consistent basis for M·q̃ computation
- [ ] Q4: Understand the factor of 5 in eq. 6.59
- [ ] Q5: Clarify what small_curves_vols.dat contains

### Phase 1: Data Exploration (after Q1-Q5 resolved)
- [ ] Load dual_curves.dat and dual_curves_gv.dat
- [ ] Project 9D curves to 4D using determined method
- [ ] Compute q̃·p for each curve

### Phase 2: Identify Leading Curves
- [ ] Sort curves by q̃·p
- [ ] Find curves with q̃·p = 32/110 and 33/110
- [ ] Verify their GV invariants match eq. 6.58

### Phase 3: Compute Effective Coefficients
- [ ] Compute M·q̃ for leading curves
- [ ] Compute (M·q̃) × N_q̃
- [ ] Verify we get effective coefficients -1 and +512

### Phase 4: Build W_flux(τ)
- [ ] Assemble the truncated sum with correct prefactor
- [ ] Verify it matches eq. 6.59 structure
- [ ] Solve F-term numerically and compare to 528

---

## Dependencies

- numpy
- McAllister data files in `resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/`

---

## References

- McAllister et al. arXiv:2107.09064, Section 6, eqs. 6.55-6.59
- Demirtas et al. arXiv:1912.10047 for perturbatively flat vacua theory
- eq. 2.22-2.23 for W_flux general formula
