# Plan: Implement Period Computation for Cosmological Constant

## Status: PARTIAL SUCCESS (2024-12-11)

### What Works ✓
1. **cygv integration**: Installed and working via CYTools wrapper
2. **GV invariants**: Match McAllister's data (252, -9252, 848628, etc.)
3. **V₀ formula**: Verified V₀ = -5.5×10⁻²⁰³ using McAllister's W₀

### What Doesn't Work ✗
1. **Cannot compute W₀ from first principles** - McAllister's W₀ = 2.3×10⁻⁹⁰ comes from numerical optimization over complex structure moduli space, not direct evaluation
2. **No explicit complex structure point** - McAllister doesn't publish the z values where W₀ is minimized
3. **Basis mismatch** - Their flux vectors are 4D but full period space is 10D

### Key Discovery
**cygv computes GV invariants, NOT periods directly.** To get periods we need:
1. Assemble prepotential F(z) from GV invariants
2. Find the "perturbatively flat direction" p in CS moduli space
3. Optimize W₀ along that direction
4. This is a complex numerical search problem (Section 5 of their paper)

### Practical Conclusion
For our GA, use McAllister's published W₀ value directly. The V₀ formula is verified.

---

## Original Goal
Compute the cosmological constant from first principles using `cygv` for period computation.

**Target**: Take McAllister's polytope + moduli → compute:
1. W₀ = 2.3×10⁻⁹⁰ (flux superpotential)
2. V₀(AdS) = -5.5×10⁻²⁰³ (AdS vacuum energy)

**Note on uplift**: McAllister stops at V₀(AdS). The uplift to positive de Sitter (our universe's Λ ~ +10⁻¹²²) requires anti-D3 branes in a warped throat - this is "left as future work" in their paper. The uplift must be "same order" as |V₀(AdS)|, so for our universe we'd want V₀(AdS) ~ -10⁻¹²². McAllister's -10⁻²⁰³ is much smaller. For now, we just reproduce their result; GA can later search for -10⁻¹²² scale.

## Background

### Current Architecture
```
Rust GA (searcher.rs)
  → PyO3 bridge (physics.rs)
    → physics_bridge.py
      → CYTools (Python)
```

### The Problem
`physics_bridge.py` line 568 uses **fake periods**:
```python
periods = np.exp(1j * np.arange(n_periods) * 0.1) * complex_mod[0]
```
This makes W₀ and therefore V₀ completely garbage.

## The Solution: `cygv` (Rust library)

From `resources/COMPUTING_PERIODS.md`:
- **cygv** implements the HKTY procedure (Hosono-Klemm-Theisen-Yau)
- **Rust core** with Python bindings
- Computes: fundamental period, log periods, mirror map, prepotential
- This gives us Π(z) which we need for W₀ = (F - τH) · Π

### Why Rust-native is better
Current: `Rust → PyO3 → Python → (if cygv Python) → Rust`
Better:  `Rust → cygv (Rust crate directly)`

We can call cygv directly from our Rust code in `physics.rs`, bypassing Python for the period computation entirely.

## Implementation Steps

### Phase 1: Add cygv to Rust dependencies
1. Add `cygv` to `Cargo.toml`
2. Explore cygv Rust API (https://docs.rs/cygv)
3. Understand input format: how to pass polytope data

### Phase 2: Create Rust Period Module
1. Add `src/periods.rs` module
2. Function: `compute_periods(polytope: &Polytope, z: &[Complex64]) -> Vec<Complex64>`
3. Handle HKTY series expansion at large complex structure

### Phase 3: Validate Against McAllister (Python, BEFORE Rust integration)
Use cygv Python bindings for this validation step.

1. Use McAllister's 4-214-647 polytope (dual, h¹¹=4)
2. Compute periods at their complex structure point (large complex structure)
3. Apply their fluxes K_vec, M_vec
4. Compare W₀ to published 2.3e-90
5. Verify V₀(AdS) ≈ -5.5e-203

This validates the full pipeline. Uplift is out of scope (McAllister didn't do it either).

### Phase 4: Integrate into GA Pipeline
- Compute periods in Rust via cygv
- Compute W₀ = (F - τH) · Π in Rust
- Compute V₀ = -3 eᴷ |W|² in Rust
- Only use Python/CYTools for geometry (intersection numbers, volumes)

### Phase 5: Update Tests
1. Unit test for period computation (Rust)
2. Integration test for full pipeline
3. McAllister reproduction as regression test

## Key Files to Modify
- `Cargo.toml` - Add cygv dependency
- `src/physics.rs` - Add period computation, possibly full V₀ calc
- `physics_bridge.py` - Remove fake periods, possibly simplify

## Key Files to Create
- `src/periods.rs` - Rust module wrapping cygv

## Critical Considerations

### 1. Basis Alignment
McAllister's flux vectors (K_vec.dat, M_vec.dat) are 4-dimensional (h²¹=4).
Full symplectic basis for periods has 2(h²¹+1) = 10 dimensions.
Need to understand the projection/embedding.

### 2. Precision
W₀ ~ 10⁻⁹⁰ requires:
- Arbitrary precision or careful use of log-space arithmetic
- cygv may use exact rational arithmetic internally

### 3. Large h¹¹
McAllister's primal has h¹¹=214. Period computation complexity may scale badly.
The dual (h¹¹=4) is likely what's used for periods (mirror symmetry).

### 4. Complex Structure Point
Need to identify the exact point z in moduli space where McAllister evaluates.
Likely "large complex structure" limit where HKTY expansion converges.

## Remaining Unknowns (to figure out during implementation)
1. Exact cygv API - need to read docs.rs/cygv
2. How McAllister's K_vec/M_vec map to full symplectic basis
3. Whether cygv handles h¹¹=214 or if we need the dual (h¹¹=4)
