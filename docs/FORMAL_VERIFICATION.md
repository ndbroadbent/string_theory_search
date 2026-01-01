# Formal Verification Strategy for Cyrus

## Overview

The cyrus Rust toolkit for Calabi-Yau manifold computations is a prime candidate for formal verification using Aeneas. Unlike typical software where bugs cause crashes, **physics code bugs produce plausible-looking wrong answers**. Formal verification provides mathematical certainty that our implementation matches the physics formulas.

## Why Formal Verification Matters Here

### The Problem with Physics Code

1. **Silent failures**: A sign error or factor of 2 doesn't crash - it just gives a wrong cosmological constant
2. **No ground truth**: We can't easily verify against "expected output" for novel computations
3. **Compounding errors**: Intersection tensor → divisor volumes → KKLT → V₀ (errors propagate)
4. **Subtle symmetries**: The intersection tensor κ_ijk has 6-fold symmetry that's easy to get wrong

### Bugs Already Caught (Without Formal Verification)

- **2× Jacobian factor** in divisor.rs - caught only by numerical finite-difference test
- **BBHL sign errors** in early Python versions
- **Basis transformation transpose errors** - K transforms covariant, M contravariant
- **Multiplicity counting** in symmetric tensor storage

These were caught by luck and extensive testing. Formal verification would catch them at compile time.

## Toolchain: Aeneas

**Repository**: https://github.com/AeneasVerif/aeneas

### What Aeneas Does

1. **Charon** compiles Rust to LLBC (Low-Level Borrow Calculus)
2. **Aeneas** translates LLBC to pure lambda calculus
3. Output to theorem provers: **Lean 4**, HOL4, F*, Coq

### Current Status

- Supports safe Rust subset
- Proven sound (ICFP 2022, 2024 papers)
- Lean backend is most mature with specialized tactics

### Installation

```bash
# Install globally via Nix (first run: 10-20 minutes)
nix profile install github:aeneasverif/aeneas#aeneas

# Verify
charon --help
aeneas -help
```

### Workflow

```bash
# 1. Rust → LLBC
cd crates/cyrus-core/rust && charon cargo --preset=aeneas

# 2. LLBC → Lean
aeneas -backend lean -dest ../lean cyrus_core.llbc

# 3. Write proofs in Lean
# 4. Build with lake
cd ../lean && lake build
```

## High-Value Verification Targets

### Priority 1: Intersection Tensor (intersection.rs)

The foundation of all computations. Bugs here corrupt everything downstream.

**Prove:**
```lean
-- Symmetry: all 6 permutations return same value
theorem get_symmetric (κ : Intersection) (i j k : Nat) :
  κ.get i j k = κ.get j i k ∧
  κ.get i j k = κ.get k j i ∧
  κ.get i j k = κ.get i k j ∧
  κ.get i j k = κ.get j k i ∧
  κ.get i j k = κ.get k i j

-- Canonical key is idempotent
theorem canonical_key_idempotent (i j k : Nat) :
  canonical_key (canonical_key (i, j, k)) = canonical_key (i, j, k)

-- Multiplicity correctly counts permutations
theorem multiplicity_correct (i j k : Nat) (h : i ≤ j ∧ j ≤ k) :
  symmetry_multiplicity i j k = (permutations_of (i, j, k)).length
```

### Priority 2: Divisor Volume Gradient (divisor.rs)

We JUST fixed a 2× factor bug here. The Jacobian must be the derivative of τ.

**Prove:**
```lean
-- Jacobian is the gradient of divisor volumes
theorem jacobian_is_gradient (κ : Intersection) (t : Vector n Float) :
  compute_divisor_jacobian κ t = ∇(compute_divisor_volumes κ) t

-- Equivalently: τ_i = ∂V/∂t^i where V = (1/6) κ t³
theorem tau_is_volume_derivative (κ : Intersection) (t : Vector n Float) (i : Fin n) :
  (compute_divisor_volumes κ t)[i] = ∂/∂t[i] (volume_classical κ t)
```

### Priority 3: Linear Algebra Correctness (flat_direction.rs)

Gaussian elimination with partial pivoting must produce correct solutions.

**Prove:**
```lean
-- Solution satisfies the equation
theorem solve_correct (N : Matrix n n Float) (K : Vector n Float) :
  (solve_linear_system N K = some p) → matrix_mul N p = K

-- Pivoting preserves solutions
theorem pivot_preserves_solution (A : AugmentedMatrix) (A' : AugmentedMatrix) :
  (A' = swap_rows A i j) → solutions A = solutions A'
```

### Priority 4: Volume Formula Consistency (volume.rs)

The relationship between classical volume, BBHL correction, and string frame volume.

**Prove:**
```lean
-- V_string = V_classical - BBHL (sign is critical!)
theorem v_string_formula (κ : Intersection) (t : Vector n Float) (h11 h21 : Int) :
  volume_string κ t h11 h21 = volume_classical κ t - bbhl_correction h11 h21

-- BBHL is positive when h11 > h21 (typical case)
theorem bbhl_positive (h11 h21 : Int) (h : h11 > h21) :
  bbhl_correction h11 h21 > 0
```

### Priority 5: Flat Direction Computation (flat_direction.rs)

The e^{K₀} computation is critical for the cosmological constant.

**Prove:**
```lean
-- N matrix contraction is correct
theorem n_matrix_correct (κ : Intersection) (M : Vector n Int) :
  (compute_n_matrix κ M)[a][b] = Σ_c κ.get(a, b, c) * M[c]

-- e^K₀ formula
theorem ek0_formula (κ : Intersection) (p : Vector n Float) :
  compute_ek0 κ p = 1 / ((4/3) * κ.contract_triple p)
```

## Verification Approach

### Phase 1: Core Tensor Operations

1. Set up Aeneas pipeline for cyrus-core
2. Prove intersection tensor symmetry properties
3. Prove contract_triple equals explicit sum

### Phase 2: Calculus Properties

1. Prove divisor volumes are gradients of classical volume
2. Prove Jacobian is correct derivative
3. This requires careful handling of floating-point or switching to rationals

### Phase 3: Linear Algebra

1. Prove Gaussian elimination correctness
2. Prove solve_linear_system produces valid solutions
3. Handle singular matrix detection

### Phase 4: Integration Tests as Theorems

Convert key numerical tests to theorems:
- McAllister 4-214-647 volume computation
- BBHL correction for χ = 420

## Challenges

### Floating Point

Aeneas/Lean work with exact arithmetic. Options:
1. **Use rationals** for intersection numbers (they're integers anyway)
2. **Abstract over field** and prove properties algebraically
3. **Bound errors** for floating point approximations

### Loop Patterns

Aeneas has restrictions on loop control flow. May need to restructure some iteration patterns.

### Mathlib Integration

Lean proofs can leverage Mathlib for:
- Linear algebra (matrices, vectors)
- Number theory (for intersection number properties)
- Analysis (derivatives, limits)

## Prior Art: lean_experiment

See `/Users/ndbroadbent/code/lean_experiment/` for working Aeneas examples:

- **password-verifier**: 100% verified cryptographic proofs (20.1 KB, no `sorry`)
- **utf8-parser**: Universal bitwise operation proofs
- **abc-product-divisibility**: Number theory with p-adic valuations

These demonstrate the Aeneas workflow is mature and practical.

## Expected Benefits

1. **Catch bugs at compile time** instead of through numerical tests
2. **Confidence in novel computations** where we have no reference values
3. **Documentation as proofs** - the theorem statements ARE the specification
4. **Refactoring safety** - proofs break if semantics change

## Timeline Estimate

| Phase | Scope | Effort |
|-------|-------|--------|
| Setup | Aeneas pipeline, first proof | 1-2 days |
| Tensor proofs | Symmetry, contraction | 1 week |
| Calculus proofs | Gradients, Jacobians | 2 weeks |
| Linear algebra | Gaussian elimination | 1 week |
| Full coverage | All modules | Ongoing |

## References

- Aeneas ICFP 2022: "Aeneas: Rust verification by functional translation"
- Aeneas ICFP 2024: Borrow checking soundness
- arXiv:2107.09064: McAllister et al. - The physics we're implementing
- Mathlib: https://github.com/leanprover-community/mathlib4
