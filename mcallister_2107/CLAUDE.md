# McAllister Reproduction Pipeline

## CRITICAL: NO SHORTCUTS. NO CHEATING. EVER.

**Every value must be computed from scratch. Using a `.dat` file or hardcoded constant that you haven't computed earlier in the pipeline is STRICTLY FORBIDDEN.**

### The Two Parts of Every Script

Each `compute_*.py` script has TWO distinct parts:

1. **GENERAL FUNCTION** (`compute_X(cy, inputs...)`)
   - Computes the quantity FROM SCRATCH using only:
     - CYTools geometry (polytope, triangulation, intersection numbers)
     - Values computed by EARLIER pipeline steps
   - **NEVER** reads from McAllister's `.dat` files
   - **NEVER** uses hardcoded constants from the paper
   - Must work for ANY polytope, not just McAllister's 5 examples

2. **VALIDATION TEST** (`test_example(name, ...)`)
   - Calls the general function to compute the value
   - Loads McAllister's expected value from `.dat` file
   - **COMPARES** computed vs expected
   - Prints PASS/FAIL

```python
# CORRECT PATTERN
def compute_W0(cy, kappa, p, gv_invariants, g_s):
    """Compute W0 from scratch using racetrack formula."""
    # ... actual physics computation using ONLY computed inputs ...
    return W0_computed

def test_example(example_name):
    """Validate against McAllister's data."""
    # Compute from scratch
    W0_computed = compute_W0(cy, kappa, p, gv, g_s)

    # Load expected (ONLY for comparison)
    W0_expected = float((data_dir / "W_0.dat").read_text())

    # Compare
    ratio = W0_computed / W0_expected
    return {"pass": abs(ratio - 1.0) < 0.01}
```

```python
# WRONG - THIS IS CHEATING
def compute_W0(cy, kappa, p, gv_invariants):
    """Pretends to compute but actually cheats."""
    # WRONG: Reading from .dat file
    W0 = float((data_dir / "W_0.dat").read_text())
    return W0
```

### Why This Matters

The entire point is to build a pipeline that can evaluate ANY polytope for the GA search. If we cheat by reading McAllister's pre-computed values, we have:
- A pipeline that only works for 5 specific examples
- Zero confidence it's actually computing physics correctly
- Wasted effort that produces nothing useful

**Save an hour now by cheating = waste weeks later debugging garbage.**

### The Three-Layer Architecture

The pipeline has three layers that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: compute_*.py scripts (PURE FUNCTIONS)                 │
│  - Each has compute_X() that works from scratch                 │
│  - Each has test_example() that validates against .dat files    │
│  - Run standalone: `python compute_X.py` → "All 5 PASSED"       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Individual script tests (TRUST UPSTREAM .dat FILES)   │
│  - A script can load .dat files for UPSTREAM values             │
│  - Example: compute_V_string.py loads corrected_kahler_param.dat│
│  - This is OK because compute_kahler_param.py validated those   │
│  - Each script only validates ITS OWN computation               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: full_pipeline.py (END-TO-END, NO .dat FILES)          │
│  - Imports compute_X() functions from all scripts               │
│  - Chains them together: output of step N → input of step N+1   │
│  - ONLY loads model inputs (polytope, K, M, c_i, heights, basis)│
│  - Computes EVERYTHING else from scratch                        │
│  - Final validation against expected V₀                         │
└─────────────────────────────────────────────────────────────────┘
```

**Example: How compute_V_string.py works**

```python
# Layer 1: Pure function (no .dat files)
def compute_V_string(kappa, t, h11, h21):
    """Compute V_string = (1/6)κt³ - BBHL from scratch."""
    V = einsum('ijk,i,j,k->', kappa, t, t, t) / 6.0
    BBHL = zeta(3) * 2 * (h11 - h21) / (4 * (2*pi)**3)
    return V - BBHL

# Layer 2: Validation test (loads t from .dat, trusts compute_kahler_param.py)
def test_example(example_name):
    t = load_kahler_params(example_name)  # From corrected_kahler_param.dat
    V_computed = compute_V_string(kappa, t, h11, h21)
    V_expected = load_cy_vol(example_name)  # From cy_vol.dat
    return abs(V_computed / V_expected - 1.0) < 0.001
```

**The key insight:** In Layer 2 tests, loading `corrected_kahler_param.dat` is NOT cheating because:
1. `compute_kahler_param.py` computes those t values from scratch
2. `compute_kahler_param.py` validates they match `corrected_kahler_param.dat`
3. Therefore we can TRUST those .dat files for downstream tests

**Layer 3 (full_pipeline.py) is the final proof:** It imports all `compute_X()` functions and chains them without ANY .dat files (except model inputs). If it passes, the entire pipeline is validated end-to-end.

---

## Purpose

This directory contains the **gold standard validation pipeline** for computing the cosmological constant V₀ from a Calabi-Yau compactification. We are reproducing the results of McAllister et al. (arXiv:2107.09064), who achieved V₀ ≈ -5.5 × 10⁻²⁰³ for polytope 4-214-647.

**Why this matters:** Until we can exactly reproduce a published result, our entire GA search is meaningless. This is the ground truth.

## Strategy: Backporting to CYTools 2021

### The Problem

We initially built scripts using CYTools 2025 (latest). This caused endless headaches:

1. **Basis differences**: CYTools 2021 uses basis `[3,4,5,8]`, while 2025 uses `[5,6,7,8]` for the same polytope
2. **Flux transformation**: K, M vectors must be transformed between bases - easy to get wrong
3. **Triangulation differences**: Default triangulations can differ between versions
4. **Debugging nightmare**: When something doesn't match, is it physics or basis mismatch?

### The Solution

**Use CYTools 2021 (McAllister's version) for everything except GV invariants.**

- CYTools 2021 doesn't have `compute_gvs()` - that was added later
- For GV invariants only, we use CYTools 2025's `compute_gvs(min_points=20000)`
- Everything else uses 2021, matching McAllister's exact setup

This eliminates basis transformation complexity entirely. McAllister's K, M, basis, and all other data work directly.

## Directory Structure

### Subdirectories (WHERE SCRIPTS GO)

```
mcallister_2107/
├── 2021_cytools/           # Scripts using CYTools 2021 (McAllister's version)
│   ├── compute_triangulation.py
│   ├── compute_gv_invariants.py
│   ├── compute_derived_racetrack.py
│   └── ...
├── latest_cytools/         # Scripts using latest CYTools (for comparison/debugging)
│   └── ...
├── test_*.py               # Testing/debugging/trial scripts ONLY
├── debug_*.py              # Debugging scripts ONLY
└── CLAUDE.md               # This file
```

**CRITICAL: `compute_*.py` scripts go in `2021_cytools/` or `latest_cytools/`, NOT in the root `mcallister_2107/` directory.**

The root directory is ONLY for:
- Testing/debugging/trial scripts (`test_*.py`, `debug_*.py`)
- Documentation (`.md` files)
- One-off experiments

### Main Scripts (`compute_*.py`)

Each script follows the same pattern:

```python
#!/usr/bin/env python3
"""
Compute <quantity> for a Calabi-Yau threefold.

<Physics explanation>

Validation: Tests against McAllister's 5 examples.
"""

# Imports and paths (note: .parent.parent.parent because we're in 2021_cytools/)
CYTOOLS_2021 = Path(__file__).parent.parent.parent / "vendor/cytools_mcallister_2107"
DATA_BASE = Path(__file__).parent.parent.parent / "resources/small_cc_2107.09064_source/anc/paper_data"

# General-purpose function
def compute_<quantity>(cy, ...):
    """Compute <quantity> from CY geometry."""
    ...

# Unit test / validation
def test_example(example_name: str, ...):
    """Test against one McAllister example."""
    ...

def main():
    """Run validation against all 5 McAllister examples."""
    for example in MCALLISTER_EXAMPLES:
        result = test_example(example, ...)
    print("All 5 examples PASSED" if all_passed else "FAILED")

if __name__ == "__main__":
    main()
```

**Key insight:** Each script is BOTH:
1. A **reusable function** for the GA pipeline
2. A **unit test** that validates against known data

### Pipeline Steps (in order)

| Step | Script | Computes | Status |
|------|--------|----------|--------|
| 1-4 | `compute_triangulation.py` | Polytope, triangulation, model choices | ✅ |
| 5-8 | `compute_gv_invariants.py` | GV invariants N_q (uses 2025 CYTools) | ✅ |
| 9-11 | (inline in pipeline) | N_ab, p, e^{K₀} | ✅ |
| 12-14 | `compute_derived_racetrack.py` | g_s, W₀ via racetrack | ⚠️ |
| 15 | `compute_target_tau.py` | Target divisor volumes τ_i | ✅ |
| 15 | `compute_chi_divisor.py` | χ(D_i) via Braun formula | ✅ |
| 16 | `compute_kklt_iterative.py` | Solve T_i(t) = τ_i for t | ⚠️ |
| 17 | `compute_V_string.py` | V_string with BBHL correction | ✅ |
| 18 | `full_pipeline.py` | V₀ = -3 e^{K₀} (g_s⁷/(4V)²) W₀² | 🎯 |

### Supporting Scripts

- `compute_basis_transform.py` - Transform K, M between bases (only needed for 2025 CYTools)
- `compute_c_i.py` - Dual Coxeter numbers (1 for D3, 6 for O7)
- `compute_rigidity_combinatorial.py` - Divisor rigidity via dual face interior points
- `compute_divisor_cohomology.py` - h^i(D) via cohomCalg

### Test/Debug Scripts (DO NOT DELETE)

These document edge cases and are searchable via `rg`:

- `test_superset_hypothesis.py` - Proves min_points convergence for GV
- `test_max_deg.py` - Shows max_deg is intractable vs min_points
- `debug_missing_curves.py` - Analyzed 5-113-4627 curve discrepancy
- `benchmark_*.py` - Performance measurements

### Documentation

- `REPRODUCTION_OUTLINE.md` - Complete pipeline with formulas and status
- `GV_INVARIANTS_MIN_POINTS_CONVERGENCE.md` - Why min_points=20000, not max_deg
- `CURVE_DISCREPANCY.md` - 5-113-4627 investigation and resolution
- `LATEST_CYTOOLS_CONVERSION_RESULT.md` - Basis transformation details (for reference)

## The Five McAllister Examples

| Folder | h²¹ | h¹¹ | Notes |
|--------|-----|-----|-------|
| `4-214-647` | 4 | 214 | Primary test case |
| `5-113-4627-main` | 5 | 113 | |
| `5-113-4627-alternative` | 5 | 113 | Different flux choice |
| `5-81-3213` | 5 | 81 | |
| `7-51-13590` | 7 | 51 | |

All scripts test against ALL 5 examples. A script is only "done" when all 5 pass.

## Key Technical Details

### GV Invariants: min_points, NOT max_deg

```python
# CORRECT - fast, matches McAllister
cy.compute_gvs(min_points=20000)  # ~16s for 4-214-647

# WRONG - intractable, combinatorial explosion
cy.compute_gvs(max_deg=300)  # killed after 4+ min
```

McAllister's "max degree 280" was the highest curve *sampled*, not a cutoff.

### Precision Requirements

| Quantity | Precision | Why |
|----------|-----------|-----|
| κ_ijk, GV | exact int | Combinatorial |
| N, p, e^{K₀} | float64 | Linear algebra |
| g_s, W₀ | mpmath 150+ | W₀ ~ 10⁻⁹⁰ |
| V₀ | mpmath | W₀² ~ 10⁻¹⁸⁰ |

### CYTools Import Pattern

```python
# For most computations - use 2021
import sys
sys.path.insert(0, str(CYTOOLS_2021))
from cytools import Polytope

# For GV only - use 2025
sys.path.insert(0, str(CYTOOLS_LATEST))
from cytools import Polytope  # after clearing modules
```

### Multiprocessing Warning

CYTools uses multiprocessing (e.g., `compute_gvs()`). Scripts MUST:
1. Be written to actual `.py` files (NEVER heredoc/stdin)
2. Have `if __name__ == "__main__":` guard

## End Goal

`full_pipeline.py` will:

1. Take a polytope + (K, M, orientifold) as input
2. Compute V₀ end-to-end using only CYTools 2021 (+ 2025 for GV)
3. Validate against all 5 McAllister examples
4. Serve as the physics evaluation function for the GA

Once this works, we can trust the GA to search ~400M polytopes for configurations matching our universe.

## What NOT to Do

1. **Don't CHEAT** - never read McAllister's `.dat` files in compute functions (see top of this file)
2. **Don't REIMPLEMENT** - import from sibling `compute_*.py` scripts, don't duplicate code
3. **Don't put `compute_*.py` in root** - they go in `2021_cytools/` or `latest_cytools/`
4. **Don't use 2025 CYTools for anything except `compute_gvs()`**
5. **Don't transform K, M between bases** - just use 2021 basis directly
6. **Don't delete test/debug scripts** - they're documentation
7. **Don't use max_deg** - always use min_points for GV
8. **Don't use float64 for W₀** - it's 10⁻⁹⁰, needs mpmath
9. **Don't use hardcoded constants from the paper** - compute them from scratch
10. **Don't use scipy.optimize for KKLT** - use path-following with LINEAR solves (see below)

---

## CRITICAL: compute_kahler_param.py Algorithm

**USE PREDICTOR-CORRECTOR, NOT PURE NEWTON OR SCIPY**

See FORMULAS.md Section 7.6 and `BRANCH_DISCOVERY_NOTES.md` for full details.

### The Problem
Find t such that τ(t) = τ_target where τ_i = (1/2) κ_ijk t^j t^k

### WRONG Approaches

**Pure Newton (n_steps=1):** FAILS from random initialization
```python
# DON'T DO THIS - Newton fails from random init
# Line search shrinks step to zero, gets stuck
J @ delta_t = -(tau(t) - tau_target)
t = t + delta_t  # Overshoots or wrong direction
```

**Scipy optimization:** Also fails
```python
# DON'T DO THIS - scipy treats it as nonlinear optimization
scipy.optimize.least_squares(lambda t: tau(t) - tau_target, t0)
```

### CORRECT Approach: Predictor-Corrector

**Optimal step count: n_steps=32** (tested on h¹¹=81)

| n_steps | Success Rate | Notes |
|---------|--------------|-------|
| 1-8 | 0% | Pure Newton fails |
| 16 | 40% | Minimum working |
| **32** | **70%** | **Optimal** |
| 64+ | 60-67% | Diminishing returns |

```python
def solve_predictor_corrector(kappa, tau_target, t_init, n_steps=32):
    t = t_init
    tau_init = kappa.compute_tau_kklt(t)

    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_step = (1 - alpha) * tau_init + alpha * tau_target

        # PREDICTOR: Euler step toward tau_step
        J = kappa.compute_jacobian_kklt(t)
        epsilon = np.linalg.lstsq(J, tau_step - kappa.compute_tau_kklt(t))[0]
        t_pred = t + epsilon

        # CORRECTOR: Newton iterations to converge exactly to tau_step
        t = t_pred
        for _ in range(5):  # max 5 Newton corrections
            residual = kappa.compute_tau_kklt(t) - tau_step
            if np.linalg.norm(residual) / np.linalg.norm(tau_step) < 1e-8:
                break
            J = kappa.compute_jacobian_kklt(t)
            delta = np.linalg.lstsq(J, -residual)[0]
            if np.linalg.norm(delta) > 1.0:
                delta = delta / np.linalg.norm(delta)  # Damping
            t = t + delta

    return t
```

### Why Predictor-Corrector Works
- Interpolates τ from τ_init to τ_target in small steps
- Each step: Predictor (Euler) + Corrector (Newton to converge exactly)
- Stays near solution manifold throughout
- 70% success rate with n_steps=32, random init

### Why Pure Newton Fails
- Newton assumes linearization is globally accurate
- From random init, Newton direction points wrong way
- Backtracking line search shrinks step to zero → stuck
- Need to be "close enough" to a solution for Newton to work

### Why Scipy Fails
- Treats as unconstrained nonlinear optimization over 214 variables
- Jacobian has rank ~65 (not 214) → 149-dim nullspace
- No path structure → diverges wildly
- Orders of magnitude slower

---

## basis.dat vs kklt_basis.dat

**These are DIFFERENT sets!** For 4-214-647:
- `basis.dat`: 214 divisors forming the computational h11 basis
- `kklt_basis.dat`: 214 divisors contributing to superpotential
- **Shared:** 210 divisors
- **Only in basis:** [1, 2, 46, 130]
- **Only in kklt_basis:** [8, 9, 10, 17]

### Computing τ for non-basis KKLT divisors

For basis divisors: `τ_i = (1/2) κ_ijk t^j t^k` using `in_basis=True`

For non-basis divisors (points 8, 9, 10, 17):
```python
# Use intersection_numbers(in_basis=False) for mixed indices
τ_a = (1/2) Σ_{j,k} κ_{a, basis[j], basis[k]} t^j t^k
```

This was verified: using McAllister's t values, we can compute τ for ALL 214 KKLT divisors with RMS error < 0.0001.
