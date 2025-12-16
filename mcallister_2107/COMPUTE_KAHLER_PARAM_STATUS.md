# compute_kahler_param.py Status

## Goal

Compute Kähler moduli t from scratch using KKLT stabilization, validating against `corrected_kahler_param.dat`.

## The KKLT Equation

From McAllister eq 5.12:
```
τ_i(t) = (1/2) κ_ijk t^j t^k = c_i/c_τ + χ(D_i)/24 - GV_corrections
```

Where:
- t^i are the Kähler moduli (h11 = 214 values for 4-214-647)
- τ_i are divisor volumes
- κ_ijk are intersection numbers
- c_i are dual Coxeter numbers (1 for D3-instanton, 6 for O7-plane gaugino condensation)
- c_τ = 2π / (g_s × ln(W₀⁻¹))
- χ(D_i) is divisor Euler characteristic
- GV_corrections involve polylogarithms of e^(-2πq·t)

## CORRECT ALGORITHM: Path-Following with Linear Solves

**From McAllister Section 5.2 - DO NOT USE SCIPY OPTIMIZATION**

At each step, solve a **LINEAR system** (not nonlinear optimization):
```
κᵢⱼₖ tʲ εᵏ = Δτᵢ   (linear in ε!)
```

```python
def solve_kklt_path_following(kappa, tau_target, n_steps=200):
    t = initialize_t()
    tau_init = compute_tau(kappa, t)

    for m in range(n_steps):
        alpha = (m + 1) / n_steps
        tau_step = (1 - alpha) * tau_init + alpha * tau_target

        tau_current = compute_tau(kappa, t)
        delta_tau = tau_step - tau_current

        # J_ik = κ_ijk t^j
        J = compute_jacobian(kappa, t)

        # ONE LINEAR SOLVE per step!
        epsilon = np.linalg.lstsq(J, delta_tau, rcond=1e-8)[0]
        t = t + epsilon

    return t
```

**Why this is fast:**
- Each step is ONE linear solve: O(h11³)
- Total: ~200 linear solves → runs in seconds

**Why scipy.optimize FAILS:**
- Treats as unconstrained nonlinear optimization
- Jacobian has rank ~65 (not 214) → 149-dim nullspace
- No path structure → diverges
- Orders of magnitude slower

## basis.dat vs kklt_basis.dat

**These are DIFFERENT sets!** For 4-214-647:
- `basis.dat`: 214 divisors forming the computational h11 basis
- `kklt_basis.dat`: 214 divisors contributing to superpotential
- **Shared:** 210 divisors
- **Only in basis:** [1, 2, 46, 130]
- **Only in kklt_basis:** [8, 9, 10, 17]

## Computing τ for Non-Basis KKLT Divisors

For basis divisors: `τ_i = (1/2) κ_ijk t^j t^k` using `in_basis=True`

For non-basis divisors (points 8, 9, 10, 17):
```python
# Use intersection_numbers(in_basis=False) for mixed indices
τ_a = (1/2) Σ_{j,k} κ_{a, basis[j], basis[k]} t^j t^k
```

**Verified:** Using McAllister's t values, τ for ALL 214 KKLT divisors matches with RMS < 0.0001.

## File Dependencies

```
compute_kahler_param.py
├── compute_target_tau.py (for compute_c_tau)
├── compute_rigidity_combinatorial.py (for χ(O_D) via point location)
└── cytools (for Polytope, intersection_numbers)
```

## Test Command

```bash
cd /Users/ndbroadbent/code/string_theory
uv run python mcallister_2107/2021_cytools/compute_kahler_param.py
```
