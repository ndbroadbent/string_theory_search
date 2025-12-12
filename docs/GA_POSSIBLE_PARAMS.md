# GA Possible Parameters: Ideas and Suggestions

This document tracks potential parameters and optimizations for the GA search that need further investigation or validation.

---

## GV Invariant Computation Depth

### The Problem

Gopakumar-Vafa invariants exist for infinitely many curve classes. We can't compute them all. Higher-degree curves are exponentially suppressed by:

```
e^{-2π Im(τ) (q·p)}
```

So we need a convergence criterion: compute curves until additional terms contribute below some threshold to W₀.

### CYTools Parameters

CYTools `compute_gvs()` has two relevant parameters:

- `min_points`: Minimum lattice points to sample (default: 50). More points = higher degree curves found.
- `max_deg`: Maximum curve degree to compute (optional). Explicit cutoff.

### McAllister's Data

From `dual_curves.dat` files:

| Example | # Curves | Max Degree |
|---------|----------|------------|
| 4-214-647 | 5177 | 280 |
| 5-113-4627-main | 1009 | ~100 |
| 5-113-4627-alternative | 1009 | ~100 |
| 5-81-3213 | ~800 | ~80 |
| 7-51-13590 | ~500 | ~60 |

McAllister computed until contributions to W₀ were negligible (~10^{-100} scale).

### Convergence Criterion

The physics-driven approach:

1. Compute GV invariants in batches (increasing degree)
2. For each batch, compute contribution to W₀:
   ```
   δW₀ = ζ Σ_{q in batch} (M·q) N_q Li₂(e^{-2π Im(τ) (q·p)})
   ```
3. Stop when |δW₀| / |W₀_running| < threshold (e.g., 10^{-10})

This ensures we capture all curves that matter for the physics.

### Is This a GA Parameter?

**No.** The convergence threshold is physics-driven, not a hyperparameter to optimize.

The threshold must be small enough that:
1. W₀ is computed to sufficient precision
2. Racetrack stabilization finds the correct minimum
3. V₀ = -3 e^K₀ (g_s^7 / (4V)^2) W₀^2 is accurate

A threshold of 10^{-10} relative contribution is conservative and should work universally.

**What IS a GA parameter:** How much compute time to spend per polytope. If GV computation is slow, the outer GA might learn to:
- Prefer polytopes with smaller h11 (faster GV computation)
- Use coarser initial estimates before full computation
- Batch similar polytopes together

### Implementation

```python
def compute_gv_until_convergence(cy, M, p, Im_tau, rel_threshold=1e-10):
    """
    Compute GV invariants until convergence.

    Args:
        cy: CalabiYau object
        M: Flux vector
        p: Flat direction
        Im_tau: Imaginary part of τ (= 1/g_s)
        rel_threshold: Stop when new terms contribute < this fraction

    Returns:
        Dict of {curve_class: N_q} for all relevant curves
    """
    all_gv = {}
    W0_running = 0.0

    for max_deg in [50, 100, 150, 200, 250, 300]:
        # Compute GVs up to this degree
        gv_obj = cy.compute_gvs(max_deg=max_deg)
        new_gv = extract_new_curves(gv_obj, all_gv)

        if not new_gv:
            break

        # Compute contribution from new curves
        delta_W0 = compute_W0_contribution(new_gv, M, p, Im_tau)

        # Check convergence
        if W0_running != 0 and abs(delta_W0 / W0_running) < rel_threshold:
            break

        all_gv.update(new_gv)
        W0_running += delta_W0

    return all_gv
```

### Validation

The test is simple: our convergence parameters must produce results that match or exceed McAllister's data for all 5 examples.

```python
def test_gv_convergence():
    for example in MCALLISTER_EXAMPLES:
        our_gv = compute_gv_until_convergence(...)
        mcallister_gv = load_mcallister_gv(example)

        # We must have computed AT LEAST all curves McAllister has
        for curve in mcallister_gv:
            assert curve in our_gv
            assert our_gv[curve] == mcallister_gv[curve]
```

---

## Future Parameters to Investigate

### Flux Search Range

Current: K, M integers in [-15, 15]

Questions:
- Is this range sufficient for all polytopes?
- Should range scale with h11?
- Are there physics constraints that narrow the range?

### Triangulation Selection

Current: Use McAllister's triangulation (from `dual_simplices.dat`)

Questions:
- How many triangulations exist per polytope?
- Do different triangulations give different physics?
- Should we search over triangulations?

### Orientifold Involution

Current: Fixed (from McAllister's data)

Questions:
- How to enumerate valid involutions?
- Computational cost of checking involution constraints?
- Should this be part of the GA genome?

---

## References

1. CYTools documentation: `compute_gvs()` method
2. McAllister arXiv:2107.09064 Section 5.3: GV computation
3. Demirtas arXiv:1912.10047: Racetrack construction
