# DESI Dark Energy: The Actual Numbers

## The Bottom Line First

**DESI measured TWO numbers that describe how dark energy changes over time:**

| Parameter | ΛCDM (constant) | DESI Measurement | Meaning |
|-----------|-----------------|------------------|---------|
| w₀ | -1.0 | **-0.45 ± 0.21** | Dark energy EoS today |
| wₐ | 0.0 | **-1.8 ± 0.6** (approx) | Rate of change |

The deviation from (w₀, wₐ) = (-1, 0) is at **~3σ significance**.

---

## What These Numbers Mean Physically

### The CPL Parameterization
```
w(a) = w₀ + wₐ(1 - a)
```
Where:
- `a` = scale factor of the universe (a = 1 today, a = 0 at Big Bang)
- `w` = equation of state = pressure/density

### For a cosmological constant (Λ):
- w = -1 always (constant pressure)
- w₀ = -1, wₐ = 0

### For DESI result (example: w₀ = -0.7, wₐ = -1.0):
```
Today (a = 1):      w = -0.7 + (-1.0)(0) = -0.7
At a = 0.5 (z=1):   w = -0.7 + (-1.0)(0.5) = -1.2
At a = 0.25 (z=3):  w = -0.7 + (-1.0)(0.75) = -1.45
Early times (a→0): w → -0.7 + (-1.0)(1) = -1.7
```

So dark energy was **more negative** (w < -1, "phantom-like") in the past and is becoming **less negative** (approaching w = -1 or higher) now.

---

## How Much Data Is This?

**The DESI result is NOT petabytes of raw data distilled to 2 numbers.**

Rather:
- Raw data: ~millions of galaxy spectra, ~PB scale
- Processed to: Baryon Acoustic Oscillation (BAO) distance measurements at different redshifts
- Combined with: CMB (Planck), Supernovae (Pantheon+/Union3/DES Y5)
- Fit to: CPL model with 2 free parameters (w₀, wₐ)

The "3σ" is the statistical significance of the deviation from (w₀, wₐ) = (-1, 0).

---

## Can We Trace w(t) Through Cosmic History?

Yes! Given w₀ and wₐ, we can compute w at any time:

### Converting between a, z, and t

| Quantity | Formula | Today | z=1 | z=3 | z=10 | z=1000 (CMB) |
|----------|---------|-------|-----|-----|------|--------------|
| Scale factor a | a = 1/(1+z) | 1 | 0.5 | 0.25 | 0.091 | 0.001 |
| Redshift z | z = 1/a - 1 | 0 | 1 | 3 | 10 | 1000 |
| Age (approx) | - | 13.8 Gyr | 5.9 Gyr | 2.2 Gyr | 0.5 Gyr | 380,000 yr |

### w(z) for Different Models

| Redshift z | Age | a | Λ (w=-1) | DESI (w₀=-0.7, wₐ=-1) |
|------------|-----|---|----------|----------------------|
| 0 (today) | 13.8 Gyr | 1.0 | -1.0 | **-0.7** |
| 0.5 | 8.6 Gyr | 0.67 | -1.0 | **-1.03** |
| 1 | 5.9 Gyr | 0.5 | -1.0 | **-1.2** |
| 2 | 3.3 Gyr | 0.33 | -1.0 | **-1.37** |
| 3 | 2.2 Gyr | 0.25 | -1.0 | **-1.45** |
| 10 | 0.5 Gyr | 0.091 | -1.0 | **-1.61** |

**Key observation:** The "phantom crossing" (w crossing -1) happens around z ≈ 0.3-0.5, roughly 4-5 billion years ago.

---

## What About Λ at Different Times?

**Critical point:** The DESI result is about w(t), not Λ(t)!

For a cosmological constant:
- Λ is CONSTANT (doesn't change)
- But ρ_Λ = Λ/(8πG) is also constant
- While ρ_matter ∝ a⁻³ decreases

For quintessence with w(t):
- The dark energy DENSITY ρ_DE(t) changes:
```
ρ_DE(a) = ρ_DE,0 × exp(3 ∫₁^a (1 + w(a'))/a' da')
```

For CPL parameterization:
```
ρ_DE(a) = ρ_DE,0 × a^(-3(1+w₀+wₐ)) × exp(-3wₐ(1-a))
```

### Dark Energy Density Over Time (normalized to today)

| z | Age | ρ_DE/ρ_DE,0 (Λ) | ρ_DE/ρ_DE,0 (DESI) |
|---|-----|-----------------|-------------------|
| 0 | 13.8 Gyr | 1.0 | 1.0 |
| 1 | 5.9 Gyr | 1.0 | ~0.5 |
| 3 | 2.2 Gyr | 1.0 | ~0.15 |
| 10 | 0.5 Gyr | 1.0 | ~0.01 |

So if DESI is right, dark energy was **weaker** in the early universe.

---

## The 13.8 Billion Year Question

**Can we run a simulation from t=0 to t=13.8 Gyr and arrive at Λ_today?**

In principle, YES, but:

1. **For ΛCDM:** Λ is constant, so Λ(t) = Λ₀ = 2.888 × 10⁻¹²² Mₚₗ⁴ always

2. **For quintessence:** We'd solve:
   ```
   φ̈ + 3Hφ̇ + V'(φ) = 0
   H² = (ρ_matter + ρ_radiation + ρ_φ)/(3Mₚₗ²)
   ```
   Starting from initial conditions at some early time, evolving to today.

3. **The test for a CY configuration:**
   - Compute V(φ) from the string compactification
   - Evolve the cosmology for 13.8 Gyr
   - Check: Does w₀ ≈ -0.7 and wₐ ≈ -1.0 at the end?

---

## A Concrete Test for Our CY Search

**Given a Calabi-Yau configuration (polytope, fluxes, orientifold), we want to test:**

### Old Test (Static Λ):
```python
def test_static_lambda(cy_config):
    V0 = compute_vacuum_energy(cy_config)  # Our formula
    return abs(V0 - 2.888e-122) < tolerance
```

### New Test (Dynamical w):
```python
def test_dynamical_de(cy_config):
    # 1. Get the scalar potential V(φ) for the rolling modulus
    V, dV_dphi = compute_modulus_potential(cy_config)

    # 2. Evolve from early times to today
    w_history = evolve_cosmology(V, t_initial=1e-30, t_final=13.8e9 * year)

    # 3. Fit to CPL parameterization
    w0_computed, wa_computed = fit_cpl(w_history)

    # 4. Compare to DESI
    w0_target = -0.7  # approximate
    wa_target = -1.0  # approximate

    return (abs(w0_computed - w0_target) < sigma_w0 and
            abs(wa_computed - wa_target) < sigma_wa)
```

---

## Summary: What DESI Gives Us

| What | Value | Precision | Data Volume |
|------|-------|-----------|-------------|
| w₀ | -0.45 to -0.87 | ±0.05 to ±0.21 | 2 numbers |
| wₐ | -1.0 to -1.8 | ±0.4 to ±0.6 | (depends on dataset combo) |
| Significance | 2.5-3.1σ | - | |

**The assertion for a test:**
```
IF w₀ = -0.7 ± 0.2 AND wₐ = -1.0 ± 0.5
THEN the dark energy evolves as: w(z) = -0.7 - 1.0 × z/(1+z)
```

This can be checked against ANY model that predicts w(z).

---

## Open Questions for Our Project

1. **Can we compute V(φ) for a rolling Kähler modulus from CYTools?**
   - We can compute τ(t) = ∂V/∂T for stabilized moduli
   - For rolling modulus, need the full potential shape

2. **What initial conditions at the Big Bang?**
   - String theory doesn't specify φ(t=0)
   - May need to scan over initial conditions

3. **Is 13.8 Gyr evolution computationally feasible?**
   - The cosmological equations are ODEs, not PDEs
   - Should be fast (~seconds per configuration)

4. **What if DESI significance drops?**
   - DR2 → DR3 may change numbers
   - We should support both Λ and (w₀, wₐ) targets
