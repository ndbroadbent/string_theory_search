# The Cosmological Constant

## Our Universe's Value

### Latest Measurements (PDG 2025 / Planck PR4 2024)

From **Planck 2018** satellite measurements ([arXiv:1807.06209](https://arxiv.org/abs/1807.06209)),
with updated uncertainties from **Planck PR4** ([A&A 2024](https://www.aanda.org/articles/aa/full_html/2024/02/aa48015-23/aa48015-23.html)):

| Parameter | Value | Uncertainty (68% CL) | Source |
|-----------|-------|---------------------|--------|
| Ω_Λ (dark energy density) | 0.685 | ± 0.007 | Planck 2018 |
| H₀ (Hubble constant) | 67.64 km/s/Mpc | ± 0.52 | Planck PR4 2024 |
| Ω_m (matter density) | 0.315 | ± 0.007 | Planck 2018 |
| h | 0.6764 | ± 0.0052 | Planck PR4 2024 |
| σ₈ | 0.811 | ± 0.006 | Planck 2018 |

**PDG 2025 best-fit values** (from TT+TE+EE+lowE+lensing):
- h = 0.674 ± 0.005
- σ₈ = 0.811 ± 0.006
- Ω_m = 0.315 ± 0.007
- Ω_Λ = 0.685 ± 0.007

### Derived Values

**Dark energy density (SI units):**
```
ρ_Λ = Ω_Λ × ρ_critical
    = Ω_Λ × 3H₀²/(8πG)
    = 5.85 × 10⁻²⁷ kg/m³
    ≈ 5.36 × 10⁻¹⁰ J/m³
    ≈ 3.35 GeV/m³
```

**Cosmological constant in various units:**
```
Λ = 1.1056 × 10⁻⁵² m⁻²
  = 2.888 × 10⁻¹²² (Planck units, i.e., l_P⁻²)
  ≈ 10⁻³⁵ s⁻²
```

### Conversion to Planck Units

The Planck density is:
```
ρ_Planck = c⁵/(ℏG²) = 5.178 × 10⁹⁶ kg/m³
```

The dimensionless cosmological constant in Planck units:
```
Λ = 8π × (ρ_Λ / ρ_Planck)
  = 8π × (5.85 × 10⁻²⁷ / 5.178 × 10⁹⁶)
  = 8π × 1.13 × 10⁻¹²³
  = 2.888 × 10⁻¹²²
```

**Best estimate for our search:**
```python
UNIVERSE_LAMBDA = 2.888e-122  # Planck units (Mpl⁴)
TARGET_LOG_V0 = -121.54       # log₁₀(Λ)
```

## IMPORTANT: 2025 DESI Results - Dark Energy May Evolve!

### Evidence for Time-Varying Dark Energy

The **DESI collaboration** (March 2025) has found **3.9σ evidence** that dark energy density is evolving over time:

> "We now have the first hint in over 20 years that dark energy might be changing, and if it is evolving, it must be something new, which would change our understanding of fundamental physics."

Key results:
- [DESI DR2 (March 2025)](https://newscenter.lbl.gov/2025/03/19/new-desi-results-strengthen-hints-that-dark-energy-may-evolve/): 3.5σ preference for evolving dark energy vs constant Λ
- [Combined DES+DESI+Planck analysis](https://physicalsciences.uchicago.edu/news/article/reconsidering-the-cosmological-constant/): 3.2σ preference for w₀w_a model over ΛCDM

### What This Means

If dark energy is evolving (w ≠ -1), this could be:
1. **Good news for string theory**: Quintessence models predict evolving dark energy
2. **Bad news for pure ΛCDM**: Simple cosmological constant may not be correct

**However**: Physics requires **5σ** to accept revolutionary claims. Current evidence is suggestive but not definitive. The Vera Rubin Observatory (LSST) and continued DESI observations will be decisive.

### For Our Pipeline

**Keep using Λ = 2.888 × 10⁻¹²² Mpl⁴ as the target.** This remains the PDG/Planck value. If DESI results are confirmed at 5σ, we would need to target a time-dependent dark energy model instead.

## The Cosmological Constant Problem

This is called "the worst theoretical prediction in the history of physics."

**The problem:** If we calculate the vacuum energy from quantum field theory (summing zero-point energies up to the Planck scale), we get:
```
ρ_vacuum^QFT ~ ρ_Planck ~ 10⁹⁶ kg/m³
```

**The observation:**
```
ρ_Λ^observed ~ 10⁻²⁷ kg/m³
```

**The discrepancy:**
```
ρ_QFT / ρ_observed ~ 10¹²³
```

This 123 orders of magnitude mismatch is the cosmological constant problem.

## Why This Matters for String Theory

In the string theory landscape, the cosmological constant arises from:

```
V₀ = -3 eᴷ |W|²
```

Where:
- **K** = Kähler potential (depends on moduli)
- **W** = superpotential (flux + non-perturbative contributions)

McAllister et al. ([arXiv:2107.09064](https://arxiv.org/abs/2107.09064)) achieved V₀ ~ 10⁻²⁰³ for a specific Calabi-Yau compactification. Our goal is to find configurations that produce V₀ ~ 10⁻¹²².

### The Gap

| Source | log₁₀(|V₀|) |
|--------|-------------|
| Our Universe | -121.5 |
| McAllister's best | -203 |
| Typical string vacua | -200 to -600 |

The challenge: McAllister found vacua that are **too small** by ~80 orders of magnitude. We need to find the "Goldilocks" configurations.

## References

1. **Planck 2018 Results VI** - Cosmological Parameters
   - [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
   - [A&A 641, A6 (2020)](https://www.aanda.org/articles/aa/full_html/2020/09/aa33910-18/aa33910-18.html)

2. **Planck PR4 (2024)** - Final Data Release with 10-20% tighter constraints
   - [A&A 2024](https://www.aanda.org/articles/aa/full_html/2024/02/aa48015-23/aa48015-23.html)
   - H₀ = 67.64 ± 0.52 km/s/Mpc

3. **PDG 2025** - Cosmological Parameters Review
   - [pdg.lbl.gov/2025](https://pdg.lbl.gov/2025/reviews/rpp2024-rev-cosmological-parameters.pdf)
   - Ω_Λ = 0.685 ± 0.007

4. **DESI DR2 (2025)** - Evidence for evolving dark energy
   - [Berkeley Lab News](https://newscenter.lbl.gov/2025/03/19/new-desi-results-strengthen-hints-that-dark-energy-may-evolve/)
   - 3.9σ evidence for time-varying dark energy

5. **McAllister et al.** - Small Cosmological Constants in String Theory
   - [arXiv:2107.09064](https://arxiv.org/abs/2107.09064)

## Physical Constants Used

From CODATA 2018:
```
G   = 6.67430(15) × 10⁻¹¹ m³ kg⁻¹ s⁻²
c   = 299792458 m/s (exact)
ℏ   = 1.054571817 × 10⁻³⁴ J s
l_P = √(ℏG/c³) = 1.616255 × 10⁻³⁵ m (Planck length)
t_P = √(ℏG/c⁵) = 5.391247 × 10⁻⁴⁴ s (Planck time)
m_P = √(ℏc/G)  = 2.176434 × 10⁻⁸ kg (Planck mass)
```

Conversion:
```
1 Mpc = 3.08567758149 × 10²² m
H₀ = 67.64 km/s/Mpc = 2.190 × 10⁻¹⁸ s⁻¹
```
