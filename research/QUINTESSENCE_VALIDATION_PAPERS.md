# Quintessence Validation Papers

*Generated: 2026-01-01*
*Purpose: Find papers with explicit calculations to validate our quintessence module*

## The Goal

Like McAllister (arXiv:2107.09064) gave us explicit CY configurations with published V₀, W₀, g_s values for KKLT validation, we need equivalent "ground truth" papers for quintessence.

---

## Key Finding: arXiv:2407.03405 (Cicoli et al. 2024)

**"From Inflation to Quintessence: a History of the Universe in String Theory"**

**THIS IS OUR QUINTESSENCE MCALLISTER!**

### Why This Paper Is Perfect

1. **Same framework**: Type IIB, LVS, Kähler moduli - same physics as McAllister
2. **Explicit numerical example** (Section 4.5):
   - ⟨τ₁⟩ ≈ 1.324 (fibration modulus)
   - ⟨τ₂⟩ ≈ 1122 (base modulus)
   - V ≈ 900 (CY volume in string units)
   - g_s = 0.1
   - |W₀| = 1
3. **Dark energy scale**: Λ₁⁴ ≈ 10⁻¹²⁰ Mₚ⁴ ✓
4. **Quintessence axion**:
   - f₁ ≈ 0.085 Mₚ (decay constant)
   - m₁ ~ 10⁻³² eV (mass)
5. **LaTeX source available** at `resources/inflation_to_quintessence_2407.03405.tex`

### The Model

K3-fibred Calabi-Yau with two Kähler moduli:
```
V = √τ₁ τ₂ - k τ_s^(3/2)    (Fibre Inflation volume form)
```

Late-time potential (quintessence):
```
V(φ_L) ≈ Λ₁⁴ [1 - cos(φ_L / f_L)]
```

Where f_L is enhanced via "axion alignment" (KNP mechanism).

### Validation Tests We Can Build

1. **Volume stabilization**: Given g_s, W₀, can we reproduce ⟨τ₁⟩, ⟨τ₂⟩?
2. **Dark energy scale**: From τ values, compute Λ₁⁴ ≈ 10⁻¹²⁰
3. **Decay constant**: Compute f₁ from τ₁
4. **Slow-roll**: Verify ε, η << 1

---

## Secondary Resource: ScalPy (arXiv:1503.02407)

**"ScalPy: A Python Package For Late Time Scalar Field Cosmology"**

**GitHub**: https://github.com/sum33it/scalpy
**PyPI**: `pip install scalpy`

### What It Does

- Solves cosmological ODEs for scalar fields (quintessence, tachyon, Galileon)
- Computes observables: H(z), D_L(z), D_A(z), growth rate
- Supports exponential and power-law potentials
- Integrates with emcee for MCMC

### How We Can Use It

1. **Validation**: Run ScalPy with V(φ) from string theory, compare w(z) output
2. **CPL fitting**: Use ScalPy's machinery to fit (w₀, wₐ) from our potential
3. **Benchmark**: Compare our cosmology solver against ScalPy

---

## Other Useful Papers

### arXiv:2112.10779 + 2112.10783 (Cicoli et al. 2022)
**"Quintessence and the Swampland"** (parametric + numerical control)

Key result: **Slow-roll is NOT possible in asymptotic moduli space**
- Must work in interior (finite volume, finite coupling)
- This is where McAllister works!
- Gives theoretical constraints on viable quintessence

### arXiv:2410.21243 (Butenschoen et al. 2024)
**"Cosmological tests of quintessence in quantum gravity"**

- MCMC analysis of hilltop quintessence
- Best-fit parameters against Planck + DESI
- Has corner plots in source files
- Could extract likelihood contours

### arXiv:2511.23463 (KMIX - Toomey et al. 2025)
**"Kinetically Mixed Axion-Dilaton Quintessence in Light of DESI DR2"**

- Already in our docs
- Shows phantom crossing as "illusion" from kinetic mixing
- String-motivated two-field model

### arXiv:2506.21542 (2025)
**"Quintessence and phantoms in light of DESI 2025"**

- Latest DESI DR2 analysis
- Higgs-like potential fits
- Good observational constraints

---

## Comparison: McAllister vs Cicoli

| Aspect | McAllister 2107.09064 | Cicoli 2407.03405 |
|--------|----------------------|-------------------|
| **Goal** | Small Λ (static) | Quintessence (dynamic) |
| **CY type** | h²¹=4, h¹¹=214 | K3-fibred (LVS) |
| **Framework** | KKLT | Fibre Inflation + LVS |
| **Data files** | ✓ Ancillary .dat | ✗ Numbers in text |
| **W₀** | 10⁻⁹⁰ (tiny) | 1 (natural) |
| **g_s** | 0.009 | 0.1 |
| **Volume** | ~4700 | ~900 |
| **Dark energy** | V₀ = -5.5×10⁻²⁰³ | Λ₁⁴ = 10⁻¹²⁰ |
| **Moduli** | All stabilized | τ₁, τ₂ fixed, axions roll |

### Key Insight

McAllister achieves tiny Λ via tiny W₀ (the 10⁻⁹⁰ conspiracy).
Cicoli achieves dark energy scale via poly-instanton suppression with natural O(1) parameters.

---

## Immediate Action Items

1. **Read Cicoli 2407.03405 in detail**
   - Extract all numerical constraints
   - Understand the K3-fibred volume form
   - Map to our CYTools framework

2. **Install and test ScalPy**
   ```bash
   uv add scalpy
   # Test with exponential potential
   ```

3. **Build validation test case**
   - Input: τ₁=1.324, τ₂=1122, g_s=0.1, W₀=1
   - Expected: Λ⁴ ≈ 10⁻¹²⁰, f ≈ 0.085 Mₚ

4. **Understand LVS vs KKLT**
   - LVS uses α' corrections (ξ/V³ term)
   - Different stabilization mechanism
   - May need separate code path

---

## Open Questions

1. **Can we use K3-fibred CY in CYTools?**
   - Need to find/construct appropriate polytope
   - Or adapt formulas to general fibred case

2. **How to connect to DESI observables?**
   - Cicoli doesn't quote (w₀, wₐ) directly
   - Need to solve cosmology with their potential
   - Compare to DESI contours

3. **What are the "charges" q_{ij}?**
   - Related to divisor intersection structure
   - Need to compute from CY geometry

---

## Papers Downloaded

- `resources/inflation_to_quintessence_2407.03405.pdf` + `.tex`
- `resources/quintessence_numerically_controlled_2112.10783.pdf`
- `resources/quintessence_quantum_gravity_tests_2410.21243.pdf`
- `resources/quintessence_phantoms_desi_2025_2506.21542.pdf`
- `resources/quintessence_string_moduli_2112.10779.pdf` (already had)

---

## References

1. Cicoli et al., "From Inflation to Quintessence", arXiv:2407.03405 (2024)
2. Kumar et al., "ScalPy", arXiv:1503.02407 (2015)
3. Cicoli et al., "Quintessence and the Swampland: parametric", arXiv:2112.10779 (2022)
4. Cicoli et al., "Quintessence and the Swampland: numerical", arXiv:2112.10783 (2022)
5. Toomey et al., "KMIX", arXiv:2511.23463 (2025)
6. DESI Collaboration, arXiv:2404.03002 (2024)
