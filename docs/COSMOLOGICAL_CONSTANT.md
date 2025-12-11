# The Cosmological Constant

## Our Universe's Value

From **Planck 2018** satellite measurements ([arXiv:1807.06209](https://arxiv.org/abs/1807.06209)):

| Parameter | Value | Uncertainty (68% CL) |
|-----------|-------|---------------------|
| �_� (dark energy density parameter) | 0.6847 | � 0.0073 |
| H� (Hubble constant) | 67.4 km/s/Mpc | � 0.5 |
| �_m (matter density parameter) | 0.315 | � 0.007 |

### Derived Values

**Dark energy density (SI units):**
```
�_� = �_� � �_critical
    = �_� � 3H��/(8�G)
    = 5.85 � 10{�w kg/m�
    H 5.36 � 10{�p J/m�
    H 3.35 GeV/m�
```

**Cosmological constant in various units:**
```
� = 1.1056 � 10{u� m{�
  = 2.846 � 10{��� (Planck units, i.e., l_P{�)
  = 10{�u s{�
```

### Conversion to Planck Units

The Planck density is:
```
�_Planck = cu/(G�) = 5.178 � 10yv kg/m�
```

The dimensionless cosmological constant in Planck units:
```
� = 8� � (�_� / �_Planck)
  = 8� � (5.85 � 10{�w / 5.178 � 10yv)
  = 8� � 1.13 � 10{���
  = 2.846 � 10{���
```

**Best estimate for our search:**
```python
UNIVERSE_LAMBDA = 2.846e-122  # Planck units
TARGET_LOG_V0 = -121.546      # log��(�)
```

## The Cosmological Constant Problem

This is called "the worst theoretical prediction in the history of physics."

**The problem:** If we calculate the vacuum energy from quantum field theory (summing zero-point energies up to the Planck scale), we get:
```
�_vacuum^QFT ~ �_Planck ~ 10yv kg/m�
```

**The observation:**
```
�_�^observed ~ 10{�w kg/m�
```

**The discrepancy:**
```
�_QFT / �_observed ~ 10���
```

This 123 orders of magnitude mismatch is the cosmological constant problem.

## Why This Matters for String Theory

In the string theory landscape, the cosmological constant arises from:

```
V� = -3 e7 |W|�
```

Where:
- **K** = K�hler potential (depends on moduli)
- **W** = superpotential (flux + non-perturbative contributions)

McAllister et al. ([arXiv:2107.09064](https://arxiv.org/abs/2107.09064)) achieved V� ~ 10{�p� for a specific Calabi-Yau compactification. Our goal is to find configurations that produce V� ~ 10{���.

### The Gap

| Source | log��(|V�|) |
|--------|-------------|
| Our Universe | -121.5 |
| McAllister's best | -203 |
| Typical string vacua | -200 to -600 |

The challenge: McAllister found vacua that are **too small** by ~80 orders of magnitude. We need to find the "Goldilocks" configurations.

## References

1. **Planck 2018 Results VI** - Cosmological Parameters
   - [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
   - [A&A 641, A6 (2020)](https://www.aanda.org/articles/aa/full_html/2020/09/aa33910-18/aa33910-18.html)

2. **Planck 2024 PR4** - Final Data Release
   - [A&A 2024](https://www.aanda.org/articles/aa/full_html/2024/02/aa48015-23/aa48015-23.html)
   - Constraints ~10-20% tighter than 2018

3. **PDG 2024** - Cosmological Parameters Review
   - [pdg.lbl.gov](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-cosmological-parameters.pdf)

4. **McAllister et al.** - Small Cosmological Constants in String Theory
   - [arXiv:2107.09064](https://arxiv.org/abs/2107.09064)

## Physical Constants Used

From CODATA 2018:
```
G  = 6.67430(15) � 10{�� m� kg{� s{�
c  = 299792458 m/s (exact)
  = 1.054571817 � 10{�t J s
l_P = (G/c�) = 1.616255 � 10{�u m (Planck length)
t_P = (G/cu) = 5.391247 � 10{tt s (Planck time)
m_P = (c/G)  = 2.176434 � 10{x kg (Planck mass)
```

Conversion:
```
1 Mpc = 3.08567758149 � 10�� m
H� = 67.4 km/s/Mpc = 2.1836 � 10{�x s{�
```
