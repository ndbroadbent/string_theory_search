# McAllister Group: Small Cosmological Constant Configurations

## Source Papers
- **arXiv:2107.09064** - "Small cosmological constants in string theory" (Demirtas, Kim, McAllister, Moritz 2021)
- **arXiv:1912.10047** - "Vacua with Small Flux Superpotential" (Demirtas et al. 2019)
- **arXiv:2009.03312** - "Conifold Vacua" (Demirtas et al. 2020)

## Key Achievement
They found configurations achieving:
- |W₀| ~ 10⁻⁹⁰ (for polytope 4-214-647)
- This translates to |Λ| ~ 10⁻¹²³ in Planck units
- The observed cosmological constant is ~2.888 × 10⁻¹²²

## Summary of Polytope Configurations

| Polytope ID | h¹¹ | h²¹ | |Δh| | W₀ | g_s | CY Volume | K flux | M flux |
|-------------|-----|-----|-----|-----|-----|-----------|--------|--------|
| 647 | 4 | 214 | 210 | 2.30e-90 | 0.0091 | 4712 | (-3,-5,8,6) | (10,11,-11,-5) |
| 4627 | 5 | 113 | 108 | 6.46e-62 | 0.0111 | 945 | (8,-15,11,-2,13) | (0,2,4,11,-8) |
| 3213 | 5 | 81 | 76 | 2.04e-23 | 0.0504 | 198 | (-5,5,-4,-1,5) | (3,-5,2,-2,-5) |
| 13590 | 7 | 51 | 44 | 4.08e-21 | 0.0403 | 142 | (-4,-4,-3,2,-3,3,3) | (4,4,0,-3,2,0,-2) |

## Key Observation: h²¹ Correlation
**Larger h²¹ → Smaller W₀ → Smaller Λ**

This is the "DKMM mechanism": large complex structure moduli space (high h²¹) provides more freedom for flux tuning, enabling exponentially small superpotential.

## Detailed Data Files (per polytope)

Each polytope directory contains:
- `W_0.dat` - Superpotential magnitude
- `g_s.dat` - String coupling
- `cy_vol.dat` - Calabi-Yau volume
- `K_vec.dat` / `M_vec.dat` - Flux vectors
- `kahler_param.dat` - Kähler parameters (h²¹ values)
- `dual_points.dat` - Polytope vertices
- `dual_simplices.dat` - Triangulation
- `potent_rays_gv.dat` - Gopakumar-Vafa invariants

## Polytope 647 (Best Result) - Detailed

### Vertices (dual_points.dat)
```
0,0,0,0
-1,2,-1,-1
1,-1,0,0
-1,-1,1,1
-1,-1,1,2
-1,-1,2,1
-1,-1,2,3
-1,-1,3,2
-1,-1,2,2
-1,0,1,1
-1,1,0,0
0,-1,1,1
```

### Physics Parameters
- W₀ = 2.30012 × 10⁻⁹⁰
- g_s = 0.00911134
- c_τ = 3.34109
- CY Volume = 4711.83

## Implications for Our Search

### What These Polytopes Are NOT
- These are NOT three-generation polytopes (|h¹¹ - h²¹| = 3)
- The best result (h¹¹=4, h²¹=214) has |Δh| = 210 generations
- Our search focuses on |Δh| = 3 for Standard Model physics

### What We Learn
1. **Large h²¹ helps small CC** - Confirmed empirically in our correlation analysis
2. **Small g_s is necessary** - All examples have g_s < 0.05
3. **Large CY volumes help** - Larger volumes → better perturbative control
4. **Specific flux patterns matter** - Not random; carefully chosen

### The Fundamental Tension
To get 3 generations, we need |h¹¹ - h²¹| = 3
But to get small CC, we need large |h²¹|

Possible solutions:
1. Find polytopes with h¹¹ ≈ h²¹ + 3 where h²¹ is still large (e.g., h¹¹=103, h²¹=100)
2. Use different compactification mechanism that doesn't require large h²¹
3. Accept that SM-compatible vacua may not have ultra-small CC naturally

## Correlation Analysis (from our search)

Our empirical correlations between heuristics and CC error:
| Feature | Pearson r |
|---------|-----------|
| h²¹ | -0.79 |
| h¹¹ | -0.77 |
| shannon_entropy | -0.67 |
| spikiness | -0.61 |
| coord_kurtosis | -0.58 |
| prime_count | -0.53 |
| loner_score | -0.51 |

Negative correlation means: higher value → smaller CC error (better)

This empirically confirms the theoretical finding: **large h²¹ is the strongest predictor of small CC**.
