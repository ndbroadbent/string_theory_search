# Research Questions: CYTools vs McAllister Data Compatibility

## Context

We are building a genetic algorithm to search the string theory landscape for Calabi-Yau compactifications that reproduce Standard Model physics. We use **CYTools** (a Python library) for polytope analysis and CY volume computations.

We want to validate our physics computations against published results from the McAllister group (arXiv:2107.09064), who achieved remarkably small cosmological constants (|W₀| ~ 10⁻⁹⁰).

## The Problem

We cannot reproduce McAllister's CY volume using CYTools.

**McAllister's published result for polytope 4-214-647:**
- CY Volume = 4711.83
- h11 = 4, h21 = 214

**Our CYTools result:**
- CY Volume ≈ 21 (using `_find_kahler_in_cone` method)
- CY Volume ≈ 374,000 (using first 4 values from their `kahler_param.dat`)

Neither matches.

## The Data Format Mismatch

McAllister's `kahler_param.dat` contains **214 values**, but the CY has h11 = 4.

CYTools' `compute_cy_volume(t)` expects a vector of dimension equal to the Kähler cone ambient dimension, which equals h11 = 4.

So what are the 214 values in McAllister's file?

## Our Current (Possibly Wrong) Understanding

1. **h11** = dimension of H^{1,1}(X) = number of independent Kähler moduli
2. **CYTools expects** 4 Kähler moduli t^i for `compute_cy_volume()`
3. **McAllister provides** 214 values labeled "kahler_param"

Possible interpretations of the 214 values:
- Heights for all toric divisors (one per lattice point)?
- Divisor volumes τ^a (which are quadratic in the Kähler moduli)?
- Coefficients in some extended basis?
- Something else entirely?

## Files in McAllister's Data Directory

```
4-214-647/
├── W_0.dat              # Superpotential: 2.30012e-90
├── g_s.dat              # String coupling: 0.00911134
├── cy_vol.dat           # CY volume: 4711.829675204889
├── kahler_param.dat     # 214 comma-separated floats
├── heights.dat          # 214 floats (some negative)
├── target_volumes.dat   # integers (mostly 1 or 6)
├── dual_points.dat      # 12 vertices (polytope)
├── K_vec.dat            # Flux K: -3,-5,8,6 (4 integers)
├── M_vec.dat            # Flux M: 10,11,-11,-5 (4 integers)
├── basis.dat            # 218 integers (indices?)
├── kklt_basis.dat       # 214 integers (indices?)
├── corrected_kahler_param.dat  # 214 floats (different values)
├── corrected_cy_vol.dat        # 4711.432499235554
└── ... (more files)
```

## Open Questions

### Q1: What parameterization does McAllister use?
The paper mentions "K\"ahler parameters $t_\star$" and "divisor volumes $\tau^i$". Are the 214 values in `kahler_param.dat` the divisor volumes τ, not the Kähler moduli t?

### Q2: How to convert between parameterizations?
If McAllister uses divisor volumes (214-dimensional), how do we convert to CYTools' Kähler moduli (4-dimensional)? The paper says "divisor volumes $\tau(t)$ are quadratic functions of the K\"ahler parameters $t^i$".

### Q3: What is the `basis.dat` file?
It contains 218 integers. Is this specifying which toric divisors to use? How does it relate to the 214 kahler_param values?

### Q4: Why are there "corrected" files?
There's both `kahler_param.dat` and `corrected_kahler_param.dat`, and both `cy_vol.dat` and `corrected_cy_vol.dat`. What correction was applied?

### Q5: Does CYTools use compatible conventions?
CYTools computes CY volume as V = (1/6) κ_{ijk} t^i t^j t^k where κ are triple intersection numbers. Does McAllister use the same formula and normalization?

### Q6: What is the relationship between polytope points and divisors?
The polytope has 12 vertices (in `dual_points.dat`), but there are 214+ parameters. Are these from interior lattice points? How does CYTools' `divisor_basis()` relate to McAllister's basis?

### Q7: How to reproduce McAllister's CY volume using CYTools?
Given their data files, what exact sequence of CYTools calls would reproduce cy_vol = 4711.83?

## What We Need

1. **Documentation** of McAllister's data format and parameterization
2. **Conversion formulas** between their parameterization and CYTools' expected inputs
3. **Working code** that takes McAllister's data and reproduces their CY volume using CYTools (or confirmation that this is impossible due to incompatible methods)

## References

- arXiv:2107.09064 - "Small cosmological constants in string theory" (Demirtas, Kim, McAllister, Moritz 2021)
- CYTools documentation: https://cy.tools/
- McAllister data: `resources/small_cc_2107.09064_source/anc/paper_data/4-214-647/`
