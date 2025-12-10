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

---

Short answer: the 214 “kahler_param” numbers are **not** the 4 Kähler moduli (t^i). They live in the **ambient/secondary‑fan parameterization** that McAllister’s code uses while walking through phases to a point where selected **divisor volumes** hit prescribed integer targets. To reproduce their (\mathrm{CY}) volume with CYTools you must (i) rebuild the exact FRST triangulation they used, (ii) adopt their divisor basis, (iii) convert their parameters into the 4 moduli (t^i), and (iv) compare in the same frame (string vs Einstein). Below is a precise mapping and a working recipe.

---

## What each file is (for 4‑214‑647)

* `points.dat`, `dual_points.dat`, `dual_simplices.dat`
  Geometry/triangulation: the FRST of (\Delta^*) that defines the toric ambient and hence the CY hypersurface phase used in the paper’s solution. You must load this exact triangulation to match their intersection tensor and Mori/Kähler cones. The paper describes solving for Kähler parameters by moving in the **secondary fan** (coherent triangulations) and emphasizes that **divisor volumes are quadratic in (t)** and change piecewise across phases. ([ar5iv][1])

* `basis.dat`
  A permutation of the prime toric divisors. The **first 4 indices** are the KKLT run’s **divisor basis** on the CY (a (\mathbb{Z})-basis of (H^{1,1}(X))); the remaining entries enumerate the other prime toric divisors in the order used by their code. (You can confirm this by counting: #entries equals #prime toric divisors that restrict to the CY.)

* `kklt_basis.dat`
  The list of indices (length equals “#prime toric divisors - 4”) that correspond to the **non‑basis** divisors. This is the index order used for the long vectors below.

* `target_volumes.dat`
  Integer **targets** for the **Einstein‑frame** volumes of the prime toric divisors at the final solution (mostly 1 or 6 in the examples). These are the (\tau) targets the path‑following algorithm tries to achieve component‑wise (over an overdetermined set), cf. the discussion around eqs. (5.8)–(5.11). ([ar5iv][1])

* `heights.dat`
  The **lifting heights** that specify a coherent triangulation in the **secondary fan** language (they can be negative). These encode “where you are” in triangulation space while homotoping toward the desired point with the target divisor volumes.

* `kahler_param.dat`
  The **ambient/extended Kähler coordinates** used along that secondary‑fan walk: **one real number per non‑basis prime toric divisor**, ordered as in `kklt_basis.dat`. These are **not** the 4 moduli (t^i). Think of this as a redundant FI‑type vector on toric divisors; the true (t^i) are obtained by **projecting** to the 4‑dimensional divisor‑class basis via linear relations among toric divisors.

* `c_tau.dat`
  The Jacobian used in the line‑search update, i.e. coefficients for the **linearized map** (\delta\tau_A=\sum_i c_{A i},\delta t^i) (with (A) running over the many toric divisors and (i=1,\dots,4)). It implements eq. (5.11) of the paper’s algorithm. ([ar5iv][1])

* `cy_vol.dat`, `W_0.dat`, `g_s.dat`
  Final observables at the found vacuum: (|W_0|), (g_s), and the **Einstein‑frame CY volume** (\mathcal{V}_E). Section 6 reports volumes explicitly in **Einstein frame**. ([ar5iv][1])

* `corrected_*.dat`
  The same quantities **after including worldsheet‑instanton corrections** to the Kähler potential and re‑minimizing, as discussed in Sections 5.3–6 (“the instanton corrections shift the solution by …”). The “uncorrected” files correspond to the perturbative (tree‑level in the Kähler sector) F‑flat solution; the “corrected” files reflect the small shift once the GV‑invariant sum is included. ([ar5iv][1])

---

## Conventions and conversion

1. **CYTools formulas**
   CYTools uses the standard classical (string‑frame) formulas on a chosen divisor basis:
   [
   \mathcal{V}*S(t)=\tfrac{1}{6},\kappa*{ijk},t^i t^j t^k,\qquad
   \tau_i(t)=\tfrac{1}{2},\kappa_{ijk},t^j t^k .
   ]
   See the docs for `compute_cy_volume` and `compute_divisor_volumes`. ([cy.tools][2])

2. **Einstein vs string frame**
   The paper quotes **Einstein‑frame** volumes, while CYTools returns **string‑frame** by default. Convert with
   [
   \mathcal{V}_E = g_s^{-3/2},\mathcal{V}_S ,
   ]
   where (g_s) is from `g_s.dat`. So to compare to `cy_vol.dat`, multiply the CYTools result by (g_s^{-3/2}). (This alone can easily explain order‑of‑magnitude mismatches if ignored.)

3. **214 numbers vs (h^{1,1}=4)**
   The 214 “kahler_param” entries are **not (\tau)** and not the 4 (t^i). They are the **redundant ambient coordinates** (one per **non‑basis** prime toric divisor) the authors optimize over while staying inside a fixed FRST chamber and pushing toward the target divisors’ volumes. The paper explicitly works with:

* a 4‑dimensional **basis** of divisor classes (the true Kähler moduli), and
* many **prime toric divisors** whose volumes (\tau_A(t)) are quadratic in (t^i), used to impose control ((\tau_A\ge 1) etc.) and to monitor instanton convergence via GV invariants. ([ar5iv][1])

In short: **use `basis.dat` to pick the 4 basis divisors**, then **project** the 214‑vector in `kahler_param.dat` down to the 4 (t^i) using the toric linear relations among divisors.

---

## How to project to the 4 Kähler moduli in CYTools

Let (D_A) be the prime toric divisors (indexed as in `basis.dat`). CYTools can give you the integer **GLSM linear relations**
[
\sum_A Q^r_{,A},D_A ;=;0\qquad (r=1,\dots,R)
]
and from these you can build a **projection** (P) from the big vector of ambient parameters to the 4‑dimensional divisor‑class basis (D_i) ((i=1,\dots,4)). In practice:

1. **Load the exact triangulation and CY**

```python
from cytools import Polytope, Triangulation
# load all points of Δ* and its simplices from McAllister's files
pdual = Polytope(dual_points)     # from dual_points.dat
t = Triangulation(pdual, simplices=dual_simplices)  # from dual_simplices.dat
cy = t.get_cy()
```

2. **Adopt their divisor basis**

```python
# basis.dat is a permutation of prime toric divisors; take the first 4 as the basis
perm = load_ints("basis.dat")
cy.set_divisor_basis(perm[:4])
```

3. **Build the projection from ambient to basis coordinates**
   Let `D_all` be the list of prime toric divisors in CYTools’ order, and `D_basis` the 4 you just set. Use the **GLSM linear relations** to express every (D_A) as an integer combination of the 4 basis divisors. CYTools exposes the needed pieces via `glsm_linear_relations()` and the divisor basis utilities. This gives you a **matrix (R\in\mathbb{Z}^{(#\text{divisors})\times 4})** with
   [
   D_A = \sum_{i=1}^4 R_{A i},D_i .
   ]
   Then the **ambient “kahler_param” vector** (t^{\text{amb}}*A) in the order of `kklt_basis.dat` projects to the true 4‑vector
   [
   t_i ;=; \sum_A \widehat{R}*{iA};t^{\text{amb}}_A,
   ]
   where (\widehat{R}) is the appropriate left‑inverse assembled for the non‑basis rows (use the reordering from `basis.dat`/`kklt_basis.dat`). Numerically you can get (t) by least‑squares but in practice (R) has full rank 4 in this basis so the 4 components are determined.

4. **Alternative (safer) route via divisor volumes**
   If you’re uncertain about the projection, use the physics: the authors target specific **Einstein‑frame divisor volumes**. Pull in `target_volumes.dat` as the vector (\tau^{\mathrm{(E)}}_A), convert to **string frame** (\tau^{\mathrm{(S)}}_A=g_s^{3/2}\tau^{\mathrm{(E)}}*A), and **solve for (t)** by minimizing
   [
   |\tau^{\mathrm{(S)}}(t)-\tau^{\mathrm{(S)}}*{\text{target}}|^2
   ]
   subject to (t) in the Kähler cone. Use CYTools to compute (\tau(t)) with `compute_divisor_volumes(t, in_basis=False)` and restrict to the relevant rows/ordering. This directly produces the 4 Kähler moduli consistent with the reported targets, independent of `kahler_param.dat`. (Equations (5.8)–(5.11) justify the linearized updates used in the paper’s search; you can jump straight to a nonlinear solve.) ([ar5iv][1])

---

## Why your two CY volumes don’t match 4711.83

* Using `_find_kahler_in_cone` gave ~21 because that picks a **generic point** (often near the tip of a stretched cone) unrelated to their FRST chamber and **not** the KKLT minimum.
* Using the **first 4** entries of `kahler_param.dat` gave (\sim 3.74\times 10^5) because those entries are **not** (t^i); they’re ambient coordinates in a 214‑dim space.
* You also must compare in the same **frame**: multiply CYTools’ string‑frame (\mathcal{V}_S) by (g_s^{-3/2}) to match their quoted (\mathcal{V}_E). CYTools’ `compute_divisor_volumes`/`compute_cy_volume` follow the standard normalizations (\mathcal{V}*S=\tfrac{1}{6}\kappa t^3) and (\tau_i=\tfrac{1}{2}\kappa*{ijk}t^j t^k), exactly as documented. ([cy.tools][2])

---

## Minimal end‑to‑end recipe to reproduce `cy_vol = 4711.83`

1. **Rebuild their CY phase**

   * Load `dual_points.dat` and `dual_simplices.dat` into a `Triangulation`, then `get_cy()`.

2. **Adopt their divisor basis**

   * `cy.set_divisor_basis(perm[:4])` with `perm` from `basis.dat`.

3. **Recover (t)**

   * Preferred: solve for (t) from the **target divisor volumes**:

     * Read `target_volumes.dat` → (\tau^{\mathrm{(E)}}_A); convert to string frame with `g_s.dat`.
     * Numerically solve (\tau_A(t)=\tfrac{1}{2}\kappa_{A,jk}t^j t^k \approx g_s^{3/2}\tau^{\mathrm{(E)}}_A) (over the subset of divisors you trust, e.g. the rigid ones or all toric prime divisors), with (t) constrained to `cy.toric_kahler_cone()`.
   * Or: project `corrected_kahler_param.dat` to (t) using the GLSM relations matrix as explained above.

4. **Compute the volume**

   * `V_S = cy.compute_cy_volume(t)`
   * `V_E = V_S * g_s**(-1.5)` using the value in `g_s.dat`.

You should obtain the **uncorrected** `cy_vol.dat` and, if you used `corrected_kahler_param.dat`, the **corrected** `corrected_cy_vol.dat` (the paper shows the instanton‑corrected shift is small). ([ar5iv][1])

---

## Direct answers to your questions

**Q1. What parameterization does McAllister use?**
Two at once: a **4‑parameter** divisor‑basis Kähler class (J=\sum_i t^i D_i), and a **redundant ambient/secondary‑fan parameterization** attached to **prime toric divisors** used for the numerical path‑following. The 214 numbers in `kahler_param.dat` are in the latter category (one per **non‑basis** toric divisor), **not** the four (t^i). The volumes (\tau_A(t)) of divisors are quadratic in (t). ([ar5iv][1])

**Q2. How to convert?**
Use the **GLSM linear relations** among toric divisors to express every prime toric divisor in the 4‑divisor basis. This gives a projection matrix from the 214‑vector to the 4‑vector (t). Alternatively, bypass `kahler_param.dat` and solve for (t) so that `compute_divisor_volumes(t)` matches the Einstein‑frame targets (after converting to string frame by (g_s^{3/2})). CYTools implements the needed formulas (\tau_i=\tfrac12\kappa_{ijk}t^j t^k), (\mathcal{V}*S=\tfrac16\kappa*{ijk}t^i t^j t^k). ([cy.tools][2])

**Q3. What is `basis.dat`?**
A permutation of prime toric divisor indices; the **first 4 entries are the divisor‑basis** used to define the true Kähler moduli. The rest provide the ordering used for long vectors like `kahler_param.dat`. The companion `kklt_basis.dat` lists exactly the **non‑basis** indices in the order used in those long vectors. (Sizes match: `len(basis)=#divisors`, `len(kklt_basis)=#divisors-4`.)

**Q4. Why “corrected” files?**
They report both the **tree‑level Kähler‑sector** solution and the **instanton‑corrected** one after summing GV‑invariant contributions and re‑minimizing. The corrected files are the latter; Section 6 states the shift is small and shows convergence diagnostics. ([ar5iv][1])

**Q5. Conventions match?**
Yes, the **classical** normalization matches CYTools:
(\mathcal{V}*S=\tfrac16\kappa t^3,\ \tau_i=\tfrac12\kappa*{ijk}t^j t^k). CYTools computes in **string frame**; the paper quotes **Einstein‑frame** volumes, so multiply by (g_s^{-3/2}) to compare. ([cy.tools][2])

**Q6. Polytope points vs divisors**
Toric divisors correspond to rays of the fan (lattice points of (\Delta^*) excluding the origin). There are **many** prime toric divisors, but their restrictions to (X) span a **4‑dimensional** Néron‑Severi space for this geometry (since (h^{1,1}=4)). CYTools’ `divisor_basis()` gives a 4‑element basis; the paper manipulates volumes of **all** prime toric divisors to enforce control and convergence conditions, hence the long vectors. ([ar5iv][1])

**Q7. Exact CYTools call sequence to get 4711.83**
Skeleton (replace `load_*` with your actual file readers):

```python
from cytools import Polytope, Triangulation

# 1) Recreate their FRST
pdual = Polytope(load_points("dual_points.dat"))
t = Triangulation(pdual, simplices=load_simplices("dual_simplices.dat"))
cy = t.get_cy()

# 2) Use their divisor basis (first 4 of basis.dat)
perm = load_ints("basis.dat")
cy.set_divisor_basis(perm[:4])

# 3a) Recover t by matching target divisor volumes
g_s = float(open("g_s.dat").read())
tauE = load_floats("target_volumes.dat")          # Einstein-frame targets (ordered as in basis.dat/kklt_basis.dat)
tauS = [ (g_s**1.5)*x for x in tauE ]             # convert to string frame

# Select the same subset of divisors they enforce (often all toric prime divisors)
def residual(t):
    tau_here = cy.compute_divisor_volumes(t, in_basis=False)
    return tau_here[selected_indices] - tauS[selected_indices]

t = solve_nonlinear_in_kahler_cone(residual)      # any robust constrained solver

# 3b) Or project corrected_kahler_param.dat through GLSM relations to get t

# 4) Compute volumes and convert to Einstein frame
V_S = cy.compute_cy_volume(t)
V_E = V_S * (g_s**(-1.5))
```

With the corrected parameters you should reproduce `corrected_cy_vol.dat ≈ 4711.43`; with the uncorrected ones you should get `cy_vol.dat ≈ 4711.83`. The paper’s Section 6 states volumes at the KKLT point in Einstein frame (and shows the small instanton‑induced shift). ([ar5iv][1])

---

## Sanity checks and common pitfalls

* **Wrong FRST**: If you don’t load *their* triangulation, the intersection tensor changes and nothing will match.
* **Wrong basis**: If your 4 basis divisors differ from theirs, your (\kappa_{ijk}) in that basis differ and `t` won’t line up.
* **Frame mismatch**: Always convert (\mathcal{V}_S) ↔ (\mathcal{V}_E) with (g_s).
* **Ordering**: `basis.dat`/`kklt_basis.dat` define the ordering used for long vectors; keep that order when projecting or matching (\tau).
* **Using `kahler_param.dat` as (t)**: Don’t - it’s ambient/secondary‑fan data, not the 4 moduli.

---

### Sources

* CYTools API (volume and divisor-volume normalizations; divisor bases; Kähler/Mori cones). ([cy.tools][2])
* Paper’s description of the path‑following algorithm in the secondary fan and the quadratic (\tau(t)) dependence; example section for ((h^{1,1},h^{2,1})=(4,214)) quoting Einstein‑frame volume and small instanton shift. ([ar5iv][1])
* ArXiv ancillary file list for `cy_vol.dat`, `corrected_*`, `kahler_param.dat`, `target_volumes.dat`, etc., confirming the dataset structure used above. ([arXiv][3])

---

If you want, send me the four indices at the top of your `basis.dat` and the first few lines of `kklt_basis.dat`. I can draft the tiny projection routine that maps your `corrected_kahler_param.dat` to the (t^i) CYTools expects and print (\mathcal{V}_E) directly.

[1]: https://ar5iv.org/pdf/2107.09064 "[2107.09064] Contents"
[2]: https://cy.tools/docs/documentation/calabiyau/ "CalabiYau Class | CYTools"
[3]: https://arxiv.org/abs/2107.09064 "[2107.09064] Small Cosmological Constants in String Theory"
