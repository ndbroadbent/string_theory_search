Tools that can actually help with **periods / Picard-Fuchs / GKZ** fall into two fairly different camps:

## 1) Toric CY hypersurfaces (HKTY / GKZ series at LCS)

This is the closest match to what McAllister is doing in 2107.09064 (large complex structure expansions, GKZ system, mirror map, prepotential).

### `cygv` (Rust core + Python package)

* Implements the **HKTY procedure** (Hosono-Klemm-Theisen-Yau) for toric hypersurfaces / CICYs in toric varieties, “based on *Computational Mirror Symmetry*”, written in **Rust** with **Python bindings**.
* It’s distributed on PyPI (`pip install cygv`) and explicitly targets the same mirror-symmetry pipeline you need (GKZ series, mirror map, GV/GW invariants, etc.).

Why it’s relevant for you:

* Even if your end goal is **periods Π(z)** and **W₀ = (F − τH)·Π**, anything implementing HKTY is already doing a big chunk of the “period technology” (fundamental period, logarithmic solutions, mirror map, prepotential reconstruction).
* It’s also the best “Rust-native” option I found.

Practical tip: for DKMM-style tiny values, you want a pipeline that keeps as much as possible in **exact integer/rational arithmetic** until the final numerical evaluation, then uses high precision. HKTY-style series code often enables this.

## 2) Numerical periods for projective hypersurfaces (Picard-Lefschetz based)

These are not toric-first, but they are excellent for:

* sanity checks on smaller models,
* cross-validating a Picard-Fuchs operator,
* computing period matrices for explicit projective hypersurface equations.

### `lefschetz-family` (Python, SageMath-based)

* Sage package providing “efficiently computing periods of complex projective hypersurfaces” with **certified rigorous precision bounds**, based on Picard-Lefschetz methods.
* Available on GitHub and PyPI.

Caveat:

* It targets **projective hypersurfaces** (and some related constructions), not general toric hypersurfaces. If you can represent the relevant mirror family as an explicit projective hypersurface in a way the package supports, it can be extremely useful; otherwise it’s more of a reference/checking tool.

### `PeriodSuite` (Sage + Magma)

* Software to compute periods of hypersurfaces, implementing algorithms from “Computing periods of hypersurfaces”.
* Requires **SageMath and Magma** (so not a clean open pipeline).

## 3) “Build your own toric GKZ” toolchain helpers (GKZ D-modules, canonical series)

If you want to explicitly construct GKZ systems / Picard-Fuchs operators and then generate series solutions:

### Macaulay2 `Dmodules` / `HolonomicSystems` (GKZ system construction)

* Macaulay2 has `gkz(A,b)` for building the **A-hypergeometric (GKZ) system** as a Weyl algebra ideal.
* HolonomicSystems docs include series-solution oriented workflows (canonical series style).

This is useful if:

* you want to compute the GKZ/PF operator from your toric data independently,
* or you want an algebraic check that your charge vectors / secondary fan phase match what you think.

### `FeynGKZ` (Mathematica package, but GKZ-series machinery)

* Mathematica package that derives/solves GKZ systems using triangulation and Gröbner deformation approaches, with dependencies including Macaulay2/TOPCOM/polymake.
* Not string-geometry specific, but it’s a working “GKZ-series solution generator” you can crib ideas from.

### OpenXM / Risa-Asir `mt_gkz` (GKZ via Macaulay matrix method)

* There are documented toolchains for GKZ hypergeometric systems using **mt_gkz** and related routines.
* More niche, but it’s another established “GKZ engine” you can lean on.

## What’s probably missing from your current reproduction inputs

Even with perfect period tooling, to compute **W₀ from first principles** you need to be able to line up three things consistently:

1. **A point in complex-structure moduli space** (or a 1-parameter slice like DKMM’s “p-direction” plus a specific large parameter value).
2. **A symplectic basis convention for H³** so that your flux vector lives in the same coordinates as your period vector Π.
3. **Enough precision / exact arithmetic** so that you’re not relying on cancellations at floating precision when W₀ is ~1e−90.

From your file list, the potential red flag is: if your flux data is stored in some reduced basis (like those short `K_vec.dat`, `M_vec.dat`), you’ll need the exact basis map to the full b₃-dimensional symplectic basis used for Π, otherwise you can compute a perfectly correct Π(z) and still get nonsense W₀.

## What I’d do next in your shoes (fastest path to a passing unit test)

1. **Try `cygv` first** (it’s the only thing I found that is both toric-first and (Rust+Python) and explicitly HKTY-based).
2. Use it to compute:

   * the fundamental period + log periods near LCS,
   * mirror map,
   * and the pieces needed to assemble Π(z) in your chosen basis.
3. Only if you get stuck on toric specifics, use Macaulay2’s GKZ system builder to validate your A-matrix / charge vectors and whether you’re in the right secondary-fan phase.
