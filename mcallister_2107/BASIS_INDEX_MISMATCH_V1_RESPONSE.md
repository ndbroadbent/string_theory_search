Two things jump out that can produce **exactly** the kind of “works for h11=4, fails for h11=214” situation you’re seeing:

1. **You’re probably mixing “uncorrected” vs “corrected” ancillary files.**
   For `paper_data/4-214-647/` arXiv ships *both* `kahler_param.dat`/`heights.dat` **and** `corrected_kahler_param.dat`/`corrected_heights.dat`, plus `cy_vol.dat` **and** `corrected_cy_vol.dat`. ([arXiv][1])
   If your computed 17901.22 matches the uncorrected set, but your “target” 4711.8296 came from the corrected set, you’ll get a big mismatch even with perfect indexing.

2. **There are two divisor bases in the anc: `basis.dat` and `kklt_basis.dat`.** ([arXiv][1])
   Your `kahler_param.dat` is only meaningful in the divisor basis it was generated in. If the moduli were solved/stored in the KKLT-adapted basis, using `basis.dat` will give the wrong volume even if CYTools accepts it.

Below is the fastest way to isolate which one it is.

---

## Step 1: Don’t build κ, let CYTools compute the volume

This removes any possible tensor bookkeeping issues:

```python
import numpy as np
from cytools import Polytope

points  = np.loadtxt("points.dat", delimiter=",", dtype=int)

# Try both (uncorrected, corrected)
heights = np.loadtxt("heights.dat", delimiter=",")
t       = np.loadtxt("kahler_param.dat", delimiter=",")

poly = Polytope(points)
tri  = poly.triangulate(heights=heights)
cy   = tri.get_cy()

for basis_file in ["basis.dat", "kklt_basis.dat"]:
    basis = np.loadtxt(basis_file, delimiter=",", dtype=int)
    cy.set_divisor_basis(basis)  # do NOT subtract 1
    V = cy.compute_cy_volume(t)
    print(basis_file, V)
```

Then repeat with corrected files:

```python
heights = np.loadtxt("corrected_heights.dat", delimiter=",")
t       = np.loadtxt("corrected_kahler_param.dat", delimiter=",")

tri = poly.triangulate(heights=heights)
cy  = tri.get_cy()

for basis_file in ["basis.dat", "kklt_basis.dat"]:
    basis = np.loadtxt(basis_file, delimiter=",", dtype=int)
    cy.set_divisor_basis(basis)
    V = cy.compute_cy_volume(t)
    print("corrected", basis_file, V)
```

Given the anc layout, it is very plausible that:

* `heights.dat` + `kahler_param.dat` + one of the bases reproduces ~17901, and
* `corrected: ...` reproduces ~4711.83. ([arXiv][1])

If that happens, this is not a CYTools basis-index bug. It’s “wrong file set” or “wrong basis for that file set”.

---

## Step 2: Your H1 “subtract 1” is almost certainly wrong

CYTools point indexing puts the **origin at index 0** and then boundary points after that. The docs explicitly describe a canonical ordering of polytope points with interior points first. ([CYTools][2])

So a basis file that starts at 1 is consistent with “skip the origin”. Subtracting 1 would inject index 0 (the origin) into your divisor basis, which you do not want.

---

## Step 3: If you still suspect “McAllister indices ≠ CYTools indices”, map by coordinates, not integers

CYTools may reorder points to its canonical ordering. ([CYTools][2])
Also, for reflexive polytopes the triangulation machinery typically excludes points interior to facets by default. ([CYTools][3])
So the only robust way to translate an external numbering scheme is: **basis index → lattice point → CYTools point index**.

Pattern:

```python
# points_file[i] = the i-th point in McAllister's points.dat order (including origin at i=0)
points_file = np.loadtxt("points.dat", delimiter=",", dtype=int)

poly = Polytope(points_file)

# Build coordinate -> CYTools index map
coord_to_cyidx = {tuple(p): int(poly.points_to_indices(p)) for p in points_file}

# Convert a basis expressed as indices into points.dat order
basis_file_indices = np.loadtxt("basis.dat", delimiter=",", dtype=int)

# If basis.dat is 1-based indices into points.dat including origin at line 0,
# then convert with the same convention (adjust here if needed):
basis_cy = [coord_to_cyidx[tuple(points_file[idx])] for idx in basis_file_indices]

# Now use basis_cy
tri = poly.triangulate(heights=np.loadtxt("heights.dat", delimiter=","))
cy  = tri.get_cy()
cy.set_divisor_basis(basis_cy)
```

If `basis.dat` already matches CYTools indices, this conversion will be the identity. If not, this fixes it.

(Use the same trick to reorder `t` if `t` is keyed to divisor indices rather than “basis order”.)

---

## Step 4: Why the “missing 8,9,10,17” is not suspicious by itself

In a 4D toric ambient with 218 rays, the Picard rank is typically `218 - 4 = 214`, so you expect to omit 4 prime divisors to form a basis. Your `basis.dat` is literally “all rays except 4”, which is normal.

---

## What I would bet is happening

Given the arXiv ancillary explicitly contains `corrected_*` variants for **heights, kahler params, target volumes, and cy volume** in this model ([arXiv][1]), I would first assume:

* **17901.22 is the volume corresponding to the uncorrected file set**, and
* **4711.8296 is from the corrected file set** (or from using `kklt_basis.dat` instead of `basis.dat`).

Run the 2×2 test matrix (uncorrected/corrected) × (basis/kklt_basis) above, and you’ll know immediately which.

If you paste the four volumes you get from that 2×2, I can tell you exactly which mapping is consistent and what the intended combination is.

[1]: https://arxiv.org/abs/2107.09064 "[2107.09064] Small Cosmological Constants in String Theory"
[2]: https://cy.tools/docs/documentation/polytope/ "Polytope Class | CYTools"
[3]: https://cy.tools/docs/documentation/triangulation/ "Triangulation Class | CYTools"
