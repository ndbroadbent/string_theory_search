The key thing is: **`basis.dat` is not a “4 element basis for the dual CY”. It is a 214-element divisor basis for the *primal* (mirror) CY.** Once you see that, the weird indices suddenly make sense.

Let me unpack what is going on with your two files.

---

## 1. What `basis.dat` actually is

From your `head`:

```text
basis.dat:
1,2,3,4,5,6,7,11,12,13,...,218   # 214 numbers total
```

* The primal polytope (`points.dat`) has **218 prime toric divisors** and **h11 = 214**.
* A divisor basis for (H^{1,1}) on this manifold must have **214** elements.
* `basis.dat` contains **214 indices** between 1 and 218.
* The indices that do *not* appear (you already noticed them) are:

  * 8, 9, 10, 17  (4 divisors)

So:

* **The basis is the 214 divisors listed in `basis.dat`.**
* The 4 *missing* divisors `{8,9,10,17}` are the **dependent** divisors that can be expressed as integer linear combinations of the basis. They are *not* a basis themselves. There is no way 4 divisors span a 214 dimensional divisor class group.

In other words, your earlier line

> Basis divisors: [8, 9, 10, 17] (indices NOT in basis.dat)

has the terminology flipped. Those are the 4 non-basis divisors. The basis is everything *except* those four.

---

## 2. What `kklt_basis.dat` is

From your `head`:

```text
kklt_basis.dat:
3,4,5,6,7,8,9,10,11,12,...,218   # 214 numbers total
        ...
        43,44,45,47,48,49,...    # note 46 missing here
```

Again:

* It also has **214** entries between 1 and 218.
* So it is **another choice of divisor basis** for the same primal CY.
* The 4 numbers that do *not* appear in `kklt_basis.dat` are another set of 4 dependent divisors (different from `{8,9,10,17}`).

Meaning:

* `basis.dat` and `kklt_basis.dat` are two different **bases of (H^{1,1})** for the **primal** manifold with (h^{1,1}=214).
* They give you two 214 x 218 selection matrices:

  * (S_{\text{basis}}): rows pick the indices in `basis.dat`
  * (S_{\text{kklt}}): rows pick the indices in `kklt_basis.dat`
* You can move between them via an integer 214 x 214 change-of-basis matrix.

`kahler_param.dat` and `corrected_kahler_param.dat` are 214-vectors that live in **one of these bases** (in practice: in the KKLT basis, hence the filename).

---

## 3. Why this clashes with your dual CY in CYTools

On the **dual polytope** (`dual_points.dat`) side, you already saw:

* `cy_dual.h11() = 4`
* Only **8 prime toric divisors** with indices 1..8 in CYTools.

So:

* The indices in `basis.dat` and `kklt_basis.dat` run from 1 to 218 because they are indexing the **primal toric divisors**, not the dual ones.
* That is why trying to do something like `cy_dual.set_divisor_basis([1,2,3,4,...])` immediately hits a wall: those divisors literally do not exist on the dual.

Conceptually:

* `points.dat` + `basis.dat` + `kklt_basis.dat` + `kahler_param.dat` describe the **mirror CY with (h^{1,1}=214, h^{2,1}=4)**.
* `dual_points.dat` + `dual_simplices.dat` describe the **target CY with (h^{1,1}=4, h^{2,1}=214)** where you are doing IIB flux + KKLT.

So you are mixing data from **two different geometries**:

* primal: used for GV invariants, instanton control, etc.
* dual: used for flux compactification (your CYTools `get_cy()` with 8 divisors and 4 Kähler moduli).

The CYTools object you currently have for 4-214-647 is the **dual one**, so `basis.dat` and `kklt_basis.dat` are **not directly usable** as divisor indices there.

---

## 4. What this means for your reproduction attempt

Given all that:

1. **You cannot use `basis.dat` as “the basis for the 4-moduli CY” in CYTools.**
   It is a 214-element basis on the mirror.

2. **`kahler_param.dat` and `corrected_kahler_param.dat` are 214 Kähler coordinates on the mirror, not the 4 Kähler moduli on the dual.**
   Mapping them to the 4 dual Kähler moduli is a full mirror-map problem (Computational Mirror Symmetry style), which is *not* encoded in these anc files and not implemented by CYTools out of the box.

3. For the **dual, h11=4 CY** that you actually use in your GA:

   * The only clean geometric inputs from the anc that live on that side are:

     * `dual_points.dat`, `dual_simplices.dat` (triangulation)
     * `g_s.dat`
     * `cy_vol.dat` and `corrected_cy_vol.dat` (Einstein frame) or equivalently string frame volumes
     * the flux vectors K, M and W0 as scalars.
   * There is **no file** that directly says “here are the 4 Kähler moduli (t^i) on the dual in CYTools’ basis”.

4. That is why:

   * Your optimization that matches (\mathcal V_S) can get a correct **volume**, but not a uniquely defined “McAllister point” in the 4-dimensional dual Kähler space.
   * And it is why trying to interpret the 214-vector as something in the dual space keeps breaking: they do not live in that space.

---

## 5. Practical takeaway

So, very concretely:

* **You are correct** in your latest note:

  > "The basis.dat file is for the primal parametrization with 218 divisors, not the dual with 8 divisors."

* `basis.dat` and `kklt_basis.dat` should be treated as:

  * chooser matrices for an (h^{1,1}=214) divisor basis on the **mirror**;
  * useful if you want to reproduce their GV computations, instanton control, mirror-side Kähler geometry etc;
  * not something to feed directly into the dual CYTools object with 8 divisors.

For your GA / physics_bridge on the **4-moduli CY**:

* Ignore `basis.dat` and `kklt_basis.dat` when working with the dual CY inside CYTools.
* Use:

  * `dual_points.dat` + `dual_simplices.dat` to build the CY,
  * `g_s.dat` to convert frames,
  * `cy_vol.dat` / `corrected_cy_vol.dat` as the gold standard volume,
  * the flux vectors and W0 as fixed external inputs.
* If you want the *exact* dual (t) consistent with their vacuum, you will need the **mirror map** between the primal’s 214 Kähler parameters and the dual’s 214 complex structure moduli, then the KKLT stabilization on that side, then map back to the dual Kähler sector. That pipeline is essentially what their “Computational Mirror Symmetry” + KKLT code does. It is not encoded in basis.dat itself.

If you like, next step could be: you paste the output of `cy_primal.h11()`, `cy_primal.num_toric_divisors()` etc for `points.dat`, and I can sketch how to verify that `basis.dat` is indeed a 214-divisor basis on that side and how `kklt_basis.dat` is just another basis. That will at least close the loop on “what do these files *actually* mean” before you decide whether you want to chase the full mirror map or treat the 4-moduli (t) as “effective parameters” for your GA.
