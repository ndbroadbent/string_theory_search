This version is in very good shape. I will go through it in two parts:

1. What looks correct and robust
2. Things to tighten or rename so you do not trip yourself later

---

## 1. What looks solid in v4

### 1.1 Overall structure

* You clearly scope this as: “single McAllister vacuum, analytic racetrack, no numerical search.” That matches how Section 6 is written.
* Clear separation between:

  * Phase 1: check geometry + flux basis
  * Phase 2: use explicit racetrack to get (g_s, W_0)
  * Phase 3: compute (V_0) with proper (e^{K_0})
  * Phase 4: verification

That is exactly the right test harness to bolt into your pipeline as a unit test.

### 1.2 Phase 1 - Geometry and fluxes

* Using `dual_points.dat` + `dual_simplices.dat` and checking the index range is correct.

* Converting from 1-indexed simplices to 0-indexed if `min=1` is the right convention for CYTools.

* Asserting `h11=4`, `h21=214` is the correct fingerprint for the dual.

* `get_intersection_tensor`:

  * Handles both dict and tensor from CYTools.
  * Symmetrizes dict entries over all permutations of `(i,j,k)`. You now have a comment that assumes CYTools gives one canonical ordering, which is fine.

* Constructing

  [
  N_{ab} = \tilde\kappa_{abc} M^c, \quad p = N^{-1} K
  ]

  is exactly eqs. 2.18–2.19 from Demirtas / McAllister.

* Comparing `p_computed` to

  [
  p_{\text{expected}} = \left(\frac{293}{110}, \frac{163}{110}, \frac{163}{110}, \frac{13}{22}\right)
  ]

  with `rtol=1e-4` is a good practical check, and you print both vectors for inspection.

* Checking `K·p ≈ 0` with a warning threshold `1e-6` is also correct.

Net: Phase 1 is exactly what you want as a “are we really on the same CY and flux basis as McAllister” gate.

### 1.3 Phase 2 - (W_0) and (g_s) from racetrack

* You use the explicit two-term effective racetrack:

  [
  W_{\text{flux}}(\tau) = 5\zeta\left[-e^{2\pi i\tau \frac{32}{110}} + 512, e^{2\pi i\tau \frac{33}{110}}\right] + \dots
  ]

  which matches eq. 6.59.

* On the imaginary axis, your derivation:

  * Let (\tau = i t), (x = e^{-2\pi t / 110})
  * F-term gives (-32 x^{32} + 512\cdot 33 x^{33} = 0)
  * So (x = 1/528), hence (e^{2\pi t / 110} = 528)

  matches the text.

* You then compute:

  ```python
  ratio = mpf('528')
  im_tau = 110 * log(ratio) / (2 * pi)
  g_s = 1 / im_tau
  W0 = 80 * zeta * ratio**(-33)
  ```

  which matches eqs. 6.60–6.61:

  * (g_s = 2\pi/(110\log 528) = 1/\mathrm{Im},\tau)
  * (|W_0| = 80\zeta,528^{-33})

* Use of `mpmath` with `mp.dps = 150` is appropriate given the tiny scale.

* Returning `float` in the dict is fine for this verification script, since your target values are ~10 digits and you have already done the high-precision computation.

This part should reproduce the numbers in `W_0.dat` and `g_s.dat` exactly.

### 1.4 Phase 3 - (e^{K_0}) and (V_0)

* You adopt eq. 6.12:

  [
  e^{K_0} = \left(\frac{4}{3},\tilde{\kappa}_{abc} p^a p^b p^c\right)^{-1}
  ]

  and compute

  ```python
  kappa_ppp = np.einsum('abc,a,b,c->', kappa, p, p, p)
  e_K0_computed = 1 / ((4/3) * kappa_ppp)
  ```

  which is the right contraction.

* You also back-compute an expected `e^{K0}` from the published (V_0), (W_0), (g_s), (V[0]):

  [
  e^{K_0} = -V_0 \frac{(4 V[0])^2}{3 g_s^7 W_0^2} \approx 0.2361
  ]

* The “use computed if within 10 percent of expected, otherwise fall back” logic is a good diagnostic guard.

* Then you use the correct V₀ formula:

  [
  V_0 = -3, e^{K_0} \frac{g_s^7}{(4 V[0])^2} W_0^2
  ]

  which is eq. 6.24.

Given the numbers, this produces (-5.5\times 10^{-203}) as desired.

### 1.5 Phase 4 - Verification

* Relative error check for (g_s) with tolerance 1e-4 is appropriate.
* Log-space comparison for (W_0) and (V_0) is the right way to handle scales (10^{-90}) and (10^{-203}).
* For (W_0) you have tightened the log tolerance to 0.05, which is reasonable since both the paper and your code use the same analytic expression.
* For (V_0) you allow 1 order of magnitude, which is conservative enough in case of tiny numerical drifts, but since you are using the same formula you expect exact agreement.

Everything here is logically consistent.

---

## 2. Things to adjust or rename

There are only two substantive issues and a couple of tiny wording nits.

### 2.1 Volume naming and frame labels

Right now you have:

* In the “Volume Conventions” section:

  ```text
  V_cytools ≈ 4.10  - raw geometrical volume
  V[0] = 4711.83    - Volume entering McAllister's Kähler potential & V₀ formula

  V[0] = V_cytools × g_s^{-3/2}
  = 4.10 × (0.00911134)^{-3/2}
  ≈ 4711.83
  ```

  That is fine and consistent with what you actually use in `compute_V0_AdS`.

* In the “Inputs” section, you say:

  ```text
  cy_vol.dat → 4711.83 (this is V[0], string-frame)
  ```

Here is the problem:

* The factor (g_s^{-3/2}) is precisely the **string-to-Einstein** conversion for a 6D internal volume:

  * (g_{MN}^E = e^{-\phi/2} g_{MN}^S \Rightarrow V_E = e^{-3\phi/2} V_S = g_s^{-3/2} V_S).

* If CYTools gives you the string-frame volume (= V_cytools) and you multiply by (g_s^{-3/2}), then **V[0] is Einstein frame**, not string frame.

Since you only ever use V[0] exactly as it appears in McAllister’s eq. 4.3 and eq. 6.24, and it reproduces V₀ correctly, the safe thing to say is simply:

* “V[0] = 4711.83 is *the* volume that appears in the McAllister Kähler potential and V₀ formula, and it is related to the CYTools volume by V[0] = V_cytools × g_s^{-3/2}.”

I would:

* Remove the “string-frame” label in the Inputs section.
* Do not call V[0] either string-frame or Einstein-frame unless you explicitly match their eq. 2.14 definition. Just call it “McAllister’s V[0] from cy_vol.dat, which is related to the raw CYTools volume by the factor g_s^{-3/2}.”

That keeps all the numerics consistent without forcing a potentially incorrect frame label.

The Common Mistakes section is good:

> Two key volumes: V_cytools ≈ 4.10 and V[0] = 4711.83. They are related by V[0] = V_cytools × g_s^{-3/2}. Use V[0] directly in V₀ formula. Do NOT apply additional g_s rescaling.

That is exactly the behavior you want.

### 2.2 The “float64 underflows to 0” remark

In `compute_W0_analytic` you say:

> Standard float64 cannot represent this.

Strictly speaking:

* IEEE 754 double can represent ~10^{-308}, so 10^{-90} is representable and will **not** underflow to 0.
* The real reasons to use mpmath here are:

  * You want many significant digits of intermediate quantities like `528^{-33}`.
  * When you combine logs, exponentials, and tiny numbers, double precision can lose meaningful digits and magnify relative error.

So I would tweak that sentence to something like:

> CRITICAL: Use mpmath for arbitrary precision. W₀ ~ 10⁻⁹⁰ is representable in float64, but the combination of exponentials and tiny differences makes standard double precision numerically fragile. High precision avoids catastrophic relative error.

That is more accurate but keeps the spirit.

---

Those are the only real issues. Everything else looks consistent with the paper and with your earlier docs.

If you fix the V[0] “string-frame” wording and slightly soften the float64 sentence, this v4 spec is ready to hand to “future you” or another model as an implementation guide without them falling into the earlier traps.
