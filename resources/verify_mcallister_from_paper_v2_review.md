### 1. Review of `verify_mcallister_full_pipeline_v2.md`

Overall: this version is *much* closer to the actual McAllister/DKMM machinery. The key change - “just use the explicit 2–term racetrack for 4–214–647” - is exactly right and avoids a ton of unnecessary (and fragile) numerics.

I’ll go through the main parts.

---

#### 1.1 High‑level strategy

* Using the dual polytope + McAllister’s *exact* triangulation (`dual_points.dat`, `dual_simplices.dat`) is correct and matches how they build the mirror CY.
* Using fluxes
  `M = (10, 11, -11, -5)` and `K = (-3, -5, 8, 6)`
  matches eq. (6.55).
* Using
  `p = (293/110, 163/110, 163/110, 13/22)`
  from eq. (6.56) is correct.
* The racetrack formula
  `W_flux(τ) = 5 ζ [ - e^{2π i τ·(32/110)} + 512 e^{2π i τ·(33/110)} ] + O(e^{2π i τ·13/22})`
  is exactly eq. (6.59).

So the conceptual skeleton of the v2 doc is aligned with the paper.

---

#### 1.2 Phase 1: geometry + basis alignment

```python
N_ab = Σ_c κ_abc M^c
p = N^{-1} K
```

* This is exactly the DKMM/McAllister definition: in Demirtas 1912.10047 the perturbatively flat direction is defined by `N_ab = κ̃_abc M^c` and `p = N^{-1} K`.
* Checking `K·p = 0` is also the right orthogonality condition.
* Using CYTools `intersection_numbers()` as κ̃_abc on the dual (mirror) CY is correct.

**Implementation caveats (not conceptual errors):**

* You will have to be careful about index conventions:

  * CYTools often numbers divisors from 0..h11-1, while the paper labels components 1..h11.
  * `dual_simplices.dat` might be 1‑based; CYTools triangulation expects 0‑based. Your pseudocode does not mention the `-1` shift but you almost certainly need it.
* `compute_N_matrix`:

  * The dict branch assumes κ keys `(a,b,c)` already in 0..3. If CYTools returns `(i,j,k)` including ambient indices or origin, you will need to map into the CY divisor basis explicitly.

Those are “implementation gotchas,” not conceptual mistakes.

---

#### 1.3 Phase 2: analytic racetrack solution

Your derivation in `compute_W0_analytic()` matches the paper exactly.

From eq. (6.59):

[
W_{\text{flux}}(\tau) = 5\zeta\left(- e^{2\pi i\tau \frac{32}{110}} + 512 e^{2\pi i\tau\frac{33}{110}}\right) + O!\left(e^{2\pi i\tau\frac{13}{22}}\right)
]

On the imaginary axis (\tau = i t):

* Set the F‑term equation `∂τ W = 0` using only the two leading terms:
  [
  -\frac{32}{110} e^{-2\pi t \frac{32}{110}}

  * 512 \frac{33}{110} e^{-2\pi t \frac{33}{110}} = 0
    ]
* With (x = e^{-2\pi t / 110}), this becomes
  (-32 x^{32} + 512\cdot 33, x^{33} = 0)
  ⇒ (x = 32 / (512\cdot 33) = 1/528).
* Therefore (e^{2\pi t /110} = 528) and
  (t = \frac{110}{2\pi}\log(528)), so
  [
  g_s = \frac{1}{\text{Im},\tau} = \frac{2\pi}{110\log(528)}
  ]
  exactly eq. (6.60).

Your code:

```python
ratio = mpf('528')
im_tau = 110 * log(ratio) / (2 * pi)
g_s = 1 / im_tau
```

is correct.

Then you evaluate W at that τ:

* With (y = e^{-2\pi t / 110} = 1/528),
  [
  \begin{aligned}
  W_{\text{flux}} &= 5\zeta\left(-y^{32} + 512y^{33}\right) \
  &= 5\zeta\cdot 528^{-33}(-528 + 512) \
  &= -80\zeta \cdot 528^{-33}
  \end{aligned}
  ]
* So (|W_0| = 80\zeta,528^{-33}), exactly eq. (6.61).

Your code:

```python
zeta = 1 / (2**1.5 * pi**2.5)
W0 = 80 * zeta * ratio**(-33)
```

is exactly that. Good.

**Minor wording issue:** in the “Inputs” section you describe

> GV coefficients: N₁ = 1, N₂ = 512 [eq. 6.59]

That is slightly misleading. In eq. (6.59) the numbers `−1` and `512` are *net racetrack coefficients* (schematically (M·\tilde q \times N_{\tilde q}), with some additional factors) rather than raw GV invariants. The actual GV vector is shown in eq. (6.58): (N_{\tilde q} = (1, -2, 252, -2)).

For the purpose of this script it doesn’t matter (you only need the combined coefficients), but I would rephrase that line to “effective racetrack coefficients” instead of “GV coefficients.”

---

#### 1.4 Phase 3: verification

The `verify_results` logic is sound:

* Comparing g_s with relative error < 1e−4 is fine. Analytic formula will match much better than that.
* Comparing log₁₀(W₀) within ±1 is conservative and safe, but in this particular case the analytic expression should agree to machine precision with what you get by evaluating the same formula in double precision when McAllister wrote the data file.

No conceptual problems here.

---

#### 1.5 Phase 4: `compute_V0_AdS`

This is the one part that is *still* off.

You currently have:

```python
e_K0 = g_s / (8 * V_string**2)
V0 = -3 * e_K0 * g_s**7 / (4 * V_string)**2 * W0**2
```

and you comment:

> From McAllister eq. 6.24.

But eq. (6.24) is

[
V_0 \approx -3,e^{K_0} \frac{g_s^7}{(4,V_{[0]})^2} W_0^2
]

with (e^{K_0}) defined by the *complex structure* data:

[
e^{K_0} = \left(\frac{4}{3} \kappã_{abc} p^a p^b p^c \right)^{-1}
]

for the mirror threefold.

If you set (e^{K_0} = 1/(8 V^2)) you get a vacuum energy that is way too small. Quick numeric check with your numbers:

```python
W0 = 2.30012e-90
gs = 0.00911134
V  = 4711.83   # using your V_CY

eK0 = 1/(8*V*V)
V0  = -3*eK0*(gs**7)/(4*V)**2 * W0**2
# = -1.3e-210, not -5.5e-203
```

So:

* The **code as written does NOT reproduce** (-5.5\times 10^{-203}); it gives roughly (-1.3\times10^{-210}).
* To get the published value, you need (e^{K_0} \approx 0.2361). This matches the value you back‑computed earlier:
  [
  e^{K_0} = \frac{-V_0}{3,g_s^7 W_0^2 /(4 V_{[0]})^2} \approx 0.2361
  ]
  using eq. (6.24) with the published (V_0, g_s, W_0, V_{[0]}).

So for correctness:

* Either:

  * Compute (e^{K_0}) from (\kappã_{abc}) and (p) using eq. (6.12) (preferred), or
  * Treat (e^{K_0}) as an *input* for this verification script (hardcode 0.2361 for this example), or
  * Explicitly state that `compute_V0_AdS` is a rough “toy” formula and that the sample output `V0 = -5.5e-203` is *not* what this code will produce.

Right now the doc claims:

> V₀ = -5.5e-203 ✓

for that function, which is simply false.

**Frame wording inconsistency:**

* In the “Inputs” box you still say:
  `cy_vol.dat → 4711.83 (Einstein frame)`
* But in this function you label the argument as
  `V_CY (string) = {float(V_string):.2f}`

From the paper:

* They find (V_{[0]} \approx 4711) and Einstein volume (V_E \approx 5.4\times 10^6).
* In your own earlier analysis you observed CYTools gives (V_{\text{string}} \approx 4.10) and (V_E \approx 4711) using (V_E = V_S g_s^{-3/2}).

So there are *three* volumes:

1. CYTools string-frame volume (V_S \approx 4.10).
2. McAllister’s corrected Kähler‑potential volume (V_{[0]} \approx 4711).
3. Einstein-frame volume (V_E \approx 5.4\times 10^6).

Your docs currently conflate these. For this script, pick one convention and stick to it:

* If `V_CY` is meant to be (V_{[0]}) (as in eq. 6.24), say so explicitly.
* Do not call 4711 “Einstein frame” in one place and “string” in another.

---

#### 1.6 Expected output section

The sample output:

> V₀ = -5.5e-203

will *not* be produced by the code as currently written, for the reasons above. Everything about g_s and W₀ is fine, but you should either:

* Remove V₀ from the “SUCCESS” criteria, or
* Fix `compute_V0_AdS` to use the correct (e^{K_0}).

---

### 2. Review of `COMPUTING_PERIODS.md`

This doc is mostly a conceptual roadmap for the general case. It is broadly consistent with the McAllister paper and the Demirtas “Vacua with small flux superpotential” construction. I’ll highlight the key points and the couple of places where it goes off.

---

#### 2.1 Prepotential and periods

You have:

```text
F_poly = -1/6 κ̃_abc z^a z^b z^c
         + 1/2 ã_ab z^a z^b
         + 1/24 c̃_a z^a
         + ζ(3)χ(X̃)/(2(2πi)³)

F_inst = -1/(2πi)³ Σ_q N_q Li₃(e^(2πi q·z))
```

* The *instanton* piece matches eq. (2.3) in McAllister:
  (F_{\text{inst}}(z) = -\frac{1}{(2\pi i)^3}\sum_{\tilde q} N_{\tilde q} \operatorname{Li}_3(e^{2\pi i \tilde q\cdot z})).
* The overall form of (F_{\text{poly}}) (cubic term in (\kappã), quadratic term, linear term, ζ(3)χ term) is the standard HKTY structure. Exact numerical coefficients on the subleading terms depend on conventions, but your pattern is consistent with the literature.

The period vector:

```text
Π = (F_A, z^A)^T
```

This is just a choice of ordering. In McAllister they use ((z^A, F_A)) and then contract with a symplectic matrix Σ. As long as you use the same convention consistently in `W_flux = sqrt(2/π) Π^T Σ (f - τ h)` it is fine. Right now the doc mixes:

* Sect. 1.1: Π = (F_A, z^A)
* Sect. 3.3: Π = (∫_A Ω, ∫_B Ω) = (z^I, F_I)

Those are the same objects but in reversed order: make sure your code agrees with the convention in eq. (2.11) of McAllister.

---

#### 2.2 Flux superpotential and ζ factor

You wrote:

```text
W_flux(τ, z) = √(2/π) Π^T Σ (f - τh)
```

This matches eq. (2.11):

[
W_{\text{flux}}(\tau,z^a) = \sqrt{\frac{2}{\pi}} \int (F_3 - \tau H_3) \wedge \Omega
= \sqrt{\frac{2}{\pi}},\Pi^T \Sigma (f - \tau h)
]

Later you use the effective racetrack expression

```text
W_eff(τ) = ζ Σ_q (M·q) N_q Li₂(e^{2πiτ p·q})
```

with

[
\zeta = \frac{1}{2^{3/2}\pi^{5/2}}
]

which is exactly eq. (2.23) and (2.22) in the paper.

So the ζ normalisation is correct.

---

#### 2.3 Perturbatively flat vacua and p = N⁻¹K

Your summary of the lemma:

* (N_{ab} = κ̃_{abc} M^c) invertible.
* (p = N^{-1} K) inside the Kähler cone.
* (K·p = 0).

matches Demirtas 1912.10047 and the McAllister discussion around eqs. (2.18)–(2.22).

The description of how W₀ arises from a racetrack in (\tau) using two instanton terms with small ε = p·(q₂ − q₁) is also faithful to the paper’s Section 2 and Section 6.

---

#### 2.4 Periods from prepotential

You state:

> GV invariants determine the prepotential F(z)
> Prepotential determines periods Π(z) = (z^A, F_A)

This is exactly right in the large‑complex‑structure frame. Given:

* κ̃_abc and the topological data for the polynomial part, and
* GV invariants for the instanton part,

you can construct F(z), then F_A = ∂F/∂z^A, and hence Π.

So the overall story of “cygv gives GV invariants → build F → get Π” is sound.

---

#### 2.5 The big problem: `compute_V0_AdS` in this doc

Same issue as in the v2 pipeline doc. Here you write:

```python
def compute_V0_AdS(W_0, g_s, V_CY):
    """
    V₀ = -3 e^K₀ (g_s⁷ / (4V)²) W₀²

    At the KKLT minimum with |W₀| << 1.
    """
    # Kähler potential at minimum
    # e^K₀ ≈ 1 / (8 V^2) for large volume
    e_K0 = 1 / (8 * V_CY**2)

    # Prefactor from supergravity
    prefactor = -3 * e_K0 * (g_s**7) / (4*V_CY)**2

    V0 = prefactor * W_0**2
    return V0

# McAllister values:
V0 = compute_V0_AdS(2.3e-90, 0.00911134, 4711.83)
# V0 ≈ -5.5e-203 ✓
```

As above, plugging these numbers in gives ~−1.3×10⁻²¹⁰, not −5.5×10⁻²⁰³.

What eq. (6.24) actually says is:

[
V_0 \approx -3 e^{K_0} \frac{g_s^7}{(4V_{[0]})^2} W_0^2
]

with

[
e^{K_0} = \left(\frac{4}{3} \kappã_{abc} p^a p^b p^c \right)^{-1}
]

not (1/(8V^2)).

So the code snippet and the “✓” comment are inconsistent with the actual physics. You already know the correct e^{K0} for the 4–214–647 example (≈ 0.2361); that is what makes eq. (6.24) reproduce the published V₀. The “1/(8V²)” thing should be clearly marked as an *approximate toy* and not claimed to give the paper’s value.

---

#### 2.6 Frames and volumes

In the “Key Parameters for McAllister’s 4‑214‑647 Example” section you write:

> V_CY = 4711.83 (Einstein frame volume)

But from the paper:

* They say (V_{[0]} \approx 4711) and Einstein frame volume (V_E \approx 5.4\times 10^6).
* From your own earlier CYTools debugging:

  * CYTools gives a “raw” CY volume ≈ 4.10.
  * Multiplying by (g_s^{-3/2} ≈ 1150) gives 4711.8, which you identified as McAllister’s *Einstein* frame volume.

There is a real three‑way normalization tangle here:

1. CYTools’ `compute_cy_volume(t)` volume.
2. The string‑frame (V_{[0]}) that enters the Kähler potential (and eq. 4.3, 4.4).
3. The Einstein‑frame volume V_E defined in eq. (2.14).

Your docs currently call 4711 “Einstein,” “string,” and “V[0]” in different places. That will absolutely bite you when you wire the pipeline together.

I would explicitly standardize:

* Let `V_cytools` be what CYTools gives.
* Let `V_string` (or `V[0]`) be what enters the Kähler potential.
* Let `V_Einstein` be what the paper calls V_E.

Then write down exactly how CYTools’ normalization relates to V_string and V_E for this example. You have enough information from eqs. (2.14), (4.3), (4.4), and (6.59–6.62) to do that mapping precisely.

---

#### 2.7 Everything else

The rest of `COMPUTING_PERIODS.md`:

* The racetrack description, ε = p·(q₂ − q₁), δ hierarchy, etc, matches the structure of eqs. (2.24) and the worked examples in Section 6.
* The multi‑objective fitness hierarchy (Tier 1: topology, Tier 2: gauge couplings, Tier 3: W₀/Λ) is logically sound and matches your implementation constraints.
* The suggestion to use cygv and, optionally, Lefschetz/Picard‑Fuchs codes to cross‑check periods is reasonable.

I do not see any other hard physics errors beyond:

* The V₀ formula and its “✓” comment.
* The frame/volume naming inconsistencies.
* The minor Π ordering convention drift.

---

### TL;DR

* `verify_mcallister_full_pipeline_v2.md` is conceptually correct for reproducing **g_s** and **W₀** for 4‑214‑647, *provided*:

  * You fix or clearly demote the approximate `compute_V0_AdS` and clean up the volume/frame naming.
  * You treat 1 and 512 in eq. (6.59) as “effective racetrack coefficients,” not bare GV invariants.

* `COMPUTING_PERIODS.md` is a good high‑level map of the full DKMM/McAllister machinery, with the same two caveats:

  * The V₀ helper function is numerically inconsistent with eq. (6.24) and the sample numbers.
  * The string vs Einstein vs CYTools volume labels around “4711.83” need to be disentangled and made consistent.

If you clean up those points, both docs will be solid foundations for the actual Python implementation and for plugging the 4‑214‑647 vacuum cleanly into your GA.
