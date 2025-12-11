You didn’t miss anything. “CYTools can’t compute periods” is only true in the narrow sense that CYTools itself stops at toric and Kähler-side geometry (intersection numbers, cones, GV via hooks, etc.). Periods are absolutely computable with other tooling, and in fact for reproducing **McAllister’s specific W₀ in 4-214-647** you can go even further: you can reproduce W₀ and gₛ **without ever constructing the full period vector** by following the paper’s derived 1D effective superpotential.

## Open-source tools that can help with periods (and what they’re actually good for)

### 1) `cygv` (Rust + Python)

Implements the HKTY procedure to compute **GV/GW invariants** for toric CY hypersurfaces/complete intersections. This is exactly the ingredient McAllister uses to build the instanton part of the prepotential. ([GitHub][1])

What it helps with:

* Getting **N_q** at large degree efficiently
* Building **F_inst** as a polylog sum (Li₃), and then derivatives for Li₂ that appear in W_flux

What it does *not* automatically give you:

* A fully normalized **symplectic period basis Π** at a generic moduli point
* Flux vacuum solving end-to-end (that’s your pipeline)

### 2) `lefschetz-family` (Python; GitHub + GitLab + PyPI)

Numerical periods using **Picard-Lefschetz / Lefschetz fibration** algorithms. This is a good sanity-check route when you want an independent numerical period computation. ([GitHub][2])

What it helps with:

* Numerical evaluation of periods for explicit hypersurface families (when you can present the defining polynomial family to it)
* Cross-checking normalization issues (up to basis choices)

Caveat:

* Scaling to “big” multiparameter families can be hard, and you still must align bases/conventions with your flux period basis.

### 3) `PeriodSuite`

Software to compute periods of hypersurfaces (period integrals, and related structure). ([GitHub][3])
This is another “independent check” style tool.

### 4) A repo that’s directly in your problem neighborhood

`AndreasSchachner/kklt_de_sitter_vacua` explicitly says it uses CYTools and includes GV invariants, and recommends `cygv` for additional invariants. ([GitHub][4])
Even if it’s not your exact mechanism, it’s a useful example of “CYTools + GV data + KKLT-ish physics plumbing.”

## Key point: for this McAllister reproduction, you don’t need full periods

McAllister derive an effective flux superpotential along the perturbatively flat direction in terms of **GV invariants and dilogarithms**:

[
W_{\mathrm{flux}}(\tau)= -\zeta \sum_{\tilde q\in M(\tilde X)} (M\cdot \tilde q), N_{\tilde q}, \mathrm{Li}_2!\left(e^{2\pi i \tau, \tilde q\cdot p}\right)
]
with
[
\zeta=\frac{1}{2^{3/2}\pi^{5/2}}.
]


So you can “compute periods” only insofar as you need the instanton-generated terms, but you **do not** need to build Π(z) and then do ((F-\tau H)\cdot \Pi) to reproduce their published W₀ example. Their pipeline already reduces it to this 1D object.

## The most important fact check: your revised plan still has several hard mistakes

Below I’m going phase-by-phase against your latest pseudocode/spec.

### Phase 1: Load geometry with exact triangulation

Good idea: use `dual_points.dat` + `dual_simplices.dat` so you’re on the same phase/triangulation.

But you *must* add a basis-consistency check:

* The flux vectors (M,K) and the intersection tensor (\tilde\kappa_{abc}) must be expressed in the **same basis** as the paper’s formulas.
* In the paper’s (4,214) example they report the flat direction (p=(293/110,163/110,163/110,13/22)).
  So the sanity check is: compute (p=N^{-1}K) (with your computed (\kappa) and the given (M,K)) and verify you reproduce that exact rational vector. If you don’t, you are in the wrong basis (even if you are on the right triangulation).

### Phase 2: Load GV invariants

Your doc currently says two incompatible things:

* “Load McAllister’s precomputed 9D curves” and then later
* “Just use CYTools `compute_gvs()` 4D basis”

Pick one for the implementation spec. Mixing them is how you keep reintroducing the 9D/4D confusion.

Also, **this is a concrete bug**:

> `q = np.array(q_tuple[:4])  # Use first 4 components if 9D`

That is not a valid projection. Ever.

If you must use the 9D ambient curve vectors, you need an actual linear map (GLSM-related) into the 4D basis used for (p,M). Otherwise your p·q and M·q are meaningless.

### Phase 3: “Verify perturbatively flat conditions”

The structure is roughly right, but the *cone test* is wrong.

You currently do:

* “p in Kähler cone (simplified: all p^a > 0)”

That is not equivalent to “p is in the Kähler cone” in general. Use the actual cone inequalities. In McAllister’s construction this is the cone (K_\infty) for the mirror Kähler moduli (the large-volume phase), not “componentwise positivity.”

Also you are missing conditions from the lemma used in the companion work (quantization constraints involving the polynomial terms of the prepotential). In Demirtaş et al. the conditions include integrality constraints like (a\cdot M\in\mathbb Z), (b\cdot M\in\mathbb Z) in addition to invertibility and the null condition.
You don’t necessarily need them to reproduce the already-published vacuum (since you already have M,K), but you should not claim “from first principles” without either:

* implementing these checks, or
* explicitly scoping them out (“we assume the published M,K already satisfy flux quantization constraints”).

### Phase 4: Find racetrack curve pairs

For reproducing the published example: you do not need a heuristic search.

McAllister literally give:

* The leading instanton charge vectors (\tilde q_i) (columns of a 4×4 integer matrix) and
* Their corresponding GV invariants (N_{\tilde q_i}),
  for the (h₂,₁,h₁,₁)=(4,214) vacuum.

They then give the truncated superpotential:
[
W_{\mathrm{flux}}(\tau)=5\zeta\left(-e^{2\pi i\tau\cdot 32/110}+512,e^{2\pi i\tau\cdot 33/110}\right)+O!\left(e^{2\pi i\tau\cdot 13/22}\right).
]


If your goal is a **gold-standard regression/unit test**, the correct approach is:

* Hardcode those two leading terms (or derive them from the listed (\tilde q_i) and (p))
* Solve analytically for Im(τ)

This avoids “find pairs” heuristics and removes a massive source of non-reproducibility.

### Phase 5: W_effective (polylog sum)

Two issues:

1. **Sign mismatch**
   Paper has a leading minus sign: (W_{\mathrm{flux}}=-\zeta\sum\cdots).
   Sign won’t change |W₀| but it matters if you’re debugging phases and cancellations.

2. **You are implicitly assuming you can sum over all curves**
   That might be fine, but then:

* `cy.compute_gvs(min_points=100)` is unlikely to give you the same dataset as McAllister’s `dual_curves_gv.dat` (5177 entries). If you rely on a truncated set, your Wflux(τ) and its derivative will be wrong unless you can prove the truncation is negligible at Im(τ)≈110.

For this example, the truncation is actually justified because everything beyond the leading q·p values is exponentially tiny at Im(τ)≈110, but you should justify it by using the same dominant charges McAllister present.

### Phase 6: Solve ∂W/∂τ = 0

This part is still not correct as written.

* You dropped the factor (2\pi i) in the derivative but still take `.imag` as if the derivative is imaginary. That makes the root-finding logic inconsistent.

* Binary search assumes monotonicity and a bracketed sign change. There is no guarantee of either for a general Li₂ sum.

For this *specific* vacuum, there is an exact analytic solution coming from the 2-term racetrack:
[
e^{2\pi,\mathrm{Im}(\tau)/110}=512\cdot \frac{33}{32}=528
\Rightarrow
g_s=\frac{1}{\mathrm{Im}(\tau)}=\frac{2\pi}{110\log 528}.
]
And then:
[
W_0 \approx 80,\zeta,528^{-33}.
]
Both are explicitly stated in the paper.

If your goal is “EXACT reproduction,” this analytic route is the correct one.

### Phase 7: Compute V₀(AdS)

Your plan’s formula

* (e^K \approx g_s/(8V^2))

does **not** match what McAllister use for their quoted vacuum energy scale.

In their section 6 they use the structure (example shown in eq. 6.24):
[
V_0 \approx -3,e^{K_0},\frac{g_s^7}{(4V[0])^2},W_0^2,
]

and they quote the final vacuum energy for the (4,214) example:
[
V_0 \approx -5.5\times 10^{-203} M_{\mathrm{pl}}^4.
]


So Phase 7 in your spec must be rewritten to match this normalization, including:

* the (g_s^7) factor,
* the (e^{K_0}) factor (which they compute from (\tilde\kappa_{abc}p^ap^bp^c) in other examples),
* and the precise meaning of (V[0]) (they treat it as the string-frame volume and separately report (V_E)).

## So did “we can’t compute periods” miss your research?

Yes in the sense that:

* There are real open-source tools for period computation (lefschetz-family, PeriodSuite) and for the GV/prepotential side (cygv). ([GitHub][1])

But also: for reproducing **this McAllister W₀**, computing the full Π(z) is not the most direct or most reliable route. The paper already reduces W₀ to a racetrack problem in τ with inputs you can validate directly from CYTools + GV data.

## What I would change in your spec right now (minimal edits, maximal correctness)

1. Add a “basis alignment” gate immediately after Phase 1:

* Compute (p) and confirm it matches eq. (6.56) for this example. If it doesn’t, stop and fix the divisor basis / transformation.

2. Delete every mention of “q_tuple[:4]”.

* Either you work in 4D curve basis throughout (recommended), or you define the explicit 9D→4D map.

3. Replace Phase 4–6 with the paper’s explicit leading-term racetrack for the unit test:

* Use eqs. (6.59)–(6.61) directly.
  This gives you deterministic reproduction without numerical fragility.

4. Replace Phase 7 with the same normalization as eq. (6.24) and the quoted (6.63)-scale result.

If you want, paste the current contents of `resources/COMPUTING_PERIODS.md` sections that list the repos and your intended “period strategy,” and I’ll map each repo/tool to a concrete pipeline stage (GV → prepotential → mirror map → periods → Wflux) and call out where basis conventions can silently break things.

[1]: https://github.com/ariostas/cygv?utm_source=chatgpt.com "ariostas/cygv: Compute GV and GW invariants of CY ..."
[2]: https://github.com/ericpipha/lefschetz-family?utm_source=chatgpt.com "ericpipha/lefschetz-family: This package implements ..."
[3]: https://github.com/emresertoz/PeriodSuite?utm_source=chatgpt.com "emresertoz/PeriodSuite: Software to compute periods of ..."
[4]: https://github.com/AndreasSchachner/kklt_de_sitter_vacua?utm_source=chatgpt.com "AndreasSchachner/kklt_de_sitter_vacua"
