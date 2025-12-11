# Re-derive Two-Term Racetrack from GV Inputs (v2)

**Goal**: Starting from geometry + flux + GV invariants, reconstruct the effective 1D superpotential
\[ W_{\text{flux}}(\tau) \]
and show that, for 4-214-647 with McAllister’s flux choice, it reduces to the 2-term racetrack of eq. (6.59):
\[
W_{\text{flux}}(\tau) \approx 5 \zeta \left[-e^{2 \pi i \tau \cdot \frac{32}{110}} + 512 \, e^{2 \pi i \tau \cdot \frac{33}{110}}\right] + \mathcal{O}(e^{2 \pi i \tau \cdot 13/22})
\]

**Why this matters**:
- Validates that we can go from **GV invariants + fluxes** to the racetrack data (exponents and coefficients) *without* hard-coding eq. (6.59).
- This is the missing link between the “from first principles” pipeline and the analytic shortcuts used in v6.

**Status**: PARTIALLY BLOCKED - basis alignment is still the main issue. This doc refines the open questions and proposes concrete diagnostic steps.

---

## 0. What We Already Know / Verified

From the existing code and notes:

1. **Geometry / GV side**
   - CYTools, with McAllister’s triangulation for 4-214-647 (dual), reproduces:
     - \( h^{1,1} = 4 \), \( h^{2,1} = 214 \)
     - GV invariants for simple curve classes, e.g. \( (1,0,0,0) \mapsto 252 \), match their data.
   - `dual_curves.dat` + `dual_curves_gv.dat`:
     - 5177 curves in a 9D *ambient toric* basis.
     - The first few rows have GV values matching eq. (6.58):
       - GV list \( (1, -2, 252, -2) \) appears.

2. **“Small curves” side**
   - `small_curves.dat`: 344 curves in a 219D *primal ambient* basis.
   - `small_curves_gv.dat`: GV for those 344 curves, mostly 1 and -2, **no 252**.
   - `small_curves_vols.dat`: 344 floats, with:
     - Some negative values.
     - Values in [0,1] that look like “small something” (volumes / q·p / related).
     - Numbers near 0.29 and 0.30 but not equal to 32/110 and 33/110 exactly.

3. **Flux & flat direction**
   - Fluxes (in McAllister’s moduli basis) from eq. (6.55):
     - \( K = (-3,-5,8,6) \)
     - \( M = (10,11,-11,-5) \)
   - Flat direction (same basis) from eq. (6.56):
     - \( p = (293/110, \, 163/110, \, 163/110, \, 13/22) \)
   - These satisfy the Demirtas perturbatively flat conditions *in the paper’s basis*.

4. **Analytic racetrack**
   - Eq. (6.58) gives four special curves with GV \( (1,-2,252,-2) \).
   - Eq. (6.59) re-expresses all relevant contributions as a **2-term racetrack** with:
     - Effective exponents \( p \cdot \tilde q = 32/110, 33/110 \) and \(13/22\).
     - Effective coefficients \(-1\) and \(+512\).
   - v6 script shows that plugging these back into the 1D W(τ) indeed reproduces:
     - \( g_s = 0.00911134 \)
     - \( W_0 \approx 2.3 \times 10^{-90} \)
     - \( V_0 \approx -5.5 \times 10^{-203} \)

**The missing piece**: Show that if you start from **curves + GV + fluxes only**, and apply the general formula
\[
W_{\text{flux}}(\tau) = -\zeta \sum_{\tilde q} (M \cdot \tilde q) N_{\tilde q} \, \text{Li}_2(e^{2\pi i \tau \, (\tilde q \cdot p)})
\]
you automatically rediscover those two leading exponents and the effective coefficients. That requires putting \(M\), \(p\), and \(\tilde q\) in a **consistent basis**.

---

## 1. Clarifying the Basis Problem

### 1.1 Different spaces / bases involved

- **Ambient divisors (dual side)**:
  - 9 toric rays (indices 0..8) correspond to divisors \(D_i\) in the ambient toric variety.
  - `dual_curves.dat` stores curve classes as 9D integer vectors \(q^{\text{amb}} \in \mathbb{Z}^9\).

- **CY divisor classes (dual CY)**:
  - \( h^{1,1} = 4 \) independent divisor classes.
  - CYTools picks some 4 linear combinations of ambient divisors as its **basis**.
    - `cy.divisor_basis()` returns the ambient indices used, e.g. `[5,6,7,8]`.

- **Curve classes in H₂(X̃)**:
  - The Mori cone is rank 4. A curve class \(\tilde q\) can be represented:
    - As a 4-vector in some H₂ basis (McAllister’s “moduli basis”).
    - Or as intersection numbers with the ambient divisors (your 9D vector).

- **Flux & moduli basis (paper)**:
  - McAllister’s flux vectors \(M, K\) and flat direction \(p\) are 4D vectors in a particular H₂/H⁴ basis.
  - CYTools’ 4D divisor basis is related to the paper’s basis by some unknown \(T \in GL(4,\mathbb{Z})\):
    - Divisors: \(D^{\text{paper}} = T \, D^{\text{cytools}}\)
    - Curves: \(\tilde q^{\text{paper}} = (T^{-1})^T \, \tilde q^{\text{cytools}}\)
    - Flux components: \(M^{\text{cytools}} = T \, M^{\text{paper}}\), etc.

We do **not** currently know \(T\). This is the core obstruction for doing everything in the CYTools basis.

### 1.2 Important realization

For **this v1/v2 racetrack re-derivation**, we do not *necessarily* need \(T\) explicitly. Possible approaches:

1. Work purely in the **paper basis**, using McAllister’s own “small_curves” data if that data is already expressed in that basis (or at least already contracted with \(p\)).
2. Or solve for \(T\) once, using relations between:
   - CYTools intersection numbers κ(in_basis=True)
   - Known p in paper basis, and p_cytools computed via N⁻¹K in CYTools basis
   - Possibly the small_curves_vols entries

Option 1 is much simpler if `small_curves_vols.dat` really are the \( \tilde q \cdot p \) values used in Section 6.

---

## 2. Refining the Open Questions

### Q1: How to project 9D ambient curves to 4D h¹¹ basis?

In abstract toric geometry:

- There is a GLSM charge matrix \(Q\) of shape \((9-4) \times 9 = 5 \times 9\), encoding linear relations among the divisor classes.
- Divisor classes in H²(X̃) are 4-dimensional; curves in H₂(X̃) live in the dual space.
- A 9D ambient curve vector \(q^{\text{amb}}\) is effectively its intersection with the toric divisors; to go to a 4D basis you need to know how those divisors map to the 4 H² generators.

In CYTools:

- `cy.glsm_charge_matrix(include_origin=False)` (or similar) gives you the GLSM charges \(Q\).
- `cy.divisor_basis()` gives you the ambient indices picked as the H² basis.
- In principle, you can:
  1. Express each ambient divisor \(D_i\) as a linear combination of the basis divisors.
  2. Then express each ambient curve vector \(q^{\text{amb}}\) as the corresponding 4D vector \(q^{\text{cytools}}\) via duality.

**But**: this is not a one-liner; it requires careful solving of linear systems and verifying integrality. We do not currently have a tested recipe that does this correctly.

**Plan for Q1**:

- [ ] Add a sandbox script to:
  - Retrieve `Q = cy.glsm_charge_matrix(include_origin=True/False)`.
  - Retrieve `basis = cy.divisor_basis()`.
  - For each non-basis divisor D_i, solve for its coordinates in the basis using Q.
  - Verify that intersection numbers κ in the computed basis match `intersection_numbers(in_basis=True)`.

- [ ] Once that is working, derive a mapping that takes 9D curves → 4D curves in CYTools basis.

At this stage, Q1 remains **open**, but we now have a concrete experiment to run.

---

### Q2: What basis is McAllister’s p in?

We know:

- p_paper = (293/110, 163/110, 163/110, 13/22) is defined by N⁻¹K with N_ab = κ̃_abc M^c in *McAllister’s* basis.
- CYTools in-basis κ and the same (K,M) do **not** give that p; they give some p_cytools instead.

That means:

- There exists a GL(4,Z) matrix T such that:
  - κ_paper = (T^{-1})^T κ_cytools T^{-1}
  - M_paper = T^{-1} M_cytools
  - K_paper = T^{-1} K_cytools
  - p_paper = T^{-1} p_cytools

Solving for T from the data you have is not trivial, but:

- You know p_paper and p_cytools numerically.
- You know κ_cytools.
- You know N_ab in CYTools basis, and you know that in paper basis N_paper p_paper = K_paper.

So in principle, there is an overdetermined system that could recover T.

**However**: this is a messy integer linear algebra problem and may itself deserve its own doc and code.

For the purpose of *re-deriving the racetrack* rather than *rebuilding the entire Demirtas lemma machinery*, we can temporarily avoid solving for T explicitly if we can work in the paper basis via their own ancillary data (small_curves).

So Q2 stays **open**, with the note:

- Long-term, solving for T is essential to generalize the pipeline across different tools/bases.
- Short-term, racetrack re-derivation may be simpler by staying in the “McAllister basis world” and not touching CYTools for curves.

---

### Q3: How to compute M·q in a consistent basis?

This is essentially the same basis problem as Q1+Q2:

- Formula needs \(M \cdot \tilde q\).
- You currently know:
  - M_paper as a 4-vector.
  - q as 9D ambient (dual_curves) or 219D primal (small_curves).
- Without a known linear map from those ambient coordinates to the 4D moduli basis, you cannot compute \(M \cdot \tilde q\) directly.

**Potential shortcut**:

- If the McAllister ancillary code already precomputes the effective coefficients \((M \cdot \tilde q) N_{\tilde q}\) and stores them somewhere (e.g. another file), you could treat those as inputs.
- So far, the only files you’ve inspected are:
  - dual_curves(+_gv)
  - small_curves(+_gv/_vols/_cutoff)
- It is worth checking the rest of the anc files for any:
  - `*_coeffs.dat`, `small_curves_*`, or similar.

Until we find that, Q3 is **blocked** in the strict sense.

---

### Q4: Where does the factor 5 in eq. 6.59 come from?

Your options:

- Degeneracy: maybe there are 5 curves (or 5 symmetry-related orbits) contributing equally to the leading pair, so the sum over them multiplies by 5.
- Normalization: maybe the 5 comes from some prefactor in the reduction from full 3D periods to 1D effective theory.

Without the ability to read the relevant section text right now, the safe strategy is:

- Treat 5 as the **sum over contributions of a specific set of small curves**:
  \[
  5(-1, 512) \equiv \sum_{i \in \text{orbit}} (M \cdot \tilde q_i) N_{\tilde q_i}
  \]
- Once Q1–Q3 are solved, you should be able to *compute* that sum and see if it equals 5 times the leading pair.

So for v2: Q4 remains a **“to be checked numerically”** item after we can compute \(M\cdot \tilde q\).

---

### Q5: What is small_curves_vols.dat actually storing?

Your observations:

- 344 floats, some negative.
- Many values around 0.2–0.8.
- Some near 0.2888 and 0.2996, close to 32/110 ≈ 0.2909 and 33/110 = 0.3, but not exact.
- There is a `small_curves_cutoff.dat` containing `1.0`, suggesting curves with “volume” < 1 are selected.

Candidate interpretations:

1. **Hypothesis H1**: `small_curves_vols` are exactly \( \tilde q \cdot p \) in McAllister’s basis, evaluated at the flat point p.
   - Then:
     - The cutoff 1.0 is literally “q·p < 1”.
     - Differences 0.2888 vs 32/110 might be due to:
       - Use of a slightly different p (e.g. approximate numerical p, not the rational one).
       - Different normalization (units).
   - This is *testable*:
     - For each small curve row, compute the *intersection* of that curve with the divisors corresponding to the moduli basis, contract with p, and compare.

2. **Hypothesis H2**: `small_curves_vols` are *actual curve volumes* in string units:
   - Something like \( \text{Vol}(C) = \int_C J = q \cdot t \) where t are Kähler moduli, not p.
   - Negative entries might indicate curves lying outside some cone but included for completeness.

3. **Hypothesis H3**: They are volumes in the *primal* 214-dimensional picture, not the dual 4D one.

For v2, we should **not rely on small_curves_vols as q·p** until H1 is confirmed.

**Concrete experiment for H1** (you can run locally):

1. Take two of the candidate curves near 0.29 and 0.30 in small_curves_vols.dat.
2. See which rows of small_curves.dat they correspond to.
3. Figure out, using the paper’s description or ancillary code, how the 219D ambient basis there maps to the 4D moduli basis where p is defined (they might have documented this in a separate file).
4. Contract the mapped 4D q with p and see if it matches the `small_curves_vols` entry.

Until that is done, we treat Q5 as **open**.

---

## 3. Revised “Re-derive Racetrack” Plan

We now distinguish two levels of ambition:

- **Level 1 (we can likely achieve soon)**:
  - Working entirely in “McAllister’s basis world”, use their small_curves datasets and p, M, K to reconstruct numerically:
    - The list of q·p for relevant curves.
    - The effective coefficients (M·q) N_q.
    - Show that the sum over these reproduces eq. (6.59) to numerical precision.

- **Level 2 (harder, generalizable)**:
  - Starting strictly from:
    - dual_points.dat, dual_simplices.dat
    - dual_curves.dat, dual_curves_gv.dat
    - K_vec.dat, M_vec.dat
  - Use CYTools + linear algebra to:
    - Map curves from 9D ambient basis to 4D CYTools basis.
    - Solve for GL(4,Z) transformation T to map to McAllister’s moduli basis.
    - Then rederive eq. (6.59) from the raw GV data.

For the GA use-case, Level 2 is ultimately what you want, but Level 1 is a good “sanity milestone” and avoids touching T at first.

### 3.1 Level 1: “McAllister-basis-only” pipeline

Assuming we can understand small_curves.dat and small_curves_vols.dat:

```python
def build_racetrack_from_small_curves():
    # Load small_curves.dat, small_curves_gv.dat, small_curves_vols.dat
    curves = load_small_curves()      # 344 x 219 ints
    gvs = load_small_curves_gv()      # 344 ints (mostly 1 or -2)
    vols = load_small_curves_vols()   # 344 floats (hypothesis: q·p or closely related)

    # Flux vector M in the same basis as small_curves (to be clarified!)
    # This is the main missing piece for Level 1 right now.
    M = M_paper  # if small_curves are already built in McAllister's 4D basis
                 # otherwise, need the projection.

    terms = []
    for i in range(len(curves)):
        q_p = vols[i]            # Hypothesis: q·p
        N_q = gvs[i]             # GV invariant
        M_dot_q = ...            # Need basis for M·q

        if 0 < q_p < 1:
            eff = M_dot_q * N_q
            terms.append({
                'idx': i,
                'q_dot_p': q_p,
                'N_q': N_q,
                'M_dot_q': M_dot_q,
                'coeff': eff,
            })

    terms.sort(key=lambda t: t['q_dot_p'])
    return terms[:10]
