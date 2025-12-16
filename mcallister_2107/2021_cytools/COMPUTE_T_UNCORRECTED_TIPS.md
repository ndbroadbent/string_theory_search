There is no known “one-shot” deterministic way to go from a target divisor-volume vector τ* to “the right phase / triangulation” of the CY at large h11, and McAllister et al. explicitly say this: given only τi, they don’t know an algorithm to construct the corresponding phase of the CY hypersurface.  That is why their phase-search starts from random points in the secondary fan and then follows a path.

But: your Phase 1 script is not doing that hardest part. You already fix a triangulation via `heights=corrected_heights.dat`, so κijk is fixed. In that fixed-chamber problem, you can usually solve
[
\tau_i(t)=\frac12 \kappa_{ijk} t^j t^k = c_i
]
much faster than “100 random starts × 150 steps”.

Below are the practical changes that typically cut runtime by orders of magnitude.

---

## 1) Stop integrating with tiny fixed steps: use Newton (with damping)

Their step rule is the linearization you’re already using:
[
\kappa_{ijk} t^j \epsilon^k = \tau^{i}*{m+1}-\tau^{i}*{m}
]
(eq. 5.11).

That matrix
[
J_{ik}(t) := \frac{\partial \tau_i}{\partial t^k} = \kappa_{ijk} t^j
]
is the exact Jacobian of the nonlinear system (F(t)=\tau(t)-c=0).

So instead of taking 150 tiny Euler steps along a τ-line, just do Newton on (F(t)=0):

**Newton step**

* residual: (r = \tau(t)-c)
* solve: (J(t),\Delta t = -r)
* update: (t \leftarrow t + \lambda \Delta t) with a backtracking line search (λ ≤ 1) that enforces “physical” constraints.

### What to enforce in line search

At minimum:

* `V(t) > 0`
* `tau(t)` stays positive (or at least doesn’t flip wildly)
* optionally `t > 0` componentwise (not a true Kähler cone test, but often a good cheap guardrail)

This alone usually turns “50-100 attempts” into “1 attempt, 5-20 iterations”.

### Why this is so much faster

Your current method does ~15,000 linear solves per example (100 × 150). Newton usually needs ~10-ish solves total if you start from a decent interior point and damp correctly.

---

## 2) If you want continuation, use predictor-corrector + adaptive step size

If you really need to follow the τ-line (because you’re worried about jumping branches), keep the idea of eqs. 5.8–5.11  but upgrade the numerics:

**Predictor**

* Do what you already do: solve (5.11) once to predict (t_{m+1}^{pred}).

**Corrector**

* Run 1-3 damped Newton iterations to solve the full nonlinear equation (\tau(t)=\tau_{m+1}) starting from (t_{m+1}^{pred}).

**Adaptive step**

* If the corrector fails, halve the step in α and retry.
* If it succeeds easily, increase step (so you don’t waste 150 steps when 20 suffice).

This is standard continuation logic and is far more reliable than fixed-step Euler. It also removes most of the need for multistart.

---

## 3) Pick a much better single start t_init (this matters more than people think)

Random `abs(randn)` in 200 dimensions is almost always near “some bad wall” in some sense.

Better single-shot initializations:

### Option A (cheap, often works): scaled uniform interior

Start with `t = s * ones` and scale using the homogeneity you’re already using.

Then do damped Newton.

This is already in your “Strategy 1”, but right now you still do full path-follow for each one. If you switch Strategy 1 to Newton, you’ll likely be done.

### Option B (best if you can): maximize margin to Kähler cone walls

If you can get Mori generators (q_a) (curve classes), the Kähler cone in that phase is approximately
[
q_a \cdot t > 0 ;;\forall a
]
Then solve a small LP to find a point deep in the cone (Chebyshev center style). That t_init is dramatically more robust than random.

Even if you don’t do an LP, just sampling random positive t and rejecting those with small min(q·t) is way better than blind random.

---

## 4) Don’t build dense κ tensors and don’t use dense algebra where you don’t need it

This is “second order” compared to fixing the algorithm, but still worth it:

* You build `kappa_tensor[h11,h11,h11]` just to compute V. At h11=214 that’s big and slow to fill.
* You can compute `V(t)` directly from sparse triple intersections without materializing the full tensor.

Same for τ and J:

* τ is quadratic in t.
* J is linear in t and symmetric: (J_{ik}=J_{ki}).
* Both can be assembled from sparse κ entries in O(nnz(κ)) time.

Also, since J is symmetric (often indefinite), LDLᵀ factorization (or a robust symmetric solver) is typically more stable than generic LU.

---

## 5) Why multi-start is still sometimes necessary (but you can make it rare)

Even in the paper, they explicitly note the path-follow can fail in some examples (negative volume for some divisor, etc.).

So: it’s not crazy that you sometimes need restarts.

But needing 50-100 starts usually means one of these is happening:

1. **Your stepper is too brittle** (fixed-step Euler, no corrector).
2. **You don’t enforce constraints during updates**, so iterates wander into garbage regions and never come back.
3. **You’re not actually solving the tree-level system** in Phase 1 (for example if `compute_tau_kklt` includes χ(Di)/24 or instanton Li2 terms). If Phase 1 includes corrections, it becomes much harder. The paper’s clean “quadratic solve” is explicitly the tree-level piece (\frac12 \kappa t t = c_i).
4. **Non-favorable / basis subtleties** (your 7-51-13590 exception smells like this): mixing CYTools versions and bases can easily make “τ in kklt_basis” not be the same τ you think you’re solving for.

---

## Concrete “drop-in” plan

If you want the shortest path to speed:

1. Implement `compute_tau_tree(t)` that is definitely exactly (\tau_i=\frac12\kappa_{ijk}t^j t^k) in the correct basis (kklt basis if that’s what `c_i` is expressed in).

2. Implement `jacobian_tau_tree(t)` that returns (J_{ik}=\kappa_{ijk}t^j).

3. Run damped Newton with backtracking:

* accept a step only if it reduces `||tau(t)-c||` and keeps `V(t)>0`.

4. Only if Newton fails, fall back to continuation with adaptive α and Newton corrector.

That should reduce typical runs from minutes to seconds per example, and reduce “n_attempts” to single digits.

---

## What I’d need to be sure we’re not missing a landmine

I can give much more specific advice if you paste these two functions (or confirm their behavior):

* `SparseKappa.compute_tau_kklt(t)`
  Does it include any of the corrected terms like χ(Di)/24 or Li2 sums (cf. (4.12) and (5.13))?
  Phase 1 should not.

* `_path_follow(...)`
  Does it do pure Euler stepping, or does it do any corrector? Does it enforce any positivity constraints?

If you paste those, I can tell you exactly which change will eliminate the 50-100 attempts in your setup.
