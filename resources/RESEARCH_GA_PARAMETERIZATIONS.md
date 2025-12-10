# Research Question: Optimal Parameterization for GA Search in the String Landscape

## Context

We have a working genetic algorithm (GA) searching through Calabi-Yau compactifications to find configurations matching Standard Model physics. We have successfully reproduced McAllister et al.'s published result (arXiv:2107.09064) for polytope 4-214-647:

- **CY Volume (Einstein frame)**: 4711.83
- **W₀**: 2.30×10⁻⁹⁰
- **g_s**: 0.00911134
- **Kähler moduli** (our solution): t ≈ [49.6, 38.9, 35.0, 26.2]

This provides a **gold standard test case** where we know the exact optimal point in parameter space.

## The Core Problem

There are multiple ways to parameterize the same physical configuration. Each parameterization creates a different fitness landscape for the GA. We need to understand:

1. Which parameterization creates the smoothest gradients?
2. Which one allows small mutations to produce small fitness changes?
3. Can we design a parameterization specifically optimized for GA search?

## Known Parameterizations

### 1. Kähler Moduli (h¹¹-dimensional) — What CYTools Uses

The fundamental parameters: t^i for i = 1, ..., h¹¹ (4 values for polytope 647).

**CY Volume formula**: V_S = (1/6) κ_{ijk} t^i t^j t^k (cubic polynomial)

**Divisor volumes**: τ_A = (1/2) κ_{Ajk} t^j t^k (quadratic in t)

**Pros**:
- Minimal dimension (just h¹¹ parameters)
- Directly enters physics formulas
- CYTools computes volumes from these

**Cons**:
- Must stay in Kähler cone (convex but complex constraints)
- Small changes in t can cause large changes in V if near cone boundary
- Nonlinear (cubic) relationship to volume

### 2. Ambient/Secondary-Fan Parameters (214-dimensional for 647) — What McAllister Uses

One parameter per non-basis toric divisor. These are what McAllister stores in `kahler_param.dat`.

**Relationship**: The 214 values can be projected down to 4 Kähler moduli via GLSM linear relations. Alternatively, they encode "where you are" in the secondary fan / triangulation space.

**Pros**:
- More direct control over individual divisor volumes
- Path-following algorithms can walk through triangulation phases
- Natural for enforcing τ_A ≥ 1 constraints

**Cons**:
- Highly overdetermined (214 params → 4 moduli)
- Redundant degrees of freedom
- Not clear how to mutate meaningfully

### 3. Target Divisor Volumes (Einstein frame)

Specify the desired volumes τ_A for each toric divisor, then solve for t.

**What McAllister does**: `target_volumes.dat` contains integers (1 or 6), representing target Einstein-frame divisor volumes.

**Pros**:
- Directly controls the physics (instanton contributions depend on τ)
- Integer targets give discrete search space

**Cons**:
- Must solve nonlinear system to get t
- Not all target volume combinations are achievable
- Overdetermined constraints

### 4. Our "Ray Direction" Parameterization (h¹¹-dimensional)

Our current GA approach in `physics_bridge.py`:

```python
def _find_kahler_in_cone(self, cy_data, genome):
    # Get direction from genome (normalize to unit vector)
    direction = normalize(genome["kahler_moduli"])

    # Start from tip of cone (guaranteed inside)
    tip = cone.tip_of_stretched_cone(1.0)

    # Binary search to find where ray exits cone
    # Return point at 50% of way from tip to boundary
    t_final = t_min * 0.5
    return tip + t_final * direction
```

**How it works**:
1. Genome specifies a **direction** in h¹¹-dimensional space
2. We start at the cone's tip (always valid)
3. We trace a ray in that direction until it exits the cone
4. We return a point at 50% of the way to the boundary

**Key property**: Small direction changes → small output changes (smooth gradients)

**Pros**:
- Always produces valid Kähler point (automatic constraint satisfaction)
- Direction mutations are meaningful (rotate the ray)
- Scales naturally with cone geometry

**Cons**:
- Can't reach arbitrary points in the cone
- Biased toward points near the tip
- Volume range limited by ray length

### 5. Frame Conversion Factor

Another degree of freedom: **string coupling g_s**.

**Frame conversion**: V_E = V_S × g_s^{-3/2}

For McAllister's g_s = 0.00911134, this factor is ~1150.

**Implication**: The GA could search in (V_S, g_s) space instead of just V_S. Small g_s dramatically amplifies string-frame volume to Einstein-frame volume.

## The Key Questions

### Q1: Gradient Smoothness Analysis

Starting from McAllister's optimal point t* = [49.6, 38.9, 35.0, 26.2]:

**Experiment**: Take steps of size ε along each coordinate axis:
- t' = t* + ε × e_i (where e_i is the i-th unit vector)
- Compute fitness(t') for ε = 0.01, 0.1, 1.0, 10.0

**What we want to learn**:
1. Is fitness(t' ) a smooth function of ε?
2. Does fitness degrade monotonically as we move away?
3. Are there cliffs or discontinuities?

If the gradient is smooth, standard gradient descent would work. If it's noisy/discontinuous, GA crossover might accidentally jump to bad regions.

### Q2: Coordinate System Comparison

For the same physical perturbation, which coordinate system shows the smoothest fitness change?

**Experiment**: Define a "physics perturbation" (e.g., increase CY volume by 1%):
1. In Kähler moduli: scale all t^i by α = 1.01^{1/3} (since V ~ t³)
2. In ambient params: adjust all 214 values proportionally
3. In target volumes: increase τ targets by some factor
4. In ray direction: rotate direction slightly

**Measure**: |Δfitness| / |Δparameter| — the gradient magnitude

The parameterization with the most consistent, moderate gradient is best for GA.

### Q3: Can We Invent a Better Parameterization?

Ideas to explore:

**A. Log-space Kähler moduli**: Use log(t^i) instead of t^i
- Pros: Handles orders-of-magnitude changes better
- Makes multiplicative mutations additive

**B. Volume-preserving mutations**: Parameterize as (V, shape) where shape is constrained to V=const hypersurface
- Decouple volume optimization from shape optimization

**C. Principal component space**: Transform to eigenbasis of the intersection tensor κ_{ijk}
- Might diagonalize the fitness landscape

**D. Divisor volume space with regularization**: Optimize τ_A directly, project to t via least-squares
- Natural for enforcing τ ≥ 1 constraints

### Q4: McAllister as Ground Truth

**The ultimate test**: Starting from a random point, can we GA our way to McAllister's optimal?

**Setup**:
1. Use polytope 4-214-647 with h¹¹=4
2. Target: reproduce V_E = 4711.83 and/or W₀ = 2.3×10⁻⁹⁰
3. Try each parameterization
4. Measure: generations to reach target, success rate, wall-clock time

**What this tells us**: Which parameterization makes the "basin of attraction" around the optimum largest and smoothest?

## Specific Data Needed

To answer these questions, we need:

1. **Intersection tensor κ_{ijk}** for polytope 647 — determines how t affects V
2. **GLSM charge matrix** — converts between 214-space and 4-space
3. **McAllister's exact triangulation** from `dual_simplices.dat` — ensures same geometry
4. **Divisor basis mapping** from `basis.dat` — ensures same indexing

## Proposed Experiments

### Experiment 1: Local Gradient Analysis
Starting from t* = [49.6, 38.9, 35.0, 26.2], compute:
- V(t* + ε·e_i) for i=1,2,3,4 and ε ∈ [-10, 10]
- Plot V vs ε for each direction
- Check if V changes smoothly or has discontinuities

### Experiment 2: Basin of Attraction Mapping
Sample 1000 random starting points in Kähler cone. Run gradient descent toward t*. Measure:
- How many converge to t*?
- What's the average path length?
- Are there local minima trapping the descent?

### Experiment 3: Parameterization Tournament
Run GA with each parameterization, targeting V_E = 4711.83:
1. Standard Kähler moduli t^i
2. Log-Kähler moduli log(t^i)
3. Ray direction (our current method)
4. Target divisor volumes

Measure: success rate, generations to converge, final fitness.

### Experiment 4: Sensitivity Analysis
For each parameterization, compute:
- ∂V/∂param_i at the optimum
- Condition number of the Jacobian
- Eigenvalue spectrum

The best parameterization has moderate, well-conditioned gradients (not too large, not too small, not too variable).

## Summary Questions for Deep Research

1. **What is the fitness landscape topology** around McAllister's optimum in each parameterization? (Smooth bowl vs rugged terrain)

2. **Which parameterization has the largest basin of attraction** around known optima?

3. **Is there a theoretical reason** why one parameterization should work better for GA? (E.g., does log-space linearize the cubic V(t) relationship?)

4. **Can we learn the optimal parameterization** from data? (Meta-learn a coordinate transform that makes the landscape smooth)

5. **How do the different parameterizations compose with GA operators** (crossover, mutation)? Which one preserves good solutions under genetic operations?

The answer to these questions will determine whether our current raytracing approach is optimal, or whether we should switch to a fundamentally different parameterization.


------------------------------


Answers from ChatGPT 5.1 Pro:

Treat “how you parameterize J” as an engineering choice that shapes curvature, constraints, and mutation semantics.

Below is a concise, actionable plan that you can drop into your pipeline. It culminates in a parameterization that is both cone-safe and well‑conditioned near the McAllister point.

---

## Key facts to exploit

* Classical string‑frame volume:

  * ( \mathcal V_S(t)=\tfrac16,\kappa_{ijk}t^it^jt^k )
  * ( \partial \mathcal V_S/\partial t^i=\tfrac12,\kappa_{ijk}t^jt^k=\tau_i(t) )
    So the gradient in a divisor basis **is exactly the divisor volumes**.
  * Hessian: ( H_{ij}(t)=\kappa_{ijk}t^k ) - linear in (t).
* Scaling: ( \mathcal V_S(\alpha t)=\alpha^3 \mathcal V_S(t) ). So separating “scale” and “shape” is natural.
* Einstein vs string: ( \mathcal V_E=g_s^{-3/2}\mathcal V_S ) - a constant rescaling if fluxes (and hence (g_s)) are fixed.

These make it easy to design smooth coordinates and to analytically precondition the landscape.

---

## Recommendation at a glance

**Use “scale-shape, cone‑generator, log‑coords”**:

1. Fix an FRST, basis, and (\kappa_{ijk}) as you already do for 4-214-647.
2. Express the Kähler cone ( \mathcal K ) as the positive span of its rays (r_\alpha) in your divisor basis:
   [
   t = e^{s}, T \lambda,\quad \lambda_\alpha>0,\quad T=[r_1,r_2,\dots r_{m}] \in \mathbb R^{h^{1,1}\times m}
   ]

   * (s\in\mathbb R) is the **scale** gene.
   * (\lambda\in\mathbb R_{>0}^{m}) are **shape** genes in the cone.
   * Enforce interior with (\lambda_\alpha=\lambda_{\min}+\mathrm{softplus}(z_\alpha)) where (z_\alpha\in\mathbb R) are unconstrained genes and (\lambda_{\min}>0) is a small margin.
3. Optionally **normalize shape**: impose (\sum_\alpha \lambda_\alpha=1) (simplex) so scale lives entirely in (s). Implement with (\lambda=\mathrm{softmax}(z)) then multiply by a free positive “shape scale” gene if you prefer.

**Why this is good**

* Always inside the cone - constraint solved by construction.
* Mutations in (z) and (s) map smoothly to (t).
* Natural separation: (s) tunes overall volume; (z) tunes shape.
* Derivatives are tame and analytic:
  [
  \mathcal V_S = \tfrac16,\kappa'*{\alpha\beta\gamma}\lambda*\alpha\lambda_\beta\lambda_\gamma \cdot e^{3s},\quad
  \frac{\partial \mathcal V_S}{\partial s}=3\mathcal V_S,\quad
  \frac{\partial \mathcal V_S}{\partial \lambda_\alpha}=\tfrac12,\kappa'*{\alpha\beta\gamma}\lambda*\beta\lambda_\gamma\cdot e^{3s}
  ]
  with (\kappa'*{\alpha\beta\gamma}=\kappa*{ijk}T_{i\alpha}T_{j\beta}T_{k\gamma}).

This outperforms raw (t), ambient 214‑vectors, and pure ray‑direction for GA stability.

---

## How it compares to your current options

* **Raw (t)**: minimal dimension but boundary handling is hard. Near facets, small (\Delta t) can cause large (\Delta \mathcal V).
* **log (t)**: good scale handling, but still needs inequality handling.
* **Ray direction from tip**: safe but biased toward the tip and restricts reachable region on each ray.
* **Ambient 214**: redundant, hard to mutate meaningfully; keep for constraints or diagnostics, not as genes.
* **Target (\tau)**: physically meaningful but requires a solve to map back to (t). Use it as a penalty/regularizer, not as the coordinate system.

---

## Make it numerically smooth: whitening near the optimum

At your McAllister point (t_\star=[49.6,38.9,35.0,26.2]), build a **local preconditioner** that makes curvature isotropic:

1. Compute Hessian (H_{ij}=\kappa_{ijk}t_\star^k) in your 4‑divisor basis.
2. Cholesky (H=L L^\top) (it is positive definite in the interior).
3. Define **whitened shape coordinates** (\tilde u = L^\top (t/|t|-t_\star/|t_\star|)) or, in the cone‑generator form, whiten the (\lambda) direction space: (\tilde z = W \cdot \mathrm{logit}(\lambda)) with (W) from the SVD of the Jacobian (\partial t/\partial z) at (z_\star).

In practice: keep the cone‑generator parameterization, but mutate in the **whitened latent** (\tilde z). This equalizes sensitivities across genes and yields very smooth fitness responses to small mutations.

---

## Where to use (g_s)

* If fluxes are fixed, (g_s) is fixed and just rescales the fitness by a constant factor (g_s^{-3/2}). Do not include (g_s) as a gene unless you are simultaneously varying flux quanta and solving the CS/dilaton F-terms.
* If you do vary fluxes: track (\mathcal V_E=\mathcal V_S g_s^{-3/2}) but keep the geometry parameterization independent. This keeps the search disentangled.

---

## Experiments to run quickly

All experiments assume you load the exact FRST and intersection tensor for 4-214-647.

### E1 - Local smoothness

* Around (t_\star), evaluate (\mathcal V_E) along coordinate directions in each parameterization:

  * raw (t)
  * log (t)
  * your ray direction
  * cone‑generator log‑coords ( (s,z) )
* Probe step sizes (\varepsilon\in{10^{-2},10^{-1},1,10}).
* Record monotonicity, curvature, and whether any move leaves the cone. Expect the cone‑generator log‑coords to be the most monotone and boundary‑safe.

### E2 - Condition numbers

* At (t_\star), compute Jacobians of (\mathcal V_S) w.r.t. each coordinate system and the spectrum of the corresponding local quadratic model. Lower condition numbers are better for GA stability. Expect whitening to normalize this.

### E3 - Basin mapping toward (t_\star)

* Sample 1000 feasible points. Do a simple second‑order line search toward (\mathcal V_E(t_\star)) (or whatever metric you want, e.g. MSE on selected (\tau_A)). Compare convergence rates across parameterizations. Expect the cone‑generator plus whitening to have the highest hit rate.

### E4 - GA tournament

* Run your GA with identical operators and budgets on:

  1. raw (t)
  2. log (t)
  3. ray direction
  4. cone‑generator log‑coords
  5. cone‑generator log‑coords + whitening
* Fitness: e.g. ( |\mathcal V_E-4711.83| ) plus penalties (see below). Report success rate and generations to hit tolerance.

---

## Practical details you’ll need

### Get cone rays or inequalities

* If CYTools gives the Kähler cone as (A t \ge 0), compute its V‑representation once offline with a polyhedral library and cache the rays (r_\alpha). If it already exposes rays, just collect those in (T).
* Keep only **extreme rays**. If there are many, you can use a reduced set spanning the cone with margin.

### Keep a safety margin to avoid steep walls

* Use (\lambda_\alpha=\lambda_{\min}+\mathrm{softplus}(z_\alpha)) with (\lambda_{\min}\approx 0.05) to stay away from facets. This prevents large curvature spikes.

### Penalize physics constraints in (\tau)-space, not in (t)-space

* Pick a small set (\mathcal S) of toric divisors that matter for control, e.g. rigid ones and those with large GV support. Penalize
  [
  \Phi_{\tau} = \sum_{A\in\mathcal S}\mathrm{softplus}(\tau_{\min}-\tau_A(t))
  ]
  using **string frame** (\tau). This is cheap and smooth. It also works naturally with the cone‑generator coordinates.

### Precompute tensors in the new basis

* Build (\kappa'_{\alpha\beta\gamma}) once from (T). Evaluations then boil down to a tiny 3‑tensor contraction. Caching gives a big speedup for GA inner loops.

---

## Minimal code scaffolding

```python
# Given: kappa[i,j,k] in your divisor basis, and Kähler cone rays r[:, alpha]
import numpy as np

def build_kappa_prime(kappa, T):  # T shape: (h11, m)
    h11, m = T.shape[0], T.shape[1]
    kp = np.zeros((m, m, m))
    for a in range(m):
        for b in range(m):
            for c in range(m):
                # contract: kappa_ijk T_{i a} T_{j b} T_{k c}
                kp[a,b,c] = np.einsum('ijk,i,j,k->', kappa, T[:,a], T[:,b], T[:,c])
    return kp

def t_from_sz(s, z, T, lam_min=0.05):
    lam = lam_min + np.log1p(np.exp(z))  # softplus
    t = np.exp(s) * (T @ lam)
    return t, lam

def V_string_from_sz(s, z, kappa_prime, T, lam_min=0.05):
    lam = lam_min + np.log1p(np.exp(z))
    cubic = np.einsum('abc,a,b,c->', kappa_prime, lam, lam, lam)
    return np.exp(3*s) * (cubic / 6.0)

def tau_basis_from_t(kappa, t):
    # tau_i = 1/2 kappa_{ijk} t^j t^k
    return 0.5 * np.einsum('ijk,j,k->i', kappa, t, t)

def V_einstein(V_string, g_s):
    return V_string * g_s**(-1.5)
```

Whitening near (t_\star):

```python
def hessian_at_t(kappa, t):
    # H_{ij} = kappa_{ijk} t^k
    return np.einsum('ijk,k->ij', kappa, t)

def whiten_transform(kappa, t_star):
    H = hessian_at_t(kappa, t_star)
    L = np.linalg.cholesky(H)   # H = L L^T
    W = np.linalg.inv(L.T)      # so that W^T H W = I
    return W  # use to precondition mutations in z
```

Mutation in whitened space:

* Draw (\delta \tilde z \sim \mathcal N(0,\sigma^2 I)), set (\delta z = W,\delta \tilde z).

---

## Answers to your study questions

* **Q1 smoothness**: In the cone‑generator log‑coords, both (\mathcal V) and (\tau) are low‑degree polynomials times (e^{3s}); there are no discontinuities as long as you stay in a fixed FRST. Cliffs only appear near facets; the softplus margin avoids them.
* **Q2 best coordinate system**: Cone‑generator log‑coords with Hessian whitening give the most uniform, moderate gradient. log‑(t) is second best if you add a barrier penalty.
* **Q3 better parameterization**: Yes - the proposed scale‑shape cone‑generator parameterization, optionally whitened. It is tailored to the cubic homogeneity and the cone geometry.
* **Q4 McAllister as ground truth**: Use your gold point (t_\star) to set (W) once and compare convergence and robustness across parameterizations as in E4.

---

## When to consider the 214‑dim data

* Use it to **define penalties** (e.g., keep selected (\tau_A\ge\tau_{\min})), or to **warm start** via linearized updates (C,\delta t \approx \delta\tau) with (C_{A i}=\partial \tau_A/\partial t^i=\kappa_{Aik}t^k).
* Do not expose all 214 as GA genes.

---

## Final checklist to switch your GA

1. Build (T) from cone rays, compute (\kappa').
2. Use genes ((s,z)) with softplus margin for (\lambda).
3. Add optional whitening using (H(t_\star)).
4. Fitness in Einstein frame if you want to compare to the paper; otherwise work in string frame and convert at reporting time.
5. Add (\tau)-based soft penalties.
6. Run the tournament in E4 and keep the winner.
