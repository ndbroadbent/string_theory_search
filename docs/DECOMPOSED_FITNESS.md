Decomposed fitness is the right move. Treat observables as functions of a small set of components, optimize those components directly, and let a learned controller suggest parameter-specific tweaks.

---

## 1) Components and exact sensitivities

Use log-coordinates for smoothness.

**Gauge couplings at the string scale**
[
\alpha_a=\frac{g_s}{4\pi,\tau_a(t)},\quad
\log\alpha_a=\log g_s-\log \tau_a-\log(4\pi).
]
With ( \tau_a(t)=\tfrac12,\kappa_{a jk} t^j t^k ),
[
\frac{\partial \log\alpha_a}{\partial t^i}= -\frac{\kappa_{a i k} t^k}{\tau_a},\qquad
\frac{\partial \log\alpha_a}{\partial \log g_s}=1.
]

Define components that disentangle magnitude vs ratios:

* Magnitude: (M \equiv \frac{1}{n}\sum_a \log\alpha_a)  (controlled by (\log g_s) and mean (\log\tau_a)).
* Ratios: (R_a \equiv \log\alpha_a - \frac{1}{n}\sum_b \log\alpha_b = -\big(\log\tau_a - \frac{1}{n}\sum_b \log\tau_b\big))  (independent of (g_s)).

**Weinberg angle** at high scale using SU(2) and U(1):
[
\sin^2\theta_W=\frac{\alpha_1}{\alpha_1+\alpha_2},\quad
W \equiv \operatorname{logit}(\sin^2\theta_W)=\log\alpha_1-\log\alpha_2,
]
so (W) depends only on the difference of (\log\tau)s, not on (g_s).

**Volume and CC (schematic KKLT)**
[
\mathcal V_S=\tfrac16,\kappa_{ijk} t^i t^j t^k,\quad
\frac{\partial \mathcal V_S}{\partial t^i}=\tau_i=\tfrac12,\kappa_{i jk} t^j t^k.
]
Work in logs: (V\equiv\log \mathcal V_S), then (\partial V/\partial t^i=\tau_i/\mathcal V_S).
For a decomposed CC loss, use components

* (C_{W0}\equiv \log|W_0|) (flux controlled),
* (C_V\equiv \log \mathcal V_S) (Kähler controlled),
* (C_D\equiv \log D) (uplift controlled),
  and compare to the KKLT scaling (\Lambda \sim D,\mathcal V^{-4/3}-3|W_0|^2 e^K \mathcal V^{-2}). This lets you steer with orthogonal levers.

---

## 2) Parameterization to pair with components

Use the cone-safe scale-shape log parameterization from earlier:

* (t = e^{s},T,\lambda(z)), with (\lambda_\alpha=\lambda_{\min}+\text{softplus}(z_\alpha)) and rays (T=[r_1,\dots,r_m]).
* Optional simplex normalization on (\lambda) to separate scale (s) from shape.
* Mutate in a whitened latent for smooth, well-conditioned steps.

This gives clean control knobs:

* (\log g_s) changes all (\log\alpha_a) by +1 each.
* Shape latents (z) change (\log\tau_a) differences, hence ratios and (W).
* Scale (s) mostly changes (\mathcal V) and thus the CC component.

---

## 3) Multi-objective objectives in component space

Replace one scalar fitness with:

* (L_{\text{mag}} = \big|\ M - M_{\text{tgt}}\ \big|)  (or a banded/hinge loss)
* (L_{\text{rat}} = \sum_a \big| R_a - R_{a,\text{tgt}} \big|)
* (L_W = \big|\ W - W_{\text{tgt}}\ \big|)
* (L_{\Lambda} = \big|\ \log|\Lambda| - \log|\Lambda|*{\text{tgt}}\ \big|) or a decomposed loss in ((C*{W0}, C_V, C_D))

Run NSGA-II or MOEA/D so partial progress is preserved on the Pareto front. Optionally stage it:

1. ratios and (W),
2. magnitudes via (\log g_s) and mild shape adjustments,
3. CC via (C_{W0}, C_V, C_D).

Use soft target bands to define “solved” components and downweight them dynamically.

---

## 4) Component-directed mutation operators

Add small, interpretable step types:

* G-step (gauge magnitude): (\delta\log g_s=\eta). Leaves ratios and (W) unchanged at tree level.
* R-step (ratio-preserving volume): choose (\delta t) that satisfies (\sum_a w_a,\delta \log\tau_a=0) with (\sum_i \tau_i,\delta t^i=0) to keep (V) nearly fixed while moving (R_a) or (W). Solve the small linear system using (\partial \log\tau_a/\partial t^i=(\kappa_{a i k}t^k)/\tau_a).
* V-step (volume only): move along the eigenvector of (H_{ij}=\kappa_{ijk}t^k) aligned with (t) to change (V) with minimal effect on ratios.
* Λ-step: if you allow flux changes, propose (\delta \log|W_0|) or (\delta \log D); otherwise favor V-step toward the side that reduces the current CC component.

These operators align with components, so one mutation mostly moves one objective.

---

## 5) Learn a tweak policy from GA data

You can train a small neural controller that, given current component errors, proposes a parameter step that helps with probability > 0.75.

Data you already log per evaluation:
[
(x_t,\ \text{components}*t,\ \Delta x_t,\ \text{components}*{t+1})
]
Derive labels like (\Delta L_k<0) for each component (k).

Two practical learners:

**A. Local Jacobian learner**

* Train regressors for each component (k) to predict (\Delta c_k \approx J_k(x),\Delta x).
* Use them to pick (\Delta x) that most reduces (L_k) subject to trust-region and cone constraints.
* Start with analytic seeds for (J) from the formulas above and learn residuals.

**B. Operator chooser (mixture-of-experts)**

* Treat each step type {G, R, V, Λ, …} as an expert with parameterized step size.
* Train a classifier (p(\text{expert}\mid \text{component errors}, \text{features})) with the target “which expert reduced (L_k) last time”.
* Execute the most probable expert, sample step size from a small learned regressor.

Add monotonic priors where physics is exact:

* (\partial \log\alpha_a/\partial \log g_s = 1) (embed as a hard feature or a constrained layer).
* Ratio changes are insensitive to (\log g_s) at tree level.

Use an ensemble for uncertainty; when the controller is uncertain, fall back to safe analytic operators.

---

## 6) Diagnostics to verify the idea fast

* Compute at the McAllister point (t_\star):

  * SVD of the Jacobian of ((M,{R_a},W,V)) w.r.t. ((\log g_s, s, z)).
  * Expect one singular vector almost aligned with (\log g_s) for (M), one or two aligned with shape for ratios and (W), and one aligned with scale for (V).
* Plot one-dimensional slices of each component along each operator direction. You should see smooth, near-monotone curves inside the FRST.

---

## 7) Practical loss shapes

Use banded or hinge losses to turn targets into regions:

* (L_{\text{band}}(y)=\max(0, |y-y_{\text{tgt}}|-b)).
* For Λ, use log error with an upper cap so progress remains measurable when values are tiny.

---

## 8) Notes and caveats

* Keep FRST fixed while doing component analysis. Crossing a wall changes (\kappa) and derivatives.
* Stay away from the cone boundary with a softplus margin in (\lambda).
* RG running and thresholds shift the mapping between string-scale and low-scale quantities. Start with tree-level components to structure the search, then add learned threshold corrections as residuals.

---

## Minimal sketch to drop in

```python
# Components
def components(t, gs, stacks):
    # stacks: list of brane divisors A with tau_A(t) = 0.5 kappa_{A,jk} t^j t^k
    taus = np.array([0.5*np.einsum('jk,j,k->', kA, t, t) for kA in stacks])
    log_alpha = np.log(gs) - np.log(taus) - np.log(4*np.pi)
    M = log_alpha.mean()
    R = log_alpha - M
    V = np.log( (1.0/6.0)*np.einsum('ijk,i,j,k->', kappa, t, t, t) )
    return dict(M=M, R=R, V=V)

# G-step: adjust magnitude only
def g_step(gs, eta): return gs*np.exp(eta)

# R-step: solve small linear system for delta t to change a chosen ratio while approx preserving V
# Use Jacobians d log tau_a / d t^i and tau_i for dV/dt^i, then least squares with constraints.
```

---

**Bottom line**: yes, decompose. Put the GA in component space, use cone-safe log coordinates, add component-aligned operators, and learn a small controller that picks which operator to apply next. You will get smoother gradients, clearer partial progress signals, and far better sample efficiency than a single scalar fitness.
