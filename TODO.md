# String Theory Landscape Explorer - TODO

## High Priority: GA Parameterization Improvements

The recent research (see `resources/RESEARCH_GA_PARAMETERIZATIONS.md`) provides clear guidance on optimal parameterization.

### Scale-Shape Cone-Generator Implementation
- [ ] Extract Kähler cone rays from CYTools for each polytope
- [ ] Implement `build_kappa_prime(kappa, T)` to transform intersection tensor to cone basis
- [ ] Implement `t_from_sz(s, z, T, lam_min)` - convert (s, z) genes to Kähler moduli
- [ ] Add softplus transform for z genes: `λ = λ_min + softplus(z)` to guarantee cone interior
- [ ] Separate scale gene (s) from shape genes (z) in genome structure
- [ ] Update `physics_bridge.py` to support new parameterization alongside existing modes

### Hessian Whitening (Optional Enhancement)
- [ ] Compute Hessian `H_ij = κ_ijk t^k` at reference points
- [ ] Implement Cholesky-based whitening transform
- [ ] Apply whitening to mutation operator for equalized sensitivities

### Validation with McAllister Ground Truth
- [ ] Run parameterization tournament on polytope 4-214-647
  - Raw t (current)
  - Log t
  - Ray direction (current GA mode)
  - Cone-generator log-coords (new)
  - Cone-generator + whitening (new)
- [ ] Measure: generations to converge, success rate, final fitness
- [ ] Document results and select best parameterization

## High Priority: Web Dashboard / Playground

From plan file (`~/.claude/plans/cheeky-mapping-boot.md`):

### Schema Migration for Content-Addressable Evaluations
- [ ] Create `migrations/008_content_addressable.sql`
  - Add `input_hash TEXT` (SHA256 of inputs for dedup/caching)
  - Add `model_version TEXT` (invalidates cache on physics code changes)
  - Add `source TEXT DEFAULT 'ga'` ('ga', 'playground')
  - Add `label TEXT` (optional human-readable name)
  - Add `vertices_json TEXT` (for non-DB polytopes)
  - Add `h11 INTEGER`, `h21 INTEGER` columns
  - Create unique index on `input_hash`

### Model Versioning
- [ ] Add `PHYSICS_MODEL_VERSION = "1.0.0"` to `physics_bridge.py`
- [ ] Add `pub const PHYSICS_MODEL_VERSION: &str = "1.0.0"` to `src/constants.rs`
- [ ] Include version in hash computation for automatic cache invalidation

### Playground Page
- [ ] Create `web/src/routes/playground.tsx`
  - Polytope source selector (DB vs external/custom vertices)
  - Kähler moduli JSON input
  - Complex moduli JSON input
  - Flux vectors input
  - g_s input
  - Predefined configs (McAllister 4-214-647, quintic, etc.)
- [ ] Create `web/src/server/playground.ts`
  - `runEvaluation(params)` - spawn Rust binary with `--json --save`
  - Return evaluation result from SQLite
- [ ] Add `--json` output mode to `src/bin/evaluate.rs`

### Shared Components
- [ ] Create `web/src/components/EvaluationCard.tsx`
  - Unified display for all evaluation types
  - Physics values, fitness, moduli summary
  - Source badge (ga/playground/reference)
- [ ] Update existing pages to use shared component

## Medium Priority: Decomposed Fitness

Explore breaking observables into constituent components (see `NOTES.md` section "Decomposed Fitness"):

### Physics Decomposition Analysis
- [ ] Compute Jacobian ∂(observable)/∂(parameter) at McAllister optimum
- [ ] SVD analysis to identify principal component directions
- [ ] Check if components align with interpretable physics
- [ ] Measure gradient smoothness along components vs original axes

### Multi-Objective GA
- [ ] Track separate objectives:
  - `gauge_magnitude_error`
  - `gauge_ratio_error`
  - `weinberg_error`
  - `cc_log_error`
- [ ] Implement Pareto-front selection (NSGA-II style)
- [ ] Compare convergence vs single-objective approach

### Soft/Banded Targets
- [ ] Implement target ranges instead of exact values
- [ ] `fitness = 0 if value ∈ [min, max] else penalty`
- [ ] Test if finding valid regions is easier than exact points

## Medium Priority: Unit Tests

### Physics Bridge Validation
- [ ] Create `tests/test_physics_bridge.py`
- [ ] Create `tests/fixtures/mcallister_4_214_647.json` with gold standard data:
  - vertices, Kähler moduli, fluxes, g_s
  - expected: W₀ = 2.30×10⁻⁹⁰, CY volume = 4711.83
- [ ] Verify physics_bridge reproduces McAllister results within tolerance
- [ ] Add tests for GA mode vs fixed mode

## Lower Priority: Infrastructure

### Outlier Score Pipeline
- [ ] Create `PopulationStats` class with Welford's algorithm
- [ ] Save/load running stats to `population_stats.json`
- [ ] Compute outlier scores incrementally as polytopes are processed
- [ ] Add `--recompute-outliers` flag for batch recomputation

### Visualization
- [ ] Genealogy trees with ancestry tracking
- [ ] 2D projections of 4D polytope vertices
- [ ] Real-time fitness plot over generations
- [ ] Population diversity metrics

### Parallelization
- [ ] Island model with migration between workers
- [ ] Shared hall of fame across nodes
- [ ] Coordinate multiple servers

## Completed

### Physics Bridge Dual-Mode Support
- [x] Implement `_get_kahler_moduli()` dispatcher
- [x] GA mode: raytracing from cone tip along genome direction
- [x] Fixed mode: use exact Kähler values, project into cone if needed
- [x] Test both modes with quintic polytope (`tools/test_physics_bridge_modes.py`)

### McAllister Volume Reproduction
- [x] Create `tools/reproduce_mcallister_volume.py`
- [x] Understand frame conversion: V_E = V_S × g_s^{-3/2}
- [x] Validate CY volume = 4711.83 is achievable

### GA Parameterization Research
- [x] Document parameterization options in `resources/RESEARCH_GA_PARAMETERIZATIONS.md`
- [x] Get deep research answer: scale-shape cone-generator with log-coords
- [x] Document Hessian whitening approach

### Decomposed Fitness Exploration
- [x] Add "Decomposed Fitness: Multi-Dimensional Gradient Descent" section to NOTES.md
- [x] Document physics decomposition examples (gauge couplings, Weinberg angle, CC)
- [x] Outline implementation approaches (multi-objective, hierarchical, dynamic weighting)

## Reference Documents

- `NOTES.md` - Technical notes, physics background, decomposed fitness ideas
- `CLAUDE.md` - Project setup, build instructions, architecture
- `resources/RESEARCH_GA_PARAMETERIZATIONS.md` - Optimal parameterization research
- `resources/MCALLISTER_SMALL_CC_DETAILS.md` - McAllister paper analysis
- `~/.claude/plans/cheeky-mapping-boot.md` - Playground/evaluation system plan
