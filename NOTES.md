# String Theory Landscape Explorer - Technical Notes

## Overview

This project uses a genetic algorithm to search the string theory landscape for Calabi-Yau compactifications that could reproduce Standard Model physics. The search space is the Kreuzer-Skarke database of ~473 million 4D reflexive polytopes.

## Physics Background

### The Goal
Find Calabi-Yau manifolds whose compactification gives:
- 3 generations of fermions (quarks + leptons)
- SU(3) × SU(2) × U(1) gauge group (Standard Model)
- Correct particle masses and mixing angles
- Small cosmological constant

### String Theory Dimensions
String theory requires 10D (Type IIA/IIB, Heterotic) or 11D (M-theory). To get our 4D spacetime:
- **10D = 4D spacetime + 6D Calabi-Yau (CY3)**
- The "4D polytopes" in Kreuzer-Skarke define 6D CY3 manifolds via toric geometry

### Key Constraints

**3-Generation Constraint:**
- Euler characteristic χ = 2(h11 - h21)
- For 3 generations: |h11 - h21| = 3
- This filters 473M polytopes down to a much smaller set

**Known Working Examples:**
- (h11, h21) = (1, 4): Gives exactly 3 generations, can break E6 → Standard Model
- (h11, h21) = (1, 1): Minimal case from 24-cell construction
- Small Hodge numbers are generally preferred for model building

### Hodge Numbers
- h11: Number of Kähler moduli (shape deformations)
- h21: Number of complex structure moduli
- These determine many physical properties of the compactification

## Physics Tools

### CYTools (Cornell/McAllister Group)
Primary tool for polytope analysis and CY computations.

**Repository:** https://github.com/LiamMcAllisterGroup/cytools

**Key Features:**
- `Polytope(vertices)` - Create polytope from vertices
- `p.triangulate()` - Get triangulation
- `cy.h11()`, `cy.h21()`, `cy.chi()` - Hodge numbers
- `cy.intersection_numbers()` - Triple intersection numbers κᵢⱼₖ
- `cy.compute_cy_volume(t)` - CY volume from Kähler moduli
- `cy.compute_divisor_volumes(t)` - 4-cycle volumes (for gauge couplings)
- `cy.second_chern_class()` - c₂ for anomaly cancellation
- `cy.toric_kahler_cone()` - Valid Kähler moduli range

**Dependencies:**
- pplpy, python-flint, pypalp, cygv, ortools

### cymyc (JAX-based CY Metrics)
Numerical differential geometry on CY manifolds.

**Repository:** https://github.com/Justin-Tan/cymyc

**Key Features:**
- Neural network approximation of Ricci-flat metrics
- Curvature computations (Riemann, Ricci, scalar)
- Yukawa coupling calculations
- Complex structure moduli space investigations

**Dependencies:**
- JAX, equinox, optax, sympy

### PALP (Polytope Analysis Lattice Package)
Classic tool for toric geometry computations.

**Repository:** https://gitlab.com/stringstuwien/PALP

**Used for:**
- Hodge number computation
- Triangulation algorithms
- Point counting

## Data Pipeline

### 1. Download (download_all_polytopes.py)
- Source: HuggingFace `calabi-yau-data/polytopes-4d`
- Format: 32 parquet files, ~15.8 GB compressed
- Contains: vertices, h11, h12 (=h21), vertex_count, facet_count, etc.
- Resume-capable: validates existing files, skips completed downloads

### 2. Filter (filter_three_gen.py)
- Filters to |h11 - h21| = 3 (3-generation candidates)
- Sorts by h11 + h21 (smaller = simpler manifolds)
- Outputs: `polytopes_three_gen.json`
- Expected reduction: 473M → maybe 1-10M polytopes

### 3. GA Search (real_physics binary)
- Loads filtered polytopes
- Runs genetic algorithm with CYTools/cymyc physics evaluation
- Persists state for resume capability

## Physics Computations (physics_bridge.py)

### Architecture
```
real_physics.rs (Rust binary)
       │
       ▼
physics.rs (PyO3 bridge)
       │
       ▼
physics_bridge.py
       │
       ├─► CYTools: Polytope analysis, volumes, intersection numbers
       ├─► cymyc: Numerical metrics, curvature (when trained)
       └─► KKLT: Moduli stabilization, cosmological constant
```

### Gauge Coupling Computation
In Type IIB with D7-branes:
```
1/g_a² = Vol(Σ_a) / g_s
```
Where Σ_a is the 4-cycle wrapped by the D7-brane stack.

At tree level:
- `α_a = g_s / (4π Vol_a)` at string scale
- Run to Z scale using 1-loop SM β-functions: b₁=41/10, b₂=-19/6, b₃=-7

### KKLT Moduli Stabilization
Scalar potential:
```
V = e^K [ K^{ij̄} D_i W D_j̄ W̄ - 3|W|² ] + V_uplift
```
- W = W_flux + W_np (flux + non-perturbative superpotential)
- W_flux = ∫ G₃ ∧ Ω (Gukov-Vafa-Witten)
- V_uplift = D/V^{4/3} (anti-D3 brane at warped throat)

### Tadpole Constraint
```
N_flux + N_D3 ≤ χ(CY)/24
```

## Search Architecture

### The Problem
- 12M+ candidate polytopes (filtered for 3 generations)
- Each physics evaluation takes ~100ms (CYTools + KKLT)
- Random search is useless at this scale
- Need to learn which geometric features predict good physics

### Multi-Level Learning System

The search is **iterative** - we learn as we go:

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION LOOP                          │
│                                                             │
│   GA selects polytope → Run physics → Record result         │
│         ▲                                    │              │
│         │                                    ▼              │
│   ┌─────┴─────┐                    ┌─────────────────┐     │
│   │  Ranker   │◄───── train ──────│ Evaluation Log  │     │
│   │  Model    │                    │ (geometry,fit)  │     │
│   └─────┬─────┘                    └─────────────────┘     │
│         │                                    │              │
│         │ score                              │ cluster      │
│         ▼                                    ▼              │
│   ┌───────────┐                    ┌─────────────────┐     │
│   │ Candidate │                    │ Feature Space   │     │
│   │ Ranking   │                    │ Clusters        │     │
│   └───────────┘                    └─────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Evaluation Recording

Every physics evaluation records:
```json
{
  "polytope_id": 12345,
  "geometry_features": [h11, h21, vertex_stats...],
  "physics_result": {fitness, alpha_em, alpha_s, ...},
  "timestamp": "..."
}
```

Persisted to `evaluations.jsonl` - accumulates across runs.

### Layer 2: Feature Clustering

Cluster polytopes in **geometry feature space** (not just h11/h21):
- Vertex coordinate statistics
- Shape characteristics (spread, aspect ratio)
- Combinatorial properties

Track per-cluster:
- Number of evaluations
- Average/best fitness
- UCB score for exploration/exploitation balance

### Layer 3: Learned Ranker

After N evaluations (e.g., 1000+), train a simple model:
- Input: geometry features (cheap to compute)
- Output: predicted fitness
- Model: small feedforward NN or gradient boosting

Use ranker to:
- Score unevaluated polytopes
- Bias selection toward predicted-good candidates
- Still explore (don't trust model completely)

### Layer 4: Active Learning

Select polytopes that would most improve the model:
- High uncertainty (model unsure)
- Near decision boundary
- In under-explored clusters

Balance: exploit (high predicted score) vs explore (high uncertainty)

### Feedback Loop

```
Run N generations of GA
        │
        ▼
Train/update ranker on all evaluations
        │
        ▼
Score candidate polytopes with ranker
        │
        ▼
Update cluster statistics
        │
        ▼
Adjust selection probabilities
        │
        └──► Repeat
```

The ranker gets better over time → smarter selection → faster convergence.

## Genetic Algorithm Design

### Genome
Each individual encodes:
- `polytope_id` - Index into polytope database
- `kahler_moduli` - Array of h11 Kähler parameters
- `complex_moduli` - Array of h21 complex structure parameters
- `flux_f`, `flux_h` - Integer flux quanta
- `g_s` - String coupling
- `n_antiD3` - Number of anti-D3 branes for uplift

### Fitness Components
1. **Generation score**: |N_gen - 3| (should be 0)
2. **Gauge coupling scores**: Log-error from observed α_em, α_s, sin²θ_W
3. **Cosmological constant**: Log-error from observed Λ
4. **Tadpole**: Penalty if constraint violated

### Polytope Feature Vectors

**Geometric Features (cheap - from vertices only):**
- h11, h21, Euler characteristic
- Vertex count, coordinate statistics (mean, std, min, max)
- Shape characteristics (aspect ratio, spread, centroid distance)
- Combinatorial (zero count, negative count, coord sums)

**Physics Features (expensive - requires evaluation):**
- α_em error, α_s error, sin²θ_W error
- N_gen error, Λ error
- CY volume, flux tadpole
- Overall fitness

### Selection Strategy

1. **Cluster-weighted**: UCB score per cluster
   ```
   weight = avg_fitness + c * sqrt(log(total_evals) / cluster_evals)
   ```

2. **Ranker-guided**: Bias toward high predicted fitness
   ```
   P(select) ∝ exp(ranker_score / temperature)
   ```

3. **Hot polytope reuse**: Polytopes with good offspring get resampled

4. **Exploration floor**: Always sample some random polytopes

## Infrastructure

### Server Setup (Ansible)
- LXC container at 10.5.7.33
- 32 cores, 8GB RAM
- ZFS mount at /data/polytopes (6.1 TB)

**Installed Tools:**
- PALP from GitLab
- CYTools from GitHub (LiamMcAllisterGroup)
- cymyc from GitHub (Justin-Tan)
- Rust toolchain with Python 3.10

### File Locations on Server
```
/root/string_theory/           # Project root
/root/palp_source/             # PALP installation
/root/cytools_source/          # CYTools source
/root/cymyc_source/            # cymyc source
/data/polytopes/parquet/       # Raw parquet files
/data/polytopes/polytopes_three_gen.json  # Filtered data
```

### Running the GA
```bash
# On server
cd /root/string_theory
source venv/bin/activate
./target/release/real_physics
```

### Monitoring
```bash
# Check GA progress
ls -la results/
cat results/best_*.json | jq .fitness

# Check cluster state
cat cluster_state.json | jq '.clusters | length'
```

## Key Files

| File | Purpose |
|------|---------|
| `download_all_polytopes.py` | Download parquet files from HuggingFace |
| `filter_three_gen.py` | Filter to 3-generation candidates |
| `physics_bridge.py` | CYTools + cymyc physics computations |
| `src/bin/real_physics.rs` | Main GA binary |
| `src/real_genetic.rs` | GA implementation with clustering |
| `src/physics.rs` | Rust-Python bridge via PyO3 |
| `ansible/playbook.yml` | Server setup automation |

## References

### Data Sources
- Kreuzer-Skarke database: https://huggingface.co/datasets/calabi-yau-data/polytopes-4d

### Physics Tools
- PALP: https://gitlab.com/stringstuwien/PALP
- CYTools: https://github.com/LiamMcAllisterGroup/cytools
- cymyc: https://github.com/Justin-Tan/cymyc

### Papers
- Kreuzer & Skarke, "Complete classification of reflexive polyhedra in four dimensions" [arXiv:hep-th/0002240](https://arxiv.org/abs/hep-th/0002240)
- Braun et al., "A three-generation Calabi-Yau manifold with small Hodge numbers" [arXiv:0909.3947](https://arxiv.org/abs/0909.3947)
- He, "The Calabi-Yau Landscape" [arXiv:1812.02893](https://arxiv.org/abs/1812.02893)
- cymyc paper: [arXiv:2410.19728](https://arxiv.org/abs/2410.19728)

## Progress

### Completed
- [x] Download pipeline for polytope data
- [x] Filter script for 3-generation candidates
- [x] Update real_physics to use polytopes_three_gen.json
- [x] Implement cluster-based adaptive selection with UCB
- [x] Add mutation pattern tracking
- [x] Persist cluster state to disk
- [x] Add polytope feature vectors (embeddings)
- [x] Rewrite physics_bridge.py to use CYTools + cymyc
- [x] Update ansible to install CYTools and cymyc

### TODO

**Layer 1: Evaluation Recording**
- [ ] Persist every evaluation to `evaluations.jsonl`
- [ ] Include full geometry features + physics results
- [ ] Load history on startup to resume learning

**Layer 2: Feature Clustering**
- [ ] Cluster by full geometry features (not just h11/h21)
- [ ] Use k-means or HDBSCAN in feature space
- [ ] Track cluster statistics (evals, avg fitness, best fitness)
- [ ] UCB-based cluster selection

**Layer 3: Learned Ranker**
- [ ] Train simple NN: geometry → predicted fitness
- [ ] Trigger training after N evaluations (e.g., 1000)
- [ ] Use ranker scores to bias polytope selection
- [ ] Retrain periodically as more data accumulates

**Layer 4: Active Learning**
- [ ] Track model uncertainty per polytope
- [ ] Balance exploit (high score) vs explore (high uncertainty)
- [ ] Prioritize under-explored clusters

**Infrastructure**
- [ ] Add `-v` verbose logging (DONE)
- [ ] Double Ctrl+C force quit (DONE)
- [ ] Create visualization for cluster/ranker data
- [ ] Add monitoring dashboard
- [ ] Integrate cymyc trained metric models
