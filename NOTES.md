# String Theory Landscape Explorer - Technical Notes

## Overview

This project uses a **meta-genetic algorithm** to search the string theory landscape for Calabi-Yau compactifications that could reproduce Standard Model physics. The search space is the Kreuzer-Skarke database filtered to 12.2 million three-generation candidates.

**Key insight**: Instead of manually tuning GA hyperparameters, we evolve the search strategies themselves. The meta-GA discovers which geometric features predict good physics, optimal mutation rates, and effective polytope selection strategies.

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

### 3. Meta-GA Search (search binary)
- Loads filtered polytopes
- Runs as distributed worker in meta-GA system
- Acquires algorithm from database, runs trials, records results
- Multiple workers coordinate via SQLite WAL mode

## Physics Computations (physics_bridge.py)

### Architecture
```
search binary (Rust, src/bin/search/)
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
| `src/bin/search/` | Meta-GA worker binary (multi-file) |
| `src/searcher.rs` | Inner GA implementation |
| `src/meta_ga.rs` | Meta-evolution functions |
| `src/db.rs` | SQLite persistence layer |
| `src/physics.rs` | Rust-Python bridge via PyO3 |
| `migrations/*.sql` | Database schema |
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

## McAllister Group: Small Cosmological Constant Results

### Key Achievement
The Cornell group (Demirtas, Kim, McAllister, Moritz, Rios-Tascon) achieved **|Λ| < 10⁻¹²³** in Planck units - matching/beating the observed cosmological constant.

**Papers:**
- [Small Cosmological Constants in String Theory](https://arxiv.org/abs/2107.09064) (JHEP 2021)
- [Vacua with Small Flux Superpotential](https://arxiv.org/abs/1912.10047) (PRL 2020)
- [Conifold Vacua with Small Flux Superpotential](https://arxiv.org/abs/2009.03312)
- [Exponentially Small Cosmological Constant](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.011602) (PRL)

### How They Did It (DKMM Mechanism)

The key insight: **exponentially small W₀ (flux superpotential)** is achievable through careful flux selection, not random search.

**Two-step construction:**
1. Find quantized fluxes where F-terms and superpotential **vanish perturbatively** along a flat direction (ignoring non-perturbative corrections)
2. Restore non-perturbative corrections → flat direction is lifted → W₀ becomes exponentially small

**Requirements for this to work:**
- **Large h²¹** (many complex structure moduli) - their example: (h¹¹, h²¹) = (2, 272)
- **Large complex structure limit** - working far from singular points
- **Weak string coupling** (g_s << 1)
- **Specific flux quantization** - not random, but satisfying algebraic conditions

### Why Random Search Won't Find These

The problem is NP-complete (subset sum variant):
- Need flux integers that sum to give exponentially small W₀
- Random sampling has essentially zero probability of hitting this
- The 10⁻¹²³ result requires **number-theoretic structure** in the flux choices

### What Properties They Selected For

**Geometric criteria:**
1. **Large h²¹** - More complex structure moduli = more "knobs" to tune
2. **Favorable polytopes** - Kähler forms descend from ambient toric variety
3. **Specific orientifolds** - O3/O7 planes with right involution
4. **Controllable α' expansion** - Verified via Gopakumar-Vafa invariants

**Flux selection criteria:**
1. F-terms vanish at leading order in prepotential expansion
2. Superpotential W₀ vanishes perturbatively along a flat direction
3. Non-perturbative corrections are exponentially suppressed
4. Tadpole constraint: N_flux ≤ χ(CY)/24

### Computable Properties (CYTools)

CYTools can compute these properties that matter for KKLT/small-Λ:

**From Polytope class:**
- Hodge numbers (h¹¹, h²¹)
- Euler characteristic χ
- Favorability (whether Kähler forms descend)
- All triangulations (fine, regular, star)

**From CalabiYau class:**
- Intersection numbers κᵢⱼₖ
- Kähler cone (valid moduli range)
- Mori cone
- Second Chern class c₂

**From triangulations:**
- Different geometric phases
- Extended Kähler cone (all birational phases)

**Gopakumar-Vafa invariants:**
- Encode BPS spectrum
- Verify α' expansion is controlled
- Computed via topological string methods

### Implications for Our Search

**What we could do differently:**

1. **Prioritize large h²¹** - Our 3-gen filter gives |h¹¹ - h²¹| = 3, but within that, prefer large h²¹ (e.g., h²¹ = 100+ means h¹¹ = 97 or 103)

2. **Focus on favorable polytopes** - CYTools can check this

3. **Compute Gopakumar-Vafa invariants** - These predict whether small Λ is achievable for a given CY

4. **Structured flux search** - Instead of random flux integers, search for fluxes satisfying the DKMM algebraic conditions

5. **Conifold proximity** - Vacua near conifold points can have exponentially small W₀

### The Hard Truth

The McAllister results came from **analytical understanding**, not brute-force search:
- They knew *what to look for* mathematically
- They derived conditions fluxes must satisfy
- Then found explicit examples satisfying those conditions

Our GA approach is unlikely to find 10⁻¹²² by chance. But it might:
- Find **locally good** regions worth deeper analytical study
- Discover **correlations** between heuristics and physics that inform theory
- Serve as a **validation tool** for theoretical predictions

### Future Directions

1. **Implement DKMM flux conditions** as a search constraint
2. **Compute GV invariants** for promising polytopes
3. **Focus on conifold-adjacent** regions in moduli space
4. **Use heuristics to predict** which polytopes are worth expensive GV computation

## Progress

### Completed
- [x] Download pipeline for polytope data
- [x] Filter script for 3-generation candidates
- [x] Implement cluster-based adaptive selection with UCB
- [x] Add mutation pattern tracking
- [x] Persist cluster state to disk
- [x] Add polytope feature vectors (embeddings)
- [x] Rewrite physics_bridge.py to use CYTools + cymyc
- [x] Update ansible to install CYTools and cymyc
- [x] SQLite database layer with migrations
- [x] Meta-GA schema (algorithms, trials, fitness)
- [x] Worker locking with PID + heartbeat
- [x] Multi-file search binary structure
- [x] Generation 0 initialization
- [x] Meta-evolution (crossover, mutation)
- [x] Trial execution and metrics
- [x] Web dashboard (SolidStart + better-sqlite3)

### TODO

**Meta-GA Enhancements**
- [ ] Feature weight application to polytope selection
- [ ] Similarity search using weighted distance
- [ ] Path interpolation between good polytopes
- [ ] Learned ranker from evaluation history

**Infrastructure**
- [ ] Add meta-GA views to web dashboard
- [ ] Visualization of meta-fitness over generations
- [ ] Feature importance analysis from evolved weights

**Physics**
- [ ] Integrate cymyc trained metric models
- [ ] More accurate gauge coupling computation

## Polytope Transition Graph

### The Key Insight

Instead of treating the 12.2M polytopes as isolated points to sample randomly, we can model the **transition graph** between them:

- **Nodes**: Each polytope in our filtered database
- **Edges**: Valid topological transitions between polytopes

This transforms the search from "random jumps in 12M space" to "navigable graph exploration".

### Valid Transitions

Reflexive polytopes can be connected via:

1. **Vertex addition/removal** (if preserves reflexivity)
2. **Face subdivision**
3. **Different triangulations** (same polytope, different CY geometry)
4. **Conifold transitions** (connect different topologies through singular limits)
5. **Flops** (birational transformations)

The constraint |h11 - h21| = 3 must be preserved, which limits valid transitions.

### Graph Search Algorithms

Once the transition graph is computed (or computed on-demand), we can use:

```
BFS: "All polytopes within N transitions of current best"
A*:  "Shortest path from polytope A to polytope B"
Random walk: "Stochastic exploration of neighborhood"
Beam search: "Track top-K promising paths simultaneously"
```

### Vector Embedding for Similarity Search

Use ChromaDB or similar to index polytopes by structural features:

```python
embedding_v1 = [
    h11, h21,                    # Hodge numbers
    vertex_count,                # Combinatorial
    *vertex_coord_stats,         # Mean, std, min, max of coordinates
    *shape_features,             # Aspect ratio, spread, etc.
]
```

This enables:
- "Find 100 nearest polytopes to current best"
- "Explore in direction of improving fitness"
- Validate: does embedding proximity correlate with transition reachability?

### Evolving the Embedding

The embedding itself can evolve:

1. **v1**: Hand-crafted structural features
2. **v2**: Learned from which polytopes give similar physics
3. **v3**: Learned from which polytopes are transition-connected

Store multiple embedding versions, track which performs best for different query types.

### Implementation Plan

1. **Compute transition edges** (at least for promising polytopes)
   - Check if CYTools provides transition enumeration
   - Or compute: for each polytope, enumerate valid modifications, check if result is in our database

2. **Index in ChromaDB** with structural embedding
   - Start with simple features
   - Query neighbors, evaluate, track correlation

3. **Hybrid search**:
   - GA optimizes parameters on current polytope
   - Hit fitness ceiling → query transition graph for neighbors
   - Evaluate neighbors → move to best one
   - Repeat

4. **Learn better embeddings** from accumulated (polytope, fitness) data

## Meta-Genetic Algorithm

### The Problem with Fixed Algorithms

A traditional GA has fixed hyperparameters:
- Mutation rate
- Crossover strategy
- Selection pressure
- Polytope sampling strategy
- Embedding/similarity metric

We don't know which settings work best. Worse, the optimal settings might **change** as we explore different regions of the landscape.

### Meta-Evolution: Let Evolution Guide Evolution

Instead of one fixed algorithm, run a **population of algorithms**, each with different parameterized strategies:

```
Meta-GA Population:
├── Algorithm A: {mutation_rate: 0.1, polytope_strategy: "random", embedding: "v1"}
├── Algorithm B: {mutation_rate: 0.3, polytope_strategy: "nearest_neighbor", embedding: "v2"}
├── Algorithm C: {mutation_rate: 0.05, polytope_strategy: "transition_walk", embedding: "v1"}
└── ... (many variants)
```

### Meta-Fitness Function

The fitness of an **algorithm** is not a single number, but **the rate of improvement** it achieves:

```
meta_fitness(algorithm) = ∫ d(fitness)/dt over run duration
                        ≈ (final_fitness - initial_fitness) / generations
                        or: area under the fitness curve
```

Example comparison:
- Algorithm A: 0.30 → 0.50 in 1000 gens = **0.0002/gen**
- Algorithm B: 0.30 → 0.49 in 1000 gens = **0.00019/gen**
- Algorithm A wins, even though both ended near 0.5

Better yet, weight **early improvements** more (diminishing returns):
```
meta_fitness = Σ (fitness_improvement[t] * decay^t)
```

### Evolvable Parameters

Everything becomes a gene in the meta-genome:

**Search Strategy:**
- Polytope selection: random, nearest_k, transition_walk, cluster_ucb
- Neighborhood size for queries
- Exploration vs exploitation balance

**Mutation Operators:**
- Mutation rate (continuous)
- Which parameters to mutate (Kähler, complex, flux, g_s)
- Mutation magnitude distribution
- Adaptive mutation (increase when stuck)

**Embedding/Similarity:**
- Which features to include
- Feature weights
- Distance metric (L2, cosine, learned)

**Crossover:**
- Crossover rate
- Strategy: uniform, single-point, blend
- Cross polytopes or just parameters?

### Implementation

```
outer_loop:
    meta_population = [random_algorithm() for _ in range(N)]

    for meta_generation in range(M):
        # Run each algorithm for K generations
        for algo in meta_population:
            fitness_curve = run_inner_ga(algo, generations=K)
            algo.meta_fitness = compute_improvement_rate(fitness_curve)

        # Evolve the algorithms themselves
        meta_population = meta_select(meta_population)
        meta_population = meta_mutate(meta_population)
        meta_population = meta_crossover(meta_population)

        # Track which strategies are winning
        log_best_algorithms(meta_population)
```

### What This Discovers

The meta-GA will naturally discover:
- Which polytope exploration strategies actually help
- Whether transition graphs matter (vs random jumps)
- Optimal mutation rates for different search phases
- Which embedding features correlate with good physics
- When to explore vs exploit

We don't have to guess - **evolution figures out what works**.

### Practical Considerations

- **Compute cost**: Each meta-fitness evaluation requires a full inner GA run
- **Parallelization**: Inner runs are independent, perfect for distributed compute
- **Checkpointing**: Save meta-population state, can resume
- **Warmstart**: Seed meta-population with human-designed "reasonable" algorithms

## Speculative Shape Heuristics

### Philosophy

We don't know which geometric features predict good physics. The obvious ones (Hodge numbers, vertex counts) might not be the right ones. Some geometric property that seems "meaningless" mathematically might correlate with physics for reasons we don't understand.

**Strategy**: Compute *everything* we can think of, let the meta-GA discover what matters.

### "π-ness" / Circularity

How "circular" or "spherical" is a polytope vs angular/sharp?

**Candidate metrics:**
```python
# Ratio of inscribed to circumscribed sphere radii
pi_ratio = r_inscribed / r_circumscribed  # 1.0 = perfect sphere

# Deviation from sphericity
vertices_centered = vertices - centroid
distances = np.linalg.norm(vertices_centered, axis=1)
sphericity = 1 - np.std(distances) / np.mean(distances)

# Surface area to volume ratio vs sphere's ratio
# For a sphere: A/V = 3/r, for our polytope compare to ideal
isotropy_score = (4π)^(1/3) * (3V)^(2/3) / A  # 1.0 for sphere

# Angle distribution - are vertex angles close to π?
# Count how many vertex angles are near 180° (flat) vs sharp
flat_angle_fraction = count(angles > 150°) / total_angles

# Moment of inertia tensor eigenvalue ratios
# Equal eigenvalues = spherical, different = elongated/flat
I = inertia_tensor(vertices)
eigenvalues = sorted(np.linalg.eigvalsh(I))
inertia_isotropy = eigenvalues[0] / eigenvalues[2]  # 1.0 = spherical
```

### "Spirality" / Helical Structure

Does the polytope have any spiral or helical character?

**Candidate metrics:**
```python
# Project vertices onto axis, measure angular progression
# A spiral would show correlated angle-vs-distance pattern
for axis in [x, y, z, w]:
    projected = vertices.dot(axis)
    angles = np.arctan2(v[:, 1], v[:, 0])  # around axis
    spiral_correlation = np.corrcoef(projected, angles)[0, 1]

# Torsion: rate of change of binormal vector along paths
# High torsion = twisty, helical character

# Writhing number (from knot theory)
# Measures how much a curve wraps around itself
```

### Chirality / Handedness

Does the polytope have a preferred "handedness"? Is it distinguishable from its mirror image?

**Candidate metrics:**
```python
# Chirality = structure ≠ mirror image
# Compute: can we superimpose vertices with reflection(vertices)?

def chirality_score(vertices):
    # Try to align original with mirrored version
    mirrored = vertices.copy()
    mirrored[:, 0] *= -1  # Mirror across YZW plane

    # Optimal rotation alignment (Kabsch algorithm)
    R, rmsd = kabsch_align(vertices, mirrored)

    # High RMSD = chiral (can't superimpose)
    return rmsd

# For 4D, need to consider all reflection planes
# Compute chirality score for each axis reflection

# Determinant of vertex matrix (sign indicates handedness)
# This is crude but catches some chirality
handedness = np.sign(np.linalg.det(vertices[:4, :]))

# Triple/quadruple products of vertex vectors
# These flip sign under reflection
chiral_products = [np.dot(v1, np.cross(v2, v3)) for ...]  # 3D
# Generalize to 4D using Levi-Civita tensor
```

### Symmetry Analysis

What symmetry operations leave the polytope invariant?

**Candidate metrics:**
```python
# Count of symmetry operations
# For each candidate operation (rotation, reflection, etc.):
#   Check if transformed vertices ≈ original vertices (up to permutation)

def count_symmetries(vertices):
    symmetry_count = 0

    # Rotations around each axis (90°, 180°, 270°)
    for axis in axes:
        for angle in [90, 180, 270]:
            if is_invariant(vertices, rotate(axis, angle)):
                symmetry_count += 1

    # Reflections across planes
    for plane in planes:
        if is_invariant(vertices, reflect(plane)):
            symmetry_count += 1

    return symmetry_count

# Per-axis symmetry scores
# "How symmetric is it around the X axis vs Y axis?"
axis_symmetry = {}
for i, axis in enumerate(['x', 'y', 'z', 'w']):
    reflected = vertices.copy()
    reflected[:, i] *= -1
    axis_symmetry[axis] = vertex_overlap(vertices, reflected)

# Symmetry breaking: which symmetries are present vs absent
# e.g., has XY reflection but not XZ reflection

# Approximate symmetry: even if not exactly symmetric,
# how close to symmetric?
approx_symmetry_score = mean([best_alignment_rmsd(v, op(v)) for op in all_ops])
```

### Other Wild Ideas

**"Flatness" Measures:**
```python
# How close to living in a lower dimension?
# PCA: what fraction of variance is in top k dimensions?
pca = PCA(n_components=4)
pca.fit(vertices)
flatness_3d = sum(pca.explained_variance_ratio_[:3])  # How 3D is it?
flatness_2d = sum(pca.explained_variance_ratio_[:2])  # How 2D?

# Intrinsic dimension estimation
```

**"Regularity" / Uniformity:**
```python
# How uniform are edge lengths?
edge_length_uniformity = 1 - cv(edge_lengths)  # cv = std/mean

# How uniform are face areas?
face_area_uniformity = 1 - cv(face_areas)

# How uniform are dihedral angles?
dihedral_uniformity = 1 - cv(dihedral_angles)
```

**"Spikiness" / "Blobiness":**
```python
# Ratio of convex hull to actual volume (if different)
# For convex polytopes this is 1, but we can measure "protrusions"

# Vertex "exposure": how much each vertex sticks out
for v in vertices:
    exposure = distance_from_centroid(v) / mean_distance

# Max/mean ratio of distances = spikiness
spikiness = max(distances) / mean(distances)
```

**"Connectedness Patterns":**
```python
# Edge graph properties
G = construct_edge_graph(polytope)
clustering_coef = nx.average_clustering(G)
diameter = nx.diameter(G)
avg_path_length = nx.average_shortest_path_length(G)

# Spectrum of graph Laplacian (encodes connectivity)
laplacian_spectrum = np.linalg.eigvalsh(nx.laplacian_matrix(G))
```

**"Golden Ratio" / "Fibonacci" Patterns:**
```python
# How many vertex coordinate ratios are close to φ = 1.618...?
phi = (1 + np.sqrt(5)) / 2
phi_count = 0
for v in vertices:
    for i in range(4):
        for j in range(i+1, 4):
            if v[j] != 0:
                ratio = abs(v[i] / v[j])
                if abs(ratio - phi) < 0.1 or abs(ratio - 1/phi) < 0.1:
                    phi_count += 1

# Also check: 1, 2, 3, 5, 8 appearing in coordinates (Fibonacci)
```

**Integer Structure:**
```python
# Since polytope vertices are integers, patterns in those integers matter
# Prime factor distributions
# GCD/LCM patterns across coordinates
# Pythagorean relationships (a² + b² = c²)
```

### Concentration / Outlier Structure

How "conformist" vs "weird" is the vertex distribution? Is mass concentrated centrally with a few outliers, or spread uniformly?

**Candidate metrics:**
```python
# Core vs outlier ratio
distances = np.linalg.norm(vertices - centroid, axis=1)
median_dist = np.median(distances)
core_count = np.sum(distances < median_dist)
outlier_count = np.sum(distances > 2 * median_dist)
conformity_ratio = core_count / (outlier_count + 1)

# Kurtosis: heavy tails = more outliers
from scipy.stats import kurtosis
distance_kurtosis = kurtosis(distances)  # High = spiky outliers

# "Loner score": how isolated is the most isolated vertex?
from scipy.spatial.distance import cdist
pairwise = cdist(vertices, vertices)
np.fill_diagonal(pairwise, np.inf)
nearest_neighbor_dists = pairwise.min(axis=1)
max_isolation = np.max(nearest_neighbor_dists)
mean_isolation = np.mean(nearest_neighbor_dists)
loner_score = max_isolation / mean_isolation

# Cluster tendency: does it naturally split into groups?
# Silhouette score for k=2,3,4 clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
cluster_scores = {}
for k in [2, 3, 4]:
    if len(vertices) > k:
        km = KMeans(n_clusters=k).fit(vertices)
        cluster_scores[k] = silhouette_score(vertices, km.labels_)

# Density variation: local density at each vertex
from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=1.0).fit(vertices)
log_densities = kde.score_samples(vertices)
density_variation = np.std(np.exp(log_densities))
```

### Statistical Distribution Tests

Treat vertex coordinates as a dataset and apply every statistical test imaginable:

**Basic Distribution Stats:**
```python
coords_flat = vertices.flatten()

# Central tendency
coord_mean = np.mean(coords_flat)
coord_median = np.median(coords_flat)
coord_mode = scipy.stats.mode(coords_flat, keepdims=False).mode
mean_median_diff = abs(coord_mean - coord_median)  # Skewness indicator

# Spread
coord_std = np.std(coords_flat)
coord_iqr = np.percentile(coords_flat, 75) - np.percentile(coords_flat, 25)
coord_range = np.max(coords_flat) - np.min(coords_flat)

# Shape
coord_skewness = scipy.stats.skew(coords_flat)
coord_kurtosis = scipy.stats.kurtosis(coords_flat)

# Per-axis versions of all the above
for axis in range(4):
    axis_coords = vertices[:, axis]
    stats[f'mean_axis_{axis}'] = np.mean(axis_coords)
    stats[f'median_axis_{axis}'] = np.median(axis_coords)
    stats[f'std_axis_{axis}'] = np.std(axis_coords)
    # ... etc
```

**Information Theory / Entropy:**
```python
# Shannon entropy of coordinate distribution
# Higher entropy = more "random"/spread out, lower = more structured
from scipy.stats import entropy

# Discretize coordinates into bins
hist, _ = np.histogram(coords_flat, bins=20, density=True)
hist = hist[hist > 0]  # Remove zeros for log
shannon_entropy = entropy(hist)

# Per-axis entropy
axis_entropies = []
for axis in range(4):
    hist, _ = np.histogram(vertices[:, axis], bins=10, density=True)
    hist = hist[hist > 0]
    axis_entropies.append(entropy(hist))

# Joint entropy (how much info in the full 4D distribution)
# Approximated via histogram in 4D bins
joint_hist, _ = np.histogramdd(vertices, bins=5)
joint_hist_flat = joint_hist.flatten()
joint_hist_flat = joint_hist_flat[joint_hist_flat > 0]
joint_entropy = entropy(joint_hist_flat / joint_hist_flat.sum())

# Mutual information between axes
# High MI = axes are correlated, low = independent
from sklearn.metrics import mutual_info_score
for i in range(4):
    for j in range(i+1, 4):
        # Discretize for MI calculation
        xi = np.digitize(vertices[:, i], bins=10)
        xj = np.digitize(vertices[:, j], bins=10)
        mi[f'{i}_{j}'] = mutual_info_score(xi, xj)
```

**Compressibility / Kolmogorov Complexity Proxy:**
```python
import zlib
import json

# How well does the vertex data compress?
# Lower = more regular/patterned, higher = more random
vertex_bytes = json.dumps(vertices.tolist()).encode()
compressed = zlib.compress(vertex_bytes, level=9)
compression_ratio = len(compressed) / len(vertex_bytes)

# Also try different serializations
vertex_str = ' '.join(map(str, coords_flat))
compressed_str = zlib.compress(vertex_str.encode())
string_compression_ratio = len(compressed_str) / len(vertex_str.encode())

# Sorted compression (does sorting help? = has order structure)
sorted_coords = np.sort(coords_flat)
sorted_bytes = json.dumps(sorted_coords.tolist()).encode()
sorted_compressed = zlib.compress(sorted_bytes)
sort_compression_gain = len(compressed) / len(sorted_compressed)
```

**Axis Balance / Distribution Shape:**
```python
# How balanced is the distribution along each axis?
# Perfectly symmetric around 0 → balance = 1
axis_balance = {}
for axis in range(4):
    positive = np.sum(vertices[:, axis] > 0)
    negative = np.sum(vertices[:, axis] < 0)
    zero = np.sum(vertices[:, axis] == 0)
    axis_balance[axis] = min(positive, negative) / max(positive, negative, 1)

# Which axis has the most/least spread?
axis_spreads = [np.std(vertices[:, i]) for i in range(4)]
spread_ratio = max(axis_spreads) / min(axis_spreads)  # Elongation

# Quartile analysis per axis
for axis in range(4):
    q1, q2, q3 = np.percentile(vertices[:, axis], [25, 50, 75])
    stats[f'q1_axis_{axis}'] = q1
    stats[f'q2_axis_{axis}'] = q2  # median
    stats[f'q3_axis_{axis}'] = q3
    stats[f'iqr_axis_{axis}'] = q3 - q1
    # Quartile skewness
    stats[f'quartile_skew_{axis}'] = (q3 + q1 - 2*q2) / (q3 - q1 + 1e-10)
```

**Normality and Distribution Tests:**
```python
from scipy import stats

# Is the distribution normal? (probably not, but how far off?)
_, normality_pvalue = stats.normaltest(coords_flat)

# Shapiro-Wilk test (better for small samples)
if len(coords_flat) < 5000:
    _, shapiro_pvalue = stats.shapiro(coords_flat[:5000])

# Is it uniform?
_, uniform_ks_stat = stats.kstest(
    (coords_flat - coords_flat.min()) / (coords_flat.max() - coords_flat.min()),
    'uniform'
)

# Anderson-Darling test against various distributions
for dist in ['norm', 'expon', 'logistic']:
    result = stats.anderson(coords_flat, dist=dist)
    stats[f'anderson_{dist}'] = result.statistic
```

**Correlation Structure:**
```python
# Correlation matrix between axes
corr_matrix = np.corrcoef(vertices.T)

# Extract unique correlations
correlations = {
    'xy': corr_matrix[0, 1],
    'xz': corr_matrix[0, 2],
    'xw': corr_matrix[0, 3],
    'yz': corr_matrix[1, 2],
    'yw': corr_matrix[1, 3],
    'zw': corr_matrix[2, 3],
}

# Overall correlation strength
mean_abs_correlation = np.mean(np.abs(corr_matrix[np.triu_indices(4, k=1)]))

# Eigenvalues of correlation matrix (PCA-like)
corr_eigenvalues = np.linalg.eigvalsh(corr_matrix)
# High first eigenvalue = strong dominant direction
```

**Runs Test / Sequence Structure:**
```python
# If we sort coordinates, how many "runs" of increasing/decreasing?
# More runs = more random, fewer = more structured
from statsmodels.sandbox.stats.runs import runstest_1samp

sorted_coords = np.sort(coords_flat)
diffs = np.diff(sorted_coords)
runs_stat, runs_pvalue = runstest_1samp(diffs, correction=False)

# Autocorrelation of sorted differences
autocorr = np.correlate(diffs, diffs, mode='full')
autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
autocorr_decay = autocorr[1] / autocorr[0] if autocorr[0] > 0 else 0
```

### Why These Might Matter

Physics often cares about:
- **Symmetry** → gauge groups, conservation laws
- **Chirality** → matter/antimatter asymmetry, weak force parity violation
- **Regularity** → stability, moduli stabilization
- **Connectivity** → interaction patterns

A polytope's geometry determines the Calabi-Yau topology, which determines the 4D physics. Geometric features we think are "random" might encode deep physics.

**We don't know what we don't know.** Throw everything at it, let evolution sort it out.

### Scaling Outlier Scores to 12M Polytopes

The current outlier score computation loads all heuristics into memory and computes z-scores. This won't scale to 12M polytopes (~11GB just for the matrix).

**Streaming Approach for Scale:**

1. **Welford's Algorithm for Running Stats**
   ```python
   class RunningStats:
       """Compute mean/variance in single pass, O(1) memory per dimension."""
       def __init__(self, n_dims):
           self.n = 0
           self.mean = np.zeros(n_dims)
           self.M2 = np.zeros(n_dims)  # Sum of squared differences

       def update(self, x):
           self.n += 1
           delta = x - self.mean
           self.mean += delta / self.n
           delta2 = x - self.mean
           self.M2 += delta * delta2

       def variance(self):
           return self.M2 / self.n if self.n > 1 else np.zeros_like(self.M2)

       def std(self):
           return np.sqrt(self.variance())
   ```

2. **Two-File Architecture**
   - `population_stats.json` - Running mean/std/count per dimension
   - `heuristics_sample.json` - Per-polytope heuristics (unchanged)

3. **Workflow**
   - **First pass**: Compute heuristics, update running stats
   - **Second pass (optional)**: Compute outlier scores using stored stats
   - **Incremental**: New polytopes update running stats, get outlier score immediately

4. **Outlier Score Computation**
   ```python
   def compute_outlier_score(heuristics: dict, stats: RunningStats) -> dict:
       """Compute outlier score for single polytope against population stats."""
       flat = flatten_heuristics(heuristics)
       z_scores = (flat - stats.mean) / (stats.std() + 1e-10)

       heuristics['outlier_score'] = float(np.mean(np.abs(z_scores)))
       heuristics['outlier_max_zscore'] = float(np.max(np.abs(z_scores)))
       heuristics['outlier_max_dim'] = dimension_names[np.argmax(np.abs(z_scores))]
       heuristics['outlier_count_3sigma'] = int(np.sum(np.abs(z_scores) > 3))

       return heuristics
   ```

5. **Incremental Updates**
   - When adding new polytopes: update stats, compute their outlier scores
   - Existing polytope outlier scores become stale as population grows
   - Can periodically recompute all outlier scores in batch (or accept slight staleness)

**Implementation TODO:**
- [ ] Create `PopulationStats` class with Welford's algorithm
- [ ] Save/load stats to `population_stats.json`
- [ ] Modify `compute_heuristics.py` to update stats incrementally
- [ ] Add `--recompute-outliers` flag for batch recomputation
- [ ] Consider: outlier scores relative to local cluster vs global population

## Decomposed Fitness: Multi-Dimensional Gradient Descent

### The Problem with Scalar Fitness

Currently, we search for specific target values:
- Fine Structure Constant: α_em = 7.297×10⁻³
- Strong Coupling: α_s = 0.118
- Weinberg Angle: sin²θ_W = 0.231
- Cosmological Constant: Λ = 2.888×10⁻¹²²

The fitness function combines these into a single scalar (weighted sum of log-errors). This creates a **deceptively smooth** landscape that hides the underlying complexity.

**Key insight**: These observables are **computed quantities**—they're functions of more fundamental parameters. If we decompose them into their constituent parts, we might find that:
1. Different "components" have different fitness gradients
2. The landscape is smoother along some component axes than others
3. We can make progress on multiple fronts simultaneously

### Physics Decomposition Examples

#### 1. Gauge Couplings

At tree level in Type IIB:
```
α_a = g_s / (4π τ_a)
```
where τ_a is the 4-cycle volume wrapped by the D7-brane.

**Decomposition:**
- Component A: g_s (string coupling) — affects ALL gauge couplings equally
- Component B: τ_a ratios — affects RATIOS between gauge couplings
- Component C: Overall volume scale — affects absolute magnitudes

Instead of searching for "α_em = 0.00729", search for:
- g_s in correct range
- τ_ratios that give correct coupling ratios
- Volume scale that gives correct absolute values

Each component might have a smoother fitness landscape than the combined observable.

#### 2. Weinberg Angle

The Weinberg angle relates SU(2) and U(1) couplings:
```
sin²θ_W = g'² / (g² + g'²)
         = α_1 / (α_1 + α_2)   (at unification scale)
```

This depends on:
- The 4-cycle volumes wrapped by different brane stacks
- The RG running from string scale to Z mass
- The specific GUT embedding (SU(5), SO(10), etc.)

**Decomposition:**
- Component X: Cycle volume ratio τ_1/τ_2 → determines g'/g ratio at string scale
- Component Y: RG running distance → depends on unification scale
- Component Z: GUT breaking pattern → discrete choice

We might find: X is easy to optimize, Y has smooth gradients, Z is a discrete branching choice.

#### 3. Cosmological Constant

From KKLT:
```
Λ ≈ V_uplift + V_AdS
  = D/V^(4/3) - 3|W₀|²/V² × e^K
```

**Decomposition:**
- Component P: W₀ magnitude — from flux choice, can be exponentially small
- Component Q: CY volume V — from Kähler moduli
- Component R: Uplift contribution D — from anti-D3 brane count

The McAllister group achieved 10⁻¹²² by making W₀ exponentially small (DKMM mechanism), not by fine-tuning all components equally.

### Why This Might Help

#### A. Gradient Isolation

Currently: Moving in parameter space changes ALL observables simultaneously.

With decomposition: We can identify directions that primarily affect ONE component.

**Example**: If we discover that:
- g_s primarily affects the "magnitude scale" component
- Kähler ratios primarily affect the "ratio" component
- Flux integers primarily affect the "W₀" component

Then we can do **coordinate descent**: optimize one component at a time.

#### B. Partial Success Recognition

Currently: A configuration with α_em=0.007, α_s=0.12, sin²θ_W=0.5, Λ=10⁻³⁰ gets a mediocre fitness score.

With decomposition: We might recognize:
- "Coupling magnitude": ✓ Correct (both α values close)
- "Coupling ratio": ✓ Correct (α_s/α_em ≈ 16, target is 16.2)
- "Weinberg angle": ✗ Wrong (0.5 vs 0.23)
- "CC magnitude": ✗ Wrong (10⁻³⁰ vs 10⁻¹²²)

This tells us: **2/4 components are solved**. Focus search on the unsolved components.

#### C. Transfer Learning Between Targets

Different components might be transferable:
- A configuration that achieves good "coupling ratios" might be a good starting point for optimizing "CC magnitude"
- The "correct g_s range" might be the same for many good solutions

### Parameterization Connection

This relates to the RESEARCH_GA_PARAMETERIZATIONS.md question: **which parameterization creates the smoothest fitness gradients?**

The answer might be: parameterize in **component space**, not observable space.

**Observable space**: (α_em, α_s, sin²θ_W, Λ)
**Component space**: (g_s, τ_ratio_1, τ_ratio_2, V_scale, W₀_magnitude, ...)

The component space might:
- Have more dimensions (more degrees of freedom)
- But smoother gradients per dimension
- And independent optimization axes

### Implementation Ideas

#### 1. Multi-Objective GA

Instead of single fitness, track multiple objectives:
```python
objectives = {
    "gauge_magnitude_error": ...,
    "gauge_ratio_error": ...,
    "weinberg_error": ...,
    "cc_log_error": ...,
}
```

Use Pareto-front based selection (NSGA-II style). An individual is "good" if no other individual beats it on ALL objectives.

#### 2. Hierarchical Search

1. First optimize gauge coupling ratios (ignore magnitudes)
2. Then fix ratios, optimize magnitudes
3. Then fix gauge sector, optimize CC

This assumes the components are somewhat independent.

#### 3. Component Fitness Weighting

Dynamic weights based on current progress:
```python
if gauge_ratios_solved:
    weight_ratios *= 0.1  # De-emphasize solved component
    weight_cc *= 10       # Focus on unsolved component
```

#### 4. Feature Engineering for Components

Compute component-specific geometric features:
- "g_s range compatibility" — which geometric features correlate with g_s being in the right range?
- "τ ratio structure" — which features predict good cycle volume ratios?

### Open Questions

1. **What are the actual components?**
   The physics determines this. Need to carefully analyze which parameter combinations affect which observables.

2. **Are components independent?**
   Probably not perfectly. But even partial independence helps.

3. **Do smoother component gradients exist?**
   Empirical question. Try it and see.

4. **Does the 214 vs 4 Kähler parameterization matter here?**
   The 214-dimensional space might naturally align with component axes better than the 4-dimensional Kähler moduli space.

### Experiment: Component Gradient Analysis

Using McAllister's polytope 4-214-647 as ground truth:

1. At the optimal point, compute Jacobian: ∂(observable)/∂(parameter) for each observable and parameter
2. Do SVD of Jacobian to find "principal component directions"
3. Check: do the principal components align with interpretable physics?
4. Measure: gradient smoothness along principal components vs along original parameter axes

If components exist and have smooth gradients, SVD should reveal them.

### Related: "Soft" Targets

Instead of hard targets:
```
fitness = distance(α_em, 0.00729)
```

Use soft/banded targets:
```
fitness = 0 if α_em ∈ [0.006, 0.009] else penalty
```

This turns the optimization from "find exact point" to "find valid region". Regions might be much easier to find, then refine within region.
