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
