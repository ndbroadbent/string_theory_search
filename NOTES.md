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
- Runs genetic algorithm with PALP physics evaluation
- Persists state for resume capability

## Genetic Algorithm Design

### Current Implementation
- Population of candidate compactifications
- Each individual: polytope + moduli parameters
- Fitness: combination of physics scores (gauge group, generations, masses, etc.)
- Selection: tournament selection
- Crossover: parameter blending
- Mutation: polytope swap, parameter perturbation

### Fitness Components
1. **Gauge group score**: How close to SU(3)×SU(2)×U(1)
2. **Generation score**: Penalize deviation from 3
3. **Mass hierarchy score**: Correct fermion mass ratios
4. **Mixing angle score**: CKM/PMNS matrix elements
5. **Cosmological constant**: Should be tiny positive

## Adaptive Polytope Selection (TODO)

### Problem
Random polytope selection wastes evaluations on unpromising regions of the search space.

### Solution: Learned Prior via Clustering

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                   Polytope Clustering                    │
├─────────────────────────────────────────────────────────┤
│  Features for clustering:                                │
│  - h11, h21 (Hodge numbers)                             │
│  - vertex_count                                          │
│  - vertex coordinate statistics (mean, variance, etc)    │
│  - facet structure hash                                  │
│                                                          │
│  Cluster → fitness history → selection weight            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Cluster A (h11~1-5)     ──► avg_fitness: 0.82  ──► 40% │
│   Cluster B (h11~6-10)    ──► avg_fitness: 0.45  ──► 15% │
│   Cluster C (h11~11-20)   ──► avg_fitness: 0.31  ──► 10% │
│   Cluster D (small vertex)──► avg_fitness: 0.71  ──► 35% │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**State to Persist (cluster_state.json):**
```json
{
  "clusters": {
    "cluster_0": {
      "centroid": {"h11": 2.3, "h21": 5.3, "vertex_count": 8.1},
      "polytope_ids": [123, 456, 789, ...],
      "evaluations": 1542,
      "fitness_sum": 847.3,
      "fitness_best": 0.92,
      "selection_weight": 0.35
    },
    ...
  },
  "hot_polytopes": [
    {"id": 456, "fitness": 0.92, "offspring_success_rate": 0.73},
    ...
  ],
  "mutation_patterns": {
    "toward_small_h11": {"attempts": 234, "improvements": 89},
    "toward_small_vertex": {"attempts": 156, "improvements": 67}
  },
  "total_evaluations": 50000,
  "last_updated": "2024-12-08T13:45:00Z"
}
```

**Selection Algorithm:**
1. With probability p_exploit (e.g., 0.7): sample from weighted clusters
2. With probability p_explore (e.g., 0.2): sample uniformly (explore)
3. With probability p_hot (e.g., 0.1): sample from hot polytopes

**Weight Update (after each evaluation):**
```python
def update_cluster_weight(cluster, fitness):
    cluster.evaluations += 1
    cluster.fitness_sum += fitness
    avg = cluster.fitness_sum / cluster.evaluations

    # UCB-like exploration bonus
    exploration_bonus = sqrt(log(total_evals) / cluster.evaluations)

    cluster.selection_weight = avg + 0.1 * exploration_bonus
```

**Mutation Guidance:**
Track which mutations improve fitness:
- "Move toward cluster X" → record success/failure
- After enough data, bias mutations toward successful patterns

### Implementation Plan

**Phase 1: Basic Clustering**
- Cluster polytopes by (h11, h21, vertex_count)
- Simple k-means or binning
- Track per-cluster fitness statistics

**Phase 2: Adaptive Selection**
- Implement weighted sampling
- UCB exploration bonus
- Persist and resume cluster state

**Phase 3: Mutation Learning**
- Track mutation directions that improve fitness
- Learn which transformations help
- Guide mutations toward promising regions

## Infrastructure

### Server Setup (Ansible)
- LXC container at 10.5.7.33
- 32 cores, 8GB RAM
- ZFS mount at /data/polytopes (6.1 TB)
- PALP installed from GitLab
- Rust toolchain with Python 3.10

### File Locations on Server
```
/root/string_theory/           # Project root
/root/palp_source/             # PALP installation
/data/polytopes/parquet/       # Raw parquet files
/data/polytopes/download.log   # Download progress
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
# Check download progress
tail -f /data/polytopes/download.log

# Check GA progress
ls -la results/
cat results/best_*.json | jq .fitness
```

## Key Files

| File | Purpose |
|------|---------|
| `download_all_polytopes.py` | Download parquet files from HuggingFace |
| `filter_three_gen.py` | Filter to 3-generation candidates |
| `src/bin/real_physics.rs` | Main GA binary |
| `src/real_genetic.rs` | GA implementation |
| `src/physics.rs` | Physics calculations via PALP |
| `ansible/playbook.yml` | Server setup automation |

## References

- Kreuzer-Skarke database: https://huggingface.co/datasets/calabi-yau-data/polytopes-4d
- PALP: https://gitlab.com/stringstuwien/PALP
- Three-generation CY: https://arxiv.org/abs/0909.3947
- CY Landscape ML: https://arxiv.org/abs/1812.02893

## TODO

- [ ] Finish downloading all 32 parquet files
- [ ] Run filter_three_gen.py to create filtered dataset
- [ ] Update real_physics to use polytopes_three_gen.json
- [ ] Implement cluster-based adaptive selection
- [ ] Add mutation pattern learning
- [ ] Persist cluster state to disk
- [ ] Add monitoring/visualization for cluster weights
- [ ] Consider ensemble of GAs exploring different regions
