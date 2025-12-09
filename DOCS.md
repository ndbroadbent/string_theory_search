# String Theory Landscape Explorer - Technical Documentation

## Overview

This project searches through the string theory landscape to find Calabi-Yau compactifications that reproduce Standard Model physics. It uses a **meta-genetic algorithm** - a GA that evolves the parameters of other GAs - to discover optimal search strategies.

## The Physics Problem

String theory predicts our 10-dimensional universe compactifies down to 4 dimensions, with the extra 6 dimensions curled up into a Calabi-Yau manifold. The shape of this manifold determines the physics we observe:

- **Gauge couplings** (α_em, α_s, sin²θ_W) - strengths of forces
- **Number of generations** (3 families of fermions)
- **Cosmological constant** (Λ ≈ 10⁻¹²² in Planck units)

The "landscape" contains ~10⁵⁰⁰ possible compactifications. We're searching for the needle in this cosmic haystack.

## Five-Level Hierarchy

The system operates at five nested levels:

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: META-GENERATION                                        │
│ A population of algorithms (e.g., 16 algorithms)                │
│ Each algorithm is a different search strategy                   │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: ALGORITHM                                              │
│ One specific set of parameters:                                 │
│ - 50+ feature weights for polytope similarity                   │
│ - GA parameters (population, mutation rate, etc.)               │
│ - Search strategy (similarity_radius, interpolation_weight)     │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: TRIAL                                                  │
│ One complete run of an algorithm (10 trials per algorithm)      │
│ Multiple trials needed because fitness is noisy                 │
├─────────────────────────────────────────────────────────────────┤
│ Level 4: INNER GENERATION                                       │
│ One generation within a trial's GA                              │
│ Evolves moduli parameters for a given polytope                  │
├─────────────────────────────────────────────────────────────────┤
│ Level 5: EVALUATION                                             │
│ One physics computation via CYTools + cymyc                     │
│ Computes gauge couplings, CC, etc. for specific genome          │
└─────────────────────────────────────────────────────────────────┘
```

## The Meta-GA

### What It Evolves

The meta-GA evolves **search strategies**, not solutions. Each "algorithm" (meta-individual) contains:

#### Feature Weights (~50 parameters)
```json
{
  "sphericity": 1.5,
  "chirality_optimal": 2.0,
  "outlier_score": 0.5,
  "h11": 1.0,
  "shannon_entropy": 0.8,
  ...
}
```
These weights determine how "similarity" between polytopes is computed. High weight = this feature matters more when finding similar polytopes.

#### Search Strategy
- `similarity_radius` (0.1-1.0): How wide to search around good polytopes
- `interpolation_weight` (0.0-1.0): Balance between similarity search and path-walking between distant good polytopes

#### Inner GA Parameters
- `population_size`: 30-150
- `max_generations`: 5-20
- `mutation_rate`: 0.2-0.6
- `mutation_strength`: 0.2-0.5
- `crossover_rate`: 0.6-0.9
- `tournament_size`: 3-7
- `elite_count`: 5-20

#### Polytope Switching
- `polytope_patience`: Generations before considering switch
- `switch_threshold`: Minimum improvement to stay
- `switch_probability`: Random switch chance

### Meta-Fitness

An algorithm's meta-fitness is computed from aggregated trial results:

```
meta_fitness = 0.3 × mean_improvement_rate
             + 0.3 × best_final_fitness
             + 0.4 × (1 / (1 + mean_cc_log_error))
```

This rewards algorithms that:
1. Improve fitness quickly (good search dynamics)
2. Find high absolute fitness values
3. Get close to the cosmological constant target

### Evolution Process

1. **Generation 0**: Random algorithms created
2. **Trials**: Each algorithm runs 10 trials to measure noisy performance
3. **Selection**: Top performers selected (elite_count = 4)
4. **Reproduction**: Mutation (40%) and crossover (60%) create next generation
5. **Repeat**: Process continues indefinitely

## Database Schema

All data lives in SQLite with WAL mode for concurrent workers.

### Core Tables

```sql
-- Every physics evaluation ever performed
evaluations (
    polytope_id, run_id, generation,
    g_s, kahler_moduli, complex_moduli, flux_f, flux_h,
    fitness, alpha_em, alpha_s, sin2_theta_w, n_generations,
    cosmological_constant, success, error
)

-- Polytope metadata (lazy-loaded on first eval)
polytopes (
    id, h11, h21, vertex_count, vertices,
    eval_count, fitness_sum, fitness_min, fitness_max
)

-- Computed heuristics per polytope (~50 features)
heuristics (
    polytope_id,
    sphericity, chirality_optimal, outlier_score, ...
)
```

### Meta-GA Tables

```sql
-- Algorithm definitions (the "genome" of search strategies)
meta_algorithms (
    id, name, version, meta_generation,
    feature_weights,  -- JSON with ~50 keys
    similarity_radius, interpolation_weight,
    population_size, max_generations, mutation_rate, ...
    trials_required,
    status, locked_by_pid, last_heartbeat_at
)

-- Trial results
meta_trials (
    algorithm_id, run_id, generations_run,
    initial_fitness, final_fitness, improvement_rate,
    fitness_auc, best_cc_log_error, physics_success_rate
)

-- Aggregated meta-fitness per algorithm
meta_fitness (
    algorithm_id, trial_count,
    mean_improvement_rate, best_final_fitness,
    best_cc_log_error, meta_fitness
)
```

## Worker Architecture

### Distributed Processing

Multiple workers can run simultaneously on different machines. They coordinate via the database:

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Worker1 │     │ Worker2 │     │ Worker3 │
│ PID 123 │     │ PID 456 │     │ PID 789 │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              ┌──────┴──────┐
              │   SQLite    │
              │  (WAL mode) │
              └─────────────┘
```

### Locking Protocol

Workers use a PID + heartbeat mechanism:

1. **Acquire**: Write PID and timestamp, read back to verify
2. **Heartbeat**: Update timestamp every 20 seconds
3. **Stale detection**: If heartbeat > 60s old, algorithm is abandoned
4. **Complete**: Clear lock when all trials done

```sql
-- Acquiring an algorithm
UPDATE meta_algorithms
SET locked_by_pid = 123, last_heartbeat_at = '2024-01-15 10:30:00'
WHERE id = 42;

-- Verify we got it (another worker might have won)
SELECT locked_by_pid, last_heartbeat_at FROM meta_algorithms WHERE id = 42;
```

### Worker Loop

```
┌─────────────────────────────────────────────┐
│                 Worker Loop                  │
├─────────────────────────────────────────────┤
│  1. Check if gen 0 needs initialization     │
│  2. Check if current gen complete → evolve  │
│  3. Try to acquire algorithm                │
│     - If none available, wait 30s           │
│  4. Start heartbeat thread                  │
│  5. Run trial (inner GA)                    │
│  6. Record trial results                    │
│  7. If trials_required met → mark complete  │
│  8. Loop back to step 1                     │
└─────────────────────────────────────────────┘
```

## Code Structure

### Library (`src/lib.rs`)

```
src/
├── lib.rs           # Module exports
├── constants.rs     # Target physics values
├── physics.rs       # PyO3 bridge to CYTools/cymyc
├── db.rs            # SQLite layer, MetaAlgorithm struct
├── meta_ga.rs       # Evolution functions, feature weights
└── searcher.rs      # Inner GA: LandscapeSearcher, Individual
```

### Binary (`src/bin/search/`)

```
src/bin/search/
├── main.rs          # Entry point, initialization
├── config.rs        # CLI args, config file parsing
├── heartbeat.rs     # Background heartbeat thread
├── trial.rs         # Run one trial, compute metrics
└── worker.rs        # Main loop, algorithm acquisition
```

## Configuration

### config.toml

```toml
[paths]
polytopes = "polytopes_three_gen.jsonl"
output_dir = "results"
database = "data/string_theory.db"

[meta_ga]
algorithms_per_generation = 16
trials_required = 10
```

### Environment Variables

- `STRING_THEORY_DB`: Override database path
- `VIRTUAL_ENV`: Required for PyO3 to find Python packages
- `PYO3_PYTHON`: Python interpreter for building

## Running

### Single Worker

```bash
# Build
PYO3_PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 cargo build --release

# Run
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search
```

### Multiple Workers

Just run multiple instances - they coordinate automatically:

```bash
# Terminal 1
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search

# Terminal 2
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search

# Terminal 3
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search
```

### On Server (16+ workers)

```bash
# Start workers in tmux/screen sessions
for i in {1..16}; do
    tmux new-session -d -s "worker$i" './target/release/search'
done
```

## Monitoring

### Database Queries

```sql
-- Current meta-generation status
SELECT meta_generation, status, COUNT(*)
FROM meta_algorithms
GROUP BY meta_generation, status;

-- Top performing algorithms
SELECT a.id, a.meta_generation, f.meta_fitness, f.trial_count
FROM meta_algorithms a
JOIN meta_fitness f ON f.algorithm_id = a.id
ORDER BY f.meta_fitness DESC
LIMIT 10;

-- Worker activity
SELECT id, locked_by_pid, last_heartbeat_at, status
FROM meta_algorithms
WHERE status = 'running';

-- Best physics results ever
SELECT polytope_id, fitness, cosmological_constant
FROM evaluations
WHERE success = 1
ORDER BY fitness DESC
LIMIT 20;
```

## Key Insights

### Why Meta-GA?

The inner GA has ~10 tunable parameters plus ~50 feature weights. Manual tuning is:
1. Slow (each run takes hours)
2. Biased (human intuition fails in 50+ dimensions)
3. Non-adaptive (what works early may not work later)

The meta-GA automates hyperparameter optimization with:
- Parallel exploration of parameter space
- Statistical validation (multiple trials)
- Continuous adaptation as we learn more about the landscape

### Why Feature Weights?

Polytopes are characterized by ~50 geometric heuristics (sphericity, chirality, entropy, etc.). The meta-GA learns which features predict good physics:

- Maybe spherical polytopes → good gauge couplings
- Maybe high chirality → correct generations
- Maybe low entropy → small CC

We don't know the correlations a priori - the meta-GA discovers them.

### Why Multiple Trials?

Physics fitness is noisy:
- Same algorithm may find fitness 0.3 or 0.5 depending on random initialization
- One lucky run doesn't mean the algorithm is good
- 10 trials gives statistical confidence in meta-fitness

## Future Directions

1. **Feature importance analysis**: Which heuristics actually correlate with good physics?
2. **Transfer learning**: Use learned weights to warm-start new runs
3. **Adaptive trials**: Run more trials for uncertain algorithms
4. **Multi-objective**: Pareto-optimize for different physics targets
5. **Neural surrogate**: Train NN to predict fitness without full physics computation
