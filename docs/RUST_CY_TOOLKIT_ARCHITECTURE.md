# Calabi-Yau Manifold Toolkit (Rust)

*A high-performance, formally verified toolkit for computational string theory*

## Vision

Build the definitive open-source toolkit for evaluating Calabi-Yau compactifications, capable of:
1. Reproducing McAllister's KKLT results (static Λ)
2. Computing Cicoli's quintessence dynamics (w₀, wₐ)
3. Scaling to search 10⁸+ configurations via genetic algorithms

**Goal**: A tool so valuable to the HET physics community that it becomes the standard, potentially cited in future breakthrough papers.

---

## Design Principles

### From rack-gateway
- Ultra-strict linting (clippy::pedantic, deny unsafe unless justified)
- Clean, minimal abstractions
- Comprehensive benchmarking

### Beyond rack-gateway
- **100% test coverage** for core math
- **Formal verification** via Aeneas for critical algorithms
- **Self-optimizing caches** that evolve via GA
- **Reproducibility**: Every computation traceable to paper/equation

---

## Crate Structure

```
cytools-rs/                     # Workspace root
├── Cargo.toml                  # Workspace manifest
│
├── crates/
│   ├── cy-core/                # Core mathematical primitives
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── lattice.rs      # Lattice point enumeration
│   │   │   ├── polytope.rs     # Convex polytope operations
│   │   │   ├── triangulation.rs # FRST, star triangulations
│   │   │   ├── intersection.rs  # κ_ijk computation
│   │   │   ├── kahler.rs       # Kähler cone, volumes
│   │   │   └── gv.rs           # Gopakumar-Vafa invariants
│   │   └── tests/              # Property-based tests
│   │
│   ├── cy-moduli/              # Moduli stabilization
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── kklt.rs         # KKLT stabilization (τ = c_i solver)
│   │   │   ├── lvs.rs          # Large Volume Scenario
│   │   │   ├── racetrack.rs    # g_s, W₀ computation
│   │   │   └── potential.rs    # Scalar potential V(T)
│   │   └── tests/
│   │
│   ├── cy-cosmology/           # Cosmological evolution
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── friedmann.rs    # Friedmann equations
│   │   │   ├── quintessence.rs # Scalar field evolution
│   │   │   ├── cpl.rs          # CPL fitting (w₀, wₐ)
│   │   │   └── observables.rs  # H(z), D_L(z), etc.
│   │   └── tests/
│   │
│   ├── cy-cache/               # Intelligent caching system
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── strategy.rs     # Cache strategy enum/trait
│   │   │   ├── memory.rs       # In-memory LRU/LFU
│   │   │   ├── disk.rs         # Disk-backed with mmap
│   │   │   ├── tiered.rs       # RAM → NVMe → HDD
│   │   │   ├── bloom.rs        # Bloom filter for second-use caching
│   │   │   ├── compression.rs  # zstd with trained dictionaries
│   │   │   └── metrics.rs      # Hit rates, timing, optimization
│   │   └── tests/
│   │
│   ├── cy-ga/                  # Genetic algorithm core
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── genome.rs       # Compactification genome
│   │   │   ├── fitness.rs      # Physics fitness function
│   │   │   ├── operators.rs    # Crossover, mutation
│   │   │   ├── selection.rs    # Tournament, elitism
│   │   │   └── parallel.rs     # Rayon-based parallelism
│   │   └── tests/
│   │
│   ├── cy-meta-ga/             # Meta-optimization (GA for GA)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── cache_genome.rs # Cache strategy as genome
│   │   │   ├── arch_genome.rs  # Architecture parameters
│   │   │   └── meta_fitness.rs # Combined: result quality + speed
│   │   └── tests/
│   │
│   └── cy-verified/            # Formally verified core (Aeneas)
│       ├── src/
│       │   ├── lib.rs
│       │   ├── intersection_verified.rs
│       │   └── solver_verified.rs
│       ├── aeneas/             # Lean proofs
│       └── tests/
│
├── bins/
│   ├── cy-search/              # Main GA search binary
│   ├── cy-validate/            # Validate against papers
│   └── cy-bench/               # Benchmarking suite
│
├── benches/                    # Criterion benchmarks
│   ├── intersection.rs
│   ├── triangulation.rs
│   └── kklt_solver.rs
│
└── data/
    ├── mcallister/             # McAllister validation data
    ├── cicoli/                 # Cicoli validation data
    └── kreuzer-skarke/         # KS database subset
```

---

## Crate Dependency Graph

```
                    ┌─────────────┐
                    │  cy-cache   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   cy-core     │  │  cy-moduli    │  │ cy-cosmology  │
│  (primitives) │  │ (stabilization│  │  (evolution)  │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │    cy-ga    │
                    │   (search)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  cy-meta-ga │
                    │ (self-opt)  │
                    └─────────────┘
```

---

## Caching Architecture

### Strategy Enum

```rust
#[derive(Clone, Debug)]
pub enum CacheStrategy {
    /// No caching - pure computation
    None,

    /// In-memory with eviction policy
    Memory {
        max_bytes: usize,
        policy: EvictionPolicy,  // LRU, LFU, ARC
    },

    /// Disk-backed with optional memory layer
    Disk {
        path: PathBuf,
        max_bytes: u64,
        compression: Compression,
        index: IndexStrategy,
    },

    /// Tiered: RAM → NVMe → HDD
    Tiered {
        ram: Box<CacheStrategy>,
        nvme: Box<CacheStrategy>,
        hdd: Box<CacheStrategy>,
    },

    /// Bloom filter gated (cache on second use)
    BloomGated {
        inner: Box<CacheStrategy>,
        bloom_bits: usize,
        hash_count: u8,
    },

    /// Load from precomputed database
    Precomputed {
        db_path: PathBuf,
        fallback: Box<CacheStrategy>,
    },
}
```

### Compression Options

```rust
pub enum Compression {
    None,
    Zstd { level: i32 },
    ZstdTrained {
        level: i32,
        dict: Arc<zstd::dict::EncoderDictionary<'static>>,
    },
}
```

### Index Strategies

```rust
pub enum IndexStrategy {
    /// No index - linear scan
    None,

    /// Hash map on serialized key
    HashMap,

    /// B-tree for range queries
    BTree,

    /// Parameter-linked: param1 → param2 → ... → value
    LinkedParams {
        param_order: Vec<String>,
    },

    /// Blocked storage for related data
    Blocked {
        block_size: usize,  // e.g., 256MB
        clustering: ClusteringStrategy,
    },
}
```

### Cache Metrics

```rust
pub struct CacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,
    pub avg_read_ns: AtomicU64,
    pub avg_write_ns: AtomicU64,
    pub compression_ratio: AtomicU64,  // Fixed-point
}
```

### Auto-Optimization

```rust
/// Meta-GA genome for cache optimization
pub struct CacheGenome {
    /// Strategy for each function
    pub strategies: HashMap<FunctionId, CacheStrategy>,

    /// Resource constraints
    pub ram_budget: usize,
    pub nvme_budget: u64,
    pub hdd_budget: u64,
}

/// Fitness function for cache optimization
pub fn cache_fitness(genome: &CacheGenome, workload: &Workload) -> f64 {
    let throughput = measure_throughput(genome, workload);
    let memory_efficiency = genome.ram_budget as f64 / actual_ram_used;
    let hit_rate = total_hits as f64 / total_accesses as f64;

    // Combined fitness
    0.5 * throughput + 0.3 * hit_rate + 0.2 * memory_efficiency
}
```

---

## Validation Strategy

### Two-Paper Validation

1. **McAllister (arXiv:2107.09064)** - Static Λ
   - 5 explicit examples with published data
   - Target: V₀ = -5.5 × 10⁻²⁰³ for 4-214-647
   - Test: `cy-validate mcallister`

2. **Cicoli (arXiv:2407.03405)** - Quintessence
   - K3-fibred CY with explicit parameters
   - Target: Λ⁴ ≈ 10⁻¹²⁰, f ≈ 0.085 Mₚₗ
   - Test: `cy-validate cicoli`

### Implementation Order

Only implement functions **as needed** for validation:

```
Phase 1: McAllister KKLT
├── cy-core::polytope (from points.dat)
├── cy-core::triangulation (FRST)
├── cy-core::intersection (κ_ijk)
├── cy-moduli::racetrack (g_s, W₀)
├── cy-moduli::kklt (τ = c_i solver)
└── cy-core::kahler (V_string)

Phase 2: Cicoli Quintessence
├── cy-moduli::lvs (LVS stabilization)
├── cy-cosmology::friedmann
├── cy-cosmology::quintessence
└── cy-cosmology::cpl
```

---

## Formal Verification Plan

### Critical Functions for Aeneas

1. **Intersection number computation** (`κ_ijk`)
   - Must be symmetric: κ_ijk = κ_jik = κ_kij
   - Integer output for integer inputs
   - Topological invariant

2. **KKLT solver convergence**
   - τ(t) = c_i must be satisfied to tolerance
   - V > 0 constraint
   - Jacobian properties

3. **Volume computation**
   - V = (1/6) κ_ijk t^i t^j t^k ≥ 0 in Kähler cone
   - BBHL correction sign

### Aeneas Workflow

```bash
# Extract Rust to LLBC
aeneas -backend lean cy-verified/src/intersection_verified.rs

# Generate Lean definitions
# Prove properties in Lean
# Back-annotate Rust with verified guarantees
```

---

## Benchmarking Suite

### Criterion Benchmarks

```rust
// benches/intersection.rs
fn bench_intersection(c: &mut Criterion) {
    let polytope = load_mcallister_polytope("4-214-647");

    c.bench_function("kappa_ijk_h11_214", |b| {
        b.iter(|| compute_intersection_numbers(&polytope))
    });
}

// benches/kklt_solver.rs
fn bench_kklt_solver(c: &mut Criterion) {
    let (kappa, c_i) = load_kklt_inputs("5-81-3213");

    c.bench_function("kklt_solver_h11_81", |b| {
        b.iter(|| solve_kklt(&kappa, &c_i, SolverParams::default()))
    });
}
```

### Performance Targets

| Function | Current (Python) | Target (Rust) |
|----------|------------------|---------------|
| κ_ijk (h¹¹=214) | ~2s | <50ms |
| KKLT solver (h¹¹=81) | ~13s | <500ms |
| GV invariants | ~16s | <2s |
| Full pipeline | ~45s | <5s |

---

## Resource Management

### Configuration

```toml
# cy-config.toml
[resources]
ram_gb = 16
nvme_gb = 512
hdd_gb = 1024

[cache]
# Evolve these via meta-GA
strategy = "tiered"
bloom_filter_bits = 1_000_000
zstd_level = 3

[ga]
population = 1000
generations = 50000
parallel_workers = 8
```

### Automatic Resource Discovery

```rust
pub struct SystemResources {
    pub ram_total: u64,
    pub ram_available: u64,
    pub nvme_paths: Vec<(PathBuf, u64)>,  // (path, free_bytes)
    pub hdd_paths: Vec<(PathBuf, u64)>,
    pub cpu_cores: usize,
}

impl SystemResources {
    pub fn detect() -> Self { ... }

    pub fn suggest_config(&self) -> CacheConfig {
        // Heuristics for initial config
        // Meta-GA will optimize from here
    }
}
```

---

## Meta-GA: Self-Optimization

### Architecture Genome

```rust
pub struct ArchitectureGenome {
    // Cache parameters (per function)
    pub cache_strategies: Vec<CacheStrategy>,

    // Solver parameters
    pub kklt_n_steps: u32,
    pub kklt_tolerance: f64,
    pub newton_max_iter: u32,

    // GA parameters
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism_fraction: f64,
}
```

### Meta-Fitness Function

```rust
pub fn meta_fitness(
    arch: &ArchitectureGenome,
    physics_results: &[PhysicsResult],
    timing: &TimingStats,
) -> f64 {
    // How good are the physics results?
    let result_quality = physics_results.iter()
        .map(|r| r.fitness)
        .max()
        .unwrap_or(0.0);

    // How fast did we improve?
    let improvement_rate = timing.fitness_improvement_per_second;

    // How quickly did we find good results?
    let discovery_speed = 1.0 / timing.time_to_first_good_result;

    // Combined meta-fitness
    0.4 * result_quality
    + 0.3 * improvement_rate
    + 0.3 * discovery_speed
}
```

---

## Migration Path

### Phase 1: Core Math (2-3 weeks)
- Port polytope/triangulation from CYTools
- Implement κ_ijk computation
- Validate against McAllister

### Phase 2: KKLT Solver (1-2 weeks)
- Predictor-corrector in Rust
- Multi-branch exploration
- Validate V_string

### Phase 3: Cosmology (1-2 weeks)
- Friedmann ODE solver
- Quintessence evolution
- CPL fitting

### Phase 4: Caching (1 week)
- Basic tiered cache
- Metrics collection

### Phase 5: GA (1-2 weeks)
- Port GA from existing Rust code
- Integrate with new physics

### Phase 6: Meta-GA (ongoing)
- Self-optimization
- Continuous improvement

---

## Success Criteria

1. **Correctness**: Reproduce both papers exactly
2. **Performance**: 10-100× faster than Python
3. **Reliability**: 100% test coverage, formal verification
4. **Usability**: Clean API, good documentation
5. **Citability**: Published, documented, maintained

---

*This document is the north star for the Rust CY toolkit development.*
