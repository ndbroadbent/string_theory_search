# String Theory Landscape Explorer

A meta-genetic algorithm to search the string theory landscape for Calabi-Yau compactifications that reproduce Standard Model physics.

## What This Does

Searches through 12.2 million three-generation polytopes from the Kreuzer-Skarke database to find Calabi-Yau manifolds whose compactification gives:
- **3 generations** of fermions (quarks + leptons)
- **SU(3) × SU(2) × U(1)** gauge group
- Correct gauge couplings (α_em, α_s, sin²θ_W)
- Small cosmological constant (Λ ≈ 10⁻¹²² Planck units)

## Key Innovation: Meta-GA

Instead of tuning hyperparameters manually, this project uses a **meta-genetic algorithm** that evolves search strategies:

```
Meta-GA evolves:
├── 50+ feature weights (which polytope properties matter?)
├── GA parameters (mutation rate, population size, etc.)
├── Search strategy (similarity radius, path interpolation)
└── Polytope switching behavior
```

Multiple workers run in parallel, each trying different evolved strategies. The strategies that find good physics faster get selected for the next meta-generation.

## Quick Start

```bash
# 1. Install dependencies
uv venv --python /opt/homebrew/opt/python@3.11/bin/python3.11 .venv
uv sync
uv pip install ../cytools_source ../cymyc_source

# 2. Build
PYO3_PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 cargo build --release --bin search

# 3. Run a worker
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search
```

For multiple parallel workers:
```bash
# Terminal 1, 2, 3, ...
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    META-GA (evolves strategies)                 │
├─────────────────────────────────────────────────────────────────┤
│  Generation 0: 16 random algorithms                             │
│  Each algorithm runs 10 trials                                  │
│  Top performers reproduce → Generation 1                        │
│  Repeat...                                                      │
├─────────────────────────────────────────────────────────────────┤
│                    INNER GA (per trial)                         │
├─────────────────────────────────────────────────────────────────┤
│  Population of compactifications on polytopes                   │
│  Evolve: Kähler moduli, complex moduli, fluxes, g_s             │
│  Fitness = match to observed physics                            │
├─────────────────────────────────────────────────────────────────┤
│                    PHYSICS (per evaluation)                     │
├─────────────────────────────────────────────────────────────────┤
│  CYTools → triangulation, intersection numbers                  │
│  cymyc → numerical CY metrics                                   │
│  KKLT → moduli stabilization, cosmological constant             │
└─────────────────────────────────────────────────────────────────┘
```

## Files

```
src/
├── lib.rs              # Module exports
├── constants.rs        # Target physics values
├── physics.rs          # PyO3 bridge to CYTools/cymyc
├── db.rs               # SQLite persistence layer
├── meta_ga.rs          # Meta-evolution functions
└── searcher.rs         # Inner GA implementation

src/bin/search/
├── main.rs             # Worker entry point
├── config.rs           # CLI args and config parsing
├── heartbeat.rs        # Background heartbeat thread
├── trial.rs            # Trial execution
└── worker.rs           # Main worker loop

migrations/
├── 001_initial_schema.sql
└── 002_meta_ga_schema.sql
```

## Configuration

Create `config.toml`:
```toml
[paths]
polytopes = "polytopes_three_gen.jsonl"
output_dir = "results"
database = "data/string_theory.db"

[meta_ga]
algorithms_per_generation = 16
trials_required = 10
```

## Requirements

- **Rust** 1.70+
- **Python** 3.11 (Homebrew recommended for PyO3)
- **CYTools** - polytope analysis
- **cymyc** - numerical CY metrics
- **~4GB** polytope data

## Documentation

- **[DOCS.md](DOCS.md)** - Comprehensive technical documentation
- **[NOTES.md](NOTES.md)** - Physics background and design notes
- **[CLAUDE.md](CLAUDE.md)** - Development instructions

## Physics Background

The project targets these Standard Model values:

| Constant | Symbol | Value |
|----------|--------|-------|
| Fine structure | α_em | 7.297 × 10⁻³ |
| Strong coupling | α_s | 0.118 |
| Weinberg angle | sin²θ_W | 0.231 |
| Generations | N_gen | 3 |
| Cosmological constant | Λ | 2.888 × 10⁻¹²² |

The three-generation constraint (|h11 - h21| = 3) filters 473M polytopes down to 12.2M candidates.

## References

- [DOCS.md](DOCS.md) for detailed architecture
- CYTools: https://github.com/LiamMcAllisterGroup/cytools
- cymyc: https://github.com/Justin-Tan/cymyc
- Kreuzer-Skarke: https://huggingface.co/datasets/calabi-yau-data/polytopes-4d

## License

MIT
