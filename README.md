# String Theory Landscape Explorer

A genetic algorithm to search the string theory landscape for Calabi-Yau compactifications that reproduce Standard Model physics.

## What This Does

Searches through ~473 million 4D reflexive polytopes (Kreuzer-Skarke database) to find Calabi-Yau manifolds whose compactification could give:
- **3 generations** of fermions (quarks + leptons)
- **SU(3) × SU(2) × U(1)** gauge group
- Correct particle masses and mixing angles

## Quick Start

```bash
# 1. Download polytope data (~15.8 GB)
python download_all_polytopes.py --output-dir polytope_data

# 2. Filter to 3-generation candidates
python filter_three_gen.py --parquet-dir polytope_data --output polytopes_three_gen.json

# 3. Build and run the GA
cargo build --release --bin real_physics
./target/release/real_physics
```

## Physics Background

### The 3-Generation Constraint

The number of fermion generations is determined by the Euler characteristic:
```
χ = 2(h11 - h21)
```

For 3 generations: **|h11 - h21| = 3**

This constraint alone filters 473M polytopes down to a much smaller candidate set.

### Hodge Numbers

- **h11**: Kähler moduli (shape deformations)
- **h21**: Complex structure moduli

Small Hodge numbers are preferred for realistic model building. Known working examples:
- (h11, h21) = (1, 4) → 3 generations, can break to Standard Model
- (h11, h21) = (1, 1) → Minimal case

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Kreuzer-Skarke  │────►│  3-Gen Filter    │────►│  Genetic         │
│  Database        │     │  |h11-h21| = 3   │     │  Algorithm       │
│  (473M polytopes)│     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                         ┌──────────────────┐              │
                         │  PALP            │◄─────────────┘
                         │  (Physics Eval)  │
                         └──────────────────┘
```

### Adaptive Selection (Planned)

Instead of random polytope selection, the GA learns which regions produce good results:

1. **Cluster polytopes** by (h11, h21, vertex structure)
2. **Track fitness** per cluster
3. **Weight selection** toward promising clusters
4. **Learn mutation patterns** that improve fitness

See [NOTES.md](NOTES.md) for detailed design.

## Files

| File | Description |
|------|-------------|
| `download_all_polytopes.py` | Download parquet files from HuggingFace |
| `filter_three_gen.py` | Filter to 3-generation candidates |
| `src/bin/real_physics.rs` | Main GA binary |
| `src/real_genetic.rs` | Genetic algorithm implementation |
| `src/physics.rs` | PALP interface for physics calculations |
| `ansible/` | Server deployment automation |
| `NOTES.md` | Technical design notes |

## Requirements

- Rust 1.70+
- Python 3.10+
- PALP (Polytope Analysis Lattice Package)
- ~20GB disk space for polytope data

### Python Dependencies

```bash
pip install numpy pyarrow requests
```

### PALP Installation

```bash
git clone https://gitlab.com/stringstuwien/PALP.git
cd PALP
make
```

## Server Deployment

Uses Ansible for automated setup:

```bash
cd ansible
ansible-playbook -i inventory.yml playbook.yml
```

This sets up:
- Rust toolchain
- Python venv with dependencies
- PALP from source
- Polytope data download (background)

## Data Sources

- **Polytopes**: [calabi-yau-data/polytopes-4d](https://huggingface.co/datasets/calabi-yau-data/polytopes-4d) on HuggingFace
- **PALP**: [GitLab](https://gitlab.com/stringstuwien/PALP)

## References

- Kreuzer & Skarke, "Complete classification of reflexive polyhedra in four dimensions" [arXiv:hep-th/0002240](https://arxiv.org/abs/hep-th/0002240)
- Braun et al., "A three-generation Calabi-Yau manifold with small Hodge numbers" [arXiv:0909.3947](https://arxiv.org/abs/0909.3947)
- He, "The Calabi-Yau Landscape" [arXiv:1812.02893](https://arxiv.org/abs/1812.02893)

## License

MIT
