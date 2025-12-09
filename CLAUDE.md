# String Theory Landscape Explorer

## What This Does
Genetic algorithm searching through Calabi-Yau compactifications to find configurations that reproduce Standard Model physics (gauge couplings, generations, cosmological constant).

## Local Setup (Mac)
- Python: Homebrew 3.11 (`/opt/homebrew/opt/python@3.11/bin/python3.11`)
  - NOTE: mise/asdf Python has broken PyO3 embedding paths, use homebrew
- Package manager: `uv` (never pip/venv directly)
- numpy<2 required (cytools/ortools need numpy 1.x)
- CYTools: `../cytools_source` (compiled, installed via `uv pip install`)
- cymyc: `../cymyc_source` (compiled, installed via `uv pip install`)
- Polytopes: `polytopes_three_gen.jsonl` (3.4GB, 12.2M three-gen candidates)

### First-time setup:
```bash
# Create venv with homebrew Python 3.11
uv venv --python /opt/homebrew/opt/python@3.11/bin/python3.11 .venv
uv sync
uv pip install ../cytools_source ../cymyc_source

# Build with correct Python
PYO3_PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 cargo build --release --bin search
```

## Server Setup
- Host: `root@10.5.7.33`
- Project: `/root/string_theory` (NVMe)
- Data: `/root/string_theory/data/` (NVMe) - filtered polytopes, SQLite, ChromaDB
- Raw polytopes: `/data/polytopes/parquet/` (ZFS spinning disk) - 400M raw polytopes
- CYTools: `/root/cytools_source`
- cymyc: `/root/cymyc_source`
- Venv: `/root/string_theory/venv`

### Deploy
```bash
ansible-playbook -i ansible/inventory.yml ansible/playbook.yml
```

### Server Services
```bash
# Web dashboard (auto-starts, port 3000)
systemctl status string-theory-web
systemctl restart string-theory-web

# Search workers (stopped by default, 2 instances configured)
systemctl start string-theory-search.target   # Start 2 workers
systemctl stop string-theory-search.target    # Stop all workers
systemctl start string-theory-search@3        # Start additional worker
journalctl -u string-theory-search@1 -f       # Follow worker logs
```

## Build & Run
```bash
# Build (must set PYO3_PYTHON)
PYO3_PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 cargo build --release --bin search

# Run locally (must set VIRTUAL_ENV for PyO3 to find packages)
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search

# Run on server
./target/release/search -c config.server.toml

# With options
VIRTUAL_ENV=$(pwd)/.venv ./target/release/search -t 3600 -g 50000 --population 1000
```

## Key Files
```
src/bin/search.rs    - main binary with CLI (clap)
src/searcher.rs      - GA: GaConfig, LandscapeSearcher, Individual
src/physics.rs       - PyO3 bridge to Python, Compactification genome
src/constants.rs     - target physics values (α_em, α_s, sin²θ_W, Λ, etc.)
physics_bridge.py    - CYTools + cymyc computations (NO FALLBACKS)
config.toml          - local config
config.server.toml   - server config (/data/polytopes path)
```

## Physics Bridge
- Uses CYTools for polytope analysis, triangulations, intersection numbers
- Uses cymyc (JAX) for numerical CY metrics
- Crashes if not available - no fallbacks ever
- Computes: gauge couplings, cosmological constant, generation count

## Target Constants (what we're matching)
- α_em = 7.297e-3 (fine structure)
- α_s = 0.118 (strong coupling at M_Z)
- sin²θ_W = 0.231 (Weinberg angle)
- N_gen = 3 (fermion generations)
- Λ = 2.888e-122 (cosmological constant in Planck units)

## Genome Structure (Compactification)
- polytope_id: index into Kreuzer-Skarke database
- kahler_moduli: h11 real positive values (cycle volumes)
- complex_moduli: h21 complex values
- flux_f, flux_h: integer flux quanta
- g_s: string coupling

## Three-Generation Filter
Polytopes filtered by |h11 - h21| = 3 (gives 3 fermion generations).
473M polytopes → 12.2M candidates.

## Scripts
- `download_all_polytopes.py` - download Kreuzer-Skarke from HuggingFace
- `filter_three_gen.py` - filter for |h11 - h21| = 3, outputs JSONL

## Output
Results saved to `results/run_XX/`:
- `best_XXXXXX.json` - best compactification found
- `state_*.json` - full GA state for resume
- `cluster_state.json` - polytope clustering data

## Ansible
Inventory: `ansible/inventory.yml` (set string_theory host)
Playbook installs: Rust, Python venv, PALP, CYTools, cymyc
