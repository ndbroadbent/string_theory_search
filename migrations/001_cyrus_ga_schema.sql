-- Migration 001: cyrus-ga schema
-- Fresh schema for the cyrus-ga string-landscape genetic algorithm.
-- The old meta-GA pipeline (evaluations, meta_algorithms, meta_fitness,
-- meta_state, runs, workers) is dead and its tables are not recreated.

-- Per-polytope aggregate state, keyed by pool name (e.g. "h21_4_0")
CREATE TABLE IF NOT EXISTS polytopes (
    id TEXT PRIMARY KEY,
    h11 INTEGER,
    h21 INTEGER,
    favorable INTEGER,
    q_d3 REAL,
    vertices TEXT, -- JSON array of 4D integer points
    rounds INTEGER DEFAULT 0,
    evals INTEGER DEFAULT 0,
    valid_seen INTEGER DEFAULT 0,
    best_fitness REAL,
    updated_at INTEGER
);

-- Best-candidate improvements per polytope (from candidates.jsonl)
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY,
    polytope_id TEXT NOT NULL,
    k TEXT NOT NULL,  -- JSON array of flux quanta K
    m TEXT NOT NULL,  -- JSON array of flux quanta M
    fitness REAL,
    tier TEXT,        -- 'valid' or free-text failure reason
    q_flux REAL,
    g_s REAL,
    w0 REAL,          -- |W_0| flux superpotential (NOT the CPL w0!)
    log10_abs_v0 REAL,
    cpl_w0 REAL,
    cpl_wa REAL,
    round INTEGER,
    ts INTEGER,
    UNIQUE(polytope_id, k, m)
);

-- GA round log (from rounds.jsonl)
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY,
    round INTEGER,
    polytope_id TEXT,
    evals INTEGER,      -- cumulative evals for this polytope
    valid INTEGER,
    best_here REAL,
    global_best REAL,
    live INTEGER,
    ts INTEGER
);

-- Ingester bookkeeping (byte offsets, last ingest timestamp, ...)
CREATE TABLE IF NOT EXISTS run_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Geometric heuristics per polytope (computed by the `heuristics` binary).
-- Same metric columns as before, but keyed by TEXT pool name.
CREATE TABLE IF NOT EXISTS heuristics (
    polytope_id TEXT PRIMARY KEY,
    h11 INTEGER,
    h21 INTEGER,
    vertex_count INTEGER,
    sphericity REAL,
    inertia_isotropy REAL,
    chirality_optimal REAL,
    chirality_x REAL,
    chirality_y REAL,
    chirality_z REAL,
    chirality_w REAL,
    handedness_det REAL,
    symmetry_x REAL,
    symmetry_y REAL,
    symmetry_z REAL,
    symmetry_w REAL,
    flatness_3d REAL,
    flatness_2d REAL,
    intrinsic_dim_estimate REAL,
    spikiness REAL,
    max_exposure REAL,
    conformity_ratio REAL,
    distance_kurtosis REAL,
    loner_score REAL,
    coord_mean REAL,
    coord_median REAL,
    coord_std REAL,
    coord_skewness REAL,
    coord_kurtosis REAL,
    shannon_entropy REAL,
    joint_entropy REAL,
    compression_ratio REAL,
    sorted_compression_ratio REAL,
    sort_compression_gain REAL,
    phi_ratio_count INTEGER,
    fibonacci_count INTEGER,
    zero_count INTEGER,
    one_count INTEGER,
    prime_count INTEGER,
    outlier_score REAL,
    outlier_max_zscore REAL,
    outlier_max_dim TEXT,
    outlier_count_2sigma INTEGER,
    outlier_count_3sigma INTEGER,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Manual / pipeline verification results for promising candidates
CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY,
    polytope_id TEXT,
    k TEXT,
    m TEXT,
    status TEXT,
    blocker TEXT,
    details TEXT,
    ts INTEGER
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_candidates_polytope ON candidates(polytope_id);
CREATE INDEX IF NOT EXISTS idx_candidates_fitness ON candidates(fitness DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_tier ON candidates(tier);
CREATE INDEX IF NOT EXISTS idx_rounds_polytope ON rounds(polytope_id);
CREATE INDEX IF NOT EXISTS idx_rounds_ts ON rounds(ts);
CREATE INDEX IF NOT EXISTS idx_polytopes_best_fitness ON polytopes(best_fitness DESC);
CREATE INDEX IF NOT EXISTS idx_verifications_polytope ON verifications(polytope_id);
