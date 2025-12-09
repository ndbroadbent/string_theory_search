-- Migration 001: Initial Schema
-- Creates core tables for unified data storage

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now')),
    description TEXT
);

-- GA run metadata
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT,
    ended_at TEXT,
    config TEXT,
    polytope_filter TEXT,
    total_generations INTEGER,
    total_evaluations INTEGER,
    best_fitness REAL,
    best_polytope_id INTEGER
);

-- Polytope metadata (cached from JSONL for fast access)
CREATE TABLE IF NOT EXISTS polytopes (
    id INTEGER PRIMARY KEY,
    h11 INTEGER NOT NULL,
    h21 INTEGER NOT NULL,
    vertex_count INTEGER NOT NULL,
    vertices TEXT NOT NULL,
    eval_count INTEGER DEFAULT 0,
    fitness_sum REAL DEFAULT 0,
    fitness_sum_sq REAL DEFAULT 0,
    fitness_min REAL,
    fitness_max REAL
);

-- Every physics evaluation ever performed
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    polytope_id INTEGER NOT NULL,
    run_id TEXT,
    generation INTEGER,
    g_s REAL,
    kahler_moduli TEXT,
    complex_moduli TEXT,
    flux_f TEXT,
    flux_h TEXT,
    fitness REAL NOT NULL,
    alpha_em REAL,
    alpha_s REAL,
    sin2_theta_w REAL,
    n_generations INTEGER,
    cosmological_constant REAL,
    success INTEGER,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (polytope_id) REFERENCES polytopes(id)
);

-- Computed heuristics per polytope
CREATE TABLE IF NOT EXISTS heuristics (
    polytope_id INTEGER PRIMARY KEY,
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
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (polytope_id) REFERENCES polytopes(id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_evaluations_polytope ON evaluations(polytope_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_fitness ON evaluations(fitness DESC);
CREATE INDEX IF NOT EXISTS idx_heuristics_outlier ON heuristics(outlier_score DESC);
CREATE INDEX IF NOT EXISTS idx_polytopes_eval_count ON polytopes(eval_count DESC);
