-- Migration 002: Meta-GA Schema
-- Tables for meta-genetic algorithm that evolves search strategies
-- The database itself IS the meta-run - no separate tracking table needed

-- Meta-algorithm definitions (the "genome" of each search strategy)
CREATE TABLE IF NOT EXISTS meta_algorithms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,  -- Optional human-readable name

    -- Version of the algorithm schema (v1, v2, etc.)
    -- When we add new parameters, bump version so we know which params exist
    version INTEGER DEFAULT 1,

    -- Locking mechanism: process writes both, reads back, if matches it owns this
    locked_by_pid INTEGER,
    last_heartbeat_at TEXT,  -- Updated every 20s, stale after 60s

    -- Status
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    completed_at TEXT,
    trials_required INTEGER DEFAULT 10,  -- How many trials needed before complete

    -- Which meta-generation this algorithm belongs to
    meta_generation INTEGER DEFAULT 0,

    -- === THE CORE: Feature weights for polytope search ===
    -- JSON object with ~50 keys matching heuristic columns
    -- e.g. {"sphericity": 1.5, "chirality_optimal": 2.0, "outlier_score": 0.5, ...}
    feature_weights TEXT NOT NULL,

    -- === Search strategy ===
    similarity_radius REAL DEFAULT 0.5,     -- How wide to search around good polytopes
    interpolation_weight REAL DEFAULT 0.5,  -- Balance: 0=pure similarity, 1=pure path-walking

    -- === GA parameters ===
    population_size INTEGER DEFAULT 50,
    max_generations INTEGER DEFAULT 10,
    mutation_rate REAL DEFAULT 0.4,
    mutation_strength REAL DEFAULT 0.35,
    crossover_rate REAL DEFAULT 0.85,
    tournament_size INTEGER DEFAULT 5,
    elite_count INTEGER DEFAULT 10,

    -- === Polytope switching ===
    polytope_patience INTEGER DEFAULT 5,      -- Gens before considering switch
    switch_threshold REAL DEFAULT 0.01,       -- Min improvement to stay
    switch_probability REAL DEFAULT 0.1,      -- Random switch chance

    -- === Fitness weights ===
    cc_weight REAL DEFAULT 10.0,              -- Cosmological constant importance

    -- Lineage tracking
    parent_id INTEGER,  -- Which algorithm this was mutated/crossed from

    created_at TEXT DEFAULT (datetime('now')),

    FOREIGN KEY (parent_id) REFERENCES meta_algorithms(id)
);

-- Meta-algorithm performance tracking
-- Each row = one "trial run" of an algorithm
CREATE TABLE IF NOT EXISTS meta_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm_id INTEGER NOT NULL,
    run_id TEXT,  -- Links to runs table

    -- Trial parameters
    generations_run INTEGER,

    -- Performance metrics (for computing meta-fitness)
    initial_fitness REAL,
    final_fitness REAL,
    fitness_improvement REAL,  -- final - initial
    improvement_rate REAL,  -- fitness_improvement / generations_run

    -- Area under fitness curve (weighted by time)
    fitness_auc REAL,

    -- Best cosmological constant achieved (absolute value, closer to target is better)
    best_cc_log_error REAL,  -- log10(|computed_cc / target_cc|)

    -- Physics success rate
    physics_success_rate REAL,

    -- Diversity metrics
    unique_polytopes_tried INTEGER,

    started_at TEXT,
    ended_at TEXT,

    FOREIGN KEY (algorithm_id) REFERENCES meta_algorithms(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Aggregate meta-fitness per algorithm
-- Updated after each trial
CREATE TABLE IF NOT EXISTS meta_fitness (
    algorithm_id INTEGER PRIMARY KEY,

    trial_count INTEGER DEFAULT 0,

    -- Aggregated performance
    mean_improvement_rate REAL,
    best_improvement_rate REAL,
    mean_fitness_auc REAL,
    best_final_fitness REAL,

    -- Cosmological constant performance
    best_cc_log_error REAL,
    mean_cc_log_error REAL,

    -- Combined meta-fitness score (computed from above)
    meta_fitness REAL,

    updated_at TEXT DEFAULT (datetime('now')),

    FOREIGN KEY (algorithm_id) REFERENCES meta_algorithms(id)
);

-- Feature importance learned from evaluations
-- Which heuristics correlate with good fitness?
CREATE TABLE IF NOT EXISTS feature_importance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    feature_name TEXT NOT NULL,

    -- Correlation with fitness
    fitness_correlation REAL,

    -- Correlation with cosmological constant accuracy
    cc_correlation REAL,

    -- Importance from gradient boosting or similar
    importance_score REAL,

    -- Sample size this was computed from
    sample_count INTEGER,

    updated_at TEXT DEFAULT (datetime('now'))
);

-- Global meta-state (single row, updated atomically)
CREATE TABLE IF NOT EXISTS meta_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Only one row ever
    current_generation INTEGER DEFAULT 0,
    algorithms_per_generation INTEGER DEFAULT 16,
    best_meta_fitness REAL,
    best_algorithm_id INTEGER,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Initialize meta_state with single row
INSERT OR IGNORE INTO meta_state (id) VALUES (1);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_meta_algorithms_generation ON meta_algorithms(meta_generation);
CREATE INDEX IF NOT EXISTS idx_meta_algorithms_parent ON meta_algorithms(parent_id);
CREATE INDEX IF NOT EXISTS idx_meta_algorithms_status ON meta_algorithms(status);
CREATE INDEX IF NOT EXISTS idx_meta_algorithms_heartbeat ON meta_algorithms(last_heartbeat_at);
CREATE INDEX IF NOT EXISTS idx_meta_trials_algorithm ON meta_trials(algorithm_id);
CREATE INDEX IF NOT EXISTS idx_meta_fitness_score ON meta_fitness(meta_fitness DESC);
