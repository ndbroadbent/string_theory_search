-- Migration 005: Unify runs and meta_trials into single runs table
--
-- Hierarchy:
--   Generation -> Algorithm (N per gen) -> Run (M per algo) -> Evaluation (many per run)
--
-- Before: meta_trials and runs were separate, confusing
-- After: runs is the single table for GA executions (what was meta_trials)

-- Disable foreign key constraints during migration
PRAGMA foreign_keys = OFF;

-- Step 1: Drop the old runs table (had TEXT primary key, was separate from meta_trials)
DROP TABLE IF EXISTS runs;

-- Step 2: Create new runs table from meta_trials (with INTEGER id)
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm_id INTEGER NOT NULL,
    run_number INTEGER NOT NULL DEFAULT 1,

    -- Execution
    started_at TEXT,
    ended_at TEXT,

    -- Results
    generations_run INTEGER,
    initial_fitness REAL,
    final_fitness REAL,
    fitness_improvement REAL,
    improvement_rate REAL,
    fitness_auc REAL,
    best_cc_log_error REAL,
    physics_success_rate REAL,
    unique_polytopes_tried INTEGER,

    FOREIGN KEY (algorithm_id) REFERENCES meta_algorithms(id)
);

-- Step 3: Insert from meta_trials, extracting run_number from run_id
-- run_id format: "meta_{algo_id}_{run_num}" -> extract last part after last underscore
-- Since run_id like "meta_17_2", the last char(s) after final _ is the run number
INSERT INTO runs (
    id, algorithm_id, run_number, started_at, ended_at,
    generations_run, initial_fitness, final_fitness,
    fitness_improvement, improvement_rate, fitness_auc,
    best_cc_log_error, physics_success_rate, unique_polytopes_tried
)
SELECT
    id,
    algorithm_id,
    -- Extract run number: for "meta_17_2", get everything after the last underscore
    CAST(SUBSTR(run_id, LENGTH(run_id) - LENGTH(run_id) + LENGTH(REPLACE(run_id, '_', '')) + 3) AS INTEGER),
    started_at,
    ended_at,
    generations_run,
    initial_fitness,
    final_fitness,
    fitness_improvement,
    improvement_rate,
    fitness_auc,
    best_cc_log_error,
    physics_success_rate,
    unique_polytopes_tried
FROM meta_trials;

-- Simpler approach: just count trials per algorithm and assign run_number
-- This works because trials were inserted in order
UPDATE runs SET run_number = (
    SELECT COUNT(*)
    FROM runs r2
    WHERE r2.algorithm_id = runs.algorithm_id
    AND r2.id <= runs.id
);

-- Step 4: Drop old meta_trials table
DROP TABLE meta_trials;

-- Step 5: Create index on runs
CREATE INDEX idx_runs_algorithm ON runs(algorithm_id);

-- Step 6: Update evaluations table - change run_id from TEXT to INTEGER
-- Create new evaluations table with INTEGER run_id
CREATE TABLE evaluations_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    polytope_id INTEGER NOT NULL,
    run_id INTEGER,
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

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (polytope_id) REFERENCES polytopes(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Copy data, setting run_id to NULL (old data used string IDs that don't map cleanly)
INSERT INTO evaluations_new (
    id, polytope_id, run_id, generation,
    g_s, kahler_moduli, complex_moduli, flux_f, flux_h,
    fitness, alpha_em, alpha_s, sin2_theta_w, n_generations,
    cosmological_constant, success, error, created_at
)
SELECT
    id, polytope_id, NULL, generation,
    g_s, kahler_moduli, complex_moduli, flux_f, flux_h,
    fitness, alpha_em, alpha_s, sin2_theta_w, n_generations,
    cosmological_constant, success, error, created_at
FROM evaluations;

DROP TABLE evaluations;
ALTER TABLE evaluations_new RENAME TO evaluations;

CREATE INDEX idx_evaluations_polytope ON evaluations(polytope_id);
CREATE INDEX idx_evaluations_run ON evaluations(run_id);
CREATE INDEX idx_evaluations_fitness ON evaluations(fitness DESC);

-- Step 7: Rename trials_required to runs_required in meta_algorithms
ALTER TABLE meta_algorithms RENAME COLUMN trials_required TO runs_required;

-- Step 8: Rename trial_count to run_count in meta_fitness
ALTER TABLE meta_fitness RENAME COLUMN trial_count TO run_count;

-- Re-enable foreign key constraints
PRAGMA foreign_keys = ON;
