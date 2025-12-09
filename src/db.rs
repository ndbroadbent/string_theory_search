//! SQLite database layer for persistent storage
//!
//! Single source of truth for all evaluation data, heuristics, and run metadata.
//! Uses schema versioning with migrations to evolve safely over time.

use rusqlite::{Connection, Result, params};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use crate::physics::{Compactification, PhysicsOutput};

/// Global database connection (thread-safe singleton)
static DB_CONNECTION: OnceLock<Arc<Mutex<Connection>>> = OnceLock::new();

/// Default database path
pub const DEFAULT_DB_PATH: &str = "data/string_theory.db";

/// Get or create the database connection
pub fn get_connection() -> Arc<Mutex<Connection>> {
    DB_CONNECTION.get_or_init(|| {
        let path = std::env::var("STRING_THEORY_DB")
            .unwrap_or_else(|_| DEFAULT_DB_PATH.to_string());
        init_database(&path).expect("Failed to initialize database")
    }).clone()
}

/// Initialize database with a specific path
pub fn init_database(path: &str) -> Result<Arc<Mutex<Connection>>> {
    // Ensure parent directory exists
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let conn = Connection::open(path)?;

    // Enable foreign keys and WAL mode for better concurrency
    // busy_timeout prevents "database is locked" errors under contention
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 30000;
         PRAGMA wal_autocheckpoint = 1000;"
    )?;

    // Run migrations
    run_migrations(&conn)?;

    Ok(Arc::new(Mutex::new(conn)))
}

/// Run all pending migrations
fn run_migrations(conn: &Connection) -> Result<()> {
    // First, ensure schema_version table exists
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime('now')),
            description TEXT
        )",
        [],
    )?;

    // Get current version
    let current_version: i32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Load and apply migrations
    let migrations = get_migrations();
    for (version, description, sql) in migrations {
        if version > current_version {
            log::info!("Applying migration {}: {}", version, description);
            conn.execute_batch(sql)?;
            conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (?1, ?2)",
                params![version, description],
            )?;
        }
    }

    Ok(())
}

/// Get all migrations as (version, description, sql)
fn get_migrations() -> Vec<(i32, &'static str, &'static str)> {
    vec![
        (1, "Initial schema", include_str!("../migrations/001_initial_schema.sql")),
        (2, "Meta-GA schema", include_str!("../migrations/002_meta_ga_schema.sql")),
        (3, "Heuristics Hodge numbers", include_str!("../migrations/003_heuristics_hodge_numbers.sql")),
        (4, "RNG seed for reproducibility", include_str!("../migrations/004_rng_seed.sql")),
    ]
}

/// Insert a new run record
pub fn insert_run(
    conn: &Connection,
    run_id: &str,
    config_json: &str,
    polytope_filter_json: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO runs (id, started_at, config, polytope_filter)
         VALUES (?1, datetime('now'), ?2, ?3)",
        params![run_id, config_json, polytope_filter_json],
    )?;
    Ok(())
}

/// Update run statistics on completion
pub fn update_run_stats(
    conn: &Connection,
    run_id: &str,
    total_generations: i32,
    total_evaluations: i64,
    best_fitness: f64,
    best_polytope_id: Option<i64>,
) -> Result<()> {
    conn.execute(
        "UPDATE runs SET
            ended_at = datetime('now'),
            total_generations = ?2,
            total_evaluations = ?3,
            best_fitness = ?4,
            best_polytope_id = ?5
         WHERE id = ?1",
        params![run_id, total_generations, total_evaluations, best_fitness, best_polytope_id],
    )?;
    Ok(())
}

/// Upsert a polytope (insert on first evaluation)
pub fn upsert_polytope(
    conn: &Connection,
    polytope_id: i64,
    h11: i32,
    h21: i32,
    vertex_count: i32,
    vertices_json: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO polytopes (id, h11, h21, vertex_count, vertices)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(id) DO NOTHING",
        params![polytope_id, h11, h21, vertex_count, vertices_json],
    )?;
    Ok(())
}

/// Record an evaluation result
pub fn insert_evaluation(
    conn: &Connection,
    polytope_id: i64,
    run_id: Option<&str>,
    generation: Option<i32>,
    genome: &Compactification,
    physics: &PhysicsOutput,
    fitness: f64,
) -> Result<i64> {
    // Serialize genome fields to JSON
    let kahler_json = serde_json::to_string(&genome.kahler_moduli).unwrap_or_default();
    let complex_json = serde_json::to_string(&genome.complex_moduli).unwrap_or_default();
    let flux_f_json = serde_json::to_string(&genome.flux_f).unwrap_or_default();
    let flux_h_json = serde_json::to_string(&genome.flux_h).unwrap_or_default();

    conn.execute(
        "INSERT INTO evaluations (
            polytope_id, run_id, generation,
            g_s, kahler_moduli, complex_moduli, flux_f, flux_h,
            fitness, alpha_em, alpha_s, sin2_theta_w, n_generations,
            cosmological_constant, success, error
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        params![
            polytope_id,
            run_id,
            generation,
            genome.g_s,
            kahler_json,
            complex_json,
            flux_f_json,
            flux_h_json,
            fitness,
            physics.alpha_em,
            physics.alpha_s,
            physics.sin2_theta_w,
            physics.n_generations,
            physics.cosmological_constant,
            physics.success as i32,
            physics.error,
        ],
    )?;

    // Update polytope aggregate stats
    update_polytope_stats(conn, polytope_id, fitness)?;

    Ok(conn.last_insert_rowid())
}

/// Update polytope aggregate fitness statistics
fn update_polytope_stats(conn: &Connection, polytope_id: i64, fitness: f64) -> Result<()> {
    conn.execute(
        "UPDATE polytopes SET
            eval_count = eval_count + 1,
            fitness_sum = fitness_sum + ?2,
            fitness_sum_sq = fitness_sum_sq + (?2 * ?2),
            fitness_min = CASE
                WHEN fitness_min IS NULL THEN ?2
                ELSE MIN(fitness_min, ?2)
            END,
            fitness_max = CASE
                WHEN fitness_max IS NULL THEN ?2
                ELSE MAX(fitness_max, ?2)
            END
         WHERE id = ?1",
        params![polytope_id, fitness],
    )?;
    Ok(())
}

/// Polytope fitness statistics
#[derive(Debug, Clone)]
pub struct PolytopeFitnessStats {
    pub polytope_id: i64,
    pub eval_count: i64,
    pub fitness_mean: f64,
    pub fitness_min: f64,
    pub fitness_max: f64,
    pub fitness_variance: f64,
}

/// Get fitness statistics for a polytope
pub fn get_polytope_stats(conn: &Connection, polytope_id: i64) -> Result<Option<PolytopeFitnessStats>> {
    let result = conn.query_row(
        "SELECT id, eval_count, fitness_sum, fitness_sum_sq, fitness_min, fitness_max
         FROM polytopes WHERE id = ?1 AND eval_count > 0",
        params![polytope_id],
        |row| {
            let id: i64 = row.get(0)?;
            let count: i64 = row.get(1)?;
            let sum: f64 = row.get(2)?;
            let sum_sq: f64 = row.get(3)?;
            let min: f64 = row.get(4)?;
            let max: f64 = row.get(5)?;

            let mean = sum / count as f64;
            let variance = (sum_sq / count as f64) - (mean * mean);

            Ok(PolytopeFitnessStats {
                polytope_id: id,
                eval_count: count,
                fitness_mean: mean,
                fitness_min: min,
                fitness_max: max,
                fitness_variance: variance.max(0.0),
            })
        },
    );

    match result {
        Ok(stats) => Ok(Some(stats)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Get top polytopes by mean fitness
pub fn get_top_polytopes(conn: &Connection, limit: i32, min_evals: i32) -> Result<Vec<PolytopeFitnessStats>> {
    let mut stmt = conn.prepare(
        "SELECT id, eval_count, fitness_sum, fitness_sum_sq, fitness_min, fitness_max
         FROM polytopes
         WHERE eval_count >= ?1
         ORDER BY (fitness_sum / eval_count) DESC
         LIMIT ?2"
    )?;

    let rows = stmt.query_map(params![min_evals, limit], |row| {
        let id: i64 = row.get(0)?;
        let count: i64 = row.get(1)?;
        let sum: f64 = row.get(2)?;
        let sum_sq: f64 = row.get(3)?;
        let min: f64 = row.get(4)?;
        let max: f64 = row.get(5)?;

        let mean = sum / count as f64;
        let variance = (sum_sq / count as f64) - (mean * mean);

        Ok(PolytopeFitnessStats {
            polytope_id: id,
            eval_count: count,
            fitness_mean: mean,
            fitness_min: min,
            fitness_max: max,
            fitness_variance: variance.max(0.0),
        })
    })?;

    rows.collect()
}

/// Upsert heuristics for a polytope
pub fn upsert_heuristics(
    conn: &Connection,
    polytope_id: i64,
    heuristics: &HeuristicsData,
) -> Result<()> {
    conn.execute(
        "INSERT INTO heuristics (
            polytope_id, h11, h21, vertex_count,
            sphericity, inertia_isotropy,
            chirality_optimal, chirality_x, chirality_y, chirality_z, chirality_w, handedness_det,
            symmetry_x, symmetry_y, symmetry_z, symmetry_w,
            flatness_3d, flatness_2d, intrinsic_dim_estimate,
            spikiness, max_exposure, conformity_ratio, distance_kurtosis, loner_score,
            coord_mean, coord_median, coord_std, coord_skewness, coord_kurtosis,
            shannon_entropy, joint_entropy,
            compression_ratio, sorted_compression_ratio, sort_compression_gain,
            phi_ratio_count, fibonacci_count, zero_count, one_count, prime_count,
            outlier_score, outlier_max_zscore, outlier_max_dim, outlier_count_2sigma, outlier_count_3sigma,
            updated_at
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
            ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31,
            ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40, ?41, ?42, ?43, ?44, datetime('now')
        )
        ON CONFLICT(polytope_id) DO UPDATE SET
            h11 = excluded.h11,
            h21 = excluded.h21,
            vertex_count = excluded.vertex_count,
            sphericity = excluded.sphericity,
            inertia_isotropy = excluded.inertia_isotropy,
            chirality_optimal = excluded.chirality_optimal,
            chirality_x = excluded.chirality_x,
            chirality_y = excluded.chirality_y,
            chirality_z = excluded.chirality_z,
            chirality_w = excluded.chirality_w,
            handedness_det = excluded.handedness_det,
            symmetry_x = excluded.symmetry_x,
            symmetry_y = excluded.symmetry_y,
            symmetry_z = excluded.symmetry_z,
            symmetry_w = excluded.symmetry_w,
            flatness_3d = excluded.flatness_3d,
            flatness_2d = excluded.flatness_2d,
            intrinsic_dim_estimate = excluded.intrinsic_dim_estimate,
            spikiness = excluded.spikiness,
            max_exposure = excluded.max_exposure,
            conformity_ratio = excluded.conformity_ratio,
            distance_kurtosis = excluded.distance_kurtosis,
            loner_score = excluded.loner_score,
            coord_mean = excluded.coord_mean,
            coord_median = excluded.coord_median,
            coord_std = excluded.coord_std,
            coord_skewness = excluded.coord_skewness,
            coord_kurtosis = excluded.coord_kurtosis,
            shannon_entropy = excluded.shannon_entropy,
            joint_entropy = excluded.joint_entropy,
            compression_ratio = excluded.compression_ratio,
            sorted_compression_ratio = excluded.sorted_compression_ratio,
            sort_compression_gain = excluded.sort_compression_gain,
            phi_ratio_count = excluded.phi_ratio_count,
            fibonacci_count = excluded.fibonacci_count,
            zero_count = excluded.zero_count,
            one_count = excluded.one_count,
            prime_count = excluded.prime_count,
            outlier_score = excluded.outlier_score,
            outlier_max_zscore = excluded.outlier_max_zscore,
            outlier_max_dim = excluded.outlier_max_dim,
            outlier_count_2sigma = excluded.outlier_count_2sigma,
            outlier_count_3sigma = excluded.outlier_count_3sigma,
            updated_at = datetime('now')",
        params![
            polytope_id,
            heuristics.h11,
            heuristics.h21,
            heuristics.vertex_count,
            heuristics.sphericity,
            heuristics.inertia_isotropy,
            heuristics.chirality_optimal,
            heuristics.chirality_x,
            heuristics.chirality_y,
            heuristics.chirality_z,
            heuristics.chirality_w,
            heuristics.handedness_det,
            heuristics.symmetry_x,
            heuristics.symmetry_y,
            heuristics.symmetry_z,
            heuristics.symmetry_w,
            heuristics.flatness_3d,
            heuristics.flatness_2d,
            heuristics.intrinsic_dim_estimate,
            heuristics.spikiness,
            heuristics.max_exposure,
            heuristics.conformity_ratio,
            heuristics.distance_kurtosis,
            heuristics.loner_score,
            heuristics.coord_mean,
            heuristics.coord_median,
            heuristics.coord_std,
            heuristics.coord_skewness,
            heuristics.coord_kurtosis,
            heuristics.shannon_entropy,
            heuristics.joint_entropy,
            heuristics.compression_ratio,
            heuristics.sorted_compression_ratio,
            heuristics.sort_compression_gain,
            heuristics.phi_ratio_count,
            heuristics.fibonacci_count,
            heuristics.zero_count,
            heuristics.one_count,
            heuristics.prime_count,
            heuristics.outlier_score,
            heuristics.outlier_max_zscore,
            heuristics.outlier_max_dim,
            heuristics.outlier_count_2sigma,
            heuristics.outlier_count_3sigma,
        ],
    )?;
    Ok(())
}

/// Heuristics data structure (matches schema)
#[derive(Debug, Clone, Default)]
pub struct HeuristicsData {
    // Hodge numbers (stored directly for fast lookup)
    pub h11: Option<i32>,
    pub h21: Option<i32>,
    pub vertex_count: Option<i32>,
    // Shape metrics
    pub sphericity: Option<f64>,
    pub inertia_isotropy: Option<f64>,
    pub chirality_optimal: Option<f64>,
    pub chirality_x: Option<f64>,
    pub chirality_y: Option<f64>,
    pub chirality_z: Option<f64>,
    pub chirality_w: Option<f64>,
    pub handedness_det: Option<f64>,
    pub symmetry_x: Option<f64>,
    pub symmetry_y: Option<f64>,
    pub symmetry_z: Option<f64>,
    pub symmetry_w: Option<f64>,
    pub flatness_3d: Option<f64>,
    pub flatness_2d: Option<f64>,
    pub intrinsic_dim_estimate: Option<f64>,
    pub spikiness: Option<f64>,
    pub max_exposure: Option<f64>,
    pub conformity_ratio: Option<f64>,
    pub distance_kurtosis: Option<f64>,
    pub loner_score: Option<f64>,
    pub coord_mean: Option<f64>,
    pub coord_median: Option<f64>,
    pub coord_std: Option<f64>,
    pub coord_skewness: Option<f64>,
    pub coord_kurtosis: Option<f64>,
    pub shannon_entropy: Option<f64>,
    pub joint_entropy: Option<f64>,
    pub compression_ratio: Option<f64>,
    pub sorted_compression_ratio: Option<f64>,
    pub sort_compression_gain: Option<f64>,
    pub phi_ratio_count: Option<i32>,
    pub fibonacci_count: Option<i32>,
    pub zero_count: Option<i32>,
    pub one_count: Option<i32>,
    pub prime_count: Option<i32>,
    pub outlier_score: Option<f64>,
    pub outlier_max_zscore: Option<f64>,
    pub outlier_max_dim: Option<String>,
    pub outlier_count_2sigma: Option<i32>,
    pub outlier_count_3sigma: Option<i32>,
}

impl HeuristicsData {
    /// Convert to a HashMap<String, f64> for weighted distance computation
    /// Only includes non-None numeric values
    pub fn to_map(&self) -> std::collections::HashMap<String, f64> {
        use std::collections::HashMap;
        let mut map = HashMap::new();

        // Helper macro to insert Option<f64> values
        macro_rules! insert_f64 {
            ($map:expr, $($name:ident),+) => {
                $(
                    if let Some(v) = self.$name {
                        $map.insert(stringify!($name).to_string(), v);
                    }
                )+
            };
        }

        // Helper macro to insert Option<i32> values as f64
        macro_rules! insert_i32 {
            ($map:expr, $($name:ident),+) => {
                $(
                    if let Some(v) = self.$name {
                        $map.insert(stringify!($name).to_string(), v as f64);
                    }
                )+
            };
        }

        insert_f64!(map,
            sphericity, inertia_isotropy,
            chirality_optimal, chirality_x, chirality_y, chirality_z, chirality_w, handedness_det,
            symmetry_x, symmetry_y, symmetry_z, symmetry_w,
            flatness_3d, flatness_2d, intrinsic_dim_estimate,
            spikiness, max_exposure, conformity_ratio, distance_kurtosis, loner_score,
            coord_mean, coord_median, coord_std, coord_skewness, coord_kurtosis,
            shannon_entropy, joint_entropy,
            compression_ratio, sorted_compression_ratio, sort_compression_gain,
            outlier_score, outlier_max_zscore
        );

        insert_i32!(map,
            phi_ratio_count, fibonacci_count, zero_count, one_count, prime_count,
            outlier_count_2sigma, outlier_count_3sigma
        );

        map
    }
}

/// Get heuristics for a polytope
pub fn get_heuristics(conn: &Connection, polytope_id: i64) -> Result<Option<HeuristicsData>> {
    let result = conn.query_row(
        "SELECT * FROM heuristics WHERE polytope_id = ?1",
        params![polytope_id],
        |row| {
            Ok(HeuristicsData {
                h11: row.get("h11")?,
                h21: row.get("h21")?,
                vertex_count: row.get("vertex_count")?,
                sphericity: row.get("sphericity")?,
                inertia_isotropy: row.get("inertia_isotropy")?,
                chirality_optimal: row.get("chirality_optimal")?,
                chirality_x: row.get("chirality_x")?,
                chirality_y: row.get("chirality_y")?,
                chirality_z: row.get("chirality_z")?,
                chirality_w: row.get("chirality_w")?,
                handedness_det: row.get("handedness_det")?,
                symmetry_x: row.get("symmetry_x")?,
                symmetry_y: row.get("symmetry_y")?,
                symmetry_z: row.get("symmetry_z")?,
                symmetry_w: row.get("symmetry_w")?,
                flatness_3d: row.get("flatness_3d")?,
                flatness_2d: row.get("flatness_2d")?,
                intrinsic_dim_estimate: row.get("intrinsic_dim_estimate")?,
                spikiness: row.get("spikiness")?,
                max_exposure: row.get("max_exposure")?,
                conformity_ratio: row.get("conformity_ratio")?,
                distance_kurtosis: row.get("distance_kurtosis")?,
                loner_score: row.get("loner_score")?,
                coord_mean: row.get("coord_mean")?,
                coord_median: row.get("coord_median")?,
                coord_std: row.get("coord_std")?,
                coord_skewness: row.get("coord_skewness")?,
                coord_kurtosis: row.get("coord_kurtosis")?,
                shannon_entropy: row.get("shannon_entropy")?,
                joint_entropy: row.get("joint_entropy")?,
                compression_ratio: row.get("compression_ratio")?,
                sorted_compression_ratio: row.get("sorted_compression_ratio")?,
                sort_compression_gain: row.get("sort_compression_gain")?,
                phi_ratio_count: row.get("phi_ratio_count")?,
                fibonacci_count: row.get("fibonacci_count")?,
                zero_count: row.get("zero_count")?,
                one_count: row.get("one_count")?,
                prime_count: row.get("prime_count")?,
                outlier_score: row.get("outlier_score")?,
                outlier_max_zscore: row.get("outlier_max_zscore")?,
                outlier_max_dim: row.get("outlier_max_dim")?,
                outlier_count_2sigma: row.get("outlier_count_2sigma")?,
                outlier_count_3sigma: row.get("outlier_count_3sigma")?,
            })
        },
    );

    match result {
        Ok(h) => Ok(Some(h)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Get total evaluation count
pub fn get_total_evaluations(conn: &Connection) -> Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM evaluations", [], |row| row.get(0))
}

// =============================================================================
// Meta-GA Functions
// =============================================================================

/// Current algorithm schema version
pub const META_ALGORITHM_VERSION: i32 = 1;

/// Meta-algorithm genome (evolvable search strategy parameters)
/// The core is `feature_weights` - ~50 weights for heuristic-based polytope selection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetaAlgorithm {
    pub id: Option<i64>,
    pub name: Option<String>,
    pub version: i32,

    // === THE CORE: Feature weights for polytope search ===
    // JSON object with ~50 keys matching heuristic columns
    pub feature_weights: String,

    // === Search strategy ===
    pub similarity_radius: f64,      // How wide to search around good polytopes
    pub interpolation_weight: f64,   // Balance: 0=pure similarity, 1=pure path-walking

    // === GA parameters ===
    pub population_size: i32,
    pub max_generations: i32,
    pub mutation_rate: f64,
    pub mutation_strength: f64,
    pub crossover_rate: f64,
    pub tournament_size: i32,
    pub elite_count: i32,

    // === Polytope switching ===
    pub polytope_patience: i32,
    pub switch_threshold: f64,
    pub switch_probability: f64,

    // === Fitness weights ===
    pub cc_weight: f64,

    // === Lineage ===
    pub parent_id: Option<i64>,
    pub meta_generation: i32,

    // === Trials ===
    pub trials_required: i32,

    // === RNG seed for reproducibility ===
    pub rng_seed: u64,
}

impl Default for MetaAlgorithm {
    fn default() -> Self {
        Self {
            id: None,
            name: None,
            version: META_ALGORITHM_VERSION,
            feature_weights: crate::meta_ga::default_feature_weights_json(),
            similarity_radius: 0.5,
            interpolation_weight: 0.5,
            population_size: 50,
            max_generations: 10,
            mutation_rate: 0.4,
            mutation_strength: 0.35,
            crossover_rate: 0.85,
            tournament_size: 5,
            elite_count: 10,
            polytope_patience: 5,
            switch_threshold: 0.01,
            switch_probability: 0.1,
            cc_weight: 10.0,
            parent_id: None,
            meta_generation: 0,
            trials_required: 10,
            rng_seed: 0, // Will be set when creating algorithm
        }
    }
}

/// Derive a trial seed from algorithm seed and trial number
/// This ensures each trial is reproducible given the algorithm seed
pub fn derive_trial_seed(algo_seed: u64, trial_number: i32) -> u64 {
    // Use a simple but deterministic combination
    algo_seed.wrapping_add(trial_number as u64).wrapping_mul(2654435761)
}

/// Insert a new meta-algorithm
pub fn insert_meta_algorithm(conn: &Connection, algo: &MetaAlgorithm) -> Result<i64> {
    conn.execute(
        "INSERT INTO meta_algorithms (
            name, version, feature_weights,
            similarity_radius, interpolation_weight,
            population_size, max_generations,
            mutation_rate, mutation_strength, crossover_rate,
            tournament_size, elite_count,
            polytope_patience, switch_threshold, switch_probability,
            cc_weight, parent_id, meta_generation, trials_required, rng_seed
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
        params![
            algo.name,
            algo.version,
            algo.feature_weights,
            algo.similarity_radius,
            algo.interpolation_weight,
            algo.population_size,
            algo.max_generations,
            algo.mutation_rate,
            algo.mutation_strength,
            algo.crossover_rate,
            algo.tournament_size,
            algo.elite_count,
            algo.polytope_patience,
            algo.switch_threshold,
            algo.switch_probability,
            algo.cc_weight,
            algo.parent_id,
            algo.meta_generation,
            algo.trials_required,
            algo.rng_seed as i64,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Get a meta-algorithm by ID
pub fn get_meta_algorithm(conn: &Connection, id: i64) -> Result<Option<MetaAlgorithm>> {
    let result = conn.query_row(
        "SELECT id, name, version, feature_weights,
                similarity_radius, interpolation_weight,
                population_size, max_generations,
                mutation_rate, mutation_strength, crossover_rate,
                tournament_size, elite_count,
                polytope_patience, switch_threshold, switch_probability,
                cc_weight, parent_id, meta_generation, trials_required, rng_seed
         FROM meta_algorithms WHERE id = ?1",
        params![id],
        |row| {
            Ok(MetaAlgorithm {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                version: row.get(2)?,
                feature_weights: row.get(3)?,
                similarity_radius: row.get(4)?,
                interpolation_weight: row.get(5)?,
                population_size: row.get(6)?,
                max_generations: row.get(7)?,
                mutation_rate: row.get(8)?,
                mutation_strength: row.get(9)?,
                crossover_rate: row.get(10)?,
                tournament_size: row.get(11)?,
                elite_count: row.get(12)?,
                polytope_patience: row.get(13)?,
                switch_threshold: row.get(14)?,
                switch_probability: row.get(15)?,
                cc_weight: row.get(16)?,
                parent_id: row.get(17)?,
                meta_generation: row.get(18)?,
                trials_required: row.get(19)?,
                rng_seed: row.get::<_, Option<i64>>(20)?.unwrap_or(0) as u64,
            })
        },
    );

    match result {
        Ok(algo) => Ok(Some(algo)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Meta-trial result
#[derive(Debug, Clone)]
pub struct MetaTrial {
    pub id: Option<i64>,
    pub algorithm_id: i64,
    pub run_id: Option<String>,
    pub generations_run: i32,
    pub initial_fitness: f64,
    pub final_fitness: f64,
    pub fitness_improvement: f64,
    pub improvement_rate: f64,
    pub fitness_auc: f64,
    pub best_cc_log_error: f64,
    pub physics_success_rate: f64,
    pub unique_polytopes_tried: i32,
}

/// Insert a meta-trial result
pub fn insert_meta_trial(conn: &Connection, trial: &MetaTrial) -> Result<i64> {
    conn.execute(
        "INSERT INTO meta_trials (
            algorithm_id, run_id, generations_run,
            initial_fitness, final_fitness, fitness_improvement, improvement_rate,
            fitness_auc, best_cc_log_error, physics_success_rate, unique_polytopes_tried,
            started_at, ended_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'), datetime('now'))",
        params![
            trial.algorithm_id,
            trial.run_id,
            trial.generations_run,
            trial.initial_fitness,
            trial.final_fitness,
            trial.fitness_improvement,
            trial.improvement_rate,
            trial.fitness_auc,
            trial.best_cc_log_error,
            trial.physics_success_rate,
            trial.unique_polytopes_tried,
        ],
    )?;

    // Update aggregate meta_fitness
    update_meta_fitness(conn, trial.algorithm_id)?;

    Ok(conn.last_insert_rowid())
}

/// Update aggregate meta-fitness for an algorithm
fn update_meta_fitness(conn: &Connection, algorithm_id: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO meta_fitness (algorithm_id, trial_count, mean_improvement_rate, best_improvement_rate,
                                   mean_fitness_auc, best_final_fitness, best_cc_log_error, mean_cc_log_error, meta_fitness)
         SELECT
             algorithm_id,
             COUNT(*) as trial_count,
             AVG(improvement_rate) as mean_improvement_rate,
             MAX(improvement_rate) as best_improvement_rate,
             AVG(fitness_auc) as mean_fitness_auc,
             MAX(final_fitness) as best_final_fitness,
             MIN(best_cc_log_error) as best_cc_log_error,
             AVG(best_cc_log_error) as mean_cc_log_error,
             -- Combined meta-fitness: weighted sum of improvement rate and CC performance
             (AVG(improvement_rate) * 0.3 + MAX(final_fitness) * 0.3 + (1.0 / (1.0 + AVG(best_cc_log_error))) * 0.4) as meta_fitness
         FROM meta_trials
         WHERE algorithm_id = ?1
         GROUP BY algorithm_id
         ON CONFLICT(algorithm_id) DO UPDATE SET
             trial_count = excluded.trial_count,
             mean_improvement_rate = excluded.mean_improvement_rate,
             best_improvement_rate = excluded.best_improvement_rate,
             mean_fitness_auc = excluded.mean_fitness_auc,
             best_final_fitness = excluded.best_final_fitness,
             best_cc_log_error = excluded.best_cc_log_error,
             mean_cc_log_error = excluded.mean_cc_log_error,
             meta_fitness = excluded.meta_fitness,
             updated_at = datetime('now')",
        params![algorithm_id],
    )?;
    Ok(())
}

/// Get top meta-algorithms by meta-fitness
pub fn get_top_meta_algorithms(conn: &Connection, generation: i32, limit: i32) -> Result<Vec<(MetaAlgorithm, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT a.id, a.name, a.version, a.feature_weights,
                a.similarity_radius, a.interpolation_weight,
                a.population_size, a.max_generations,
                a.mutation_rate, a.mutation_strength, a.crossover_rate,
                a.tournament_size, a.elite_count,
                a.polytope_patience, a.switch_threshold, a.switch_probability,
                a.cc_weight, a.parent_id, a.meta_generation, a.trials_required, a.rng_seed,
                COALESCE(f.meta_fitness, 0) as meta_fitness
         FROM meta_algorithms a
         LEFT JOIN meta_fitness f ON f.algorithm_id = a.id
         WHERE a.meta_generation = ?1 AND a.status = 'completed'
         ORDER BY meta_fitness DESC
         LIMIT ?2"
    )?;

    let rows = stmt.query_map(params![generation, limit], |row| {
        let algo = MetaAlgorithm {
            id: Some(row.get(0)?),
            name: row.get(1)?,
            version: row.get(2)?,
            feature_weights: row.get(3)?,
            similarity_radius: row.get(4)?,
            interpolation_weight: row.get(5)?,
            population_size: row.get(6)?,
            max_generations: row.get(7)?,
            mutation_rate: row.get(8)?,
            mutation_strength: row.get(9)?,
            crossover_rate: row.get(10)?,
            tournament_size: row.get(11)?,
            elite_count: row.get(12)?,
            polytope_patience: row.get(13)?,
            switch_threshold: row.get(14)?,
            switch_probability: row.get(15)?,
            cc_weight: row.get(16)?,
            parent_id: row.get(17)?,
            meta_generation: row.get(18)?,
            trials_required: row.get(19)?,
            rng_seed: row.get::<_, Option<i64>>(20)?.unwrap_or(0) as u64,
        };
        let fitness: f64 = row.get(21)?;
        Ok((algo, fitness))
    })?;

    rows.collect()
}

/// Get evaluation count for a polytope
pub fn get_polytope_eval_count(conn: &Connection, polytope_id: i64) -> Result<i64> {
    conn.query_row(
        "SELECT COALESCE(eval_count, 0) FROM polytopes WHERE id = ?1",
        params![polytope_id],
        |row| row.get(0),
    ).or(Ok(0))
}

// =============================================================================
// Meta-GA Worker Locking
// =============================================================================

/// Stale heartbeat threshold in seconds (process considered dead after this)
pub const HEARTBEAT_STALE_SECONDS: i64 = 60;

/// Try to acquire an algorithm for this worker.
/// Returns the algorithm ID if successful, None if no available algorithm.
///
/// Strategy:
/// 1. Find pending algorithms, or running algorithms with stale heartbeat
/// 2. Write our PID and heartbeat timestamp
/// 3. Read it back - if it matches, we own it
pub fn try_acquire_algorithm(conn: &Connection, my_pid: i32) -> Result<Option<i64>> {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let stale_cutoff = (chrono::Utc::now() - chrono::Duration::seconds(HEARTBEAT_STALE_SECONDS))
        .format("%Y-%m-%d %H:%M:%S").to_string();

    // Find candidate: running with stale heartbeat (resume), or pending
    // Prioritize stale running algorithms so we resume interrupted work
    let candidate: Option<i64> = conn.query_row(
        "SELECT id FROM meta_algorithms
         WHERE status = 'pending'
            OR (status = 'running' AND (last_heartbeat_at IS NULL OR last_heartbeat_at < ?1))
         ORDER BY
            CASE WHEN status = 'running' THEN 0 ELSE 1 END,  -- prefer resuming stale running
            meta_generation ASC,  -- oldest generation first
            id ASC
         LIMIT 1",
        params![stale_cutoff],
        |row| row.get(0),
    ).ok();

    let algo_id = match candidate {
        Some(id) => id,
        None => return Ok(None),
    };

    // Try to acquire: write our PID and timestamp
    conn.execute(
        "UPDATE meta_algorithms
         SET locked_by_pid = ?1, last_heartbeat_at = ?2, status = 'running'
         WHERE id = ?3",
        params![my_pid, now, algo_id],
    )?;

    // Read back to verify we got it
    let (read_pid, read_heartbeat): (Option<i32>, Option<String>) = conn.query_row(
        "SELECT locked_by_pid, last_heartbeat_at FROM meta_algorithms WHERE id = ?1",
        params![algo_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;

    if read_pid == Some(my_pid) && read_heartbeat.as_deref() == Some(&now) {
        Ok(Some(algo_id))
    } else {
        // Someone else got it
        Ok(None)
    }
}

/// Update heartbeat for an algorithm we own
pub fn update_heartbeat(conn: &Connection, algo_id: i64, my_pid: i32) -> Result<bool> {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

    let rows = conn.execute(
        "UPDATE meta_algorithms
         SET last_heartbeat_at = ?1
         WHERE id = ?2 AND locked_by_pid = ?3 AND status = 'running'",
        params![now, algo_id, my_pid],
    )?;

    Ok(rows > 0)
}

/// Mark algorithm as completed and release lock
pub fn complete_algorithm(conn: &Connection, algo_id: i64, my_pid: i32) -> Result<bool> {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

    let rows = conn.execute(
        "UPDATE meta_algorithms
         SET status = 'completed', completed_at = ?1, locked_by_pid = NULL, last_heartbeat_at = NULL
         WHERE id = ?2 AND locked_by_pid = ?3",
        params![now, algo_id, my_pid],
    )?;

    Ok(rows > 0)
}

/// Mark algorithm as failed and release lock
pub fn fail_algorithm(conn: &Connection, algo_id: i64, my_pid: i32) -> Result<bool> {
    let rows = conn.execute(
        "UPDATE meta_algorithms
         SET status = 'failed', locked_by_pid = NULL, last_heartbeat_at = NULL
         WHERE id = ?1 AND locked_by_pid = ?2",
        params![algo_id, my_pid],
    )?;

    Ok(rows > 0)
}

/// Get current meta-generation (highest generation with any completed algorithms)
pub fn get_current_meta_generation(conn: &Connection) -> Result<i32> {
    conn.query_row(
        "SELECT COALESCE(MAX(meta_generation), 0) FROM meta_algorithms WHERE status = 'completed'",
        [],
        |row| row.get(0),
    ).or(Ok(0))
}

/// Get count of algorithms per status in a generation
pub fn get_generation_status(conn: &Connection, generation: i32) -> Result<(i32, i32, i32, i32)> {
    // (pending, running, completed, failed)
    let pending: i32 = conn.query_row(
        "SELECT COUNT(*) FROM meta_algorithms WHERE meta_generation = ?1 AND status = 'pending'",
        params![generation], |row| row.get(0)
    ).unwrap_or(0);
    let running: i32 = conn.query_row(
        "SELECT COUNT(*) FROM meta_algorithms WHERE meta_generation = ?1 AND status = 'running'",
        params![generation], |row| row.get(0)
    ).unwrap_or(0);
    let completed: i32 = conn.query_row(
        "SELECT COUNT(*) FROM meta_algorithms WHERE meta_generation = ?1 AND status = 'completed'",
        params![generation], |row| row.get(0)
    ).unwrap_or(0);
    let failed: i32 = conn.query_row(
        "SELECT COUNT(*) FROM meta_algorithms WHERE meta_generation = ?1 AND status = 'failed'",
        params![generation], |row| row.get(0)
    ).unwrap_or(0);

    Ok((pending, running, completed, failed))
}

/// Check if a generation is complete (all algorithms done)
pub fn is_generation_complete(conn: &Connection, generation: i32) -> Result<bool> {
    let (pending, running, _, _) = get_generation_status(conn, generation)?;
    Ok(pending == 0 && running == 0)
}

/// Get meta_state (current generation, algorithms per generation)
pub fn get_meta_state(conn: &Connection) -> Result<(i32, i32)> {
    conn.query_row(
        "SELECT current_generation, algorithms_per_generation FROM meta_state WHERE id = 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
}

/// Update meta_state current generation
pub fn set_current_generation(conn: &Connection, generation: i32) -> Result<()> {
    conn.execute(
        "UPDATE meta_state SET current_generation = ?1, updated_at = datetime('now') WHERE id = 1",
        params![generation],
    )?;
    Ok(())
}

/// Count algorithms in a generation
pub fn count_algorithms_in_generation(conn: &Connection, generation: i32) -> Result<i32> {
    conn.query_row(
        "SELECT COUNT(*) FROM meta_algorithms WHERE meta_generation = ?1",
        params![generation],
        |row| row.get(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper to create a test database
    fn test_db() -> (tempfile::TempDir, Arc<Mutex<Connection>>) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = init_database(db_path.to_str().unwrap()).unwrap();
        (dir, conn)
    }

    // =========================================================================
    // Database Initialization Tests
    // =========================================================================

    #[test]
    fn test_database_init_creates_all_tables() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let expected_tables = [
            "schema_version",
            "evaluations",
            "polytopes",
            "heuristics",
            "runs",
            "meta_algorithms",
            "meta_trials",
            "meta_fitness",
            "meta_state",
        ];

        for table in expected_tables {
            let count: i32 = locked
                .query_row(
                    &format!(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'",
                        table
                    ),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 1, "Table '{}' should exist", table);
        }
    }

    #[test]
    fn test_database_init_is_idempotent() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // Initialize twice - should not error
        let _conn1 = init_database(db_path.to_str().unwrap()).unwrap();
        let _conn2 = init_database(db_path.to_str().unwrap()).unwrap();
    }

    #[test]
    fn test_schema_version_tracking() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let version: i32 = locked
            .query_row(
                "SELECT MAX(version) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap();

        // Should have at least version 2 (meta-GA schema)
        assert!(version >= 2, "Schema version should be at least 2");
    }

    // =========================================================================
    // Polytope Tests
    // =========================================================================

    #[test]
    fn test_upsert_polytope() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        upsert_polytope(&locked, 42, 3, 6, 5, "[[1,0,0,0],[0,1,0,0]]").unwrap();

        let (h11, h21): (i32, i32) = locked
            .query_row(
                "SELECT h11, h21 FROM polytopes WHERE id = 42",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(h11, 3);
        assert_eq!(h21, 6);
    }

    #[test]
    fn test_upsert_polytope_does_not_overwrite() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Insert first time
        upsert_polytope(&locked, 1, 3, 6, 5, "[[0,0,0,0]]").unwrap();

        // Try to insert again with different data - should NOT update
        upsert_polytope(&locked, 1, 99, 99, 99, "[[1,1,1,1]]").unwrap();

        let h11: i32 = locked
            .query_row("SELECT h11 FROM polytopes WHERE id = 1", [], |row| {
                row.get(0)
            })
            .unwrap();

        assert_eq!(h11, 3, "Upsert should not overwrite existing polytope");
    }

    #[test]
    fn test_polytope_stats_aggregation() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        upsert_polytope(&locked, 1, 3, 6, 5, "[[0,0,0,0]]").unwrap();

        // Add multiple fitness values
        update_polytope_stats(&locked, 1, 0.5).unwrap();
        update_polytope_stats(&locked, 1, 0.7).unwrap();
        update_polytope_stats(&locked, 1, 0.3).unwrap();

        let stats = get_polytope_stats(&locked, 1).unwrap().unwrap();
        assert_eq!(stats.eval_count, 3);
        assert!((stats.fitness_mean - 0.5).abs() < 0.01);
        assert!((stats.fitness_min - 0.3).abs() < 0.01);
        assert!((stats.fitness_max - 0.7).abs() < 0.01);

        // Check variance: var([0.3, 0.5, 0.7]) = 0.0267
        assert!(stats.fitness_variance > 0.02 && stats.fitness_variance < 0.03);
    }

    #[test]
    fn test_polytope_stats_none_for_missing() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let stats = get_polytope_stats(&locked, 999999).unwrap();
        assert!(stats.is_none());
    }

    #[test]
    fn test_get_top_polytopes() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Create polytopes with different fitness levels
        for i in 1..=5 {
            upsert_polytope(&locked, i, 3, 6, 5, "[]").unwrap();
            for _ in 0..3 {
                update_polytope_stats(&locked, i, i as f64 * 0.1).unwrap();
            }
        }

        let top = get_top_polytopes(&locked, 3, 2).unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].polytope_id, 5); // Highest fitness
        assert_eq!(top[1].polytope_id, 4);
        assert_eq!(top[2].polytope_id, 3);
    }

    // =========================================================================
    // Meta-Algorithm Tests
    // =========================================================================

    #[test]
    fn test_insert_and_get_meta_algorithm() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let algo = MetaAlgorithm {
            id: None,
            name: Some("test_algo".to_string()),
            version: 1,
            feature_weights: r#"{"sphericity": 1.5}"#.to_string(),
            similarity_radius: 0.6,
            interpolation_weight: 0.4,
            population_size: 100,
            max_generations: 15,
            mutation_rate: 0.35,
            mutation_strength: 0.3,
            crossover_rate: 0.8,
            tournament_size: 5,
            elite_count: 10,
            polytope_patience: 5,
            switch_threshold: 0.01,
            switch_probability: 0.1,
            cc_weight: 10.0,
            parent_id: None,
            meta_generation: 0,
            trials_required: 10,
            rng_seed: 12345,
        };

        let id = insert_meta_algorithm(&locked, &algo).unwrap();
        assert!(id > 0);

        let retrieved = get_meta_algorithm(&locked, id).unwrap().unwrap();
        assert_eq!(retrieved.name, Some("test_algo".to_string()));
        assert_eq!(retrieved.population_size, 100);
        assert!((retrieved.similarity_radius - 0.6).abs() < 0.001);
        assert_eq!(retrieved.trials_required, 10);
    }

    #[test]
    fn test_get_meta_algorithm_none_for_missing() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let algo = get_meta_algorithm(&locked, 999999).unwrap();
        assert!(algo.is_none());
    }

    #[test]
    fn test_meta_algorithm_default() {
        let algo = MetaAlgorithm::default();
        assert_eq!(algo.version, META_ALGORITHM_VERSION);
        assert_eq!(algo.trials_required, 10);
        assert!(!algo.feature_weights.is_empty());
    }

    // =========================================================================
    // Meta-Trial Tests
    // =========================================================================

    #[test]
    fn test_insert_meta_trial_updates_fitness() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Create an algorithm first
        let algo = MetaAlgorithm::default();
        let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();

        // Insert a trial (run_id=None to avoid foreign key constraint on runs table)
        let trial = MetaTrial {
            id: None,
            algorithm_id: algo_id,
            run_id: None,
            generations_run: 10,
            initial_fitness: 0.1,
            final_fitness: 0.5,
            fitness_improvement: 0.4,
            improvement_rate: 0.04,
            fitness_auc: 3.0,
            best_cc_log_error: 50.0,
            physics_success_rate: 0.8,
            unique_polytopes_tried: 100,
        };

        let trial_id = insert_meta_trial(&locked, &trial).unwrap();
        assert!(trial_id > 0);

        // Check that meta_fitness was updated
        let (count, fitness): (i32, f64) = locked
            .query_row(
                "SELECT trial_count, meta_fitness FROM meta_fitness WHERE algorithm_id = ?",
                params![algo_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(count, 1);
        assert!(fitness > 0.0);
    }

    #[test]
    fn test_multiple_trials_aggregate_correctly() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let algo = MetaAlgorithm::default();
        let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();

        // Insert multiple trials (run_id=None to avoid foreign key constraint)
        for i in 1..=3 {
            let trial = MetaTrial {
                id: None,
                algorithm_id: algo_id,
                run_id: None,
                generations_run: 10,
                initial_fitness: 0.1,
                final_fitness: 0.3 + (i as f64 * 0.1),
                fitness_improvement: 0.2 + (i as f64 * 0.1),
                improvement_rate: 0.02 + (i as f64 * 0.01),
                fitness_auc: 2.0 + (i as f64),
                best_cc_log_error: 60.0 - (i as f64 * 10.0),
                physics_success_rate: 0.7 + (i as f64 * 0.05),
                unique_polytopes_tried: 50 + (i as i32 * 10),
            };
            insert_meta_trial(&locked, &trial).unwrap();
        }

        let (count, best_final): (i32, f64) = locked
            .query_row(
                "SELECT trial_count, best_final_fitness FROM meta_fitness WHERE algorithm_id = ?",
                params![algo_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(count, 3);
        assert!((best_final - 0.6).abs() < 0.01); // 0.3 + 0.3 = 0.6
    }

    // =========================================================================
    // Worker Locking Tests
    // =========================================================================

    #[test]
    fn test_acquire_algorithm_basic() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Create a pending algorithm
        let algo = MetaAlgorithm::default();
        let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();

        // Acquire it
        let acquired = try_acquire_algorithm(&locked, 12345).unwrap();
        assert_eq!(acquired, Some(algo_id));

        // Check it's now running
        let status: String = locked
            .query_row(
                "SELECT status FROM meta_algorithms WHERE id = ?",
                params![algo_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "running");
    }

    #[test]
    fn test_acquire_algorithm_none_when_all_running() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Create and acquire an algorithm
        let algo = MetaAlgorithm::default();
        let _algo_id = insert_meta_algorithm(&locked, &algo).unwrap();
        try_acquire_algorithm(&locked, 12345).unwrap();

        // Try to acquire with different PID - should get None
        let acquired = try_acquire_algorithm(&locked, 99999).unwrap();
        assert!(acquired.is_none());
    }

    #[test]
    fn test_heartbeat_update() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let algo = MetaAlgorithm::default();
        let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();
        try_acquire_algorithm(&locked, 12345).unwrap();

        // Update heartbeat
        let updated = update_heartbeat(&locked, algo_id, 12345).unwrap();
        assert!(updated);

        // Wrong PID should fail
        let updated = update_heartbeat(&locked, algo_id, 99999).unwrap();
        assert!(!updated);
    }

    #[test]
    fn test_complete_algorithm() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let algo = MetaAlgorithm::default();
        let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();
        try_acquire_algorithm(&locked, 12345).unwrap();

        // Complete it
        let completed = complete_algorithm(&locked, algo_id, 12345).unwrap();
        assert!(completed);

        let status: String = locked
            .query_row(
                "SELECT status FROM meta_algorithms WHERE id = ?",
                params![algo_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "completed");

        // Lock should be cleared
        let locked_by: Option<i32> = locked
            .query_row(
                "SELECT locked_by_pid FROM meta_algorithms WHERE id = ?",
                params![algo_id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(locked_by.is_none());
    }

    #[test]
    fn test_complete_algorithm_wrong_pid_fails() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let algo = MetaAlgorithm::default();
        let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();
        try_acquire_algorithm(&locked, 12345).unwrap();

        // Try to complete with wrong PID
        let completed = complete_algorithm(&locked, algo_id, 99999).unwrap();
        assert!(!completed);

        // Should still be running
        let status: String = locked
            .query_row(
                "SELECT status FROM meta_algorithms WHERE id = ?",
                params![algo_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "running");
    }

    #[test]
    fn test_fail_algorithm() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let algo = MetaAlgorithm::default();
        let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();
        try_acquire_algorithm(&locked, 12345).unwrap();

        let failed = fail_algorithm(&locked, algo_id, 12345).unwrap();
        assert!(failed);

        let status: String = locked
            .query_row(
                "SELECT status FROM meta_algorithms WHERE id = ?",
                params![algo_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "failed");
    }

    // =========================================================================
    // Generation Management Tests
    // =========================================================================

    #[test]
    fn test_count_algorithms_in_generation() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Insert algorithms in different generations
        for gen in 0..=2 {
            for _ in 0..(gen + 1) {
                let mut algo = MetaAlgorithm::default();
                algo.meta_generation = gen;
                insert_meta_algorithm(&locked, &algo).unwrap();
            }
        }

        assert_eq!(count_algorithms_in_generation(&locked, 0).unwrap(), 1);
        assert_eq!(count_algorithms_in_generation(&locked, 1).unwrap(), 2);
        assert_eq!(count_algorithms_in_generation(&locked, 2).unwrap(), 3);
        assert_eq!(count_algorithms_in_generation(&locked, 3).unwrap(), 0);
    }

    #[test]
    fn test_generation_status() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Create algorithms and set status directly via SQL
        // (try_acquire_algorithm grabs any pending one, which makes deterministic testing hard)
        let statuses = ["pending", "running", "completed", "failed"];
        for status in &statuses {
            let algo = MetaAlgorithm::default();
            let id = insert_meta_algorithm(&locked, &algo).unwrap();
            locked
                .execute(
                    "UPDATE meta_algorithms SET status = ?1 WHERE id = ?2",
                    params![status, id],
                )
                .unwrap();
        }

        let (pending, running, completed, failed) = get_generation_status(&locked, 0).unwrap();
        assert_eq!(pending, 1);
        assert_eq!(running, 1);
        assert_eq!(completed, 1);
        assert_eq!(failed, 1);
    }

    #[test]
    fn test_is_generation_complete() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Create and complete two algorithms
        for _ in 0..2 {
            let algo = MetaAlgorithm::default();
            let id = insert_meta_algorithm(&locked, &algo).unwrap();
            try_acquire_algorithm(&locked, 12345).unwrap();
            complete_algorithm(&locked, id, 12345).unwrap();
        }

        assert!(is_generation_complete(&locked, 0).unwrap());

        // Add a pending one
        insert_meta_algorithm(&locked, &MetaAlgorithm::default()).unwrap();
        assert!(!is_generation_complete(&locked, 0).unwrap());
    }

    #[test]
    fn test_meta_state() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        let (gen, per_gen) = get_meta_state(&locked).unwrap();
        assert_eq!(gen, 0);
        assert_eq!(per_gen, 16); // default

        set_current_generation(&locked, 5).unwrap();
        let (gen, _) = get_meta_state(&locked).unwrap();
        assert_eq!(gen, 5);
    }

    #[test]
    fn test_get_top_meta_algorithms() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // Create algorithms and trials
        // Set status directly to avoid try_acquire_algorithm's non-deterministic behavior
        for i in 1..=3 {
            let algo = MetaAlgorithm::default();
            let algo_id = insert_meta_algorithm(&locked, &algo).unwrap();

            // Mark as completed directly
            locked
                .execute(
                    "UPDATE meta_algorithms SET status = 'completed' WHERE id = ?",
                    params![algo_id],
                )
                .unwrap();

            // Insert trial (run_id=None to avoid foreign key constraint)
            let trial = MetaTrial {
                id: None,
                algorithm_id: algo_id,
                run_id: None,
                generations_run: 10,
                initial_fitness: 0.1,
                final_fitness: i as f64 * 0.2,
                fitness_improvement: i as f64 * 0.1,
                improvement_rate: i as f64 * 0.01,
                fitness_auc: i as f64,
                best_cc_log_error: 100.0 / i as f64,
                physics_success_rate: 0.8,
                unique_polytopes_tried: 100,
            };
            insert_meta_trial(&locked, &trial).unwrap();
        }

        let top = get_top_meta_algorithms(&locked, 0, 2).unwrap();
        assert_eq!(top.len(), 2);

        // First should have highest meta_fitness
        assert!(top[0].1 >= top[1].1);
    }

    // =========================================================================
    // HeuristicsData Tests
    // =========================================================================

    #[test]
    fn test_heuristics_data_to_map_empty() {
        let data = HeuristicsData::default();
        let map = data.to_map();
        assert!(map.is_empty());
    }

    #[test]
    fn test_heuristics_data_to_map_with_values() {
        let data = HeuristicsData {
            sphericity: Some(0.5),
            inertia_isotropy: Some(0.8),
            chirality_optimal: Some(1.2),
            zero_count: Some(10),
            fibonacci_count: Some(3),
            ..Default::default()
        };

        let map = data.to_map();

        assert_eq!(map.get("sphericity"), Some(&0.5));
        assert_eq!(map.get("inertia_isotropy"), Some(&0.8));
        assert_eq!(map.get("chirality_optimal"), Some(&1.2));
        assert_eq!(map.get("zero_count"), Some(&10.0));
        assert_eq!(map.get("fibonacci_count"), Some(&3.0));
        assert!(!map.contains_key("coord_mean")); // Not set
    }

    #[test]
    fn test_heuristics_data_to_map_all_numeric_fields() {
        let data = HeuristicsData {
            h11: Some(42),
            h21: Some(39),
            vertex_count: Some(10),
            sphericity: Some(1.0),
            inertia_isotropy: Some(2.0),
            chirality_optimal: Some(3.0),
            chirality_x: Some(4.0),
            chirality_y: Some(5.0),
            chirality_z: Some(6.0),
            chirality_w: Some(7.0),
            handedness_det: Some(8.0),
            symmetry_x: Some(9.0),
            symmetry_y: Some(10.0),
            symmetry_z: Some(11.0),
            symmetry_w: Some(12.0),
            flatness_3d: Some(13.0),
            flatness_2d: Some(14.0),
            intrinsic_dim_estimate: Some(15.0),
            spikiness: Some(16.0),
            max_exposure: Some(17.0),
            conformity_ratio: Some(18.0),
            distance_kurtosis: Some(19.0),
            loner_score: Some(20.0),
            coord_mean: Some(21.0),
            coord_median: Some(22.0),
            coord_std: Some(23.0),
            coord_skewness: Some(24.0),
            coord_kurtosis: Some(25.0),
            shannon_entropy: Some(26.0),
            joint_entropy: Some(27.0),
            compression_ratio: Some(28.0),
            sorted_compression_ratio: Some(29.0),
            sort_compression_gain: Some(30.0),
            phi_ratio_count: Some(31),
            fibonacci_count: Some(32),
            zero_count: Some(33),
            one_count: Some(34),
            prime_count: Some(35),
            outlier_score: Some(36.0),
            outlier_max_zscore: Some(37.0),
            outlier_max_dim: Some("x".to_string()),  // String, not in map
            outlier_count_2sigma: Some(38),
            outlier_count_3sigma: Some(39),
        };

        let map = data.to_map();

        // Should have all numeric fields
        assert_eq!(map.len(), 39);  // 32 f64 + 7 i32

        // Verify f64 fields
        assert_eq!(map.get("sphericity"), Some(&1.0));
        assert_eq!(map.get("outlier_max_zscore"), Some(&37.0));

        // Verify i32 fields converted to f64
        assert_eq!(map.get("phi_ratio_count"), Some(&31.0));
        assert_eq!(map.get("outlier_count_3sigma"), Some(&39.0));

        // String field not included
        assert!(!map.contains_key("outlier_max_dim"));
    }

    #[test]
    fn test_upsert_heuristics_stores_hodge_numbers() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // First create the polytope (FK constraint)
        upsert_polytope(&locked, 12345, 24, 21, 15, "[]").unwrap();

        // Create heuristics with h11/h21/vertex_count
        let data = HeuristicsData {
            h11: Some(24),
            h21: Some(21),
            vertex_count: Some(15),
            sphericity: Some(0.75),
            ..Default::default()
        };

        upsert_heuristics(&locked, 12345, &data).unwrap();

        // Verify by querying directly
        let (h11, h21, vertex_count): (Option<i32>, Option<i32>, Option<i32>) = locked
            .query_row(
                "SELECT h11, h21, vertex_count FROM heuristics WHERE polytope_id = 12345",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();

        assert_eq!(h11, Some(24), "h11 should be stored correctly");
        assert_eq!(h21, Some(21), "h21 should be stored correctly");
        assert_eq!(vertex_count, Some(15), "vertex_count should be stored correctly");
    }

    #[test]
    fn test_get_heuristics_retrieves_hodge_numbers() {
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // First create the polytope (FK constraint)
        upsert_polytope(&locked, 99999, 30, 27, 20, "[]").unwrap();

        // Insert heuristics with h11/h21/vertex_count
        let data = HeuristicsData {
            h11: Some(30),
            h21: Some(27),
            vertex_count: Some(20),
            sphericity: Some(0.85),
            spikiness: Some(0.42),
            ..Default::default()
        };

        upsert_heuristics(&locked, 99999, &data).unwrap();

        // Retrieve and verify
        let retrieved = get_heuristics(&locked, 99999).unwrap().expect("Should find heuristics");

        assert_eq!(retrieved.h11, Some(30), "h11 should be retrieved correctly");
        assert_eq!(retrieved.h21, Some(27), "h21 should be retrieved correctly");
        assert_eq!(retrieved.vertex_count, Some(20), "vertex_count should be retrieved correctly");
        assert_eq!(retrieved.sphericity, Some(0.85));
        assert_eq!(retrieved.spikiness, Some(0.42));
    }

    #[test]
    fn test_heuristics_hodge_numbers_not_zero_when_set() {
        // This test specifically verifies the bug fix for h11/h21 always being zero
        let (_dir, conn) = test_db();
        let locked = conn.lock().unwrap();

        // First create the polytope (FK constraint)
        upsert_polytope(&locked, 777, 42, 39, 25, "[]").unwrap();

        // Insert with non-zero h11/h21
        let data = HeuristicsData {
            h11: Some(42),
            h21: Some(39),
            vertex_count: Some(25),
            ..Default::default()
        };

        upsert_heuristics(&locked, 777, &data).unwrap();

        // Retrieve and verify they are NOT zero
        let retrieved = get_heuristics(&locked, 777).unwrap().expect("Should find heuristics");

        assert_ne!(retrieved.h11, Some(0), "h11 should NOT be zero");
        assert_ne!(retrieved.h21, Some(0), "h21 should NOT be zero");
        assert_eq!(retrieved.h11, Some(42));
        assert_eq!(retrieved.h21, Some(39));
    }
}
