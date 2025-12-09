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
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;"
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
            polytope_id,
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
            ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40, ?41, datetime('now')
        )
        ON CONFLICT(polytope_id) DO UPDATE SET
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

/// Get heuristics for a polytope
pub fn get_heuristics(conn: &Connection, polytope_id: i64) -> Result<Option<HeuristicsData>> {
    let result = conn.query_row(
        "SELECT * FROM heuristics WHERE polytope_id = ?1",
        params![polytope_id],
        |row| {
            Ok(HeuristicsData {
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

/// Get evaluation count for a polytope
pub fn get_polytope_eval_count(conn: &Connection, polytope_id: i64) -> Result<i64> {
    conn.query_row(
        "SELECT COALESCE(eval_count, 0) FROM polytopes WHERE id = ?1",
        params![polytope_id],
        |row| row.get(0),
    ).or(Ok(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_database_init() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = init_database(db_path.to_str().unwrap()).unwrap();

        // Check that tables exist
        let locked = conn.lock().unwrap();
        let count: i32 = locked
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='evaluations'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_polytope_stats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn_arc = init_database(db_path.to_str().unwrap()).unwrap();
        let conn = conn_arc.lock().unwrap();

        // Insert a polytope
        upsert_polytope(&conn, 1, 3, 6, 5, "[[0,0,0,0]]").unwrap();

        // Update stats manually
        update_polytope_stats(&conn, 1, 0.5).unwrap();
        update_polytope_stats(&conn, 1, 0.7).unwrap();
        update_polytope_stats(&conn, 1, 0.3).unwrap();

        // Check stats
        let stats = get_polytope_stats(&conn, 1).unwrap().unwrap();
        assert_eq!(stats.eval_count, 3);
        assert!((stats.fitness_mean - 0.5).abs() < 0.01);
        assert!((stats.fitness_min - 0.3).abs() < 0.01);
        assert!((stats.fitness_max - 0.7).abs() < 0.01);
    }
}
