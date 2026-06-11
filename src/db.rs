//! SQLite database layer for the cyrus-ga dashboard pipeline
//!
//! The database is shared with the TypeScript ingester (web/scripts/ingest.ts),
//! which fills the polytopes/candidates/rounds tables from the cyrus-ga run
//! directory. This module only manages schema setup and the heuristics table.

use rusqlite::{Connection, Result, params};
use std::path::Path;

/// Default database path
pub const DEFAULT_DB_PATH: &str = "data/string_theory.db";

/// Initialize database with a specific path
pub fn init_database(path: &str) -> Result<Connection> {
    // Ensure parent directory exists
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let conn = Connection::open(path)?;

    // WAL mode + busy timeout: the ingester and web server read concurrently
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 30000;
         PRAGMA wal_autocheckpoint = 1000;",
    )?;

    run_migrations(&conn)?;

    Ok(conn)
}

/// Run all pending migrations
fn run_migrations(conn: &Connection) -> Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime('now')),
            description TEXT
        )",
        [],
    )?;

    let current_version: i32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

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
    vec![(
        1,
        "cyrus-ga schema",
        include_str!("../migrations/001_cyrus_ga_schema.sql"),
    )]
}

/// Upsert heuristics for a polytope (keyed by pool name, e.g. "h21_4_0")
pub fn upsert_heuristics(
    conn: &Connection,
    polytope_id: &str,
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

/// Get the set of polytope names that already have heuristics rows
pub fn get_processed_heuristics_ids(
    conn: &Connection,
) -> Result<std::collections::HashSet<String>> {
    let mut stmt = conn.prepare("SELECT polytope_id FROM heuristics")?;
    let ids = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<std::collections::HashSet<String>>>()?;
    Ok(ids)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_and_upsert_heuristics() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = init_database(db_path.to_str().unwrap()).unwrap();

        let h = HeuristicsData {
            h11: Some(28),
            h21: Some(4),
            vertex_count: Some(29),
            sphericity: Some(0.5),
            ..Default::default()
        };

        upsert_heuristics(&conn, "h21_4_0", &h).unwrap();
        // Idempotent upsert
        upsert_heuristics(&conn, "h21_4_0", &h).unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM heuristics", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);

        let ids = get_processed_heuristics_ids(&conn).unwrap();
        assert!(ids.contains("h21_4_0"));
    }
}
