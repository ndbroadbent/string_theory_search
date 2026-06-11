//! Heuristics Worker
//!
//! Computes geometric heuristics for every polytope in a cyrus-ga pool file
//! (JSONL, one polytope per line keyed by "name" with a "points" array of 4D
//! integer vectors) and upserts them into the dashboard SQLite database,
//! keyed by the TEXT pool name.
//!
//! Usage:
//!     heuristics --pool /data/cyrus/pools/h21_4-7.jsonl --db data/string_theory.db
//!     heuristics -c config.toml            # paths from config [paths] pool/database
//!     heuristics --pool pool.jsonl --db test.db --limit 50 --no-resume

use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use string_theory::db::{get_processed_heuristics_ids, init_database, upsert_heuristics};
use string_theory::heuristics::compute_heuristics;

#[derive(Parser)]
#[command(name = "heuristics")]
#[command(about = "Compute heuristics for all polytopes in a cyrus-ga pool file")]
struct Args {
    /// Config file path (used for defaults when --pool/--db are not given)
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Pool JSONL file (one polytope per line: {"name", "h11", "h21", "points"})
    #[arg(long)]
    pool: Option<String>,

    /// SQLite database path
    #[arg(long)]
    db: Option<String>,

    /// Skip polytopes that already have heuristics rows
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    resume: bool,

    /// Stop after processing N polytopes
    #[arg(long)]
    limit: Option<u64>,
}

#[derive(Deserialize, Default)]
struct Config {
    #[serde(default)]
    paths: PathsConfig,
}

#[derive(Deserialize, Default)]
struct PathsConfig {
    /// Pool JSONL file (cyrus-ga pool format)
    pool: Option<String>,
    database: Option<String>,
}

/// A polytope record from the cyrus-ga pool file
#[derive(Deserialize)]
struct PoolPolytope {
    name: String,
    h11: i32,
    h21: i32,
    points: Vec<[i32; 4]>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Config is optional: only needed when --pool/--db are not provided
    let config: Config = match std::fs::read_to_string(&args.config) {
        Ok(s) => toml::from_str(&s)?,
        Err(_) => Config::default(),
    };

    let pool_path = args
        .pool
        .or(config.paths.pool)
        .ok_or("No pool file: pass --pool <pool.jsonl> or set [paths] pool in config.toml")?;
    let db_path = args
        .db
        .or(config.paths.database)
        .or_else(|| std::env::var("STRING_THEORY_DB").ok())
        .unwrap_or_else(|| string_theory::db::DEFAULT_DB_PATH.to_string());

    log::info!("Heuristics Worker");
    log::info!("  Pool: {}", pool_path);
    log::info!("  Database: {}", db_path);

    let conn = init_database(&db_path)?;

    let processed_ids = if args.resume {
        get_processed_heuristics_ids(&conn)?
    } else {
        std::collections::HashSet::new()
    };
    log::info!("  Already processed: {} polytopes", processed_ids.len());

    let file = File::open(&pool_path)?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let start = Instant::now();
    let mut processed: u64 = 0;
    let mut skipped: u64 = 0;

    for (line_no, line) in reader.lines().enumerate() {
        if let Some(limit) = args.limit {
            if processed >= limit {
                log::info!("Reached limit of {} polytopes", limit);
                break;
            }
        }

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let polytope: PoolPolytope = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Skipping malformed pool line {}: {}", line_no + 1, e);
                continue;
            }
        };

        if args.resume && processed_ids.contains(&polytope.name) {
            skipped += 1;
            continue;
        }

        // Flatten the points array for the metric computations
        let vertices: Vec<i32> = polytope.points.iter().flatten().copied().collect();
        let h = compute_heuristics(polytope.h11, polytope.h21, &vertices);

        upsert_heuristics(&conn, &polytope.name, &h)?;
        processed += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM heuristics", [], |row| row.get(0))?;

    log::info!("=== Complete ===");
    log::info!("  Processed: {} (skipped {} already done)", processed, skipped);
    log::info!("  Time: {:.1}s", elapsed);
    log::info!("  Total in database: {}", total);

    Ok(())
}
