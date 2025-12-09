//! Heuristics Worker
//!
//! Background process that computes heuristics for all polytopes in the JSONL file.
//! Designed to run continuously until all polytopes are processed.
//!
//! Usage:
//!     ./target/release/heuristics -c config.server.toml
//!     ./target/release/heuristics --batch-size 1000 --resume
//!     ./target/release/heuristics build-index -c config.server.toml

use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use string_theory::db::{init_database, upsert_heuristics, upsert_polytope};
use string_theory::heuristics::compute_heuristics;
use string_theory::vector_index::HeuristicsIndex;

#[derive(Parser)]
#[command(name = "heuristics")]
#[command(about = "Compute heuristics for all polytopes")]
struct Args {
    /// Config file path
    #[arg(short, long, default_value = "config.toml", global = true)]
    config: String,

    #[command(subcommand)]
    command: Option<Command>,

    /// Batch size for commits
    #[arg(short, long, default_value = "100")]
    batch_size: usize,

    /// Resume from where we left off
    #[arg(short, long, default_value = "true")]
    resume: bool,

    /// Start from a specific polytope index
    #[arg(long)]
    start: Option<u64>,

    /// Stop after processing N polytopes
    #[arg(long)]
    limit: Option<u64>,

    /// Progress reporting interval (seconds)
    #[arg(long, default_value = "30")]
    progress_interval: u64,
}

#[derive(Subcommand)]
enum Command {
    /// Build HNSW vector index from existing heuristics
    BuildIndex {
        /// Output index path (default: data/heuristics.usearch)
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(Deserialize)]
struct Config {
    paths: PathsConfig,
}

#[derive(Deserialize)]
struct PathsConfig {
    polytopes: String,
    #[serde(default = "default_db_path")]
    database: String,
}

fn default_db_path() -> String {
    "data/string_theory.db".to_string()
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Polytope {
    h11: i32,
    h21: i32,
    #[serde(default)]
    vertex_count: Option<i32>,
    vertices: Vec<i32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Load config
    let config_str = std::fs::read_to_string(&args.config)?;
    let config: Config = toml::from_str(&config_str)?;

    // Handle subcommands
    if let Some(command) = args.command {
        return match command {
            Command::BuildIndex { output } => {
                build_index(&config.paths.database, output)
            }
        };
    }

    // Default: run heuristics computation
    run_heuristics_worker(args, config)
}

fn build_index(db_path: &str, output: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = output.unwrap_or_else(|| {
        // Default: same directory as database
        let db_dir = Path::new(db_path).parent().unwrap_or(Path::new("data"));
        db_dir.join("heuristics.usearch").to_string_lossy().to_string()
    });

    log::info!("Building HNSW index");
    log::info!("  Database: {}", db_path);
    log::info!("  Output: {}", output_path);

    let conn = rusqlite::Connection::open(db_path)?;
    let start = Instant::now();

    let index = HeuristicsIndex::build_from_db(&conn, Path::new(&output_path))?;

    let elapsed = start.elapsed();
    log::info!("Index built successfully!");
    log::info!("  Polytopes: {}", index.len());
    log::info!("  Time: {:.1}s", elapsed.as_secs_f64());

    // List output files
    let index_path = Path::new(&output_path);
    let ids_path = index_path.with_extension("ids");
    let raw_path = index_path.with_extension("raw");

    if index_path.exists() {
        let size = std::fs::metadata(index_path)?.len();
        log::info!("  Index file: {} ({:.1} MB)", index_path.display(), size as f64 / 1_000_000.0);
    }
    if ids_path.exists() {
        let size = std::fs::metadata(&ids_path)?.len();
        log::info!("  IDs file: {} ({:.1} MB)", ids_path.display(), size as f64 / 1_000_000.0);
    }
    if raw_path.exists() {
        let size = std::fs::metadata(&raw_path)?.len();
        log::info!("  Raw vectors: {} ({:.1} MB)", raw_path.display(), size as f64 / 1_000_000.0);
    }

    Ok(())
}

fn run_heuristics_worker(args: Args, config: Config) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Heuristics Worker");
    log::info!("  Polytopes: {}", config.paths.polytopes);
    log::info!("  Database: {}", config.paths.database);
    log::info!("  Batch size: {}", args.batch_size);

    // Initialize database
    let db = init_database(&config.paths.database)?;

    // Get already-processed polytope count
    let processed_count: u64 = {
        let conn = db.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM heuristics", [], |row| row.get(0))?
    };
    log::info!("  Already processed: {} polytopes", processed_count);

    // Build index of processed IDs if resuming
    let processed_ids: std::collections::HashSet<u64> = if args.resume && args.start.is_none() {
        let conn = db.lock().unwrap();
        let mut stmt = conn.prepare("SELECT polytope_id FROM heuristics")?;
        let ids: Vec<u64> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        ids.into_iter().collect()
    } else {
        std::collections::HashSet::new()
    };
    log::info!("  Loaded {} processed IDs for resume", processed_ids.len());

    // Setup graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        log::info!("Shutdown requested...");
        r.store(false, Ordering::SeqCst);
    })?;

    // Progress counters
    let total_processed = Arc::new(AtomicU64::new(0));
    let batch_start = Instant::now();

    // Open polytopes file
    let file = File::open(&config.paths.polytopes)?;
    let reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer

    // Determine starting point
    let start_idx = args.start.unwrap_or(0);
    let mut current_idx: u64 = 0;
    let mut _batch_count = 0;
    let mut last_progress = Instant::now();

    log::info!("Starting from polytope {}", start_idx);

    for line in reader.lines() {
        if !running.load(Ordering::SeqCst) {
            log::info!("Shutting down gracefully...");
            break;
        }

        let line = line?;
        let polytope_id = current_idx;
        current_idx += 1;

        // Skip if before start
        if polytope_id < start_idx {
            continue;
        }

        // Skip if already processed (resume mode)
        if args.resume && processed_ids.contains(&polytope_id) {
            continue;
        }

        // Check limit
        if let Some(limit) = args.limit {
            if total_processed.load(Ordering::SeqCst) >= limit {
                log::info!("Reached limit of {} polytopes", limit);
                break;
            }
        }

        // Parse polytope
        let polytope: Polytope = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Failed to parse polytope {}: {}", polytope_id, e);
                continue;
            }
        };

        // Compute heuristics
        let h = compute_heuristics(
            polytope_id as i64,
            polytope.h11,
            polytope.h21,
            &polytope.vertices,
        );

        // Serialize vertices as JSON
        let vertices_json = serde_json::to_string(&polytope.vertices).unwrap_or_default();
        let vertex_count = polytope.vertex_count.unwrap_or((polytope.vertices.len() / 4) as i32);

        // Insert into database (polytope first due to FK constraint, then heuristics)
        {
            let conn = db.lock().unwrap();

            // Upsert polytope first (FK constraint)
            if let Err(e) = upsert_polytope(
                &conn,
                polytope_id as i64,
                polytope.h11,
                polytope.h21,
                vertex_count,
                &vertices_json,
            ) {
                log::error!("Failed to insert polytope {}: {}", polytope_id, e);
                continue;
            }

            // Now insert heuristics
            if let Err(e) = upsert_heuristics(&conn, polytope_id as i64, &h) {
                log::error!("Failed to insert heuristics for {}: {}", polytope_id, e);
                continue;
            }
        }

        _batch_count += 1;
        total_processed.fetch_add(1, Ordering::SeqCst);

        // Progress reporting
        if last_progress.elapsed() >= Duration::from_secs(args.progress_interval) {
            let processed = total_processed.load(Ordering::SeqCst);
            let elapsed = batch_start.elapsed().as_secs_f64();
            let rate = processed as f64 / elapsed;

            log::info!(
                "Progress: {} processed, {:.1}/s, current index: {}",
                processed,
                rate,
                polytope_id
            );

            last_progress = Instant::now();
        }
    }

    // Final stats
    let processed = total_processed.load(Ordering::SeqCst);
    let elapsed = batch_start.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };

    log::info!("=== Complete ===");
    log::info!("  Total processed: {}", processed);
    log::info!("  Time: {:.1}s", elapsed);
    log::info!("  Rate: {:.1} polytopes/s", rate);

    // Final count check
    let final_count: u64 = {
        let conn = db.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM heuristics", [], |row| row.get(0))?
    };
    log::info!("  Total in database: {}", final_count);

    Ok(())
}
