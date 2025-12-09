//! Configuration loading and CLI argument parsing

use clap::Parser;
use serde::Deserialize;

/// Default number of algorithms per generation
pub const DEFAULT_ALGORITHMS_PER_GENERATION: i32 = 16;

/// Default trials required per algorithm
pub const DEFAULT_TRIALS_REQUIRED: i32 = 10;

#[derive(Parser, Debug)]
#[command(name = "search")]
#[command(about = "Meta-GA worker for string theory landscape exploration")]
pub struct Args {
    /// Path to config file
    #[arg(short = 'c', long, default_value = "config.toml")]
    pub config: String,

    /// Path to polytope database (overrides config)
    #[arg(short = 'p', long)]
    pub polytopes: Option<String>,

    /// Verbose debug output (show each individual evaluation)
    #[arg(short = 'v', long)]
    pub verbose: bool,

    /// Specific polytope IDs to search (comma-separated or @filename)
    #[arg(long, value_delimiter = ',')]
    pub ids: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub paths: PathsConfig,
    #[serde(default)]
    pub meta_ga: MetaGaConfig,
}

#[derive(Debug, Deserialize)]
pub struct PathsConfig {
    #[serde(default = "default_polytopes")]
    pub polytopes: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    #[serde(default = "default_database")]
    pub database: String,
    /// Optional: path to HNSW heuristics index (default: derived from database path)
    #[serde(default)]
    pub heuristics_index: Option<String>,
}

fn default_polytopes() -> String {
    "polytopes_three_gen.jsonl".to_string()
}
fn default_output_dir() -> String {
    "results".to_string()
}
fn default_database() -> String {
    "data/string_theory.db".to_string()
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            polytopes: default_polytopes(),
            output_dir: default_output_dir(),
            database: default_database(),
            heuristics_index: None,
        }
    }
}

impl PathsConfig {
    /// Get the heuristics index path, deriving from database path if not explicitly set
    pub fn get_index_path(&self) -> String {
        self.heuristics_index.clone().unwrap_or_else(|| {
            // Derive from database path: data/string_theory.db -> data/heuristics.usearch
            let db_path = std::path::Path::new(&self.database);
            let db_dir = db_path.parent().unwrap_or(std::path::Path::new("data"));
            db_dir.join("heuristics.usearch").to_string_lossy().to_string()
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct MetaGaConfig {
    #[serde(default = "default_algorithms_per_gen")]
    pub algorithms_per_generation: i32,
    #[serde(default = "default_runs_required")]
    pub runs_required: i32,
    #[serde(default = "default_master_seed")]
    pub master_seed: u64,
}

fn default_algorithms_per_gen() -> i32 {
    DEFAULT_ALGORITHMS_PER_GENERATION
}
fn default_runs_required() -> i32 {
    DEFAULT_TRIALS_REQUIRED
}
fn default_master_seed() -> u64 {
    // Use a fixed default seed for reproducibility
    // The answer to life, the universe, and everything
    42
}

impl Default for MetaGaConfig {
    fn default() -> Self {
        Self {
            algorithms_per_generation: default_algorithms_per_gen(),
            runs_required: default_runs_required(),
            master_seed: default_master_seed(),
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_else(|e| {
                eprintln!("Warning: Failed to parse {}: {}", path, e);
                Config::default()
            }),
            Err(_) => {
                eprintln!("Warning: No config file at {}, using defaults", path);
                Config::default()
            }
        }
    }
}

/// Parse polytope IDs from CLI arguments
pub fn parse_polytope_ids(args_ids: Option<Vec<String>>) -> Option<Vec<usize>> {
    let ids = args_ids?;
    let mut result = Vec::new();

    for arg in ids {
        if arg.starts_with('@') {
            let filename = &arg[1..];
            match std::fs::read_to_string(filename) {
                Ok(content) => {
                    for line in content.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        for part in line.split(',') {
                            if let Ok(id) = part.trim().parse::<usize>() {
                                result.push(id);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to read IDs file '{}': {}", filename, e);
                }
            }
        } else if let Ok(id) = arg.parse::<usize>() {
            result.push(id);
        }
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}
