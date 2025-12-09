//! String Theory Landscape Explorer
//!
//! Search through Calabi-Yau compactifications using genetic algorithms
//! to find configurations that reproduce Standard Model physics.

use std::time::{Duration, Instant};

use clap::Parser;
use serde::Deserialize;
use string_theory::constants;
use string_theory::physics::init_physics_bridge;
use string_theory::searcher::{format_fitness_line, format_fitness_report, GaConfig, LandscapeSearcher};

#[derive(Parser, Debug)]
#[command(name = "search")]
#[command(about = "Search the string theory landscape for Standard Model compactifications")]
struct Args {
    /// Path to config file
    #[arg(short = 'c', long, default_value = "config.toml")]
    config: String,

    /// Maximum runtime in seconds (overrides config)
    #[arg(short = 't', long)]
    max_time: Option<u64>,

    /// Maximum generations (overrides config)
    #[arg(short = 'g', long)]
    max_generations: Option<usize>,

    /// Path to polytope database (overrides config)
    #[arg(short = 'p', long)]
    polytopes: Option<String>,

    /// Run ID for output directory (0-99, random if not specified)
    #[arg(short = 'r', long)]
    run_id: Option<u32>,

    /// Population size (overrides config)
    #[arg(long)]
    population: Option<usize>,

    /// Verbose debug output (show each individual evaluation)
    #[arg(short = 'v', long)]
    verbose: bool,
}

#[derive(Debug, Deserialize, Default)]
struct Config {
    #[serde(default)]
    paths: PathsConfig,
    #[serde(default)]
    search: SearchConfig,
    #[serde(default)]
    limits: LimitsConfig,
}

#[derive(Debug, Deserialize)]
struct PathsConfig {
    #[serde(default = "default_polytopes")]
    polytopes: String,
    #[serde(default = "default_output_dir")]
    output_dir: String,
}

fn default_polytopes() -> String { "polytopes_three_gen.jsonl".to_string() }
fn default_output_dir() -> String { "results".to_string() }

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            polytopes: default_polytopes(),
            output_dir: default_output_dir(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SearchConfig {
    #[serde(default = "default_population")]
    population_size: usize,
    #[serde(default = "default_elite")]
    elite_count: usize,
    #[serde(default = "default_tournament")]
    tournament_size: usize,
    #[serde(default = "default_crossover")]
    crossover_rate: f64,
    #[serde(default = "default_mutation_rate")]
    mutation_rate: f64,
    #[serde(default = "default_mutation_strength")]
    mutation_strength: f64,
    #[serde(default = "default_collapse")]
    collapse_threshold: usize,
    #[serde(default = "default_hof")]
    hall_of_fame_size: usize,
}

fn default_population() -> usize { 500 }
fn default_elite() -> usize { 20 }
fn default_tournament() -> usize { 5 }
fn default_crossover() -> f64 { 0.85 }
fn default_mutation_rate() -> f64 { 0.4 }
fn default_mutation_strength() -> f64 { 0.35 }
fn default_collapse() -> usize { 1000 }
fn default_hof() -> usize { 100 }

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            population_size: default_population(),
            elite_count: default_elite(),
            tournament_size: default_tournament(),
            crossover_rate: default_crossover(),
            mutation_rate: default_mutation_rate(),
            mutation_strength: default_mutation_strength(),
            collapse_threshold: default_collapse(),
            hall_of_fame_size: default_hof(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct LimitsConfig {
    #[serde(default = "default_max_time")]
    max_time: u64,
    #[serde(default = "default_max_gen")]
    max_generations: usize,
}

fn default_max_time() -> u64 { 86400 }
fn default_max_gen() -> usize { 100000 }

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_time: default_max_time(),
            max_generations: default_max_gen(),
        }
    }
}

fn load_config(path: &str) -> Config {
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

fn main() {
    env_logger::init();
    let args = Args::parse();

    // Load config, CLI args override
    let config = load_config(&args.config);

    let max_time = args.max_time.unwrap_or(config.limits.max_time);
    let max_generations = args.max_generations.unwrap_or(config.limits.max_generations);
    let polytope_path = args.polytopes.unwrap_or(config.paths.polytopes);
    let population_size = args.population.unwrap_or(config.search.population_size);

    let run_id = args.run_id.unwrap_or_else(|| rand::random::<u32>() % 100);
    let output_dir = format!("{}/run_{:02}", config.paths.output_dir, run_id);
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    println!("═══════════════════════════════════════════════════════════════");
    println!("  STRING THEORY LANDSCAPE EXPLORER");
    println!("  Run ID: {:02}", run_id);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Initialize Python bridge - REQUIRED
    println!("Initializing physics bridge (CYTools + cymyc)...");
    init_physics_bridge().expect("Physics bridge REQUIRED. Install CYTools and cymyc.");
    println!("  Physics bridge ready");
    println!();

    println!("Loading polytopes from {}...", polytope_path);

    let ga_config = GaConfig {
        population_size,
        elite_count: config.search.elite_count,
        tournament_size: config.search.tournament_size,
        crossover_rate: config.search.crossover_rate,
        base_mutation_rate: config.search.mutation_rate,
        base_mutation_strength: config.search.mutation_strength,
        collapse_threshold: config.search.collapse_threshold,
        hall_of_fame_size: config.search.hall_of_fame_size,
    };

    let mut searcher = LandscapeSearcher::new(ga_config.clone(), &polytope_path);
    searcher.verbose = args.verbose;

    println!("Population size: {}", ga_config.population_size);
    println!("Max time: {}s", max_time);
    println!("Max generations: {}", max_generations);
    println!("Output: {}", output_dir);
    println!();
    println!("Target constants:");
    println!("  alpha_em     = {:.6e}", constants::ALPHA_EM);
    println!("  alpha_s      = {:.4}", constants::ALPHA_STRONG);
    println!("  sin2_theta_W = {:.5}", constants::SIN2_THETA_W);
    println!("  N_gen        = {}", constants::NUM_GENERATIONS);
    println!("  Lambda       = {:.3e}", constants::COSMOLOGICAL_CONSTANT);
    println!();
    println!("Starting search (Ctrl+C to stop)...");
    println!();

    let start = Instant::now();
    let max_duration = Duration::from_secs(max_time);
    let mut last_best_fitness = 0.0;
    let mut last_saved_fitness = 0.0;
    let mut best_lambda_magnitude: f64 = 0.0;

    // Ctrl+C handler - exit immediately on second interrupt
    let interrupt_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ic = interrupt_count.clone();
    ctrlc::set_handler(move || {
        let count = ic.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count == 0 {
            eprintln!("\nInterrupt received, will exit after current evaluation...");
        } else {
            eprintln!("\nForce quit.");
            std::process::exit(1);
        }
    }).expect("Error setting Ctrl-C handler");

    while interrupt_count.load(std::sync::atomic::Ordering::SeqCst) == 0 {
        // Log start of generation
        eprint!("Gen {:5} evaluating {} individuals... ", searcher.generation + 1, ga_config.population_size);
        let gen_start = Instant::now();

        searcher.step();

        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let stats = searcher.history.last().unwrap();

        if let Some(best) = searcher.best() {
            let current_lambda = best.physics.as_ref()
                .map(|p| p.cosmological_constant.abs())
                .unwrap_or(f64::MAX);

            let is_new_best = best.fitness > last_best_fitness * 1.001;
            let is_better_lambda = current_lambda < best_lambda_magnitude * 0.1
                && current_lambda > 0.0
                && current_lambda < 1e-10;

            // Always print generation summary
            eprintln!(
                "{:.1}s | ok:{} fail:{} | best:{:.4}",
                gen_elapsed,
                stats.physics_successes,
                stats.physics_failures,
                best.fitness
            );

            if is_new_best || is_better_lambda {
                if is_new_best {
                    last_best_fitness = best.fitness;
                }
                if is_better_lambda || best_lambda_magnitude == 0.0 {
                    best_lambda_magnitude = current_lambda;
                }

                println!(
                    "  NEW BEST | {}",
                    format_fitness_line(best)
                );

                if best.fitness > last_saved_fitness * 1.001 {
                    last_saved_fitness = best.fitness;
                    if let Ok(filename) = searcher.save_best_with_fitness(&output_dir) {
                        println!("  Saved: {}", filename);
                    }
                }
            }
        } else {
            eprintln!("{:.1}s | no results", gen_elapsed);
        }

        // Check termination
        if searcher.generation >= max_generations {
            println!("\nReached max generations ({})", max_generations);
            break;
        }
        if start.elapsed() > max_duration {
            println!("\nReached max time ({}s)", max_time);
            break;
        }
    }

    // Final save
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SAVING");
    println!("═══════════════════════════════════════════════════════════════");

    if let Ok(f) = searcher.save_state(&output_dir) {
        println!("State saved to {}", f);
    }
    if let Ok(f) = searcher.save_best_with_fitness(&output_dir) {
        println!("Best saved to {}", f);
    }
    if let Ok(()) = searcher.save_cluster_state() {
        println!("Cluster state saved ({} clusters)", searcher.cluster_state.clusters.len());
    }

    // Report
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Generations: {}", searcher.generation);
    println!("Evaluated: {}", searcher.total_evaluated);
    println!("Elapsed: {:.1}s", start.elapsed().as_secs_f64());
    println!("Hall of fame: {}", searcher.hall_of_fame.len());
    println!();

    if let Some(best) = searcher.best_ever.as_ref() {
        println!("Best found:");
        println!("{}", format_fitness_report(best));
    }

    println!();
    println!("Results saved to {}/", output_dir);
}
