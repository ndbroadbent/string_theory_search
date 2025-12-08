//! Real Physics String Theory Landscape Explorer
//!
//! This binary uses actual physics computations from JAX/cymyc
//! instead of the toy approximations.
//!
//! SAVES RESULTS AUTOMATICALLY - never lose a good compactification!
//!
//! Usage: real_physics [RUN_ID]
//!   RUN_ID: optional identifier (0-99), auto-generated if not provided

use std::time::{Duration, Instant};

use string_theory::constants;
use string_theory::physics::{init_physics_bridge, is_physics_available};
use string_theory::real_genetic::{
    format_real_fitness_line, format_real_fitness_report, RealGaConfig, RealLandscapeSearcher,
};

const MAX_GENERATIONS: usize = 100000;
const MAX_RUNTIME_SECS: u64 = 86400; // 24 hours max

fn main() {
    env_logger::init();

    // Get run ID from args or generate random
    let run_id: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| rand::random::<u32>() % 100);

    let output_dir = format!("results/run_{:02}", run_id);

    // Create output directory
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    println!("═══════════════════════════════════════════════════════════════");
    println!("  STRING THEORY LANDSCAPE EXPLORER - REAL PHYSICS MODE");
    println!("  Run ID: {:02}", run_id);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Initialize Python bridge
    println!("Initializing Python/JAX physics bridge...");
    match init_physics_bridge() {
        Ok(()) => println!("✓ Physics bridge initialized"),
        Err(e) => {
            eprintln!("✗ Failed to initialize physics bridge: {}", e);
            eprintln!("  Using fallback fitness function");
        }
    }
    println!();

    // Load polytope data - use medium dataset if available, fall back to small
    let polytope_path = if std::path::Path::new("polytopes_medium.json").exists() {
        "polytopes_medium.json"
    } else {
        "polytopes_small.json"
    };
    println!("Loading polytope database from {}...", polytope_path);

    // Configure the GA
    let config = RealGaConfig {
        population_size: 500,
        elite_count: 20,
        tournament_size: 5,
        crossover_rate: 0.85,
        base_mutation_rate: 0.4,
        base_mutation_strength: 0.35,
        collapse_threshold: 1000,
        hall_of_fame_size: 100,
    };

    let mut searcher = RealLandscapeSearcher::new(config.clone(), polytope_path);

    println!("Population size: {}", config.population_size);
    println!("Physics bridge available: {}", is_physics_available());
    println!("Output directory: {}", output_dir);
    println!();
    println!("Target constants:");
    println!("  α_em = {:.6e} (fine structure)", constants::ALPHA_EM);
    println!("  α_s  = {:.4} (strong coupling)", constants::ALPHA_STRONG);
    println!("  sin²θ_W = {:.5} (Weinberg angle)", constants::SIN2_THETA_W);
    println!("  N_gen = {} (fermion generations)", constants::NUM_GENERATIONS);
    println!("  Λ = {:.3e} (cosmological constant)", constants::COSMOLOGICAL_CONSTANT);
    println!();
    println!("Starting search (Ctrl+C to stop, state will be saved)...");
    println!();

    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut last_best_fitness = 0.0;
    let mut last_saved_fitness = 0.0; // Track what we've actually saved
    let mut best_lambda_magnitude: f64 = 0.0; // Track best cosmological constant (closest to target)

    // Set up Ctrl+C handler
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n\n🛑 Interrupt received, saving state...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        searcher.step();

        // Report progress
        if let Some(best) = searcher.best() {
            let current_lambda = best.physics.as_ref()
                .map(|p| p.cosmological_constant.abs())
                .unwrap_or(f64::MAX);

            // Check if this is a new best (by fitness OR by better cosmological constant)
            let is_new_best = best.fitness > last_best_fitness * 1.001;
            let is_better_lambda = current_lambda < best_lambda_magnitude * 0.1
                && current_lambda > 0.0
                && current_lambda < 1e-10; // Only care about small lambdas

            if is_new_best || is_better_lambda {
                if is_new_best {
                    last_best_fitness = best.fitness;
                }
                if is_better_lambda || best_lambda_magnitude == 0.0 {
                    best_lambda_magnitude = current_lambda;
                }

                let stats = searcher.history.last().unwrap();
                println!(
                    "🎯 Gen {:5} | Eval: {:>8} | 🌠:{} | {}",
                    searcher.generation,
                    searcher.total_evaluated,
                    stats.landscape_collapses,
                    format_real_fitness_line(best)
                );

                // Only save if fitness actually improved (not just reported)
                if best.fitness > last_saved_fitness * 1.001 {
                    last_saved_fitness = best.fitness;
                    match searcher.save_best_with_fitness(&output_dir) {
                        Ok(filename) => println!("   💾 Saved to {}", filename),
                        Err(e) => eprintln!("   ⚠️  Failed to save: {}", e),
                    }
                }
            } else if last_report.elapsed() > Duration::from_secs(10) {
                // Periodic status
                last_report = Instant::now();
                let stats = searcher.history.last().unwrap();
                println!(
                    "   Gen {:5} | Eval: {:>8} | stag:{:4} | 🌠:{} | ok:{}/fail:{}",
                    searcher.generation,
                    searcher.total_evaluated,
                    stats.stagnation_generations,
                    stats.landscape_collapses,
                    stats.physics_successes,
                    stats.physics_failures,
                );
            }
        }

        // Check termination conditions
        if searcher.generation >= MAX_GENERATIONS {
            println!("\n✓ Reached max generations ({})", MAX_GENERATIONS);
            break;
        }
        if start.elapsed() > Duration::from_secs(MAX_RUNTIME_SECS) {
            println!("\n✓ Reached max runtime ({}s)", MAX_RUNTIME_SECS);
            break;
        }
    }

    // Final save
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FINAL SAVE");
    println!("═══════════════════════════════════════════════════════════════");

    match searcher.save_state(&output_dir) {
        Ok(filename) => println!("💾 Final state saved to {}", filename),
        Err(e) => eprintln!("⚠️  Failed to save final state: {}", e),
    }
    match searcher.save_best_with_fitness(&output_dir) {
        Ok(filename) => println!("💾 Best individual saved to {}", filename),
        Err(e) => eprintln!("⚠️  Failed to save best: {}", e),
    }

    // Final report
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FINAL REPORT");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Generations: {}", searcher.generation);
    println!("Total evaluated: {}", searcher.total_evaluated);
    println!("Elapsed: {:.1}s", start.elapsed().as_secs_f64());
    println!("Hall of fame size: {}", searcher.hall_of_fame.len());
    println!();

    if let Some(best) = searcher.best_ever.as_ref() {
        println!("Best ever found:");
        println!("{}", format_real_fitness_report(best));
    }

    println!();
    println!("All results saved to {}/", output_dir);
}
