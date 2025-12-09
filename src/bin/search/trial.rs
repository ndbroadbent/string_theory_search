//! Trial execution - running one trial of an algorithm

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rusqlite::Connection;

use string_theory::constants;
use string_theory::db::{self, MetaAlgorithm, Run};
use string_theory::searcher::{format_fitness_line, GaConfig, LandscapeSearcher, SearchStrategy};

/// Convert MetaAlgorithm to GaConfig for the inner GA
pub fn algorithm_to_ga_config(algo: &MetaAlgorithm) -> GaConfig {
    GaConfig {
        population_size: algo.population_size as usize,
        elite_count: algo.elite_count as usize,
        tournament_size: algo.tournament_size as usize,
        crossover_rate: algo.crossover_rate,
        base_mutation_rate: algo.mutation_rate,
        base_mutation_strength: algo.mutation_strength,
        collapse_threshold: 1000,
        hall_of_fame_size: 100,
    }
}

/// Run one trial of an algorithm and return the results
pub fn run_trial(
    algo: &MetaAlgorithm,
    run_number: i32,
    polytope_path: &str,
    polytope_filter: Option<Vec<usize>>,
    db_conn: Arc<Mutex<Connection>>,
    verbose: bool,
    output_dir: &str,
    interrupt_flag: &Arc<AtomicBool>,
) -> Run {
    let ga_config = algorithm_to_ga_config(algo);
    let search_strategy = SearchStrategy::from_meta_algorithm(algo);
    let max_generations = algo.max_generations as usize;

    // Derive deterministic run seed from algorithm seed and run number
    let run_seed = db::derive_run_seed(algo.rng_seed, run_number);
    println!("  Run seed: {} (derived from algo_seed={}, run={})",
             run_seed, algo.rng_seed, run_number);

    // Note: run_id is None here - evaluations won't be linked to a run record
    // The Run record is created after the trial completes with all the metrics
    let mut searcher = LandscapeSearcher::new_with_seed(
        ga_config.clone(),
        search_strategy,
        polytope_path,
        polytope_filter,
        Some(db_conn.clone()),
        None,  // No run_id yet - will be set after insert_run
        Some(run_seed),
    );
    searcher.verbose = verbose;

    // Track fitness over generations for AUC calculation
    let mut fitness_history: Vec<f64> = Vec::new();
    let mut unique_polytopes: HashSet<usize> = HashSet::new();
    let mut physics_successes = 0usize;
    let mut physics_failures = 0usize;

    // Run first generation to get initial fitness
    searcher.step();
    let initial_fitness = searcher.best().map(|b| b.fitness).unwrap_or(0.0);
    fitness_history.push(initial_fitness);

    // Track stats from first gen
    if let Some(stats) = searcher.history.last() {
        physics_successes += stats.physics_successes;
        physics_failures += stats.physics_failures;
    }
    for ind in &searcher.population {
        unique_polytopes.insert(ind.genome.polytope_id);
    }

    let mut last_best_fitness = initial_fitness;

    // Run remaining generations
    for gen in 2..=max_generations {
        if interrupt_flag.load(Ordering::Relaxed) {
            eprintln!("Interrupt received, stopping trial early");
            break;
        }

        eprint!(
            "  Gen {:3}/{} evaluating {} ... ",
            gen, max_generations, ga_config.population_size
        );
        let gen_start = Instant::now();

        searcher.step();

        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let stats = searcher.history.last().unwrap();
        physics_successes += stats.physics_successes;
        physics_failures += stats.physics_failures;

        for ind in &searcher.population {
            unique_polytopes.insert(ind.genome.polytope_id);
        }

        let current_best = searcher.best().map(|b| b.fitness).unwrap_or(0.0);
        fitness_history.push(current_best);

        eprintln!(
            "{:.1}s | ok:{} fail:{} | best:{:.4}",
            gen_elapsed, stats.physics_successes, stats.physics_failures, current_best
        );

        if current_best > last_best_fitness * 1.001 {
            last_best_fitness = current_best;
            if let Some(best) = searcher.best() {
                println!("  NEW BEST | {}", format_fitness_line(best));
            }
        }
    }

    // Save best if good
    let final_fitness = fitness_history.last().copied().unwrap_or(0.0);
    if final_fitness > 0.3 {
        if let Ok(filename) = searcher.save_best_with_fitness(output_dir) {
            println!("  Saved: {}", filename);
        }
    }

    // Compute run metrics
    compute_run_metrics(
        algo.id.unwrap_or(0),
        run_number,
        &fitness_history,
        initial_fitness,
        &searcher,
        physics_successes,
        physics_failures,
        unique_polytopes.len(),
    )
}

fn compute_run_metrics(
    algorithm_id: i64,
    run_number: i32,
    fitness_history: &[f64],
    initial_fitness: f64,
    searcher: &LandscapeSearcher,
    physics_successes: usize,
    physics_failures: usize,
    unique_polytopes_count: usize,
) -> Run {
    let generations_run = fitness_history.len() as i32;
    let final_fitness = fitness_history.last().copied().unwrap_or(0.0);
    let fitness_improvement = final_fitness - initial_fitness;
    let improvement_rate = if generations_run > 1 {
        fitness_improvement / (generations_run - 1) as f64
    } else {
        0.0
    };

    // Area under curve (simple trapezoidal)
    let fitness_auc: f64 = fitness_history
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .sum();

    // Best CC log error
    let best_cc_log_error = searcher
        .best_ever
        .as_ref()
        .and_then(|b| b.physics.as_ref())
        .map(|p| {
            let cc = p.cosmological_constant;
            if cc.abs() > 1e-200 && constants::COSMOLOGICAL_CONSTANT.abs() > 1e-200 {
                (cc / constants::COSMOLOGICAL_CONSTANT).abs().log10().abs()
            } else {
                200.0
            }
        })
        .unwrap_or(200.0);

    let total_evals = physics_successes + physics_failures;
    let physics_success_rate = if total_evals > 0 {
        physics_successes as f64 / total_evals as f64
    } else {
        0.0
    };

    Run {
        id: None,
        algorithm_id,
        run_number,
        generations_run,
        initial_fitness,
        final_fitness,
        fitness_improvement,
        improvement_rate,
        fitness_auc,
        best_cc_log_error,
        physics_success_rate,
        unique_polytopes_tried: unique_polytopes_count as i32,
    }
}
