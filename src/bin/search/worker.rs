//! Worker loop for the meta-GA

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rusqlite::Connection;

use string_theory::db;
use string_theory::meta_ga;

use crate::config::{Config, DEFAULT_ALGORITHMS_PER_GENERATION};
use crate::heartbeat::HeartbeatThread;
use crate::trial::run_trial;

/// Meta-GA evolution parameters
const META_ELITE_COUNT: i32 = 4;
const META_MUTATION_RATE: f64 = 0.4;
const META_MUTATION_STRENGTH: f64 = 0.5;

/// Run the main worker loop
pub fn run_worker_loop(
    config: &Config,
    db_conn: Arc<Mutex<Connection>>,
    polytope_path: &str,
    polytope_filter: Option<Vec<usize>>,
    verbose: bool,
    interrupt_flag: Arc<AtomicBool>,
    heartbeat: &HeartbeatThread,
    my_pid: i32,
) {
    println!("Starting meta-GA worker loop...");
    println!();

    loop {
        if interrupt_flag.load(Ordering::Relaxed) {
            println!("Interrupt flag set, exiting worker loop");
            break;
        }

        // Try to acquire an algorithm
        let acquisition = acquire_algorithm(&config, &db_conn, my_pid);

        let (algo_id, algo) = match acquisition {
            AcquisitionResult::Acquired(id, algo) => (id, algo),
            AcquisitionResult::NoneAvailable => {
                println!("No available algorithms, waiting 30s...");
                thread::sleep(Duration::from_secs(30));
                continue;
            }
            AcquisitionResult::Error(e) => {
                eprintln!("Failed to acquire algorithm: {}", e);
                thread::sleep(Duration::from_secs(5));
                continue;
            }
            AcquisitionResult::Fatal(e) => {
                eprintln!("Fatal error: {}", e);
                break;
            }
        };

        // Update heartbeat thread with current algorithm ID
        heartbeat.set_algorithm(algo_id);

        // Get current trial count
        let trial_number = {
            let conn = db_conn.lock().unwrap();
            meta_ga::get_trial_count(&conn, algo_id).unwrap_or(0) + 1
        };

        print_algorithm_header(&algo, algo_id, trial_number);

        // Generate unique run ID and output directory
        let run_id = format!("meta_{}_{}", algo_id, trial_number);
        let output_dir = format!("{}/algo_{}", config.paths.output_dir, algo_id);
        std::fs::create_dir_all(&output_dir).ok();

        // Run the trial
        let trial_start = Instant::now();
        let trial = run_trial(
            &algo,
            polytope_path,
            polytope_filter.clone(),
            db_conn.clone(),
            run_id,
            verbose,
            &output_dir,
            &interrupt_flag,
        );
        let trial_elapsed = trial_start.elapsed();

        print_trial_summary(&trial, trial_number, trial_elapsed.as_secs_f64());

        // Record trial result and check completion
        record_trial_and_check_completion(&db_conn, &trial, algo_id, &algo, my_pid);

        // Clear the algo ID for heartbeat
        heartbeat.clear_algorithm();

        // Small delay before next acquisition
        thread::sleep(Duration::from_millis(500));
    }
}

enum AcquisitionResult {
    Acquired(i64, db::MetaAlgorithm),
    NoneAvailable,
    Error(String),
    Fatal(String),
}

fn acquire_algorithm(
    config: &Config,
    db_conn: &Arc<Mutex<Connection>>,
    my_pid: i32,
) -> AcquisitionResult {
    let conn = db_conn.lock().unwrap();

    // Check if we need to initialize generation 0
    let algo_count = db::count_algorithms_in_generation(&conn, 0).unwrap_or(0);
    if algo_count == 0 {
        println!("No algorithms in database, initializing generation 0...");
        if let Err(e) = meta_ga::init_generation_zero(
            &conn,
            config.meta_ga.algorithms_per_generation,
            config.meta_ga.trials_required,
        ) {
            return AcquisitionResult::Fatal(format!("Failed to initialize generation 0: {}", e));
        }
        println!(
            "Created {} algorithms for generation 0",
            config.meta_ga.algorithms_per_generation
        );
    }

    // Check if current generation is complete and we need to evolve
    let (current_gen, _) =
        db::get_meta_state(&conn).unwrap_or((0, DEFAULT_ALGORITHMS_PER_GENERATION));
    if db::is_generation_complete(&conn, current_gen).unwrap_or(false) {
        let next_gen_count = db::count_algorithms_in_generation(&conn, current_gen + 1).unwrap_or(0);
        if next_gen_count == 0 {
            println!(
                "Generation {} complete, evolving to generation {}...",
                current_gen,
                current_gen + 1
            );
            if let Err(e) = meta_ga::evolve_next_generation(
                &conn,
                current_gen,
                config.meta_ga.algorithms_per_generation,
                config.meta_ga.trials_required,
                META_ELITE_COUNT,
                META_MUTATION_RATE,
                META_MUTATION_STRENGTH,
            ) {
                eprintln!("Failed to evolve next generation: {}", e);
            } else {
                println!("Created generation {}", current_gen + 1);
            }
        }
    }

    // Try to acquire an algorithm
    match db::try_acquire_algorithm(&conn, my_pid) {
        Ok(Some(id)) => match db::get_meta_algorithm(&conn, id) {
            Ok(Some(algo)) => AcquisitionResult::Acquired(id, algo),
            Ok(None) => AcquisitionResult::Error(format!(
                "Algorithm {} not found after acquiring",
                id
            )),
            Err(e) => AcquisitionResult::Error(format!("Failed to get algorithm {}: {}", id, e)),
        },
        Ok(None) => AcquisitionResult::NoneAvailable,
        Err(e) => AcquisitionResult::Error(e.to_string()),
    }
}

fn print_algorithm_header(algo: &db::MetaAlgorithm, algo_id: i64, trial_number: i32) {
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  ALGORITHM {} (gen {}) - Trial {}/{}",
        algo_id, algo.meta_generation, trial_number, algo.trials_required
    );
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Population: {}, Generations: {}",
        algo.population_size, algo.max_generations
    );
    println!(
        "  Mutation rate: {:.2}, Crossover rate: {:.2}",
        algo.mutation_rate, algo.crossover_rate
    );
    println!(
        "  Similarity radius: {:.2}, Interpolation weight: {:.2}",
        algo.similarity_radius, algo.interpolation_weight
    );
    println!();
}

fn print_trial_summary(trial: &db::MetaTrial, trial_number: i32, elapsed_secs: f64) {
    println!();
    println!("Trial {} completed in {:.1}s", trial_number, elapsed_secs);
    println!("  Initial fitness: {:.4}", trial.initial_fitness);
    println!("  Final fitness: {:.4}", trial.final_fitness);
    println!("  Improvement rate: {:.6}", trial.improvement_rate);
    println!("  CC log error: {:.1}", trial.best_cc_log_error);
    println!();
}

fn record_trial_and_check_completion(
    db_conn: &Arc<Mutex<Connection>>,
    trial: &db::MetaTrial,
    algo_id: i64,
    algo: &db::MetaAlgorithm,
    my_pid: i32,
) {
    let conn = db_conn.lock().unwrap();

    if let Err(e) = db::insert_meta_trial(&conn, trial) {
        eprintln!("Failed to insert trial result: {}", e);
    }

    // Check if algorithm is complete
    let trial_count = meta_ga::get_trial_count(&conn, algo_id).unwrap_or(0);
    if trial_count >= algo.trials_required {
        println!(
            "Algorithm {} completed all {} trials",
            algo_id, algo.trials_required
        );
        if let Err(e) = db::complete_algorithm(&conn, algo_id, my_pid) {
            eprintln!("Failed to mark algorithm complete: {}", e);
        }
    } else {
        println!(
            "Algorithm {} has {}/{} trials complete",
            algo_id, trial_count, algo.trials_required
        );
    }
}
