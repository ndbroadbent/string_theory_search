//! String Theory Landscape Explorer - Meta-GA Worker
//!
//! This binary runs as a meta-GA worker that:
//! 1. Acquires an algorithm from the database (or creates generation 0 if empty)
//! 2. Runs one trial using the algorithm's parameters
//! 3. Records trial results
//! 4. Marks algorithm complete when all trials done
//! 5. Loops back to acquire next algorithm

mod config;
mod heartbeat;
mod trial;
mod worker;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use clap::Parser;

use string_theory::constants;
use string_theory::db;
use string_theory::physics::init_physics_bridge;

use config::{Args, Config, parse_polytope_ids};
use heartbeat::HeartbeatThread;
use worker::run_worker_loop;

fn main() {
    env_logger::init();
    let args = Args::parse();

    let config = Config::load(&args.config);
    let polytope_path = args.polytopes.unwrap_or(config.paths.polytopes.clone());
    let polytope_filter = parse_polytope_ids(args.ids);

    let my_pid = std::process::id() as i32;

    print_banner(my_pid);

    // Initialize Python bridge - REQUIRED
    println!("Initializing physics bridge (CYTools + cymyc)...");
    init_physics_bridge().expect("Physics bridge REQUIRED. Install CYTools and cymyc.");
    println!("  Physics bridge ready");

    // Initialize database
    println!("Initializing database at {}...", config.paths.database);
    let db_conn =
        db::init_database(&config.paths.database).expect("Failed to initialize database");
    println!("  Database ready");
    println!();

    print_target_constants();

    // Set up Ctrl+C handler
    let interrupt_flag = Arc::new(AtomicBool::new(false));
    setup_interrupt_handler(interrupt_flag.clone());

    // Start heartbeat thread
    let heartbeat = HeartbeatThread::start(config.paths.database.clone(), my_pid);

    // Run the main worker loop
    run_worker_loop(
        &config,
        db_conn,
        &polytope_path,
        polytope_filter,
        args.verbose,
        interrupt_flag,
        &heartbeat,
        my_pid,
    );

    // Clean shutdown
    heartbeat.stop();
    println!();
    println!("Worker exiting.");
}

fn print_banner(pid: i32) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STRING THEORY LANDSCAPE EXPLORER - Meta-GA Worker");
    println!("  PID: {}", pid);
    println!("═══════════════════════════════════════════════════════════════");
    println!();
}

fn print_target_constants() {
    println!("Target constants:");
    println!("  alpha_em     = {:.6e}", constants::ALPHA_EM);
    println!("  alpha_s      = {:.4}", constants::ALPHA_STRONG);
    println!("  sin2_theta_W = {:.5}", constants::SIN2_THETA_W);
    println!("  N_gen        = {}", constants::NUM_GENERATIONS);
    println!("  Lambda       = {:.3e}", constants::COSMOLOGICAL_CONSTANT);
    println!();
}

fn setup_interrupt_handler(interrupt_flag: Arc<AtomicBool>) {
    let interrupt_count = Arc::new(AtomicUsize::new(0));
    let ic = interrupt_count.clone();
    let if_clone = interrupt_flag.clone();

    ctrlc::set_handler(move || {
        let count = ic.fetch_add(1, Ordering::SeqCst);
        if_clone.store(true, Ordering::SeqCst);
        if count == 0 {
            eprintln!("\nInterrupt received, will exit after current trial...");
        } else {
            eprintln!("\nForce quit.");
            std::process::exit(1);
        }
    })
    .expect("Error setting Ctrl-C handler");
}
