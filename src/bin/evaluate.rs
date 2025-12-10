//! Evaluate specific compactification configurations
//!
//! This binary allows evaluating:
//! 1. Custom configurations from command line arguments (for playground)
//! 2. Custom configurations from JSON files
//!
//! Usage:
//!   # Evaluate with specific parameters (playground mode)
//!   ./evaluate --vertices "[[0,0,0,0],[-1,2,-1,-1],...]" \
//!              --g-s 0.0091 \
//!              --flux-k "[-3,-5,8,6]" \
//!              --flux-m "[10,11,-11,-5]" \
//!              --kahler "[1.0, 2.0]" \
//!              --complex "[1.0]" \
//!              --json --save --source playground
//!
//!   # Evaluate from JSON file
//!   ./evaluate --file config.json

use clap::Parser;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use string_theory::physics::{
    init_physics_bridge, compute_physics, Compactification, PhysicsOutput,
};
use string_theory::constants;

#[derive(Parser, Debug)]
#[command(name = "evaluate")]
#[command(about = "Evaluate specific string compactification configurations")]
struct Args {
    /// Config file path
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Evaluate from JSON file
    #[arg(long)]
    file: Option<PathBuf>,

    /// Polytope vertices as JSON array (for custom evaluation)
    #[arg(long)]
    vertices: Option<String>,

    /// String coupling g_s
    #[arg(long)]
    g_s: Option<f64>,

    /// K flux vector as JSON array (F_3 flux)
    #[arg(long)]
    flux_k: Option<String>,

    /// M flux vector as JSON array (H_3 flux)
    #[arg(long)]
    flux_m: Option<String>,

    /// Kahler moduli as JSON array
    #[arg(long)]
    kahler: Option<String>,

    /// Complex structure moduli as JSON array
    #[arg(long)]
    complex: Option<String>,

    /// Hodge number h11 (for external polytopes)
    #[arg(long)]
    h11: Option<i32>,

    /// Hodge number h21 (for external polytopes)
    #[arg(long)]
    h21: Option<i32>,

    /// Database path
    #[arg(long, default_value = "data/string_theory.db")]
    database: PathBuf,

    /// Output JSON instead of human-readable text
    #[arg(long)]
    json: bool,

    /// Save result to database
    #[arg(long)]
    save: bool,

    /// Evaluation source (ga, playground, test)
    #[arg(long, default_value = "playground")]
    source: String,

    /// Input hash for caching (SHA256)
    #[arg(long)]
    hash: Option<String>,

    /// Physics model version
    #[arg(long)]
    model_version: Option<String>,

    /// Optional label for the evaluation
    #[arg(long)]
    label: Option<String>,

    /// Verbose output (ignored in JSON mode)
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CustomConfig {
    vertices: Vec<Vec<i32>>,
    g_s: f64,
    kahler_moduli: Vec<f64>,
    complex_moduli: Vec<f64>,
    flux_f: Vec<i32>,
    flux_h: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct JsonOutput {
    success: bool,
    evaluation_id: Option<i64>,
    cached: bool,
    error: Option<String>,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    // In JSON mode, suppress all stdout except the final result
    if !args.json {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  STRING THEORY CONFIGURATION EVALUATOR");
        println!("═══════════════════════════════════════════════════════════════");
        println!();
    }

    // Initialize Python bridge
    if !args.json {
        println!("Initializing physics bridge...");
    }
    if let Err(e) = init_physics_bridge() {
        if args.json {
            let output = JsonOutput {
                success: false,
                evaluation_id: None,
                cached: false,
                error: Some(format!("Failed to initialize physics bridge: {}", e)),
            };
            println!("{}", serde_json::to_string(&output).unwrap());
        } else {
            eprintln!("Failed to initialize physics bridge: {}", e);
            eprintln!("Make sure CYTools and cymyc are installed.");
        }
        std::process::exit(1);
    }
    if !args.json {
        println!("Physics bridge ready.");
        println!();
        print_target_constants();
    }

    if let Some(ref file) = args.file {
        evaluate_from_file(&args, file);
    } else if args.vertices.is_some() {
        evaluate_custom(&args);
    } else {
        if args.json {
            let output = JsonOutput {
                success: false,
                evaluation_id: None,
                cached: false,
                error: Some("No evaluation mode specified. Use --vertices or --file".to_string()),
            };
            println!("{}", serde_json::to_string(&output).unwrap());
        } else {
            eprintln!("No evaluation mode specified. Use --help for options.");
        }
        std::process::exit(1);
    }
}

fn print_target_constants() {
    println!("Target constants:");
    println!("  alpha_em     = {:.6e}", constants::ALPHA_EM);
    println!("  alpha_s      = {:.4}", constants::ALPHA_STRONG);
    println!("  sin2_theta_w = {:.5}", constants::SIN2_THETA_W);
    println!("  N_gen        = {}", constants::NUM_GENERATIONS);
    println!("  Lambda       = {:.3e}", constants::COSMOLOGICAL_CONSTANT);
    println!();
}

fn evaluate_from_file(args: &Args, file: &PathBuf) {
    let content = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => {
            if args.json {
                let output = JsonOutput {
                    success: false,
                    evaluation_id: None,
                    cached: false,
                    error: Some(format!("Failed to read file: {}", e)),
                };
                println!("{}", serde_json::to_string(&output).unwrap());
            } else {
                eprintln!("Failed to read file: {}", e);
            }
            std::process::exit(1);
        }
    };

    let config: CustomConfig = match serde_json::from_str(&content) {
        Ok(c) => c,
        Err(e) => {
            if args.json {
                let output = JsonOutput {
                    success: false,
                    evaluation_id: None,
                    cached: false,
                    error: Some(format!("Invalid JSON: {}", e)),
                };
                println!("{}", serde_json::to_string(&output).unwrap());
            } else {
                eprintln!("Invalid JSON: {}", e);
            }
            std::process::exit(1);
        }
    };

    let h11 = config.kahler_moduli.len() as i32;
    let h21 = config.complex_moduli.len() as i32;

    let genome = Compactification {
        polytope_id: usize::MAX, // External polytope (sentinel value)
        kahler_moduli: config.kahler_moduli,
        complex_moduli: config.complex_moduli,
        flux_f: config.flux_f,
        flux_h: config.flux_h,
        g_s: config.g_s,
        h11,
        h21,
    };

    if !args.json {
        println!("Evaluating custom configuration from {:?}", file);
        println!("  h11 = {}, h21 = {}", h11, h21);
    }

    let result = compute_physics(&genome, &config.vertices);
    handle_result(args, &genome, &config.vertices, &result);
}

fn evaluate_custom(args: &Args) {
    let vertices_str = args.vertices.as_ref().expect("--vertices required");
    let vertices: Vec<Vec<i32>> = match serde_json::from_str(vertices_str) {
        Ok(v) => v,
        Err(e) => {
            if args.json {
                let output = JsonOutput {
                    success: false,
                    evaluation_id: None,
                    cached: false,
                    error: Some(format!("Invalid vertices JSON: {}", e)),
                };
                println!("{}", serde_json::to_string(&output).unwrap());
            } else {
                eprintln!("Invalid vertices JSON: {}", e);
            }
            std::process::exit(1);
        }
    };

    let g_s = args.g_s.unwrap_or(0.01);

    let flux_k: Vec<i32> = args.flux_k.as_ref()
        .map(|s| serde_json::from_str(s).expect("Invalid flux_k JSON"))
        .unwrap_or_default();
    let flux_m: Vec<i32> = args.flux_m.as_ref()
        .map(|s| serde_json::from_str(s).expect("Invalid flux_m JSON"))
        .unwrap_or_default();
    let kahler: Vec<f64> = args.kahler.as_ref()
        .map(|s| serde_json::from_str(s).expect("Invalid kahler JSON"))
        .unwrap_or_else(|| vec![1.0; 4]);
    let complex: Vec<f64> = args.complex.as_ref()
        .map(|s| serde_json::from_str(s).expect("Invalid complex JSON"))
        .unwrap_or_else(|| vec![1.0; 4]);

    // Use explicit h11/h21 if provided, otherwise infer from moduli lengths
    let h11 = args.h11.unwrap_or(kahler.len() as i32);
    let h21 = args.h21.unwrap_or(complex.len() as i32);

    let genome = Compactification {
        polytope_id: usize::MAX, // External polytope (sentinel value)
        kahler_moduli: kahler,
        complex_moduli: complex,
        flux_f: flux_k,
        flux_h: flux_m,
        g_s,
        h11,
        h21,
    };

    if !args.json {
        println!("Evaluating custom configuration");
        println!("  h11 = {}, h21 = {}", h11, h21);
        println!("  g_s = {}", g_s);
    }

    let result = compute_physics(&genome, &vertices);
    handle_result(args, &genome, &vertices, &result);
}

fn handle_result(
    args: &Args,
    genome: &Compactification,
    vertices: &[Vec<i32>],
    result: &PhysicsOutput,
) {
    // Compute fitness
    let fitness = compute_fitness(result);

    if args.save {
        // Save to database
        match save_evaluation(args, genome, vertices, result, fitness) {
            Ok(eval_id) => {
                if args.json {
                    let output = JsonOutput {
                        success: result.success,
                        evaluation_id: Some(eval_id),
                        cached: false,
                        error: result.error.clone(),
                    };
                    println!("{}", serde_json::to_string(&output).unwrap());
                } else {
                    println!();
                    println!("Saved to database with ID: {}", eval_id);
                }
            }
            Err(e) => {
                if args.json {
                    let output = JsonOutput {
                        success: false,
                        evaluation_id: None,
                        cached: false,
                        error: Some(format!("Failed to save: {}", e)),
                    };
                    println!("{}", serde_json::to_string(&output).unwrap());
                } else {
                    eprintln!("Failed to save to database: {}", e);
                }
                std::process::exit(1);
            }
        }
    } else if args.json {
        // JSON output without saving
        let output = JsonOutput {
            success: result.success,
            evaluation_id: None,
            cached: false,
            error: result.error.clone(),
        };
        println!("{}", serde_json::to_string(&output).unwrap());
    } else {
        // Human-readable output
        print_physics_result(result, fitness, args.verbose);
    }
}

fn compute_fitness(result: &PhysicsOutput) -> f64 {
    if !result.success {
        return 0.0;
    }

    let cc = result.cosmological_constant;
    let cc_target = constants::COSMOLOGICAL_CONSTANT;
    let cc_log_error = if cc.abs() > 1e-200 {
        (cc.abs().log10() - cc_target.log10()).abs()
    } else {
        200.0
    };

    let alpha_em_err = ((result.alpha_em - constants::ALPHA_EM) / constants::ALPHA_EM).powi(2);
    let alpha_s_err = ((result.alpha_s - constants::ALPHA_STRONG) / constants::ALPHA_STRONG).powi(2);
    let sw_err = ((result.sin2_theta_w - constants::SIN2_THETA_W) / constants::SIN2_THETA_W).powi(2);
    let gen_err = ((result.n_generations - constants::NUM_GENERATIONS as i32) as f64).powi(2);
    let cc_err = cc_log_error.powi(2) / 122.0_f64.powi(2);

    1.0 / (1.0 + alpha_em_err + alpha_s_err + sw_err + gen_err + cc_err)
}

/// Convert 2D vertices array to flat array format used in DB
fn vertices_to_flat(vertices: &[Vec<i32>]) -> Vec<i32> {
    vertices.iter().flat_map(|v| v.iter().copied()).collect()
}

/// Find or insert polytope, returning its ID
fn get_or_create_polytope(conn: &Connection, vertices: &[Vec<i32>], h11: i32, h21: i32) -> Result<i64, rusqlite::Error> {
    let flat = vertices_to_flat(vertices);
    let vertices_json = serde_json::to_string(&flat).unwrap();

    // Try to find existing polytope with same vertices
    let existing: Option<i64> = conn.query_row(
        "SELECT id FROM polytopes WHERE vertices = ?",
        [&vertices_json],
        |row| row.get(0),
    ).ok();

    if let Some(id) = existing {
        return Ok(id);
    }

    // Insert new polytope
    conn.execute(
        "INSERT INTO polytopes (h11, h21, vertex_count, vertices) VALUES (?, ?, ?, ?)",
        rusqlite::params![h11, h21, vertices.len() as i32, vertices_json],
    )?;

    Ok(conn.last_insert_rowid())
}

fn save_evaluation(
    args: &Args,
    genome: &Compactification,
    vertices: &[Vec<i32>],
    result: &PhysicsOutput,
    fitness: f64,
) -> Result<i64, rusqlite::Error> {
    let conn = Connection::open(&args.database)?;

    // Get or create polytope ID
    let db_polytope_id = if genome.polytope_id == usize::MAX {
        get_or_create_polytope(&conn, vertices, genome.h11, genome.h21)?
    } else {
        genome.polytope_id as i64
    };

    // Insert the evaluation
    conn.execute(
        "INSERT INTO evaluations (
            polytope_id, run_id, generation, g_s,
            kahler_moduli, complex_moduli, flux_f, flux_h,
            fitness, alpha_em, alpha_s, sin2_theta_w, n_generations,
            cosmological_constant, success, error,
            input_hash, model_version, source, label, vertices_json, h11, h21
        ) VALUES (
            ?1, NULL, NULL, ?2,
            ?3, ?4, ?5, ?6,
            ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14,
            ?15, ?16, ?17, ?18, ?19, ?20, ?21
        )",
        rusqlite::params![
            db_polytope_id,
            genome.g_s,
            serde_json::to_string(&genome.kahler_moduli).unwrap(),
            serde_json::to_string(&genome.complex_moduli).unwrap(),
            serde_json::to_string(&genome.flux_f).unwrap(),
            serde_json::to_string(&genome.flux_h).unwrap(),
            fitness,
            result.alpha_em,
            result.alpha_s,
            result.sin2_theta_w,
            result.n_generations,
            result.cosmological_constant,
            result.success as i32,
            result.error.as_deref(),
            args.hash.as_deref(),
            args.model_version.as_deref().unwrap_or(constants::PHYSICS_MODEL_VERSION),
            args.source.as_str(),
            args.label.as_deref(),
            serde_json::to_string(vertices).unwrap(),
            genome.h11,
            genome.h21,
        ],
    )?;

    Ok(conn.last_insert_rowid())
}

fn print_physics_result(result: &PhysicsOutput, fitness: f64, verbose: bool) {
    if !result.success {
        println!("  EVALUATION FAILED: {}", result.error.as_deref().unwrap_or("Unknown error"));
        return;
    }

    println!("  Physics computed successfully:");
    println!("    alpha_em     = {:.6e}  (target: {:.6e}, error: {:.2}x)",
             result.alpha_em, constants::ALPHA_EM,
             result.alpha_em / constants::ALPHA_EM);
    println!("    alpha_s      = {:.4}       (target: {:.4}, error: {:.2}x)",
             result.alpha_s, constants::ALPHA_STRONG,
             result.alpha_s / constants::ALPHA_STRONG);
    println!("    sin2_theta_w = {:.5}      (target: {:.5}, error: {:.2}x)",
             result.sin2_theta_w, constants::SIN2_THETA_W,
             result.sin2_theta_w / constants::SIN2_THETA_W);
    println!("    N_gen        = {}          (target: {})",
             result.n_generations, constants::NUM_GENERATIONS);

    let cc = result.cosmological_constant;
    let cc_target = constants::COSMOLOGICAL_CONSTANT;
    let cc_log_error = if cc.abs() > 1e-200 {
        (cc.abs().log10() - cc_target.log10()).abs()
    } else {
        200.0
    };
    println!("    Lambda       = {:.3e}  (target: {:.3e})", cc, cc_target);
    println!("    CC log error = {:.2} orders of magnitude", cc_log_error);

    if verbose {
        println!();
        println!("    V_CY         = {:.2}", result.cy_volume);
        println!("    g_s (out)    = {:.6}", result.string_coupling);
        println!("    |W0|         = {:.6e}", result.superpotential_abs);
        println!("    Q_flux       = {:.2}", result.flux_tadpole);
    }

    println!();
    println!("    Overall fitness: {:.6}", fitness);
}
