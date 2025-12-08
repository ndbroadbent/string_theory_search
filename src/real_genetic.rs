//! Genetic algorithm for real string theory landscape search
//!
//! This module implements a GA that uses actual physics computations
//! from the Python/JAX bridge instead of toy approximations.

use crate::constants;
use crate::physics::{
    compute_physics, init_physics_bridge, is_physics_available,
    PhysicsOutput, PolytopeData, RealCompactification,
};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for the real physics GA
#[derive(Debug, Clone)]
pub struct RealGaConfig {
    pub population_size: usize,
    pub elite_count: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub base_mutation_rate: f64,
    pub base_mutation_strength: f64,
    pub collapse_threshold: usize,
    pub hall_of_fame_size: usize,
}

impl Default for RealGaConfig {
    fn default() -> Self {
        Self {
            population_size: 500,  // Smaller due to expensive physics computations
            elite_count: 10,
            tournament_size: 5,
            crossover_rate: 0.85,
            base_mutation_rate: 0.4,
            base_mutation_strength: 0.35,
            collapse_threshold: 2000,
            hall_of_fame_size: 50,
        }
    }
}

/// An individual in the real physics GA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealIndividual {
    pub genome: RealCompactification,
    pub physics: Option<PhysicsOutput>,
    pub fitness: f64,
}

impl RealIndividual {
    /// Create a new random individual
    pub fn random<R: Rng>(rng: &mut R, polytope_data: &PolytopeData) -> Self {
        let genome = RealCompactification::random(rng, polytope_data);
        Self {
            genome,
            physics: None,
            fitness: 0.0,
        }
    }

    /// Evaluate this individual's fitness using real physics
    pub fn evaluate(&mut self) {
        if !is_physics_available() {
            panic!("Physics bridge not available! Cannot run without real physics.");
        }

        let physics = compute_physics(&self.genome);

        if physics.success {
            self.fitness = compute_fitness(&physics);
            self.physics = Some(physics);
        } else {
            self.fitness = 0.0;
            self.physics = Some(physics);
        }
    }
}

/// Compute fitness from physics output
///
/// Higher is better. We want to match observed physical constants.
pub fn compute_fitness(physics: &PhysicsOutput) -> f64 {
    if !physics.success {
        return 0.0;
    }

    // Target values (observed physics)
    let targets = [
        (physics.alpha_em, constants::ALPHA_EM, 1.0, "α_em"),
        (physics.alpha_s, constants::ALPHA_STRONG, 1.0, "α_s"),
        (physics.sin2_theta_w, constants::SIN2_THETA_W, 1.0, "sin²θ_W"),
        (physics.n_generations as f64, constants::NUM_GENERATIONS as f64, 2.0, "N_gen"),
    ];

    let mut total_score = 0.0;
    let mut total_weight = 0.0;

    for (computed, target, weight, _name) in targets {
        // Log-ratio scoring: how close are we on a log scale?
        let ratio = if target.abs() > 1e-30 && computed.abs() > 1e-30 {
            (computed / target).abs()
        } else {
            0.0
        };

        // Score: 1.0 when ratio = 1, decreasing as ratio deviates
        // Use log to handle wide range of values
        let log_ratio = if ratio > 0.0 { ratio.ln().abs() } else { 10.0 };
        let score = (-log_ratio).exp();

        total_score += weight * score;
        total_weight += weight;
    }

    // Cosmological constant needs special handling (tiny target)
    let cc_ratio = if constants::COSMOLOGICAL_CONSTANT.abs() > 1e-150 {
        (physics.cosmological_constant / constants::COSMOLOGICAL_CONSTANT).abs()
    } else {
        0.0
    };
    let cc_log = if cc_ratio > 0.0 { cc_ratio.ln().abs() } else { 300.0 };
    let cc_score = (-cc_log / 100.0).exp();  // Scale down log penalty
    total_score += 0.5 * cc_score;
    total_weight += 0.5;

    // Mass ratios
    let me_ratio = if constants::ELECTRON_PLANCK_RATIO > 0.0 && physics.m_e_planck_ratio > 0.0 {
        (physics.m_e_planck_ratio / constants::ELECTRON_PLANCK_RATIO).abs()
    } else {
        0.0
    };
    let me_log = if me_ratio > 0.0 { me_ratio.ln().abs() } else { 50.0 };
    let me_score = (-me_log / 10.0).exp();
    total_score += 0.3 * me_score;
    total_weight += 0.3;

    // Bonus for valid geometry
    if physics.cy_volume > 0.1 && physics.cy_volume < 100.0 {
        total_score += 0.2;
        total_weight += 0.2;
    }

    // Tadpole constraint satisfaction (should be < some bound)
    if physics.flux_tadpole.abs() < 50.0 {
        total_score += 0.1;
        total_weight += 0.1;
    }

    // Normalize to [0, 1]
    let fitness = total_score / total_weight;

    // Ensure non-negative
    fitness.max(0.0)
}

/// Generation statistics for real physics GA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealGenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub worst_fitness: f64,
    pub diversity: f64,
    pub total_evaluated: u64,
    pub stagnation_generations: usize,
    pub landscape_collapses: usize,
    pub physics_successes: usize,
    pub physics_failures: usize,
}

/// The real physics landscape searcher
pub struct RealLandscapeSearcher {
    pub config: RealGaConfig,
    pub polytope_data: Arc<PolytopeData>,
    pub population: Vec<RealIndividual>,
    pub best_ever: Option<RealIndividual>,
    pub hall_of_fame: Vec<RealIndividual>,
    pub history: Vec<RealGenerationStats>,
    pub generation: usize,
    pub total_evaluated: u64,
    pub stagnation_count: usize,
    pub collapse_count: usize,
    rng: StdRng,
}

impl RealLandscapeSearcher {
    /// Create a new searcher
    pub fn new(config: RealGaConfig, polytope_path: &str) -> Self {
        // Initialize Python bridge
        if let Err(e) = init_physics_bridge() {
            eprintln!("Warning: Could not initialize physics bridge: {}", e);
            eprintln!("Using fallback fitness function");
        }

        // Load polytope data
        let polytope_data = Arc::new(PolytopeData::load_or_default(polytope_path));
        println!("Using {} polytopes from Kreuzer-Skarke database", polytope_data.polytopes.len());

        let mut rng = StdRng::from_entropy();

        // Initialize population
        let population: Vec<RealIndividual> = (0..config.population_size)
            .map(|_| RealIndividual::random(&mut rng, &polytope_data))
            .collect();

        Self {
            config,
            polytope_data,
            population,
            best_ever: None,
            hall_of_fame: Vec::new(),
            history: Vec::new(),
            generation: 0,
            total_evaluated: 0,
            stagnation_count: 0,
            collapse_count: 0,
            rng,
        }
    }

    /// Run one generation of the GA
    pub fn step(&mut self) {
        self.generation += 1;

        // Evaluate population (can be parallelized, but Python GIL limits this)
        for individual in &mut self.population {
            if individual.physics.is_none() {
                individual.evaluate();
                self.total_evaluated += 1;
            }
        }

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Track best
        if let Some(best) = self.population.first() {
            let dominated = match &self.best_ever {
                Some(prev_best) => best.fitness > prev_best.fitness,
                None => true,
            };

            if dominated {
                self.best_ever = Some(best.clone());
                self.stagnation_count = 0;
            } else {
                self.stagnation_count += 1;
            }

            // Update hall of fame
            self.update_hall_of_fame(best.clone());
        }

        // Record statistics
        let stats = self.compute_stats();
        self.history.push(stats);

        // Check for landscape collapse
        if self.stagnation_count >= self.config.collapse_threshold {
            self.landscape_collapse();
        }

        // Create next generation
        self.evolve();
    }

    fn compute_stats(&self) -> RealGenerationStats {
        let fitnesses: Vec<f64> = self.population.iter().map(|i| i.fitness).collect();
        let best = fitnesses.first().copied().unwrap_or(0.0);
        let worst = fitnesses.last().copied().unwrap_or(0.0);
        let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        // Diversity: std dev of fitness
        let variance = fitnesses.iter()
            .map(|f| (f - avg).powi(2))
            .sum::<f64>() / fitnesses.len() as f64;
        let diversity = variance.sqrt();

        // Count physics successes/failures
        let physics_successes = self.population.iter()
            .filter(|i| i.physics.as_ref().map(|p| p.success).unwrap_or(false))
            .count();
        let physics_failures = self.population.len() - physics_successes;

        RealGenerationStats {
            generation: self.generation,
            best_fitness: best,
            avg_fitness: avg,
            worst_fitness: worst,
            diversity,
            total_evaluated: self.total_evaluated,
            stagnation_generations: self.stagnation_count,
            landscape_collapses: self.collapse_count,
            physics_successes,
            physics_failures,
        }
    }

    fn update_hall_of_fame(&mut self, candidate: RealIndividual) {
        // Check if this is unique enough
        let dominated = self.hall_of_fame.iter()
            .any(|existing| {
                // Simple uniqueness check based on polytope and moduli
                existing.genome.polytope_id == candidate.genome.polytope_id &&
                (existing.genome.g_s - candidate.genome.g_s).abs() < 0.01
            });

        if !dominated {
            self.hall_of_fame.push(candidate);
            self.hall_of_fame.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            self.hall_of_fame.truncate(self.config.hall_of_fame_size);
        }
    }

    fn landscape_collapse(&mut self) {
        println!("🌠 LANDSCAPE COLLAPSE #{} - Exploring new region of moduli space!", self.collapse_count + 1);
        self.collapse_count += 1;
        self.stagnation_count = 0;

        // Keep elites
        let elite_count = self.config.elite_count;

        // Replace half with completely fresh random individuals
        let fresh_count = self.population.len() / 2;
        for i in elite_count..(elite_count + fresh_count) {
            self.population[i] = RealIndividual::random(&mut self.rng, &self.polytope_data);
        }

        // Inject some from hall of fame (with heavy mutation)
        let hof_inject = (self.hall_of_fame.len() / 4).min(self.population.len() / 10);
        for i in 0..hof_inject {
            if elite_count + fresh_count + i < self.population.len() {
                let idx = self.rng.gen_range(0..self.hall_of_fame.len());
                let mut mutant = self.hall_of_fame[idx].clone();
                mutant.genome.mutate(&mut self.rng, 1.0, &self.polytope_data);
                mutant.physics = None;  // Force re-evaluation
                self.population[elite_count + fresh_count + i] = mutant;
            }
        }

        // Rest: heavily mutated current population
        for i in (elite_count + fresh_count + hof_inject)..self.population.len() {
            self.population[i].genome.mutate(&mut self.rng, 1.0, &self.polytope_data);
            self.population[i].physics = None;
        }
    }

    fn evolve(&mut self) {
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Elitism: keep best individuals
        for i in 0..self.config.elite_count {
            new_population.push(self.population[i].clone());
        }

        // Fill rest with offspring
        while new_population.len() < self.config.population_size {
            // Tournament selection - get indices
            let idx1 = self.tournament_select_idx();
            let idx2 = self.tournament_select_idx();
            let parent1 = self.population[idx1].clone();
            let parent2 = self.population[idx2].clone();

            // Crossover
            let mut child = if self.rng.gen::<f64>() < self.config.crossover_rate {
                let child_genome = parent1.genome.crossover(&parent2.genome, &mut self.rng);
                RealIndividual {
                    genome: child_genome,
                    physics: None,
                    fitness: 0.0,
                }
            } else {
                let mut child = parent1;
                child.physics = None;
                child
            };

            // Adaptive mutation
            let (rate, strength) = adaptive_mutation(
                self.stagnation_count,
                self.config.base_mutation_rate,
                self.config.base_mutation_strength,
                &mut self.rng,
            );

            if self.rng.gen::<f64>() < rate {
                child.genome.mutate(&mut self.rng, strength, &self.polytope_data);
            }

            new_population.push(child);
        }

        self.population = new_population;
    }

    fn tournament_select_idx(&mut self) -> usize {
        let mut best_idx = self.rng.gen_range(0..self.population.len());
        let mut best_fitness = self.population[best_idx].fitness;

        for _ in 1..self.config.tournament_size {
            let idx = self.rng.gen_range(0..self.population.len());
            if self.population[idx].fitness > best_fitness {
                best_idx = idx;
                best_fitness = self.population[idx].fitness;
            }
        }

        best_idx
    }

    /// Get current best individual
    pub fn best(&self) -> Option<&RealIndividual> {
        self.population.first()
    }

    /// Save state to timestamped JSON file
    pub fn save_state(&self, dir: &str) -> Result<String, Box<dyn std::error::Error>> {
        use std::io::Write;

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("{}/state_{}.json", dir, timestamp);

        #[derive(Serialize)]
        struct SavedState<'a> {
            generation: usize,
            total_evaluated: u64,
            best_ever: &'a Option<RealIndividual>,
            hall_of_fame: &'a Vec<RealIndividual>,
            collapse_count: usize,
        }

        let state = SavedState {
            generation: self.generation,
            total_evaluated: self.total_evaluated,
            best_ever: &self.best_ever,
            hall_of_fame: &self.hall_of_fame,
            collapse_count: self.collapse_count,
        };

        let json = serde_json::to_string_pretty(&state)?;
        let mut file = std::fs::File::create(&filename)?;
        file.write_all(json.as_bytes())?;

        Ok(filename)
    }

    /// Save just the best individual to a simple file (for quick reference)
    pub fn save_best(&self, dir: &str) -> Result<String, Box<dyn std::error::Error>> {
        use std::io::Write;

        if let Some(best) = &self.best_ever {
            let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            let filename = format!("{}/best_{}.json", dir, timestamp);

            let json = serde_json::to_string_pretty(best)?;
            let mut file = std::fs::File::create(&filename)?;
            file.write_all(json.as_bytes())?;

            Ok(filename)
        } else {
            Err("No best individual to save".into())
        }
    }

    /// Save best individual with fitness in filename for easy sorting
    pub fn save_best_with_fitness(&self, dir: &str) -> Result<String, Box<dyn std::error::Error>> {
        use std::io::Write;

        if let Some(best) = &self.best_ever {
            let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            // Format fitness as 4 decimal places, e.g., 0.9187 -> "0_9187"
            let fitness_str = format!("{:.4}", best.fitness).replace('.', "_");
            let filename = format!("{}/fit{}_{}.json", dir, fitness_str, timestamp);

            let json = serde_json::to_string_pretty(best)?;
            let mut file = std::fs::File::create(&filename)?;
            file.write_all(json.as_bytes())?;

            Ok(filename)
        } else {
            Err("No best individual to save".into())
        }
    }
}

/// Adaptive mutation based on stagnation
fn adaptive_mutation<R: Rng>(
    stagnation: usize,
    base_rate: f64,
    base_strength: f64,
    _rng: &mut R,
) -> (f64, f64) {
    // Ramp up mutation as stagnation increases
    if stagnation < 10 {
        (base_rate, base_strength)
    } else if stagnation < 50 {
        (base_rate * 1.5, base_strength * 2.0)
    } else if stagnation < 200 {
        (base_rate * 2.0, base_strength * 3.0)
    } else if stagnation < 500 {
        (0.9, base_strength * 4.0)  // Hypermutation
    } else {
        (1.0, 1.0)  // Maximum chaos
    }
}

/// Format a physics output for display
pub fn format_real_fitness_line(individual: &RealIndividual) -> String {
    if let Some(ref physics) = individual.physics {
        if physics.success {
            format!(
                "fit:{:.4} | α_em:{:.2e} α_s:{:.3} sin²θ:{:.3} N:{} Λ:{:.1e}",
                individual.fitness,
                physics.alpha_em,
                physics.alpha_s,
                physics.sin2_theta_w,
                physics.n_generations,
                physics.cosmological_constant,
            )
        } else {
            format!(
                "fit:{:.4} | ERROR: {}",
                individual.fitness,
                physics.error.as_deref().unwrap_or("unknown")
            )
        }
    } else {
        format!("fit:{:.4} | not evaluated", individual.fitness)
    }
}

/// Format detailed report
pub fn format_real_fitness_report(individual: &RealIndividual) -> String {
    let mut report = String::new();

    report.push_str(&format!("Fitness: {:.6}\n", individual.fitness));
    report.push_str(&format!("\nGenome:\n"));
    report.push_str(&format!("  Polytope ID: {}\n", individual.genome.polytope_id));
    report.push_str(&format!("  h11 = {}, h21 = {}\n", individual.genome.h11, individual.genome.h21));
    report.push_str(&format!("  Kähler moduli: {:?}\n", individual.genome.kahler_moduli));
    report.push_str(&format!("  String coupling g_s = {:.4}\n", individual.genome.g_s));

    if let Some(ref physics) = individual.physics {
        report.push_str(&format!("\nPhysics output:\n"));
        if physics.success {
            report.push_str(&format!("  α_em = {:.6e}  (target: {:.6e})\n", physics.alpha_em, constants::ALPHA_EM));
            report.push_str(&format!("  α_s  = {:.6}  (target: {:.4})\n", physics.alpha_s, constants::ALPHA_STRONG));
            report.push_str(&format!("  sin²θ_W = {:.6}  (target: {:.5})\n", physics.sin2_theta_w, constants::SIN2_THETA_W));
            report.push_str(&format!("  N_gen = {}  (target: {})\n", physics.n_generations, constants::NUM_GENERATIONS));
            report.push_str(&format!("  Λ = {:.3e}  (target: {:.3e})\n", physics.cosmological_constant, constants::COSMOLOGICAL_CONSTANT));
            report.push_str(&format!("  CY volume = {:.4}\n", physics.cy_volume));
            report.push_str(&format!("  Flux tadpole = {:.2}\n", physics.flux_tadpole));
        } else {
            report.push_str(&format!("  ERROR: {}\n", physics.error.as_deref().unwrap_or("unknown")));
        }
    }

    report
}
