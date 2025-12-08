//! Fitness evaluation for compactifications.
//!
//! We measure how close a compactification's predicted physics
//! matches our universe's observed constants.

use crate::compactification::{Compactification, PhysicsOutput};
use crate::constants::{
    GENERATION_BONUS, GENERATION_PENALTY, NUM_GENERATIONS, TARGETS, TARGET_NAMES, WEIGHTS,
};

/// Evaluated individual in the population
#[derive(Clone, Debug)]
pub struct Individual {
    pub genome: Compactification,
    pub physics: PhysicsOutput,
    pub fitness: f64,
    pub component_errors: [f64; 6],
}

impl Individual {
    pub fn new(genome: Compactification) -> Self {
        let physics = genome.compute_physics();
        let (fitness, errors) = compute_fitness(&physics);
        Self {
            genome,
            physics,
            fitness,
            component_errors: errors,
        }
    }

    pub fn reevaluate(&mut self) {
        self.physics = self.genome.compute_physics();
        let (fitness, errors) = compute_fitness(&self.physics);
        self.fitness = fitness;
        self.component_errors = errors;
    }
}

/// Compute fitness of a physics output against target constants.
///
/// We use log-scale comparison because these constants span many orders of magnitude.
/// A perfect match returns fitness approaching 1.0, worse matches approach 0.
fn compute_fitness(physics: &PhysicsOutput) -> (f64, [f64; 6]) {
    let predicted = physics.to_array();
    let mut errors = [0.0; 6];
    let mut total_weighted_error = 0.0;
    let mut total_weight = 0.0;

    for i in 0..6 {
        // Use log ratio for scale-invariant comparison
        let log_ratio = if predicted[i] > 0.0 && TARGETS[i] > 0.0 {
            (predicted[i].ln() - TARGETS[i].ln()).abs()
        } else {
            100.0 // Penalty for zero/negative values
        };

        errors[i] = log_ratio;
        total_weighted_error += log_ratio * WEIGHTS[i];
        total_weight += WEIGHTS[i];
    }

    let avg_error = total_weighted_error / total_weight;

    // Convert to fitness in [0, 1] range
    // Error of 0 → fitness 1.0
    // Error of 10 (factor of e^10 ≈ 22000 off) → fitness ≈ 0.0001
    let mut fitness = (-avg_error * 0.5).exp();

    // Generation count bonus/penalty
    if physics.n_generations == NUM_GENERATIONS {
        fitness *= 1.0 + GENERATION_BONUS;
    } else {
        fitness *= 1.0 - GENERATION_PENALTY;
    }

    // Clamp to [0, 1]
    fitness = fitness.clamp(0.0, 1.0);

    (fitness, errors)
}

/// Format fitness details for display
pub fn format_fitness_report(ind: &Individual) -> String {
    let predicted = ind.physics.to_array();
    let mut report = String::new();

    report.push_str(&format!(
        "═══════════════════════════════════════════════════════════════\n"
    ));
    report.push_str(&format!(
        "  FITNESS: {:.6e}    Generations: {} (target: {})\n",
        ind.fitness, ind.physics.n_generations, NUM_GENERATIONS
    ));
    report.push_str(&format!(
        "═══════════════════════════════════════════════════════════════\n"
    ));

    for i in 0..6 {
        let ratio = if TARGETS[i] > 0.0 {
            predicted[i] / TARGETS[i]
        } else {
            f64::NAN
        };

        let status = if ind.component_errors[i] < 0.1 {
            "✓"
        } else if ind.component_errors[i] < 1.0 {
            "~"
        } else if ind.component_errors[i] < 5.0 {
            "×"
        } else {
            "✗"
        };

        report.push_str(&format!(
            "  {} {:26} pred={:12.4e}  target={:12.4e}  ratio={:8.2e}\n",
            status, TARGET_NAMES[i], predicted[i], TARGETS[i], ratio
        ));
    }

    report.push_str(&format!(
        "───────────────────────────────────────────────────────────────\n"
    ));

    // Key compactification parameters
    report.push_str(&format!(
        "  Topology: h^{{1,1}}={}, h^{{2,1}}={}, χ={}\n",
        ind.genome.hodge_numbers.0, ind.genome.hodge_numbers.1, ind.genome.euler_char
    ));
    report.push_str(&format!(
        "  g_s={:.4}, V_CY={:.2}, α'={:.3}\n",
        ind.genome.string_coupling, ind.genome.cy_volume, ind.genome.alpha_prime_corrections
    ));

    report
}

/// Compact single-line summary
pub fn format_fitness_line(ind: &Individual) -> String {
    format!(
        "fit={:.4e} α={:.4e} α_s={:.4e} θ_W={:.4} Λ={:.2e} gen={}",
        ind.fitness,
        ind.physics.alpha_em,
        ind.physics.alpha_strong,
        ind.physics.sin2_theta_w,
        ind.physics.cosmological_constant,
        ind.physics.n_generations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_fitness_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let genome = Compactification::random(&mut rng);
            let ind = Individual::new(genome);
            assert!(ind.fitness >= 0.0, "Fitness below 0: {}", ind.fitness);
            assert!(ind.fitness <= 1.0, "Fitness above 1: {}", ind.fitness);
        }
    }

    #[test]
    fn test_physics_reasonable() {
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..50 {
            let genome = Compactification::random(&mut rng);
            let physics = genome.compute_physics();

            // Basic sanity checks
            assert!(physics.alpha_em > 0.0);
            assert!(physics.alpha_strong > 0.0);
            assert!(physics.sin2_theta_w >= 0.0 && physics.sin2_theta_w <= 1.0);
            assert!(physics.cosmological_constant > 0.0);
        }
    }
}
