//! Genetic algorithm implementation for searching the string landscape.
//!
//! Techniques from physics_simulations/genetic_logic_shapes:
//! - Adaptive mutation rate based on stagnation
//! - Asteroid impact (catastrophic reset with Hall of Fame)
//! - Multi-objective fitness (physics match + simplicity)
//! - Stagnation tracking per individual

use crate::compactification::{
    Compactification, MAX_FLUX, NUM_BRANE_STACKS, NUM_COMPLEX, NUM_FLUXES, NUM_KAHLER,
};
use crate::fitness::Individual;
use rand::Rng;
use rayon::prelude::*;

/// Configuration for the genetic algorithm
#[derive(Clone, Debug)]
pub struct GaConfig {
    pub population_size: usize,
    pub elite_count: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub base_mutation_rate: f64,
    pub base_mutation_strength: f64,
    /// Generations without improvement before asteroid impact
    pub asteroid_threshold: usize,
    /// Hall of fame size
    pub hall_of_fame_size: usize,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population_size: 2000,
            elite_count: 20,
            tournament_size: 5,
            crossover_rate: 0.85,
            base_mutation_rate: 0.15,
            base_mutation_strength: 0.15,
            asteroid_threshold: 50,
            hall_of_fame_size: 100,
        }
    }
}

/// Statistics for a generation
#[derive(Clone, Debug, Default)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub worst_fitness: f64,
    pub diversity: f64,
    pub total_evaluated: u64,
    pub best_n_generations: u8,
    pub stagnation_generations: usize,
    pub asteroid_impacts: usize,
}

/// Hall of Fame entry - stores compactified best solutions
#[derive(Clone, Debug)]
struct HallOfFameEntry {
    genome: Compactification,
    fitness: f64,
    generation_found: usize,
}

/// The genetic algorithm searcher
pub struct LandscapeSearcher {
    pub config: GaConfig,
    pub population: Vec<Individual>,
    pub best_ever: Option<Individual>,
    pub generation: usize,
    pub total_evaluated: u64,
    pub history: Vec<GenerationStats>,
    /// Per-individual stagnation tracking
    stagnation_ages: Vec<usize>,
    /// Generations since global best improved
    generations_since_improvement: usize,
    /// Hall of Fame - archive of best solutions
    hall_of_fame: Vec<HallOfFameEntry>,
    /// Total asteroid impacts
    asteroid_count: usize,
}

impl LandscapeSearcher {
    /// Create a new searcher with random initial population
    pub fn new(config: GaConfig) -> Self {
        let population: Vec<Individual> = (0..config.population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                Individual::new(Compactification::random(&mut rng))
            })
            .collect();

        let best = population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .cloned();

        let total = config.population_size as u64;
        let stagnation_ages = vec![0; config.population_size];

        Self {
            config,
            population,
            best_ever: best,
            generation: 0,
            total_evaluated: total,
            history: Vec::new(),
            stagnation_ages,
            generations_since_improvement: 0,
            hall_of_fame: Vec::new(),
            asteroid_count: 0,
        }
    }

    /// Run one generation of evolution
    pub fn step(&mut self) {
        self.generation += 1;

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Check for global improvement
        let current_best_fitness = self.population.first().map(|i| i.fitness).unwrap_or(0.0);
        let previous_best = self.best_ever.as_ref().map(|i| i.fitness).unwrap_or(0.0);

        if current_best_fitness > previous_best * 1.001 { // 0.1% improvement threshold
            self.generations_since_improvement = 0;
            self.best_ever = self.population.first().cloned();

            // Add to Hall of Fame
            if let Some(best) = self.population.first().cloned() {
                self.add_to_hall_of_fame(&best);
            }
        } else {
            self.generations_since_improvement += 1;
        }

        // Check for asteroid impact
        if self.generations_since_improvement >= self.config.asteroid_threshold {
            self.asteroid_impact();
        }

        // Collect statistics
        let stats = self.compute_stats();
        self.history.push(stats);

        // Create next generation
        let mut next_gen = Vec::with_capacity(self.config.population_size);
        let mut next_stagnation = Vec::with_capacity(self.config.population_size);

        // Elitism: keep best individuals unchanged
        for (i, ind) in self.population.iter().take(self.config.elite_count).enumerate() {
            next_gen.push(ind.clone());
            next_stagnation.push(self.stagnation_ages.get(i).copied().unwrap_or(0) + 1);
        }

        // Fill rest with offspring
        let offspring_needed = self.config.population_size - self.config.elite_count;

        // Clone data for parallel access
        let pop_clone = self.population.clone();
        let stagnation_clone = self.stagnation_ages.clone();
        let config = self.config.clone();
        let generation = self.generation;
        let hall_of_fame = self.hall_of_fame.clone();

        let offspring: Vec<(Individual, usize)> = (0..offspring_needed)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();

                // Tournament selection for parents
                let (parent1_idx, parent1) = tournament_select_with_idx(&pop_clone, config.tournament_size, &mut rng);
                let parent1_stagnation = stagnation_clone.get(parent1_idx).copied().unwrap_or(0);

                // Sometimes pick from Hall of Fame for second parent (crossover with history)
                let (parent2, parent2_stagnation) = if !hall_of_fame.is_empty() && rng.gen::<f64>() < 0.2 {
                    let hof_idx = rng.gen_range(0..hall_of_fame.len());
                    let hof_genome = hall_of_fame[hof_idx].genome.clone();
                    (Individual::new(hof_genome), 0)
                } else {
                    let (idx, p) = tournament_select_with_idx(&pop_clone, config.tournament_size, &mut rng);
                    (p.clone(), stagnation_clone.get(idx).copied().unwrap_or(0))
                };

                // Adaptive mutation rate based on parent stagnation
                let max_stagnation = parent1_stagnation.max(parent2_stagnation);
                let (mutation_rate, mutation_strength) = adaptive_mutation(
                    max_stagnation,
                    config.base_mutation_rate,
                    config.base_mutation_strength,
                    &mut rng,
                );

                // Crossover
                let mut child = if rng.gen::<f64>() < config.crossover_rate {
                    crossover(&parent1.genome, &parent2.genome, &mut rng)
                } else {
                    parent1.genome.clone()
                };

                // Mutation (always mutate, but with adaptive strength)
                mutate(&mut child, mutation_rate, mutation_strength, &mut rng);

                let child_ind = Individual::new(child);

                // Inherit stagnation age, reset if improved
                let child_stagnation = if child_ind.fitness > parent1.fitness.max(parent2.fitness) {
                    0 // Improved! Reset stagnation
                } else {
                    max_stagnation + 1 // Inherit and increment
                };

                (child_ind, child_stagnation)
            })
            .collect();

        for (ind, stag) in offspring {
            next_gen.push(ind);
            next_stagnation.push(stag);
        }

        self.total_evaluated += offspring_needed as u64;
        self.population = next_gen;
        self.stagnation_ages = next_stagnation;
    }

    /// ASTEROID IMPACT: Catastrophic reset with Hall of Fame preservation
    fn asteroid_impact(&mut self) {
        self.asteroid_count += 1;
        self.generations_since_improvement = 0;

        eprintln!(
            "\n🌠 ASTEROID IMPACT #{} at generation {} - Rebuilding population from Hall of Fame...\n",
            self.asteroid_count, self.generation
        );

        // Save current best to Hall of Fame if not already there
        if let Some(best) = self.population.first().cloned() {
            self.add_to_hall_of_fame(&best);
        }

        let mut rng = rand::thread_rng();
        let mut new_population = Vec::with_capacity(self.config.population_size);
        let mut new_stagnation = Vec::with_capacity(self.config.population_size);

        // Keep the absolute best (the King)
        if let Some(best) = &self.best_ever {
            new_population.push(best.clone());
            new_stagnation.push(0);
        }

        // 20% fresh random (exploration)
        let fresh_count = self.config.population_size / 5;
        for _ in 0..fresh_count {
            let genome = Compactification::random(&mut rng);
            new_population.push(Individual::new(genome));
            new_stagnation.push(0);
        }

        // 80% recombinations from Hall of Fame (genetic blender)
        let recomb_count = self.config.population_size - new_population.len();

        if self.hall_of_fame.len() >= 2 {
            for _ in 0..recomb_count {
                // Pick two random Hall of Fame members
                let idx1 = rng.gen_range(0..self.hall_of_fame.len());
                let idx2 = rng.gen_range(0..self.hall_of_fame.len());

                let parent1 = &self.hall_of_fame[idx1].genome;
                let parent2 = &self.hall_of_fame[idx2].genome;

                // Aggressive crossover + mutation
                let mut child = crossover(parent1, parent2, &mut rng);
                mutate(&mut child, 0.5, 0.3, &mut rng); // High mutation for exploration

                new_population.push(Individual::new(child));
                new_stagnation.push(0);
            }
        } else {
            // Not enough Hall of Fame members, fill with random
            for _ in 0..recomb_count {
                let genome = Compactification::random(&mut rng);
                new_population.push(Individual::new(genome));
                new_stagnation.push(0);
            }
        }

        self.population = new_population;
        self.stagnation_ages = new_stagnation;
        self.total_evaluated += self.config.population_size as u64;
    }

    /// Add individual to Hall of Fame (if worthy)
    fn add_to_hall_of_fame(&mut self, ind: &Individual) {
        // Check if already in HoF (by fitness proximity)
        let dominated = self.hall_of_fame.iter().any(|h| {
            (h.fitness - ind.fitness).abs() < ind.fitness * 0.01 // Within 1%
        });

        if dominated {
            return;
        }

        // Remove any HoF entries this one dominates
        self.hall_of_fame.retain(|h| h.fitness > ind.fitness * 0.99);

        // Add new entry
        self.hall_of_fame.push(HallOfFameEntry {
            genome: ind.genome.clone(),
            fitness: ind.fitness,
            generation_found: self.generation,
        });

        // Trim to max size (keep best)
        if self.hall_of_fame.len() > self.config.hall_of_fame_size {
            self.hall_of_fame.sort_by(|a, b| {
                b.fitness.partial_cmp(&a.fitness).unwrap()
            });
            self.hall_of_fame.truncate(self.config.hall_of_fame_size);
        }
    }

    /// Compute generation statistics
    fn compute_stats(&self) -> GenerationStats {
        let fitnesses: Vec<f64> = self.population.iter().map(|i| i.fitness).collect();

        let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        let variance =
            fitnesses.iter().map(|f| (f - avg).powi(2)).sum::<f64>() / fitnesses.len() as f64;
        let diversity = variance.sqrt();

        let best_n_gen = self
            .population
            .first()
            .map(|i| i.physics.n_generations)
            .unwrap_or(0);

        GenerationStats {
            generation: self.generation,
            best_fitness: best,
            avg_fitness: avg,
            worst_fitness: worst,
            diversity,
            total_evaluated: self.total_evaluated,
            best_n_generations: best_n_gen,
            stagnation_generations: self.generations_since_improvement,
            asteroid_impacts: self.asteroid_count,
        }
    }

    /// Get current best individual
    pub fn best(&self) -> Option<&Individual> {
        self.population.first()
    }
}

/// Adaptive mutation based on stagnation (from genetic_logic_shapes)
fn adaptive_mutation<R: Rng>(
    stagnation: usize,
    base_rate: f64,
    base_strength: f64,
    rng: &mut R,
) -> (f64, f64) {
    let roll: f64 = rng.gen();

    let (rate, strength) = if stagnation < 3 {
        // Fresh - gentle mutations
        if roll < 0.6 { (base_rate * 0.5, base_strength * 0.5) }
        else if roll < 0.9 { (base_rate, base_strength) }
        else { (base_rate * 2.0, base_strength * 1.5) }
    } else if stagnation < 7 {
        // Getting stale - moderate mutations
        if roll < 0.3 { (base_rate, base_strength) }
        else if roll < 0.7 { (base_rate * 2.0, base_strength * 1.5) }
        else { (base_rate * 3.0, base_strength * 2.0) }
    } else if stagnation < 15 {
        // Stagnant - aggressive mutations
        if roll < 0.2 { (base_rate * 2.0, base_strength * 1.5) }
        else if roll < 0.6 { (base_rate * 4.0, base_strength * 2.5) }
        else { (base_rate * 6.0, base_strength * 3.0) }
    } else {
        // Very stagnant - desperate mutations
        if roll < 0.3 { (base_rate * 4.0, base_strength * 2.0) }
        else if roll < 0.7 { (base_rate * 8.0, base_strength * 4.0) }
        else { (1.0, base_strength * 5.0) } // Hypermutation!
    };

    (rate.min(1.0), strength.min(1.0))
}

/// Tournament selection returning index and individual
fn tournament_select_with_idx<'a, R: Rng>(
    population: &'a [Individual],
    tournament_size: usize,
    rng: &mut R,
) -> (usize, &'a Individual) {
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best = &population[best_idx];

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..population.len());
        let candidate = &population[idx];

        if candidate.fitness > best.fitness {
            best_idx = idx;
            best = candidate;
        }
    }

    (best_idx, best)
}

/// Crossover two compactifications
fn crossover<R: Rng>(
    p1: &Compactification,
    p2: &Compactification,
    rng: &mut R,
) -> Compactification {
    let mut child = p1.clone();
    let alpha = rng.gen_range(0.0..1.0);

    // Topology: inherit from one parent (discrete)
    if rng.gen() {
        child.hodge_numbers = p2.hodge_numbers;
        child.euler_char = p2.euler_char;
    }

    // Kähler moduli: blend crossover
    for i in 0..NUM_KAHLER {
        if rng.gen() {
            child.kahler_moduli[i] =
                alpha * p1.kahler_moduli[i] + (1.0 - alpha) * p2.kahler_moduli[i];
        }
    }

    // Complex structure moduli: blend crossover
    for i in 0..NUM_COMPLEX {
        if rng.gen() {
            child.complex_moduli[i] =
                alpha * p1.complex_moduli[i] + (1.0 - alpha) * p2.complex_moduli[i];
        }
    }

    // Fluxes: uniform crossover (discrete)
    for i in 0..NUM_FLUXES / 2 {
        if rng.gen() {
            child.ns_fluxes[i] = p2.ns_fluxes[i];
        }
        if rng.gen() {
            child.rr_fluxes[i] = p2.rr_fluxes[i];
        }
    }

    // Brane stacks: mix of blend and uniform
    for i in 0..NUM_BRANE_STACKS {
        if rng.gen() {
            child.brane_stacks[i][0] = p2.brane_stacks[i][0];
            child.brane_stacks[i][1] = p2.brane_stacks[i][1];
        }
        child.brane_stacks[i][2] =
            alpha * p1.brane_stacks[i][2] + (1.0 - alpha) * p2.brane_stacks[i][2];
        child.brane_stacks[i][3] =
            alpha * p1.brane_stacks[i][3] + (1.0 - alpha) * p2.brane_stacks[i][3];
    }

    if rng.gen() {
        child.string_coupling =
            alpha * p1.string_coupling + (1.0 - alpha) * p2.string_coupling;
    }
    if rng.gen() {
        child.cy_volume = alpha * p1.cy_volume + (1.0 - alpha) * p2.cy_volume;
    }
    if rng.gen() {
        child.alpha_prime_corrections =
            alpha * p1.alpha_prime_corrections + (1.0 - alpha) * p2.alpha_prime_corrections;
    }

    child
}

/// Mutate a compactification with adaptive rate and strength
fn mutate<R: Rng>(genome: &mut Compactification, rate: f64, strength: f64, rng: &mut R) {
    // Topology mutation (more likely with high strength)
    if rng.gen::<f64>() < 0.02 * strength * 5.0 {
        let delta_h11: i16 = rng.gen_range(-20..=20);
        let delta_h21: i16 = rng.gen_range(-20..=20);
        genome.hodge_numbers.0 = (genome.hodge_numbers.0 as i16 + delta_h11).clamp(2, 200) as u16;
        genome.hodge_numbers.1 = (genome.hodge_numbers.1 as i16 + delta_h21).clamp(0, 200) as u16;
        genome.euler_char = 2 * (genome.hodge_numbers.0 as i32 - genome.hodge_numbers.1 as i32);
    }

    // Kähler moduli mutations (log-scale perturbation)
    for k in &mut genome.kahler_moduli {
        if rng.gen::<f64>() < rate {
            let log_k = k.max(0.01).ln();
            let perturbation = rng.gen_range(-1.0..1.0) * strength;
            *k = (log_k + perturbation).exp().clamp(0.01, 100.0);
        }
    }

    // Complex structure mutations
    for c in &mut genome.complex_moduli {
        if rng.gen::<f64>() < rate {
            *c += rng.gen_range(-1.0..1.0) * strength;
            *c = c.clamp(-5.0, 5.0);
        }
    }

    // Flux mutations (discrete jumps, larger with higher strength)
    let flux_jump = (strength * 10.0).ceil() as i32;
    for f in &mut genome.ns_fluxes {
        if rng.gen::<f64>() < rate * 0.5 {
            let delta: i32 = rng.gen_range(-flux_jump..=flux_jump);
            *f = (*f + delta).clamp(-MAX_FLUX, MAX_FLUX);
        }
    }
    for f in &mut genome.rr_fluxes {
        if rng.gen::<f64>() < rate * 0.5 {
            let delta: i32 = rng.gen_range(-flux_jump..=flux_jump);
            *f = (*f + delta).clamp(-MAX_FLUX, MAX_FLUX);
        }
    }

    // Brane stack mutations
    for stack in &mut genome.brane_stacks {
        if rng.gen::<f64>() < rate * 0.3 {
            let delta: f64 = rng.gen_range(-2.0..=2.0) * strength;
            stack[0] = (stack[0] + delta).clamp(1.0, 10.0).floor();
        }
        if rng.gen::<f64>() < rate * 0.3 {
            stack[1] = rng.gen_range(0.0..NUM_KAHLER as f64).floor();
        }
        if rng.gen::<f64>() < rate {
            stack[2] += rng.gen_range(-0.2..0.2) * strength;
            stack[2] = stack[2].clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            stack[3] += rng.gen_range(-0.2..0.2) * strength;
            stack[3] = stack[3].clamp(0.0, 1.0);
        }
    }

    // String coupling mutation
    if rng.gen::<f64>() < rate {
        genome.string_coupling *= 1.0 + rng.gen_range(-0.3..0.3) * strength;
        genome.string_coupling = genome.string_coupling.clamp(0.01, 0.5);
    }

    // Volume mutation (log-scale, larger perturbation)
    if rng.gen::<f64>() < rate {
        let log_v = genome.cy_volume.ln();
        let new_log_v = log_v + rng.gen_range(-0.5..0.5) * strength;
        genome.cy_volume = new_log_v.exp().clamp(1.0, 1000.0);
    }

    // α' corrections
    if rng.gen::<f64>() < rate {
        genome.alpha_prime_corrections += rng.gen_range(-0.1..0.1) * strength;
        genome.alpha_prime_corrections = genome.alpha_prime_corrections.clamp(0.0, 0.5);
    }
}
