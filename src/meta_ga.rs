//! Meta-Genetic Algorithm for evolving search strategies
//!
//! The meta-GA evolves algorithm parameters (feature weights, search strategy, etc.)
//! to find configurations that discover good physics solutions faster.
//!
//! Five levels:
//! 1. Meta-generation: population of algorithms (e.g. 16)
//! 2. Algorithm: specific parameters (feature weights, similarity_radius, etc.)
//! 3. Trial: one complete run of an algorithm
//! 4. Inner generation: one generation within a trial
//! 5. Evaluation: one polytope physics computation

use rand::Rng;
use rusqlite::{Connection, params};
use std::collections::HashMap;

use crate::db::{MetaAlgorithm, META_ALGORITHM_VERSION};

/// All heuristic feature names that can be weighted
/// These must match the columns in the heuristics table
pub const HEURISTIC_FEATURES: &[&str] = &[
    // Basic geometry
    "h11",
    "h21",
    "vertex_count",
    // Circularity
    "sphericity",
    "inertia_isotropy",
    // Chirality
    "chirality_optimal",
    "chirality_x",
    "chirality_y",
    "chirality_z",
    "chirality_w",
    "handedness_det",
    // Symmetry
    "symmetry_x",
    "symmetry_y",
    "symmetry_z",
    "symmetry_w",
    // Flatness
    "flatness_3d",
    "flatness_2d",
    "intrinsic_dim_estimate",
    // Shape
    "spikiness",
    "max_exposure",
    "conformity_ratio",
    "distance_kurtosis",
    "loner_score",
    // Statistics
    "coord_mean",
    "coord_median",
    "coord_std",
    "coord_skewness",
    "coord_kurtosis",
    // Information theory
    "shannon_entropy",
    "joint_entropy",
    // Compression
    "compression_ratio",
    "sorted_compression_ratio",
    "sort_compression_gain",
    // Patterns
    "phi_ratio_count",
    "fibonacci_count",
    "zero_count",
    "one_count",
    "prime_count",
    // Outliers
    "outlier_score",
    "outlier_max_zscore",
    "outlier_count_2sigma",
    "outlier_count_3sigma",
];

/// Generate default feature weights JSON (all weights = 1.0)
pub fn default_feature_weights_json() -> String {
    let weights: HashMap<&str, f64> = HEURISTIC_FEATURES
        .iter()
        .map(|&name| (name, 1.0))
        .collect();
    serde_json::to_string(&weights).unwrap_or_else(|_| "{}".to_string())
}

/// Generate random feature weights JSON (broad: uses most features)
pub fn random_feature_weights_broad<R: Rng>(rng: &mut R) -> String {
    let weights: HashMap<&str, f64> = HEURISTIC_FEATURES
        .iter()
        .map(|&name| {
            let weight = if rng.gen::<f64>() < 0.1 {
                0.0  // 10% chance to ignore feature
            } else {
                rng.gen::<f64>() * 3.0
            };
            (name, weight)
        })
        .collect();
    serde_json::to_string(&weights).unwrap_or_else(|_| "{}".to_string())
}

/// Generate focused feature weights JSON (narrow: only 2-6 features active)
pub fn random_feature_weights_focused<R: Rng>(rng: &mut R) -> String {
    use rand::seq::SliceRandom;

    // Pick 2-6 features to focus on
    let num_features = rng.gen_range(2..=6);

    // Shuffle features and pick the first N
    let mut features: Vec<&str> = HEURISTIC_FEATURES.iter().copied().collect();
    features.shuffle(rng);

    let weights: HashMap<&str, f64> = HEURISTIC_FEATURES
        .iter()
        .map(|&name| {
            let weight = if features[..num_features].contains(&name) {
                // Active feature: random weight 1.0-3.0 (higher floor to ensure significance)
                1.0 + rng.gen::<f64>() * 2.0
            } else {
                0.0  // Inactive feature
            };
            (name, weight)
        })
        .collect();
    serde_json::to_string(&weights).unwrap_or_else(|_| "{}".to_string())
}

/// Generate random feature weights JSON (50/50 broad vs focused)
pub fn random_feature_weights_json<R: Rng>(rng: &mut R) -> String {
    if rng.gen::<f64>() < 0.5 {
        random_feature_weights_focused(rng)
    } else {
        random_feature_weights_broad(rng)
    }
}

/// Parse feature weights from JSON
pub fn parse_feature_weights(json: &str) -> HashMap<String, f64> {
    serde_json::from_str(json).unwrap_or_default()
}

/// Compute weighted distance between two heuristic vectors
pub fn weighted_distance(
    weights: &HashMap<String, f64>,
    h1: &HashMap<String, f64>,
    h2: &HashMap<String, f64>,
) -> f64 {
    let mut sum_sq = 0.0;
    let mut total_weight = 0.0;

    for (name, &weight) in weights {
        if weight > 0.0 {
            let v1 = h1.get(name).copied().unwrap_or(0.0);
            let v2 = h2.get(name).copied().unwrap_or(0.0);
            let diff = v1 - v2;
            sum_sq += weight * diff * diff;
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        (sum_sq / total_weight).sqrt()
    } else {
        f64::MAX
    }
}

// =============================================================================
// Meta-GA Evolution
// =============================================================================

/// Create a random algorithm for initial population
pub fn random_algorithm<R: Rng>(rng: &mut R, meta_generation: i32, runs_required: i32) -> MetaAlgorithm {
    // Generate a seed for this algorithm from the rng
    let algo_seed: u64 = rng.gen();

    MetaAlgorithm {
        id: None,
        name: None,
        version: META_ALGORITHM_VERSION,
        feature_weights: random_feature_weights_json(rng),
        similarity_radius: rng.gen_range(0.1..=1.0),
        interpolation_weight: rng.gen_range(0.0..=1.0),
        population_size: rng.gen_range(30..=150),
        max_generations: rng.gen_range(5..=20),
        mutation_rate: rng.gen_range(0.2..=0.6),
        mutation_strength: rng.gen_range(0.2..=0.5),
        crossover_rate: rng.gen_range(0.6..=0.9),
        tournament_size: rng.gen_range(3..=7),
        elite_count: rng.gen_range(5..=20),
        polytope_patience: rng.gen_range(3..=15),
        switch_threshold: rng.gen_range(0.0..=0.5),
        switch_probability: rng.gen_range(0.05..=0.3),
        cc_weight: rng.gen_range(5.0..=15.0),
        parent_id: None,
        meta_generation,
        runs_required,
        rng_seed: algo_seed,
    }
}

/// Mutate an algorithm's parameters
pub fn mutate_algorithm<R: Rng>(algo: &MetaAlgorithm, rng: &mut R, strength: f64) -> MetaAlgorithm {
    let mut child = algo.clone();
    child.id = None;
    child.parent_id = algo.id;

    // Mutate feature weights
    let mut weights = parse_feature_weights(&child.feature_weights);
    for weight in weights.values_mut() {
        if rng.gen::<f64>() < 0.3 {
            let delta = (rng.gen::<f64>() - 0.5) * 2.0 * strength;
            *weight = (*weight + delta).max(0.0).min(5.0);
            if rng.gen::<f64>() < 0.05 {
                *weight = 0.0;
            }
        }
    }
    child.feature_weights = serde_json::to_string(&weights).unwrap_or_default();

    // Mutate search strategy
    if rng.gen::<f64>() < 0.2 {
        child.similarity_radius = (child.similarity_radius + (rng.gen::<f64>() - 0.5) * 0.3)
            .clamp(0.05, 2.0);
    }
    if rng.gen::<f64>() < 0.2 {
        child.interpolation_weight = (child.interpolation_weight + (rng.gen::<f64>() - 0.5) * 0.3)
            .clamp(0.0, 1.0);
    }

    // Mutate GA params
    if rng.gen::<f64>() < 0.2 {
        child.population_size = ((child.population_size as f64 + (rng.gen::<f64>() - 0.5) * 40.0) as i32)
            .clamp(20, 200);
    }
    if rng.gen::<f64>() < 0.2 {
        child.max_generations = ((child.max_generations as f64 + (rng.gen::<f64>() - 0.5) * 6.0) as i32)
            .clamp(3, 30);
    }
    if rng.gen::<f64>() < 0.2 {
        child.mutation_rate = (child.mutation_rate + (rng.gen::<f64>() - 0.5) * 0.2)
            .clamp(0.1, 0.8);
    }
    if rng.gen::<f64>() < 0.2 {
        child.mutation_strength = (child.mutation_strength + (rng.gen::<f64>() - 0.5) * 0.2)
            .clamp(0.1, 0.6);
    }
    if rng.gen::<f64>() < 0.2 {
        child.crossover_rate = (child.crossover_rate + (rng.gen::<f64>() - 0.5) * 0.2)
            .clamp(0.5, 0.95);
    }
    if rng.gen::<f64>() < 0.2 {
        child.tournament_size = ((child.tournament_size as f64 + (rng.gen::<f64>() - 0.5) * 4.0) as i32)
            .clamp(2, 10);
    }
    if rng.gen::<f64>() < 0.2 {
        child.elite_count = ((child.elite_count as f64 + (rng.gen::<f64>() - 0.5) * 10.0) as i32)
            .clamp(2, 50);
    }

    // Mutate polytope switching
    if rng.gen::<f64>() < 0.2 {
        child.polytope_patience = ((child.polytope_patience as f64 + (rng.gen::<f64>() - 0.5) * 8.0) as i32)
            .clamp(1, 30);
    }
    if rng.gen::<f64>() < 0.2 {
        child.switch_threshold = (child.switch_threshold + (rng.gen::<f64>() - 0.5) * 0.3)
            .clamp(0.0, 1.0);
    }
    if rng.gen::<f64>() < 0.2 {
        child.switch_probability = (child.switch_probability + (rng.gen::<f64>() - 0.5) * 0.2)
            .clamp(0.0, 0.5);
    }

    // Mutate fitness weight
    if rng.gen::<f64>() < 0.2 {
        child.cc_weight = (child.cc_weight + (rng.gen::<f64>() - 0.5) * 5.0)
            .clamp(1.0, 20.0);
    }

    // Generate new seed for this child
    child.rng_seed = rng.gen();

    child
}

/// Crossover two algorithms to produce offspring
pub fn crossover_algorithms<R: Rng>(
    parent1: &MetaAlgorithm,
    parent2: &MetaAlgorithm,
    rng: &mut R,
) -> MetaAlgorithm {
    let mut child = parent1.clone();
    child.id = None;
    child.parent_id = parent1.id;  // Track primary parent

    // Crossover feature weights (uniform)
    let w1 = parse_feature_weights(&parent1.feature_weights);
    let w2 = parse_feature_weights(&parent2.feature_weights);
    let mut child_weights: HashMap<String, f64> = HashMap::new();
    for name in HEURISTIC_FEATURES {
        let v1 = w1.get(*name).copied().unwrap_or(1.0);
        let v2 = w2.get(*name).copied().unwrap_or(1.0);
        child_weights.insert(name.to_string(), if rng.gen::<bool>() { v1 } else { v2 });
    }
    child.feature_weights = serde_json::to_string(&child_weights).unwrap_or_default();

    // Crossover other params (50/50)
    if rng.gen::<bool>() { child.similarity_radius = parent2.similarity_radius; }
    if rng.gen::<bool>() { child.interpolation_weight = parent2.interpolation_weight; }
    if rng.gen::<bool>() { child.population_size = parent2.population_size; }
    if rng.gen::<bool>() { child.max_generations = parent2.max_generations; }
    if rng.gen::<bool>() { child.mutation_rate = parent2.mutation_rate; }
    if rng.gen::<bool>() { child.mutation_strength = parent2.mutation_strength; }
    if rng.gen::<bool>() { child.crossover_rate = parent2.crossover_rate; }
    if rng.gen::<bool>() { child.tournament_size = parent2.tournament_size; }
    if rng.gen::<bool>() { child.elite_count = parent2.elite_count; }
    if rng.gen::<bool>() { child.polytope_patience = parent2.polytope_patience; }
    if rng.gen::<bool>() { child.switch_threshold = parent2.switch_threshold; }
    if rng.gen::<bool>() { child.switch_probability = parent2.switch_probability; }
    if rng.gen::<bool>() { child.cc_weight = parent2.cc_weight; }

    // Generate new seed for this child
    child.rng_seed = rng.gen();

    child
}

// =============================================================================
// Generation Management
// =============================================================================

/// Initialize first generation with random algorithms
/// Uses master_seed to derive all algorithm seeds deterministically
pub fn init_generation_zero(
    conn: &Connection,
    population_size: i32,
    runs_required: i32,
    master_seed: u64,
) -> rusqlite::Result<()> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(master_seed);

    for i in 0..population_size {
        let mut algo = random_algorithm(&mut rng, 0, runs_required);
        algo.name = Some(format!("gen0_algo{}", i));
        crate::db::insert_meta_algorithm(conn, &algo)?;
    }

    // Store master seed in meta_state
    conn.execute(
        "UPDATE meta_state SET master_seed = ?1 WHERE id = 1",
        params![master_seed as i64],
    )?;

    Ok(())
}

/// Create next generation from top performers
/// Derives RNG seed from master_seed and generation number for reproducibility
pub fn evolve_next_generation(
    conn: &Connection,
    current_gen: i32,
    population_size: i32,
    runs_required: i32,
    elite_count: i32,
    mutation_rate: f64,
    mutation_strength: f64,
) -> rusqlite::Result<()> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let next_gen = current_gen + 1;

    // Get master seed from meta_state
    let master_seed: i64 = conn.query_row(
        "SELECT COALESCE(master_seed, 42) FROM meta_state WHERE id = 1",
        [],
        |row| row.get(0),
    )?;

    // Derive generation-specific seed from master_seed and generation number
    let gen_seed = (master_seed as u64)
        .wrapping_add(next_gen as u64)
        .wrapping_mul(6364136223846793005);
    let mut rng = StdRng::seed_from_u64(gen_seed);

    // Get top performers from current generation
    let top = crate::db::get_top_meta_algorithms(conn, current_gen, population_size)?;
    if top.is_empty() {
        // No completed algorithms yet, can't evolve
        return Ok(());
    }

    let elite: Vec<_> = top.into_iter().take(elite_count as usize).collect();

    // Create next generation
    for i in 0..population_size {
        let algo = if i < elite_count {
            // Keep elite unchanged (but as new entry) with NEW seed
            let (parent, _fitness) = &elite[i as usize % elite.len()];
            let mut child = parent.clone();
            child.id = None;
            child.parent_id = parent.id;
            child.meta_generation = next_gen;
            child.runs_required = runs_required;
            child.name = Some(format!("gen{}_elite{}", next_gen, i));
            child.rng_seed = rng.gen();  // New seed so it explores different polytopes
            child
        } else if rng.gen::<f64>() < mutation_rate {
            // Mutate a random elite
            let (parent, _) = &elite[rng.gen_range(0..elite.len())];
            let mut child = mutate_algorithm(parent, &mut rng, mutation_strength);
            child.meta_generation = next_gen;
            child.runs_required = runs_required;
            child.name = Some(format!("gen{}_mutant{}", next_gen, i));
            child
        } else {
            // Crossover two random elites
            let (p1, _) = &elite[rng.gen_range(0..elite.len())];
            let (p2, _) = &elite[rng.gen_range(0..elite.len())];
            let mut child = crossover_algorithms(p1, p2, &mut rng);
            // Also apply light mutation
            child = mutate_algorithm(&child, &mut rng, mutation_strength * 0.5);
            child.meta_generation = next_gen;
            child.runs_required = runs_required;
            child.name = Some(format!("gen{}_cross{}", next_gen, i));
            child
        };

        crate::db::insert_meta_algorithm(conn, &algo)?;
    }

    // Update meta_state
    crate::db::set_current_generation(conn, next_gen)?;

    Ok(())
}

/// Get number of completed trials for an algorithm
pub fn get_trial_count(conn: &Connection, algorithm_id: i64) -> rusqlite::Result<i32> {
    conn.query_row(
        "SELECT COUNT(*) FROM runs WHERE algorithm_id = ?1",
        params![algorithm_id],
        |row| row.get(0),
    )
}

/// Check if an algorithm has completed all required trials
pub fn is_algorithm_complete(conn: &Connection, algorithm_id: i64, runs_required: i32) -> rusqlite::Result<bool> {
    let count = get_trial_count(conn, algorithm_id)?;
    Ok(count >= runs_required)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use tempfile::tempdir;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(12345)
    }

    fn test_db() -> (tempfile::TempDir, Connection) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn_arc = crate::db::init_database(db_path.to_str().unwrap()).unwrap();
        let conn = std::sync::Arc::try_unwrap(conn_arc)
            .map(|m| m.into_inner().unwrap())
            .unwrap_or_else(|arc| {
                let _guard = arc.lock().unwrap();
                // Can't extract, but for tests we can just open a new connection
                Connection::open(db_path).unwrap()
            });
        (dir, conn)
    }

    // =========================================================================
    // Feature Weights Tests
    // =========================================================================

    #[test]
    fn test_default_feature_weights_contains_all_features() {
        let json = default_feature_weights_json();
        let weights = parse_feature_weights(&json);

        // All features should be present
        assert_eq!(weights.len(), HEURISTIC_FEATURES.len());

        // Each feature should have weight 1.0
        for feature in HEURISTIC_FEATURES {
            let weight = weights.get(*feature);
            assert!(weight.is_some(), "Missing feature: {}", feature);
            assert_eq!(*weight.unwrap(), 1.0, "Wrong default weight for {}", feature);
        }
    }

    #[test]
    fn test_random_feature_weights_produces_varied_values() {
        let mut rng = seeded_rng();
        let json = random_feature_weights_json(&mut rng);
        let weights = parse_feature_weights(&json);

        assert_eq!(weights.len(), HEURISTIC_FEATURES.len());

        // Check that values are in expected range [0, 3]
        for (name, &weight) in &weights {
            assert!(
                weight >= 0.0 && weight <= 3.0,
                "Weight {} for {} out of range [0, 3]",
                weight, name
            );
        }

        // With high probability, not all weights should be the same
        let values: Vec<f64> = weights.values().copied().collect();
        let first = values[0];
        let all_same = values.iter().all(|&v| (v - first).abs() < 0.001);
        assert!(!all_same, "All random weights are the same - unlikely with seed");
    }

    #[test]
    fn test_random_feature_weights_can_produce_zeros() {
        // Run enough times to see at least one zero (10% chance per feature)
        let mut found_zero = false;
        for seed in 0..50 {
            let mut rng = StdRng::seed_from_u64(seed);
            let json = random_feature_weights_json(&mut rng);
            let weights = parse_feature_weights(&json);
            if weights.values().any(|&w| w == 0.0) {
                found_zero = true;
                break;
            }
        }
        assert!(found_zero, "No zero weights found after 50 random generations");
    }

    #[test]
    fn test_focused_feature_weights_has_few_active() {
        // Test that focused weights only have 2-6 active features
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let json = random_feature_weights_focused(&mut rng);
            let weights = parse_feature_weights(&json);

            let active_count = weights.values().filter(|&&w| w > 0.0).count();
            assert!(
                active_count >= 2 && active_count <= 6,
                "Focused weights should have 2-6 active features, got {} (seed {})",
                active_count, seed
            );

            // Active weights should be in range [1.0, 3.0]
            for (name, &weight) in &weights {
                if weight > 0.0 {
                    assert!(
                        weight >= 1.0 && weight <= 3.0,
                        "Active weight {} for {} should be in [1.0, 3.0] (seed {})",
                        weight, name, seed
                    );
                }
            }
        }
    }

    #[test]
    fn test_broad_vs_focused_split() {
        // Test that random_feature_weights_json produces both broad and focused
        let mut broad_count = 0;
        let mut focused_count = 0;

        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let json = random_feature_weights_json(&mut rng);
            let weights = parse_feature_weights(&json);

            let active_count = weights.values().filter(|&&w| w > 0.0).count();
            if active_count <= 6 {
                focused_count += 1;
            } else {
                broad_count += 1;
            }
        }

        // With 50/50 split, expect roughly equal distribution (allow 20-80 range)
        assert!(
            broad_count >= 20 && broad_count <= 80,
            "Expected roughly 50/50 split, got {} broad vs {} focused",
            broad_count, focused_count
        );
    }

    #[test]
    fn test_parse_feature_weights_handles_empty_json() {
        let weights = parse_feature_weights("{}");
        assert!(weights.is_empty());
    }

    #[test]
    fn test_parse_feature_weights_handles_invalid_json() {
        let weights = parse_feature_weights("not valid json");
        assert!(weights.is_empty());
    }

    #[test]
    fn test_parse_feature_weights_roundtrip() {
        let json = default_feature_weights_json();
        let weights = parse_feature_weights(&json);
        let json2 = serde_json::to_string(&weights).unwrap();
        let weights2 = parse_feature_weights(&json2);
        assert_eq!(weights, weights2);
    }

    // =========================================================================
    // Weighted Distance Tests
    // =========================================================================

    #[test]
    fn test_weighted_distance_zero_for_identical() {
        let weights: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 2.0),
        ].into_iter().collect();

        let h: HashMap<String, f64> = [
            ("a".to_string(), 5.0),
            ("b".to_string(), 3.0),
        ].into_iter().collect();

        let dist = weighted_distance(&weights, &h, &h);
        assert!((dist - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_weighted_distance_respects_weights() {
        let h1: HashMap<String, f64> = [
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
        ].into_iter().collect();

        let h2: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 1.0),
        ].into_iter().collect();

        // Equal weights: both dimensions contribute equally
        let equal_weights: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 1.0),
        ].into_iter().collect();
        let d1 = weighted_distance(&equal_weights, &h1, &h2);

        // Only weight a: only a contributes
        let a_only: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.0),
        ].into_iter().collect();
        let d2 = weighted_distance(&a_only, &h1, &h2);

        // d1 = sqrt((1*1 + 1*1)/2) = 1.0
        // d2 = sqrt((1*1)/1) = 1.0
        assert!((d1 - 1.0).abs() < 0.0001);
        assert!((d2 - 1.0).abs() < 0.0001);

        // Higher weight on 'a' should increase distance when h2.a differs more
        let h3: HashMap<String, f64> = [
            ("a".to_string(), 10.0),
            ("b".to_string(), 1.0),
        ].into_iter().collect();

        let high_a: HashMap<String, f64> = [
            ("a".to_string(), 10.0),
            ("b".to_string(), 1.0),
        ].into_iter().collect();

        let low_a: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 10.0),
        ].into_iter().collect();

        let dist_high = weighted_distance(&high_a, &h1, &h3);
        let dist_low = weighted_distance(&low_a, &h1, &h3);

        // High weight on 'a' emphasizes the large diff (10), low weight on 'a' emphasizes small diff (1)
        assert!(dist_high > dist_low);
    }

    #[test]
    fn test_weighted_distance_missing_features() {
        let weights: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 1.0),
            ("c".to_string(), 1.0),
        ].into_iter().collect();

        // h1 missing "b", h2 missing "c"
        let h1: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("c".to_string(), 2.0),
        ].into_iter().collect();

        let h2: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 3.0),
        ].into_iter().collect();

        // Missing values default to 0.0
        // diff_a = 0, diff_b = 0 - 3 = -3, diff_c = 2 - 0 = 2
        // distance = sqrt((0 + 9 + 4) / 3) = sqrt(13/3) ≈ 2.08
        let dist = weighted_distance(&weights, &h1, &h2);
        assert!((dist - (13.0_f64 / 3.0).sqrt()).abs() < 0.0001);
    }

    #[test]
    fn test_weighted_distance_all_zero_weights() {
        let weights: HashMap<String, f64> = [
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
        ].into_iter().collect();

        let h1: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let h2: HashMap<String, f64> = [("a".to_string(), 100.0)].into_iter().collect();

        let dist = weighted_distance(&weights, &h1, &h2);
        assert_eq!(dist, f64::MAX);
    }

    #[test]
    fn test_weighted_distance_empty_weights() {
        let weights: HashMap<String, f64> = HashMap::new();
        let h1: HashMap<String, f64> = [("a".to_string(), 1.0)].into_iter().collect();
        let h2: HashMap<String, f64> = [("a".to_string(), 2.0)].into_iter().collect();

        let dist = weighted_distance(&weights, &h1, &h2);
        assert_eq!(dist, f64::MAX);
    }

    // =========================================================================
    // Random Algorithm Tests
    // =========================================================================

    #[test]
    fn test_feature_weights() {
        let json = default_feature_weights_json();
        let weights = parse_feature_weights(&json);
        assert_eq!(weights.len(), HEURISTIC_FEATURES.len());
        assert_eq!(weights.get("sphericity"), Some(&1.0));
    }

    #[test]
    fn test_random_algorithm() {
        let mut rng = rand::thread_rng();
        let algo = random_algorithm(&mut rng, 0, 10);
        assert!(algo.population_size >= 30 && algo.population_size <= 150);
        assert!(algo.similarity_radius >= 0.1 && algo.similarity_radius <= 1.0);
    }

    #[test]
    fn test_random_algorithm_respects_bounds() {
        let mut rng = seeded_rng();

        for _ in 0..100 {
            let algo = random_algorithm(&mut rng, 5, 15);

            // Check all bounds
            assert!(algo.similarity_radius >= 0.1 && algo.similarity_radius <= 1.0);
            assert!(algo.interpolation_weight >= 0.0 && algo.interpolation_weight <= 1.0);
            assert!(algo.population_size >= 30 && algo.population_size <= 150);
            assert!(algo.max_generations >= 5 && algo.max_generations <= 20);
            assert!(algo.mutation_rate >= 0.2 && algo.mutation_rate <= 0.6);
            assert!(algo.mutation_strength >= 0.2 && algo.mutation_strength <= 0.5);
            assert!(algo.crossover_rate >= 0.6 && algo.crossover_rate <= 0.9);
            assert!(algo.tournament_size >= 3 && algo.tournament_size <= 7);
            assert!(algo.elite_count >= 5 && algo.elite_count <= 20);
            assert!(algo.polytope_patience >= 3 && algo.polytope_patience <= 15);
            assert!(algo.switch_threshold >= 0.0 && algo.switch_threshold <= 0.5);
            assert!(algo.switch_probability >= 0.05 && algo.switch_probability <= 0.3);
            assert!(algo.cc_weight >= 5.0 && algo.cc_weight <= 15.0);

            // Check fixed values
            assert_eq!(algo.meta_generation, 5);
            assert_eq!(algo.runs_required, 15);
            assert!(algo.id.is_none());
            assert!(algo.parent_id.is_none());
        }
    }

    #[test]
    fn test_random_algorithm_has_valid_feature_weights() {
        let mut rng = seeded_rng();
        let algo = random_algorithm(&mut rng, 0, 10);
        let weights = parse_feature_weights(&algo.feature_weights);

        assert_eq!(weights.len(), HEURISTIC_FEATURES.len());
        for &weight in weights.values() {
            assert!(weight >= 0.0 && weight <= 3.0);
        }
    }

    // =========================================================================
    // Mutation Tests
    // =========================================================================

    #[test]
    fn test_mutate() {
        let mut rng = rand::thread_rng();
        let algo = random_algorithm(&mut rng, 0, 10);
        let mutated = mutate_algorithm(&algo, &mut rng, 0.5);
        // Should be different (with high probability)
        assert!(mutated.feature_weights != algo.feature_weights ||
                mutated.similarity_radius != algo.similarity_radius);
    }

    #[test]
    fn test_mutate_preserves_bounds() {
        let mut rng = seeded_rng();

        // Start with extreme values
        let extreme_algo = MetaAlgorithm {
            id: Some(1),
            name: Some("extreme".to_string()),
            version: META_ALGORITHM_VERSION,
            feature_weights: default_feature_weights_json(),
            similarity_radius: 0.05,  // Below minimum after clamp
            interpolation_weight: 0.0,
            population_size: 20,      // Below minimum after clamp
            max_generations: 3,       // Below minimum after clamp
            mutation_rate: 0.1,
            mutation_strength: 0.1,
            crossover_rate: 0.5,
            tournament_size: 2,
            elite_count: 2,
            polytope_patience: 1,
            switch_threshold: 0.0,
            switch_probability: 0.0,
            cc_weight: 1.0,
            parent_id: None,
            meta_generation: 0,
            runs_required: 10,
            rng_seed: 99999,
        };

        // Mutate many times and check bounds
        let mut algo = extreme_algo.clone();
        for _ in 0..100 {
            algo = mutate_algorithm(&algo, &mut rng, 1.0);

            assert!(algo.similarity_radius >= 0.05 && algo.similarity_radius <= 2.0);
            assert!(algo.interpolation_weight >= 0.0 && algo.interpolation_weight <= 1.0);
            assert!(algo.population_size >= 20 && algo.population_size <= 200);
            assert!(algo.max_generations >= 3 && algo.max_generations <= 30);
            assert!(algo.mutation_rate >= 0.1 && algo.mutation_rate <= 0.8);
            assert!(algo.mutation_strength >= 0.1 && algo.mutation_strength <= 0.6);
            assert!(algo.crossover_rate >= 0.5 && algo.crossover_rate <= 0.95);
            assert!(algo.tournament_size >= 2 && algo.tournament_size <= 10);
            assert!(algo.elite_count >= 2 && algo.elite_count <= 50);
            assert!(algo.polytope_patience >= 1 && algo.polytope_patience <= 30);
            assert!(algo.switch_threshold >= 0.0 && algo.switch_threshold <= 1.0);
            assert!(algo.switch_probability >= 0.0 && algo.switch_probability <= 0.5);
            assert!(algo.cc_weight >= 1.0 && algo.cc_weight <= 20.0);
        }
    }

    #[test]
    fn test_mutate_tracks_parent() {
        let mut rng = seeded_rng();
        let parent = MetaAlgorithm {
            id: Some(42),
            ..random_algorithm(&mut rng, 0, 10)
        };

        let child = mutate_algorithm(&parent, &mut rng, 0.5);
        assert_eq!(child.parent_id, Some(42));
        assert!(child.id.is_none());
    }

    #[test]
    fn test_mutate_feature_weights_stay_bounded() {
        let mut rng = seeded_rng();
        let algo = random_algorithm(&mut rng, 0, 10);

        // Mutate many times
        let mut mutated = algo.clone();
        for _ in 0..50 {
            mutated = mutate_algorithm(&mutated, &mut rng, 1.0);
            let weights = parse_feature_weights(&mutated.feature_weights);
            for &w in weights.values() {
                assert!(w >= 0.0 && w <= 5.0, "Weight {} out of bounds [0, 5]", w);
            }
        }
    }

    // =========================================================================
    // Crossover Tests
    // =========================================================================

    #[test]
    fn test_crossover_mixes_parents() {
        let mut rng = seeded_rng();

        let p1 = MetaAlgorithm {
            id: Some(1),
            name: Some("parent1".to_string()),
            version: META_ALGORITHM_VERSION,
            feature_weights: r#"{"sphericity": 1.0, "h11": 1.0}"#.to_string(),
            similarity_radius: 0.1,
            interpolation_weight: 0.1,
            population_size: 30,
            max_generations: 5,
            mutation_rate: 0.2,
            mutation_strength: 0.2,
            crossover_rate: 0.6,
            tournament_size: 3,
            elite_count: 5,
            polytope_patience: 3,
            switch_threshold: 0.0,
            switch_probability: 0.05,
            cc_weight: 5.0,
            parent_id: None,
            meta_generation: 0,
            runs_required: 10,
            rng_seed: 11111,
        };

        let p2 = MetaAlgorithm {
            id: Some(2),
            name: Some("parent2".to_string()),
            version: META_ALGORITHM_VERSION,
            feature_weights: r#"{"sphericity": 3.0, "h11": 3.0}"#.to_string(),
            similarity_radius: 0.9,
            interpolation_weight: 0.9,
            population_size: 150,
            max_generations: 20,
            mutation_rate: 0.6,
            mutation_strength: 0.5,
            crossover_rate: 0.9,
            tournament_size: 7,
            elite_count: 20,
            polytope_patience: 15,
            switch_threshold: 0.5,
            switch_probability: 0.3,
            cc_weight: 15.0,
            parent_id: None,
            meta_generation: 0,
            runs_required: 10,
            rng_seed: 22222,
        };

        // Run crossover many times and verify mixing
        let mut saw_p1_radius = false;
        let mut saw_p2_radius = false;

        for _ in 0..50 {
            let child = crossover_algorithms(&p1, &p2, &mut rng);

            if (child.similarity_radius - p1.similarity_radius).abs() < 0.001 {
                saw_p1_radius = true;
            }
            if (child.similarity_radius - p2.similarity_radius).abs() < 0.001 {
                saw_p2_radius = true;
            }

            // Parent tracking
            assert_eq!(child.parent_id, Some(1));
            assert!(child.id.is_none());
        }

        // Should see values from both parents with 50 trials
        assert!(saw_p1_radius && saw_p2_radius, "Crossover not mixing parents");
    }

    #[test]
    fn test_crossover_feature_weights_mix() {
        let mut rng = seeded_rng();

        let p1 = MetaAlgorithm {
            feature_weights: r#"{"sphericity": 0.0, "h11": 0.0, "h21": 0.0}"#.to_string(),
            ..random_algorithm(&mut rng, 0, 10)
        };

        let p2 = MetaAlgorithm {
            feature_weights: r#"{"sphericity": 5.0, "h11": 5.0, "h21": 5.0}"#.to_string(),
            ..random_algorithm(&mut rng, 0, 10)
        };

        // With enough crossovers, we should see mixed weights
        let mut saw_mixed = false;
        for _ in 0..20 {
            let child = crossover_algorithms(&p1, &p2, &mut rng);
            let weights = parse_feature_weights(&child.feature_weights);

            // Check if we got a mix (not all 0 or all 5 for tested features)
            let s = weights.get("sphericity").copied().unwrap_or(1.0);
            let h11 = weights.get("h11").copied().unwrap_or(1.0);
            if (s < 0.01 && h11 > 4.99) || (s > 4.99 && h11 < 0.01) {
                saw_mixed = true;
                break;
            }
        }
        assert!(saw_mixed, "Crossover should mix feature weights");
    }

    // =========================================================================
    // Database Integration Tests
    // =========================================================================

    #[test]
    fn test_init_generation_zero() {
        let (_dir, conn) = test_db();

        init_generation_zero(&conn, 5, 10, 12345).unwrap();

        let count: i32 = conn
            .query_row("SELECT COUNT(*) FROM meta_algorithms WHERE meta_generation = 0", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 5);

        // Check all have correct runs_required
        let trials: Vec<i32> = conn
            .prepare("SELECT runs_required FROM meta_algorithms")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert!(trials.iter().all(|&t| t == 10));
    }

    #[test]
    fn test_init_generation_zero_deterministic() {
        // With same seed, should produce identical algorithms
        let (_dir1, conn1) = test_db();
        let (_dir2, conn2) = test_db();

        init_generation_zero(&conn1, 5, 10, 99999).unwrap();
        init_generation_zero(&conn2, 5, 10, 99999).unwrap();

        // Fetch algorithm parameters from both
        let get_algo_params = |conn: &Connection| -> Vec<(f64, f64, i32)> {
            conn.prepare("SELECT similarity_radius, mutation_rate, population_size FROM meta_algorithms ORDER BY id")
                .unwrap()
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .unwrap()
                .map(|r| r.unwrap())
                .collect()
        };

        let params1 = get_algo_params(&conn1);
        let params2 = get_algo_params(&conn2);

        assert_eq!(params1, params2, "Same seed should produce identical algorithms");
    }

    #[test]
    fn test_get_trial_count_zero_initially() {
        let (_dir, conn) = test_db();

        let mut rng = seeded_rng();
        let algo = random_algorithm(&mut rng, 0, 10);
        let algo_id = crate::db::insert_meta_algorithm(&conn, &algo).unwrap();

        let count = get_trial_count(&conn, algo_id).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_get_trial_count_increments() {
        let (_dir, conn) = test_db();

        let mut rng = seeded_rng();
        let algo = random_algorithm(&mut rng, 0, 10);
        let algo_id = crate::db::insert_meta_algorithm(&conn, &algo).unwrap();

        // Insert runs
        for i in 0..3 {
            let run_id = crate::db::create_run(&conn, algo_id, (i + 1) as i32).unwrap();
            let run = crate::db::Run {
                id: Some(run_id),
                algorithm_id: algo_id,
                run_number: (i + 1) as i32,
                generations_run: 10,
                initial_fitness: 0.1,
                final_fitness: 0.5,
                fitness_improvement: 0.4,
                improvement_rate: 0.04,
                fitness_auc: 3.0,
                best_cc_log_error: 50.0,
                physics_success_rate: 0.8,
                unique_polytopes_tried: 100,
            };
            crate::db::complete_run(&conn, &run, None).unwrap();
            assert_eq!(get_trial_count(&conn, algo_id).unwrap(), i + 1);
        }
    }

    #[test]
    fn test_is_algorithm_complete() {
        let (_dir, conn) = test_db();

        let mut rng = seeded_rng();
        let algo = random_algorithm(&mut rng, 0, 3);  // Only 3 trials required
        let algo_id = crate::db::insert_meta_algorithm(&conn, &algo).unwrap();

        // Not complete with 0 trials
        assert!(!is_algorithm_complete(&conn, algo_id, 3).unwrap());

        // Add runs
        for i in 0..2 {
            let run_id = crate::db::create_run(&conn, algo_id, i + 1).unwrap();
            let run = crate::db::Run {
                id: Some(run_id),
                algorithm_id: algo_id,
                run_number: i + 1,
                generations_run: 10,
                initial_fitness: 0.1,
                final_fitness: 0.5,
                fitness_improvement: 0.4,
                improvement_rate: 0.04,
                fitness_auc: 3.0,
                best_cc_log_error: 50.0,
                physics_success_rate: 0.8,
                unique_polytopes_tried: 100,
            };
            crate::db::complete_run(&conn, &run, None).unwrap();
        }
        assert!(!is_algorithm_complete(&conn, algo_id, 3).unwrap());

        // Add one more - now complete
        let run_id = crate::db::create_run(&conn, algo_id, 3).unwrap();
        let run = crate::db::Run {
            id: Some(run_id),
            algorithm_id: algo_id,
            run_number: 3,
            generations_run: 10,
            initial_fitness: 0.1,
            final_fitness: 0.5,
            fitness_improvement: 0.4,
            improvement_rate: 0.04,
            fitness_auc: 3.0,
            best_cc_log_error: 50.0,
            physics_success_rate: 0.8,
            unique_polytopes_tried: 100,
        };
        crate::db::complete_run(&conn, &run, None).unwrap();
        assert!(is_algorithm_complete(&conn, algo_id, 3).unwrap());
    }

    #[test]
    fn test_evolve_next_generation_empty_returns_ok() {
        let (_dir, conn) = test_db();

        // No algorithms in generation 0, should return Ok without creating anything
        evolve_next_generation(&conn, 0, 10, 10, 4, 0.4, 0.3).unwrap();

        let count: i32 = conn
            .query_row("SELECT COUNT(*) FROM meta_algorithms", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_evolve_next_generation_creates_offspring() {
        let (_dir, conn) = test_db();

        // Create gen 0 with completed algorithms
        for i in 0..4 {
            let mut rng = StdRng::seed_from_u64(i);
            let algo = random_algorithm(&mut rng, 0, 1);
            let algo_id = crate::db::insert_meta_algorithm(&conn, &algo).unwrap();

            // Mark completed
            conn.execute(
                "UPDATE meta_algorithms SET status = 'completed' WHERE id = ?",
                params![algo_id],
            ).unwrap();

            // Add a run with varying fitness
            let run_id = crate::db::create_run(&conn, algo_id, 1).unwrap();
            let run = crate::db::Run {
                id: Some(run_id),
                algorithm_id: algo_id,
                run_number: 1,
                generations_run: 10,
                initial_fitness: 0.1,
                final_fitness: 0.1 * (i + 1) as f64,
                fitness_improvement: 0.05 * (i + 1) as f64,
                improvement_rate: 0.005 * (i + 1) as f64,
                fitness_auc: 1.0 * (i + 1) as f64,
                best_cc_log_error: 100.0 / (i + 1) as f64,
                physics_success_rate: 0.8,
                unique_polytopes_tried: 100,
            };
            crate::db::complete_run(&conn, &run, None).unwrap();
        }

        // Evolve to generation 1
        evolve_next_generation(&conn, 0, 6, 5, 2, 0.4, 0.3).unwrap();

        let gen1_count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM meta_algorithms WHERE meta_generation = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(gen1_count, 6);

        // Check current generation was updated
        let (current_gen, _) = crate::db::get_meta_state(&conn).unwrap();
        assert_eq!(current_gen, 1);
    }
}
