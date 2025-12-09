//! Genetic algorithm for real string theory landscape search
//!
//! This module implements a GA that uses actual physics computations
//! from the Python/JAX bridge instead of toy approximations.

use crate::constants;
use crate::db;
use crate::meta_ga::{parse_feature_weights, weighted_distance};
use crate::physics::{
    compute_physics, is_physics_available,
    PhysicsOutput, PolytopeData, Compactification,
};
use rand::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Configuration for the real physics GA
#[derive(Debug, Clone, Serialize)]
pub struct GaConfig {
    pub population_size: usize,
    pub elite_count: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub base_mutation_rate: f64,
    pub base_mutation_strength: f64,
    pub collapse_threshold: usize,
    pub hall_of_fame_size: usize,
}

impl Default for GaConfig {
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

/// Search strategy parameters evolved by the meta-GA
/// These control HOW we explore the polytope landscape
#[derive(Debug, Clone, Serialize)]
pub struct SearchStrategy {
    /// Weights for computing polytope similarity (from meta-GA feature_weights)
    /// Keys match heuristic column names in the database
    pub feature_weights: HashMap<String, f64>,

    /// How wide to search around good polytopes (0.1 = very local, 1.0 = broad)
    pub similarity_radius: f64,

    /// Balance between similarity search (0) and path-walking between good polytopes (1)
    pub interpolation_weight: f64,

    /// Generations before considering switching to a different polytope
    pub polytope_patience: i32,

    /// Minimum fitness improvement required to stay with current polytope
    pub switch_threshold: f64,

    /// Random probability of switching polytopes regardless of improvement
    pub switch_probability: f64,

    /// Weight for cosmological constant in fitness computation
    pub cc_weight: f64,
}

impl Default for SearchStrategy {
    fn default() -> Self {
        Self {
            feature_weights: HashMap::new(),  // Empty = use default (unweighted)
            similarity_radius: 0.5,
            interpolation_weight: 0.5,
            polytope_patience: 5,
            switch_threshold: 0.01,
            switch_probability: 0.1,
            cc_weight: 10.0,
        }
    }
}

impl SearchStrategy {
    /// Create from a MetaAlgorithm's parameters
    pub fn from_meta_algorithm(algo: &db::MetaAlgorithm) -> Self {
        Self {
            feature_weights: parse_feature_weights(&algo.feature_weights),
            similarity_radius: algo.similarity_radius,
            interpolation_weight: algo.interpolation_weight,
            polytope_patience: algo.polytope_patience,
            switch_threshold: algo.switch_threshold,
            switch_probability: algo.switch_probability,
            cc_weight: algo.cc_weight,
        }
    }

    /// Check if feature weights are defined (not empty)
    pub fn has_feature_weights(&self) -> bool {
        !self.feature_weights.is_empty()
    }
}

/// Feature vector for a polytope - like an embedding
/// Combines geometric features with physics evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolytopeFeatures {
    // === GEOMETRIC FEATURES ===

    // Hodge numbers (most important for physics)
    pub h11: f64,
    pub h21: f64,
    pub euler_char: f64,  // 2*(h11 - h21)

    // Vertex statistics
    pub vertex_count: f64,
    pub vertex_mean: [f64; 4],      // Mean of each coordinate
    pub vertex_std: [f64; 4],       // Std dev of each coordinate
    pub vertex_min: [f64; 4],
    pub vertex_max: [f64; 4],
    pub vertex_spread: f64,         // Max - min across all coords

    // Shape characteristics
    pub centroid_dist: f64,         // Avg distance from centroid
    pub max_vertex_norm: f64,       // Largest vertex magnitude
    pub min_vertex_norm: f64,
    pub aspect_ratio: f64,          // Elongation measure

    // Combinatorial
    pub coord_sum: f64,             // Sum of all vertex coordinates
    pub coord_abs_sum: f64,         // Sum of absolute values
    pub zero_count: f64,            // Number of zero coordinates
    pub negative_count: f64,        // Number of negative coordinates

    // === PHYSICS FEATURES (from PALP evaluation) ===

    // How close to universal constants (log-ratio, 0 = perfect match)
    pub alpha_em_error: f64,        // Fine structure constant error
    pub alpha_s_error: f64,         // Strong coupling error
    pub sin2_theta_w_error: f64,    // Weinberg angle error
    pub n_gen_error: f64,           // Generation count error (should be 0 for 3-gen)
    pub lambda_error: f64,          // Cosmological constant error

    // Mass ratios
    pub m_e_planck_error: f64,      // Electron mass ratio error

    // Geometry quality
    pub cy_volume: f64,             // Calabi-Yau volume
    pub flux_tadpole: f64,          // Flux tadpole (constraint)

    // Overall fitness
    pub fitness: f64,
    pub physics_success: bool,
}

impl PolytopeFeatures {
    /// Compute geometric features from a polytope's vertices (physics features set to default)
    pub fn from_polytope(vertices: &[Vec<i32>], h11: i32, h21: i32) -> Self {
        let n_vertices = vertices.len();
        let mut vertex_mean = [0.0f64; 4];
        let mut vertex_min = [f64::MAX; 4];
        let mut vertex_max = [f64::MIN; 4];
        let mut coord_sum = 0.0f64;
        let mut coord_abs_sum = 0.0f64;
        let mut zero_count = 0.0f64;
        let mut negative_count = 0.0f64;

        // First pass: compute means, mins, maxs
        for vertex in vertices {
            for (j, &coord) in vertex.iter().enumerate().take(4) {
                let v = coord as f64;
                vertex_mean[j] += v;
                vertex_min[j] = vertex_min[j].min(v);
                vertex_max[j] = vertex_max[j].max(v);
                coord_sum += v;
                coord_abs_sum += v.abs();
                if v == 0.0 { zero_count += 1.0; }
                if v < 0.0 { negative_count += 1.0; }
            }
        }

        if n_vertices > 0 {
            for j in 0..4 {
                vertex_mean[j] /= n_vertices as f64;
            }
        }

        // Second pass: compute std devs and distances
        let mut vertex_std = [0.0f64; 4];
        let mut centroid_dist_sum = 0.0f64;
        let mut max_vertex_norm = 0.0f64;
        let mut min_vertex_norm = f64::MAX;

        for vertex in vertices {
            let mut dist_sq = 0.0f64;
            let mut norm_sq = 0.0f64;
            for (j, &coord) in vertex.iter().enumerate().take(4) {
                let v = coord as f64;
                let diff = v - vertex_mean[j];
                vertex_std[j] += diff * diff;
                dist_sq += diff * diff;
                norm_sq += v * v;
            }
            centroid_dist_sum += dist_sq.sqrt();
            let norm = norm_sq.sqrt();
            max_vertex_norm = max_vertex_norm.max(norm);
            min_vertex_norm = min_vertex_norm.min(norm);
        }

        if n_vertices > 0 {
            for j in 0..4 {
                vertex_std[j] = (vertex_std[j] / n_vertices as f64).sqrt();
            }
        }

        let vertex_spread = (0..4)
            .map(|j| vertex_max[j] - vertex_min[j])
            .fold(0.0f64, |a, b| a.max(b));

        let aspect_ratio = if min_vertex_norm > 0.0 {
            max_vertex_norm / min_vertex_norm
        } else {
            max_vertex_norm
        };

        Self {
            // Geometric features
            h11: h11 as f64,
            h21: h21 as f64,
            euler_char: 2.0 * (h11 as f64 - h21 as f64),
            vertex_count: n_vertices as f64,
            vertex_mean,
            vertex_std,
            vertex_min,
            vertex_max,
            vertex_spread,
            centroid_dist: centroid_dist_sum / n_vertices.max(1) as f64,
            max_vertex_norm,
            min_vertex_norm,
            aspect_ratio,
            coord_sum,
            coord_abs_sum,
            zero_count,
            negative_count,
            // Physics features (set after evaluation)
            alpha_em_error: f64::MAX,
            alpha_s_error: f64::MAX,
            sin2_theta_w_error: f64::MAX,
            n_gen_error: f64::MAX,
            lambda_error: f64::MAX,
            m_e_planck_error: f64::MAX,
            cy_volume: 0.0,
            flux_tadpole: f64::MAX,
            fitness: 0.0,
            physics_success: false,
        }
    }

    /// Update physics features after PALP evaluation
    pub fn update_physics(&mut self, physics: &crate::physics::PhysicsOutput, fitness: f64) {
        use crate::constants;

        self.physics_success = physics.success;
        self.fitness = fitness;

        if physics.success {
            // Compute log-ratio errors (0 = perfect, higher = worse)
            self.alpha_em_error = if physics.alpha_em > 0.0 && constants::ALPHA_EM > 0.0 {
                (physics.alpha_em / constants::ALPHA_EM).ln().abs()
            } else {
                10.0
            };

            self.alpha_s_error = if physics.alpha_s > 0.0 && constants::ALPHA_STRONG > 0.0 {
                (physics.alpha_s / constants::ALPHA_STRONG).ln().abs()
            } else {
                10.0
            };

            self.sin2_theta_w_error = if physics.sin2_theta_w > 0.0 && constants::SIN2_THETA_W > 0.0 {
                (physics.sin2_theta_w / constants::SIN2_THETA_W).ln().abs()
            } else {
                10.0
            };

            // Generation error: |computed - 3|
            self.n_gen_error = (physics.n_generations as f64 - 3.0).abs();

            // Cosmological constant is tricky (target is tiny)
            self.lambda_error = if physics.cosmological_constant.abs() > 0.0 {
                (physics.cosmological_constant.abs() / constants::COSMOLOGICAL_CONSTANT.abs()).ln().abs() / 100.0
            } else {
                3.0  // Better than huge values
            };

            self.m_e_planck_error = if physics.m_e_planck_ratio > 0.0 && constants::ELECTRON_PLANCK_RATIO > 0.0 {
                (physics.m_e_planck_ratio / constants::ELECTRON_PLANCK_RATIO).ln().abs()
            } else {
                50.0
            };

            self.cy_volume = physics.cy_volume;
            self.flux_tadpole = physics.flux_tadpole;
        }
    }

    /// Convert to a flat vector for similarity computations
    /// Includes both geometric and physics features
    pub fn to_vector(&self) -> Vec<f64> {
        let mut v = vec![
            // Geometric features (normalized where possible)
            self.h11 / 100.0,           // Scale Hodge numbers
            self.h21 / 100.0,
            self.euler_char / 10.0,
            self.vertex_count / 50.0,
            self.vertex_spread / 10.0,
            self.centroid_dist / 10.0,
            self.max_vertex_norm / 10.0,
            self.min_vertex_norm / 10.0,
            self.aspect_ratio / 10.0,
            self.coord_sum / 100.0,
            self.coord_abs_sum / 100.0,
            self.zero_count / 50.0,
            self.negative_count / 50.0,
        ];
        // Vertex statistics (already reasonably scaled)
        for &m in &self.vertex_mean { v.push(m / 5.0); }
        for &s in &self.vertex_std { v.push(s / 5.0); }

        // Physics features (errors - already log-scale, lower = better)
        // Clamp to reasonable range for similarity computation
        v.push(self.alpha_em_error.min(10.0) / 10.0);
        v.push(self.alpha_s_error.min(10.0) / 10.0);
        v.push(self.sin2_theta_w_error.min(10.0) / 10.0);
        v.push(self.n_gen_error.min(10.0) / 10.0);
        v.push(self.lambda_error.min(10.0) / 10.0);
        v.push(self.m_e_planck_error.min(50.0) / 50.0);
        v.push(self.cy_volume.min(100.0) / 100.0);
        v.push(self.flux_tadpole.abs().min(100.0) / 100.0);
        v.push(self.fitness);  // 0-1 already

        v
    }

    /// Number of features in the vector
    pub const FEATURE_COUNT: usize = 13 + 8 + 9;  // geo + vertex_stats + physics

    /// Cosine similarity between two feature vectors
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Euclidean distance (normalized)
    pub fn distance(&self, other: &Self) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        v1.iter().zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Statistics for a cluster of polytopes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    pub centroid: Vec<f64>,         // Cluster centroid in feature space
    pub evaluations: u64,
    pub fitness_sum: f64,
    pub fitness_best: f64,
    pub polytope_ids: Vec<usize>,
    pub feature_sum: Vec<f64>,      // Running sum for centroid update
}

impl ClusterStats {
    pub fn new(n_features: usize) -> Self {
        Self {
            centroid: vec![0.0; n_features],
            evaluations: 0,
            fitness_sum: 0.0,
            fitness_best: 0.0,
            polytope_ids: Vec::new(),
            feature_sum: vec![0.0; n_features],
        }
    }

    pub fn avg_fitness(&self) -> f64 {
        if self.evaluations > 0 {
            self.fitness_sum / self.evaluations as f64
        } else {
            0.0
        }
    }

    /// Update centroid with new features
    pub fn update_centroid(&mut self, features: &[f64]) {
        for (i, &f) in features.iter().enumerate() {
            if i < self.feature_sum.len() {
                self.feature_sum[i] += f;
            }
        }
        if self.evaluations > 0 {
            for (i, c) in self.centroid.iter_mut().enumerate() {
                *c = self.feature_sum[i] / self.evaluations as f64;
            }
        }
    }

    /// UCB-style selection weight: exploitation + exploration bonus
    pub fn selection_weight(&self, total_evals: u64, exploration_factor: f64) -> f64 {
        if self.evaluations == 0 {
            return f64::MAX; // Unexplored clusters get priority
        }
        let exploitation = self.avg_fitness();
        let exploration = exploration_factor * ((total_evals as f64).ln() / self.evaluations as f64).sqrt();
        exploitation + exploration
    }
}

/// Mutation pattern tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationPattern {
    pub attempts: u64,
    pub improvements: u64,
}

impl MutationPattern {
    pub fn new() -> Self {
        Self { attempts: 0, improvements: 0 }
    }

    pub fn success_rate(&self) -> f64 {
        if self.attempts > 0 {
            self.improvements as f64 / self.attempts as f64
        } else {
            0.5 // Prior: 50% success rate
        }
    }
}

/// Hot polytope - one that has shown good results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPolytope {
    pub id: usize,
    pub fitness: f64,
    pub offspring_success_rate: f64,
}

/// Persistent cluster state for adaptive polytope selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    /// Clusters keyed by (h11, h21, vertex_bucket)
    pub clusters: HashMap<String, ClusterStats>,
    /// Polytopes that have produced good results
    pub hot_polytopes: Vec<HotPolytope>,
    /// Track which mutations work
    pub mutation_patterns: HashMap<String, MutationPattern>,
    /// Total evaluations across all clusters
    pub total_evaluations: u64,
    /// Last update timestamp
    pub last_updated: String,
}

impl ClusterState {
    pub fn new() -> Self {
        Self {
            clusters: HashMap::new(),
            hot_polytopes: Vec::new(),
            mutation_patterns: HashMap::new(),
            total_evaluations: 0,
            last_updated: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Load from file or create new
    pub fn load_or_new(path: &str) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(state) = serde_json::from_str(&data) {
                println!("Loaded cluster state from {}", path);
                return state;
            }
        }
        println!("Creating new cluster state");
        Self::new()
    }

    /// Save to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Get cluster key for a polytope (based on Hodge numbers and vertex bucket)
    pub fn cluster_key(h11: u32, h21: u32, vertex_count: usize) -> String {
        // Bucket vertex count into ranges
        let vertex_bucket = match vertex_count {
            0..=7 => "v5-7",
            8..=10 => "v8-10",
            11..=15 => "v11-15",
            16..=20 => "v16-20",
            21..=25 => "v21-25",
            _ => "v26+",
        };
        format!("h{}_{}__{}", h11, h21, vertex_bucket)
    }

    /// Update cluster stats after evaluation with full feature vector
    pub fn update_with_features(&mut self, features: &PolytopeFeatures, polytope_id: usize) {
        let key = Self::cluster_key(
            features.h11 as u32,
            features.h21 as u32,
            features.vertex_count as usize,
        );

        let n_features = PolytopeFeatures::FEATURE_COUNT;
        let cluster = self.clusters.entry(key).or_insert_with(|| ClusterStats::new(n_features));

        cluster.evaluations += 1;
        cluster.fitness_sum += features.fitness;
        if features.fitness > cluster.fitness_best {
            cluster.fitness_best = features.fitness;
        }

        // Update centroid with new feature vector
        let fv = features.to_vector();
        cluster.update_centroid(&fv);

        if !cluster.polytope_ids.contains(&polytope_id) && cluster.polytope_ids.len() < 1000 {
            cluster.polytope_ids.push(polytope_id);
        }

        self.total_evaluations += 1;
        self.last_updated = chrono::Utc::now().to_rfc3339();

        // Track hot polytopes (good fitness)
        if features.fitness > 0.4 {
            self.add_hot_polytope(polytope_id, features.fitness);
        }
    }

    /// Legacy update without features (for compatibility)
    pub fn update(&mut self, h11: u32, h21: u32, vertex_count: usize, polytope_id: usize, fitness: f64) {
        let key = Self::cluster_key(h11, h21, vertex_count);
        let n_features = PolytopeFeatures::FEATURE_COUNT;
        let cluster = self.clusters.entry(key).or_insert_with(|| ClusterStats::new(n_features));

        cluster.evaluations += 1;
        cluster.fitness_sum += fitness;
        if fitness > cluster.fitness_best {
            cluster.fitness_best = fitness;
        }
        if !cluster.polytope_ids.contains(&polytope_id) && cluster.polytope_ids.len() < 1000 {
            cluster.polytope_ids.push(polytope_id);
        }

        self.total_evaluations += 1;
        self.last_updated = chrono::Utc::now().to_rfc3339();

        if fitness > 0.4 {
            self.add_hot_polytope(polytope_id, fitness);
        }
    }

    /// Find the nearest cluster to a feature vector
    pub fn find_nearest_cluster(&self, features: &PolytopeFeatures) -> Option<&str> {
        let fv = features.to_vector();
        let mut best_key: Option<&str> = None;
        let mut best_dist = f64::MAX;

        for (key, cluster) in &self.clusters {
            if cluster.centroid.len() == fv.len() {
                let dist: f64 = fv.iter()
                    .zip(cluster.centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_key = Some(key.as_str());
                }
            }
        }
        best_key
    }

    /// Add or update a hot polytope
    fn add_hot_polytope(&mut self, id: usize, fitness: f64) {
        if let Some(existing) = self.hot_polytopes.iter_mut().find(|p| p.id == id) {
            if fitness > existing.fitness {
                existing.fitness = fitness;
            }
        } else {
            self.hot_polytopes.push(HotPolytope {
                id,
                fitness,
                offspring_success_rate: 0.5,
            });
            // Keep top 100
            self.hot_polytopes.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            self.hot_polytopes.truncate(100);
        }
    }

    /// Track mutation outcome
    pub fn record_mutation(&mut self, pattern_name: &str, improved: bool) {
        let pattern = self.mutation_patterns.entry(pattern_name.to_string())
            .or_insert_with(MutationPattern::new);
        pattern.attempts += 1;
        if improved {
            pattern.improvements += 1;
        }
    }

    /// Select a cluster using UCB
    pub fn select_cluster<R: Rng>(&self, rng: &mut R, exploration_factor: f64) -> Option<&str> {
        if self.clusters.is_empty() {
            return None;
        }

        let weights: Vec<(f64, &str)> = self.clusters.iter()
            .map(|(k, v)| (v.selection_weight(self.total_evaluations, exploration_factor), k.as_str()))
            .collect();

        let total_weight: f64 = weights.iter().map(|(w, _)| *w).sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut r = rng.gen::<f64>() * total_weight;
        for (weight, key) in weights {
            r -= weight;
            if r <= 0.0 {
                return Some(key);
            }
        }

        self.clusters.keys().next().map(|s| s.as_str())
    }

    /// Get a random polytope ID from a cluster
    pub fn random_from_cluster<R: Rng>(&self, cluster_key: &str, rng: &mut R) -> Option<usize> {
        self.clusters.get(cluster_key)
            .and_then(|c| {
                if c.polytope_ids.is_empty() {
                    None
                } else {
                    Some(c.polytope_ids[rng.gen_range(0..c.polytope_ids.len())])
                }
            })
    }

    /// Get mutation guidance - which patterns have worked
    pub fn get_mutation_bias(&self) -> HashMap<String, f64> {
        self.mutation_patterns.iter()
            .map(|(k, v)| (k.clone(), v.success_rate()))
            .collect()
    }
}

/// An individual in the real physics GA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub genome: Compactification,
    pub physics: Option<PhysicsOutput>,
    pub fitness: f64,
}

impl Individual {
    /// Create a new random individual
    pub fn random<R: Rng>(rng: &mut R, polytope_data: &PolytopeData) -> Self {
        Self::random_filtered(rng, polytope_data, None)
    }

    /// Create a new random individual, optionally filtering to specific polytope IDs
    pub fn random_filtered<R: Rng>(rng: &mut R, polytope_data: &PolytopeData, filter: Option<&[usize]>) -> Self {
        let genome = Compactification::random_filtered(rng, polytope_data, filter);
        Self {
            genome,
            physics: None,
            fitness: 0.0,
        }
    }

    /// Evaluate this individual's fitness using real physics
    pub fn evaluate(&mut self, polytope_data: &PolytopeData) {
        if !is_physics_available() {
            panic!("Physics bridge not available! Cannot run without real physics.");
        }

        let vertices = polytope_data
            .get(self.genome.polytope_id)
            .map(|p| p.vertices)
            .unwrap_or_default();

        let physics = compute_physics(&self.genome, &vertices);

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

    let mut total_score = 0.0;
    let mut total_weight = 0.0;

    // Helper: score a value against target, with physical validity check
    // Returns (score, is_valid) where score is 0 if value is physically impossible
    let score_value = |computed: f64, target: f64, must_be_positive: bool| -> f64 {
        // Physical validity: if must be positive and isn't, score is 0
        if must_be_positive && computed <= 0.0 {
            return 0.0;
        }

        // Log-ratio scoring: how close are we on a log scale?
        let ratio = if target.abs() > 1e-30 && computed.abs() > 1e-30 {
            (computed / target).abs()
        } else {
            return 0.0;
        };

        // Score: 1.0 when ratio = 1, decreasing as ratio deviates
        let log_ratio = ratio.ln().abs();
        (-log_ratio).exp()
    };

    // Gauge couplings - MUST be positive (they come from 4-cycle volumes)
    // Negative values are physically impossible, so they score 0
    let alpha_em_score = score_value(physics.alpha_em, constants::ALPHA_EM, true);
    total_score += 1.0 * alpha_em_score;
    total_weight += 1.0;

    let alpha_s_score = score_value(physics.alpha_s, constants::ALPHA_STRONG, true);
    total_score += 1.0 * alpha_s_score;
    total_weight += 1.0;

    // sin²θ_W - must be in [0, 1] (it's sin² of an angle)
    // Values outside this range are physically impossible
    let sin2_score = if physics.sin2_theta_w >= 0.0 && physics.sin2_theta_w <= 1.0 {
        score_value(physics.sin2_theta_w, constants::SIN2_THETA_W, false)
    } else {
        0.0
    };
    total_score += 1.0 * sin2_score;
    total_weight += 1.0;

    // Number of generations - discrete value, exact match preferred
    let n_gen_score = score_value(physics.n_generations as f64, constants::NUM_GENERATIONS as f64, false);
    total_score += 2.0 * n_gen_score;
    total_weight += 2.0;

    // Cosmological constant - THE key target, weight heavily (10x others)
    // This is the famous fine-tuning problem: Λ ≈ 10^-122 in Planck units
    let cc_ratio = if constants::COSMOLOGICAL_CONSTANT.abs() > 1e-150 {
        (physics.cosmological_constant / constants::COSMOLOGICAL_CONSTANT).abs()
    } else {
        0.0
    };
    let cc_log = if cc_ratio > 0.0 { cc_ratio.ln().abs() } else { 300.0 };
    // More aggressive scoring: each order of magnitude off reduces score
    let cc_score = (-cc_log / 50.0).exp();
    total_score += 10.0 * cc_score;
    total_weight += 10.0;

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
pub struct LandscapeSearcher {
    pub config: GaConfig,
    /// Search strategy evolved by meta-GA (feature weights, similarity params, etc.)
    pub strategy: SearchStrategy,
    pub polytope_data: Arc<PolytopeData>,
    /// If set, only search within these polytope IDs
    pub polytope_filter: Option<Vec<usize>>,
    pub population: Vec<Individual>,
    pub best_ever: Option<Individual>,
    pub hall_of_fame: Vec<Individual>,
    pub history: Vec<RealGenerationStats>,
    pub generation: usize,
    pub total_evaluated: u64,
    pub stagnation_count: usize,
    pub collapse_count: usize,
    pub cluster_state: ClusterState,
    pub verbose: bool,
    cluster_state_path: String,
    rng: StdRng,
    /// Database connection for persistent storage
    db_conn: Option<Arc<Mutex<Connection>>>,
    /// Run ID for tracking evaluations
    run_id: Option<i64>,
    /// Cache of heuristics per polytope for weighted similarity search
    heuristics_cache: HashMap<usize, HashMap<String, f64>>,
    /// Generations since last polytope switch (for polytope_patience)
    gens_on_current_polytope: usize,
    /// Best fitness achieved on current polytope (for switch_threshold)
    best_fitness_on_current_polytope: f64,
}

impl LandscapeSearcher {
    /// Create a new searcher with default strategy
    pub fn new(config: GaConfig, polytope_path: &str) -> Self {
        Self::new_with_seed(config, SearchStrategy::default(), polytope_path, None, None, None, None)
    }

    /// Create a new searcher with an optional polytope filter
    pub fn new_with_filter(config: GaConfig, polytope_path: &str, polytope_filter: Option<Vec<usize>>) -> Self {
        Self::new_with_seed(config, SearchStrategy::default(), polytope_path, polytope_filter, None, None, None)
    }

    /// Create a new searcher with database connection and run ID (legacy, uses default strategy)
    pub fn new_with_db(
        config: GaConfig,
        polytope_path: &str,
        polytope_filter: Option<Vec<usize>>,
        db_conn: Option<Arc<Mutex<Connection>>>,
        run_id: Option<i64>,
    ) -> Self {
        Self::new_with_seed(config, SearchStrategy::default(), polytope_path, polytope_filter, db_conn, run_id, None)
    }

    /// Create a new searcher with full configuration including meta-GA search strategy
    /// Uses entropy-based RNG (non-deterministic)
    pub fn new_with_strategy(
        config: GaConfig,
        strategy: SearchStrategy,
        polytope_path: &str,
        polytope_filter: Option<Vec<usize>>,
        db_conn: Option<Arc<Mutex<Connection>>>,
        run_id: Option<i64>,
    ) -> Self {
        Self::new_with_seed(config, strategy, polytope_path, polytope_filter, db_conn, run_id, None)
    }

    /// Create a new searcher with full configuration and deterministic seed
    /// If seed is None, uses entropy (non-deterministic). If seed is Some(u64), results are reproducible.
    pub fn new_with_seed(
        config: GaConfig,
        strategy: SearchStrategy,
        polytope_path: &str,
        polytope_filter: Option<Vec<usize>>,
        db_conn: Option<Arc<Mutex<Connection>>>,
        run_id: Option<i64>,
        seed: Option<u64>,
    ) -> Self {
        // Physics bridge must be initialized by caller (search.rs)
        assert!(is_physics_available(), "Physics bridge REQUIRED but not initialized");

        // Load polytope data
        let polytope_data = Arc::new(PolytopeData::load(polytope_path).expect("Failed to load polytope data"));
        println!("Using {} polytopes from Kreuzer-Skarke database", polytope_data.len());

        // Validate filter IDs
        if let Some(ref filter) = polytope_filter {
            let max_id = polytope_data.len();
            let valid_ids: Vec<usize> = filter.iter()
                .filter(|&&id| id < max_id)
                .copied()
                .collect();
            if valid_ids.len() < filter.len() {
                eprintln!("Warning: {} polytope IDs were out of range and skipped",
                    filter.len() - valid_ids.len());
            }
            println!("Filtering to {} specific polytope IDs", valid_ids.len());
        }

        // Log strategy info if feature weights are defined
        if strategy.has_feature_weights() {
            println!("Using evolved search strategy with {} feature weights",
                strategy.feature_weights.len());
            println!("  similarity_radius: {:.3}, interpolation_weight: {:.3}",
                strategy.similarity_radius, strategy.interpolation_weight);
            println!("  polytope_patience: {}, switch_threshold: {:.3}, switch_prob: {:.3}",
                strategy.polytope_patience, strategy.switch_threshold, strategy.switch_probability);
        }

        // Load or create cluster state
        let cluster_state_path = "cluster_state.json".to_string();
        let cluster_state = ClusterState::load_or_new(&cluster_state_path);
        println!("Cluster state: {} clusters, {} total evaluations",
            cluster_state.clusters.len(), cluster_state.total_evaluations);

        // Note: Run registration now happens via insert_run() with a Run struct
        // For meta-GA runs, this is done in run_trial() which creates the Run record
        // For standalone runs, evaluations are still recorded but without a run_id

        // Use deterministic seed if provided, otherwise entropy
        let mut rng = match seed {
            Some(s) => {
                println!("Using deterministic seed: {}", s);
                StdRng::seed_from_u64(s)
            }
            None => StdRng::from_entropy(),
        };

        // Initialize population
        let population: Vec<Individual> = (0..config.population_size)
            .map(|_| Individual::random_filtered(&mut rng, &polytope_data, polytope_filter.as_deref()))
            .collect();

        Self {
            config,
            strategy,
            polytope_data,
            polytope_filter,
            population,
            best_ever: None,
            hall_of_fame: Vec::new(),
            history: Vec::new(),
            generation: 0,
            total_evaluated: 0,
            stagnation_count: 0,
            collapse_count: 0,
            cluster_state,
            verbose: false,
            cluster_state_path,
            rng,
            db_conn,
            run_id,
            heuristics_cache: HashMap::new(),
            gens_on_current_polytope: 0,
            best_fitness_on_current_polytope: 0.0,
        }
    }

    /// Run one generation of the GA
    pub fn step(&mut self) {
        self.generation += 1;

        // Evaluate population and update cluster state with full feature vectors
        let pop_size = self.population.len();
        for (i, individual) in self.population.iter_mut().enumerate() {
            if individual.physics.is_none() {
                if self.verbose {
                    eprint!("  [{}/{}] polytope {} ... ", i + 1, pop_size, individual.genome.polytope_id);
                }
                individual.evaluate(&self.polytope_data);
                self.total_evaluated += 1;
                if self.verbose {
                    if let Some(ref p) = individual.physics {
                        if p.success {
                            eprintln!("fit:{:.4} α_em:{:.2e} N:{}", individual.fitness, p.alpha_em, p.n_generations);
                        } else {
                            eprintln!("FAIL: {}", p.error.as_deref().unwrap_or("?"));
                        }
                    }
                }

                // Record evaluation to database
                if let (Some(ref conn), Some(ref physics)) = (&self.db_conn, &individual.physics) {
                    if let Ok(locked) = conn.lock() {
                        // Ensure polytope exists in db
                        if let Some(polytope) = self.polytope_data.get(individual.genome.polytope_id) {
                            let vertices_json = serde_json::to_string(&polytope.vertices).unwrap_or_default();
                            let _ = db::upsert_polytope(
                                &locked,
                                individual.genome.polytope_id as i64,
                                polytope.h11,
                                polytope.h12,
                                polytope.vertices.len() as i32,
                                &vertices_json,
                            );
                        }

                        // Record evaluation
                        let _ = db::insert_evaluation(
                            &locked,
                            individual.genome.polytope_id as i64,
                            self.run_id,
                            Some(self.generation as i32),
                            &individual.genome,
                            physics,
                            individual.fitness,
                        );
                    }
                }

                // Compute full feature vector (geometric + physics)
                if let Some(polytope) = self.polytope_data.get(individual.genome.polytope_id) {
                    let mut features = PolytopeFeatures::from_polytope(
                        &polytope.vertices,
                        polytope.h11,
                        polytope.h12,  // h12 = h21 for CY3
                    );

                    // Add physics features from evaluation
                    if let Some(ref physics) = individual.physics {
                        features.update_physics(physics, individual.fitness);
                    }

                    // Update cluster state with full feature vector
                    self.cluster_state.update_with_features(&features, individual.genome.polytope_id);
                }
            }
        }

        // Periodically save cluster state (every 100 generations)
        if self.generation % 100 == 0 {
            if let Err(e) = self.cluster_state.save(&self.cluster_state_path) {
                eprintln!("Warning: Failed to save cluster state: {}", e);
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

    fn update_hall_of_fame(&mut self, candidate: Individual) {
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
        println!("Stagnation detected, reinitializing population");
        self.collapse_count += 1;
        self.stagnation_count = 0;

        // Save cluster state before collapse
        let _ = self.cluster_state.save(&self.cluster_state_path);

        // Keep elites
        let elite_count = self.config.elite_count;

        // Replace half with cluster-guided random individuals
        let fresh_count = self.population.len() / 2;
        for i in elite_count..(elite_count + fresh_count) {
            self.population[i] = self.create_cluster_guided_individual();
        }

        // Inject some from hall of fame (with heavy mutation)
        let hof_inject = (self.hall_of_fame.len() / 4).min(self.population.len() / 10);
        for i in 0..hof_inject {
            if elite_count + fresh_count + i < self.population.len() {
                let idx = self.rng.gen_range(0..self.hall_of_fame.len());
                let mut mutant = self.hall_of_fame[idx].clone();
                mutant.genome.mutate_filtered(&mut self.rng, 1.0, &self.polytope_data, self.polytope_filter.as_deref());
                mutant.physics = None;  // Force re-evaluation
                self.population[elite_count + fresh_count + i] = mutant;
            }
        }

        // Inject from hot polytopes if available
        let hot_inject = self.cluster_state.hot_polytopes.len().min(self.population.len() / 20);
        for i in 0..hot_inject {
            let idx = elite_count + fresh_count + hof_inject + i;
            if idx < self.population.len() && i < self.cluster_state.hot_polytopes.len() {
                let hot_id = self.cluster_state.hot_polytopes[i].id;
                if hot_id < self.polytope_data.len() {
                    let mut individual = Individual::random_filtered(&mut self.rng, &self.polytope_data, self.polytope_filter.as_deref());
                    individual.genome.polytope_id = hot_id;
                    individual.physics = None;
                    self.population[idx] = individual;
                }
            }
        }

        // Rest: heavily mutated current population
        for i in (elite_count + fresh_count + hof_inject + hot_inject)..self.population.len() {
            self.population[i].genome.mutate_filtered(&mut self.rng, 1.0, &self.polytope_data, self.polytope_filter.as_deref());
            self.population[i].physics = None;
        }
    }

    /// Get heuristics for a polytope from database (or cache)
    fn get_polytope_heuristics(&mut self, polytope_id: usize) -> Option<HashMap<String, f64>> {
        // Check cache first
        if let Some(heuristics) = self.heuristics_cache.get(&polytope_id) {
            return Some(heuristics.clone());
        }

        // Query database if available
        if let Some(ref conn) = self.db_conn {
            if let Ok(locked) = conn.lock() {
                if let Ok(Some(heuristics_data)) = db::get_heuristics(&locked, polytope_id as i64) {
                    let heuristics_map = heuristics_data.to_map();
                    // Cache for future use
                    self.heuristics_cache.insert(polytope_id, heuristics_map.clone());
                    return Some(heuristics_map);
                }
            }
        }

        None
    }

    /// Find polytopes similar to a reference polytope using weighted distance
    /// Returns up to `limit` polytope IDs sorted by similarity (closest first)
    fn find_similar_polytopes(&mut self, reference_id: usize, limit: usize) -> Vec<usize> {
        // Need feature weights to do weighted search
        if !self.strategy.has_feature_weights() {
            return Vec::new();
        }

        let ref_heuristics = match self.get_polytope_heuristics(reference_id) {
            Some(h) => h,
            None => return Vec::new(),
        };

        // Clone filter to avoid borrow conflicts
        let filter = self.polytope_filter.clone();
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        // Collect hot polytope IDs first to avoid borrow conflicts
        let hot_poly_ids: Vec<usize> = self.cluster_state.hot_polytopes.iter().map(|h| h.id).collect();

        // Search through cached heuristics or hot polytopes
        // (We can't search all 12M polytopes efficiently here)
        for &hot_poly in &hot_poly_ids {
            if hot_poly == reference_id {
                continue;
            }

            // Check filter
            let in_filter = filter.is_none() || filter.as_ref().unwrap().contains(&hot_poly);
            if !in_filter {
                continue;
            }

            if let Some(heuristics) = self.get_polytope_heuristics(hot_poly) {
                let dist = weighted_distance(&self.strategy.feature_weights, &ref_heuristics, &heuristics);
                if dist < self.strategy.similarity_radius * 10.0 {  // Scale radius for distance units
                    candidates.push((hot_poly, dist));
                }
            }
        }

        // Collect cluster polytope IDs first to avoid borrow conflicts
        let cluster_poly_ids: Vec<usize> = self.cluster_state.clusters.values()
            .flat_map(|c| c.polytope_ids.iter().copied())
            .filter(|&id| id != reference_id && !candidates.iter().any(|(cid, _)| *cid == id))
            .collect();

        for poly_id in cluster_poly_ids {
            let in_filter = filter.is_none() || filter.as_ref().unwrap().contains(&poly_id);
            if !in_filter {
                continue;
            }

            if let Some(heuristics) = self.get_polytope_heuristics(poly_id) {
                let dist = weighted_distance(&self.strategy.feature_weights, &ref_heuristics, &heuristics);
                if dist < self.strategy.similarity_radius * 10.0 {
                    candidates.push((poly_id, dist));
                }
            }
        }

        // Sort by distance (ascending) and take top `limit`
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);

        candidates.into_iter().map(|(id, _)| id).collect()
    }

    /// Select a polytope using similarity-weighted search around good performers
    fn select_polytope_by_similarity(&mut self) -> Option<usize> {
        if !self.strategy.has_feature_weights() || self.cluster_state.hot_polytopes.is_empty() {
            return None;
        }

        // Pick a reference polytope (from hot list with probability weighted by fitness)
        let hot = &self.cluster_state.hot_polytopes;
        let total_fitness: f64 = hot.iter().map(|h| h.fitness).sum();
        if total_fitness <= 0.0 {
            return None;
        }

        let mut r = self.rng.gen::<f64>() * total_fitness;
        let mut reference_id = hot[0].id;
        for h in hot {
            r -= h.fitness;
            if r <= 0.0 {
                reference_id = h.id;
                break;
            }
        }

        // Find similar polytopes
        let similar = self.find_similar_polytopes(reference_id, 10);
        if similar.is_empty() {
            return None;
        }

        // Pick one randomly from similar set (biased toward closer ones)
        let idx = (self.rng.gen::<f64>() * self.rng.gen::<f64>() * similar.len() as f64) as usize;
        let idx = idx.min(similar.len() - 1);
        Some(similar[idx])
    }

    /// Create an individual using cluster-guided selection
    fn create_cluster_guided_individual(&mut self) -> Individual {
        let r = self.rng.gen::<f64>();
        // Clone filter to avoid borrow conflicts with self-mutating methods
        let filter = self.polytope_filter.clone();
        let filter_slice = filter.as_deref();

        // If we have feature weights, use similarity-weighted selection with probability
        // controlled by interpolation_weight (0 = never use similarity, 1 = always use)
        if self.strategy.has_feature_weights() && r < self.strategy.interpolation_weight {
            if let Some(polytope_id) = self.select_polytope_by_similarity() {
                let mut individual = Individual::random_filtered(&mut self.rng, &self.polytope_data, filter_slice);
                individual.genome.polytope_id = polytope_id;
                return individual;
            }
        }

        // Adjust remaining probability space
        let r_adjusted = if self.strategy.has_feature_weights() {
            let denom = 1.0 - self.strategy.interpolation_weight;
            if denom > 0.001 {
                (r - self.strategy.interpolation_weight) / denom
            } else {
                r
            }
        } else {
            r
        };

        // 70% exploit (use high-performing clusters), 20% explore, 10% hot polytopes
        if r_adjusted < 0.7 && !self.cluster_state.clusters.is_empty() {
            // Exploit: select from promising cluster
            if let Some(cluster_key) = self.cluster_state.select_cluster(&mut self.rng, 0.1) {
                if let Some(polytope_id) = self.cluster_state.random_from_cluster(cluster_key, &mut self.rng) {
                    // Only use this polytope if it's in the filter (or no filter)
                    let in_filter = filter_slice.is_none() || filter_slice.unwrap().contains(&polytope_id);
                    if polytope_id < self.polytope_data.len() && in_filter {
                        let mut individual = Individual::random_filtered(&mut self.rng, &self.polytope_data, filter_slice);
                        individual.genome.polytope_id = polytope_id;
                        return individual;
                    }
                }
            }
        } else if r_adjusted < 0.9 {
            // Explore: random selection
            return Individual::random_filtered(&mut self.rng, &self.polytope_data, filter_slice);
        } else if !self.cluster_state.hot_polytopes.is_empty() {
            // Hot polytope selection
            let idx = self.rng.gen_range(0..self.cluster_state.hot_polytopes.len());
            let hot_id = self.cluster_state.hot_polytopes[idx].id;
            // Only use hot polytope if it's in the filter (or no filter)
            let in_filter = filter_slice.is_none() || filter_slice.unwrap().contains(&hot_id);
            if hot_id < self.polytope_data.len() && in_filter {
                let mut individual = Individual::random_filtered(&mut self.rng, &self.polytope_data, filter_slice);
                individual.genome.polytope_id = hot_id;
                return individual;
            }
        }

        // Fallback to random
        Individual::random_filtered(&mut self.rng, &self.polytope_data, filter_slice)
    }

    /// Save cluster state
    pub fn save_cluster_state(&self) -> std::io::Result<()> {
        self.cluster_state.save(&self.cluster_state_path)
    }

    /// Get best fitness achieved
    pub fn get_best_fitness(&self) -> f64 {
        self.best_ever.as_ref().map(|b| b.fitness).unwrap_or(0.0)
    }

    /// Get best polytope ID
    pub fn get_best_polytope_id(&self) -> Option<i64> {
        self.best_ever.as_ref().map(|b| b.genome.polytope_id as i64)
    }

    /// Get total evaluations
    pub fn get_total_evaluated(&self) -> u64 {
        self.total_evaluated
    }

    /// Get current generation
    pub fn get_generation(&self) -> usize {
        self.generation
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
                Individual {
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
                child.genome.mutate_filtered(&mut self.rng, strength, &self.polytope_data, self.polytope_filter.as_deref());
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
    pub fn best(&self) -> Option<&Individual> {
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
            best_ever: &'a Option<Individual>,
            hall_of_fame: &'a Vec<Individual>,
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
pub fn format_fitness_line(individual: &Individual) -> String {
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
pub fn format_fitness_report(individual: &Individual) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SearchStrategy Tests
    // =========================================================================

    #[test]
    fn test_search_strategy_default() {
        let strategy = SearchStrategy::default();

        assert!(strategy.feature_weights.is_empty());
        assert!((strategy.similarity_radius - 0.5).abs() < 0.001);
        assert!((strategy.interpolation_weight - 0.5).abs() < 0.001);
        assert_eq!(strategy.polytope_patience, 5);
        assert!((strategy.switch_threshold - 0.01).abs() < 0.001);
        assert!((strategy.switch_probability - 0.1).abs() < 0.001);
        assert!((strategy.cc_weight - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_search_strategy_has_feature_weights() {
        let empty = SearchStrategy::default();
        assert!(!empty.has_feature_weights());

        let with_weights = SearchStrategy {
            feature_weights: [("sphericity".to_string(), 1.0)].into_iter().collect(),
            ..Default::default()
        };
        assert!(with_weights.has_feature_weights());
    }

    #[test]
    fn test_search_strategy_from_meta_algorithm() {
        use crate::db::MetaAlgorithm;
        use crate::meta_ga::default_feature_weights_json;

        let algo = MetaAlgorithm {
            id: Some(1),
            name: Some("test".to_string()),
            version: 1,
            feature_weights: default_feature_weights_json(),
            similarity_radius: 0.3,
            interpolation_weight: 0.7,
            population_size: 100,
            max_generations: 15,
            mutation_rate: 0.4,
            mutation_strength: 0.3,
            crossover_rate: 0.8,
            tournament_size: 5,
            elite_count: 10,
            polytope_patience: 8,
            switch_threshold: 0.05,
            switch_probability: 0.15,
            cc_weight: 12.0,
            parent_id: None,
            meta_generation: 0,
            runs_required: 10,
            rng_seed: 54321,
        };

        let strategy = SearchStrategy::from_meta_algorithm(&algo);

        assert!(!strategy.feature_weights.is_empty());
        assert_eq!(strategy.feature_weights.get("sphericity"), Some(&1.0));
        assert!((strategy.similarity_radius - 0.3).abs() < 0.001);
        assert!((strategy.interpolation_weight - 0.7).abs() < 0.001);
        assert_eq!(strategy.polytope_patience, 8);
        assert!((strategy.switch_threshold - 0.05).abs() < 0.001);
        assert!((strategy.switch_probability - 0.15).abs() < 0.001);
        assert!((strategy.cc_weight - 12.0).abs() < 0.001);
    }

    // =========================================================================
    // GaConfig Tests
    // =========================================================================

    #[test]
    fn test_ga_config_default() {
        let config = GaConfig::default();

        assert_eq!(config.population_size, 500);
        assert_eq!(config.elite_count, 10);
        assert_eq!(config.tournament_size, 5);
        assert!((config.crossover_rate - 0.85).abs() < 0.001);
        assert!((config.base_mutation_rate - 0.4).abs() < 0.001);
        assert!((config.base_mutation_strength - 0.35).abs() < 0.001);
        assert_eq!(config.collapse_threshold, 2000);
        assert_eq!(config.hall_of_fame_size, 50);
    }

    // =========================================================================
    // PolytopeFeatures Tests
    // =========================================================================

    #[test]
    fn test_polytope_features_from_simple_polytope() {
        // Simple unit simplex vertices
        let vertices = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ];

        let features = PolytopeFeatures::from_polytope(&vertices, 4, 1);

        assert!((features.h11 - 4.0).abs() < 0.001);
        assert!((features.h21 - 1.0).abs() < 0.001);
        assert!((features.euler_char - 6.0).abs() < 0.001);
        assert!((features.vertex_count - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_polytope_features_to_vector() {
        let vertices = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
        ];

        let features = PolytopeFeatures::from_polytope(&vertices, 3, 0);
        let vec = features.to_vector();

        assert_eq!(vec.len(), PolytopeFeatures::FEATURE_COUNT);
    }

    #[test]
    fn test_polytope_features_cosine_similarity_identical() {
        let vertices = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
        ];

        let f1 = PolytopeFeatures::from_polytope(&vertices, 3, 0);
        let f2 = PolytopeFeatures::from_polytope(&vertices, 3, 0);

        let sim = f1.cosine_similarity(&f2);
        assert!((sim - 1.0).abs() < 0.0001, "Identical features should have similarity 1.0");
    }

    #[test]
    fn test_polytope_features_distance_identical() {
        let vertices = vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
        ];

        let f1 = PolytopeFeatures::from_polytope(&vertices, 3, 0);
        let f2 = PolytopeFeatures::from_polytope(&vertices, 3, 0);

        let dist = f1.distance(&f2);
        assert!(dist < 0.0001, "Identical features should have distance 0");
    }

    // =========================================================================
    // ClusterState Tests
    // =========================================================================

    #[test]
    fn test_cluster_key_format() {
        assert_eq!(ClusterState::cluster_key(5, 2, 6), "h5_2__v5-7");
        assert_eq!(ClusterState::cluster_key(10, 7, 9), "h10_7__v8-10");
        assert_eq!(ClusterState::cluster_key(100, 97, 30), "h100_97__v26+");
    }

    #[test]
    fn test_cluster_state_new() {
        let state = ClusterState::new();

        assert!(state.clusters.is_empty());
        assert!(state.hot_polytopes.is_empty());
        assert!(state.mutation_patterns.is_empty());
        assert_eq!(state.total_evaluations, 0);
    }

    #[test]
    fn test_cluster_state_update() {
        let mut state = ClusterState::new();

        state.update(5, 2, 6, 12345, 0.75);

        assert_eq!(state.total_evaluations, 1);
        assert!(state.clusters.contains_key("h5_2__v5-7"));

        let cluster = state.clusters.get("h5_2__v5-7").unwrap();
        assert_eq!(cluster.evaluations, 1);
        assert!((cluster.fitness_sum - 0.75).abs() < 0.001);
        assert!((cluster.fitness_best - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_cluster_stats_avg_fitness() {
        let mut stats = ClusterStats::new(5);
        assert!((stats.avg_fitness() - 0.0).abs() < 0.001);

        stats.evaluations = 4;
        stats.fitness_sum = 2.0;
        assert!((stats.avg_fitness() - 0.5).abs() < 0.001);
    }
}
