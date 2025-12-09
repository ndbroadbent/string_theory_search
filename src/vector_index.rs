//! HNSW vector index for fast approximate nearest neighbor search
//!
//! Uses usearch for memory-mapped HNSW index over polytope heuristics.
//! Query flow:
//! 1. Get top-K candidates by unweighted cosine similarity (fast ANN)
//! 2. Rerank candidates by weighted distance (exact) using raw vectors
//! 3. Return best match for similarity/interpolation

use std::collections::HashMap;
use std::path::Path;
use memmap2::Mmap;
use rusqlite::Connection;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::meta_ga::HEURISTIC_FEATURES;

/// Number of dimensions in heuristic vectors
pub const VECTOR_DIM: usize = 42;

/// Number of candidates to retrieve from ANN before reranking
pub const ANN_CANDIDATES: usize = 10_000;

/// HNSW index for polytope heuristics with raw vectors for reranking
pub struct HeuristicsIndex {
    index: Index,
    /// Polytope IDs in index order (index position -> polytope_id)
    polytope_ids: Vec<i64>,
    /// Reverse map: polytope_id -> index position
    id_to_position: HashMap<i64, usize>,
    /// Memory-mapped raw vectors for weighted reranking (f32 * VECTOR_DIM * count)
    raw_vectors: Option<Mmap>,
}

impl HeuristicsIndex {
    /// Build index from SQLite database with parallel insertion
    pub fn build_from_db(conn: &Connection, index_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        println!("Building HNSW index from database...");

        // Count polytopes
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM heuristics",
            [],
            |row| row.get(0),
        )?;
        println!("  Found {} polytopes with heuristics", count);

        let num_threads = rayon::current_num_threads();
        println!("  Using {} threads for parallel indexing", num_threads);

        // Create index with thread support
        let options = IndexOptions {
            dimensions: VECTOR_DIM,
            metric: MetricKind::Cos,  // Cosine similarity for normalized vectors
            quantization: ScalarKind::F32,
            connectivity: 16,  // M parameter - connections per node
            expansion_add: 128,  // efConstruction
            expansion_search: 64,  // ef
            multi: false,
        };

        let index = Index::new(&options)?;
        // Reserve with thread count for parallel insertion
        index.reserve(count as usize)?;
        // Note: Index implements Send + Sync, so parallel add() is safe

        // Phase 1: Load all data from SQLite (single-threaded, SQLite limitation)
        println!("  Phase 1: Loading data from database...");
        let mut polytope_ids = Vec::with_capacity(count as usize);
        let mut raw_vectors_data: Vec<f32> = Vec::with_capacity(count as usize * VECTOR_DIM);
        let mut normalized_vectors: Vec<[f32; VECTOR_DIM]> = Vec::with_capacity(count as usize);

        let mut stmt = conn.prepare(
            "SELECT polytope_id, sphericity, inertia_isotropy, flatness_3d, flatness_2d,
                    spikiness, loner_score, conformity_ratio, intrinsic_dim_estimate,
                    coord_mean, coord_std, coord_median, coord_skewness, coord_kurtosis,
                    distance_kurtosis, shannon_entropy, compression_ratio, sorted_compression_ratio,
                    sort_compression_gain, joint_entropy, symmetry_x, symmetry_y, symmetry_z, symmetry_w,
                    chirality_x, chirality_y, chirality_z, chirality_w, chirality_optimal, handedness_det,
                    zero_count, one_count, prime_count, fibonacci_count, phi_ratio_count,
                    outlier_score, outlier_count_2sigma, outlier_count_3sigma, outlier_max_zscore, max_exposure,
                    vertex_count, h11, h21
             FROM heuristics
             ORDER BY polytope_id"
        )?;

        let mut rows = stmt.query([])?;
        let mut loaded = 0usize;

        while let Some(row) = rows.next()? {
            let polytope_id: i64 = row.get(0)?;

            // Extract all 42 features into a vector
            let mut raw_vec = [0.0f32; VECTOR_DIM];
            for i in 0..VECTOR_DIM {
                raw_vec[i] = row.get::<_, Option<f64>>(i + 1)?.unwrap_or(0.0) as f32;
            }

            // Store raw vector for reranking
            raw_vectors_data.extend_from_slice(&raw_vec);

            // Normalize for HNSW cosine similarity
            let mut normalized_vec = raw_vec;
            let norm: f32 = normalized_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for v in &mut normalized_vec {
                    *v /= norm;
                }
            }

            polytope_ids.push(polytope_id);
            normalized_vectors.push(normalized_vec);
            loaded += 1;

            if loaded % 1_000_000 == 0 {
                println!("    Loaded {} polytopes...", loaded);
            }
        }
        println!("  Loaded {} polytopes from database", loaded);

        // Phase 2: Parallel index insertion
        println!("  Phase 2: Building HNSW index in parallel...");
        let progress = AtomicUsize::new(0);
        let total = normalized_vectors.len();

        // Parallel insertion - Index is Send + Sync so this is safe
        normalized_vectors
            .par_iter()
            .enumerate()
            .for_each(|(position, vec)| {
                if let Err(e) = index.add(position as u64, vec) {
                    eprintln!("Warning: Failed to add vector {}: {}", position, e);
                }

                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 1_000_000 == 0 {
                    println!("    Indexed {} / {} polytopes ({:.1}%)", done, total, 100.0 * done as f64 / total as f64);
                }
            });

        let position = normalized_vectors.len();
        println!("  Indexed {} polytopes total", position);

        // Build id_to_position map
        let id_to_position: HashMap<i64, usize> = polytope_ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| (id, pos))
            .collect();

        // Save index
        println!("  Saving index to {:?}...", index_path);
        index.save(index_path.to_str().ok_or("Invalid path")?)?;

        // Save polytope IDs mapping
        let ids_path = index_path.with_extension("ids");
        let ids_bytes: Vec<u8> = polytope_ids.iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        std::fs::write(&ids_path, &ids_bytes)?;
        println!("  Saved polytope ID mapping to {:?}", ids_path);

        // Save raw vectors for reranking
        let raw_path = index_path.with_extension("raw");
        let raw_bytes: Vec<u8> = raw_vectors_data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&raw_path, &raw_bytes)?;
        println!("  Saved raw vectors to {:?} ({:.1} MB)", raw_path, raw_bytes.len() as f64 / 1_000_000.0);

        // Memory-map the raw vectors
        let raw_file = std::fs::File::open(&raw_path)?;
        let raw_vectors = Some(unsafe { Mmap::map(&raw_file)? });

        Ok(Self {
            index,
            polytope_ids,
            id_to_position,
            raw_vectors,
        })
    }

    /// Load existing index from disk
    pub fn load(index_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading HNSW index from {:?}...", index_path);

        let options = IndexOptions {
            dimensions: VECTOR_DIM,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            multi: false,
        };

        let index = Index::new(&options)?;
        index.load(index_path.to_str().ok_or("Invalid path")?)?;

        // Load polytope IDs
        let ids_path = index_path.with_extension("ids");
        let ids_bytes = std::fs::read(&ids_path)?;
        let polytope_ids: Vec<i64> = ids_bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let id_to_position: HashMap<i64, usize> = polytope_ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| (id, pos))
            .collect();

        // Memory-map raw vectors
        let raw_path = index_path.with_extension("raw");
        let raw_vectors = if raw_path.exists() {
            let raw_file = std::fs::File::open(&raw_path)?;
            Some(unsafe { Mmap::map(&raw_file)? })
        } else {
            println!("  Warning: raw vectors file not found, weighted reranking disabled");
            None
        };

        println!("  Loaded index with {} polytopes", polytope_ids.len());

        Ok(Self {
            index,
            polytope_ids,
            id_to_position,
            raw_vectors,
        })
    }

    /// Load or build index
    pub fn load_or_build(index_path: &Path, db_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if index_path.exists() && index_path.with_extension("ids").exists() {
            Self::load(index_path)
        } else {
            let conn = Connection::open(db_path)?;
            Self::build_from_db(&conn, index_path)
        }
    }

    /// Get raw heuristics for a polytope by index position
    fn get_raw_vector(&self, position: usize) -> Option<[f32; VECTOR_DIM]> {
        let mmap = self.raw_vectors.as_ref()?;
        let start = position * VECTOR_DIM * 4; // f32 = 4 bytes
        let end = start + VECTOR_DIM * 4;
        if end > mmap.len() {
            return None;
        }

        let bytes = &mmap[start..end];
        let mut vec = [0.0f32; VECTOR_DIM];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            vec[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        Some(vec)
    }

    /// Compute weighted distance between query and a candidate
    fn weighted_distance(
        &self,
        query: &[f32; VECTOR_DIM],
        candidate_position: usize,
        weights: &HashMap<String, f64>,
    ) -> f64 {
        let Some(candidate) = self.get_raw_vector(candidate_position) else {
            return f64::MAX;
        };

        let mut sum_sq = 0.0;
        let mut total_weight = 0.0;

        for (i, name) in HEURISTIC_FEATURES.iter().enumerate() {
            let weight = weights.get(*name).copied().unwrap_or(0.0);
            if weight > 0.0 {
                let diff = (query[i] - candidate[i]) as f64;
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

    /// Find similar polytopes using ANN + weighted reranking
    ///
    /// 1. Get query polytope's normalized vector
    /// 2. Find top ANN_CANDIDATES by cosine similarity (fast)
    /// 3. Rerank by weighted distance using raw vectors
    /// 4. Return top `limit` polytope IDs with distances
    pub fn find_similar(
        &self,
        query_heuristics: &HashMap<String, f64>,
        feature_weights: &HashMap<String, f64>,
        limit: usize,
    ) -> Vec<(i64, f64)> {
        // Build raw query vector
        let mut query_raw = [0.0f32; VECTOR_DIM];
        for (i, name) in HEURISTIC_FEATURES.iter().enumerate() {
            query_raw[i] = query_heuristics.get(*name).copied().unwrap_or(0.0) as f32;
        }

        // Normalize for ANN search
        let mut query_normalized = query_raw;
        let norm: f32 = query_normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut query_normalized {
                *v /= norm;
            }
        }

        // ANN search for candidates
        let results = match self.index.search(&query_normalized, ANN_CANDIDATES) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        // Rerank by weighted distance using raw vectors
        let mut candidates: Vec<(i64, f64)> = Vec::with_capacity(results.keys.len());

        for (idx, &key) in results.keys.iter().enumerate() {
            let position = key as usize;
            if position >= self.polytope_ids.len() {
                continue;
            }
            let polytope_id = self.polytope_ids[position];

            // Compute weighted distance if we have raw vectors, else use ANN distance
            let dist = if self.raw_vectors.is_some() {
                self.weighted_distance(&query_raw, position, feature_weights)
            } else {
                results.distances[idx] as f64
            };

            candidates.push((polytope_id, dist));
        }

        // Sort by weighted distance and return top `limit`
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);
        candidates
    }

    /// Find single most similar polytope (convenience method)
    pub fn find_most_similar(
        &self,
        query_heuristics: &HashMap<String, f64>,
        feature_weights: &HashMap<String, f64>,
    ) -> Option<i64> {
        self.find_similar(query_heuristics, feature_weights, 1)
            .into_iter()
            .next()
            .map(|(id, _)| id)
    }

    /// Find polytopes along interpolation path between two reference points
    ///
    /// Given two good polytopes A and B, find polytopes near the line A->B
    /// in the weighted feature space
    pub fn find_along_path(
        &self,
        heuristics_a: &HashMap<String, f64>,
        heuristics_b: &HashMap<String, f64>,
        feature_weights: &HashMap<String, f64>,
        t: f64,  // interpolation parameter 0..1
        limit: usize,
    ) -> Vec<(i64, f64)> {
        // Interpolated query point
        let mut query: HashMap<String, f64> = HashMap::new();
        for name in HEURISTIC_FEATURES {
            let va = heuristics_a.get(*name).copied().unwrap_or(0.0);
            let vb = heuristics_b.get(*name).copied().unwrap_or(0.0);
            query.insert(name.to_string(), va + t * (vb - va));
        }

        self.find_similar(&query, feature_weights, limit)
    }

    /// Find single polytope along interpolation path (convenience method)
    pub fn find_one_along_path(
        &self,
        heuristics_a: &HashMap<String, f64>,
        heuristics_b: &HashMap<String, f64>,
        feature_weights: &HashMap<String, f64>,
        t: f64,
    ) -> Option<i64> {
        self.find_along_path(heuristics_a, heuristics_b, feature_weights, t, 1)
            .into_iter()
            .next()
            .map(|(id, _)| id)
    }

    /// Get polytope ID at index position
    pub fn get_polytope_id(&self, position: usize) -> Option<i64> {
        self.polytope_ids.get(position).copied()
    }

    /// Get index position for polytope ID
    pub fn get_position(&self, polytope_id: i64) -> Option<usize> {
        self.id_to_position.get(&polytope_id).copied()
    }

    /// Get raw heuristics as HashMap for a polytope
    pub fn get_heuristics(&self, polytope_id: i64) -> Option<HashMap<String, f64>> {
        let position = self.get_position(polytope_id)?;
        let raw = self.get_raw_vector(position)?;

        let mut map = HashMap::with_capacity(VECTOR_DIM);
        for (i, name) in HEURISTIC_FEATURES.iter().enumerate() {
            map.insert(name.to_string(), raw[i] as f64);
        }
        Some(map)
    }

    /// Number of indexed polytopes
    pub fn len(&self) -> usize {
        self.polytope_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.polytope_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_dim() {
        assert_eq!(VECTOR_DIM, HEURISTIC_FEATURES.len());
    }
}
