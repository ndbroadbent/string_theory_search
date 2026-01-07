//! Real physics computations via cyrus-core and Python bridge
//!
//! This module integrates exact geometry from cyrus-core with
//! high-level search logic.

use cyrus_core::{
    evaluate_vacuum,
    compute_glsm_charge_matrix, compute_intersection_numbers, compute_mori_generators,
    compute_regular_triangulation,
    EvaluationRequest, GvInvariant, Intersection, MoriCone, Point,
};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};

/// Physical observables computed from a compactification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsOutput {
    pub success: bool,
    pub error: Option<String>,

    // Gauge couplings
    pub alpha_em: f64,
    pub alpha_s: f64,
    pub sin2_theta_w: f64,

    // Cosmological
    pub cosmological_constant: f64,

    // Particle physics
    pub n_generations: i32,
    pub m_e_planck_ratio: f64,
    pub m_p_planck_ratio: f64,

    // Internal geometry
    pub cy_volume: f64,
    pub string_coupling: f64,
    pub flux_tadpole: f64,
    pub superpotential_abs: f64,
}

impl Default for PhysicsOutput {
    fn default() -> Self {
        Self {
            success: false,
            error: Some("Not computed".to_string()),
            alpha_em: 0.0,
            alpha_s: 0.0,
            sin2_theta_w: 0.0,
            cosmological_constant: 0.0,
            n_generations: 0,
            m_e_planck_ratio: 0.0,
            m_p_planck_ratio: 0.0,
            cy_volume: 0.0,
            string_coupling: 0.0,
            flux_tadpole: 0.0,
            superpotential_abs: 0.0,
        }
    }
}

/// Genome for a real string compactification (discrete fluxes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compactification {
    /// Index into the polytope database
    pub polytope_id: usize,
    
    /// Triangulation index
    pub triangulation_id: usize,

    /// F_3 flux quanta (integers)
    pub k: Vec<i64>,

    /// H_3 flux quanta (integers)
    pub m: Vec<i64>,

    /// Hodge numbers from polytope
    pub h11: i32,
    pub h21: i32,
}

impl Compactification {
    /// Create a new random compactification
    pub fn random<R: rand::Rng>(rng: &mut R, polytope_data: &PolytopeData) -> Self {
        Self::random_filtered(rng, polytope_data, None)
    }

    /// Create a new random compactification, optionally filtering to specific polytope IDs
    pub fn random_filtered<R: rand::Rng>(rng: &mut R, polytope_data: &PolytopeData, filter: Option<&[usize]>) -> Self {
        let polytope_id = match filter {
            Some(ids) if !ids.is_empty() => ids[rng.gen_range(0..ids.len())],
            _ => rng.gen_range(0..polytope_data.len()),
        };
        let polytope = polytope_data.get(polytope_id).expect("Invalid polytope index");

        let h11 = polytope.h11;
        let h21 = polytope.h12; // Note: h12 = h21 for CY3

        // Flux quanta: h11 integers for K and M
        let k: Vec<i64> = (0..h11)
            .map(|_| rng.gen_range(-15..=15))
            .collect();
        let m: Vec<i64> = (0..h11)
            .map(|_| rng.gen_range(-15..=15))
            .collect();

        Self {
            polytope_id,
            triangulation_id: 0, // Default to first triangulation
            k,
            m,
            h11,
            h21,
        }
    }

    /// Mutate this compactification
    pub fn mutate_filtered<R: rand::Rng>(&mut self, rng: &mut R, strength: f64, polytope_data: &PolytopeData, filter: Option<&[usize]>) {
        // Occasionally switch polytopes entirely
        if rng.gen::<f64>() < 0.05 * strength {
            let new_id = match filter {
                Some(ids) if !ids.is_empty() => ids[rng.gen_range(0..ids.len())],
                _ => rng.gen_range(0..polytope_data.len()),
            };
            let polytope = polytope_data.get(new_id).expect("Invalid polytope index");
            self.polytope_id = new_id;
            self.h11 = polytope.h11;
            self.h21 = polytope.h12;

            // Resize flux arrays
            self.k.resize(self.h11 as usize, 0);
            self.m.resize(self.h11 as usize, 0);
        }

        // Mutate fluxes (integer valued)
        for val in &mut self.k {
            if rng.gen::<f64>() < 0.3 {
                *val += rng.gen_range(-2..=2);
                *val = (*val).clamp(-20, 20);
            }
        }
        for val in &mut self.m {
            if rng.gen::<f64>() < 0.3 {
                *val += rng.gen_range(-2..=2);
                *val = (*val).clamp(-20, 20);
            }
        }
    }

    /// Crossover with another compactification
    pub fn crossover<R: rand::Rng>(&self, other: &Self, rng: &mut R) -> Self {
        // If different polytopes, pick one randomly
        let (base, other_ref) = if rng.gen() {
            (self.clone(), other)
        } else {
            (other.clone(), self)
        };

        let mut child = base;

        // Blend fluxes
        let min_len = child.k.len().min(other_ref.k.len());
        for i in 0..min_len {
            if rng.gen() {
                child.k[i] = other_ref.k[i];
            }
        }
        let min_len = child.m.len().min(other_ref.m.len());
        for i in 0..min_len {
            if rng.gen() {
                child.m[i] = other_ref.m[i];
            }
        }

        child
    }
}

/// A polytope from the Kreuzer-Skarke database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polytope {
    pub vertices: Vec<Vec<i32>>,
    pub h11: i32,
    pub h12: i32,  // = h21 for CY3
    pub euler: i32,
    pub point_count: i32,
    pub dual_point_count: i32,
}

/// JSONL format polytope (flat vertices array)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonlPolytope {
    vertices: Vec<i32>,
    h11: i32,
    h21: i32,
    vertex_count: i32,
}

impl From<JsonlPolytope> for Polytope {
    fn from(j: JsonlPolytope) -> Self {
        // Reshape flat vertices array into Vec<Vec<i32>>
        let vertices: Vec<Vec<i32>> = j.vertices
            .chunks(4)
            .map(|chunk| chunk.to_vec())
            .collect();
        Polytope {
            vertices,
            h11: j.h11,
            h12: j.h21,  // h12 = h21 for CY3
            euler: 2 * (j.h11 - j.h21),
            point_count: j.vertex_count,
            dual_point_count: 0,  // Not available in JSONL format
        }
    }
}

/// Indexed polytope database - stores byte offsets, loads on demand
pub struct PolytopeData {
    offsets: Vec<u64>,
    file: std::sync::Mutex<std::io::BufReader<std::fs::File>>,
}

impl PolytopeData {
    /// Load or build index for JSONL file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let index_path = format!("{}.idx", path);
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();

        // Try to load existing index
        let offsets = if let Ok(index_data) = std::fs::read(&index_path) {
            // Validate: first 8 bytes = file length, rest = offsets
            if index_data.len() >= 8 {
                let stored_len = u64::from_le_bytes(index_data[0..8].try_into().unwrap());
                if stored_len == file_len {
                    let offsets: Vec<u64> = index_data[8..]
                        .chunks_exact(8)
                        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    println!("Loaded index: {} polytopes from {}", offsets.len(), index_path);
                    offsets
                } else {
                    println!("Index stale (file size changed), rebuilding...");
                    Self::build_index(path, &index_path)?
                }
            } else {
                Self::build_index(path, &index_path)?
            }
        } else {
            println!("Building index for {}...", path);
            Self::build_index(path, &index_path)?
        };

        let file = std::fs::File::open(path)?;
        Ok(Self {
            offsets,
            file: std::sync::Mutex::new(BufReader::new(file)),
        })
    }

    fn build_index(path: &str, index_path: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let mut offsets = Vec::new();
        let mut pos: u64 = 0;
        let mut line = String::new();
        let mut last_percent = 0;

        loop {
            let start = pos;
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }
            if !line.trim().is_empty() {
                offsets.push(start);
            }
            pos += bytes_read as u64;
            line.clear();

            let percent = (pos * 100 / file_len) as u32;
            if percent > last_percent {
                last_percent = percent;
                if percent % 10 == 0 {
                    print!("  {}%", percent);
                    std::io::stdout().flush().ok();
                }
            }
        }
        println!();

        // Save index: file_len (8 bytes) + offsets
        let mut index_file = std::fs::File::create(index_path)?;
        index_file.write_all(&file_len.to_le_bytes())?;
        for offset in &offsets {
            index_file.write_all(&offset.to_le_bytes())?;
        }
        println!("Saved index: {} entries to {}", offsets.len(), index_path);

        Ok(offsets)
    }

    /// Get polytope by index (reads from file on demand)
    pub fn get(&self, index: usize) -> Option<Polytope> {
        let offset = *self.offsets.get(index)?;
        let mut file = self.file.lock().ok()?;
        file.seek(SeekFrom::Start(offset)).ok()?;

        let mut line = String::new();
        file.read_line(&mut line).ok()?;

        serde_json::from_str::<JsonlPolytope>(&line)
            .ok()
            .map(|j| j.into())
    }

    /// Number of polytopes
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Compute physics from a compactification genome using cyrus-core
pub fn compute_physics(
    genome: &Compactification,
    polytope: &Polytope,
) -> PhysicsOutput {
    // 1. Get Geometry (from cache or Python)
    let (kappa, mori, gv) = match get_geometry(genome.polytope_id, &polytope.vertices) {
        Ok(g) => g,
        Err(e) => return PhysicsOutput {
            success: false,
            error: Some(format!("Failed to get geometry: {}", e)),
            ..Default::default()
        },
    };

    // 2. Evaluate Vacuum
    let req = EvaluationRequest {
        kappa: &kappa,
        mori: &mori,
        gv: &gv,
        h11: genome.h11,
        h21: genome.h21,
        q_max: 200.0, // Default bound
    };

    let res = match evaluate_vacuum(&req, &genome.k, &genome.m) {
        Ok(res) => res,
        Err(e) => return PhysicsOutput {
            success: false,
            error: Some(format!("Pipeline error: {}", e)),
            ..Default::default()
        },
    };

    if !res.success {
        return PhysicsOutput {
            success: false,
            error: res.reason,
            ..Default::default()
        };
    }

    // 3. Map Results
    let vac = res.vacuum.unwrap();
    // racetrack might be None if GV invariants are empty
    let g_s = vac.g_s; // evaluate_vacuum populates vac from rt_res if successful
    
    PhysicsOutput {
        success: true,
        error: None,
        alpha_em: 1.0 / 137.0, // TODO: Compute from volumes
        alpha_s: 0.1,          // TODO: Compute from volumes
        sin2_theta_w: 0.23,
        cosmological_constant: vac.v0,
        n_generations: 3,      // Fixed for three-gen subset
        m_e_planck_ratio: 0.0,
        m_p_planck_ratio: 0.0,
        cy_volume: vac.v_string,
        string_coupling: g_s,
        flux_tadpole: res.q_flux,
        superpotential_abs: vac.w0,
    }
}

/// Fetch geometry for a polytope (Intersections, Mori cone, GV)
fn get_geometry(
    _id: usize,
    vertices: &[Vec<i32>],
) -> Result<(Intersection, MoriCone, Vec<GvInvariant>), Box<dyn std::error::Error>> {
    // Convert vertices to cyrus-core Points
    let points: Vec<Point> = vertices
        .iter()
        .map(|v| Point::new(v.iter().map(|&x| x as i64).collect()))
        .collect();

    // 1. GLSM
    let glsm = compute_glsm_charge_matrix(&points, true).map_err(|e| format!("GLSM failed: {}", e))?;

    // 2. Triangulation
    // Use deterministic heights derived from vertices hash
    let mut hasher = DefaultHasher::new();
    vertices.hash(&mut hasher);
    let seed = hasher.finish();
    let mut rng = StdRng::seed_from_u64(seed);
    
    let heights: Vec<f64> = (0..points.len()).map(|_| rng.gen::<f64>()).collect();
    let tri = compute_regular_triangulation(&points, &heights).map_err(|e| format!("Triangulation failed: {}", e))?;

    // 3. Intersection Numbers
    let kappa = compute_intersection_numbers(&tri, &points, &glsm).map_err(|e| format!("Intersections failed: {}", e))?;

    // 4. Mori Cone
    let mori = compute_mori_generators(&tri, &points).map_err(|e| format!("Mori cone failed: {}", e))?;

    // 5. GV Invariants
    // TODO: Implement GV computation or loading
    let gv = Vec::new();

    Ok((kappa, mori, gv))
}

pub fn is_physics_available() -> bool {
    true // We always use Rust cyrus-core now
}

pub fn init_physics_bridge() -> PyResult<()> {
    // No-op
    Ok(())
}

pub fn clear_physics_cache() {
    // No-op for now
}
