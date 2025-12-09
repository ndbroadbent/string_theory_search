//! Real physics computations via Python/JAX bridge
//!
//! This module uses PyO3 to call into our Python physics_bridge.py,
//! which uses cymyc (JAX) and PALP for actual Calabi-Yau computations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Cached Python physics bridge instance
static PHYSICS_BRIDGE: OnceLock<Py<PyAny>> = OnceLock::new();

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

/// Genome for a real string compactification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compactification {
    /// Index into the polytope database
    pub polytope_id: usize,

    /// Kähler moduli (control cycle volumes)
    pub kahler_moduli: Vec<f64>,

    /// Complex structure moduli
    pub complex_moduli: Vec<f64>,

    /// F_3 flux quanta (integers)
    pub flux_f: Vec<i32>,

    /// H_3 flux quanta (integers)
    pub flux_h: Vec<i32>,

    /// String coupling g_s
    pub g_s: f64,

    /// Hodge numbers from polytope
    pub h11: i32,
    pub h21: i32,
}

impl Compactification {
    /// Create a new random compactification
    pub fn random<R: rand::Rng>(rng: &mut R, polytope_data: &PolytopeData) -> Self {
        let polytope_id = rng.gen_range(0..polytope_data.len());
        let polytope = polytope_data.get(polytope_id).expect("Invalid polytope index");

        let h11 = polytope.h11;
        let h21 = polytope.h12; // Note: h12 = h21 for CY3

        // Kähler moduli: h11 real positive parameters
        let kahler_moduli: Vec<f64> = (0..h11)
            .map(|_| rng.gen_range(0.5..5.0))
            .collect();

        // Complex structure moduli: h21 complex parameters (store as 2*h21 reals)
        let complex_moduli: Vec<f64> = (0..h21)
            .map(|_| rng.gen_range(0.1..2.0))
            .collect();

        // Flux quanta: 2*(h21+1) integers for F_3 and H_3
        let n_flux = 2 * (h21 + 1) as usize;
        let flux_f: Vec<i32> = (0..n_flux)
            .map(|_| rng.gen_range(-5..=5))
            .collect();
        let flux_h: Vec<i32> = (0..n_flux)
            .map(|_| rng.gen_range(-5..=5))
            .collect();

        // String coupling
        let g_s = rng.gen_range(0.01..0.5);

        Self {
            polytope_id,
            kahler_moduli,
            complex_moduli,
            flux_f,
            flux_h,
            g_s,
            h11,
            h21,
        }
    }

    /// Mutate this compactification
    pub fn mutate<R: rand::Rng>(&mut self, rng: &mut R, strength: f64, polytope_data: &PolytopeData) {
        // Occasionally switch polytopes entirely
        if rng.gen::<f64>() < 0.05 * strength {
            let new_id = rng.gen_range(0..polytope_data.len());
            let polytope = polytope_data.get(new_id).expect("Invalid polytope index");
            self.polytope_id = new_id;
            self.h11 = polytope.h11;
            self.h21 = polytope.h12;

            // Resize moduli arrays
            self.kahler_moduli.resize(self.h11 as usize, 1.0);
            self.complex_moduli.resize(self.h21 as usize, 1.0);

            let n_flux = 2 * (self.h21 + 1) as usize;
            self.flux_f.resize(n_flux, 0);
            self.flux_h.resize(n_flux, 0);
        }

        // Mutate Kähler moduli
        for t in &mut self.kahler_moduli {
            if rng.gen::<f64>() < 0.3 {
                *t *= 1.0 + rng.gen_range(-strength..strength);
                *t = t.clamp(0.1, 10.0);
            }
        }

        // Mutate complex structure moduli
        for z in &mut self.complex_moduli {
            if rng.gen::<f64>() < 0.3 {
                *z *= 1.0 + rng.gen_range(-strength..strength);
                *z = z.clamp(0.01, 5.0);
            }
        }

        // Mutate fluxes (integer valued)
        for f in &mut self.flux_f {
            if rng.gen::<f64>() < 0.2 {
                *f += rng.gen_range(-2..=2);
                *f = (*f).clamp(-10, 10);
            }
        }
        for h in &mut self.flux_h {
            if rng.gen::<f64>() < 0.2 {
                *h += rng.gen_range(-2..=2);
                *h = (*h).clamp(-10, 10);
            }
        }

        // Mutate string coupling
        if rng.gen::<f64>() < 0.3 {
            self.g_s *= 1.0 + rng.gen_range(-strength..strength);
            self.g_s = self.g_s.clamp(0.001, 1.0);
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

        // Blend Kähler moduli
        let min_len = child.kahler_moduli.len().min(other_ref.kahler_moduli.len());
        for i in 0..min_len {
            if rng.gen() {
                child.kahler_moduli[i] = other_ref.kahler_moduli[i];
            }
        }

        // Blend complex moduli
        let min_len = child.complex_moduli.len().min(other_ref.complex_moduli.len());
        for i in 0..min_len {
            if rng.gen() {
                child.complex_moduli[i] = other_ref.complex_moduli[i];
            }
        }

        // Blend fluxes
        let min_len = child.flux_f.len().min(other_ref.flux_f.len());
        for i in 0..min_len {
            if rng.gen() {
                child.flux_f[i] = other_ref.flux_f[i];
            }
        }
        let min_len = child.flux_h.len().min(other_ref.flux_h.len());
        for i in 0..min_len {
            if rng.gen() {
                child.flux_h[i] = other_ref.flux_h[i];
            }
        }

        // Blend string coupling
        if rng.gen() {
            child.g_s = other_ref.g_s;
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

/// Indexed polytope database - stores byte offsets, loads on demand
pub struct PolytopeData {
    offsets: Vec<u64>,
    file: std::sync::Mutex<std::io::BufReader<std::fs::File>>,
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

impl PolytopeData {
    /// Load or build index for JSONL file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::io::BufReader;

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
        use std::io::{BufRead, BufReader, Write};

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
        use std::io::{BufRead, Seek, SeekFrom};

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

/// Initialize the Python physics bridge
pub fn init_physics_bridge() -> PyResult<()> {
    Python::with_gil(|py| {
        // Add our source directory to Python path
        let sys = py.import("sys")?;
        let path: Bound<'_, PyList> = sys.getattr("path")?.downcast_into()?;

        let bridge_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."));
        path.insert(0, bridge_dir.to_string_lossy().to_string())?;

        // Add venv site-packages - try multiple Python versions
        for py_ver in ["python3.11", "python3.12", "python3.10"] {
            let venv_path = bridge_dir.join(format!(".venv/lib/{}/site-packages", py_ver));
            if venv_path.exists() {
                path.insert(0, venv_path.to_string_lossy().to_string())?;
                break;
            }
            // Also check VIRTUAL_ENV env var
            if let Ok(venv_dir) = std::env::var("VIRTUAL_ENV") {
                let venv_path = PathBuf::from(&venv_dir).join(format!("lib/{}/site-packages", py_ver));
                if venv_path.exists() {
                    path.insert(0, venv_path.to_string_lossy().to_string())?;
                    break;
                }
            }
        }

        // Import and instantiate the bridge
        let bridge_module = py.import("physics_bridge")?;
        let bridge_class = bridge_module.getattr("PhysicsBridge")?;
        let bridge_instance = bridge_class.call0()?;

        // Store globally
        PHYSICS_BRIDGE.get_or_init(|| bridge_instance.unbind());

        println!("Python physics bridge initialized successfully");
        Ok(())
    })
}

/// Compute physics from a compactification genome
pub fn compute_physics(genome: &Compactification, vertices: &[Vec<i32>]) -> PhysicsOutput {
    let result = Python::with_gil(|py| -> PyResult<PhysicsOutput> {
        let bridge = PHYSICS_BRIDGE
            .get()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Physics bridge not initialized"))?;

        let bridge = bridge.bind(py);

        // Build genome dict for Python
        let genome_dict = PyDict::new(py);
        genome_dict.set_item("polytope_id", genome.polytope_id)?;
        genome_dict.set_item("vertices", vertices)?;
        genome_dict.set_item("kahler_moduli", &genome.kahler_moduli)?;
        genome_dict.set_item("complex_moduli", &genome.complex_moduli)?;
        genome_dict.set_item("flux_f", &genome.flux_f)?;
        genome_dict.set_item("flux_h", &genome.flux_h)?;
        genome_dict.set_item("g_s", genome.g_s)?;
        genome_dict.set_item("h11", genome.h11)?;
        genome_dict.set_item("h21", genome.h21)?;

        // Call compute_physics
        let result = bridge.call_method1("compute_physics", (genome_dict,))?;

        // Extract results
        let success: bool = result.get_item("success")?.extract()?;

        if success {
            Ok(PhysicsOutput {
                success: true,
                error: None,
                alpha_em: result.get_item("alpha_em")?.extract()?,
                alpha_s: result.get_item("alpha_s")?.extract()?,
                sin2_theta_w: result.get_item("sin2_theta_w")?.extract()?,
                cosmological_constant: result.get_item("cosmological_constant")?.extract()?,
                n_generations: result.get_item("n_generations")?.extract()?,
                m_e_planck_ratio: result.get_item("m_e_planck_ratio")?.extract()?,
                m_p_planck_ratio: result.get_item("m_p_planck_ratio")?.extract()?,
                cy_volume: result.get_item("cy_volume")?.extract()?,
                string_coupling: result.get_item("string_coupling")?.extract()?,
                flux_tadpole: result.get_item("flux_tadpole")?.extract()?,
                superpotential_abs: result.get_item("superpotential_abs")?.extract()?,
            })
        } else {
            let error: String = result.get_item("error")?.extract()?;
            Ok(PhysicsOutput {
                success: false,
                error: Some(error),
                ..Default::default()
            })
        }
    });

    result.unwrap_or_else(|e| {
        panic!("Python error - this is a bug, fix it!\n{}", e);
    })
}

/// Check if physics bridge is available
pub fn is_physics_available() -> bool {
    PHYSICS_BRIDGE.get().is_some()
}

// Tests require polytopes_three_gen.jsonl - run manually
