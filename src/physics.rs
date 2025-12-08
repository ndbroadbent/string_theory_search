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
pub struct RealCompactification {
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

impl RealCompactification {
    /// Create a new random compactification
    pub fn random<R: rand::Rng>(rng: &mut R, polytope_data: &PolytopeData) -> Self {
        let polytope_id = rng.gen_range(0..polytope_data.polytopes.len());
        let polytope = &polytope_data.polytopes[polytope_id];

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
            let new_id = rng.gen_range(0..polytope_data.polytopes.len());
            let polytope = &polytope_data.polytopes[new_id];
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

/// Loaded polytope database
#[derive(Debug, Clone)]
pub struct PolytopeData {
    pub polytopes: Vec<Polytope>,
}

impl PolytopeData {
    /// Load polytopes from JSON file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let polytopes: Vec<Polytope> = serde_json::from_str(&data)?;
        println!("Loaded {} polytopes from {}", polytopes.len(), path);
        Ok(Self { polytopes })
    }

    /// Create minimal test data if file doesn't exist
    pub fn load_or_default(path: &str) -> Self {
        match Self::load(path) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Warning: Could not load polytope data from {}: {}", path, e);
                eprintln!("Using minimal built-in polytopes");

                // Minimal built-in polytopes for testing
                Self {
                    polytopes: vec![
                        // Quintic threefold
                        Polytope {
                            vertices: vec![
                                vec![1, 0, 0, 0],
                                vec![0, 1, 0, 0],
                                vec![0, 0, 1, 0],
                                vec![0, 0, 0, 1],
                                vec![-1, -1, -1, -1],
                            ],
                            h11: 1,
                            h12: 101,
                            euler: -200,
                            point_count: 6,
                            dual_point_count: 126,
                        },
                        // A simpler CY
                        Polytope {
                            vertices: vec![
                                vec![1, 0, 0, 0],
                                vec![0, 1, 0, 0],
                                vec![0, 0, 1, 0],
                                vec![1, 2, 3, 5],
                                vec![-2, -3, -4, -5],
                            ],
                            h11: 21,
                            h12: 1,
                            euler: 40,
                            point_count: 6,
                            dual_point_count: 26,
                        },
                    ],
                }
            }
        }
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

        // Also add the venv site-packages
        let venv_path = bridge_dir.join(".venv/lib/python3.14/site-packages");
        if venv_path.exists() {
            path.insert(0, venv_path.to_string_lossy().to_string())?;
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
pub fn compute_physics(genome: &RealCompactification) -> PhysicsOutput {
    let result = Python::with_gil(|py| -> PyResult<PhysicsOutput> {
        let bridge = PHYSICS_BRIDGE
            .get()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Physics bridge not initialized"))?;

        let bridge = bridge.bind(py);

        // Build genome dict for Python
        let genome_dict = PyDict::new(py);
        genome_dict.set_item("polytope_id", genome.polytope_id)?;
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

    result.unwrap_or_else(|e| PhysicsOutput {
        success: false,
        error: Some(format!("Python error: {}", e)),
        ..Default::default()
    })
}

/// Check if physics bridge is available
pub fn is_physics_available() -> bool {
    PHYSICS_BRIDGE.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polytope_loading() {
        let data = PolytopeData::load_or_default("polytopes_small.json");
        assert!(!data.polytopes.is_empty());
        println!("Loaded {} polytopes", data.polytopes.len());
    }

    #[test]
    fn test_random_compactification() {
        let data = PolytopeData::load_or_default("polytopes_small.json");
        let mut rng = rand::thread_rng();
        let comp = RealCompactification::random(&mut rng, &data);

        println!("Random compactification:");
        println!("  Polytope ID: {}", comp.polytope_id);
        println!("  h11 = {}, h21 = {}", comp.h11, comp.h21);
        println!("  Kähler moduli: {:?}", comp.kahler_moduli);
        println!("  g_s = {}", comp.g_s);
    }
}
