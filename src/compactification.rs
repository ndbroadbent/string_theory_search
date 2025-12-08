//! Compactification parameter space.
//!
//! In string theory, the extra dimensions are "compactified" - curled up into
//! tiny geometric shapes. The parameters here represent choices in this process:
//!
//! - Calabi-Yau manifold topology (discrete choices)
//! - Moduli (continuous parameters describing the shape/size)
//! - Flux integers (discrete quantum numbers for field strengths)
//! - Brane configurations (where D-branes wrap cycles)
//!
//! Physics formulas based on:
//! - Gauge couplings: 1/g² ∝ Vol(cycle) in string units
//! - Yukawa couplings: overlap integrals of wavefunctions, depend on complex structure
//! - Cosmological constant: KKLT-style flux contributions with uplift

use rand::Rng;
use std::f64::consts::PI;

/// Number of compactified dimensions (10D string theory → 4D spacetime)
pub const COMPACT_DIMS: usize = 6;

/// Number of Kähler moduli (control 2-cycle volumes)
/// Typical Calabi-Yau has h^{1,1} ~ 2 to 500
pub const NUM_KAHLER: usize = 32;

/// Number of complex structure moduli (control 3-cycle shapes)
/// Typical Calabi-Yau has h^{2,1} ~ 0 to 500
pub const NUM_COMPLEX: usize = 32;

/// Number of flux integers (NS-NS and R-R through 3-cycles)
/// Each 3-cycle can carry both types of flux
pub const NUM_FLUXES: usize = 64;

/// Number of D-brane stacks (each can wrap different cycles)
pub const NUM_BRANE_STACKS: usize = 8;

/// Parameters per brane stack (wrapping numbers, position moduli)
pub const PARAMS_PER_BRANE: usize = 4;

/// Maximum value for flux integers (tadpole constraint limits these)
pub const MAX_FLUX: i32 = 50;

/// A point in the string theory landscape.
/// This represents one possible compactification of the extra dimensions.
#[derive(Clone, Debug)]
pub struct Compactification {
    /// Hodge numbers characterizing topology: (h^{1,1}, h^{2,1})
    /// These determine the number of moduli
    pub hodge_numbers: (u16, u16),

    /// Euler characteristic χ = 2(h^{1,1} - h^{2,1})
    /// |χ|/2 gives number of fermion generations in some constructions
    pub euler_char: i32,

    /// Kähler moduli t_i - control volumes of 2-cycles
    /// Gauge couplings: 1/g_a² = Re(f_a) where f_a depends on t_i
    pub kahler_moduli: [f64; NUM_KAHLER],

    /// Complex structure moduli z_α - control shape of Calabi-Yau
    /// Yukawa couplings depend on these through period integrals
    pub complex_moduli: [f64; NUM_COMPLEX],

    /// NS-NS flux integers through 3-cycles
    pub ns_fluxes: [i32; NUM_FLUXES / 2],

    /// R-R flux integers through 3-cycles
    pub rr_fluxes: [i32; NUM_FLUXES / 2],

    /// D-brane configurations
    /// Each stack: [wrapping_number, cycle_index, position_1, position_2]
    pub brane_stacks: [[f64; PARAMS_PER_BRANE]; NUM_BRANE_STACKS],

    /// String coupling g_s = e^φ where φ is dilaton
    /// Perturbative regime requires g_s < 1
    pub string_coupling: f64,

    /// Overall volume of Calabi-Yau in string units
    /// V = (1/6) κ_{ijk} t^i t^j t^k where κ is triple intersection
    pub cy_volume: f64,

    /// α' correction parameter (stringy corrections to geometry)
    pub alpha_prime_corrections: f64,
}

impl Compactification {
    /// Create a random compactification
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let h11: u16 = rng.gen_range(2..100);
        let h21: u16 = rng.gen_range(0..100);

        let mut kahler = [0.0; NUM_KAHLER];
        let mut complex = [0.0; NUM_COMPLEX];
        let mut ns_flux = [0; NUM_FLUXES / 2];
        let mut rr_flux = [0; NUM_FLUXES / 2];
        let mut branes = [[0.0; PARAMS_PER_BRANE]; NUM_BRANE_STACKS];

        // Kähler moduli must be positive (they're volumes)
        // Use log-uniform distribution to explore different scales
        for (i, k) in kahler.iter_mut().enumerate() {
            if i < h11 as usize {
                *k = (rng.gen_range(-1.0..3.0_f64)).exp(); // Range ~0.37 to ~20
            }
        }

        // Complex structure moduli - can be complex, we track magnitudes
        for (i, c) in complex.iter_mut().enumerate() {
            if i < h21 as usize {
                *c = rng.gen_range(-3.0..3.0);
            }
        }

        // Fluxes are integers, constrained by tadpole cancellation
        // Total flux contribution must satisfy: Σ F ∧ H ≤ L (tadpole bound)
        let tadpole_budget = rng.gen_range(10..100);
        let mut used = 0i32;
        for f in &mut ns_flux {
            if used.abs() < tadpole_budget {
                *f = rng.gen_range(-MAX_FLUX..=MAX_FLUX);
                used += f.abs();
            }
        }
        used = 0;
        for f in &mut rr_flux {
            if used.abs() < tadpole_budget {
                *f = rng.gen_range(-MAX_FLUX..=MAX_FLUX);
                used += f.abs();
            }
        }

        // D-brane stacks
        for stack in &mut branes {
            stack[0] = rng.gen_range(1.0_f64..5.0).floor(); // Wrapping number (integer)
            stack[1] = rng.gen_range(0.0..NUM_KAHLER as f64).floor(); // Which cycle
            stack[2] = rng.gen_range(0.0..1.0); // Position modulus 1
            stack[3] = rng.gen_range(0.0..1.0); // Position modulus 2
        }

        Self {
            hodge_numbers: (h11, h21),
            euler_char: 2 * (h11 as i32 - h21 as i32),
            kahler_moduli: kahler,
            complex_moduli: complex,
            ns_fluxes: ns_flux,
            rr_fluxes: rr_flux,
            brane_stacks: branes,
            string_coupling: rng.gen_range(0.01..0.5), // Perturbative regime
            cy_volume: (rng.gen_range(1.0..5.0_f64)).exp(), // Log-uniform
            alpha_prime_corrections: rng.gen_range(0.0..0.3),
        }
    }

    /// Compute effective 4D physical constants from this compactification.
    ///
    /// Based on Type IIB / F-theory compactification formulas:
    /// - Gauge couplings from D7-brane cycle volumes
    /// - Yukawa from wavefunction overlaps
    /// - Cosmological constant from KKLT-type flux stabilization
    pub fn compute_physics(&self) -> PhysicsOutput {
        let (h11, h21) = self.hodge_numbers;

        // === GAUGE COUPLINGS ===
        // In Type IIB, gauge coupling for D7-brane on divisor D:
        // 1/g² = τ_D / g_s where τ_D = Vol(D) in string units

        // U(1)_Y (hypercharge) - associated with first brane stack
        let tau_y = self.effective_cycle_volume(0);
        let g_y_squared = self.string_coupling / (tau_y + 0.1);

        // SU(2)_L - associated with second brane stack
        let tau_2 = self.effective_cycle_volume(1);
        let g_2_squared = self.string_coupling / (tau_2 + 0.1);

        // SU(3)_c - associated with third brane stack
        let tau_3 = self.effective_cycle_volume(2);
        let g_3_squared = self.string_coupling / (tau_3 + 0.1);

        // Fine structure constant: α = e²/4π where e comes from U(1)_em
        // In GUT normalization: 1/α = (5/3)/g_Y² + 1/g_2²
        // Simplified: use hypercharge contribution
        let alpha_em = g_y_squared * g_2_squared / (g_y_squared + g_2_squared) / (4.0 * PI);

        // Strong coupling α_s = g_3²/4π
        let alpha_strong = g_3_squared / (4.0 * PI);

        // Weinberg angle: sin²θ_W = g'²/(g² + g'²)
        let sin2_theta_w = g_y_squared / (g_y_squared + g_2_squared);

        // === YUKAWA COUPLINGS / FERMION MASSES ===
        // Yukawa ~ exp(-S_inst) where S_inst depends on complex structure
        // Physical Yukawa also depends on Kähler metric normalization

        // Electron Yukawa - exponentially sensitive to complex structure
        let cs_factor: f64 = self.complex_moduli[..h21.min(8) as usize]
            .iter()
            .enumerate()
            .map(|(i, z)| (-0.5 * (i as f64 + 1.0) * z.abs()).exp())
            .product();

        // Normalize by volume (wavefunction normalization)
        let y_e = cs_factor / self.cy_volume.sqrt();

        // Electron mass / Planck mass ~ y_e * v / M_Pl ~ y_e * 10^{-17}
        let electron_planck = y_e * 1e-17;

        // Up quark Yukawa - different complex structure dependence
        let y_u: f64 = self.complex_moduli[..h21.min(4) as usize]
            .iter()
            .map(|z| (-0.3 * z.abs()).exp())
            .product::<f64>()
            / self.cy_volume.sqrt();

        // Proton mass comes from QCD confinement scale Λ_QCD
        // Λ_QCD / M_Pl ~ exp(-8π²/(b₀ g_3²)) where b₀ = 7 for SM
        let b0 = 7.0;
        let lambda_qcd_ratio = (-8.0 * PI * PI / (b0 * g_3_squared)).exp();
        let proton_planck = 3.0 * lambda_qcd_ratio; // m_p ~ 3 Λ_QCD

        // === COSMOLOGICAL CONSTANT ===
        // KKLT scenario: Λ comes from balance of:
        // 1. Flux superpotential W_0 (from NS and RR fluxes)
        // 2. Non-perturbative corrections (instantons, gaugino condensation)
        // 3. Uplift from anti-D3 branes

        // Flux superpotential: W_0 ~ Σ (N_i + τ M_i) where τ = C_0 + i/g_s
        let w0_real: f64 = self.ns_fluxes.iter().map(|n| *n as f64).sum();
        let w0_imag: f64 = self.rr_fluxes.iter().map(|m| *m as f64 / self.string_coupling).sum();
        let w0_squared = w0_real * w0_real + w0_imag * w0_imag;

        // Gravitino mass: m_{3/2} ~ |W_0| e^{K/2} / M_Pl
        // Kähler potential K ~ -2 ln(V) for large volume
        let exp_k_half = 1.0 / self.cy_volume;
        let m32_squared = w0_squared * exp_k_half * exp_k_half;

        // AdS vacuum depth before uplift: V_AdS ~ -3 m_{3/2}²
        let v_ads = -3.0 * m32_squared;

        // Uplift from anti-D3 branes: δV ~ D / V^{4/3}
        // D is a discrete parameter (number of anti-branes)
        let n_anti_d3 = (self.brane_stacks[0][0].abs() as i32).max(1);
        let uplift = (n_anti_d3 as f64) * 0.01 / self.cy_volume.powf(4.0 / 3.0);

        // Total cosmological constant (in Planck units)
        // Must nearly cancel to get Λ ~ 10^{-122}
        let cc_raw = (v_ads + uplift) / self.cy_volume.powi(2);

        // Apply topology-dependent discrete shifts
        let topology_shift = (self.euler_char as f64 * 0.001).sin() * 1e-124;

        let cosmological_constant = (cc_raw + topology_shift).abs();

        // === NUMBER OF GENERATIONS ===
        // In many constructions: N_gen = |χ|/2
        let n_generations = (self.euler_char.abs() / 2) as u8;

        PhysicsOutput {
            alpha_em: alpha_em.clamp(1e-5, 1.0),
            alpha_strong: alpha_strong.clamp(1e-3, 10.0),
            sin2_theta_w: sin2_theta_w.clamp(0.0, 1.0),
            electron_planck_ratio: electron_planck.abs().clamp(1e-40, 1e-15),
            proton_planck_ratio: proton_planck.abs().clamp(1e-30, 1e-10),
            cosmological_constant: cosmological_constant.clamp(1e-150, 1e-50),
            n_generations,
        }
    }

    /// Compute effective cycle volume for brane stack i
    fn effective_cycle_volume(&self, stack_idx: usize) -> f64 {
        let stack = &self.brane_stacks[stack_idx];
        let wrapping = stack[0].max(1.0);
        let cycle_idx = (stack[1] as usize) % NUM_KAHLER;

        // Volume = wrapping number × Kähler modulus of that cycle
        // Plus corrections from α' and other moduli
        let base_vol = wrapping * self.kahler_moduli[cycle_idx].max(0.1);

        // Include intersection with other cycles (simplified)
        let correction: f64 = self.kahler_moduli
            .iter()
            .take(self.hodge_numbers.0 as usize)
            .map(|t| 1.0 + self.alpha_prime_corrections * t / self.cy_volume)
            .product();

        base_vol * correction
    }

    /// Encode as flat vector for visualization (normalized to [0,1])
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(150);

        // Topology
        v.push(self.hodge_numbers.0 as f32 / 100.0);
        v.push(self.hodge_numbers.1 as f32 / 100.0);

        // Kähler moduli (log scale)
        for k in &self.kahler_moduli {
            v.push(((k.max(0.01).ln() + 3.0) / 6.0).clamp(0.0, 1.0) as f32);
        }

        // Complex structure
        for c in &self.complex_moduli {
            v.push(((*c + 3.0) / 6.0).clamp(0.0, 1.0) as f32);
        }

        // Fluxes
        for f in &self.ns_fluxes {
            v.push((*f as f32 / MAX_FLUX as f32 * 0.5 + 0.5).clamp(0.0, 1.0));
        }
        for f in &self.rr_fluxes {
            v.push((*f as f32 / MAX_FLUX as f32 * 0.5 + 0.5).clamp(0.0, 1.0));
        }

        // Coupling and volume
        v.push((self.string_coupling * 2.0) as f32);
        v.push(((self.cy_volume.ln() + 2.0) / 7.0).clamp(0.0, 1.0) as f32);

        v
    }

    /// Total parameter count (dimensionality of search space)
    pub fn parameter_count() -> usize {
        2 + NUM_KAHLER + NUM_COMPLEX + NUM_FLUXES + NUM_BRANE_STACKS * PARAMS_PER_BRANE + 3
    }
}

/// Output physics from a compactification
#[derive(Clone, Debug)]
pub struct PhysicsOutput {
    pub alpha_em: f64,
    pub alpha_strong: f64,
    pub sin2_theta_w: f64,
    pub electron_planck_ratio: f64,
    pub proton_planck_ratio: f64,
    pub cosmological_constant: f64,
    pub n_generations: u8,
}

impl PhysicsOutput {
    /// Convert to array for fitness comparison (excluding discrete n_generations)
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.alpha_em,
            self.alpha_strong,
            self.sin2_theta_w,
            self.electron_planck_ratio,
            self.proton_planck_ratio,
            self.cosmological_constant,
        ]
    }
}
