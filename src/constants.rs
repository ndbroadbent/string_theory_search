//! Physical constants of our universe that we're trying to match.
//! These are the "target" values our genetic algorithm is searching for.
//!
//! Sources:
//! - PDG (Particle Data Group) 2024
//! - CODATA 2018 fundamental constants

/// Physics model version for cache invalidation.
/// Bump this when physics_bridge.py or computation logic changes.
/// This is included in the evaluation input hash - changing it invalidates all cached results.
pub const PHYSICS_MODEL_VERSION: &str = "1.0.0";

/// Fine structure constant (electromagnetic coupling)
/// α = e²/(4πε₀ℏc) ≈ 1/137.035999084
pub const ALPHA_EM: f64 = 7.2973525693e-3;

/// Strong coupling constant at Z mass scale
/// α_s(M_Z) ≈ 0.1179 ± 0.0010
pub const ALPHA_STRONG: f64 = 0.1179;

/// Weak mixing angle (Weinberg angle)
/// sin²θ_W(M_Z) ≈ 0.23121 ± 0.00004 (MS-bar scheme)
pub const SIN2_THETA_W: f64 = 0.23121;

/// Ratio of electron mass to Planck mass
/// m_e = 0.51099895 MeV, M_Pl = 1.220890e19 GeV
/// m_e / M_Pl ≈ 4.185e-23
pub const ELECTRON_PLANCK_RATIO: f64 = 4.185e-23;

/// Ratio of proton mass to Planck mass
/// m_p = 938.27208816 MeV, M_Pl = 1.220890e19 GeV
/// m_p / M_Pl ≈ 7.685e-20
pub const PROTON_PLANCK_RATIO: f64 = 7.685e-20;

/// Cosmological constant in Planck units
/// Λ ≈ 1.1056e-52 m^-2
/// In Planck units (l_Pl = 1.616e-35 m): Λ_Pl = Λ * l_Pl² ≈ 2.888e-122
pub const COSMOLOGICAL_CONSTANT: f64 = 2.888e-122;

/// Number of generations of fermions
pub const NUM_GENERATIONS: u8 = 3;

/// Target constants as an array for fitness evaluation
pub const TARGETS: [f64; 6] = [
    ALPHA_EM,
    ALPHA_STRONG,
    SIN2_THETA_W,
    ELECTRON_PLANCK_RATIO,
    PROTON_PLANCK_RATIO,
    COSMOLOGICAL_CONSTANT,
];

/// Names for display
pub const TARGET_NAMES: [&str; 6] = [
    "α_em (fine structure)",
    "α_s (strong coupling)",
    "sin²θ_W (Weinberg angle)",
    "m_e/M_Pl (electron mass)",
    "m_p/M_Pl (proton mass)",
    "Λ (cosmological const)",
];

/// Weights for each constant in fitness calculation.
/// Higher weight = more important to match.
///
/// Rationale:
/// - Gauge couplings (α_em, α_s, sin²θ_W): directly from compactification geometry
/// - Mass ratios: involve Yukawa couplings, harder to predict precisely
/// - Cosmological constant: the famous fine-tuning problem, weight heavily
/// - Generation count: discrete, handle separately
pub const WEIGHTS: [f64; 6] = [
    2.0,  // α_em - well measured, directly from geometry
    2.0,  // α_s - well measured, directly from geometry
    1.5,  // sin²θ_W - depends on GUT breaking pattern
    1.0,  // m_e/M_Pl - involves Yukawa, more model-dependent
    1.0,  // m_p/M_Pl - involves QCD, somewhat indirect
    3.0,  // Λ - the holy grail, extremely hard to get right
];

/// Bonus/penalty for getting number of generations right
pub const GENERATION_BONUS: f64 = 0.2; // 20% bonus if N_gen = 3
pub const GENERATION_PENALTY: f64 = 0.5; // 50% penalty if N_gen ≠ 3
