// Types mirroring Rust structs from src/physics.rs

/** Physical observables computed from a compactification */
export interface PhysicsOutput {
  success: boolean;
  error: string | null;

  // Gauge couplings
  alpha_em: number;
  alpha_s: number;
  sin2_theta_w: number;

  // Cosmological
  cosmological_constant: number;

  // Particle physics
  n_generations: number;
  m_e_planck_ratio: number;
  m_p_planck_ratio: number;

  // Internal geometry
  cy_volume: number;
  string_coupling: number;
  flux_tadpole: number;
  superpotential_abs: number;
}

/** Genome for a real string compactification */
export interface Compactification {
  /** Index into the polytope database */
  polytope_id: number;

  /** Kahler moduli (control cycle volumes) */
  kahler_moduli: number[];

  /** Complex structure moduli */
  complex_moduli: number[];

  /** F_3 flux quanta (integers) */
  flux_f: number[];

  /** H_3 flux quanta (integers) */
  flux_h: number[];

  /** String coupling g_s */
  g_s: number;

  /** Hodge number h11 */
  h11: number;

  /** Hodge number h21 */
  h21: number;
}

/** Combined genome result with physics and fitness */
export interface GenomeResult {
  genome: Compactification;
  physics: PhysicsOutput;
  fitness: number;
}

/** Polytope data from the JSONL file */
export interface PolytopeEntry {
  h11: number;
  h21: number;
  vertex_count: number;
  vertices: number[]; // Flat array of 4D coords, reshape to [vertex_count][4]
}

/** Target physics constants */
export const TARGET_PHYSICS = {
  alpha_em: 7.297e-3, // Fine structure constant
  alpha_s: 0.118, // Strong coupling at M_Z
  sin2_theta_w: 0.231, // Weinberg angle
  n_generations: 3, // Fermion generations
  cosmological_constant: 2.888e-122, // In Planck units
} as const;

/** Run metadata */
export interface RunInfo {
  id: string;
  path: string;
  genomeCount: number;
  bestFitness: number;
  timestamp: Date;
}

/** Generation statistics from GA evolution */
export interface GenerationStats {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  worst_fitness: number;
  diversity: number;
  physics_success_rate: number;
}

/** All computed heuristics for a polytope */
export interface PolytopeHeuristics {
  polytope_id: number;
  h11: number;
  h21: number;
  vertex_count: number;

  // Basic geometry
  centroid: number[];

  // Circularity (π-ness)
  sphericity: number;
  inertia_isotropy: number;
  inertia_eigenvalues: number[];

  // Spirality
  spiral_correlations: Record<string, number>;

  // Chirality
  chirality_optimal: number;
  chirality_x: number;
  chirality_y: number;
  chirality_z: number;
  chirality_w: number;
  handedness_det: number;

  // Symmetry
  symmetry_x: number;
  symmetry_y: number;
  symmetry_z: number;
  symmetry_w: number;

  // Flatness (PCA)
  pca_variance_ratios: number[];
  flatness_3d: number;
  flatness_2d: number;
  intrinsic_dim_estimate: number;

  // Regularity
  edge_length_cv: number;

  // Spikiness
  spikiness: number;
  max_exposure: number;

  // Concentration / Outliers
  conformity_ratio: number;
  distance_kurtosis: number;
  loner_score: number;

  // Basic stats
  coord_mean: number;
  coord_median: number;
  coord_mode: number;
  coord_std: number;
  coord_iqr: number;
  coord_range: number;
  coord_skewness: number;
  coord_kurtosis: number;
  mean_median_diff: number;

  // Per-axis stats
  axis_means: number[];
  axis_medians: number[];
  axis_stds: number[];
  axis_skewness: number[];
  axis_kurtosis: number[];

  // Entropy / Information
  shannon_entropy: number;
  axis_entropies: number[];
  joint_entropy: number;

  // Compressibility
  compression_ratio: number;
  sorted_compression_ratio: number;
  sort_compression_gain: number;

  // Axis balance
  axis_balance: number[];
  spread_ratio: number;

  // Quartiles
  axis_q1: number[];
  axis_q2: number[];
  axis_q3: number[];
  axis_iqr: number[];
  axis_quartile_skew: number[];

  // Distribution tests
  normality_pvalue: number;
  uniform_ks_stat: number;

  // Correlation
  correlations: Record<string, number>;
  mean_abs_correlation: number;
  corr_eigenvalues: number[];

  // Patterns
  phi_ratio_count: number;
  fibonacci_count: number;
  zero_count: number;
  one_count: number;
  neg_one_count: number;
  prime_count: number;
}
