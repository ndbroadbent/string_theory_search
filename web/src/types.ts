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
