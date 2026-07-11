// Types for the cyrus-ga string-landscape dashboard.
// Mirrors the SQLite schema in migrations/001_cyrus_ga_schema.sql,
// which is filled by web/scripts/ingest.ts from the cyrus-ga run dir.

/** Physics targets for the DESI dark-energy search */
export const TARGETS = {
  /** log10|V0| of the observed dark-energy scale (Planck units) */
  log10AbsV0: -121.54,
  /** DESI CPL measurement: w0 = -0.45 +/- 0.21 */
  desiW0: { value: -0.45, sigma: 0.21 },
  /** DESI CPL measurement: wa = -1.8 +/- 0.6 */
  desiWa: { value: -1.8, sigma: 0.6 },
  /** Cosmological constant (Lambda-CDM) point in the (w0, wa) plane */
  lambda: { w0: -1, wa: 0 },
} as const;

/** Aggregate run status for the dashboard status strip */
export interface RunStatus {
  totalRounds: number;
  totalEvals: number;
  validSeen: number;
  validCandidates: number;
  polytopesTracked: number;
  poolTotal: number | null;
  live: number | null;
  dead: number | null;
  globalBest: number | null;
  /** evals/sec over the last 10 minutes of round events */
  evalsPerSec: number | null;
  /** Unix ts of the last round event */
  lastRoundTs: number | null;
  /** Unix ts of the last ingester cycle */
  lastIngestTs: number | null;
  /** Unix ts of the round that set the current global best */
  lastImprovementTs: number | null;
  /** The polytope with the most rounds in the last 10 minutes, and its share
   * of that window's rounds (0..1). A share near 1 means the bandit has
   * collapsed onto a single arm. */
  topArm: { polytope: string; share: number } | null;
}

/** A candidate row (best-improvement events from candidates.jsonl) */
export interface Candidate {
  id: number;
  polytope_id: string;
  k: number[];
  m: number[];
  fitness: number | null;
  tier: string | null;
  q_flux: number | null;
  g_s: number | null;
  /** |W_0| flux superpotential (NOT the CPL dark-energy w0!) */
  w0: number | null;
  log10_abs_v0: number | null;
  cpl_w0: number | null;
  cpl_wa: number | null;
  round: number | null;
  ts: number | null;
}

/** Leaderboard entry: best candidate per polytope + polytope context */
export interface LeaderboardEntry extends Candidate {
  h11: number | null;
  h21: number | null;
  /** D3 tadpole bound (h11+h21)/2 + 1 */
  q_d3: number | null;
}

/** A polytope row from the polytopes table (id = pool name) */
export interface PolytopeRow {
  id: string;
  h11: number | null;
  h21: number | null;
  favorable: number | null;
  q_d3: number | null;
  rounds: number;
  evals: number;
  valid_seen: number;
  best_fitness: number | null;
  updated_at: number | null;
}

/** Polytope row including parsed 4D vertices */
export interface PolytopeDetail extends PolytopeRow {
  vertices: number[][] | null;
}

/** A verification record for a promising candidate */
export interface Verification {
  id: number;
  polytope_id: string;
  k: string | null;
  m: string | null;
  status: string | null;
  blocker: string | null;
  details: string | null;
  ts: number | null;
}

/** All computed heuristics for a polytope (numeric metric columns) */
export interface PolytopeHeuristics {
  polytope_id: string;
  h11: number | null;
  h21: number | null;
  vertex_count: number | null;
  [metric: string]: string | number | null;
}

/** Per-metric Pearson correlations against search outcomes */
export interface CorrelationRow {
  metric: string;
  /** vs polytopes.valid_seen > 0 (point-biserial) */
  fertility: number | null;
  fertilityN: number;
  /** vs polytopes.best_fitness */
  bestFitness: number | null;
  bestFitnessN: number;
  /** vs best |log10_abs_v0 + 121.54| per polytope */
  v0Distance: number | null;
  v0DistanceN: number;
}

/** One polytope's data point for correlation scatter plots */
export interface CorrelationPoint {
  polytope_id: string;
  metrics: Record<string, number | null>;
  outcomes: {
    fertility: number | null;
    bestFitness: number | null;
    v0Distance: number | null;
  };
}

/** Sigma distance to a DESI measurement */
export function sigmaDistance(
  value: number | null,
  target: { value: number; sigma: number }
): number | null {
  if (value == null) return null;
  return (value - target.value) / target.sigma;
}
