/**
 * Server functions for polytope heuristics data
 *
 * Reads from SQLite database (data/string_theory.db)
 */

import { createServerFn } from '@tanstack/react-start';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { Database } from 'bun:sqlite';
import type { PolytopeHeuristics } from '../types';

function getProjectRoot(): string {
  const cwd = process.cwd();
  // If we're in /web, go up one level
  if (cwd.endsWith('/web') || cwd.endsWith('\\web')) {
    return dirname(cwd);
  }
  // If import.meta.dir is available (Bun), use it
  if (import.meta.dir) {
    return dirname(dirname(dirname(import.meta.dir)));
  }
  return cwd;
}

function getDbPath(): string {
  const projectRoot = getProjectRoot();
  return join(projectRoot, 'data', 'string_theory.db');
}

/**
 * Convert SQLite row to PolytopeHeuristics type
 */
function rowToHeuristics(row: Record<string, unknown>): PolytopeHeuristics {
  return {
    polytope_id: row.polytope_id as number,
    h11: (row.h11 as number) ?? 0,
    h21: (row.h21 as number) ?? 0,
    vertex_count: (row.vertex_count as number) ?? 0,
    sphericity: row.sphericity as number | undefined,
    inertia_isotropy: row.inertia_isotropy as number | undefined,
    chirality_optimal: row.chirality_optimal as number | undefined,
    chirality_x: row.chirality_x as number | undefined,
    chirality_y: row.chirality_y as number | undefined,
    chirality_z: row.chirality_z as number | undefined,
    chirality_w: row.chirality_w as number | undefined,
    handedness_det: row.handedness_det as number | undefined,
    symmetry_x: row.symmetry_x as number | undefined,
    symmetry_y: row.symmetry_y as number | undefined,
    symmetry_z: row.symmetry_z as number | undefined,
    symmetry_w: row.symmetry_w as number | undefined,
    flatness_3d: row.flatness_3d as number | undefined,
    flatness_2d: row.flatness_2d as number | undefined,
    intrinsic_dim_estimate: row.intrinsic_dim_estimate as number | undefined,
    spikiness: row.spikiness as number | undefined,
    max_exposure: row.max_exposure as number | undefined,
    conformity_ratio: row.conformity_ratio as number | undefined,
    distance_kurtosis: row.distance_kurtosis as number | undefined,
    loner_score: row.loner_score as number | undefined,
    coord_mean: row.coord_mean as number | undefined,
    coord_median: row.coord_median as number | undefined,
    coord_std: row.coord_std as number | undefined,
    coord_skewness: row.coord_skewness as number | undefined,
    coord_kurtosis: row.coord_kurtosis as number | undefined,
    shannon_entropy: row.shannon_entropy as number | undefined,
    joint_entropy: row.joint_entropy as number | undefined,
    compression_ratio: row.compression_ratio as number | undefined,
    sorted_compression_ratio: row.sorted_compression_ratio as number | undefined,
    sort_compression_gain: row.sort_compression_gain as number | undefined,
    phi_ratio_count: row.phi_ratio_count as number | undefined,
    fibonacci_count: row.fibonacci_count as number | undefined,
    zero_count: row.zero_count as number | undefined,
    one_count: row.one_count as number | undefined,
    prime_count: row.prime_count as number | undefined,
    outlier_score: row.outlier_score as number | undefined,
    outlier_max_zscore: row.outlier_max_zscore as number | undefined,
    outlier_max_dim: row.outlier_max_dim as string | undefined,
    outlier_count_2sigma: row.outlier_count_2sigma as number | undefined,
    outlier_count_3sigma: row.outlier_count_3sigma as number | undefined,
  };
}

/**
 * Get heuristics from SQLite
 */
async function loadHeuristics(): Promise<PolytopeHeuristics[]> {
  const dbPath = getDbPath();

  if (!existsSync(dbPath)) {
    console.error('Database not found:', dbPath);
    return [];
  }

  try {
    const db = new Database(dbPath, { readonly: true });

    // Read directly from heuristics table (h11, h21, vertex_count are stored there)
    const rows = db.prepare(`
      SELECT * FROM heuristics
      ORDER BY outlier_score DESC
    `).all() as Record<string, unknown>[];

    db.close();

    return rows.map(rowToHeuristics);
  } catch (error) {
    console.error('Error loading from SQLite:', error);
    return [];
  }
}

/**
 * Get all computed heuristics
 */
export const getHeuristics = createServerFn({ method: 'GET' }).handler(
  async (): Promise<PolytopeHeuristics[]> => {
    return loadHeuristics();
  }
);

/**
 * Get heuristics for a specific polytope by ID
 */
export const getHeuristicsForPolytope = createServerFn({ method: 'GET' })
  .inputValidator((data: { polytopeId: number }) => data)
  .handler(async ({ data: { polytopeId } }): Promise<PolytopeHeuristics | null> => {
    const dbPath = getDbPath();

    if (!existsSync(dbPath)) {
      return null;
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const row = db.prepare(`
        SELECT * FROM heuristics WHERE polytope_id = ?
      `).get(polytopeId) as Record<string, unknown> | null;

      db.close();

      if (row) {
        return rowToHeuristics(row);
      }
      return null;
    } catch (error) {
      console.error('Error loading from SQLite:', error);
      return null;
    }
  });

/**
 * Get polytope fitness statistics
 */
export const getPolytopeFitnessStats = createServerFn({ method: 'GET' })
  .inputValidator((data: { polytopeId: number }) => data)
  .handler(async ({ data: { polytopeId } }): Promise<{
    eval_count: number;
    fitness_mean: number;
    fitness_min: number;
    fitness_max: number;
    fitness_variance: number;
  } | null> => {
    const dbPath = getDbPath();

    if (!existsSync(dbPath)) {
      return null;
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const row = db.prepare(`
        SELECT
          eval_count,
          fitness_sum,
          fitness_sum_sq,
          fitness_min,
          fitness_max
        FROM polytopes
        WHERE id = ? AND eval_count > 0
      `).get(polytopeId) as {
        eval_count: number;
        fitness_sum: number;
        fitness_sum_sq: number;
        fitness_min: number;
        fitness_max: number;
      } | null;

      db.close();

      if (!row || row.eval_count === 0) {
        return null;
      }

      const mean = row.fitness_sum / row.eval_count;
      const variance = Math.max(0, (row.fitness_sum_sq / row.eval_count) - (mean * mean));

      return {
        eval_count: row.eval_count,
        fitness_mean: mean,
        fitness_min: row.fitness_min,
        fitness_max: row.fitness_max,
        fitness_variance: variance,
      };
    } catch (error) {
      console.error('Error loading fitness stats:', error);
      return null;
    }
  });

/**
 * Get top polytopes by mean fitness
 */
export const getTopPolytopes = createServerFn({ method: 'GET' })
  .inputValidator((data: { limit?: number; minEvals?: number }) => data)
  .handler(async ({ data: { limit = 100, minEvals = 5 } }): Promise<Array<{
    polytope_id: number;
    h11: number;
    h21: number;
    eval_count: number;
    fitness_mean: number;
    fitness_min: number;
    fitness_max: number;
  }>> => {
    const dbPath = getDbPath();

    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const rows = db.prepare(`
        SELECT
          id as polytope_id,
          h11,
          h21,
          eval_count,
          fitness_sum / eval_count as fitness_mean,
          fitness_min,
          fitness_max
        FROM polytopes
        WHERE eval_count >= ?
        ORDER BY (fitness_sum / eval_count) DESC
        LIMIT ?
      `).all(minEvals, limit) as Array<{
        polytope_id: number;
        h11: number;
        h21: number;
        eval_count: number;
        fitness_mean: number;
        fitness_min: number;
        fitness_max: number;
      }>;

      db.close();
      return rows;
    } catch (error) {
      console.error('Error loading top polytopes:', error);
      return [];
    }
  });

/**
 * Get run statistics
 */
export const getRunStats = createServerFn({ method: 'GET' }).handler(
  async (): Promise<Array<{
    run_id: string;
    started_at: string | null;
    ended_at: string | null;
    total_generations: number | null;
    total_evaluations: number | null;
    best_fitness: number | null;
    best_polytope_id: number | null;
  }>> => {
    const dbPath = getDbPath();

    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const rows = db.prepare(`
        SELECT
          id as run_id,
          started_at,
          ended_at,
          total_generations,
          total_evaluations,
          best_fitness,
          best_polytope_id
        FROM runs
        ORDER BY started_at DESC
      `).all() as Array<{
        run_id: string;
        started_at: string | null;
        ended_at: string | null;
        total_generations: number | null;
        total_evaluations: number | null;
        best_fitness: number | null;
        best_polytope_id: number | null;
      }>;

      db.close();
      return rows;
    } catch (error) {
      console.error('Error loading run stats:', error);
      return [];
    }
  }
);
