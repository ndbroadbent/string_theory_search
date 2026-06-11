/**
 * Server functions for polytope heuristics + outcome correlations
 *
 * The heuristics table is filled by the Rust binary:
 *   cargo run --release --bin heuristics -- --pool <pool.jsonl> --db <path>
 */

import { createServerFn } from '@tanstack/react-start';
import { withDb } from './db';
import type {
  PolytopeHeuristics,
  CorrelationRow,
  CorrelationPoint,
} from '../types';
import { TARGETS } from '../types';

/** Non-metric columns in the heuristics table */
const NON_METRIC_COLUMNS = new Set(['polytope_id', 'outlier_max_dim', 'updated_at']);

function rowToHeuristics(row: Record<string, unknown>): PolytopeHeuristics {
  const result: Record<string, string | number | null> = {};
  for (const [key, value] of Object.entries(row)) {
    if (key === 'updated_at') continue;
    result[key] =
      typeof value === 'number' || typeof value === 'string' ? value : null;
  }
  return result as unknown as PolytopeHeuristics;
}

/** Get all computed heuristics */
export const getHeuristics = createServerFn({ method: 'GET' }).handler(
  async (): Promise<PolytopeHeuristics[]> => {
    return withDb((db) => {
      const rows = db
        .prepare('SELECT * FROM heuristics ORDER BY polytope_id LIMIT 2000')
        .all() as Record<string, unknown>[];
      return rows.map(rowToHeuristics);
    }, []);
  }
);

/** Get heuristics for a specific polytope by pool name */
export const getHeuristicsForPolytope = createServerFn({ method: 'GET' })
  .inputValidator((data: { polytopeId: string }) => data)
  .handler(
    async ({ data: { polytopeId } }): Promise<PolytopeHeuristics | null> => {
      return withDb((db) => {
        const row = db
          .prepare('SELECT * FROM heuristics WHERE polytope_id = ?1')
          .get(polytopeId) as Record<string, unknown> | null;
        return row ? rowToHeuristics(row) : null;
      }, null);
    }
  );

/** Pearson correlation between two paired series (nulls excluded pairwise) */
function pearson(
  pairs: Array<[number | null, number | null]>
): { r: number | null; n: number } {
  const valid = pairs.filter(
    (p): p is [number, number] =>
      p[0] != null && p[1] != null && Number.isFinite(p[0]) && Number.isFinite(p[1])
  );
  const n = valid.length;
  if (n < 3) return { r: null, n };
  const meanX = valid.reduce((s, p) => s + p[0], 0) / n;
  const meanY = valid.reduce((s, p) => s + p[1], 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (const [x, y] of valid) {
    const dx = x - meanX;
    const dy = y - meanY;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx === 0 || syy === 0) return { r: null, n };
  return { r: sxy / Math.sqrt(sxx * syy), n };
}

/**
 * Correlations of every heuristic metric against three search outcomes:
 *   - fertility: polytopes.valid_seen > 0 (0/1)
 *   - bestFitness: polytopes.best_fitness
 *   - v0Distance: best |log10_abs_v0 + 121.54| per polytope
 */
export const getCorrelations = createServerFn({ method: 'GET' }).handler(
  async (): Promise<{ correlations: CorrelationRow[]; points: CorrelationPoint[] }> => {
    return withDb(
      (db) => {
        const rows = db
          .prepare(
            `SELECT h.*,
               p.valid_seen,
               p.best_fitness,
               (SELECT MIN(ABS(c.log10_abs_v0 - (?1)))
                FROM candidates c
                WHERE c.polytope_id = h.polytope_id
                  AND c.log10_abs_v0 IS NOT NULL) AS v0_distance
             FROM heuristics h
             JOIN polytopes p ON p.id = h.polytope_id`
          )
          .all(TARGETS.log10AbsV0) as Record<string, unknown>[];

        if (rows.length === 0) {
          return { correlations: [], points: [] };
        }

        // Numeric metric columns from the heuristics table
        const outcomeCols = new Set(['valid_seen', 'best_fitness', 'v0_distance']);
        const metricKeys = Object.keys(rows[0]).filter(
          (key) => !NON_METRIC_COLUMNS.has(key) && !outcomeCols.has(key)
        );

        const points: CorrelationPoint[] = rows.map((row) => {
          const metrics: Record<string, number | null> = {};
          for (const key of metricKeys) {
            const v = row[key];
            metrics[key] = typeof v === 'number' ? v : null;
          }
          const validSeen = row.valid_seen as number | null;
          return {
            polytope_id: row.polytope_id as string,
            metrics,
            outcomes: {
              fertility: validSeen == null ? null : validSeen > 0 ? 1 : 0,
              bestFitness: row.best_fitness as number | null,
              v0Distance: row.v0_distance as number | null,
            },
          };
        });

        const correlations: CorrelationRow[] = metricKeys.map((metric) => {
          const fertility = pearson(
            points.map((p) => [p.metrics[metric], p.outcomes.fertility])
          );
          const bestFitness = pearson(
            points.map((p) => [p.metrics[metric], p.outcomes.bestFitness])
          );
          const v0Distance = pearson(
            points.map((p) => [p.metrics[metric], p.outcomes.v0Distance])
          );
          return {
            metric,
            fertility: fertility.r,
            fertilityN: fertility.n,
            bestFitness: bestFitness.r,
            bestFitnessN: bestFitness.n,
            v0Distance: v0Distance.r,
            v0DistanceN: v0Distance.n,
          };
        });

        return { correlations, points };
      },
      { correlations: [], points: [] }
    );
  }
);
