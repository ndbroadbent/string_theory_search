/**
 * Server functions for reading genome results from SQLite
 */

import { createServerFn } from '@tanstack/react-start';
import type { GenomeResult, RunInfo } from '../types';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { Database } from 'bun:sqlite';

/** Get the project root directory (parent of web/) */
function getProjectRoot(): string {
  const cwd = process.cwd();
  // When running from web/ directory (dev mode)
  if (cwd.endsWith('/web') || cwd.endsWith('\\web')) {
    return dirname(cwd);
  }
  // Check if we're at project root
  if (existsSync(join(cwd, 'data', 'string_theory.db'))) {
    return cwd;
  }
  // Try Bun-specific import.meta.dir (web/src/server/)
  const metaDir = (import.meta as unknown as { dir?: string }).dir;
  if (metaDir && existsSync(join(dirname(dirname(dirname(metaDir))), 'data', 'string_theory.db'))) {
    return dirname(dirname(dirname(metaDir)));
  }
  // Fallback: try to find db relative to __dirname-like paths
  // In Vite SSR, file URLs are used
  const url = (import.meta as unknown as { url?: string }).url;
  if (url) {
    const filePath = url.startsWith('file://') ? url.slice(7) : url;
    const serverDir = dirname(filePath);
    const potentialRoot = dirname(dirname(dirname(serverDir)));
    if (existsSync(join(potentialRoot, 'data', 'string_theory.db'))) {
      return potentialRoot;
    }
  }
  return cwd;
}

function getDbPath(): string {
  return join(getProjectRoot(), 'data', 'string_theory.db');
}

/** List all available runs from SQLite */
export const listRuns = createServerFn({ method: 'GET' }).handler(
  async (): Promise<RunInfo[]> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      // New schema: runs table has algorithm_id, run_number, final_fitness, etc.
      const rows = db.prepare(`
        SELECT
          r.id,
          r.algorithm_id,
          r.run_number,
          r.started_at,
          r.generations_run,
          r.final_fitness,
          a.meta_generation,
          (SELECT COUNT(*) FROM evaluations WHERE run_id = r.id) as eval_count
        FROM runs r
        JOIN meta_algorithms a ON a.id = r.algorithm_id
        ORDER BY r.started_at DESC
      `).all() as Array<{
        id: number;
        algorithm_id: number;
        run_number: number;
        started_at: string | null;
        generations_run: number | null;
        final_fitness: number | null;
        meta_generation: number;
        eval_count: number;
      }>;

      db.close();

      // Map runs - for now use unique_polytopes_tried as proxy for genome count
      // since evaluations.run_id is NULL for old data
      return rows.map(row => ({
        id: String(row.id),
        path: `/meta/gen/${row.meta_generation}/algo/${row.algorithm_id}/run/${row.run_number}`,
        genomeCount: row.eval_count || row.generations_run || 0,
        bestFitness: row.final_fitness ?? 0,
        timestamp: row.started_at ? new Date(row.started_at) : new Date(0),
      }));
    } catch (e) {
      console.error('Failed to list runs:', e);
      return [];
    }
  }
);

/** Get all evaluations (not filtered by run - for overview) */
export const getAllEvaluations = createServerFn({ method: 'GET' })
  .inputValidator((data: { limit?: number }) => data)
  .handler(async ({ data: { limit = 100 } }): Promise<GenomeResult[]> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const rows = db.prepare(`
        SELECT
          e.*,
          p.h11,
          p.h21
        FROM evaluations e
        LEFT JOIN polytopes p ON p.id = e.polytope_id
        WHERE e.success = 1
        ORDER BY e.fitness DESC
        LIMIT ?
      `).all(limit) as Record<string, unknown>[];

      db.close();

      return rows.map(rowToGenomeResult);
    } catch (e) {
      console.error('Failed to read evaluations:', e);
      return [];
    }
  });

/** List top evaluations in a run */
export const listGenomes = createServerFn({ method: 'GET' })
  .inputValidator((data: { runId: string }) => data)
  .handler(
    async ({
      data: { runId },
    }): Promise<{ filename: string; fitness: number }[]> => {
      const dbPath = getDbPath();
      if (!existsSync(dbPath)) {
        return [];
      }

      try {
        const db = new Database(dbPath, { readonly: true });

        const rows = db.prepare(`
          SELECT id, fitness
          FROM evaluations
          WHERE run_id = ?
          ORDER BY fitness DESC
          LIMIT 100
        `).all(runId) as Array<{ id: number; fitness: number }>;

        db.close();

        return rows.map(row => ({
          filename: `eval_${row.id}`,
          fitness: row.fitness,
        }));
      } catch (e) {
        console.error('Failed to list genomes:', e);
        return [];
      }
    }
  );

/** Read a single evaluation result by eval ID */
export const getGenome = createServerFn({ method: 'GET' })
  .inputValidator((data: { runId: string; evalId: number }) => data)
  .handler(async ({ data: { runId, evalId } }): Promise<GenomeResult | null> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return null;
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const row = db.prepare(`
        SELECT
          e.*,
          p.h11,
          p.h21
        FROM evaluations e
        LEFT JOIN polytopes p ON p.id = e.polytope_id
        WHERE e.id = ? AND e.run_id = ?
      `).get(evalId, runId) as Record<string, unknown> | null;

      db.close();

      if (!row) return null;

      return rowToGenomeResult(row);
    } catch (e) {
      console.error('Failed to read genome:', e);
      return null;
    }
  });

/** Get best genome for a polytope in a specific run */
export const getGenomeByPolytope = createServerFn({ method: 'GET' })
  .inputValidator((data: { runId: string; polytopeId: number }) => data)
  .handler(async ({ data: { runId, polytopeId } }): Promise<GenomeResult | null> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return null;
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const row = db.prepare(`
        SELECT
          e.*,
          p.h11,
          p.h21
        FROM evaluations e
        LEFT JOIN polytopes p ON p.id = e.polytope_id
        WHERE e.polytope_id = ? AND e.run_id = ?
        ORDER BY e.fitness DESC
        LIMIT 1
      `).get(polytopeId, runId) as Record<string, unknown> | null;

      db.close();

      if (!row) return null;

      return rowToGenomeResult(row);
    } catch (e) {
      console.error('Failed to read genome by polytope:', e);
      return null;
    }
  });

/** Read all evaluations from a run (for scatter plots) */
export const getAllGenomes = createServerFn({ method: 'GET' })
  .inputValidator((data: { runId: string }) => data)
  .handler(async ({ data: { runId } }): Promise<GenomeResult[]> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const rows = db.prepare(`
        SELECT
          e.*,
          p.h11,
          p.h21
        FROM evaluations e
        LEFT JOIN polytopes p ON p.id = e.polytope_id
        WHERE e.run_id = ?
        ORDER BY e.fitness DESC
      `).all(runId) as Record<string, unknown>[];

      db.close();

      return rows.map(rowToGenomeResult);
    } catch (e) {
      console.error('Failed to read genomes:', e);
      return [];
    }
  });

/** Convert a database row to GenomeResult */
function rowToGenomeResult(row: Record<string, unknown>): GenomeResult {
  return {
    genome: {
      polytope_id: row.polytope_id as number,
      kahler_moduli: JSON.parse((row.kahler_moduli as string) || '[]'),
      complex_moduli: JSON.parse((row.complex_moduli as string) || '[]'),
      flux_f: JSON.parse((row.flux_f as string) || '[]'),
      flux_h: JSON.parse((row.flux_h as string) || '[]'),
      g_s: row.g_s as number,
      h11: (row.h11 as number) ?? 0,
      h21: (row.h21 as number) ?? 0,
    },
    physics: {
      success: (row.success as number) === 1,
      error: row.error as string | null,
      alpha_em: row.alpha_em as number,
      alpha_s: row.alpha_s as number,
      sin2_theta_w: row.sin2_theta_w as number,
      cosmological_constant: row.cosmological_constant as number,
      n_generations: row.n_generations as number,
      m_e_planck_ratio: 0,
      m_p_planck_ratio: 0,
      cy_volume: 0,
      string_coupling: row.g_s as number,
      flux_tadpole: 0,
      superpotential_abs: 0,
    },
    fitness: row.fitness as number,
  };
}
