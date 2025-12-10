/**
 * Server functions for playground evaluation
 *
 * Flow:
 * 1. Frontend sends evaluation params
 * 2. Compute SHA256 input hash
 * 3. Check for cached result by hash
 * 4. If cache miss, spawn Rust evaluate binary
 * 5. Return evaluation result
 */

import { createServerFn } from '@tanstack/react-start';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { Database } from 'bun:sqlite';
import { createHash } from 'node:crypto';
import type { Evaluation, EvaluationSource } from '../types';

/** Physics model version - MUST match src/constants.rs and physics_bridge.py */
const PHYSICS_MODEL_VERSION = '1.0.0';

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

/** Parameters for a playground evaluation */
export interface PlaygroundParams {
  // Polytope source
  polytopeId?: number;              // If from our DB
  verticesJson?: string;            // If external (JSON array of 4D coords)

  // Hodge numbers (for external polytopes)
  h11?: number;
  h21?: number;

  // Physics parameters
  g_s: number;
  kahlerModuli: number[];
  complexModuli: number[];
  fluxF: number[];
  fluxH: number[];

  // Optional metadata
  label?: string;
}

/** Compute SHA256 hash of evaluation inputs for caching */
function computeInputHash(params: PlaygroundParams, vertices: number[][] | null): string {
  const hashInput = {
    model_version: PHYSICS_MODEL_VERSION,
    vertices: vertices,
    kahler_moduli: params.kahlerModuli,
    complex_moduli: params.complexModuli,
    flux_f: params.fluxF,
    flux_h: params.fluxH,
    g_s: params.g_s,
  };
  return createHash('sha256').update(JSON.stringify(hashInput)).digest('hex');
}

/** Get vertices for a polytope from our DB */
function getPolytopeVertices(db: Database, polytopeId: number): number[][] | null {
  const row = db.prepare('SELECT vertices FROM polytopes WHERE id = ?').get(polytopeId) as { vertices: string } | null;
  if (!row) return null;
  try {
    // Vertices are stored as flat array in DB, need to reshape
    const flat = JSON.parse(row.vertices) as number[];
    const vertices: number[][] = [];
    for (let i = 0; i < flat.length; i += 4) {
      vertices.push([flat[i], flat[i + 1], flat[i + 2], flat[i + 3]]);
    }
    return vertices;
  } catch {
    return null;
  }
}

/** Convert database row to Evaluation */
function rowToEvaluation(row: Record<string, unknown>): Evaluation {
  return {
    id: row.id as number,
    polytope_id: row.polytope_id as number,
    run_id: row.run_id as number | null,
    generation: row.generation as number | null,
    g_s: row.g_s as number | null,
    kahler_moduli: row.kahler_moduli ? JSON.parse(row.kahler_moduli as string) : null,
    complex_moduli: row.complex_moduli ? JSON.parse(row.complex_moduli as string) : null,
    flux_f: row.flux_f ? JSON.parse(row.flux_f as string) : null,
    flux_h: row.flux_h ? JSON.parse(row.flux_h as string) : null,
    fitness: row.fitness as number,
    alpha_em: row.alpha_em as number | null,
    alpha_s: row.alpha_s as number | null,
    sin2_theta_w: row.sin2_theta_w as number | null,
    n_generations: row.n_generations as number | null,
    cosmological_constant: row.cosmological_constant as number | null,
    success: Boolean(row.success),
    error: row.error as string | null,
    created_at: row.created_at as string,
    input_hash: row.input_hash as string | null,
    model_version: row.model_version as string | null,
    source: row.source as EvaluationSource | null,
    label: row.label as string | null,
    vertices_json: row.vertices_json as string | null,
    h11: row.h11 as number | null,
    h21: row.h21 as number | null,
  };
}

/** Check cache for existing evaluation by input hash */
export const getEvaluationByHash = createServerFn({ method: 'GET' })
  .inputValidator((data: { hash: string }) => data)
  .handler(async ({ data: { hash } }): Promise<Evaluation | null> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) return null;

    try {
      const db = new Database(dbPath, { readonly: true });
      const row = db.prepare(`
        SELECT * FROM evaluations WHERE input_hash = ?
      `).get(hash) as Record<string, unknown> | null;
      db.close();

      if (!row) return null;
      return rowToEvaluation(row);
    } catch (e) {
      console.error('Error fetching evaluation by hash:', e);
      return null;
    }
  });

/** Run a playground evaluation */
export const runPlaygroundEvaluation = createServerFn({ method: 'POST' })
  .inputValidator((data: PlaygroundParams) => data)
  .handler(async ({ data: params }): Promise<{ evaluation: Evaluation | null; cached: boolean; error: string | null }> => {
    if (!params) {
      return { evaluation: null, cached: false, error: 'No parameters received' };
    }

    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return { evaluation: null, cached: false, error: `Database not found at ${dbPath}` };
    }

    try {
      const db = new Database(dbPath);

      // 1. Get vertices (from DB or external)
      let vertices: number[][] | null = null;
      let polytopeId = -1;

      if (params.polytopeId !== undefined && params.polytopeId !== -1) {
        vertices = getPolytopeVertices(db, params.polytopeId);
        if (!vertices) {
          db.close();
          return { evaluation: null, cached: false, error: `Polytope ${params.polytopeId} not found` };
        }
        polytopeId = params.polytopeId;
      } else if (params.verticesJson) {
        try {
          vertices = JSON.parse(params.verticesJson);
        } catch {
          db.close();
          return { evaluation: null, cached: false, error: 'Invalid vertices JSON' };
        }
      } else {
        db.close();
        return { evaluation: null, cached: false, error: 'No polytope specified' };
      }

      // 2. Compute input hash
      const inputHash = computeInputHash(params, vertices);

      // 3. Check cache
      const cached = db.prepare('SELECT * FROM evaluations WHERE input_hash = ?').get(inputHash) as Record<string, unknown> | null;
      if (cached) {
        db.close();
        return { evaluation: rowToEvaluation(cached), cached: true, error: null };
      }

      // 4. Run evaluation via Rust binary
      const projectRoot = getProjectRoot();
      const evaluateBin = join(projectRoot, 'target', 'release', 'evaluate');

      if (!existsSync(evaluateBin)) {
        db.close();
        return { evaluation: null, cached: false, error: 'Evaluate binary not found. Run: cargo build --release --bin evaluate' };
      }

      // Build command arguments
      const args = [
        '--database', dbPath,
        '--vertices', JSON.stringify(vertices),
        '--g-s', String(params.g_s),
        '--kahler', JSON.stringify(params.kahlerModuli),
        '--complex', JSON.stringify(params.complexModuli),
        '--flux-k', JSON.stringify(params.fluxF),
        '--flux-m', JSON.stringify(params.fluxH),
        '--json',  // Output JSON result
        '--save',  // Save to database
        '--source', 'playground',
        '--hash', inputHash,
        '--model-version', PHYSICS_MODEL_VERSION,
      ];

      if (params.label) {
        args.push('--label', params.label);
      }

      if (params.h11 !== undefined) {
        args.push('--h11', String(params.h11));
      }
      if (params.h21 !== undefined) {
        args.push('--h21', String(params.h21));
      }

      // Spawn the evaluate binary
      const proc = Bun.spawn([evaluateBin, ...args], {
        cwd: projectRoot,
        env: {
          ...process.env,
          VIRTUAL_ENV: join(projectRoot, '.venv'),
        },
        stdout: 'pipe',
        stderr: 'pipe',
      });

      const stdout = await new Response(proc.stdout).text();
      const stderr = await new Response(proc.stderr).text();
      const exitCode = await proc.exited;

      if (exitCode !== 0) {
        db.close();
        console.error('Evaluate stderr:', stderr);
        return { evaluation: null, cached: false, error: `Evaluation failed: ${stderr || stdout}` };
      }

      // 5. Parse result - the binary outputs the evaluation ID
      let evalId: number;
      try {
        const result = JSON.parse(stdout.trim());
        evalId = result.evaluation_id;
      } catch {
        db.close();
        return { evaluation: null, cached: false, error: `Failed to parse evaluation result: ${stdout}` };
      }

      // 6. Fetch the saved evaluation from DB
      const newEval = db.prepare('SELECT * FROM evaluations WHERE id = ?').get(evalId) as Record<string, unknown> | null;
      db.close();

      if (!newEval) {
        return { evaluation: null, cached: false, error: 'Evaluation was not saved to database' };
      }

      return { evaluation: rowToEvaluation(newEval), cached: false, error: null };
    } catch (e) {
      console.error('Error running playground evaluation:', e);
      return { evaluation: null, cached: false, error: String(e) };
    }
  });

/** Get recent playground evaluations */
export const getPlaygroundEvaluations = createServerFn({ method: 'GET' })
  .inputValidator((data: { limit?: number } | undefined) => data ?? { limit: 50 })
  .handler(async ({ data }): Promise<Evaluation[]> => {
    const limit = data?.limit ?? 50;
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });
      const rows = db.prepare(`
        SELECT * FROM evaluations
        WHERE source = 'playground'
        ORDER BY created_at DESC
        LIMIT ?
      `).all(limit) as Record<string, unknown>[];
      db.close();

      return rows.map(rowToEvaluation);
    } catch (e) {
      console.error('Error fetching playground evaluations:', e);
      return [];
    }
  });

/** Predefined configurations for the playground dropdown */
export interface PredefinedConfig {
  id: string;
  name: string;
  description: string;
  params: Omit<PlaygroundParams, 'label'>;
}

/** Get predefined configurations including McAllister data */
export const getPredefinedConfigs = createServerFn({ method: 'GET' }).handler(
  async (): Promise<PredefinedConfig[]> => {
    // These are loaded from a static list - McAllister data is used as test fixtures
    // but also made available as predefined playground configs
    const configs: PredefinedConfig[] = [
      {
        id: 'quintic',
        name: 'Quintic Threefold',
        description: 'The simplest CY3: P^4[5] with h11=1, h21=101',
        params: {
          verticesJson: JSON.stringify([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, -1, -1, -1],
          ]),
          h11: 1,
          h21: 101,
          g_s: 0.1,
          kahlerModuli: [2.0],
          complexModuli: [1.0],
          fluxF: [1, 0, 0, 0],
          fluxH: [0, 1, 0, 0],
        },
      },
      // McAllister et al. 4-214-647 configuration
      // This is from arXiv:2107.09064 - achieving |W0| ~ 10^-90
      {
        id: 'mcallister-4-214-647',
        name: 'McAllister 4-214-647',
        description: 'Small CC config from arXiv:2107.09064 (h11=4, h21=214, |W0|~10^-90)',
        params: {
          // Vertices from dual_points.dat
          verticesJson: JSON.stringify([
            [0, 0, 0, 0],
            [-1, 2, -1, -1],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [1, -1, 1, 1],
          ]),
          h11: 4,
          h21: 214,
          g_s: 0.00911134,
          // First 4 Kahler moduli from kahler_param.dat
          kahlerModuli: [0.738461538461538, 10.4666666666667, 0.469230769230769, 9.73333333333333],
          complexModuli: Array(214).fill(1.0),
          fluxF: [-3, -5, 8, 6],  // K flux from K_vec.dat
          fluxH: [10, 11, -11, -5],  // M flux from M_vec.dat
        },
      },
    ];

    return configs;
  }
);
