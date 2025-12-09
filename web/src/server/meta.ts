/**
 * Server functions for meta-GA data access
 */

import { createServerFn } from '@tanstack/react-start';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import Database from 'better-sqlite3';
import type {
  MetaAlgorithm,
  MetaAlgorithmWithFitness,
  MetaTrial,
  MetaState,
  GenerationStatus,
} from '../types';

function getProjectRoot(): string {
  const cwd = process.cwd();
  if (cwd.endsWith('/web') || cwd.endsWith('\\web')) {
    return dirname(cwd);
  }
  // Bun-specific: import.meta.dir
  const metaDir = (import.meta as unknown as { dir?: string }).dir;
  if (metaDir) {
    return dirname(dirname(dirname(metaDir)));
  }
  return cwd;
}

function getDbPath(): string {
  return join(getProjectRoot(), 'data', 'string_theory.db');
}

/** Parse feature_weights JSON string to object */
function parseFeatureWeights(json: string): Record<string, number> {
  try {
    return JSON.parse(json);
  } catch {
    return {};
  }
}

/** Convert database row to MetaAlgorithm */
function rowToAlgorithm(row: Record<string, unknown>): MetaAlgorithm {
  return {
    id: row.id as number,
    name: row.name as string | null,
    version: row.version as number,
    feature_weights: parseFeatureWeights(row.feature_weights as string),
    similarity_radius: row.similarity_radius as number,
    interpolation_weight: row.interpolation_weight as number,
    population_size: row.population_size as number,
    max_generations: row.max_generations as number,
    mutation_rate: row.mutation_rate as number,
    mutation_strength: row.mutation_strength as number,
    crossover_rate: row.crossover_rate as number,
    tournament_size: row.tournament_size as number,
    elite_count: row.elite_count as number,
    polytope_patience: row.polytope_patience as number,
    switch_threshold: row.switch_threshold as number,
    switch_probability: row.switch_probability as number,
    cc_weight: row.cc_weight as number,
    parent_id: row.parent_id as number | null,
    meta_generation: row.meta_generation as number,
    rng_seed: String(row.rng_seed ?? '0'),
    status: row.status as MetaAlgorithm['status'],
    trials_required: row.trials_required as number,
    locked_by_pid: row.locked_by_pid as number | null,
    last_heartbeat_at: row.last_heartbeat_at as string | null,
    completed_at: row.completed_at as string | null,
    created_at: row.created_at as string,
  };
}

/** Convert database row to MetaAlgorithmWithFitness */
function rowToAlgorithmWithFitness(row: Record<string, unknown>): MetaAlgorithmWithFitness {
  return {
    ...rowToAlgorithm(row),
    trial_count: (row.trial_count as number) ?? 0,
    mean_improvement_rate: row.mean_improvement_rate as number | null,
    best_improvement_rate: row.best_improvement_rate as number | null,
    mean_fitness_auc: row.mean_fitness_auc as number | null,
    best_final_fitness: row.best_final_fitness as number | null,
    best_cc_log_error: row.best_cc_log_error as number | null,
    mean_cc_log_error: row.mean_cc_log_error as number | null,
    meta_fitness: row.meta_fitness as number | null,
  };
}

/** Convert database row to MetaTrial */
function rowToTrial(row: Record<string, unknown>): MetaTrial {
  return {
    id: row.id as number,
    algorithm_id: row.algorithm_id as number,
    run_id: row.run_id as string | null,
    generations_run: row.generations_run as number,
    initial_fitness: row.initial_fitness as number,
    final_fitness: row.final_fitness as number,
    fitness_improvement: row.fitness_improvement as number,
    improvement_rate: row.improvement_rate as number,
    fitness_auc: row.fitness_auc as number,
    best_cc_log_error: row.best_cc_log_error as number,
    physics_success_rate: row.physics_success_rate as number,
    unique_polytopes_tried: row.unique_polytopes_tried as number,
    started_at: row.started_at as string | null,
    ended_at: row.ended_at as string | null,
  };
}

/**
 * Get meta-state (global meta-GA status)
 */
export const getMetaState = createServerFn({ method: 'GET' }).handler(
  async (): Promise<MetaState | null> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return null;
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const row = db.prepare(`
        SELECT current_generation, algorithms_per_generation,
               best_meta_fitness, best_algorithm_id, master_seed, updated_at
        FROM meta_state WHERE id = 1
      `).get() as Record<string, unknown> | undefined;

      db.close();

      if (!row) return null;

      return {
        current_generation: row.current_generation as number,
        algorithms_per_generation: row.algorithms_per_generation as number,
        best_meta_fitness: row.best_meta_fitness as number | null,
        best_algorithm_id: row.best_algorithm_id as number | null,
        master_seed: row.master_seed ? String(row.master_seed) : null,
        updated_at: row.updated_at as string | null,
      };
    } catch (error) {
      console.error('Error loading meta state:', error);
      return null;
    }
  }
);

/**
 * Get generation status counts
 */
export const getGenerationStatus = createServerFn({ method: 'GET' })
  .inputValidator((data: { generation: number }) => data)
  .handler(async ({ data: { generation } }): Promise<GenerationStatus> => {
    const dbPath = getDbPath();
    const defaultStatus: GenerationStatus = {
      generation,
      pending: 0,
      running: 0,
      completed: 0,
      failed: 0,
      total: 0,
    };

    if (!existsSync(dbPath)) {
      return defaultStatus;
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const counts = db.prepare(`
        SELECT
          status,
          COUNT(*) as count
        FROM meta_algorithms
        WHERE meta_generation = ?
        GROUP BY status
      `).all(generation) as Array<{ status: string; count: number }>;

      db.close();

      const result = { ...defaultStatus };
      for (const { status, count } of counts) {
        if (status in result) {
          result[status as keyof GenerationStatus] = count;
        }
        result.total += count;
      }

      return result;
    } catch (error) {
      console.error('Error loading generation status:', error);
      return defaultStatus;
    }
  });

/**
 * Get all algorithms with fitness data
 */
export const getAlgorithms = createServerFn({ method: 'GET' })
  .inputValidator((data: { generation?: number; limit?: number }) => data)
  .handler(async ({ data: { generation, limit = 100 } }): Promise<MetaAlgorithmWithFitness[]> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      let query = `
        SELECT
          a.*,
          COALESCE(f.trial_count, 0) as trial_count,
          f.mean_improvement_rate,
          f.best_improvement_rate,
          f.mean_fitness_auc,
          f.best_final_fitness,
          f.best_cc_log_error,
          f.mean_cc_log_error,
          f.meta_fitness
        FROM meta_algorithms a
        LEFT JOIN meta_fitness f ON f.algorithm_id = a.id
      `;

      const params: unknown[] = [];
      if (generation !== undefined) {
        query += ' WHERE a.meta_generation = ?';
        params.push(generation);
      }

      query += ' ORDER BY COALESCE(f.meta_fitness, 0) DESC, a.created_at DESC LIMIT ?';
      params.push(limit);

      const rows = db.prepare(query).all(...params) as Record<string, unknown>[];
      db.close();

      return rows.map(rowToAlgorithmWithFitness);
    } catch (error) {
      console.error('Error loading algorithms:', error);
      return [];
    }
  });

/**
 * Get a single algorithm by ID
 */
export const getAlgorithm = createServerFn({ method: 'GET' })
  .inputValidator((data: { id: number }) => data)
  .handler(async ({ data: { id } }): Promise<MetaAlgorithmWithFitness | null> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return null;
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const row = db.prepare(`
        SELECT
          a.*,
          COALESCE(f.trial_count, 0) as trial_count,
          f.mean_improvement_rate,
          f.best_improvement_rate,
          f.mean_fitness_auc,
          f.best_final_fitness,
          f.best_cc_log_error,
          f.mean_cc_log_error,
          f.meta_fitness
        FROM meta_algorithms a
        LEFT JOIN meta_fitness f ON f.algorithm_id = a.id
        WHERE a.id = ?
      `).get(id) as Record<string, unknown> | undefined;

      db.close();

      if (!row) return null;
      return rowToAlgorithmWithFitness(row);
    } catch (error) {
      console.error('Error loading algorithm:', error);
      return null;
    }
  });

/**
 * Get trials for an algorithm
 */
export const getTrials = createServerFn({ method: 'GET' })
  .inputValidator((data: { algorithmId: number }) => data)
  .handler(async ({ data: { algorithmId } }): Promise<MetaTrial[]> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const rows = db.prepare(`
        SELECT * FROM meta_trials
        WHERE algorithm_id = ?
        ORDER BY started_at DESC
      `).all(algorithmId) as Record<string, unknown>[];

      db.close();

      return rows.map(rowToTrial);
    } catch (error) {
      console.error('Error loading trials:', error);
      return [];
    }
  });

/**
 * Get all generations with their status
 */
export const getAllGenerations = createServerFn({ method: 'GET' }).handler(
  async (): Promise<GenerationStatus[]> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      const rows = db.prepare(`
        SELECT
          meta_generation as generation,
          SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
          SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
          SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
          SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
          COUNT(*) as total
        FROM meta_algorithms
        GROUP BY meta_generation
        ORDER BY meta_generation DESC
      `).all() as Array<{
        generation: number;
        pending: number;
        running: number;
        completed: number;
        failed: number;
        total: number;
      }>;

      db.close();

      return rows;
    } catch (error) {
      console.error('Error loading generations:', error);
      return [];
    }
  }
);

/**
 * Get algorithm lineage (ancestors up to root)
 */
export const getAlgorithmLineage = createServerFn({ method: 'GET' })
  .inputValidator((data: { id: number }) => data)
  .handler(async ({ data: { id } }): Promise<MetaAlgorithm[]> => {
    const dbPath = getDbPath();
    if (!existsSync(dbPath)) {
      return [];
    }

    try {
      const db = new Database(dbPath, { readonly: true });

      // Walk up the parent chain
      const lineage: MetaAlgorithm[] = [];
      let currentId: number | null = id;

      while (currentId !== null && lineage.length < 100) {
        const row = db.prepare(`
          SELECT * FROM meta_algorithms WHERE id = ?
        `).get(currentId) as Record<string, unknown> | undefined;

        if (!row) break;

        lineage.push(rowToAlgorithm(row));
        currentId = row.parent_id as number | null;
      }

      db.close();

      return lineage;
    } catch (error) {
      console.error('Error loading algorithm lineage:', error);
      return [];
    }
  });
