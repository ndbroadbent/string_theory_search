/**
 * Server functions for reading genome results
 */

import { createServerFn } from '@tanstack/react-start';
import type { GenomeResult, RunInfo } from '../types';
import { readdir, readFile, stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';

/** Get the project root directory (parent of web/) */
function getProjectRoot(): string {
  // In development, process.cwd() is the web/ directory
  // Project root is one level up
  const cwd = process.cwd();
  // If we're in web/, go up one level
  if (cwd.endsWith('/web') || cwd.endsWith('\\web')) {
    return dirname(cwd);
  }
  // If import.meta.dir is available (Bun), use it
  if (import.meta.dir) {
    // import.meta.dir points to src/server/, go up 3 levels
    return dirname(dirname(dirname(import.meta.dir)));
  }
  // Fallback: assume cwd is project root
  return cwd;
}

/** List all available runs */
export const listRuns = createServerFn({ method: 'GET' }).handler(
  async (): Promise<RunInfo[]> => {
    const resultsDir = join(getProjectRoot(), 'results');
    const runs: RunInfo[] = [];

    try {
      const entries = await readdir(resultsDir, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isDirectory() && entry.name.startsWith('run_')) {
          const runPath = join(resultsDir, entry.name);
          const files = await readdir(runPath);

          // Count genome files and find best fitness
          const genomeFiles = files.filter(
            (f) => f.startsWith('fit') && f.endsWith('.json')
          );

          let bestFitness = 0;
          let timestamp = new Date(0);

          for (const f of genomeFiles) {
            // Parse fitness from filename: fit0_5788_20251209_030240.json
            const match = f.match(/fit(\d+)_(\d+)/);
            if (match) {
              const fitness = parseFloat(`${match[1]}.${match[2]}`);
              if (fitness > bestFitness) bestFitness = fitness;
            }

            // Get modification time
            try {
              const s = await stat(join(runPath, f));
              if (s.mtime > timestamp) timestamp = s.mtime;
            } catch {
              // ignore stat errors
            }
          }

          runs.push({
            id: entry.name,
            path: runPath,
            genomeCount: genomeFiles.length,
            bestFitness,
            timestamp,
          });
        }
      }

      // Sort by timestamp descending
      runs.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    } catch (e) {
      console.error('Failed to list runs:', e);
    }

    return runs;
  }
);

/** List genome files in a run */
export const listGenomes = createServerFn({ method: 'GET' })
  .inputValidator((data: { runId: string }) => data)
  .handler(
    async ({
      data: { runId },
    }): Promise<{ filename: string; fitness: number }[]> => {
      const runDir = join(getProjectRoot(), 'results', runId);
      const genomes: { filename: string; fitness: number }[] = [];

      try {
        const files = await readdir(runDir);

        for (const f of files) {
          if (f.startsWith('fit') && f.endsWith('.json')) {
            const match = f.match(/fit(\d+)_(\d+)/);
            if (match) {
              genomes.push({
                filename: f,
                fitness: parseFloat(`${match[1]}.${match[2]}`),
              });
            }
          }
        }

        // Sort by fitness descending
        genomes.sort((a, b) => b.fitness - a.fitness);
      } catch (e) {
        console.error('Failed to list genomes:', e);
      }

      return genomes;
    }
  );

/** Read a single genome result */
export const getGenome = createServerFn({ method: 'GET' })
  .inputValidator((data: { runId: string; filename: string }) => data)
  .handler(async ({ data: { runId, filename } }): Promise<GenomeResult | null> => {
    const filePath = join(getProjectRoot(), 'results', runId, filename);

    try {
      const content = await readFile(filePath, 'utf-8');
      return JSON.parse(content) as GenomeResult;
    } catch (e) {
      console.error('Failed to read genome:', e);
      return null;
    }
  });

/** Read all genomes from a run (for scatter plots) */
export const getAllGenomes = createServerFn({ method: 'GET' })
  .inputValidator((data: { runId: string }) => data)
  .handler(async ({ data: { runId } }): Promise<GenomeResult[]> => {
    const runDir = join(getProjectRoot(), 'results', runId);
    const results: GenomeResult[] = [];

    try {
      const files = await readdir(runDir);

      for (const f of files) {
        if (f.startsWith('fit') && f.endsWith('.json')) {
          try {
            const content = await readFile(join(runDir, f), 'utf-8');
            results.push(JSON.parse(content) as GenomeResult);
          } catch {
            // Skip invalid files
          }
        }
      }

      // Sort by fitness descending
      results.sort((a, b) => b.fitness - a.fitness);
    } catch (e) {
      console.error('Failed to read genomes:', e);
    }

    return results;
  });
