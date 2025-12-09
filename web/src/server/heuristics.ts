/**
 * Server functions for polytope heuristics data
 */

import { createServerFn } from '@tanstack/react-start';
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
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

/**
 * Get all computed heuristics from the JSON file
 */
export const getHeuristics = createServerFn({ method: 'GET' }).handler(
  async (): Promise<PolytopeHeuristics[]> => {
    const projectRoot = getProjectRoot();
    const heuristicsPath = join(projectRoot, 'heuristics_sample.json');

    try {
      const content = await readFile(heuristicsPath, 'utf-8');
      const heuristics = JSON.parse(content) as PolytopeHeuristics[];
      return heuristics;
    } catch (error) {
      console.error('Error loading heuristics:', error);
      return [];
    }
  }
);
