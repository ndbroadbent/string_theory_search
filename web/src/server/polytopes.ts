/**
 * Server functions for reading polytope data
 *
 * Uses the binary index file for O(1) lookups into the JSONL
 * Index format: 8 bytes file_len + N*8 bytes of u64 offsets (little endian)
 */

import { createServerFn } from '@tanstack/react-start';
import type { PolytopeEntry } from '../types';
import { readFile, open } from 'node:fs/promises';
import { join, dirname } from 'node:path';

/** Cache for index offsets */
let indexCache: bigint[] | null = null;

/** Get the project root directory (parent of web/) */
function getProjectRoot(): string {
  const cwd = process.cwd();
  if (cwd.endsWith('/web') || cwd.endsWith('\\web')) {
    return dirname(cwd);
  }
  if (import.meta.dir) {
    return dirname(dirname(dirname(import.meta.dir)));
  }
  return cwd;
}

/** Load the binary index file into memory */
async function loadIndex(): Promise<bigint[]> {
  if (indexCache) return indexCache;

  const root = getProjectRoot();
  const indexPath = join(root, 'data', 'polytopes_three_gen.jsonl.idx');

  try {
    const buffer = await readFile(indexPath);

    // First 8 bytes = file length (skip it)
    // Rest = u64 offsets in little endian
    const offsets: bigint[] = [];
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    // Skip first 8 bytes (file length)
    for (let i = 8; i < buffer.length; i += 8) {
      offsets.push(view.getBigUint64(i, true)); // little endian
    }

    console.log(`Loaded index with ${offsets.length} polytopes`);
    indexCache = offsets;
    return offsets;
  } catch (e) {
    console.error('Failed to load polytope index:', e);
    return [];
  }
}

/** Get a single polytope by ID */
export const getPolytope = createServerFn({ method: 'GET' })
  .inputValidator((data: { polytopeId: number }) => data)
  .handler(async ({ data: { polytopeId } }): Promise<PolytopeEntry | null> => {
    const offsets = await loadIndex();

    if (polytopeId < 0 || polytopeId >= offsets.length) {
      console.error(`Polytope ID ${polytopeId} out of range (0-${offsets.length - 1})`);
      return null;
    }

    const offset = offsets[polytopeId];
    const root = getProjectRoot();
    const dataPath = join(root, 'data', 'polytopes_three_gen.jsonl');

    try {
      const handle = await open(dataPath, 'r');

      // Read a chunk starting at offset (max 64KB should be enough for any polytope)
      const buffer = Buffer.alloc(65536);
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, Number(offset));
      await handle.close();

      // Find the newline to get the complete line
      let lineEnd = 0;
      for (let i = 0; i < bytesRead; i++) {
        if (buffer[i] === 0x0a) { // newline
          lineEnd = i;
          break;
        }
      }

      const line = buffer.toString('utf-8', 0, lineEnd || bytesRead).trim();
      return JSON.parse(line) as PolytopeEntry;
    } catch (e) {
      console.error('Failed to read polytope:', e);
      return null;
    }
  });

/** Get total polytope count */
export const getPolytopeCount = createServerFn({ method: 'GET' }).handler(
  async (): Promise<number> => {
    const offsets = await loadIndex();
    return offsets.length;
  }
);

/** Get multiple polytopes by ID (for batch loading) */
export const getPolytopes = createServerFn({ method: 'GET' })
  .inputValidator((data: { polytopeIds: number[] }) => data)
  .handler(
    async ({
      data: { polytopeIds },
    }): Promise<(PolytopeEntry | null)[]> => {
      const offsets = await loadIndex();
      const root = getProjectRoot();
      const dataPath = join(root, 'data', 'polytopes_three_gen.jsonl');

      const results: (PolytopeEntry | null)[] = [];

      try {
        const handle = await open(dataPath, 'r');
        const buffer = Buffer.alloc(65536);

        for (const id of polytopeIds) {
          if (id < 0 || id >= offsets.length) {
            results.push(null);
            continue;
          }

          const offset = offsets[id];
          const { bytesRead } = await handle.read(buffer, 0, buffer.length, Number(offset));

          let lineEnd = 0;
          for (let i = 0; i < bytesRead; i++) {
            if (buffer[i] === 0x0a) {
              lineEnd = i;
              break;
            }
          }

          try {
            const line = buffer.toString('utf-8', 0, lineEnd || bytesRead).trim();
            results.push(JSON.parse(line) as PolytopeEntry);
          } catch {
            results.push(null);
          }
        }

        await handle.close();
      } catch (e) {
        console.error('Failed to read polytopes:', e);
        while (results.length < polytopeIds.length) {
          results.push(null);
        }
      }

      return results;
    }
  );
