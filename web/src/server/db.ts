/**
 * Shared SQLite access for server functions
 *
 * DB path resolution order:
 *   1. STRING_THEORY_DB environment variable
 *   2. config.toml [paths] database (relative to project root)
 *   3. <project root>/data/string_theory.db
 */

import { existsSync, readFileSync } from 'node:fs';
import { join, dirname, isAbsolute } from 'node:path';
import { Database } from 'bun:sqlite';

/** Get the project root directory (parent of web/) */
export function getProjectRoot(): string {
  const cwd = process.cwd();
  if (cwd.endsWith('/web') || cwd.endsWith('\\web')) {
    return dirname(cwd);
  }
  if (existsSync(join(cwd, 'web'))) {
    return cwd;
  }
  // Bun-specific: import.meta.dir points at web/src/server/
  const metaDir = (import.meta as unknown as { dir?: string }).dir;
  if (metaDir && existsSync(join(dirname(dirname(dirname(metaDir))), 'config.toml'))) {
    return dirname(dirname(dirname(metaDir)));
  }
  return cwd;
}

/** Read [paths] database from config.toml (minimal TOML parsing) */
function configDbPath(root: string): string | null {
  const configPath = join(root, 'config.toml');
  if (!existsSync(configPath)) return null;
  try {
    const text = readFileSync(configPath, 'utf-8');
    const match = text.match(/^\s*database\s*=\s*"([^"]+)"/m);
    if (match) return match[1];
  } catch {
    // fall through
  }
  return null;
}

export function getDbPath(): string {
  const env = process.env.STRING_THEORY_DB;
  if (env) return env;
  const root = getProjectRoot();
  const fromConfig = configDbPath(root);
  if (fromConfig) {
    return isAbsolute(fromConfig) ? fromConfig : join(root, fromConfig);
  }
  return join(root, 'data', 'string_theory.db');
}

/**
 * Run a query against the dashboard DB (read-only).
 * Returns `fallback` if the DB doesn't exist or the query fails
 * (e.g. ingester hasn't created the schema yet).
 */
export function withDb<T>(fn: (db: Database) => T, fallback: T): T {
  const dbPath = getDbPath();
  if (!existsSync(dbPath)) {
    console.error('Database not found:', dbPath);
    return fallback;
  }
  let db: Database | null = null;
  try {
    db = new Database(dbPath, { readonly: true });
    return fn(db);
  } catch (error) {
    console.error('Database query failed:', error);
    return fallback;
  } finally {
    db?.close();
  }
}
