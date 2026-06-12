/**
 * cyrus-ga -> SQLite ingest daemon
 *
 * Tails the append-only JSONL event streams written by the cyrus-ga systemd
 * service (rounds.jsonl + candidates.jsonl) and upserts them into the
 * dashboard SQLite database. Byte offsets are persisted in run_meta so
 * restarts never re-ingest or skip lines. Idempotent and resilient:
 * malformed lines are skipped with a log message.
 *
 * Usage:
 *   bun scripts/ingest.ts --run-dir /data/cyrus/runs/landscape \
 *                         --pool /data/cyrus/pools/h21_4-7.jsonl \
 *                         --db /path/to/string_theory.db
 *   bun scripts/ingest.ts ... --once       # single cycle (for testing)
 *
 * Defaults (when flags are omitted) come from config.toml [paths]
 * (run_dir, pool, database) with STRING_THEORY_DB env override for the DB.
 */

import { Database } from 'bun:sqlite';
import {
  closeSync,
  existsSync,
  openSync,
  readFileSync,
  readSync,
  statSync,
} from 'node:fs';
import { dirname, isAbsolute, join } from 'node:path';

// ---------------------------------------------------------------------------
// CLI + config

interface Options {
  runDir: string;
  pool: string;
  db: string;
  once: boolean;
  intervalMs: number;
}

function parseArgs(argv: string[]): Partial<Options> {
  const opts: Partial<Options> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--run-dir':
        opts.runDir = argv[++i];
        break;
      case '--pool':
        opts.pool = argv[++i];
        break;
      case '--db':
        opts.db = argv[++i];
        break;
      case '--once':
        opts.once = true;
        break;
      case '--interval':
        opts.intervalMs = Number(argv[++i]) * 1000;
        break;
      default:
        console.error(`Unknown argument: ${arg}`);
        process.exit(1);
    }
  }
  return opts;
}

function projectRoot(): string {
  // scripts/ lives in web/, project root is its parent
  const scriptDir = import.meta.dir;
  return dirname(dirname(scriptDir));
}

function configPaths(root: string): { runDir?: string; pool?: string; db?: string } {
  const configPath = join(root, 'config.toml');
  if (!existsSync(configPath)) return {};
  const text = readFileSync(configPath, 'utf-8');
  const get = (key: string): string | undefined =>
    text.match(new RegExp(`^\\s*${key}\\s*=\\s*"([^"]+)"`, 'm'))?.[1];
  const resolve = (p: string | undefined) =>
    p == null ? undefined : isAbsolute(p) ? p : join(root, p);
  return {
    runDir: resolve(get('run_dir')),
    pool: resolve(get('pool')),
    db: resolve(get('database')),
  };
}

function resolveOptions(): Options {
  const root = projectRoot();
  const fromConfig = configPaths(root);
  const fromArgs = parseArgs(process.argv.slice(2));

  const runDir = fromArgs.runDir ?? fromConfig.runDir;
  const pool = fromArgs.pool ?? fromConfig.pool;
  const db =
    fromArgs.db ??
    process.env.STRING_THEORY_DB ??
    fromConfig.db ??
    join(root, 'data', 'string_theory.db');

  if (!runDir || !pool) {
    console.error(
      'Usage: bun scripts/ingest.ts --run-dir <dir> --pool <pool.jsonl> --db <path> [--once] [--interval <secs>]'
    );
    process.exit(1);
  }

  return { runDir, pool, db, once: fromArgs.once ?? false, intervalMs: fromArgs.intervalMs ?? 15_000 };
}

// ---------------------------------------------------------------------------
// Schema

function ensureSchema(db: Database) {
  const migrationPath = join(projectRoot(), 'migrations', '001_cyrus_ga_schema.sql');
  const sql = readFileSync(migrationPath, 'utf-8');
  db.exec(`CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now')),
    description TEXT
  )`);
  db.exec(sql); // all statements are CREATE ... IF NOT EXISTS (idempotent)
  db.exec(
    `INSERT OR IGNORE INTO schema_version (version, description) VALUES (1, 'cyrus-ga schema')`
  );
}

// ---------------------------------------------------------------------------
// Pool index: name -> byte offset into the pool JSONL

interface PoolIndex {
  path: string;
  offsets: Map<string, { offset: number; length: number }>;
}

function buildPoolIndex(path: string): PoolIndex {
  const offsets = new Map<string, { offset: number; length: number }>();
  const buffer = readFileSync(path);
  let start = 0;
  while (start < buffer.length) {
    let end = buffer.indexOf(0x0a, start);
    if (end === -1) end = buffer.length;
    const line = buffer.toString('utf-8', start, end).trim();
    if (line.length > 0) {
      try {
        const record = JSON.parse(line) as { name?: string };
        if (record.name) {
          offsets.set(record.name, { offset: start, length: end - start });
        } else {
          console.warn(`[pool] line at offset ${start} has no "name", skipping`);
        }
      } catch (e) {
        console.warn(`[pool] malformed line at offset ${start}, skipping: ${e}`);
      }
    }
    start = end + 1;
  }
  console.log(`[pool] indexed ${offsets.size} polytopes from ${path}`);
  return { path, offsets };
}

interface PoolRecord {
  name: string;
  h11: number;
  h21: number;
  favorable: boolean;
  points: number[][];
}

function readPoolRecord(index: PoolIndex, name: string): PoolRecord | null {
  const entry = index.offsets.get(name);
  if (!entry) return null;
  const fd = openSync(index.path, 'r');
  try {
    const buffer = Buffer.alloc(entry.length);
    readSync(fd, buffer, 0, entry.length, entry.offset);
    return JSON.parse(buffer.toString('utf-8')) as PoolRecord;
  } catch (e) {
    console.warn(`[pool] failed to read record for ${name}: ${e}`);
    return null;
  } finally {
    closeSync(fd);
  }
}

// ---------------------------------------------------------------------------
// JSONL tailing with persisted byte offsets

function getMeta(db: Database, key: string): string | null {
  const row = db.prepare('SELECT value FROM run_meta WHERE key = ?1').get(key) as
    | { value: string }
    | null;
  return row?.value ?? null;
}

function setMeta(db: Database, key: string, value: string) {
  db.prepare(
    'INSERT INTO run_meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value'
  ).run(key, value);
}

/**
 * Read complete new lines from `path` starting at the persisted offset.
 * Returns the parsed JSON objects plus the new offset (positioned after the
 * last complete line; a trailing partial line is left for the next cycle).
 */
function tailJsonl(
  db: Database,
  path: string,
  offsetKey: string
): { records: unknown[]; newOffset: number; consumed: boolean } {
  if (!existsSync(path)) {
    return { records: [], newOffset: 0, consumed: false };
  }
  let offset = Number(getMeta(db, offsetKey) ?? '0');
  const size = statSync(path).size;
  if (size < offset) {
    // File was truncated/rotated: start over
    console.warn(`[tail] ${path} shrank (${size} < ${offset}), resetting offset`);
    offset = 0;
  }
  if (size === offset) {
    return { records: [], newOffset: offset, consumed: false };
  }

  const fd = openSync(path, 'r');
  let chunk: Buffer;
  try {
    chunk = Buffer.alloc(size - offset);
    readSync(fd, chunk, 0, chunk.length, offset);
  } finally {
    closeSync(fd);
  }

  const lastNewline = chunk.lastIndexOf(0x0a);
  if (lastNewline === -1) {
    // No complete line yet
    return { records: [], newOffset: offset, consumed: false };
  }

  const text = chunk.toString('utf-8', 0, lastNewline + 1);
  const records: unknown[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    try {
      records.push(JSON.parse(trimmed));
    } catch (e) {
      console.warn(`[tail] skipping malformed line in ${path}: ${e}`);
    }
  }
  return { records, newOffset: offset + lastNewline + 1, consumed: true };
}

// ---------------------------------------------------------------------------
// Event records

interface RoundEvent {
  round: number;
  polytope: string;
  evals: number;
  valid: number;
  best_here: number | null;
  global_best: number | null;
  live: number | null;
  ts: number;
}

interface CandidateEvent {
  polytope: string;
  round: number;
  genome: { k: number[]; m: number[] };
  report: {
    cpl_w0: number | null;
    cpl_wa: number | null;
    fitness: number | null;
    g_s: number | null;
    log10_abs_v0: number | null;
    q_flux: number | null;
    tier: string | null;
    w0: number | null;
  };
  ts: number;
}

function isRoundEvent(r: unknown): r is RoundEvent {
  const o = r as Record<string, unknown>;
  return (
    o != null &&
    typeof o.polytope === 'string' &&
    typeof o.round === 'number' &&
    typeof o.evals === 'number'
  );
}

function isCandidateEvent(r: unknown): r is CandidateEvent {
  const o = r as Record<string, unknown>;
  return (
    o != null &&
    typeof o.polytope === 'string' &&
    o.genome != null &&
    Array.isArray((o.genome as Record<string, unknown>).k) &&
    Array.isArray((o.genome as Record<string, unknown>).m) &&
    o.report != null
  );
}

// ---------------------------------------------------------------------------
// Ingest cycle

function ensurePolytope(db: Database, pool: PoolIndex, name: string) {
  const existing = db
    .prepare('SELECT id, h11 FROM polytopes WHERE id = ?1')
    .get(name) as { id: string; h11: number | null } | null;
  if (existing && existing.h11 != null) return;

  if (!existing) {
    db.prepare('INSERT OR IGNORE INTO polytopes (id) VALUES (?1)').run(name);
  }

  // Chamber arms ("ks_4_11#2") share the base polytope's pool record.
  const baseName = name.split('#')[0];
  const record = readPoolRecord(pool, baseName);
  if (!record) {
    console.warn(`[ingest] polytope ${name} not found in pool file`);
    return;
  }
  const qD3 = (record.h11 + record.h21) / 2 + 1;
  db.prepare(
    `UPDATE polytopes
     SET h11 = ?2, h21 = ?3, favorable = ?4, q_d3 = ?5, vertices = ?6
     WHERE id = ?1`
  ).run(
    name,
    record.h11,
    record.h21,
    record.favorable ? 1 : 0,
    qD3,
    JSON.stringify(record.points)
  );
}

function refreshAggregates(db: Database, names: Set<string>) {
  const update = db.prepare(
    `UPDATE polytopes SET
      rounds = (SELECT COUNT(*) FROM rounds r WHERE r.polytope_id = polytopes.id),
      evals = COALESCE((SELECT r.evals FROM rounds r WHERE r.polytope_id = polytopes.id
                        ORDER BY r.id DESC LIMIT 1), 0),
      valid_seen = COALESCE((SELECT SUM(r.valid) FROM rounds r
                             WHERE r.polytope_id = polytopes.id), 0),
      best_fitness = (SELECT MAX(best) FROM (
        SELECT MAX(r.best_here) AS best FROM rounds r WHERE r.polytope_id = polytopes.id
        UNION ALL
        SELECT MAX(c.fitness) FROM candidates c WHERE c.polytope_id = polytopes.id
      )),
      updated_at = (SELECT MAX(ts) FROM (
        SELECT MAX(r.ts) AS ts FROM rounds r WHERE r.polytope_id = polytopes.id
        UNION ALL
        SELECT MAX(c.ts) FROM candidates c WHERE c.polytope_id = polytopes.id
      ))
     WHERE id = ?1`
  );
  for (const name of names) {
    update.run(name);
  }
}

function ingestCycle(db: Database, opts: Options, pool: PoolIndex) {
  const roundsPath = join(opts.runDir, 'rounds.jsonl');
  const candidatesPath = join(opts.runDir, 'candidates.jsonl');

  const rounds = tailJsonl(db, roundsPath, 'ingest_offset_rounds');
  const candidates = tailJsonl(db, candidatesPath, 'ingest_offset_candidates');

  const touched = new Set<string>();
  let roundCount = 0;
  let candidateCount = 0;

  const insertRound = db.prepare(
    `INSERT INTO rounds (round, polytope_id, evals, valid, best_here, global_best, live, ts)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)`
  );
  const upsertCandidate = db.prepare(
    `INSERT INTO candidates
      (polytope_id, k, m, fitness, tier, q_flux, g_s, w0, log10_abs_v0, cpl_w0, cpl_wa, round, ts)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
     ON CONFLICT(polytope_id, k, m) DO UPDATE SET
       fitness = excluded.fitness,
       tier = excluded.tier,
       q_flux = excluded.q_flux,
       g_s = excluded.g_s,
       w0 = excluded.w0,
       log10_abs_v0 = excluded.log10_abs_v0,
       cpl_w0 = excluded.cpl_w0,
       cpl_wa = excluded.cpl_wa,
       round = excluded.round,
       ts = excluded.ts`
  );

  const runAll = db.transaction(() => {
    for (const record of rounds.records) {
      if (!isRoundEvent(record)) {
        console.warn('[ingest] skipping malformed round event:', JSON.stringify(record).slice(0, 200));
        continue;
      }
      ensurePolytope(db, pool, record.polytope);
      insertRound.run(
        record.round,
        record.polytope,
        record.evals,
        record.valid ?? 0,
        record.best_here ?? null,
        record.global_best ?? null,
        record.live ?? null,
        record.ts ?? null
      );
      touched.add(record.polytope);
      roundCount++;
    }

    for (const record of candidates.records) {
      if (!isCandidateEvent(record)) {
        console.warn('[ingest] skipping malformed candidate event:', JSON.stringify(record).slice(0, 200));
        continue;
      }
      ensurePolytope(db, pool, record.polytope);
      const report = record.report ?? {};
      upsertCandidate.run(
        record.polytope,
        JSON.stringify(record.genome.k),
        JSON.stringify(record.genome.m),
        report.fitness ?? null,
        report.tier ?? null,
        report.q_flux ?? null,
        report.g_s ?? null,
        report.w0 ?? null,
        report.log10_abs_v0 ?? null,
        report.cpl_w0 ?? null,
        report.cpl_wa ?? null,
        record.round ?? null,
        record.ts ?? null
      );
      touched.add(record.polytope);
      candidateCount++;
    }

    refreshAggregates(db, touched);

    if (rounds.consumed) {
      setMeta(db, 'ingest_offset_rounds', String(rounds.newOffset));
    }
    if (candidates.consumed) {
      setMeta(db, 'ingest_offset_candidates', String(candidates.newOffset));
    }
    setMeta(db, 'pool_total', String(pool.offsets.size));
    setMeta(db, 'last_ingest_ts', String(Math.floor(Date.now() / 1000)));
  });
  runAll();

  if (roundCount > 0 || candidateCount > 0) {
    console.log(
      `[ingest] ${new Date().toISOString()} +${roundCount} rounds, +${candidateCount} candidates (${touched.size} polytopes touched)`
    );
  }
}

// ---------------------------------------------------------------------------
// Main

async function main() {
  const opts = resolveOptions();
  console.log('[ingest] run dir:', opts.runDir);
  console.log('[ingest] pool:   ', opts.pool);
  console.log('[ingest] db:     ', opts.db);

  const db = new Database(opts.db, { create: true });
  db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA busy_timeout = 30000;');
  ensureSchema(db);

  const pool = buildPoolIndex(opts.pool);

  for (;;) {
    try {
      ingestCycle(db, opts, pool);
    } catch (e) {
      console.error('[ingest] cycle failed:', e);
    }
    if (opts.once) break;
    await new Promise((resolve) => setTimeout(resolve, opts.intervalMs));
  }

  db.close();
}

main();
