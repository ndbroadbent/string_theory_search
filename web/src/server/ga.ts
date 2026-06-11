/**
 * Server functions for cyrus-ga search results
 *
 * Reads the dashboard SQLite database filled by web/scripts/ingest.ts.
 */

import { createServerFn } from '@tanstack/react-start';
import { withDb } from './db';
import type {
  RunStatus,
  Candidate,
  LeaderboardEntry,
  PolytopeRow,
  PolytopeDetail,
  Verification,
} from '../types';

function parseIntArray(json: string | null): number[] {
  if (!json) return [];
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function rowToCandidate(row: Record<string, unknown>): Candidate {
  return {
    id: row.id as number,
    polytope_id: row.polytope_id as string,
    k: parseIntArray(row.k as string | null),
    m: parseIntArray(row.m as string | null),
    fitness: row.fitness as number | null,
    tier: row.tier as string | null,
    q_flux: row.q_flux as number | null,
    g_s: row.g_s as number | null,
    w0: row.w0 as number | null,
    log10_abs_v0: row.log10_abs_v0 as number | null,
    cpl_w0: row.cpl_w0 as number | null,
    cpl_wa: row.cpl_wa as number | null,
    round: row.round as number | null,
    ts: row.ts as number | null,
  };
}

const EMPTY_STATUS: RunStatus = {
  totalRounds: 0,
  totalEvals: 0,
  validSeen: 0,
  validCandidates: 0,
  polytopesTracked: 0,
  poolTotal: null,
  live: null,
  dead: null,
  globalBest: null,
  evalsPerSec: null,
  lastRoundTs: null,
  lastIngestTs: null,
};

/** Aggregate run status for the dashboard status strip */
export const getRunStatus = createServerFn({ method: 'GET' }).handler(
  async (): Promise<RunStatus> => {
    return withDb((db) => {
      const totals = db
        .prepare(
          `SELECT
            (SELECT COUNT(*) FROM rounds) AS total_rounds,
            (SELECT COALESCE(SUM(evals), 0) FROM polytopes) AS total_evals,
            (SELECT COALESCE(SUM(valid_seen), 0) FROM polytopes) AS valid_seen,
            (SELECT COUNT(*) FROM candidates WHERE tier = 'valid') AS valid_candidates,
            (SELECT COUNT(*) FROM polytopes) AS polytopes_tracked,
            (SELECT MAX(ts) FROM rounds) AS last_round_ts`
        )
        .get() as Record<string, number | null>;

      const latest = db
        .prepare('SELECT live, global_best FROM rounds ORDER BY id DESC LIMIT 1')
        .get() as { live: number | null; global_best: number | null } | null;

      const meta = db
        .prepare(
          `SELECT key, value FROM run_meta WHERE key IN ('pool_total', 'last_ingest_ts')`
        )
        .all() as Array<{ key: string; value: string }>;
      const metaMap = Object.fromEntries(meta.map((m) => [m.key, m.value]));
      const poolTotal = metaMap.pool_total ? Number(metaMap.pool_total) : null;
      const lastIngestTs = metaMap.last_ingest_ts ? Number(metaMap.last_ingest_ts) : null;

      // evals/sec over the last 10 minutes of round events. `evals` is
      // cumulative per polytope, so the rate is the change in the per-polytope
      // maxima across the window.
      let evalsPerSec: number | null = null;
      const lastRoundTs = totals.last_round_ts;
      if (lastRoundTs != null) {
        const windowSecs = 600;
        const cutoff = lastRoundTs - windowSecs;
        const win = db
          .prepare(
            `SELECT
              (SELECT COALESCE(SUM(emax), 0) FROM
                (SELECT MAX(evals) AS emax FROM rounds GROUP BY polytope_id)) AS now_total,
              (SELECT COALESCE(SUM(emax), 0) FROM
                (SELECT MAX(evals) AS emax FROM rounds WHERE ts < ?1 GROUP BY polytope_id)) AS cutoff_total,
              (SELECT MIN(ts) FROM rounds) AS min_ts`
          )
          .get(cutoff) as { now_total: number; cutoff_total: number; min_ts: number };
        const span = Math.max(1, Math.min(windowSecs, lastRoundTs - win.min_ts));
        evalsPerSec = (win.now_total - win.cutoff_total) / span;
      }

      const live = latest?.live ?? null;
      return {
        totalRounds: totals.total_rounds ?? 0,
        totalEvals: totals.total_evals ?? 0,
        validSeen: totals.valid_seen ?? 0,
        validCandidates: totals.valid_candidates ?? 0,
        polytopesTracked: totals.polytopes_tracked ?? 0,
        poolTotal,
        live,
        dead: poolTotal != null && live != null ? poolTotal - live : null,
        globalBest: latest?.global_best ?? null,
        evalsPerSec,
        lastRoundTs,
        lastIngestTs,
      };
    }, EMPTY_STATUS);
  }
);

/** DESI leaderboard: best candidate per polytope, best fitness first */
export const getLeaderboard = createServerFn({ method: 'GET' })
  .inputValidator(
    (data: { validOnly?: boolean; h21?: number | null; limit?: number }) => data
  )
  .handler(
    async ({
      data: { validOnly = false, h21 = null, limit = 200 },
    }): Promise<LeaderboardEntry[]> => {
      return withDb((db) => {
        const rows = db
          .prepare(
            `SELECT * FROM (
              SELECT c.*, p.h11, p.h21, p.q_d3,
                ROW_NUMBER() OVER (
                  PARTITION BY c.polytope_id ORDER BY c.fitness DESC
                ) AS rn
              FROM candidates c
              LEFT JOIN polytopes p ON p.id = c.polytope_id
              WHERE (?1 = 0 OR c.tier = 'valid')
                AND (?2 IS NULL OR p.h21 = ?2)
            ) WHERE rn = 1
            ORDER BY fitness DESC
            LIMIT ?3`
          )
          .all(validOnly ? 1 : 0, h21, limit) as Record<string, unknown>[];

        return rows.map((row) => ({
          ...rowToCandidate(row),
          h11: row.h11 as number | null,
          h21: row.h21 as number | null,
          q_d3: row.q_d3 as number | null,
        }));
      }, []);
    }
  );

/** Distinct h21 values present (for the leaderboard filter) */
export const getH21Values = createServerFn({ method: 'GET' }).handler(
  async (): Promise<number[]> => {
    return withDb((db) => {
      const rows = db
        .prepare(
          'SELECT DISTINCT h21 FROM polytopes WHERE h21 IS NOT NULL ORDER BY h21'
        )
        .all() as Array<{ h21: number }>;
      return rows.map((r) => r.h21);
    }, []);
  }
);

/** Valid candidates in the (cpl_w0, cpl_wa) plane for the DESI scatter */
export const getDesiCandidates = createServerFn({ method: 'GET' }).handler(
  async (): Promise<Candidate[]> => {
    return withDb((db) => {
      const rows = db
        .prepare(
          `SELECT * FROM candidates
           WHERE tier = 'valid' AND cpl_w0 IS NOT NULL AND cpl_wa IS NOT NULL
           ORDER BY fitness DESC
           LIMIT 2000`
        )
        .all() as Record<string, unknown>[];
      return rows.map(rowToCandidate);
    }, []);
  }
);

/** Polytopes list (sorted by best_fitness, then valid_seen) */
export const getPolytopesList = createServerFn({ method: 'GET' })
  .inputValidator((data: { limit?: number; offset?: number }) => data)
  .handler(
    async ({
      data: { limit = 200, offset = 0 },
    }): Promise<{ total: number; polytopes: PolytopeRow[] }> => {
      return withDb(
        (db) => {
          const countRow = db
            .prepare('SELECT COUNT(*) AS count FROM polytopes')
            .get() as { count: number };

          const rows = db
            .prepare(
              `SELECT id, h11, h21, favorable, q_d3, rounds, evals, valid_seen,
                      best_fitness, updated_at
               FROM polytopes
               ORDER BY valid_seen DESC, best_fitness DESC NULLS LAST, id
               LIMIT ?1 OFFSET ?2`
            )
            .all(limit, offset) as unknown as PolytopeRow[];

          return { total: countRow.count, polytopes: rows };
        },
        { total: 0, polytopes: [] }
      );
    }
  );

/** Full detail for a single polytope (by pool name) */
export const getPolytopeDetail = createServerFn({ method: 'GET' })
  .inputValidator((data: { id: string }) => data)
  .handler(
    async ({
      data: { id },
    }): Promise<{
      polytope: PolytopeDetail | null;
      candidates: Candidate[];
      verifications: Verification[];
    }> => {
      return withDb(
        (db) => {
          const row = db
            .prepare('SELECT * FROM polytopes WHERE id = ?1')
            .get(id) as Record<string, unknown> | null;

          let polytope: PolytopeDetail | null = null;
          if (row) {
            let vertices: number[][] | null = null;
            try {
              const parsed = JSON.parse((row.vertices as string) || 'null');
              if (Array.isArray(parsed)) vertices = parsed;
            } catch {
              vertices = null;
            }
            polytope = {
              id: row.id as string,
              h11: row.h11 as number | null,
              h21: row.h21 as number | null,
              favorable: row.favorable as number | null,
              q_d3: row.q_d3 as number | null,
              rounds: (row.rounds as number) ?? 0,
              evals: (row.evals as number) ?? 0,
              valid_seen: (row.valid_seen as number) ?? 0,
              best_fitness: row.best_fitness as number | null,
              updated_at: row.updated_at as number | null,
              vertices,
            };
          }

          const candidateRows = db
            .prepare(
              'SELECT * FROM candidates WHERE polytope_id = ?1 ORDER BY fitness DESC LIMIT 500'
            )
            .all(id) as Record<string, unknown>[];

          const verifications = db
            .prepare(
              'SELECT * FROM verifications WHERE polytope_id = ?1 ORDER BY ts DESC LIMIT 100'
            )
            .all(id) as unknown as Verification[];

          return {
            polytope,
            candidates: candidateRows.map(rowToCandidate),
            verifications,
          };
        },
        { polytope: null, candidates: [], verifications: [] }
      );
    }
  );
