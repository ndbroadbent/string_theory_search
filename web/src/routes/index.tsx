/**
 * Dashboard - cyrus-ga string-landscape search overview
 *
 * Status strip + DESI leaderboard (best candidate per polytope) + DESI
 * (cpl_w0, cpl_wa) scatter for valid candidates.
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { useCallback, useState } from 'react';
import {
  getRunStatus,
  getLeaderboard,
  getH21Values,
  getDesiCandidates,
} from '../server/ga';
import { DesiScatter } from '../components/desi/DesiScatter';
import type { LeaderboardEntry } from '../types';
import { TARGETS, sigmaDistance } from '../types';
import {
  fmtAgo,
  fmtInt,
  fmtNum,
  fmtSci,
  fmtSigned,
  sigmaColor,
  v0DeltaColor,
} from '../lib/format';

export const Route = createFileRoute('/')({
  component: Dashboard,
  loader: async () => {
    const [status, leaderboard, h21Values, desiCandidates] = await Promise.all([
      getRunStatus(),
      getLeaderboard({ data: { validOnly: false } }),
      getH21Values(),
      getDesiCandidates(),
    ]);
    return { status, leaderboard, h21Values, desiCandidates };
  },
});

function StatusCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
      <div className="text-xs text-gray-400">{label}</div>
      <div className="text-lg font-medium text-white font-mono">{value}</div>
      {sub && <div className="text-xs text-gray-500">{sub}</div>}
    </div>
  );
}

function FreshnessBadge({ lastIngestTs }: { lastIngestTs: number | null }) {
  const age = lastIngestTs == null ? null : Math.floor(Date.now() / 1000) - lastIngestTs;
  const color =
    age == null
      ? 'bg-slate-700 text-gray-400'
      : age < 60
        ? 'bg-green-900/60 text-green-300'
        : age < 600
          ? 'bg-yellow-900/60 text-yellow-300'
          : 'bg-red-900/60 text-red-300';
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-mono ${color}`}>
      ingest {fmtAgo(lastIngestTs)}
    </span>
  );
}

function Dashboard() {
  const { status, leaderboard: initialLeaderboard, h21Values, desiCandidates } =
    Route.useLoaderData();

  const [validOnly, setValidOnly] = useState(false);
  const [h21Filter, setH21Filter] = useState<number | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>(initialLeaderboard);

  const applyFilters = useCallback(
    async (nextValidOnly: boolean, nextH21: number | null) => {
      setValidOnly(nextValidOnly);
      setH21Filter(nextH21);
      const rows = await getLeaderboard({
        data: { validOnly: nextValidOnly, h21: nextH21 },
      });
      setLeaderboard(rows);
    },
    []
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">
              String Landscape Search
            </h1>
            <p className="text-gray-400 text-sm">
              cyrus-ga flux vacua search for dark-energy candidates · target log₁₀|V₀| ={' '}
              <span className="font-mono text-cyan-400">{TARGETS.log10AbsV0}</span>
            </p>
          </div>
          <div className="ml-auto">
            <FreshnessBadge lastIngestTs={status.lastIngestTs} />
          </div>
        </div>

        {/* Status strip */}
        <div className="mb-6 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatusCard
            label="Rounds"
            value={fmtInt(status.totalRounds)}
            sub={
              status.topArm
                ? `top arm ${status.topArm.polytope} · ${(status.topArm.share * 100).toFixed(1)}% of 10 min`
                : undefined
            }
          />
          <StatusCard label="Evaluations" value={fmtInt(status.totalEvals)} />
          <StatusCard
            label="Valid candidates"
            value={fmtInt(status.validSeen)}
            sub={`${fmtInt(status.validCandidates)} recorded improvements`}
          />
          <StatusCard
            label="Polytopes live / dead"
            value={`${fmtInt(status.live)} / ${fmtInt(status.dead)}`}
            sub={`${fmtInt(status.polytopesTracked)} of ${fmtInt(status.poolTotal)} searched`}
          />
          <StatusCard
            label="Evals / sec (10 min)"
            value={status.evalsPerSec == null ? '–' : status.evalsPerSec.toFixed(1)}
            sub={status.lastRoundTs ? `last round ${fmtAgo(status.lastRoundTs)}` : undefined}
          />
          <StatusCard
            label="Global best fitness"
            value={fmtNum(status.globalBest, 2)}
            sub={
              status.lastImprovementTs
                ? `improved ${fmtAgo(status.lastImprovementTs)}`
                : undefined
            }
          />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Leaderboard */}
          <div className="xl:col-span-2">
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
              <div className="px-4 py-3 flex flex-wrap items-center gap-4 border-b border-slate-700">
                <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wide">
                  DESI Leaderboard
                </h2>
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={validOnly}
                    onChange={(e) => applyFilters(e.target.checked, h21Filter)}
                    className="accent-cyan-500"
                  />
                  valid only
                </label>
                <select
                  value={h21Filter ?? ''}
                  onChange={(e) =>
                    applyFilters(validOnly, e.target.value === '' ? null : Number(e.target.value))
                  }
                  className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-gray-200 text-sm focus:outline-none focus:border-cyan-500"
                >
                  <option value="">all h21</option>
                  {h21Values.map((v) => (
                    <option key={v} value={v}>
                      h21 = {v}
                    </option>
                  ))}
                </select>
                <span className="ml-auto text-xs text-gray-500">
                  best candidate per polytope
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-slate-700/50 text-gray-300">
                      <th className="px-3 py-2 text-left font-medium">Polytope</th>
                      <th className="px-3 py-2 text-left font-medium">h11/h21</th>
                      <th className="px-3 py-2 text-left font-medium">K · M</th>
                      <th className="px-3 py-2 text-right font-medium">Q_flux</th>
                      <th className="px-3 py-2 text-right font-medium">g_s</th>
                      <th className="px-3 py-2 text-right font-medium">|W₀|</th>
                      <th className="px-3 py-2 text-right font-medium">log₁₀|V₀|</th>
                      <th className="px-3 py-2 text-right font-medium">w₀ / wₐ (σ)</th>
                      <th className="px-3 py-2 text-right font-medium">Fitness</th>
                      <th className="px-3 py-2 text-left font-medium">Tier</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.map((entry) => {
                      const v0Delta =
                        entry.log10_abs_v0 == null
                          ? null
                          : entry.log10_abs_v0 - TARGETS.log10AbsV0;
                      const sigW0 = sigmaDistance(entry.cpl_w0, TARGETS.desiW0);
                      const sigWa = sigmaDistance(entry.cpl_wa, TARGETS.desiWa);
                      return (
                        <tr
                          key={entry.id}
                          className="border-t border-slate-700/50 hover:bg-slate-700/30"
                        >
                          <td className="px-3 py-2">
                            <Link
                              to="/polytope/$id"
                              params={{ id: entry.polytope_id }}
                              className="text-cyan-400 hover:text-cyan-300 font-mono"
                            >
                              {entry.polytope_id}
                            </Link>
                          </td>
                          <td className="px-3 py-2 text-gray-300 font-mono whitespace-nowrap">
                            {entry.h11 ?? '–'}/{entry.h21 ?? '–'}
                          </td>
                          <td className="px-3 py-2 text-gray-400 font-mono whitespace-nowrap">
                            [{entry.k.join(',')}] · [{entry.m.join(',')}]
                          </td>
                          <td className="px-3 py-2 text-right text-gray-300 font-mono whitespace-nowrap">
                            {fmtNum(entry.q_flux, 1)}
                            <span className="text-gray-500"> / {fmtNum(entry.q_d3, 0)}</span>
                          </td>
                          <td className="px-3 py-2 text-right text-gray-300 font-mono">
                            {entry.g_s == null ? '–' : entry.g_s.toFixed(5)}
                          </td>
                          <td className="px-3 py-2 text-right text-gray-300 font-mono">
                            {fmtSci(entry.w0)}
                          </td>
                          <td className="px-3 py-2 text-right whitespace-nowrap">
                            {entry.log10_abs_v0 == null ? (
                              <span className="text-gray-500">–</span>
                            ) : (
                              <>
                                <span className="text-gray-300 font-mono">
                                  {entry.log10_abs_v0.toFixed(1)}
                                </span>{' '}
                                <span
                                  className={`px-1.5 py-0.5 rounded font-mono ${v0DeltaColor(v0Delta)}`}
                                >
                                  {fmtSigned(v0Delta)}
                                </span>
                              </>
                            )}
                          </td>
                          <td className="px-3 py-2 text-right font-mono whitespace-nowrap">
                            {entry.cpl_w0 == null && entry.cpl_wa == null ? (
                              <span className="text-gray-500">–</span>
                            ) : (
                              <>
                                <span className={sigmaColor(sigW0)}>
                                  {fmtNum(entry.cpl_w0)} ({fmtSigned(sigW0)}σ)
                                </span>
                                <br />
                                <span className={sigmaColor(sigWa)}>
                                  {fmtNum(entry.cpl_wa)} ({fmtSigned(sigWa)}σ)
                                </span>
                              </>
                            )}
                          </td>
                          <td className="px-3 py-2 text-right text-gray-200 font-mono">
                            {fmtNum(entry.fitness, 2)}
                          </td>
                          <td className="px-3 py-2">
                            {entry.tier === 'valid' ? (
                              <span className="px-1.5 py-0.5 rounded bg-green-900/60 text-green-300 border border-green-700/50">
                                valid
                              </span>
                            ) : (
                              <span
                                className="text-gray-500 max-w-[16rem] truncate inline-block align-bottom"
                                title={entry.tier ?? undefined}
                              >
                                {entry.tier ?? '–'}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {leaderboard.length === 0 && (
                  <div className="px-4 py-8 text-center text-gray-400 text-sm">
                    {validOnly
                      ? 'No valid candidates yet. Uncheck "valid only" to see the best attempts per polytope.'
                      : 'No candidates ingested yet. Is the ingester running?'}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* DESI plane */}
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
            <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wide mb-2">
              DESI Plane (valid candidates)
            </h2>
            {desiCandidates.length > 0 ? (
              <DesiScatter candidates={desiCandidates} />
            ) : (
              <div className="h-72 flex items-center justify-center text-gray-500 text-sm text-center px-4">
                No valid candidates with CPL parameters yet.
                <br />
                The DESI 1σ box and Λ will populate as the GA finds vacua.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
