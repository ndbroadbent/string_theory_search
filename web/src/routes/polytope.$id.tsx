/**
 * Polytope detail - 4D viewer + cyrus-ga search results + heuristics
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { Suspense, lazy, useEffect, useState } from 'react';
import { getPolytopeDetail } from '../server/ga';
import { getHeuristicsForPolytope } from '../server/heuristics';
import { PolytopeControls } from '../components/polytope/PolytopeControls';
import { TARGETS, sigmaDistance } from '../types';
import {
  fmtInt,
  fmtNum,
  fmtSci,
  fmtSigned,
  sigmaColor,
  v0DeltaColor,
} from '../lib/format';

// Lazy load Three.js component to avoid SSR issues
const Polytope4D = lazy(() =>
  import('../components/polytope/Polytope4D').then((m) => ({
    default: m.Polytope4D,
  }))
);

export const Route = createFileRoute('/polytope/$id')({
  component: PolytopeExplorer,
  loader: async ({ params }) => {
    const [detail, heuristics] = await Promise.all([
      getPolytopeDetail({ data: { id: params.id } }),
      getHeuristicsForPolytope({ data: { polytopeId: params.id } }),
    ]);
    return { id: params.id, ...detail, heuristics };
  },
});

// Group heuristics by category for display
const HEURISTIC_CATEGORIES = {
  Basic: ['h11', 'h21', 'vertex_count'],
  Circularity: ['sphericity', 'inertia_isotropy'],
  Chirality: ['chirality_optimal', 'chirality_x', 'chirality_y', 'chirality_z', 'chirality_w', 'handedness_det'],
  Symmetry: ['symmetry_x', 'symmetry_y', 'symmetry_z', 'symmetry_w'],
  Flatness: ['flatness_3d', 'flatness_2d', 'intrinsic_dim_estimate'],
  Shape: ['spikiness', 'max_exposure', 'conformity_ratio', 'distance_kurtosis', 'loner_score'],
  Statistics: ['coord_mean', 'coord_median', 'coord_std', 'coord_skewness', 'coord_kurtosis', 'shannon_entropy', 'joint_entropy'],
  Compression: ['compression_ratio', 'sorted_compression_ratio', 'sort_compression_gain'],
  Patterns: ['phi_ratio_count', 'fibonacci_count', 'zero_count', 'one_count', 'prime_count'],
  Outlier: ['outlier_score', 'outlier_max_zscore', 'outlier_count_2sigma', 'outlier_count_3sigma', 'outlier_max_dim'],
};

function formatValue(value: unknown): string {
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toString();
    return value.toFixed(4);
  }
  if (typeof value === 'string') return value;
  if (value == null) return '–';
  return JSON.stringify(value);
}

/** Render children only after mount (Three.js cannot load during SSR) */
function ClientOnly({ children, fallback }: { children: React.ReactNode; fallback: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return <>{mounted ? children : fallback}</>;
}

function PolytopeExplorer() {
  const { id, polytope, candidates, verifications, heuristics } = Route.useLoaderData();

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-center gap-4 flex-wrap">
          <Link to="/polytopes" className="text-gray-400 hover:text-white transition-colors">
            ← Polytopes
          </Link>
          <h1 className="text-2xl font-bold text-white font-mono">{id}</h1>
          {polytope && (
            <span className="text-gray-400 text-sm">
              h11={polytope.h11 ?? '–'}, h21={polytope.h21 ?? '–'}
              {polytope.favorable != null && (polytope.favorable ? ', favorable' : ', non-favorable')}
            </span>
          )}
          {polytope && polytope.valid_seen > 0 && (
            <span className="px-2 py-1 rounded bg-green-900/60 text-green-300 text-sm font-mono">
              {fmtInt(polytope.valid_seen)} valid
            </span>
          )}
        </div>

        {!polytope ? (
          <div className="bg-slate-800/50 rounded-lg p-8 border border-slate-700 text-center text-gray-400">
            Polytope <span className="font-mono">{id}</span> not found in the database. It may
            not have been searched yet.
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left column - 3D Visualization + candidates */}
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
                <div className="aspect-[4/3]">
                  <ClientOnly
                    fallback={
                      <div className="w-full h-full flex items-center justify-center text-gray-400">
                        Loading 3D view...
                      </div>
                    }
                  >
                    <Suspense
                      fallback={
                        <div className="w-full h-full flex items-center justify-center text-gray-400">
                          Loading 3D view...
                        </div>
                      }
                    >
                      {polytope.vertices && polytope.vertices.length > 0 ? (
                        <Polytope4D vertices={polytope.vertices} />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-gray-400">
                          No vertex data available
                        </div>
                      )}
                    </Suspense>
                  </ClientOnly>
                </div>
              </div>

              <PolytopeControls />

              {/* Candidates */}
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-700">
                  <h3 className="text-sm font-medium text-gray-300">
                    Best candidates ({candidates.length})
                  </h3>
                </div>
                <div className="overflow-x-auto max-h-96 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-slate-700/50 text-gray-300 sticky top-0">
                        <th className="px-3 py-2 text-left font-medium">Round</th>
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
                      {candidates.map((c) => {
                        const v0Delta =
                          c.log10_abs_v0 == null ? null : c.log10_abs_v0 - TARGETS.log10AbsV0;
                        const sigW0 = sigmaDistance(c.cpl_w0, TARGETS.desiW0);
                        const sigWa = sigmaDistance(c.cpl_wa, TARGETS.desiWa);
                        return (
                          <tr key={c.id} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                            <td className="px-3 py-2 text-gray-400 font-mono">{c.round ?? '–'}</td>
                            <td className="px-3 py-2 text-gray-300 font-mono whitespace-nowrap">
                              [{c.k.join(',')}] · [{c.m.join(',')}]
                            </td>
                            <td className="px-3 py-2 text-right text-gray-300 font-mono">
                              {fmtNum(c.q_flux, 1)}
                              <span className="text-gray-500"> / {fmtNum(polytope.q_d3, 0)}</span>
                            </td>
                            <td className="px-3 py-2 text-right text-gray-300 font-mono">
                              {c.g_s == null ? '–' : c.g_s.toFixed(5)}
                            </td>
                            <td className="px-3 py-2 text-right text-gray-300 font-mono">{fmtSci(c.w0)}</td>
                            <td className="px-3 py-2 text-right whitespace-nowrap">
                              {c.log10_abs_v0 == null ? (
                                <span className="text-gray-500">–</span>
                              ) : (
                                <>
                                  <span className="text-gray-300 font-mono">
                                    {c.log10_abs_v0.toFixed(1)}
                                  </span>{' '}
                                  <span className={`px-1.5 py-0.5 rounded font-mono ${v0DeltaColor(v0Delta)}`}>
                                    {fmtSigned(v0Delta)}
                                  </span>
                                </>
                              )}
                            </td>
                            <td className="px-3 py-2 text-right font-mono whitespace-nowrap">
                              {c.cpl_w0 == null && c.cpl_wa == null ? (
                                <span className="text-gray-500">–</span>
                              ) : (
                                <>
                                  <span className={sigmaColor(sigW0)}>
                                    {fmtNum(c.cpl_w0)} ({fmtSigned(sigW0)}σ)
                                  </span>
                                  <br />
                                  <span className={sigmaColor(sigWa)}>
                                    {fmtNum(c.cpl_wa)} ({fmtSigned(sigWa)}σ)
                                  </span>
                                </>
                              )}
                            </td>
                            <td className="px-3 py-2 text-right text-gray-200 font-mono">
                              {fmtNum(c.fitness, 2)}
                            </td>
                            <td className="px-3 py-2">
                              {c.tier === 'valid' ? (
                                <span className="px-1.5 py-0.5 rounded bg-green-900/60 text-green-300">
                                  valid
                                </span>
                              ) : (
                                <span
                                  className="text-gray-500 max-w-[14rem] truncate inline-block align-bottom"
                                  title={c.tier ?? undefined}
                                >
                                  {c.tier ?? '–'}
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  {candidates.length === 0 && (
                    <div className="px-4 py-6 text-center text-gray-400 text-sm">
                      No candidate improvements recorded yet.
                    </div>
                  )}
                </div>
              </div>

              {/* Verifications */}
              {verifications.length > 0 && (
                <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-700">
                    <h3 className="text-sm font-medium text-gray-300">
                      Verifications ({verifications.length})
                    </h3>
                  </div>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-slate-700/50 text-gray-300">
                        <th className="px-3 py-2 text-left font-medium">Status</th>
                        <th className="px-3 py-2 text-left font-medium">K · M</th>
                        <th className="px-3 py-2 text-left font-medium">Blocker</th>
                        <th className="px-3 py-2 text-left font-medium">Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {verifications.map((v) => (
                        <tr key={v.id} className="border-t border-slate-700/50">
                          <td className="px-3 py-2 text-gray-200 font-mono">{v.status ?? '–'}</td>
                          <td className="px-3 py-2 text-gray-400 font-mono">
                            {v.k ?? '–'} · {v.m ?? '–'}
                          </td>
                          <td className="px-3 py-2 text-gray-400">{v.blocker ?? '–'}</td>
                          <td className="px-3 py-2 text-gray-500">{v.details ?? '–'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Right column - stats + heuristics */}
            <div className="space-y-4 max-h-[calc(100vh-8rem)] overflow-y-auto">
              {/* Search stats */}
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-cyan-400 mb-3">Search Stats</h3>
                <div className="space-y-1 text-xs">
                  {[
                    ['Rounds', fmtInt(polytope.rounds)],
                    ['Evaluations', fmtInt(polytope.evals)],
                    ['Valid seen', fmtInt(polytope.valid_seen)],
                    ['Best fitness', fmtNum(polytope.best_fitness, 4)],
                    ['Q_D3 bound', fmtNum(polytope.q_d3, 1)],
                  ].map(([label, value]) => (
                    <div key={label} className="flex justify-between">
                      <span className="text-gray-400">{label}:</span>
                      <span className="text-gray-200 font-mono">{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Heuristics */}
              {heuristics ? (
                Object.entries(HEURISTIC_CATEGORIES).map(([category, keys]) => (
                  <div key={category} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                    <h3 className="text-sm font-medium text-cyan-400 mb-3">{category}</h3>
                    <div className="space-y-1 text-xs">
                      {keys.map((key) => {
                        const value = (heuristics as Record<string, unknown>)[key];
                        if (value === undefined) return null;
                        return (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-400">{key}:</span>
                            <span className="text-gray-200 font-mono">{formatValue(value)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))
              ) : (
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                  <p className="text-gray-400 text-sm">No heuristics computed for this polytope yet.</p>
                  <p className="text-gray-500 text-xs mt-2 font-mono">
                    cargo run --release --bin heuristics -- --pool &lt;pool.jsonl&gt; --db &lt;db&gt;
                  </p>
                </div>
              )}

              {/* Vertex data */}
              {polytope.vertices && (
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                  <h3 className="text-sm font-medium text-gray-300 mb-3">
                    Points ({polytope.vertices.length} in 4D)
                  </h3>
                  <div className="bg-slate-900/50 rounded p-2 max-h-48 overflow-y-auto font-mono text-xs">
                    {polytope.vertices.map((v, i) => (
                      <div key={i} className="text-gray-300">
                        [{v.map((c) => c.toString().padStart(3)).join(', ')}]
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
