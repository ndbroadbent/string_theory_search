/**
 * Correlations - Pearson correlation of every heuristic metric against
 * cyrus-ga search outcomes:
 *   - fertility (polytope has produced any valid candidate)
 *   - best fitness
 *   - best |log10|V0| - target| (distance to the dark-energy scale)
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { useMemo, useState } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { getCorrelations } from '../server/heuristics';
import { getLabel } from './heuristics';
import type { CorrelationRow } from '../types';

export const Route = createFileRoute('/correlations')({
  component: CorrelationsPage,
  loader: async () => {
    return getCorrelations();
  },
});

type OutcomeKey = 'fertility' | 'bestFitness' | 'v0Distance';

const OUTCOMES: Record<OutcomeKey, { label: string; description: string }> = {
  fertility: {
    label: 'Fertility',
    description: 'valid_seen > 0 (any valid candidate found)',
  },
  bestFitness: {
    label: 'Best Fitness',
    description: 'polytopes.best_fitness',
  },
  v0Distance: {
    label: '|log₁₀|V₀| + 121.54|',
    description: 'best distance to the dark-energy scale (lower is better)',
  },
};

function corrColor(r: number | null): string {
  if (r == null) return 'text-gray-600';
  const abs = Math.abs(r);
  if (abs >= 0.5) return r > 0 ? 'text-green-400' : 'text-red-400';
  if (abs >= 0.25) return r > 0 ? 'text-green-500/80' : 'text-red-500/80';
  return 'text-gray-400';
}

function CorrBar({ r }: { r: number | null }) {
  if (r == null) return null;
  const pct = Math.min(100, Math.abs(r) * 100);
  return (
    <div className="w-16 h-1.5 bg-slate-700 rounded overflow-hidden inline-block align-middle">
      <div
        className={`h-full ${r > 0 ? 'bg-green-500' : 'bg-red-500'}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function CorrelationsPage() {
  const { correlations, points } = Route.useLoaderData();

  const [sortBy, setSortBy] = useState<OutcomeKey>('bestFitness');
  const [selected, setSelected] = useState<{ metric: string; outcome: OutcomeKey } | null>(null);

  const sorted = useMemo(() => {
    return [...correlations].sort(
      (a, b) => Math.abs(b[sortBy] ?? 0) - Math.abs(a[sortBy] ?? 0)
    );
  }, [correlations, sortBy]);

  const scatterData = useMemo(() => {
    if (!selected) return [];
    return points
      .map((p) => ({
        polytope_id: p.polytope_id,
        x: p.metrics[selected.metric],
        y: p.outcomes[selected.outcome],
      }))
      .filter((p): p is { polytope_id: string; x: number; y: number } =>
        p.x != null && p.y != null
      );
  }, [points, selected]);

  const selectedRow: CorrelationRow | undefined = selected
    ? correlations.find((c) => c.metric === selected.metric)
    : undefined;

  const headerCell = (key: OutcomeKey) => (
    <th
      className={`px-3 py-2 text-right font-medium cursor-pointer select-none hover:text-white ${
        sortBy === key ? 'text-cyan-400' : ''
      }`}
      onClick={() => setSortBy(key)}
      title={OUTCOMES[key].description}
    >
      {OUTCOMES[key].label} {sortBy === key ? '↓' : ''}
    </th>
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white mb-2">Heuristic ↔ Outcome Correlations</h1>
          <p className="text-gray-400">
            Pearson r of each geometric metric against search outcomes across{' '}
            {points.length} polytopes (sorted by |r|, click a column to re-sort, click a row
            to plot)
          </p>
        </div>

        {correlations.length === 0 ? (
          <div className="bg-slate-800/50 rounded-lg p-8 border border-slate-700 text-center text-gray-400">
            No data yet: requires both heuristics rows (Rust binary) and ingested polytopes
            (ingest daemon).
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Correlation table */}
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
              <div className="overflow-y-auto max-h-[calc(100vh-14rem)]">
                <table className="w-full text-xs">
                  <thead className="sticky top-0">
                    <tr className="bg-slate-700 text-gray-300">
                      <th className="px-3 py-2 text-left font-medium">Metric</th>
                      {headerCell('fertility')}
                      {headerCell('bestFitness')}
                      {headerCell('v0Distance')}
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((row) => (
                      <tr
                        key={row.metric}
                        onClick={() => setSelected({ metric: row.metric, outcome: sortBy })}
                        className={`border-t border-slate-700/50 cursor-pointer hover:bg-slate-700/30 ${
                          selected?.metric === row.metric ? 'bg-slate-700/40' : ''
                        }`}
                      >
                        <td className="px-3 py-2 text-gray-200">{getLabel(row.metric)}</td>
                        {(['fertility', 'bestFitness', 'v0Distance'] as const).map((key) => (
                          <td key={key} className="px-3 py-2 text-right whitespace-nowrap">
                            <span className={`font-mono mr-2 ${corrColor(row[key])}`}>
                              {row[key] == null ? '–' : row[key].toFixed(3)}
                            </span>
                            <CorrBar r={row[key]} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Scatter of selected metric vs outcome */}
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              {selected && selectedRow ? (
                <>
                  <div className="mb-2 flex items-center gap-3 flex-wrap">
                    <h3 className="text-sm font-medium text-gray-200">
                      {getLabel(selected.metric)} vs {OUTCOMES[selected.outcome].label}
                    </h3>
                    <span className={`font-mono text-sm ${corrColor(selectedRow[selected.outcome])}`}>
                      r = {selectedRow[selected.outcome]?.toFixed(3) ?? '–'}
                    </span>
                    <span className="text-xs text-gray-500">n = {scatterData.length}</span>
                    <select
                      value={selected.outcome}
                      onChange={(e) =>
                        setSelected({ metric: selected.metric, outcome: e.target.value as OutcomeKey })
                      }
                      className="ml-auto bg-slate-700 border border-slate-600 rounded px-2 py-1 text-gray-200 text-xs focus:outline-none focus:border-cyan-500"
                    >
                      {Object.entries(OUTCOMES).map(([key, o]) => (
                        <option key={key} value={key}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <ResponsiveContainer width="100%" height={460}>
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis
                        type="number"
                        dataKey="x"
                        domain={['auto', 'auto']}
                        stroke="#9ca3af"
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        label={{
                          value: getLabel(selected.metric),
                          position: 'bottom',
                          fill: '#9ca3af',
                          fontSize: 12,
                        }}
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        domain={['auto', 'auto']}
                        stroke="#9ca3af"
                        tick={{ fill: '#9ca3af', fontSize: 11 }}
                        label={{
                          value: OUTCOMES[selected.outcome].label,
                          angle: -90,
                          position: 'left',
                          fill: '#9ca3af',
                          fontSize: 12,
                        }}
                      />
                      <Tooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        content={({ payload }) => {
                          if (!payload?.length) return null;
                          const d = payload[0].payload as {
                            polytope_id: string;
                            x: number;
                            y: number;
                          };
                          return (
                            <div className="bg-slate-800 border border-slate-600 rounded p-2 text-xs">
                              <Link
                                to="/polytope/$id"
                                params={{ id: d.polytope_id }}
                                className="text-cyan-400 font-mono"
                              >
                                {d.polytope_id}
                              </Link>
                              <div className="text-gray-300 mt-1">
                                {getLabel(selected.metric)}: {d.x.toFixed(4)}
                              </div>
                              <div className="text-gray-300">
                                {OUTCOMES[selected.outcome].label}: {d.y.toFixed(4)}
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Scatter data={scatterData} fill="#22d3ee" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </>
              ) : (
                <div className="h-full min-h-72 flex items-center justify-center text-gray-400 text-sm">
                  Click a metric row to plot it against the selected outcome
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
