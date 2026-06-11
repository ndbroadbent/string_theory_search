/**
 * Heuristics Explorer - Scatter plot across geometric metric dimensions
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { useState, useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { getHeuristics } from '../server/heuristics';
import type { PolytopeHeuristics } from '../types';

export const Route = createFileRoute('/heuristics')({
  component: HeuristicsExplorer,
  loader: async () => {
    const heuristics = await getHeuristics();
    return { heuristics };
  },
});

// Human-readable labels for metric columns; metrics without an entry fall
// back to their column name.
const DIMENSION_LABELS: Record<string, string> = {
  h11: 'h11',
  h21: 'h21',
  vertex_count: 'Vertex Count',
  sphericity: 'Sphericity (π-ness)',
  inertia_isotropy: 'Inertia Isotropy',
  chirality_optimal: 'Chirality (Optimal)',
  chirality_x: 'Chirality X',
  chirality_y: 'Chirality Y',
  chirality_z: 'Chirality Z',
  chirality_w: 'Chirality W',
  handedness_det: 'Handedness',
  symmetry_x: 'Symmetry X',
  symmetry_y: 'Symmetry Y',
  symmetry_z: 'Symmetry Z',
  symmetry_w: 'Symmetry W',
  flatness_3d: 'Flatness 3D',
  flatness_2d: 'Flatness 2D',
  intrinsic_dim_estimate: 'Intrinsic Dimension',
  spikiness: 'Spikiness',
  max_exposure: 'Max Exposure',
  conformity_ratio: 'Conformity Ratio',
  distance_kurtosis: 'Distance Kurtosis',
  loner_score: 'Loner Score',
  coord_mean: 'Coord Mean',
  coord_median: 'Coord Median',
  coord_std: 'Coord Std Dev',
  coord_skewness: 'Coord Skewness',
  coord_kurtosis: 'Coord Kurtosis',
  shannon_entropy: 'Shannon Entropy',
  joint_entropy: 'Joint Entropy',
  compression_ratio: 'Compression Ratio',
  sorted_compression_ratio: 'Sorted Compression Ratio',
  sort_compression_gain: 'Sort Compression Gain',
  phi_ratio_count: 'Golden Ratio Count',
  fibonacci_count: 'Fibonacci Count',
  zero_count: 'Zero Count',
  one_count: 'One Count',
  prime_count: 'Prime Count',
  outlier_score: 'Outlier Score',
  outlier_max_zscore: 'Max Z-Score',
  outlier_count_2sigma: '2σ Outlier Count',
  outlier_count_3sigma: '3σ Outlier Count',
};

export function getLabel(key: string): string {
  return DIMENSION_LABELS[key] ?? key;
}

// Keys to exclude from dimension selectors (non-numeric identifiers)
const EXCLUDE_DIMENSIONS = new Set(['polytope_id', 'outlier_max_dim']);

// Extract numeric dimensions from the loaded heuristics
function extractNumericDimensions(rows: PolytopeHeuristics[]): string[] {
  const dimensions = new Set<string>();
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) {
      if (EXCLUDE_DIMENSIONS.has(key)) continue;
      if (typeof value === 'number' && !Number.isNaN(value)) {
        dimensions.add(key);
      }
    }
  }
  return [...dimensions].sort((a, b) => getLabel(a).localeCompare(getLabel(b)));
}

function getValue(h: PolytopeHeuristics, key: string): number {
  const value = (h as Record<string, unknown>)[key];
  return typeof value === 'number' ? value : 0;
}

function getColor(value: number, min: number, max: number): string {
  const normalized = (value - min) / (max - min + 1e-10);
  // Blue -> Cyan -> Green -> Yellow -> Red
  const hue = (1 - normalized) * 240;
  return `hsl(${hue}, 80%, 50%)`;
}

function HeuristicsExplorer() {
  const { heuristics } = Route.useLoaderData();

  const dimensions = useMemo(() => extractNumericDimensions(heuristics), [heuristics]);

  const [xAxis, setXAxis] = useState('sphericity');
  const [yAxis, setYAxis] = useState('spikiness');
  const [colorBy, setColorBy] = useState('intrinsic_dim_estimate');
  const [selectedPoint, setSelectedPoint] = useState<PolytopeHeuristics | null>(null);

  const scatterData = useMemo(() => {
    return heuristics.map((h) => ({
      ...h,
      x: getValue(h, xAxis),
      y: getValue(h, yAxis),
      color: getValue(h, colorBy),
      z: 1,
    }));
  }, [heuristics, xAxis, yAxis, colorBy]);

  const colorRange = useMemo(() => {
    const values = scatterData.map((d) => d.color);
    if (values.length === 0) return { min: 0, max: 1 };
    return { min: Math.min(...values), max: Math.max(...values) };
  }, [scatterData]);

  const xLabel = getLabel(xAxis);
  const yLabel = getLabel(yAxis);
  const colorLabel = getLabel(colorBy);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white mb-2">Polytope Heuristics Explorer</h1>
          <p className="text-gray-400">
            Explore speculative shape metrics across {heuristics.length} polytopes ·{' '}
            <Link to="/correlations" className="text-cyan-400 hover:text-cyan-300">
              correlations vs search outcomes →
            </Link>
          </p>
        </div>

        {heuristics.length === 0 ? (
          <div className="bg-slate-800/50 rounded-lg p-8 border border-slate-700 text-center text-gray-400">
            No heuristics computed yet. Run:{' '}
            <span className="font-mono text-gray-300">
              cargo run --release --bin heuristics -- --pool &lt;pool.jsonl&gt; --db &lt;db&gt;
            </span>
          </div>
        ) : (
          <>
            {/* Controls */}
            <div className="mb-6 bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {(
                  [
                    ['X Axis', xAxis, setXAxis],
                    ['Y Axis', yAxis, setYAxis],
                    ['Color By', colorBy, setColorBy],
                  ] as const
                ).map(([label, value, setter]) => (
                  <div key={label}>
                    <label className="block text-sm text-gray-400 mb-1">{label}</label>
                    <select
                      value={value}
                      onChange={(e) => setter(e.target.value)}
                      className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-gray-200 text-sm focus:outline-none focus:border-cyan-500"
                    >
                      {dimensions.map((key) => (
                        <option key={key} value={key}>
                          {getLabel(key)}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>

              {/* Color legend */}
              <div className="mt-4 flex items-center gap-2">
                <span className="text-xs text-gray-400">{colorLabel}:</span>
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-500">{colorRange.min.toFixed(2)}</span>
                  <div className="w-32 h-3 rounded bg-gradient-to-r from-blue-500 via-green-500 to-red-500" />
                  <span className="text-xs text-gray-500">{colorRange.max.toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Scatter Plot */}
              <div className="lg:col-span-2 bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <ResponsiveContainer width="100%" height={500}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name={xLabel}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      label={{ value: xLabel, position: 'bottom', fill: '#9ca3af', fontSize: 12 }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name={yLabel}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      label={{ value: yLabel, angle: -90, position: 'left', fill: '#9ca3af', fontSize: 12 }}
                    />
                    <ZAxis dataKey="z" range={[8, 8]} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ payload }) => {
                        if (!payload?.length) return null;
                        const data = payload[0].payload as PolytopeHeuristics & {
                          x: number;
                          y: number;
                          color: number;
                        };
                        return (
                          <div className="bg-slate-800 border border-slate-600 rounded p-2 text-sm">
                            <div className="text-cyan-400 font-medium mb-1 font-mono">
                              {data.polytope_id}
                            </div>
                            <div className="text-gray-300">
                              h11={data.h11 ?? '–'}, h21={data.h21 ?? '–'}
                            </div>
                            <div className="text-gray-400 mt-1">
                              {xLabel}: {data.x.toFixed(4)}
                            </div>
                            <div className="text-gray-400">
                              {yLabel}: {data.y.toFixed(4)}
                            </div>
                            <div className="text-gray-400">
                              {colorLabel}: {data.color.toFixed(4)}
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Scatter
                      data={scatterData}
                      onClick={(data) => setSelectedPoint(data as PolytopeHeuristics)}
                    >
                      {scatterData.map((entry, index) => (
                        <Cell
                          key={index}
                          fill={getColor(entry.color, colorRange.min, colorRange.max)}
                          stroke={
                            selectedPoint?.polytope_id === entry.polytope_id ? '#fff' : 'none'
                          }
                          strokeWidth={selectedPoint?.polytope_id === entry.polytope_id ? 2 : 0}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              {/* Selected Point Details */}
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 max-h-[600px] overflow-y-auto">
                {selectedPoint ? (
                  <>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-medium text-white font-mono">
                        {selectedPoint.polytope_id}
                      </h3>
                      <Link
                        to="/polytope/$id"
                        params={{ id: selectedPoint.polytope_id }}
                        className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
                      >
                        View 3D →
                      </Link>
                    </div>

                    <div className="space-y-1 text-xs">
                      {dimensions.map((key) => {
                        const value = (selectedPoint as Record<string, unknown>)[key];
                        if (typeof value !== 'number') return null;
                        return (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-400">{getLabel(key)}:</span>
                            <span className="text-gray-200 font-mono">
                              {Number.isInteger(value) ? value : value.toFixed(4)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div className="text-gray-400 text-center py-8">
                    Click a point to view details
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
