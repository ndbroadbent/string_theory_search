/**
 * Heuristics Explorer - Scatter plot across different dimensions
 */

import { createFileRoute } from '@tanstack/react-router';
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

// Labels for dimension keys - will error if a key from data is missing
const DIMENSION_LABELS: Record<string, string> = {
  // Basic
  h11: 'h11',
  h21: 'h21',
  vertex_count: 'Vertex Count',

  // Circularity
  sphericity: 'Sphericity (π-ness)',
  inertia_isotropy: 'Inertia Isotropy',

  // Chirality
  chirality_optimal: 'Chirality (Optimal)',
  chirality_x: 'Chirality X',
  chirality_y: 'Chirality Y',
  chirality_z: 'Chirality Z',
  chirality_w: 'Chirality W',
  handedness_det: 'Handedness',

  // Symmetry
  symmetry_x: 'Symmetry X',
  symmetry_y: 'Symmetry Y',
  symmetry_z: 'Symmetry Z',
  symmetry_w: 'Symmetry W',

  // Flatness
  flatness_3d: 'Flatness 3D',
  flatness_2d: 'Flatness 2D',
  intrinsic_dim_estimate: 'Intrinsic Dimension',

  // Shape
  edge_length_cv: 'Edge Length CV',
  spikiness: 'Spikiness',
  max_exposure: 'Max Exposure',
  conformity_ratio: 'Conformity Ratio',
  distance_kurtosis: 'Distance Kurtosis',
  loner_score: 'Loner Score',

  // Stats
  coord_mean: 'Coord Mean',
  coord_median: 'Coord Median',
  coord_mode: 'Coord Mode',
  coord_std: 'Coord Std Dev',
  coord_iqr: 'Coord IQR',
  coord_range: 'Coord Range',
  coord_skewness: 'Coord Skewness',
  coord_kurtosis: 'Coord Kurtosis',
  mean_median_diff: 'Mean-Median Diff',

  // Entropy
  shannon_entropy: 'Shannon Entropy',
  joint_entropy: 'Joint Entropy',

  // Compression
  compression_ratio: 'Compression Ratio',
  sorted_compression_ratio: 'Sorted Compression Ratio',
  sort_compression_gain: 'Sort Compression Gain',

  // Correlations
  mean_abs_correlation: 'Mean Abs Correlation',
  spread_ratio: 'Spread Ratio',

  // Statistical tests
  normality_pvalue: 'Normality P-value',
  uniform_ks_stat: 'Uniform KS Stat',

  // Patterns
  phi_ratio_count: 'Golden Ratio Count',
  fibonacci_count: 'Fibonacci Count',
  zero_count: 'Zero Count',
  one_count: 'One Count',
  neg_one_count: 'Neg One Count',
  prime_count: 'Prime Count',

  // Outlier scores (population-level)
  outlier_score: 'Outlier Score',
  outlier_max_zscore: 'Max Z-Score',
  outlier_count_2sigma: '2σ Outlier Count',
  outlier_count_3sigma: '3σ Outlier Count',
};

// Get label for a key, throw if missing
function getLabel(key: string): string {
  const label = DIMENSION_LABELS[key];
  if (!label) {
    throw new Error(`Missing label for dimension: ${key}. Add it to DIMENSION_LABELS.`);
  }
  return label;
}

// Keys to exclude from dimension selectors (non-geometric identifiers)
const EXCLUDE_DIMENSIONS = new Set(['polytope_id', 'outlier_max_dim']);

// Extract numeric dimensions from a sample heuristics object
function extractNumericDimensions(sample: PolytopeHeuristics): string[] {
  const dimensions: string[] = [];
  for (const [key, value] of Object.entries(sample)) {
    if (EXCLUDE_DIMENSIONS.has(key)) continue;
    if (typeof value === 'number' && !Number.isNaN(value)) {
      getLabel(key); // Will throw if label missing
      dimensions.push(key);
    }
  }
  return dimensions.sort((a, b) => getLabel(a).localeCompare(getLabel(b)));
}

function getValue(h: PolytopeHeuristics, key: string): number {
  // Handle nested properties
  if (key.startsWith('spiral_')) {
    const axis = key.replace('spiral_', '');
    return h.spiral_correlations?.[axis] ?? 0;
  }
  if (key.startsWith('corr_')) {
    const pair = key.replace('corr_', '');
    return h.correlations?.[pair] ?? 0;
  }
  const value = (h as unknown as Record<string, unknown>)[key];
  return typeof value === 'number' ? value : 0;
}

function getColor(value: number, min: number, max: number): string {
  const normalized = (value - min) / (max - min + 1e-10);
  // Blue -> Cyan -> Green -> Yellow -> Red
  const hue = (1 - normalized) * 240; // 240 = blue, 0 = red
  return `hsl(${hue}, 80%, 50%)`;
}

function HeuristicsExplorer() {
  const { heuristics } = Route.useLoaderData();

  // Derive available dimensions from the data
  const dimensions = useMemo(() => {
    if (heuristics.length === 0) return [];
    return extractNumericDimensions(heuristics[0]);
  }, [heuristics]);

  const [xAxis, setXAxis] = useState('sphericity');
  const [yAxis, setYAxis] = useState('spikiness');
  const [colorBy, setColorBy] = useState('intrinsic_dim_estimate');
  const [selectedPoint, setSelectedPoint] = useState<PolytopeHeuristics | null>(null);

  // Prepare scatter data
  const scatterData = useMemo(() => {
    return heuristics.map((h) => ({
      ...h,
      x: getValue(h, xAxis),
      y: getValue(h, yAxis),
      color: getValue(h, colorBy),
    }));
  }, [heuristics, xAxis, yAxis, colorBy]);

  // Color range
  const colorRange = useMemo(() => {
    const values = scatterData.map((d) => d.color);
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
          <h1 className="text-2xl font-bold text-white mb-2">
            Polytope Heuristics Explorer
          </h1>
          <p className="text-gray-400">
            Explore speculative shape metrics across {heuristics.length} polytopes
          </p>
        </div>

        {/* Controls */}
        <div className="mb-6 bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* X Axis */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">X Axis</label>
              <select
                value={xAxis}
                onChange={(e) => setXAxis(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-gray-200 text-sm focus:outline-none focus:border-cyan-500"
              >
                {dimensions.map((key) => (
                  <option key={key} value={key}>
                    {getLabel(key)}
                  </option>
                ))}
              </select>
            </div>

            {/* Y Axis */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Y Axis</label>
              <select
                value={yAxis}
                onChange={(e) => setYAxis(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-gray-200 text-sm focus:outline-none focus:border-cyan-500"
              >
                {dimensions.map((key) => (
                  <option key={key} value={key}>
                    {getLabel(key)}
                  </option>
                ))}
              </select>
            </div>

            {/* Color By */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Color By</label>
              <select
                value={colorBy}
                onChange={(e) => setColorBy(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-gray-200 text-sm focus:outline-none focus:border-cyan-500"
              >
                {dimensions.map((key) => (
                  <option key={key} value={key}>
                    {getLabel(key)}
                  </option>
                ))}
              </select>
            </div>
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
                  label={{
                    value: xLabel,
                    position: 'bottom',
                    fill: '#9ca3af',
                    fontSize: 12,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name={yLabel}
                  stroke="#9ca3af"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  label={{
                    value: yLabel,
                    angle: -90,
                    position: 'left',
                    fill: '#9ca3af',
                    fontSize: 12,
                  }}
                />
                <ZAxis range={[50, 200]} />
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
                        <div className="text-cyan-400 font-medium mb-1">
                          Polytope #{data.polytope_id}
                        </div>
                        <div className="text-gray-300">
                          h11={data.h11}, h21={data.h21}
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
                        selectedPoint?.polytope_id === entry.polytope_id
                          ? '#fff'
                          : 'none'
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
                <h3 className="text-lg font-medium text-white mb-3">
                  Polytope #{selectedPoint.polytope_id}
                </h3>

                <div className="space-y-4">
                  {/* Basic Info */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">Basic</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-gray-400">h11:</div>
                      <div className="text-gray-200">{selectedPoint.h11}</div>
                      <div className="text-gray-400">h21:</div>
                      <div className="text-gray-200">{selectedPoint.h21}</div>
                      <div className="text-gray-400">Vertices:</div>
                      <div className="text-gray-200">{selectedPoint.vertex_count}</div>
                    </div>
                  </div>

                  {/* Circularity */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">
                      Circularity (π-ness)
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-gray-400">Sphericity:</div>
                      <div className="text-gray-200">
                        {selectedPoint.sphericity.toFixed(4)}
                      </div>
                      <div className="text-gray-400">Inertia Isotropy:</div>
                      <div className="text-gray-200">
                        {selectedPoint.inertia_isotropy.toFixed(4)}
                      </div>
                    </div>
                  </div>

                  {/* Chirality */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">Chirality</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {['x', 'y', 'z', 'w'].map((axis) => (
                        <React.Fragment key={axis}>
                          <div className="text-gray-400">{axis.toUpperCase()}:</div>
                          <div className="text-gray-200">
                            {(
                              selectedPoint[
                                `chirality_${axis}` as keyof PolytopeHeuristics
                              ] as number
                            ).toFixed(4)}
                          </div>
                        </React.Fragment>
                      ))}
                      <div className="text-gray-400">Handedness:</div>
                      <div className="text-gray-200">
                        {selectedPoint.handedness_det > 0
                          ? 'Right'
                          : selectedPoint.handedness_det < 0
                            ? 'Left'
                            : 'Achiral'}
                      </div>
                    </div>
                  </div>

                  {/* Symmetry */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">Symmetry</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {['x', 'y', 'z', 'w'].map((axis) => (
                        <React.Fragment key={axis}>
                          <div className="text-gray-400">{axis.toUpperCase()}:</div>
                          <div className="text-gray-200">
                            {(
                              (
                                selectedPoint[
                                  `symmetry_${axis}` as keyof PolytopeHeuristics
                                ] as number
                              ) * 100
                            ).toFixed(1)}
                            %
                          </div>
                        </React.Fragment>
                      ))}
                    </div>
                  </div>

                  {/* Flatness */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">Flatness</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-gray-400">3D:</div>
                      <div className="text-gray-200">
                        {(selectedPoint.flatness_3d * 100).toFixed(1)}%
                      </div>
                      <div className="text-gray-400">2D:</div>
                      <div className="text-gray-200">
                        {(selectedPoint.flatness_2d * 100).toFixed(1)}%
                      </div>
                      <div className="text-gray-400">Intrinsic Dim:</div>
                      <div className="text-gray-200">
                        {selectedPoint.intrinsic_dim_estimate.toFixed(1)}D
                      </div>
                    </div>
                  </div>

                  {/* Shape */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">Shape</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-gray-400">Spikiness:</div>
                      <div className="text-gray-200">
                        {selectedPoint.spikiness.toFixed(4)}
                      </div>
                      <div className="text-gray-400">Loner Score:</div>
                      <div className="text-gray-200">
                        {selectedPoint.loner_score.toFixed(4)}
                      </div>
                      <div className="text-gray-400">Conformity:</div>
                      <div className="text-gray-200">
                        {selectedPoint.conformity_ratio.toFixed(2)}
                      </div>
                    </div>
                  </div>

                  {/* Entropy */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">
                      Information
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-gray-400">Shannon:</div>
                      <div className="text-gray-200">
                        {selectedPoint.shannon_entropy.toFixed(4)}
                      </div>
                      <div className="text-gray-400">Compression:</div>
                      <div className="text-gray-200">
                        {(selectedPoint.compression_ratio * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Patterns */}
                  <div>
                    <h4 className="text-sm font-medium text-cyan-400 mb-2">Patterns</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-gray-400">Golden Ratio:</div>
                      <div className="text-gray-200">
                        {selectedPoint.phi_ratio_count}
                      </div>
                      <div className="text-gray-400">Fibonacci:</div>
                      <div className="text-gray-200">
                        {selectedPoint.fibonacci_count}
                      </div>
                      <div className="text-gray-400">Primes:</div>
                      <div className="text-gray-200">{selectedPoint.prime_count}</div>
                      <div className="text-gray-400">Zeros:</div>
                      <div className="text-gray-200">{selectedPoint.zero_count}</div>
                    </div>
                  </div>

                  {/* Spiral Correlations */}
                  {selectedPoint.spiral_correlations && (
                    <div>
                      <h4 className="text-sm font-medium text-cyan-400 mb-2">
                        Spiral Correlations
                      </h4>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {Object.entries(selectedPoint.spiral_correlations).map(
                          ([axis, val]) => (
                            <React.Fragment key={axis}>
                              <div className="text-gray-400">{axis.toUpperCase()}:</div>
                              <div className="text-gray-200">{val.toFixed(4)}</div>
                            </React.Fragment>
                          )
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-gray-400 text-center py-8">
                Click a point to view details
              </div>
            )}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            {
              label: 'Avg Sphericity',
              value:
                heuristics.reduce((sum, h) => sum + h.sphericity, 0) / heuristics.length,
            },
            {
              label: 'Avg Spikiness',
              value:
                heuristics.reduce((sum, h) => sum + h.spikiness, 0) / heuristics.length,
            },
            {
              label: 'Avg Entropy',
              value:
                heuristics.reduce((sum, h) => sum + h.shannon_entropy, 0) /
                heuristics.length,
            },
            {
              label: 'Avg Intrinsic Dim',
              value:
                heuristics.reduce((sum, h) => sum + h.intrinsic_dim_estimate, 0) /
                heuristics.length,
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="bg-slate-800/50 rounded-lg p-3 border border-slate-700"
            >
              <div className="text-xs text-gray-400">{stat.label}</div>
              <div className="text-lg font-medium text-white">
                {stat.value.toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Need to import React for the fragments
import React from 'react';
