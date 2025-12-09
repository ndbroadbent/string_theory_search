/**
 * Polytope Explorer - View any polytope by ID with 3D visualization and heuristics
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { Suspense, lazy } from 'react';
import { getPolytope } from '../server/polytopes';
import { getHeuristicsForPolytope } from '../server/heuristics';
import { PolytopeControls } from '../components/polytope/PolytopeControls';
import type { PolytopeHeuristics } from '../types';

/** Reshape flat vertex array to 4D points */
function reshapeVertices(flat: number[], vertexCount: number): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < vertexCount; i++) {
    result.push(flat.slice(i * 4, i * 4 + 4));
  }
  return result;
}

// Lazy load Three.js component to avoid SSR issues
const Polytope4D = lazy(() =>
  import('../components/polytope/Polytope4D').then((m) => ({
    default: m.Polytope4D,
  }))
);

export const Route = createFileRoute('/polytope/$id')({
  component: PolytopeExplorer,
  loader: async ({ params }) => {
    const polytopeId = parseInt(params.id, 10);
    const [polytope, heuristics] = await Promise.all([
      getPolytope({ data: { polytopeId } }),
      getHeuristicsForPolytope({ data: { polytopeId } }),
    ]);
    return { polytopeId, polytope, heuristics };
  },
});

// Group heuristics by category for display
const HEURISTIC_CATEGORIES = {
  'Basic': ['h11', 'h21', 'vertex_count'],
  'Circularity': ['sphericity', 'inertia_isotropy'],
  'Chirality': ['chirality_optimal', 'chirality_x', 'chirality_y', 'chirality_z', 'chirality_w', 'handedness_det'],
  'Symmetry': ['symmetry_x', 'symmetry_y', 'symmetry_z', 'symmetry_w'],
  'Flatness': ['flatness_3d', 'flatness_2d', 'intrinsic_dim_estimate'],
  'Shape': ['spikiness', 'max_exposure', 'conformity_ratio', 'distance_kurtosis', 'loner_score'],
  'Statistics': ['coord_mean', 'coord_median', 'coord_std', 'coord_skewness', 'coord_kurtosis', 'shannon_entropy', 'joint_entropy'],
  'Compression': ['compression_ratio', 'sorted_compression_ratio', 'sort_compression_gain'],
  'Patterns': ['phi_ratio_count', 'fibonacci_count', 'zero_count', 'one_count', 'prime_count'],
  'Outlier': ['outlier_score', 'outlier_max_zscore', 'outlier_count_2sigma', 'outlier_count_3sigma', 'outlier_max_dim'],
};

function formatValue(value: unknown): string {
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toString();
    return value.toFixed(4);
  }
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function PolytopeExplorer() {
  const { polytopeId, polytope, heuristics } = Route.useLoaderData();

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-center gap-4">
          <Link
            to="/heuristics"
            className="text-gray-400 hover:text-white transition-colors"
          >
            ← Heuristics Explorer
          </Link>
          <h1 className="text-2xl font-bold text-white">
            Polytope #{polytopeId}
          </h1>
          {heuristics && (
            <span className={`font-mono text-sm px-2 py-1 rounded ${
              heuristics.outlier_score > 2 ? 'bg-red-900/50 text-red-300' :
              heuristics.outlier_score > 1 ? 'bg-yellow-900/50 text-yellow-300' :
              'bg-green-900/50 text-green-300'
            }`}>
              Outlier: {heuristics.outlier_score.toFixed(2)}
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - 3D Visualization */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
              <div className="aspect-[4/3]">
                <Suspense
                  fallback={
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      Loading 3D view...
                    </div>
                  }
                >
                  {polytope?.vertices && polytope.vertex_count > 0 ? (
                    <Polytope4D
                      vertices={reshapeVertices(polytope.vertices, polytope.vertex_count)}
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      No vertex data available
                    </div>
                  )}
                </Suspense>
              </div>
            </div>

            <PolytopeControls />

            {/* Vertex data */}
            {polytope && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Vertices ({polytope.vertex_count} points in 4D)
                </h3>
                <div className="bg-slate-900/50 rounded p-2 max-h-48 overflow-y-auto font-mono text-xs">
                  {reshapeVertices(polytope.vertices, polytope.vertex_count).map((v, i) => (
                    <div key={i} className="text-gray-300">
                      [{v.map(c => c.toString().padStart(3)).join(', ')}]
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right column - Heuristics */}
          <div className="space-y-4 max-h-[calc(100vh-8rem)] overflow-y-auto">
            {heuristics ? (
              Object.entries(HEURISTIC_CATEGORIES).map(([category, keys]) => (
                <div key={category} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                  <h3 className="text-sm font-medium text-cyan-400 mb-3">{category}</h3>
                  <div className="space-y-1 text-xs">
                    {keys.map(key => {
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
                <p className="text-gray-400 text-sm">
                  No heuristics computed for this polytope yet.
                </p>
                <p className="text-gray-500 text-xs mt-2">
                  Run: python scripts/compute_heuristics.py --ids {polytopeId}
                </p>
              </div>
            )}

            {/* Raw polytope data */}
            {polytope && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Raw Data</h3>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">h11:</span>
                    <span className="text-gray-200">{polytope.h11}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">h21:</span>
                    <span className="text-gray-200">{polytope.h21}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">vertex_count:</span>
                    <span className="text-gray-200">{polytope.vertex_count}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
