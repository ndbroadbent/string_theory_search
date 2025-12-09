/**
 * Genome Detail View with 4D Polytope Visualization
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { Suspense, lazy, useState, useEffect } from 'react';
import { getGenomeByPolytope } from '../server/results';
import { getPolytope } from '../server/polytopes';
import { PhysicsGauges } from '../components/overview/PhysicsGauges';
import { PolytopeControls } from '../components/polytope/PolytopeControls';
import type { GenomeResult, PolytopeEntry } from '../types';

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

export const Route = createFileRoute('/genome/$polytopeId')({
  component: GenomeDetail,
  validateSearch: (search: Record<string, unknown>) => ({
    runId: search.runId as string | undefined,
  }),
  loader: async ({ params }) => {
    const polytopeId = parseInt(params.polytopeId, 10);
    return { polytopeId };
  },
});

function GenomeDetail() {
  const { polytopeId } = Route.useLoaderData();
  const { runId } = Route.useSearch();

  const [genome, setGenome] = useState<GenomeResult | null>(null);
  const [polytope, setPolytope] = useState<PolytopeEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load genome and polytope data
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        // Load genome data if runId is provided
        if (runId) {
          const genomeData = await getGenomeByPolytope({
            data: { runId, polytopeId },
          });
          setGenome(genomeData);
        }

        // Load polytope data
        const polytopeData = await getPolytope({ data: { polytopeId } });
        setPolytope(polytopeData);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [polytopeId, runId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-gray-400">Loading polytope #{polytopeId}...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-red-400">{error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-center gap-4">
          <Link
            to="/"
            className="text-gray-400 hover:text-white transition-colors"
          >
            ← Dashboard
          </Link>
          <h1 className="text-2xl font-bold text-white">
            Polytope #{polytopeId}
          </h1>
          {genome && (
            <span className="text-green-400 font-mono">
              Fitness: {genome.fitness.toFixed(6)}
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
          </div>

          {/* Right column - Genome details */}
          <div className="space-y-4">
            {/* Physics output */}
            {genome && <PhysicsGauges physics={genome.physics} />}

            {/* Genome parameters */}
            {genome && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Genome Parameters
                </h3>

                <div className="space-y-3 text-xs">
                  <div>
                    <p className="text-gray-400 mb-1">Hodge Numbers:</p>
                    <p className="text-gray-200 font-mono">
                      h11 = {genome.genome.h11}, h21 = {genome.genome.h21}
                    </p>
                  </div>

                  <div>
                    <p className="text-gray-400 mb-1">String Coupling (g_s):</p>
                    <p className="text-gray-200 font-mono">
                      {genome.genome.g_s.toFixed(6)}
                    </p>
                  </div>

                  <div>
                    <p className="text-gray-400 mb-1">
                      Kahler Moduli ({genome.genome.kahler_moduli.length}):
                    </p>
                    <div className="bg-slate-900/50 rounded p-2 max-h-24 overflow-y-auto">
                      <p className="text-gray-300 font-mono text-[10px] break-all">
                        [{genome.genome.kahler_moduli.map((v) => v.toFixed(3)).join(', ')}]
                      </p>
                    </div>
                  </div>

                  <div>
                    <p className="text-gray-400 mb-1">
                      Complex Moduli ({genome.genome.complex_moduli.length}):
                    </p>
                    <div className="bg-slate-900/50 rounded p-2 max-h-24 overflow-y-auto">
                      <p className="text-gray-300 font-mono text-[10px] break-all">
                        [{genome.genome.complex_moduli.map((v) => v.toFixed(3)).join(', ')}]
                      </p>
                    </div>
                  </div>

                  <div>
                    <p className="text-gray-400 mb-1">
                      F-flux ({genome.genome.flux_f.length}):
                    </p>
                    <div className="bg-slate-900/50 rounded p-2 max-h-20 overflow-y-auto">
                      <p className="text-gray-300 font-mono text-[10px] break-all">
                        [{genome.genome.flux_f.join(', ')}]
                      </p>
                    </div>
                  </div>

                  <div>
                    <p className="text-gray-400 mb-1">
                      H-flux ({genome.genome.flux_h.length}):
                    </p>
                    <div className="bg-slate-900/50 rounded p-2 max-h-20 overflow-y-auto">
                      <p className="text-gray-300 font-mono text-[10px] break-all">
                        [{genome.genome.flux_h.join(', ')}]
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Polytope info */}
            {polytope && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Polytope Data
                </h3>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Vertices:</span>
                    <span className="text-gray-200">{polytope.vertex_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">h11:</span>
                    <span className="text-gray-200">{polytope.h11}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">h21:</span>
                    <span className="text-gray-200">{polytope.h21}</span>
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
