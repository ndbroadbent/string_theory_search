/**
 * Run Detail Page - Shows the best evaluation from a run
 * with 4D polytope visualization and genome parameters
 */

import { createFileRoute, Link, notFound } from '@tanstack/react-router';
import { Suspense, lazy } from 'react';
import { getRuns, getAlgorithm, getEvaluation } from '../../../../server/meta';
import { getPolytope } from '../../../../server/polytopes';
import { PhysicsGauges } from '../../../../components/overview/PhysicsGauges';
import { PolytopeControls } from '../../../../components/polytope/PolytopeControls';
import type { MetaRun, MetaAlgorithmWithFitness, Evaluation, PolytopeEntry, PhysicsOutput } from '../../../../types';

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
  import('../../../../components/polytope/Polytope4D').then((m) => ({
    default: m.Polytope4D,
  }))
);

interface LoaderData {
  generation: number;
  algorithm: MetaAlgorithmWithFitness;
  run: MetaRun;
  allRuns: MetaRun[];
  evaluation: Evaluation | null;
  polytope: PolytopeEntry | null;
}

export const Route = createFileRoute('/meta/gen/$gen/algo/$algo/run/$run')({
  loader: async ({ params }): Promise<LoaderData> => {
    const gen = parseInt(params.gen, 10);
    const algoId = parseInt(params.algo, 10);
    const runNumber = parseInt(params.run, 10);

    const [algorithm, runs] = await Promise.all([
      getAlgorithm({ data: { id: algoId } }),
      getRuns({ data: { algorithmId: algoId } }),
    ]);

    if (!algorithm) throw notFound();

    const run = runs.find((r) => r.run_number === runNumber);
    if (!run) throw notFound();

    // Load best evaluation and its polytope
    let evaluation: Evaluation | null = null;
    let polytope: PolytopeEntry | null = null;

    if (run.best_evaluation_id) {
      evaluation = await getEvaluation({ data: { id: run.best_evaluation_id } });
      if (evaluation) {
        polytope = await getPolytope({ data: { polytopeId: evaluation.polytope_id } });
      }
    }

    return { generation: gen, algorithm, run, allRuns: runs, evaluation, polytope };
  },
  component: RunDetail,
});

function RunDetail() {
  const { generation, algorithm, run, allRuns, evaluation, polytope } = Route.useLoaderData();

  // Convert evaluation to PhysicsOutput for PhysicsGauges
  const physics: PhysicsOutput | null = evaluation ? {
    success: evaluation.success,
    error: evaluation.error,
    alpha_em: evaluation.alpha_em ?? 0,
    alpha_s: evaluation.alpha_s ?? 0,
    sin2_theta_w: evaluation.sin2_theta_w ?? 0,
    cosmological_constant: evaluation.cosmological_constant ?? 0,
    n_generations: evaluation.n_generations ?? 0,
    m_e_planck_ratio: 0,
    m_p_planck_ratio: 0,
    cy_volume: 0,
    string_coupling: evaluation.g_s ?? 0,
    flux_tadpole: 0,
    superpotential_abs: 0,
  } : null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-center gap-4">
          <Link
            to="/meta/gen/$gen/algo/$algo"
            params={{ gen: String(generation), algo: String(algorithm.id) }}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ← Algo #{algorithm.id}
          </Link>
          <h1 className="text-2xl font-bold text-white">
            Run #{run.run_number}
          </h1>
          <span className="text-green-400 font-mono">
            Fitness: {run.final_fitness.toFixed(6)}
          </span>
        </div>

        {/* Breadcrumb */}
        <div className="text-sm text-gray-400 mb-6">
          <Link to="/meta" className="hover:text-white">Meta</Link>
          {' / '}
          <Link to="/meta/gen/$gen" params={{ gen: String(generation) }} className="hover:text-white">
            Gen {generation}
          </Link>
          {' / '}
          <Link
            to="/meta/gen/$gen/algo/$algo"
            params={{ gen: String(generation), algo: String(algorithm.id) }}
            className="hover:text-white"
          >
            Algo #{algorithm.id}
          </Link>
          {' / '}
          <span className="text-white">Run #{run.run_number}</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - 4D Visualization */}
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
                      {evaluation ? 'No vertex data available' : 'No best evaluation recorded'}
                    </div>
                  )}
                </Suspense>
              </div>
            </div>

            <PolytopeControls />

            {/* Run Performance Metrics */}
            <div className="bg-slate-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Run Performance</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <MetricBox label="Initial Fitness" value={run.initial_fitness.toFixed(4)} />
                <MetricBox label="Final Fitness" value={run.final_fitness.toFixed(4)} highlight />
                <MetricBox
                  label="Improvement"
                  value={`${run.fitness_improvement > 0 ? '+' : ''}${run.fitness_improvement.toFixed(4)}`}
                  color={run.fitness_improvement > 0 ? 'green' : 'red'}
                />
                <MetricBox label="Improvement Rate" value={run.improvement_rate.toExponential(2)} />
                <MetricBox label="Fitness AUC" value={run.fitness_auc.toFixed(4)} />
                <MetricBox label="Generations" value={run.generations_run.toString()} />
                <MetricBox label="CC Log Error" value={run.best_cc_log_error.toFixed(2)} suffix="log10" />
                <MetricBox label="Physics Success" value={`${(run.physics_success_rate * 100).toFixed(1)}%`} />
                <MetricBox label="Unique Polytopes" value={run.unique_polytopes_tried.toString()} />
              </div>
            </div>
          </div>

          {/* Right column - Evaluation details */}
          <div className="space-y-4">
            {/* Physics output */}
            {physics && <PhysicsGauges physics={physics} />}

            {/* Genome parameters */}
            {evaluation && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Genome Parameters
                </h3>

                <div className="space-y-3 text-xs">
                  <div>
                    <p className="text-gray-400 mb-1">Polytope ID:</p>
                    <Link
                      to="/polytope/$id"
                      params={{ id: String(evaluation.polytope_id) }}
                      className="text-cyan-400 hover:text-cyan-300 font-mono"
                    >
                      #{evaluation.polytope_id}
                    </Link>
                  </div>

                  {polytope && (
                    <div>
                      <p className="text-gray-400 mb-1">Hodge Numbers:</p>
                      <p className="text-gray-200 font-mono">
                        h11 = {polytope.h11}, h21 = {polytope.h21}
                      </p>
                    </div>
                  )}

                  {evaluation.g_s != null && (
                    <div>
                      <p className="text-gray-400 mb-1">String Coupling (g_s):</p>
                      <p className="text-gray-200 font-mono">
                        {evaluation.g_s.toFixed(6)}
                      </p>
                    </div>
                  )}

                  {evaluation.kahler_moduli && (
                    <div>
                      <p className="text-gray-400 mb-1">
                        Kahler Moduli ({evaluation.kahler_moduli.length}):
                      </p>
                      <div className="bg-slate-900/50 rounded p-2 max-h-24 overflow-y-auto">
                        <p className="text-gray-300 font-mono text-[10px] break-all">
                          [{evaluation.kahler_moduli.map((v) => v.toFixed(3)).join(', ')}]
                        </p>
                      </div>
                    </div>
                  )}

                  {evaluation.complex_moduli && (
                    <div>
                      <p className="text-gray-400 mb-1">
                        Complex Moduli ({evaluation.complex_moduli.length}):
                      </p>
                      <div className="bg-slate-900/50 rounded p-2 max-h-24 overflow-y-auto">
                        <p className="text-gray-300 font-mono text-[10px] break-all">
                          [{evaluation.complex_moduli.map((v) => v.toFixed(3)).join(', ')}]
                        </p>
                      </div>
                    </div>
                  )}

                  {evaluation.flux_f && (
                    <div>
                      <p className="text-gray-400 mb-1">
                        F-flux ({evaluation.flux_f.length}):
                      </p>
                      <div className="bg-slate-900/50 rounded p-2 max-h-20 overflow-y-auto">
                        <p className="text-gray-300 font-mono text-[10px] break-all">
                          [{evaluation.flux_f.join(', ')}]
                        </p>
                      </div>
                    </div>
                  )}

                  {evaluation.flux_h && (
                    <div>
                      <p className="text-gray-400 mb-1">
                        H-flux ({evaluation.flux_h.length}):
                      </p>
                      <div className="bg-slate-900/50 rounded p-2 max-h-20 overflow-y-auto">
                        <p className="text-gray-300 font-mono text-[10px] break-all">
                          [{evaluation.flux_h.join(', ')}]
                        </p>
                      </div>
                    </div>
                  )}
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

            {/* Run Navigation */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-sm font-semibold mb-3 text-gray-300">All Runs</h2>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {allRuns.map((r) => (
                  <Link
                    key={r.id}
                    to="/meta/gen/$gen/algo/$algo/run/$run"
                    params={{ gen: String(generation), algo: String(algorithm.id), run: String(r.run_number) }}
                    className={`block p-2 rounded-lg transition-colors text-sm ${
                      r.run_number === run.run_number
                        ? 'bg-cyan-500/20 text-cyan-400'
                        : 'bg-slate-700/50 hover:bg-slate-700 text-gray-300'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-mono">Run #{r.run_number}</span>
                      <span className="text-xs">{r.final_fitness.toFixed(4)}</span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>

            {/* Timing */}
            {(run.started_at || run.ended_at) && (
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Timing</h3>
                <dl className="space-y-2 text-xs">
                  {run.started_at && (
                    <div className="flex justify-between">
                      <dt className="text-gray-400">Started</dt>
                      <dd className="font-mono text-gray-200">{new Date(run.started_at).toLocaleString()}</dd>
                    </div>
                  )}
                  {run.ended_at && (
                    <div className="flex justify-between">
                      <dt className="text-gray-400">Ended</dt>
                      <dd className="font-mono text-gray-200">{new Date(run.ended_at).toLocaleString()}</dd>
                    </div>
                  )}
                  {run.started_at && run.ended_at && (
                    <div className="flex justify-between">
                      <dt className="text-gray-400">Duration</dt>
                      <dd className="font-mono text-gray-200">
                        {((new Date(run.ended_at).getTime() - new Date(run.started_at).getTime()) / 1000).toFixed(1)}s
                      </dd>
                    </div>
                  )}
                </dl>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBox({
  label,
  value,
  suffix,
  highlight,
  color,
}: {
  label: string;
  value: string;
  suffix?: string;
  highlight?: boolean;
  color?: 'green' | 'red';
}) {
  let textColor = '';
  if (color === 'green') textColor = 'text-emerald-400';
  else if (color === 'red') textColor = 'text-red-400';
  else if (highlight) textColor = 'text-cyan-400';

  return (
    <div className={`rounded-lg p-3 ${highlight ? 'bg-cyan-500/10 ring-1 ring-cyan-500/30' : 'bg-slate-700/50'}`}>
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`font-mono ${textColor}`}>
        {value}
        {suffix && <span className="text-xs text-gray-500 ml-1">{suffix}</span>}
      </div>
    </div>
  );
}
