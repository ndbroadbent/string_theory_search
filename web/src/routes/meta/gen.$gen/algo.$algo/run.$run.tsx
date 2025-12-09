/**
 * Run Detail Page
 * Shows details of a single run execution
 */

import { createFileRoute, Link, notFound } from '@tanstack/react-router';
import { getRuns, getAlgorithm } from '../../../../server/meta';
import type { MetaRun, MetaAlgorithmWithFitness } from '../../../../types';

export const Route = createFileRoute('/meta/gen/$gen/algo/$algo/run/$run')({
  loader: async ({ params }) => {
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

    return { generation: gen, algorithm, run, allRuns: runs };
  },
  component: RunDetail,
});

function RunDetail() {
  const { generation, algorithm, run, allRuns } = Route.useLoaderData();

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link
          to="/meta/gen/$gen/algo/$algo"
          params={{ gen: String(generation), algo: String(algorithm.id) }}
          className="text-gray-400 hover:text-white transition-colors"
        >
          &larr; Algo #{algorithm.id}
        </Link>
        <h1 className="text-2xl font-bold">
          Run #{run.run_number}
        </h1>
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
        {/* Main Metrics */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Performance</h2>
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

          {/* Timing */}
          {(run.started_at || run.ended_at) && (
            <div className="bg-slate-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Timing</h2>
              <dl className="space-y-2 text-sm">
                {run.started_at && (
                  <div className="flex justify-between">
                    <dt className="text-gray-400">Started</dt>
                    <dd className="font-mono">{new Date(run.started_at).toLocaleString()}</dd>
                  </div>
                )}
                {run.ended_at && (
                  <div className="flex justify-between">
                    <dt className="text-gray-400">Ended</dt>
                    <dd className="font-mono">{new Date(run.ended_at).toLocaleString()}</dd>
                  </div>
                )}
                {run.started_at && run.ended_at && (
                  <div className="flex justify-between">
                    <dt className="text-gray-400">Duration</dt>
                    <dd className="font-mono">
                      {((new Date(run.ended_at).getTime() - new Date(run.started_at).getTime()) / 1000).toFixed(1)}s
                    </dd>
                  </div>
                )}
              </dl>
            </div>
          )}
        </div>

        {/* Sidebar - Run Navigation */}
        <div className="space-y-6">
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">All Runs</h2>
            <div className="space-y-2">
              {allRuns.map((r) => (
                <Link
                  key={r.id}
                  to="/meta/gen/$gen/algo/$algo/run/$run"
                  params={{ gen: String(generation), algo: String(algorithm.id), run: String(r.run_number) }}
                  className={`block p-3 rounded-lg transition-colors ${
                    r.run_number === run.run_number
                      ? 'bg-cyan-500/20 text-cyan-400'
                      : 'bg-slate-700/50 hover:bg-slate-700 text-gray-300'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-mono">Run #{r.run_number}</span>
                    <span className="text-sm">{r.final_fitness.toFixed(4)}</span>
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* Algorithm Info */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Algorithm</h2>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-400">ID</dt>
                <dd className="font-mono">#{algorithm.id}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-400">Generation</dt>
                <dd className="font-mono">{algorithm.meta_generation}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-400">Population</dt>
                <dd className="font-mono">{algorithm.population_size}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-400">Max Gens</dt>
                <dd className="font-mono">{algorithm.max_generations}</dd>
              </div>
            </dl>
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
