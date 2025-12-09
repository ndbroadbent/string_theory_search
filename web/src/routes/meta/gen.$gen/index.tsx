/**
 * Generation Detail Page
 * Shows all algorithms in a generation
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { getAlgorithms, getGenerationStatus } from '../../../server/meta';
import type { MetaAlgorithmWithFitness, GenerationStatus } from '../../../types';

export const Route = createFileRoute('/meta/gen/$gen/')({
  loader: async ({ params }) => {
    const gen = parseInt(params.gen, 10);
    const [algorithms, status] = await Promise.all([
      getAlgorithms({ data: { generation: gen, limit: 100 } }),
      getGenerationStatus({ data: { generation: gen } }),
    ]);
    return { generation: gen, algorithms, status };
  },
  component: GenerationDetail,
});

function GenerationDetail() {
  const { generation, algorithms, status } = Route.useLoaderData();

  const progress = status.total > 0
    ? ((status.completed + status.failed) / status.total) * 100
    : 0;

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link
          to="/meta"
          className="text-gray-400 hover:text-white transition-colors"
        >
          &larr; Back
        </Link>
        <h1 className="text-2xl font-bold">Generation {generation}</h1>
      </div>

      {/* Status Overview */}
      <div className="bg-slate-800 rounded-lg p-6 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Status</h2>
          <span className="text-gray-400">{status.total} algorithms</span>
        </div>

        {/* Progress bar */}
        <div className="h-3 bg-slate-700 rounded-full mb-4 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="grid grid-cols-4 gap-4 text-center">
          <div className="bg-slate-700/50 rounded-lg p-3">
            <div className="text-2xl font-bold text-gray-400">{status.pending}</div>
            <div className="text-sm text-gray-500">Pending</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <div className="text-2xl font-bold text-yellow-400">{status.running}</div>
            <div className="text-sm text-gray-500">Running</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <div className="text-2xl font-bold text-emerald-400">{status.completed}</div>
            <div className="text-sm text-gray-500">Completed</div>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <div className="text-2xl font-bold text-red-400">{status.failed}</div>
            <div className="text-sm text-gray-500">Failed</div>
          </div>
        </div>
      </div>

      {/* Algorithms Table */}
      <section className="bg-slate-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-700/50 text-left">
                <th className="px-4 py-3 font-medium">Algorithm</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Runs</th>
                <th className="px-4 py-3 font-medium">Meta-Fitness</th>
                <th className="px-4 py-3 font-medium">Best Final</th>
                <th className="px-4 py-3 font-medium">CC Error</th>
                <th className="px-4 py-3 font-medium">Params</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {algorithms.map((algo) => (
                <AlgorithmRow key={algo.id} algorithm={algo} generation={generation} />
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function AlgorithmRow({ algorithm, generation }: { algorithm: MetaAlgorithmWithFitness; generation: number }) {
  const statusColors: Record<string, string> = {
    pending: 'bg-gray-500/20 text-gray-400',
    running: 'bg-yellow-500/20 text-yellow-400',
    completed: 'bg-emerald-500/20 text-emerald-400',
    failed: 'bg-red-500/20 text-red-400',
  };

  return (
    <tr className="hover:bg-slate-700/30 transition-colors">
      <td className="px-4 py-3">
        <Link
          to="/meta/gen/$gen/algo/$algo"
          params={{ gen: String(generation), algo: String(algorithm.id) }}
          className="text-cyan-400 hover:text-cyan-300 font-mono"
        >
          #{algorithm.id}
        </Link>
      </td>
      <td className="px-4 py-3">
        <span className={`px-2 py-1 rounded text-xs ${statusColors[algorithm.status]}`}>
          {algorithm.status}
        </span>
      </td>
      <td className="px-4 py-3 font-mono">
        {algorithm.run_count}/{algorithm.runs_required}
      </td>
      <td className="px-4 py-3 font-mono">
        {algorithm.meta_fitness?.toFixed(4) ?? '-'}
      </td>
      <td className="px-4 py-3 font-mono">
        {algorithm.best_final_fitness?.toFixed(4) ?? '-'}
      </td>
      <td className="px-4 py-3 font-mono">
        {algorithm.best_cc_log_error?.toFixed(2) ?? '-'}
      </td>
      <td className="px-4 py-3 text-xs text-gray-400">
        pop={algorithm.population_size}, mut={algorithm.mutation_rate.toFixed(2)}
      </td>
    </tr>
  );
}
