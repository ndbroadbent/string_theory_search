/**
 * Meta-GA Dashboard
 * Shows generations table and latest generation algorithms
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { getMetaState, getAlgorithms, getAllGenerations } from '../../server/meta';
import type { MetaAlgorithmWithFitness, MetaState, GenerationStatus } from '../../types';

export const Route = createFileRoute('/meta/')({
  loader: async () => {
    const [metaState, generations] = await Promise.all([
      getMetaState(),
      getAllGenerations(),
    ]);

    // Get algorithms for the latest generation
    const latestGen = generations.length > 0 ? generations[0].generation : 0;
    const latestAlgorithms = await getAlgorithms({ data: { generation: latestGen, limit: 50 } });

    return { metaState, generations, latestAlgorithms, latestGen };
  },
  component: MetaDashboard,
});

function MetaDashboard() {
  const { metaState, generations, latestAlgorithms, latestGen } = Route.useLoaderData();

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Meta-GA</h1>
        {metaState && (
          <div className="text-sm text-gray-400">
            Seed: <span className="font-mono text-cyan-400">{metaState.master_seed}</span>
          </div>
        )}
      </div>

      {/* Generations Table */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Generations</h2>
        {generations.length === 0 ? (
          <div className="bg-slate-800 rounded-lg p-6 text-gray-400">
            No generations yet. Run the search to start evolving algorithms.
          </div>
        ) : (
          <GenerationsTable generations={generations} />
        )}
      </section>

      {/* Latest Generation Algorithms */}
      <section>
        <h2 className="text-xl font-semibold mb-4">
          Generation {latestGen} - Algorithms
        </h2>
        {latestAlgorithms.length === 0 ? (
          <div className="bg-slate-800 rounded-lg p-6 text-gray-400">
            No algorithms yet.
          </div>
        ) : (
          <AlgorithmTable algorithms={latestAlgorithms} generation={latestGen} />
        )}
      </section>
    </div>
  );
}

function GenerationsTable({ generations }: { generations: GenerationStatus[] }) {
  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-700/50 text-left">
            <th className="px-4 py-3 font-medium">Generation</th>
            <th className="px-4 py-3 font-medium">Total</th>
            <th className="px-4 py-3 font-medium">Pending</th>
            <th className="px-4 py-3 font-medium">Running</th>
            <th className="px-4 py-3 font-medium">Completed</th>
            <th className="px-4 py-3 font-medium">Failed</th>
            <th className="px-4 py-3 font-medium">Progress</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700">
          {generations.map((gen) => {
            const progress = gen.total > 0
              ? ((gen.completed + gen.failed) / gen.total) * 100
              : 0;

            return (
              <tr key={gen.generation} className="hover:bg-slate-700/30 transition-colors">
                <td className="px-4 py-3">
                  <Link
                    to="/meta/gen/$gen"
                    params={{ gen: String(gen.generation) }}
                    className="text-cyan-400 hover:text-cyan-300 font-mono"
                  >
                    Gen {gen.generation}
                  </Link>
                </td>
                <td className="px-4 py-3 font-mono">{gen.total}</td>
                <td className="px-4 py-3 font-mono text-gray-400">{gen.pending}</td>
                <td className="px-4 py-3 font-mono text-yellow-400">{gen.running}</td>
                <td className="px-4 py-3 font-mono text-emerald-400">{gen.completed}</td>
                <td className="px-4 py-3 font-mono text-red-400">{gen.failed}</td>
                <td className="px-4 py-3 w-32">
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AlgorithmTable({ algorithms, generation }: { algorithms: MetaAlgorithmWithFitness[]; generation: number }) {
  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden">
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
        {algorithm.meta_fitness?.toFixed(5) ?? '-'}
      </td>
      <td className="px-4 py-3 font-mono">
        {algorithm.best_final_fitness?.toFixed(5) ?? '-'}
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
