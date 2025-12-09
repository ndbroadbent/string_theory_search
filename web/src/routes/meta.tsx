/**
 * Meta-GA Dashboard
 * Shows algorithms, generations, and global meta-state
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { getMetaState, getAlgorithms, getAllGenerations } from '../server/meta';
import type { MetaAlgorithmWithFitness, MetaState, GenerationStatus } from '../types';

export const Route = createFileRoute('/meta')({
  loader: async () => {
    const [metaState, algorithms, generations] = await Promise.all([
      getMetaState(),
      getAlgorithms({ data: { limit: 50 } }),
      getAllGenerations(),
    ]);
    return { metaState, algorithms, generations };
  },
  component: MetaDashboard,
});

function MetaDashboard() {
  const { metaState, algorithms, generations } = Route.useLoaderData();

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold mb-6">Meta-GA Dashboard</h1>

      {/* Global State */}
      <MetaStateCard state={metaState} />

      {/* Generation Overview */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Generations</h2>
        {generations.length === 0 ? (
          <div className="bg-slate-800 rounded-lg p-6 text-gray-400">
            No generations yet. Run the meta-GA to start evolving algorithms.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {generations.map((gen) => (
              <GenerationCard key={gen.generation} status={gen} />
            ))}
          </div>
        )}
      </section>

      {/* Algorithm Table */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Algorithms</h2>
        {algorithms.length === 0 ? (
          <div className="bg-slate-800 rounded-lg p-6 text-gray-400">
            No algorithms yet.
          </div>
        ) : (
          <AlgorithmTable algorithms={algorithms} />
        )}
      </section>
    </div>
  );
}

function MetaStateCard({ state }: { state: MetaState | null }) {
  if (!state) {
    return (
      <div className="bg-slate-800 rounded-lg p-6 mb-8 text-gray-400">
        Meta-GA not initialized. Run the search to start.
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-lg p-6 mb-8">
      <h2 className="text-lg font-semibold mb-4">Global State</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatBox
          label="Current Generation"
          value={state.current_generation.toString()}
        />
        <StatBox
          label="Algorithms/Gen"
          value={state.algorithms_per_generation.toString()}
        />
        <StatBox
          label="Best Meta-Fitness"
          value={state.best_meta_fitness?.toFixed(4) ?? '-'}
        />
        <StatBox
          label="Best Algorithm"
          value={state.best_algorithm_id ? `#${state.best_algorithm_id}` : '-'}
          link={state.best_algorithm_id ? `/meta/${state.best_algorithm_id}` : undefined}
        />
      </div>
    </div>
  );
}

function StatBox({
  label,
  value,
  link,
}: {
  label: string;
  value: string;
  link?: string;
}) {
  const content = (
    <div className="bg-slate-700/50 rounded-lg p-4">
      <div className="text-sm text-gray-400 mb-1">{label}</div>
      <div className="text-xl font-mono font-semibold">{value}</div>
    </div>
  );

  if (link) {
    return (
      <Link to={link} className="hover:ring-2 ring-cyan-500/50 rounded-lg transition-all">
        {content}
      </Link>
    );
  }

  return content;
}

function GenerationCard({ status }: { status: GenerationStatus }) {
  const progress = status.total > 0
    ? ((status.completed + status.failed) / status.total) * 100
    : 0;

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <div className="flex justify-between items-center mb-3">
        <span className="font-semibold">Generation {status.generation}</span>
        <span className="text-sm text-gray-400">{status.total} algorithms</span>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-slate-700 rounded-full mb-3 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Status counts */}
      <div className="flex gap-4 text-sm">
        <StatusBadge label="Pending" count={status.pending} color="gray" />
        <StatusBadge label="Running" count={status.running} color="yellow" />
        <StatusBadge label="Done" count={status.completed} color="green" />
        <StatusBadge label="Failed" count={status.failed} color="red" />
      </div>
    </div>
  );
}

function StatusBadge({
  label,
  count,
  color,
}: {
  label: string;
  count: number;
  color: 'gray' | 'yellow' | 'green' | 'red';
}) {
  const colors = {
    gray: 'text-gray-400',
    yellow: 'text-yellow-400',
    green: 'text-emerald-400',
    red: 'text-red-400',
  };

  return (
    <span className={colors[color]}>
      {count} {label}
    </span>
  );
}

function AlgorithmTable({ algorithms }: { algorithms: MetaAlgorithmWithFitness[] }) {
  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-700/50 text-left">
              <th className="px-4 py-3 font-medium">ID</th>
              <th className="px-4 py-3 font-medium">Gen</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium">Trials</th>
              <th className="px-4 py-3 font-medium">Meta-Fitness</th>
              <th className="px-4 py-3 font-medium">Best Final</th>
              <th className="px-4 py-3 font-medium">CC Error</th>
              <th className="px-4 py-3 font-medium">Params</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {algorithms.map((algo) => (
              <AlgorithmRow key={algo.id} algorithm={algo} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AlgorithmRow({ algorithm }: { algorithm: MetaAlgorithmWithFitness }) {
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
          to={`/meta/${algorithm.id}`}
          className="text-cyan-400 hover:text-cyan-300 font-mono"
        >
          #{algorithm.id}
        </Link>
      </td>
      <td className="px-4 py-3 font-mono">{algorithm.meta_generation}</td>
      <td className="px-4 py-3">
        <span className={`px-2 py-1 rounded text-xs ${statusColors[algorithm.status]}`}>
          {algorithm.status}
        </span>
      </td>
      <td className="px-4 py-3 font-mono">
        {algorithm.trial_count}/{algorithm.trials_required}
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
