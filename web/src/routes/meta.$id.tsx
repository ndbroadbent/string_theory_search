/**
 * Algorithm Detail Page
 * Shows algorithm parameters, trials, and lineage
 */

import { createFileRoute, Link, notFound } from '@tanstack/react-router';
import { getAlgorithm, getTrials, getAlgorithmLineage } from '../server/meta';
import type { MetaAlgorithmWithFitness, MetaTrial, MetaAlgorithm } from '../types';

export const Route = createFileRoute('/meta/$id')({
  loader: async ({ params }) => {
    const id = parseInt(params.id, 10);
    if (isNaN(id)) throw notFound();

    const [algorithm, trials, lineage] = await Promise.all([
      getAlgorithm({ data: { id } }),
      getTrials({ data: { algorithmId: id } }),
      getAlgorithmLineage({ data: { id } }),
    ]);

    if (!algorithm) throw notFound();

    return { algorithm, trials, lineage };
  },
  component: AlgorithmDetail,
});

function AlgorithmDetail() {
  const { algorithm, trials, lineage } = Route.useLoaderData();

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
        <h1 className="text-2xl font-bold">
          Algorithm #{algorithm.id}
          {algorithm.name && (
            <span className="text-gray-400 ml-2">({algorithm.name})</span>
          )}
        </h1>
        <StatusBadge status={algorithm.status} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Performance Summary */}
          <PerformanceCard algorithm={algorithm} />

          {/* Trials Table */}
          <section className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">
              Trials ({trials.length}/{algorithm.trials_required})
            </h2>
            {trials.length === 0 ? (
              <p className="text-gray-400">No trials completed yet.</p>
            ) : (
              <TrialsTable trials={trials} />
            )}
          </section>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Parameters */}
          <ParametersCard algorithm={algorithm} />

          {/* Feature Weights */}
          <FeatureWeightsCard weights={algorithm.feature_weights} />

          {/* Lineage */}
          {lineage.length > 1 && <LineageCard lineage={lineage} currentId={algorithm.id} />}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: MetaAlgorithm['status'] }) {
  const colors: Record<string, string> = {
    pending: 'bg-gray-500/20 text-gray-400 border-gray-500/50',
    running: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
    completed: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50',
    failed: 'bg-red-500/20 text-red-400 border-red-500/50',
  };

  return (
    <span className={`px-3 py-1 rounded-full text-sm border ${colors[status]}`}>
      {status}
    </span>
  );
}

function PerformanceCard({ algorithm }: { algorithm: MetaAlgorithmWithFitness }) {
  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold mb-4">Performance</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricBox
          label="Meta-Fitness"
          value={algorithm.meta_fitness?.toFixed(4) ?? '-'}
          highlight
        />
        <MetricBox
          label="Best Final Fitness"
          value={algorithm.best_final_fitness?.toFixed(4) ?? '-'}
        />
        <MetricBox
          label="Best CC Error"
          value={algorithm.best_cc_log_error?.toFixed(2) ?? '-'}
          suffix="log10"
        />
        <MetricBox
          label="Mean Improvement"
          value={algorithm.mean_improvement_rate?.toExponential(2) ?? '-'}
        />
        <MetricBox
          label="Best Improvement"
          value={algorithm.best_improvement_rate?.toExponential(2) ?? '-'}
        />
        <MetricBox
          label="Mean Fitness AUC"
          value={algorithm.mean_fitness_auc?.toFixed(4) ?? '-'}
        />
        <MetricBox
          label="Mean CC Error"
          value={algorithm.mean_cc_log_error?.toFixed(2) ?? '-'}
          suffix="log10"
        />
        <MetricBox
          label="Trials"
          value={`${algorithm.trial_count}/${algorithm.trials_required}`}
        />
      </div>
    </div>
  );
}

function MetricBox({
  label,
  value,
  suffix,
  highlight,
}: {
  label: string;
  value: string;
  suffix?: string;
  highlight?: boolean;
}) {
  return (
    <div className={`rounded-lg p-3 ${highlight ? 'bg-cyan-500/10 ring-1 ring-cyan-500/30' : 'bg-slate-700/50'}`}>
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`font-mono ${highlight ? 'text-cyan-400' : ''}`}>
        {value}
        {suffix && <span className="text-xs text-gray-500 ml-1">{suffix}</span>}
      </div>
    </div>
  );
}

function ParametersCard({ algorithm }: { algorithm: MetaAlgorithmWithFitness }) {
  const params = [
    { label: 'Generation', value: algorithm.meta_generation },
    { label: 'Version', value: algorithm.version },
    { label: 'Population Size', value: algorithm.population_size },
    { label: 'Max Generations', value: algorithm.max_generations },
    { label: 'Mutation Rate', value: algorithm.mutation_rate.toFixed(3) },
    { label: 'Mutation Strength', value: algorithm.mutation_strength.toFixed(3) },
    { label: 'Crossover Rate', value: algorithm.crossover_rate.toFixed(3) },
    { label: 'Tournament Size', value: algorithm.tournament_size },
    { label: 'Elite Count', value: algorithm.elite_count },
    { label: 'Similarity Radius', value: algorithm.similarity_radius.toFixed(3) },
    { label: 'Interpolation Weight', value: algorithm.interpolation_weight.toFixed(3) },
    { label: 'Polytope Patience', value: algorithm.polytope_patience },
    { label: 'Switch Threshold', value: algorithm.switch_threshold.toFixed(4) },
    { label: 'Switch Probability', value: algorithm.switch_probability.toFixed(3) },
    { label: 'CC Weight', value: algorithm.cc_weight.toFixed(2) },
  ];

  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold mb-4">Parameters</h2>
      <dl className="space-y-2 text-sm">
        {params.map(({ label, value }) => (
          <div key={label} className="flex justify-between">
            <dt className="text-gray-400">{label}</dt>
            <dd className="font-mono">{value}</dd>
          </div>
        ))}
      </dl>
      {algorithm.parent_id && (
        <div className="mt-4 pt-4 border-t border-slate-700">
          <span className="text-gray-400">Parent: </span>
          <Link
            to={`/meta/${algorithm.parent_id}`}
            className="text-cyan-400 hover:text-cyan-300 font-mono"
          >
            #{algorithm.parent_id}
          </Link>
        </div>
      )}
    </div>
  );
}

function FeatureWeightsCard({ weights }: { weights: Record<string, number> }) {
  const entries = Object.entries(weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  if (entries.length === 0) {
    return (
      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Feature Weights</h2>
        <p className="text-gray-400 text-sm">No feature weights defined.</p>
      </div>
    );
  }

  // Show top 15 by absolute value
  const topEntries = entries.slice(0, 15);

  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold mb-4">Feature Weights</h2>
      <div className="space-y-1 text-sm max-h-80 overflow-y-auto">
        {topEntries.map(([key, value]) => (
          <div key={key} className="flex justify-between items-center">
            <span className="text-gray-400 truncate mr-2" title={key}>
              {key}
            </span>
            <span className={`font-mono ${value > 0 ? 'text-emerald-400' : value < 0 ? 'text-red-400' : ''}`}>
              {value > 0 ? '+' : ''}{value.toFixed(3)}
            </span>
          </div>
        ))}
        {entries.length > 15 && (
          <div className="text-gray-500 text-xs pt-2">
            +{entries.length - 15} more features
          </div>
        )}
      </div>
    </div>
  );
}

function LineageCard({ lineage, currentId }: { lineage: MetaAlgorithm[]; currentId: number }) {
  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold mb-4">Lineage</h2>
      <div className="space-y-2">
        {lineage.map((algo, index) => (
          <div
            key={algo.id}
            className={`flex items-center gap-2 text-sm ${algo.id === currentId ? 'text-cyan-400' : 'text-gray-400'}`}
          >
            <span className="text-gray-600">{'→'.repeat(index)}</span>
            {algo.id === currentId ? (
              <span className="font-mono">#{algo.id} (current)</span>
            ) : (
              <Link
                to={`/meta/${algo.id}`}
                className="font-mono hover:text-white transition-colors"
              >
                #{algo.id}
              </Link>
            )}
            <span className="text-gray-600">gen {algo.meta_generation}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrialsTable({ trials }: { trials: MetaTrial[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-400 border-b border-slate-700">
            <th className="pb-2 font-medium">Trial</th>
            <th className="pb-2 font-medium">Run</th>
            <th className="pb-2 font-medium">Gens</th>
            <th className="pb-2 font-medium">Initial</th>
            <th className="pb-2 font-medium">Final</th>
            <th className="pb-2 font-medium">Improvement</th>
            <th className="pb-2 font-medium">CC Error</th>
            <th className="pb-2 font-medium">Physics %</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700/50">
          {trials.map((trial) => (
            <tr key={trial.id} className="hover:bg-slate-700/30">
              <td className="py-2 font-mono">#{trial.id}</td>
              <td className="py-2 font-mono text-gray-400">{trial.run_id ?? '-'}</td>
              <td className="py-2 font-mono">{trial.generations_run}</td>
              <td className="py-2 font-mono">{trial.initial_fitness.toFixed(4)}</td>
              <td className="py-2 font-mono">{trial.final_fitness.toFixed(4)}</td>
              <td className="py-2 font-mono">
                <span className={trial.fitness_improvement > 0 ? 'text-emerald-400' : 'text-red-400'}>
                  {trial.fitness_improvement > 0 ? '+' : ''}{trial.fitness_improvement.toFixed(4)}
                </span>
              </td>
              <td className="py-2 font-mono">{trial.best_cc_log_error.toFixed(2)}</td>
              <td className="py-2 font-mono">{(trial.physics_success_rate * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
