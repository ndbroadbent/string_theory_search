/**
 * Overview Dashboard - Main landing page
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { useState, useEffect, useMemo } from 'react';
import { listRuns, getAllGenomes } from '../server/results';
import { FitnessScatter } from '../components/overview/FitnessScatter';
import { PhysicsGauges } from '../components/overview/PhysicsGauges';
import { GenomeTable } from '../components/overview/GenomeTable';
import { useVisualizationStore } from '../stores/visualization';
import type { GenomeResult, RunInfo } from '../types';

export const Route = createFileRoute('/')({
  component: Dashboard,
  loader: async () => {
    const runs = await listRuns();
    return { runs };
  },
});

function Dashboard() {
  const { runs: allRuns } = Route.useLoaderData();

  // Filter out empty runs and sort by best fitness descending
  const runs = useMemo(
    () =>
      allRuns
        .filter((r) => r.genomeCount > 0)
        .sort((a, b) => b.bestFitness - a.bestFitness),
    [allRuns]
  );

  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [genomes, setGenomes] = useState<GenomeResult[]>([]);
  const [selectedGenome, setSelectedGenome] = useState<GenomeResult | null>(null);
  const [loading, setLoading] = useState(false);

  const { setRuns, setAllGenomes, setSelectedGenome: setStoreGenome } =
    useVisualizationStore();

  // Sync runs to store and set initial selection
  useEffect(() => {
    setRuns(runs);
    if (!selectedRunId && runs.length > 0) {
      setSelectedRunId(runs[0].id);
    }
  }, [runs, setRuns, selectedRunId]);

  // Load genomes when run changes
  useEffect(() => {
    if (!selectedRunId) return;

    setLoading(true);
    getAllGenomes({ data: { runId: selectedRunId } })
      .then((data) => {
        setGenomes(data);
        setAllGenomes(data);
        if (data.length > 0) {
          setSelectedGenome(data[0]);
          setStoreGenome(data[0]);
        }
      })
      .finally(() => setLoading(false));
  }, [selectedRunId, setAllGenomes, setStoreGenome]);

  const handleGenomeSelect = (genome: GenomeResult) => {
    setSelectedGenome(genome);
    setStoreGenome(genome);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white mb-2">
            String Theory Landscape Explorer
          </h1>
          <p className="text-gray-400">
            Visualizing Calabi-Yau compactifications and Standard Model physics
          </p>
        </div>

        {/* Run Selector */}
        <div className="mb-6 flex items-center gap-4">
          <label className="text-sm text-gray-400">Select Run:</label>
          <select
            value={selectedRunId ?? ''}
            onChange={(e) => setSelectedRunId(e.target.value || null)}
            className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-gray-200 text-sm focus:outline-none focus:border-cyan-500"
          >
            {runs.length === 0 && (
              <option value="">No runs available</option>
            )}
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.id} - {run.genomeCount} genomes (best: {run.bestFitness.toFixed(4)})
              </option>
            ))}
          </select>

          {selectedGenome && (
            <Link
              to="/genome/$polytopeId"
              params={{ polytopeId: String(selectedGenome.genome.polytope_id) }}
              search={{ runId: selectedRunId! }}
              className="ml-auto px-4 py-1.5 bg-cyan-600 hover:bg-cyan-700 text-white text-sm rounded transition-colors"
            >
              View Detail →
            </Link>
          )}
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-400">Loading genomes...</div>
          </div>
        ) : genomes.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-400">No genomes found in this run</div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left column - Scatter plots */}
            <div className="lg:col-span-2 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FitnessScatter
                  genomes={genomes}
                  xAxis="alpha_em"
                  yAxis="fitness"
                  onSelect={handleGenomeSelect}
                  selectedId={selectedGenome?.genome.polytope_id}
                />
                <FitnessScatter
                  genomes={genomes}
                  xAxis="alpha_s"
                  yAxis="fitness"
                  onSelect={handleGenomeSelect}
                  selectedId={selectedGenome?.genome.polytope_id}
                />
              </div>

              <FitnessScatter
                genomes={genomes}
                xAxis="alpha_em"
                yAxis="alpha_s"
                onSelect={handleGenomeSelect}
                selectedId={selectedGenome?.genome.polytope_id}
              />

              <GenomeTable
                genomes={genomes}
                onSelect={handleGenomeSelect}
                selectedId={selectedGenome?.genome.polytope_id}
              />
            </div>

            {/* Right column - Selected genome details */}
            <div className="space-y-6">
              {selectedGenome ? (
                <>
                  <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                    <h3 className="text-sm font-medium text-gray-300 mb-3">
                      Selected: Polytope #{selectedGenome.genome.polytope_id}
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Fitness:</span>
                        <span className="text-green-400 font-mono">
                          {selectedGenome.fitness.toFixed(6)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Hodge numbers:</span>
                        <span className="text-gray-300">
                          h11={selectedGenome.genome.h11}, h21={selectedGenome.genome.h21}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">String coupling:</span>
                        <span className="text-gray-300 font-mono">
                          {selectedGenome.genome.g_s.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  </div>

                  <PhysicsGauges physics={selectedGenome.physics} />
                </>
              ) : (
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 text-center text-gray-400">
                  Select a genome to view details
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
