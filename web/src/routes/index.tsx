/**
 * Overview Dashboard - Main landing page
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { useState, useEffect } from 'react';
import { getAllEvaluations } from '../server/results';
import { FitnessScatter } from '../components/overview/FitnessScatter';
import { PhysicsGauges } from '../components/overview/PhysicsGauges';
import { GenomeTable } from '../components/overview/GenomeTable';
import { useVisualizationStore } from '../stores/visualization';
import type { GenomeResult } from '../types';

export const Route = createFileRoute('/')({
  component: Dashboard,
  loader: async () => {
    const genomes = await getAllEvaluations({ data: { limit: 200 } });
    return { genomes };
  },
});

function Dashboard() {
  const { genomes } = Route.useLoaderData();

  // Initialize with first genome if available
  const [selectedGenome, setSelectedGenome] = useState<GenomeResult | null>(
    () => genomes.length > 0 ? genomes[0] : null
  );

  const { setAllGenomes, setSelectedGenome: setStoreGenome } =
    useVisualizationStore();

  // Sync genomes to store
  useEffect(() => {
    setAllGenomes(genomes);
    if (genomes.length > 0 && selectedGenome) {
      setStoreGenome(selectedGenome);
    }
  }, [genomes, setAllGenomes, selectedGenome, setStoreGenome]);

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

        {/* Stats bar */}
        <div className="mb-6 flex items-center gap-6 text-sm">
          <span className="text-gray-400">
            Showing <span className="text-cyan-400 font-mono">{genomes.length}</span> best evaluations
          </span>
          {selectedGenome && (
            <div className="ml-auto flex items-center gap-2">
              {selectedGenome.runRef && (
                <Link
                  to="/meta/gen/$gen/algo/$algo/run/$run"
                  params={{
                    gen: String(selectedGenome.runRef.metaGeneration),
                    algo: String(selectedGenome.runRef.algorithmId),
                    run: String(selectedGenome.runRef.runNumber),
                  }}
                  className="px-4 py-1.5 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded transition-colors"
                >
                  View Run →
                </Link>
              )}
              <Link
                to="/polytope/$id"
                params={{ id: String(selectedGenome.genome.polytope_id) }}
                className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-700 text-white text-sm rounded transition-colors"
              >
                View Polytope →
              </Link>
            </div>
          )}
        </div>

        {genomes.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-400">No evaluations found. Run the search to generate data.</div>
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
