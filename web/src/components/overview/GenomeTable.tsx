/**
 * Table showing top genomes
 */

import type { GenomeResult } from '../../types';

interface GenomeTableProps {
  genomes: GenomeResult[];
  onSelect: (genome: GenomeResult) => void;
  selectedId?: number;
}

export function GenomeTable({ genomes, onSelect, selectedId }: GenomeTableProps) {
  const topGenomes = genomes.slice(0, 20);

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
      <div className="p-3 border-b border-slate-700">
        <h3 className="text-sm font-medium text-gray-300">
          Top Genomes ({genomes.length} total)
        </h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-slate-900/50">
              <th className="text-left p-2 text-gray-400 font-medium">Polytope</th>
              <th className="text-right p-2 text-gray-400 font-medium">Fitness</th>
              <th className="text-right p-2 text-gray-400 font-medium">α_em</th>
              <th className="text-right p-2 text-gray-400 font-medium">α_s</th>
              <th className="text-right p-2 text-gray-400 font-medium">sin²θ</th>
              <th className="text-right p-2 text-gray-400 font-medium">h11</th>
              <th className="text-right p-2 text-gray-400 font-medium">h21</th>
            </tr>
          </thead>
          <tbody>
            {topGenomes.map((g) => (
              <tr
                key={g.genome.polytope_id}
                onClick={() => onSelect(g)}
                className={`cursor-pointer transition-colors ${
                  g.genome.polytope_id === selectedId
                    ? 'bg-cyan-900/30'
                    : 'hover:bg-slate-700/50'
                }`}
              >
                <td className="p-2 text-cyan-400 font-mono">
                  #{g.genome.polytope_id}
                </td>
                <td className="p-2 text-right text-green-400 font-mono">
                  {g.fitness.toFixed(4)}
                </td>
                <td className="p-2 text-right text-gray-300 font-mono">
                  {g.physics.alpha_em.toExponential(2)}
                </td>
                <td className="p-2 text-right text-gray-300 font-mono">
                  {g.physics.alpha_s.toExponential(2)}
                </td>
                <td className="p-2 text-right text-gray-300 font-mono">
                  {g.physics.sin2_theta_w.toFixed(3)}
                </td>
                <td className="p-2 text-right text-gray-500">{g.genome.h11}</td>
                <td className="p-2 text-right text-gray-500">{g.genome.h21}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
