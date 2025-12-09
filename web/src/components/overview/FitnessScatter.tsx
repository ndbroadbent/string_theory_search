/**
 * Scatter plot showing fitness distribution
 */

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { GenomeResult } from '../../types';
import { TARGET_PHYSICS } from '../../types';

interface FitnessScatterProps {
  genomes: GenomeResult[];
  xAxis: 'alpha_em' | 'alpha_s' | 'sin2_theta_w' | 'cosmological_constant';
  yAxis: 'alpha_em' | 'alpha_s' | 'sin2_theta_w' | 'fitness';
  onSelect?: (genome: GenomeResult) => void;
  selectedId?: number;
}

const axisLabels: Record<string, string> = {
  alpha_em: 'α_em (Fine Structure)',
  alpha_s: 'α_s (Strong Coupling)',
  sin2_theta_w: 'sin²θ_W (Weinberg)',
  cosmological_constant: 'Λ (Cosmological)',
  fitness: 'Fitness',
};

const targetLines: Record<string, number> = {
  alpha_em: TARGET_PHYSICS.alpha_em,
  alpha_s: TARGET_PHYSICS.alpha_s,
  sin2_theta_w: TARGET_PHYSICS.sin2_theta_w,
};

export function FitnessScatter({
  genomes,
  xAxis,
  yAxis,
  onSelect,
  selectedId,
}: FitnessScatterProps) {
  const data = genomes.map((g) => ({
    x: xAxis === 'fitness' ? g.fitness : g.physics[xAxis],
    y: yAxis === 'fitness' ? g.fitness : g.physics[yAxis],
    fitness: g.fitness,
    polytopeId: g.genome.polytope_id,
    genome: g,
  }));

  // Filter out NaN and Infinity
  const validData = data.filter(
    (d) =>
      Number.isFinite(d.x) &&
      Number.isFinite(d.y) &&
      Math.abs(d.x) < 1e10 &&
      Math.abs(d.y) < 1e10
  );

  const handleClick = (data: { genome: GenomeResult }) => {
    if (onSelect) {
      onSelect(data.genome);
    }
  };

  return (
    <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        {axisLabels[xAxis]} vs {axisLabels[yAxis]}
      </h3>

      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 30, left: 50 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            type="number"
            dataKey="x"
            name={axisLabels[xAxis]}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickFormatter={(v) => v.toExponential(1)}
            label={{
              value: axisLabels[xAxis],
              position: 'bottom',
              fill: '#94a3b8',
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={axisLabels[yAxis]}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickFormatter={(v) => (v < 0.01 ? v.toExponential(1) : v.toFixed(3))}
            label={{
              value: axisLabels[yAxis],
              angle: -90,
              position: 'insideLeft',
              fill: '#94a3b8',
              fontSize: 11,
            }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="bg-slate-900/95 border border-slate-600 rounded p-2 text-xs">
                  <p className="text-cyan-400">Polytope #{d.polytopeId}</p>
                  <p className="text-gray-300">
                    {axisLabels[xAxis]}: {d.x.toExponential(3)}
                  </p>
                  <p className="text-gray-300">
                    {axisLabels[yAxis]}: {d.y.toExponential(3)}
                  </p>
                  <p className="text-green-400">Fitness: {d.fitness.toFixed(4)}</p>
                </div>
              );
            }}
          />

          {/* Target reference lines */}
          {xAxis in targetLines && (
            <line
              x1={targetLines[xAxis]}
              y1={0}
              x2={targetLines[xAxis]}
              y2={1}
              stroke="#22d3ee"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
          )}

          <Scatter
            data={validData}
            fill="#06b6d4"
            onClick={handleClick}
            cursor="pointer"
          >
            {validData.map((entry, index) => (
              <Cell
                key={index}
                fill={
                  entry.polytopeId === selectedId
                    ? '#f472b6'
                    : `rgba(6, 182, 212, ${0.3 + entry.fitness * 0.7})`
                }
                stroke={entry.polytopeId === selectedId ? '#f472b6' : undefined}
                strokeWidth={entry.polytopeId === selectedId ? 2 : 0}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
