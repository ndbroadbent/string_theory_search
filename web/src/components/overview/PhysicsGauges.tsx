/**
 * Physics comparison gauges - computed vs target
 */

import type { PhysicsOutput } from '../../types';
import { TARGET_PHYSICS } from '../../types';

interface PhysicsGaugesProps {
  physics: PhysicsOutput;
}

interface GaugeProps {
  label: string;
  computed: number;
  target: number;
  format?: 'scientific' | 'decimal';
}

function Gauge({ label, computed, target, format = 'scientific' }: GaugeProps) {
  // Calculate how close we are (log scale for very different magnitudes)
  const logComputed = Math.log10(Math.abs(computed) + 1e-50);
  const logTarget = Math.log10(Math.abs(target) + 1e-50);
  const logDiff = Math.abs(logComputed - logTarget);

  // Convert to percentage (0 = way off, 100 = exact match)
  // Use exp decay: at 0 orders of magnitude diff = 100%, at 3 orders = ~5%
  const accuracy = Math.exp(-logDiff) * 100;
  const percentage = Math.min(100, Math.max(0, accuracy));

  const formatValue = (v: number) => {
    if (format === 'decimal') return v.toFixed(4);
    return v.toExponential(3);
  };

  // Color based on accuracy
  const getColor = () => {
    if (percentage > 90) return 'bg-green-500';
    if (percentage > 50) return 'bg-yellow-500';
    if (percentage > 20) return 'bg-orange-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-500">{percentage.toFixed(1)}% match</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${getColor()} transition-all duration-300`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-cyan-400">
          Computed: {Number.isFinite(computed) ? formatValue(computed) : 'N/A'}
        </span>
        <span className="text-gray-500">Target: {formatValue(target)}</span>
      </div>
    </div>
  );
}

export function PhysicsGauges({ physics }: PhysicsGaugesProps) {
  return (
    <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 space-y-4">
      <h3 className="text-sm font-medium text-gray-300">Physics Comparison</h3>

      <Gauge
        label="Fine Structure Constant (α_em)"
        computed={physics.alpha_em}
        target={TARGET_PHYSICS.alpha_em}
      />

      <Gauge
        label="Strong Coupling (α_s)"
        computed={physics.alpha_s}
        target={TARGET_PHYSICS.alpha_s}
      />

      <Gauge
        label="Weinberg Angle (sin²θ_W)"
        computed={physics.sin2_theta_w}
        target={TARGET_PHYSICS.sin2_theta_w}
        format="decimal"
      />

      <Gauge
        label="Generations"
        computed={physics.n_generations}
        target={TARGET_PHYSICS.n_generations}
        format="decimal"
      />

      {/* Internal parameters (no target) */}
      <div className="pt-2 border-t border-slate-700 space-y-2">
        <p className="text-xs text-gray-500">Internal Parameters:</p>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-gray-400">CY Volume: </span>
            <span className="text-gray-300">
              {physics.cy_volume.toExponential(2)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">g_s: </span>
            <span className="text-gray-300">
              {physics.string_coupling.toFixed(4)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">|W|: </span>
            <span className="text-gray-300">
              {physics.superpotential_abs.toExponential(2)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Tadpole: </span>
            <span className="text-gray-300">{physics.flux_tadpole}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
