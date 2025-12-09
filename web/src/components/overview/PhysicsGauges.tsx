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
  showLogDiff?: boolean; // Show orders of magnitude difference instead of percentage
}

function formatSuperscript(v: number, format: 'scientific' | 'decimal'): React.ReactNode {
  if (format === 'decimal') return v.toFixed(4);

  const exp = v.toExponential(3);
  const match = exp.match(/^(-?\d+\.?\d*)e([+-]?\d+)$/);
  if (!match) return exp;

  const [, mantissa, exponent] = match;
  const superscriptDigits: Record<string, string> = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '-': '⁻', '+': '',
  };
  const supExp = exponent.split('').map(c => superscriptDigits[c] || c).join('');

  return <>{mantissa} × 10{supExp}</>;
}

function Gauge({ label, computed, target, format = 'scientific', showLogDiff }: GaugeProps) {
  // Calculate how close we are using log scale
  const logComputed = Math.log10(Math.abs(computed) + 1e-200);
  const logTarget = Math.log10(Math.abs(target) + 1e-200);

  // Log-based percentage: compare exponents directly
  // e.g., 10^10 vs 10^100 → 10/100 = 10%
  // For negative exponents: 10^-10 vs 10^-100 → 10/100 = 10%
  const absLogComputed = Math.abs(logComputed);
  const absLogTarget = Math.abs(logTarget);
  const percentage = absLogTarget > 0
    ? Math.min(100, (Math.min(absLogComputed, absLogTarget) / Math.max(absLogComputed, absLogTarget)) * 100)
    : (absLogComputed < 1 ? 100 : 0);

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
          Computed: {Number.isFinite(computed) ? formatSuperscript(computed, format) : 'N/A'}
        </span>
        <span className="text-gray-500">Target: {formatSuperscript(target, format)}</span>
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

      <Gauge
        label="Cosmological Constant (Λ)"
        computed={physics.cosmological_constant}
        target={TARGET_PHYSICS.cosmological_constant}
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
