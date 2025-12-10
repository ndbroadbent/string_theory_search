/**
 * Shared evaluation display card component
 * Used uniformly across Dashboard, Polytope page, Playground, etc.
 */

import { Link } from '@tanstack/react-router';
import type { Evaluation, EvaluationSource } from '../../types';
import { TARGET_PHYSICS } from '../../types';

interface EvaluationCardProps {
  evaluation: Evaluation;
  showPolytope?: boolean;
  compact?: boolean;
}

const SOURCE_BADGES: Record<EvaluationSource, { label: string; className: string }> = {
  ga: { label: 'GA', className: 'bg-blue-900/50 text-blue-400 border-blue-700/50' },
  playground: { label: 'Playground', className: 'bg-purple-900/50 text-purple-400 border-purple-700/50' },
  test: { label: 'Test', className: 'bg-gray-900/50 text-gray-400 border-gray-700/50' },
};

function formatScientific(value: number | null, precision = 3): string {
  if (value === null || value === undefined) return '-';
  if (value === 0) return '0';
  return value.toExponential(precision);
}

function computeCCLogError(cc: number | null): number | null {
  if (cc === null || cc === undefined) return null;
  const absCC = Math.abs(cc);
  if (absCC < 1e-200) return 200; // Essentially zero
  return Math.abs(Math.log10(absCC) - Math.log10(TARGET_PHYSICS.cosmological_constant));
}

export function EvaluationCard({ evaluation, showPolytope = true, compact = false }: EvaluationCardProps) {
  const ccLogError = computeCCLogError(evaluation.cosmological_constant);
  const source = evaluation.source ?? 'ga';
  const sourceBadge = SOURCE_BADGES[source];

  // Determine h11/h21 display values
  const h11 = evaluation.h11;
  const h21 = evaluation.h21;
  const deltaH = h11 !== null && h21 !== null ? Math.abs(h11 - h21) : null;

  if (compact) {
    return (
      <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className={`px-2 py-0.5 rounded text-xs border ${sourceBadge.className}`}>
              {sourceBadge.label}
            </span>
            {evaluation.label && (
              <span className="text-gray-300 text-sm font-medium">{evaluation.label}</span>
            )}
            <span className="text-gray-500 text-xs">#{evaluation.id}</span>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div>
              <span className="text-gray-500">Fitness:</span>
              <span className="text-white font-mono ml-1">{evaluation.fitness.toFixed(6)}</span>
            </div>
            {ccLogError !== null && (
              <div>
                <span className="text-gray-500">CC Error:</span>
                <span className={`font-mono ml-1 ${ccLogError < 50 ? 'text-emerald-400' : ccLogError < 100 ? 'text-amber-400' : 'text-red-400'}`}>
                  {ccLogError.toFixed(1)} orders
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          <span className={`px-2 py-0.5 rounded text-xs border ${sourceBadge.className}`}>
            {sourceBadge.label}
          </span>
          {evaluation.label && (
            <span className="text-gray-200 font-medium">{evaluation.label}</span>
          )}
          <span className="text-gray-500 text-sm">ID: {evaluation.id}</span>
        </div>
        <div className="text-right text-sm">
          {showPolytope && evaluation.polytope_id !== -1 && (
            <Link
              to="/polytope/$id"
              params={{ id: String(evaluation.polytope_id) }}
              className="text-cyan-400 hover:text-cyan-300"
            >
              Polytope #{evaluation.polytope_id}
            </Link>
          )}
          {evaluation.polytope_id === -1 && (
            <span className="text-gray-500">External Polytope</span>
          )}
        </div>
      </div>

      {/* Hodge numbers and geometry */}
      {(h11 !== null || h21 !== null) && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div className="bg-slate-700/50 rounded p-2">
            <div className="text-gray-400 text-xs">h11</div>
            <div className="text-white font-mono">{h11 ?? '-'}</div>
          </div>
          <div className="bg-slate-700/50 rounded p-2">
            <div className="text-gray-400 text-xs">h21</div>
            <div className="text-white font-mono">{h21 ?? '-'}</div>
          </div>
          <div className="bg-slate-700/50 rounded p-2">
            <div className="text-gray-400 text-xs">|h11-h21|</div>
            <div className={`font-mono ${deltaH === 3 ? 'text-emerald-400' : 'text-white'}`}>
              {deltaH ?? '-'}
            </div>
          </div>
          <div className="bg-slate-700/50 rounded p-2">
            <div className="text-gray-400 text-xs">g_s</div>
            <div className="text-white font-mono">{evaluation.g_s?.toExponential(4) ?? '-'}</div>
          </div>
        </div>
      )}

      {/* Main fitness and CC */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
        <div className="bg-gradient-to-br from-emerald-900/30 to-emerald-800/20 border border-emerald-700/30 rounded p-3">
          <div className="text-emerald-400 text-xs mb-1">Fitness</div>
          <div className="text-white font-mono text-lg">{evaluation.fitness.toFixed(6)}</div>
        </div>
        <div className="bg-gradient-to-br from-cyan-900/30 to-cyan-800/20 border border-cyan-700/30 rounded p-3">
          <div className="text-cyan-400 text-xs mb-1">Cosmological Constant</div>
          <div className="text-white font-mono text-lg">
            {formatScientific(evaluation.cosmological_constant)}
          </div>
          {ccLogError !== null && (
            <div className={`text-xs ${ccLogError < 50 ? 'text-emerald-400' : ccLogError < 100 ? 'text-amber-400' : 'text-red-400'}`}>
              {ccLogError.toFixed(1)} orders from target
            </div>
          )}
        </div>
        <div className="bg-slate-700/50 rounded p-3">
          <div className="text-gray-400 text-xs mb-1">N_gen</div>
          <div className={`font-mono text-lg ${evaluation.n_generations === 3 ? 'text-emerald-400' : 'text-white'}`}>
            {evaluation.n_generations ?? '-'}
          </div>
        </div>
      </div>

      {/* Gauge couplings */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-slate-700/30 rounded p-2">
          <div className="text-gray-400 text-xs mb-1">
            alpha_em
            <span className="text-gray-500 ml-1">(target: {TARGET_PHYSICS.alpha_em.toExponential(3)})</span>
          </div>
          <div className="text-white font-mono">{formatScientific(evaluation.alpha_em)}</div>
        </div>
        <div className="bg-slate-700/30 rounded p-2">
          <div className="text-gray-400 text-xs mb-1">
            alpha_s
            <span className="text-gray-500 ml-1">(target: {TARGET_PHYSICS.alpha_s})</span>
          </div>
          <div className="text-white font-mono">{evaluation.alpha_s?.toFixed(4) ?? '-'}</div>
        </div>
        <div className="bg-slate-700/30 rounded p-2">
          <div className="text-gray-400 text-xs mb-1">
            sin2_theta_w
            <span className="text-gray-500 ml-1">(target: {TARGET_PHYSICS.sin2_theta_w})</span>
          </div>
          <div className="text-white font-mono">{evaluation.sin2_theta_w?.toFixed(5) ?? '-'}</div>
        </div>
      </div>

      {/* Error status */}
      {!evaluation.success && evaluation.error && (
        <div className="bg-red-900/30 border border-red-700/50 rounded p-3 mb-4">
          <div className="text-red-400 text-xs mb-1">Error</div>
          <div className="text-red-300 text-sm">{evaluation.error}</div>
        </div>
      )}

      {/* Metadata */}
      <div className="flex items-center justify-between text-xs text-gray-500 pt-3 border-t border-slate-700">
        <div className="flex items-center gap-4">
          {evaluation.model_version && (
            <span>Model: v{evaluation.model_version}</span>
          )}
          {evaluation.run_id && (
            <span>Run: {evaluation.run_id}</span>
          )}
          {evaluation.generation !== null && (
            <span>Gen: {evaluation.generation}</span>
          )}
        </div>
        <span>{new Date(evaluation.created_at).toLocaleString()}</span>
      </div>

      {/* Collapsible moduli details */}
      {(evaluation.kahler_moduli || evaluation.flux_f || evaluation.flux_h) && (
        <details className="mt-4 pt-3 border-t border-slate-700">
          <summary className="text-gray-400 text-sm cursor-pointer hover:text-gray-300">
            Show moduli and flux details
          </summary>
          <div className="mt-3 space-y-3">
            {evaluation.kahler_moduli && (
              <div className="bg-slate-900 rounded p-2 overflow-x-auto">
                <div className="text-gray-400 text-xs mb-1">Kahler moduli ({evaluation.kahler_moduli.length})</div>
                <div className="text-white font-mono text-xs">
                  [{evaluation.kahler_moduli.slice(0, 10).map(v => v.toFixed(4)).join(', ')}
                  {evaluation.kahler_moduli.length > 10 && `, ... (${evaluation.kahler_moduli.length - 10} more)`}]
                </div>
              </div>
            )}
            {evaluation.complex_moduli && (
              <div className="bg-slate-900 rounded p-2 overflow-x-auto">
                <div className="text-gray-400 text-xs mb-1">Complex moduli ({evaluation.complex_moduli.length})</div>
                <div className="text-white font-mono text-xs">
                  [{evaluation.complex_moduli.slice(0, 10).map(v => v.toFixed(4)).join(', ')}
                  {evaluation.complex_moduli.length > 10 && `, ... (${evaluation.complex_moduli.length - 10} more)`}]
                </div>
              </div>
            )}
            {evaluation.flux_f && (
              <div className="bg-slate-900 rounded p-2 overflow-x-auto">
                <div className="text-gray-400 text-xs mb-1">F flux ({evaluation.flux_f.length})</div>
                <div className="text-white font-mono text-xs">[{evaluation.flux_f.join(', ')}]</div>
              </div>
            )}
            {evaluation.flux_h && (
              <div className="bg-slate-900 rounded p-2 overflow-x-auto">
                <div className="text-gray-400 text-xs mb-1">H flux ({evaluation.flux_h.length})</div>
                <div className="text-white font-mono text-xs">[{evaluation.flux_h.join(', ')}]</div>
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );
}

export default EvaluationCard;
