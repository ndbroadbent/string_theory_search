/**
 * Playground - Interactive physics evaluation
 *
 * Allows manual evaluation of compactification configurations
 * with predefined configs (including McAllister) or custom parameters.
 */

import { createFileRoute } from '@tanstack/react-router';
import { useState, useCallback } from 'react';
import {
  getPredefinedConfigs,
  runPlaygroundEvaluation,
  getPlaygroundEvaluations,
  type PlaygroundParams,
  type PredefinedConfig,
} from '../server/playground';
import { EvaluationCard } from '../components/evaluation/EvaluationCard';
import type { Evaluation } from '../types';

export const Route = createFileRoute('/playground')({
  loader: async () => {
    const [predefinedConfigs, recentEvaluations] = await Promise.all([
      getPredefinedConfigs(),
      getPlaygroundEvaluations({ limit: 10 }),
    ]);
    return { predefinedConfigs, recentEvaluations };
  },
  component: PlaygroundPage,
});

function PlaygroundPage() {
  const { predefinedConfigs, recentEvaluations: initialEvaluations } = Route.useLoaderData();

  // Form state
  const [polytopeSource, setPolytopeSource] = useState<'db' | 'external'>('external');
  const [polytopeId, setPolytopeId] = useState<string>('');
  const [verticesJson, setVerticesJson] = useState<string>('');
  const [h11, setH11] = useState<string>('');
  const [h21, setH21] = useState<string>('');
  const [g_s, setGs] = useState<string>('0.1');
  const [kahlerModuli, setKahlerModuli] = useState<string>('[1.0]');
  const [complexModuli, setComplexModuli] = useState<string>('[1.0]');
  const [fluxF, setFluxF] = useState<string>('[0]');
  const [fluxH, setFluxH] = useState<string>('[0]');
  const [label, setLabel] = useState<string>('');

  // Result state
  const [result, setResult] = useState<Evaluation | null>(null);
  const [cached, setCached] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [recentEvaluations, setRecentEvaluations] = useState<Evaluation[]>(initialEvaluations);

  // Load predefined config
  const loadPredefined = useCallback((config: PredefinedConfig) => {
    const p = config.params;
    if (p.polytopeId !== undefined) {
      setPolytopeSource('db');
      setPolytopeId(String(p.polytopeId));
      setVerticesJson('');
    } else if (p.verticesJson) {
      setPolytopeSource('external');
      setPolytopeId('');
      setVerticesJson(p.verticesJson);
    }
    setH11(p.h11?.toString() ?? '');
    setH21(p.h21?.toString() ?? '');
    setGs(p.g_s.toString());
    setKahlerModuli(JSON.stringify(p.kahlerModuli));
    setComplexModuli(JSON.stringify(p.complexModuli));
    setFluxF(JSON.stringify(p.fluxF));
    setFluxH(JSON.stringify(p.fluxH));
    setLabel(config.name);
  }, []);

  // Run evaluation
  const runEvaluation = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setCached(false);

    try {
      // Parse JSON fields
      let kahler: number[], complex: number[], fF: number[], fH: number[];
      try {
        kahler = JSON.parse(kahlerModuli);
        complex = JSON.parse(complexModuli);
        fF = JSON.parse(fluxF);
        fH = JSON.parse(fluxH);
      } catch {
        setError('Invalid JSON in moduli or flux fields');
        setLoading(false);
        return;
      }

      const params: PlaygroundParams = {
        g_s: parseFloat(g_s),
        kahlerModuli: kahler,
        complexModuli: complex,
        fluxF: fF,
        fluxH: fH,
        label: label || undefined,
      };

      if (polytopeSource === 'db') {
        params.polytopeId = parseInt(polytopeId, 10);
      } else {
        params.verticesJson = verticesJson;
        if (h11) params.h11 = parseInt(h11, 10);
        if (h21) params.h21 = parseInt(h21, 10);
      }

      const response = await runPlaygroundEvaluation({ data: params });

      if (response.error) {
        setError(response.error);
      } else if (response.evaluation) {
        setResult(response.evaluation);
        setCached(response.cached);
        // Add to recent evaluations if not cached
        if (!response.cached) {
          setRecentEvaluations(prev => [response.evaluation!, ...prev.slice(0, 9)]);
        }
      }
    } catch (e) {
      setError(String(e));
    }

    setLoading(false);
  }, [polytopeSource, polytopeId, verticesJson, h11, h21, g_s, kahlerModuli, complexModuli, fluxF, fluxH, label]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">Playground</h1>
        <p className="text-gray-400">
          Evaluate custom compactification configurations. Results are cached by input hash -
          running the same parameters twice returns instantly from cache.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Form */}
        <div className="space-y-6">
          {/* Predefined configs */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-gray-300 font-semibold mb-3">Predefined Configurations</h3>
            <div className="flex flex-wrap gap-2">
              {predefinedConfigs.map(config => (
                <button
                  key={config.id}
                  onClick={() => loadPredefined(config)}
                  className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-sm text-gray-300 transition-colors"
                  title={config.description}
                >
                  {config.name}
                </button>
              ))}
            </div>
          </div>

          {/* Polytope source */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-gray-300 font-semibold mb-3">Polytope Source</h3>
            <div className="flex gap-4 mb-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="polytopeSource"
                  checked={polytopeSource === 'db'}
                  onChange={() => setPolytopeSource('db')}
                  className="accent-cyan-500"
                />
                <span className="text-gray-300">From Database</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="polytopeSource"
                  checked={polytopeSource === 'external'}
                  onChange={() => setPolytopeSource('external')}
                  className="accent-cyan-500"
                />
                <span className="text-gray-300">External/Custom</span>
              </label>
            </div>

            {polytopeSource === 'db' ? (
              <div>
                <label className="block text-gray-400 text-sm mb-1">Polytope ID</label>
                <input
                  type="number"
                  value={polytopeId}
                  onChange={e => setPolytopeId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono"
                  placeholder="e.g., 12345"
                />
              </div>
            ) : (
              <div className="space-y-3">
                <div>
                  <label className="block text-gray-400 text-sm mb-1">Vertices (JSON array of 4D coords)</label>
                  <textarea
                    value={verticesJson}
                    onChange={e => setVerticesJson(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono text-sm h-24"
                    placeholder='[[1,0,0,0], [0,1,0,0], ...]'
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-gray-400 text-sm mb-1">h11</label>
                    <input
                      type="number"
                      value={h11}
                      onChange={e => setH11(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono"
                    />
                  </div>
                  <div>
                    <label className="block text-gray-400 text-sm mb-1">h21</label>
                    <input
                      type="number"
                      value={h21}
                      onChange={e => setH21(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Physics parameters */}
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-gray-300 font-semibold mb-3">Physics Parameters</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-gray-400 text-sm mb-1">String coupling g_s</label>
                <input
                  type="number"
                  step="0.0001"
                  value={g_s}
                  onChange={e => setGs(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-1">Kahler moduli (JSON array)</label>
                <textarea
                  value={kahlerModuli}
                  onChange={e => setKahlerModuli(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono text-sm h-20"
                  placeholder="[1.0, 2.0, ...]"
                />
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-1">Complex moduli (JSON array)</label>
                <textarea
                  value={complexModuli}
                  onChange={e => setComplexModuli(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono text-sm h-20"
                  placeholder="[1.0, 1.0, ...]"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-gray-400 text-sm mb-1">F flux (JSON array)</label>
                  <input
                    type="text"
                    value={fluxF}
                    onChange={e => setFluxF(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono text-sm"
                    placeholder="[1, 0, 0, 0]"
                  />
                </div>
                <div>
                  <label className="block text-gray-400 text-sm mb-1">H flux (JSON array)</label>
                  <input
                    type="text"
                    value={fluxH}
                    onChange={e => setFluxH(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white font-mono text-sm"
                    placeholder="[0, 1, 0, 0]"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Label and run */}
          <div className="bg-slate-800 rounded-lg p-4">
            <div className="mb-4">
              <label className="block text-gray-400 text-sm mb-1">Label (optional)</label>
              <input
                type="text"
                value={label}
                onChange={e => setLabel(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white"
                placeholder="e.g., My test configuration"
              />
            </div>
            <button
              onClick={runEvaluation}
              disabled={loading}
              className="w-full bg-cyan-600 hover:bg-cyan-700 disabled:bg-slate-700 disabled:text-gray-500 text-white font-semibold py-3 px-4 rounded transition-colors"
            >
              {loading ? 'Evaluating...' : 'Run Evaluation'}
            </button>
          </div>
        </div>

        {/* Right: Results */}
        <div className="space-y-6">
          {/* Current result */}
          {error && (
            <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-4">
              <h3 className="text-red-400 font-semibold mb-2">Error</h3>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}

          {result && (
            <div>
              <div className="flex items-center gap-3 mb-3">
                <h3 className="text-gray-300 font-semibold">Result</h3>
                {cached && (
                  <span className="px-2 py-0.5 bg-emerald-900/50 text-emerald-400 border border-emerald-700/50 rounded text-xs">
                    Cached
                  </span>
                )}
              </div>
              <EvaluationCard evaluation={result} showPolytope={false} />
            </div>
          )}

          {/* Recent evaluations */}
          {recentEvaluations.length > 0 && (
            <div>
              <h3 className="text-gray-300 font-semibold mb-3">Recent Playground Evaluations</h3>
              <div className="space-y-3">
                {recentEvaluations.map(ev => (
                  <EvaluationCard key={ev.id} evaluation={ev} compact />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
