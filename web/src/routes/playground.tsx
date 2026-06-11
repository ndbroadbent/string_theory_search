/**
 * Playground - placeholder for interactive (K, M) flux evaluation
 *
 * The old playground depended on the dead meta-GA evaluation pipeline.
 * It will return as an interactive front-end to the cyrus-ga evaluator.
 */

import { createFileRoute } from '@tanstack/react-router';

export const Route = createFileRoute('/playground')({
  component: PlaygroundPage,
});

function PlaygroundPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-3xl mx-auto px-4 py-16 text-center">
        <h1 className="text-2xl font-bold text-white mb-4">Playground</h1>
        <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-10">
          <p className="text-cyan-400 text-lg mb-3">
            Coming soon: interactive (K, M) evaluation
          </p>
          <p className="text-gray-400 text-sm">
            Pick a polytope, enter flux vectors K and M, and run the cyrus-ga physics
            pipeline (orthogonality, tadpole, racetrack, V₀, CPL parameters) on demand.
          </p>
        </div>
      </div>
    </div>
  );
}
