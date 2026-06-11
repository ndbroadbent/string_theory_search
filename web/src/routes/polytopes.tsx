/**
 * Polytopes list - per-polytope search progress from the cyrus-ga pool
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { getPolytopesList } from '../server/ga';
import { fmtAgo, fmtInt, fmtNum } from '../lib/format';

export const Route = createFileRoute('/polytopes')({
  component: PolytopesPage,
  loader: async () => {
    const data = await getPolytopesList({ data: { limit: 500, offset: 0 } });
    return data;
  },
});

function PolytopesPage() {
  const { total, polytopes } = Route.useLoaderData();

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white mb-2">Polytopes</h1>
          <p className="text-gray-400">
            <span className="text-cyan-400 font-mono">{total.toLocaleString()}</span>{' '}
            polytopes searched, sorted by valid candidates and best fitness
          </p>
        </div>

        <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-700/50 text-gray-300">
                <th className="px-4 py-3 text-left font-medium">Polytope</th>
                <th className="px-4 py-3 text-right font-medium">h11</th>
                <th className="px-4 py-3 text-right font-medium">h21</th>
                <th className="px-4 py-3 text-center font-medium">Favorable</th>
                <th className="px-4 py-3 text-right font-medium">Q_D3 bound</th>
                <th className="px-4 py-3 text-right font-medium">Rounds</th>
                <th className="px-4 py-3 text-right font-medium">Evals</th>
                <th className="px-4 py-3 text-right font-medium">Valid</th>
                <th className="px-4 py-3 text-right font-medium">Best Fitness</th>
                <th className="px-4 py-3 text-right font-medium">Updated</th>
              </tr>
            </thead>
            <tbody>
              {polytopes.map((p) => (
                <tr key={p.id} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                  <td className="px-4 py-2">
                    <Link
                      to="/polytope/$id"
                      params={{ id: p.id }}
                      className="text-cyan-400 hover:text-cyan-300 font-mono"
                    >
                      {p.id}
                    </Link>
                  </td>
                  <td className="px-4 py-2 text-right text-gray-300 font-mono">{p.h11 ?? '–'}</td>
                  <td className="px-4 py-2 text-right text-gray-300 font-mono">{p.h21 ?? '–'}</td>
                  <td className="px-4 py-2 text-center">
                    {p.favorable == null ? (
                      <span className="text-gray-500">–</span>
                    ) : p.favorable ? (
                      <span className="text-green-400">yes</span>
                    ) : (
                      <span className="text-gray-500">no</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right text-gray-300 font-mono">
                    {fmtNum(p.q_d3, 1)}
                  </td>
                  <td className="px-4 py-2 text-right text-gray-300 font-mono">{fmtInt(p.rounds)}</td>
                  <td className="px-4 py-2 text-right text-gray-300 font-mono">{fmtInt(p.evals)}</td>
                  <td className="px-4 py-2 text-right font-mono">
                    {p.valid_seen > 0 ? (
                      <span className="text-green-400">{fmtInt(p.valid_seen)}</span>
                    ) : (
                      <span className="text-gray-500">0</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right text-gray-300 font-mono">
                    {fmtNum(p.best_fitness, 2)}
                  </td>
                  <td className="px-4 py-2 text-right text-gray-500 font-mono">
                    {p.updated_at ? fmtAgo(p.updated_at) : '–'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {polytopes.length === 0 && (
            <div className="px-4 py-8 text-center text-gray-400 text-sm">
              No polytopes ingested yet. Is the ingester running?
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
