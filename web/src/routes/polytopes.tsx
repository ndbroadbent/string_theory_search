import { createFileRoute, Link } from '@tanstack/react-router';
import { getPolytopes } from '../server/heuristics';

export const Route = createFileRoute('/polytopes')({
  component: PolytopesPage,
  loader: async () => {
    const data = await getPolytopes({ data: { limit: 100, offset: 0 } });
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
            <span className="text-cyan-400 font-mono">{total.toLocaleString()}</span> polytopes in database
          </p>
        </div>

        <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-700/50 text-gray-300">
                <th className="px-4 py-3 text-left font-medium">ID</th>
                <th className="px-4 py-3 text-left font-medium">h11</th>
                <th className="px-4 py-3 text-left font-medium">h21</th>
                <th className="px-4 py-3 text-left font-medium">Vertices</th>
                <th className="px-4 py-3 text-left font-medium">Evals</th>
                <th className="px-4 py-3 text-left font-medium">Mean Fitness</th>
              </tr>
            </thead>
            <tbody>
              {polytopes.map((p) => (
                <tr key={p.id} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                  <td className="px-4 py-2">
                    <Link
                      to="/polytope/$id"
                      params={{ id: String(p.id) }}
                      className="text-cyan-400 hover:text-cyan-300 font-mono"
                    >
                      {p.id}
                    </Link>
                  </td>
                  <td className="px-4 py-2 text-gray-300 font-mono">{p.h11}</td>
                  <td className="px-4 py-2 text-gray-300 font-mono">{p.h21}</td>
                  <td className="px-4 py-2 text-gray-300 font-mono">{p.vertex_count}</td>
                  <td className="px-4 py-2 text-gray-300 font-mono">{p.eval_count}</td>
                  <td className="px-4 py-2 text-gray-300 font-mono">
                    {p.fitness_mean !== null ? p.fitness_mean.toFixed(6) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-4 text-gray-500 text-sm">Showing first 100 polytopes</p>
      </div>
    </div>
  );
}
