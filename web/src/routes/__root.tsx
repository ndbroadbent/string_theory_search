/**
 * Root layout for the visualization dashboard
 */

import { HeadContent, Scripts, createRootRoute, Outlet, Link } from '@tanstack/react-router';
import appCss from '../styles.css?url';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'String Theory Landscape Explorer' },
      { name: 'description', content: 'Visualizing Calabi-Yau compactifications' },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
    ],
  }),
  component: RootDocument,
});

function RootDocument() {
  return (
    <html lang="en" className="dark">
      <head>
        <HeadContent />
      </head>
      <body className="bg-slate-900 text-white antialiased">
        {/* Navigation */}
        <nav className="bg-slate-800/80 border-b border-slate-700 px-4 py-2">
          <div className="max-w-7xl mx-auto flex items-center gap-6">
            <Link
              to="/"
              className="text-gray-300 hover:text-white text-sm font-medium transition-colors [&.active]:text-cyan-400"
            >
              Overview
            </Link>
            <Link
              to="/heuristics"
              className="text-gray-300 hover:text-white text-sm font-medium transition-colors [&.active]:text-cyan-400"
            >
              Heuristics Explorer
            </Link>
          </div>
        </nav>
        <Outlet />
        <Scripts />
      </body>
    </html>
  );
}
