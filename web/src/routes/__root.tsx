/**
 * Root layout for the visualization dashboard
 */

import { HeadContent, Scripts, createRootRoute, Outlet } from '@tanstack/react-router';
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
        <Outlet />
        <Scripts />
      </body>
    </html>
  );
}
