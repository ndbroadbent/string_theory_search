/**
 * Shared number formatting for physics quantities
 */

/** Scientific notation for tiny magnitudes like |W0| ~ 1e-51 */
export function fmtSci(value: number | null, digits = 2): string {
  if (value == null) return '–';
  if (value === 0) return '0';
  return value.toExponential(digits);
}

/** Fixed-point with dash fallback */
export function fmtNum(value: number | null, digits = 2): string {
  if (value == null) return '–';
  return value.toFixed(digits);
}

/** Signed fixed-point (e.g. +1.3 / -0.4) */
export function fmtSigned(value: number | null, digits = 1): string {
  if (value == null) return '–';
  const s = value.toFixed(digits);
  return value >= 0 ? `+${s}` : s;
}

/** Compact integer with thousands separators */
export function fmtInt(value: number | null): string {
  if (value == null) return '–';
  return value.toLocaleString();
}

/** Relative time like "12s ago" from a unix ts */
export function fmtAgo(ts: number | null): string {
  if (ts == null) return 'never';
  const secs = Math.max(0, Math.floor(Date.now() / 1000) - ts);
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

/** Tailwind classes for a |sigma| distance badge */
export function sigmaColor(sigma: number | null): string {
  if (sigma == null) return 'text-gray-500';
  const abs = Math.abs(sigma);
  if (abs <= 1) return 'text-green-400';
  if (abs <= 2) return 'text-yellow-400';
  return 'text-red-400';
}

/** Tailwind classes for the delta-to-target log10|V0| badge */
export function v0DeltaColor(delta: number | null): string {
  if (delta == null) return 'bg-slate-700 text-gray-400';
  const abs = Math.abs(delta);
  if (abs <= 2) return 'bg-green-900/60 text-green-300 border border-green-700/50';
  if (abs <= 10) return 'bg-yellow-900/60 text-yellow-300 border border-yellow-700/50';
  return 'bg-red-900/60 text-red-300 border border-red-700/50';
}
