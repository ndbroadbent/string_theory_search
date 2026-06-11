/**
 * DESI plane scatter: valid candidates in (cpl_w0, cpl_wa) with the DESI
 * 1-sigma box (w0 = -0.45 +/- 0.21, wa = -1.8 +/- 0.6) and the Lambda-CDM
 * point (-1, 0) marked.
 */

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
} from 'recharts';
import type { Candidate } from '../../types';
import { TARGETS, sigmaDistance } from '../../types';
import { fmtNum, fmtSigned } from '../../lib/format';

interface DesiScatterProps {
  candidates: Candidate[];
  height?: number;
}

export function DesiScatter({ candidates, height = 360 }: DesiScatterProps) {
  const points = candidates
    .filter((c) => c.cpl_w0 != null && c.cpl_wa != null)
    .map((c) => ({ ...c, x: c.cpl_w0 as number, y: c.cpl_wa as number }));

  const { desiW0, desiWa, lambda } = TARGETS;
  const boxX: [number, number] = [desiW0.value - desiW0.sigma, desiW0.value + desiW0.sigma];
  const boxY: [number, number] = [desiWa.value - desiWa.sigma, desiWa.value + desiWa.sigma];

  // Domain: include data, the DESI box, and the Lambda point, with padding
  const xs = [...points.map((p) => p.x), boxX[0], boxX[1], lambda.w0];
  const ys = [...points.map((p) => p.y), boxY[0], boxY[1], lambda.wa];
  const pad = 0.25;
  const xDomain: [number, number] = [Math.min(...xs) - pad, Math.max(...xs) + pad];
  const yDomain: [number, number] = [Math.min(...ys) - pad, Math.max(...ys) + pad];

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 35, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            type="number"
            dataKey="x"
            domain={xDomain}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{ value: 'CPL w₀', position: 'bottom', fill: '#9ca3af', fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={yDomain}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{ value: 'CPL wₐ', angle: -90, position: 'left', fill: '#9ca3af', fontSize: 12 }}
          />
          {/* DESI 1-sigma box */}
          <ReferenceArea
            x1={boxX[0]}
            x2={boxX[1]}
            y1={boxY[0]}
            y2={boxY[1]}
            stroke="#22d3ee"
            strokeOpacity={0.7}
            fill="#22d3ee"
            fillOpacity={0.08}
          />
          {/* DESI best fit */}
          <ReferenceDot
            x={desiW0.value}
            y={desiWa.value}
            r={5}
            fill="#22d3ee"
            stroke="none"
          />
          {/* Lambda-CDM point */}
          <ReferenceDot
            x={lambda.w0}
            y={lambda.wa}
            r={5}
            fill="none"
            stroke="#f59e0b"
            strokeWidth={2}
          />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ payload }) => {
              if (!payload?.length) return null;
              const c = payload[0].payload as Candidate & { x: number; y: number };
              return (
                <div className="bg-slate-800 border border-slate-600 rounded p-2 text-xs">
                  <div className="text-cyan-400 font-medium mb-1">{c.polytope_id}</div>
                  <div className="text-gray-300">
                    w₀ = {fmtNum(c.cpl_w0, 3)}{' '}
                    <span className="text-gray-500">
                      ({fmtSigned(sigmaDistance(c.cpl_w0, desiW0))}σ)
                    </span>
                  </div>
                  <div className="text-gray-300">
                    wₐ = {fmtNum(c.cpl_wa, 3)}{' '}
                    <span className="text-gray-500">
                      ({fmtSigned(sigmaDistance(c.cpl_wa, desiWa))}σ)
                    </span>
                  </div>
                  <div className="text-gray-400 mt-1">fitness {fmtNum(c.fitness, 2)}</div>
                </div>
              );
            }}
          />
          <Scatter data={points} fill="#4ade80" />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 px-2 text-xs text-gray-400">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm bg-cyan-400/20 border border-cyan-400" />
          DESI 1σ (w₀ = -0.45 ± 0.21, wₐ = -1.8 ± 0.6)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full border-2 border-amber-500" />
          Λ (-1, 0)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full bg-green-400" />
          valid candidates ({points.length})
        </span>
      </div>
    </div>
  );
}
