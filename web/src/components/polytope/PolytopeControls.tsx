/**
 * Controls for 4D polytope visualization
 */

import { useVisualizationStore } from '../../stores/visualization';

export function PolytopeControls() {
  const {
    isAutoRotating,
    toggleAutoRotation,
    resetRotation,
    showVertices,
    showEdges,
    toggleVertices,
    toggleEdges,
    vertexSize,
    setVertexSize,
    projectionDistance,
    setProjectionDistance,
    rotationSpeeds,
    setRotationSpeeds,
    zoom,
    setZoom,
  } = useVisualizationStore();

  return (
    <div className="bg-slate-800/90 backdrop-blur rounded-lg p-4 space-y-4">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
        Polytope Controls
      </h3>

      {/* Toggle buttons */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={toggleAutoRotation}
          className={`px-3 py-1.5 text-sm rounded transition-colors ${
            isAutoRotating
              ? 'bg-cyan-600 text-white'
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          {isAutoRotating ? 'Stop Rotation' : 'Auto Rotate'}
        </button>

        <button
          onClick={resetRotation}
          className="px-3 py-1.5 text-sm rounded bg-slate-700 text-gray-300 hover:bg-slate-600 transition-colors"
        >
          Reset View
        </button>

        <button
          onClick={toggleVertices}
          className={`px-3 py-1.5 text-sm rounded transition-colors ${
            showVertices
              ? 'bg-cyan-600 text-white'
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          Vertices
        </button>

        <button
          onClick={toggleEdges}
          className={`px-3 py-1.5 text-sm rounded transition-colors ${
            showEdges
              ? 'bg-cyan-600 text-white'
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          Edges
        </button>
      </div>

      {/* Sliders */}
      <div className="space-y-3">
        <div>
          <label className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Zoom</span>
            <span>{zoom.toFixed(1)}x</span>
          </label>
          <input
            type="range"
            min="0.5"
            max="5"
            step="0.1"
            value={zoom}
            onChange={(e) => setZoom(parseFloat(e.target.value))}
            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
        </div>

        <div>
          <label className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Vertex Size</span>
            <span>{vertexSize.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.01"
            max="0.2"
            step="0.01"
            value={vertexSize}
            onChange={(e) => setVertexSize(parseFloat(e.target.value))}
            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
        </div>

        <div>
          <label className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Projection Distance</span>
            <span>{projectionDistance.toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="2"
            max="10"
            step="0.1"
            value={projectionDistance}
            onChange={(e) => setProjectionDistance(parseFloat(e.target.value))}
            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
        </div>

        {/* 4D Rotation speeds - all 6 planes */}
        <div className="pt-2 border-t border-slate-700">
          <p className="text-xs text-gray-400 mb-2">4D Rotation Speeds:</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2">
            {(['xy', 'xz', 'xw', 'yz', 'yw', 'zw'] as const).map((plane) => (
              <div key={plane}>
                <label className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>{plane.toUpperCase()}</span>
                  <span>{rotationSpeeds[plane].toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.02"
                  value={rotationSpeeds[plane]}
                  onChange={(e) => setRotationSpeeds({ [plane]: parseFloat(e.target.value) })}
                  className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Color legend */}
      <div className="pt-2 border-t border-slate-700">
        <p className="text-xs text-gray-400 mb-2">W-depth coloring:</p>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-blue-500" />
          <span className="text-xs text-gray-400">Near (W-)</span>
          <div className="flex-1 h-2 rounded bg-gradient-to-r from-blue-500 via-white to-red-500" />
          <span className="text-xs text-gray-400">Far (W+)</span>
          <div className="w-4 h-4 rounded bg-red-500" />
        </div>
      </div>
    </div>
  );
}
