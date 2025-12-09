/**
 * Zustand store for visualization state
 */

import { create } from 'zustand';
import type { GenomeResult, RunInfo } from '../types';
import type { Rotation4DAngles } from '../three/rotation4d';
import { DEFAULT_ROTATION_SPEEDS } from '../three/rotation4d';

interface VisualizationState {
  // Current selection
  selectedRunId: string | null;
  selectedGenome: GenomeResult | null;
  allGenomes: GenomeResult[];

  // Run info
  runs: RunInfo[];

  // 4D rotation state
  rotation: Rotation4DAngles;
  rotationSpeeds: Rotation4DAngles;
  isAutoRotating: boolean;

  // Projection settings
  projectionDistance: number;

  // View settings
  showVertices: boolean;
  showEdges: boolean;
  vertexSize: number;
  zoom: number;

  // Actions
  setSelectedRunId: (runId: string | null) => void;
  setSelectedGenome: (genome: GenomeResult | null) => void;
  setAllGenomes: (genomes: GenomeResult[]) => void;
  setRuns: (runs: RunInfo[]) => void;

  // Rotation actions
  setRotation: (rotation: Partial<Rotation4DAngles>) => void;
  setRotationSpeeds: (speeds: Partial<Rotation4DAngles>) => void;
  toggleAutoRotation: () => void;
  resetRotation: () => void;

  // Projection actions
  setProjectionDistance: (distance: number) => void;

  // View actions
  toggleVertices: () => void;
  toggleEdges: () => void;
  setVertexSize: (size: number) => void;
  setZoom: (zoom: number) => void;
}

const initialRotation: Rotation4DAngles = {
  xy: 0,
  xz: 0,
  xw: 0,
  yz: 0,
  yw: 0,
  zw: 0,
};

export const useVisualizationStore = create<VisualizationState>((set) => ({
  // Initial state
  selectedRunId: null,
  selectedGenome: null,
  allGenomes: [],
  runs: [],

  rotation: { ...initialRotation },
  rotationSpeeds: { ...DEFAULT_ROTATION_SPEEDS },
  isAutoRotating: true,

  projectionDistance: 4,

  showVertices: true,
  showEdges: true,
  vertexSize: 0.05,
  zoom: 2.0, // 2x default zoom

  // Actions
  setSelectedRunId: (runId) => set({ selectedRunId: runId }),
  setSelectedGenome: (genome) => set({ selectedGenome: genome }),
  setAllGenomes: (genomes) => set({ allGenomes: genomes }),
  setRuns: (runs) => set({ runs }),

  setRotation: (rotation) =>
    set((state) => ({
      rotation: { ...state.rotation, ...rotation },
    })),

  setRotationSpeeds: (speeds) =>
    set((state) => ({
      rotationSpeeds: { ...state.rotationSpeeds, ...speeds },
    })),

  toggleAutoRotation: () =>
    set((state) => ({ isAutoRotating: !state.isAutoRotating })),

  resetRotation: () => set({ rotation: { ...initialRotation } }),

  setProjectionDistance: (distance) => set({ projectionDistance: distance }),

  toggleVertices: () => set((state) => ({ showVertices: !state.showVertices })),
  toggleEdges: () => set((state) => ({ showEdges: !state.showEdges })),
  setVertexSize: (size) => set({ vertexSize: size }),
  setZoom: (zoom) => set({ zoom }),
}));
