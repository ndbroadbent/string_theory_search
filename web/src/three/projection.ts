/**
 * 4D to 3D projection utilities
 *
 * Uses stereographic projection from 4D hypersphere to 3D space
 */

import type { Vec3, Vec4 } from './rotation4d';

/**
 * Stereographic projection from 4D to 3D
 * Projects from a point at distance `d` along the W axis
 *
 * @param point 4D point [x, y, z, w]
 * @param distance Distance of projection point from origin (default 4)
 * @returns 3D projected point [x', y', z']
 */
export function stereographicProject(
  [x, y, z, w]: Vec4,
  distance: number = 4
): Vec3 {
  const scale = distance / (distance - w);
  return [x * scale, y * scale, z * scale];
}

/**
 * Perspective projection from 4D to 3D
 * Simpler projection that just scales by W
 *
 * @param point 4D point
 * @param focalLength Focal length for perspective (default 2)
 * @returns 3D projected point
 */
export function perspectiveProject(
  [x, y, z, w]: Vec4,
  focalLength: number = 2
): Vec3 {
  const scale = focalLength / (focalLength + w);
  return [x * scale, y * scale, z * scale];
}

/**
 * Get W-depth as a value from 0-1 for coloring
 *
 * @param w W coordinate
 * @param minW Minimum expected W value
 * @param maxW Maximum expected W value
 * @returns Normalized depth 0-1
 */
export function normalizeWDepth(w: number, minW: number, maxW: number): number {
  if (maxW === minW) return 0.5;
  return (w - minW) / (maxW - minW);
}

/**
 * Color interpolation based on W depth
 * Blue (near) -> White (middle) -> Red (far)
 */
export function wDepthToColor(depth: number): [number, number, number] {
  if (depth < 0.5) {
    // Blue to White
    const t = depth * 2;
    return [t, t, 1];
  } else {
    // White to Red
    const t = (depth - 0.5) * 2;
    return [1, 1 - t, 1 - t];
  }
}

/**
 * Project multiple 4D vertices to 3D
 */
export function projectVertices(
  vertices4d: Vec4[],
  projectionDistance: number = 4
): { vertices3d: Vec3[]; wValues: number[] } {
  const vertices3d: Vec3[] = [];
  const wValues: number[] = [];

  for (const v of vertices4d) {
    vertices3d.push(stereographicProject(v, projectionDistance));
    wValues.push(v[3]);
  }

  return { vertices3d, wValues };
}

/**
 * Find edges of a convex polytope from vertices
 * Uses convex hull edge detection heuristic
 */
export function findEdges(vertices: Vec4[], threshold: number = 1.5): [number, number][] {
  const edges: [number, number][] = [];
  const n = vertices.length;

  // Compute all pairwise distances
  const distances: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = distance4d(vertices[i], vertices[j]);
      distances.push(d);
    }
  }

  // Find minimum non-zero distance
  const minDist = Math.min(...distances.filter((d) => d > 1e-10));

  // Edges are pairs with distance close to minimum
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = distance4d(vertices[i], vertices[j]);
      if (d <= minDist * threshold) {
        edges.push([i, j]);
      }
    }
  }

  return edges;
}

/** Euclidean distance in 4D */
function distance4d(a: Vec4, b: Vec4): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  const dw = a[3] - b[3];
  return Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
}

/**
 * Center vertices around origin
 */
export function centerVertices(vertices: Vec4[]): Vec4[] {
  if (vertices.length === 0) return [];

  // Compute centroid
  const centroid: Vec4 = [0, 0, 0, 0];
  for (const v of vertices) {
    centroid[0] += v[0];
    centroid[1] += v[1];
    centroid[2] += v[2];
    centroid[3] += v[3];
  }
  centroid[0] /= vertices.length;
  centroid[1] /= vertices.length;
  centroid[2] /= vertices.length;
  centroid[3] /= vertices.length;

  // Subtract centroid
  return vertices.map((v) => [
    v[0] - centroid[0],
    v[1] - centroid[1],
    v[2] - centroid[2],
    v[3] - centroid[3],
  ]);
}

/**
 * Normalize vertices to fit within unit hypersphere
 */
export function normalizeVertices(vertices: Vec4[]): Vec4[] {
  if (vertices.length === 0) return [];

  // Find max distance from origin
  let maxDist = 0;
  for (const v of vertices) {
    const d = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2);
    if (d > maxDist) maxDist = d;
  }

  if (maxDist === 0) return vertices;

  // Scale to unit hypersphere
  return vertices.map((v) => [
    v[0] / maxDist,
    v[1] / maxDist,
    v[2] / maxDist,
    v[3] / maxDist,
  ]);
}
