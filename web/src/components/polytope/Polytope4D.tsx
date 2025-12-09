/**
 * 4D Polytope visualization using Three.js
 */

import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

import { useVisualizationStore } from '../../stores/visualization';
import type { Vec4, Vec3, Rotation4DAngles } from '../../three/rotation4d';
import { composeRotation4D, transform4 } from '../../three/rotation4d';
import {
  stereographicProject,
  normalizeWDepth,
  wDepthToColor,
  centerVertices,
  normalizeVertices,
  findEdges,
} from '../../three/projection';

interface Polytope4DProps {
  vertices: number[][];
}

/** Inner component that renders the polytope */
function PolytopeGeometry({ vertices4d }: { vertices4d: Vec4[] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);

  const {
    rotation,
    rotationSpeeds,
    isAutoRotating,
    projectionDistance,
    showVertices,
    showEdges,
    vertexSize,
    zoom,
    setRotation,
  } = useVisualizationStore();

  // Center and normalize vertices
  const normalizedVertices = useMemo(() => {
    const centered = centerVertices(vertices4d);
    return normalizeVertices(centered);
  }, [vertices4d]);

  // Find edges once
  const edges = useMemo(
    () => findEdges(normalizedVertices),
    [normalizedVertices]
  );

  // Update rotation in animation loop
  useFrame((_, delta) => {
    if (!isAutoRotating) return;

    setRotation({
      xy: rotation.xy + rotationSpeeds.xy * delta,
      xz: rotation.xz + rotationSpeeds.xz * delta,
      xw: rotation.xw + rotationSpeeds.xw * delta,
      yz: rotation.yz + rotationSpeeds.yz * delta,
      yw: rotation.yw + rotationSpeeds.yw * delta,
      zw: rotation.zw + rotationSpeeds.zw * delta,
    });
  });

  // Compute projected positions and colors
  const { positions3d, colors, wRange } = useMemo(() => {
    const rotationMatrix = composeRotation4D(rotation);
    const rotated = normalizedVertices.map((v) => transform4(rotationMatrix, v));

    // Find W range for coloring
    let minW = Infinity;
    let maxW = -Infinity;
    for (const v of rotated) {
      if (v[3] < minW) minW = v[3];
      if (v[3] > maxW) maxW = v[3];
    }

    const positions3d: Vec3[] = [];
    const colors: [number, number, number][] = [];

    for (const v of rotated) {
      positions3d.push(stereographicProject(v, projectionDistance));
      const depth = normalizeWDepth(v[3], minW, maxW);
      colors.push(wDepthToColor(depth));
    }

    return { positions3d, colors, wRange: { minW, maxW } };
  }, [normalizedVertices, rotation, projectionDistance]);

  // Update instanced mesh positions and colors
  useEffect(() => {
    if (!meshRef.current) return;

    const mesh = meshRef.current;
    const tempObject = new THREE.Object3D();
    const tempColor = new THREE.Color();

    for (let i = 0; i < positions3d.length; i++) {
      const [x, y, z] = positions3d[i];
      tempObject.position.set(x * zoom, y * zoom, z * zoom);
      tempObject.scale.setScalar(zoom);
      tempObject.updateMatrix();
      mesh.setMatrixAt(i, tempObject.matrix);

      const [r, g, b] = colors[i];
      tempColor.setRGB(r, g, b);
      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [positions3d, colors, zoom]);

  // Update line positions
  useEffect(() => {
    if (!linesRef.current) return;

    const linePositions: number[] = [];
    const lineColors: number[] = [];

    for (const [i, j] of edges) {
      const [x1, y1, z1] = positions3d[i];
      const [x2, y2, z2] = positions3d[j];
      linePositions.push(x1 * zoom, y1 * zoom, z1 * zoom, x2 * zoom, y2 * zoom, z2 * zoom);

      // Average colors of endpoints
      const [r1, g1, b1] = colors[i];
      const [r2, g2, b2] = colors[j];
      lineColors.push(r1, g1, b1, r2, g2, b2);
    }

    const geometry = linesRef.current.geometry as THREE.BufferGeometry;
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(linePositions, 3)
    );
    geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(lineColors, 3)
    );
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }, [positions3d, colors, edges, zoom]);

  return (
    <>
      {/* Vertices as instanced spheres */}
      {showVertices && (
        <instancedMesh
          ref={meshRef}
          args={[undefined, undefined, positions3d.length]}
        >
          <sphereGeometry args={[vertexSize, 16, 16]} />
          <meshStandardMaterial vertexColors />
        </instancedMesh>
      )}

      {/* Edges as lines */}
      {showEdges && (
        <lineSegments ref={linesRef}>
          <bufferGeometry />
          <lineBasicMaterial vertexColors transparent opacity={0.6} />
        </lineSegments>
      )}
    </>
  );
}

/** Main polytope canvas component */
export function Polytope4D({ vertices }: Polytope4DProps) {
  // Ensure vertices are 4D (pad with 0 if needed)
  const vertices4d: Vec4[] = useMemo(() => {
    return vertices.map((v) => {
      if (v.length >= 4) return [v[0], v[1], v[2], v[3]] as Vec4;
      if (v.length === 3) return [v[0], v[1], v[2], 0] as Vec4;
      if (v.length === 2) return [v[0], v[1], 0, 0] as Vec4;
      if (v.length === 1) return [v[0], 0, 0, 0] as Vec4;
      return [0, 0, 0, 0] as Vec4;
    });
  }, [vertices]);

  if (vertices4d.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        No vertices to display
      </div>
    );
  }

  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      <PolytopeGeometry vertices4d={vertices4d} />

      <OrbitControls enableDamping dampingFactor={0.05} />

      {/* Grid helper for orientation */}
      <gridHelper args={[4, 20, '#333', '#222']} rotation={[Math.PI / 2, 0, 0]} />
    </Canvas>
  );
}
