/**
 * SO(4) rotation utilities for 4D polytope visualization
 *
 * 4D rotations occur in 6 planes (not 3 axes like 3D):
 * XY, XZ, XW, YZ, YW, ZW
 */

/** 4D point */
export type Vec4 = [number, number, number, number];

/** 3D point */
export type Vec3 = [number, number, number];

/** 4x4 rotation matrix */
export type Mat4x4 = number[][];

/** Identity 4x4 matrix */
export function identity4(): Mat4x4 {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

/** Rotation in the XY plane */
export function rotationXY(theta: number): Mat4x4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, -s, 0, 0],
    [s, c, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

/** Rotation in the XZ plane */
export function rotationXZ(theta: number): Mat4x4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, 0, -s, 0],
    [0, 1, 0, 0],
    [s, 0, c, 0],
    [0, 0, 0, 1],
  ];
}

/** Rotation in the XW plane */
export function rotationXW(theta: number): Mat4x4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, 0, 0, -s],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [s, 0, 0, c],
  ];
}

/** Rotation in the YZ plane */
export function rotationYZ(theta: number): Mat4x4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [1, 0, 0, 0],
    [0, c, -s, 0],
    [0, s, c, 0],
    [0, 0, 0, 1],
  ];
}

/** Rotation in the YW plane */
export function rotationYW(theta: number): Mat4x4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [1, 0, 0, 0],
    [0, c, 0, -s],
    [0, 0, 1, 0],
    [0, s, 0, c],
  ];
}

/** Rotation in the ZW plane */
export function rotationZW(theta: number): Mat4x4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, c, -s],
    [0, 0, s, c],
  ];
}

/** Multiply two 4x4 matrices */
export function multiply4x4(a: Mat4x4, b: Mat4x4): Mat4x4 {
  const result: Mat4x4 = [];
  for (let i = 0; i < 4; i++) {
    result[i] = [];
    for (let j = 0; j < 4; j++) {
      result[i][j] =
        a[i][0] * b[0][j] +
        a[i][1] * b[1][j] +
        a[i][2] * b[2][j] +
        a[i][3] * b[3][j];
    }
  }
  return result;
}

/** Apply 4x4 matrix to a 4D vector */
export function transform4(m: Mat4x4, v: Vec4): Vec4 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
    m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
  ];
}

/** Rotation angles for all 6 planes */
export interface Rotation4DAngles {
  xy: number;
  xz: number;
  xw: number;
  yz: number;
  yw: number;
  zw: number;
}

/** Compose all 6 rotation planes into a single 4x4 matrix */
export function composeRotation4D(angles: Rotation4DAngles): Mat4x4 {
  let result = identity4();
  result = multiply4x4(result, rotationXY(angles.xy));
  result = multiply4x4(result, rotationXZ(angles.xz));
  result = multiply4x4(result, rotationXW(angles.xw));
  result = multiply4x4(result, rotationYZ(angles.yz));
  result = multiply4x4(result, rotationYW(angles.yw));
  result = multiply4x4(result, rotationZW(angles.zw));
  return result;
}

/** Default rotation speeds for animation (radians per second) */
export const DEFAULT_ROTATION_SPEEDS: Rotation4DAngles = {
  xy: 0.1,
  xz: 0.15,
  xw: 0.2, // W-axis rotations are most interesting
  yz: 0.12,
  yw: 0.18,
  zw: 0.22,
};
