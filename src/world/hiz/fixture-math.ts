/**
 * A tiny, dependency-free camera for Hi-Z fixtures: eye at the origin looking
 * down +Z, +Y up, Babylon column-major layout (`x' = x*m[0] + y*m[4] + z*m[8] + m[12]`).
 */
import type { ShadoHiZDepthConvention, ShadoHiZViewInput } from './types';

export function hizPerspective(
  aspect: number,
  fovY: number,
  near: number,
  far: number,
  convention: ShadoHiZDepthConvention
): Float32Array {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  const range = far - near;
  if (convention === 'normal') {
    m[10] = far / range;
    m[14] = (-near * far) / range;
  } else {
    m[10] = -near / range;
    m[14] = (near * far) / range;
  }
  m[11] = 1;
  return m;
}

export function hizFixtureView(
  width: number,
  height: number,
  convention: ShadoHiZDepthConvention
): ShadoHiZViewInput {
  return {
    frameId: 1,
    viewProjection: hizPerspective(width / height, Math.PI / 3, 0.1, 100, convention),
    viewportWidth: width,
    viewportHeight: height,
    convention,
    ndcHalfZRange: true,
    topLeftOrigin: true,
    worldEpoch: 1,
    opaqueEpoch: 1,
  };
}
