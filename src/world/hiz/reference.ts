/**
 * Scalar Hi-Z reference. The WGSL in `./wgsl.ts` mirrors these functions line
 * for line; keep them in step. Small enough to reason about by hand, and the
 * oracle for the conservative-admission rules the GPU path must never relax.
 */
import {
  SHADO_HIZ_DEPTH_BIAS,
  SHADO_HIZ_MAX_LEVELS,
  SHADO_HIZ_MAX_TEXELS_PER_AXIS,
  SHADO_HIZ_PIXEL_EXPANSION,
  ShadoHiZAdmitReason,
  type ShadoHiZDepthConvention,
  type ShadoHiZProjection,
  type ShadoHiZViewInput,
} from './types';

export interface ShadoHiZLevel {
  readonly width: number;
  readonly height: number;
  /** Word offset of this level inside the flat pyramid buffer. */
  readonly offset: number;
}

export interface ShadoHiZLayout {
  readonly levels: readonly ShadoHiZLevel[];
  /** Total float words for all levels. */
  readonly words: number;
}

/**
 * Level L+1 is ceil(L / 2) on each axis, so texel i of level L+1 covers
 * texels 2i and 2i+1 of level L (the second only when it exists). Pixel x
 * therefore lands in texel x >> L at every level and no source pixel is
 * skipped on odd edges.
 */
export function shadoHiZLayout(width: number, height: number): ShadoHiZLayout {
  if (!(width >= 1 && height >= 1) || !Number.isInteger(width) || !Number.isInteger(height)) {
    throw new Error(`Hi-Z layout needs a positive integer viewport, got ${width}x${height}`);
  }
  const levels: ShadoHiZLevel[] = [];
  let w = width;
  let h = height;
  let offset = 0;
  for (;;) {
    levels.push({ width: w, height: h, offset });
    offset += w * h;
    if (w === 1 && h === 1) break;
    if (levels.length === SHADO_HIZ_MAX_LEVELS) {
      throw new Error(`Hi-Z viewport ${width}x${height} needs more than ${SHADO_HIZ_MAX_LEVELS} levels`);
    }
    w = Math.ceil(w / 2);
    h = Math.ceil(h / 2);
  }
  return { levels, words: offset };
}

/** Farthest of two depths in the convention: max for normal, min for reversed. */
function farther(convention: ShadoHiZDepthConvention, a: number, b: number): number {
  return convention === 'normal' ? Math.max(a, b) : Math.min(a, b);
}

/**
 * Builds the full conservative chain. `depth` is level 0, row-major, row 0
 * first; uncovered pixels must already hold the clear value (1 normal, 0
 * reversed) so they keep every parent far.
 */
export function buildShadoHiZPyramid(
  depth: ArrayLike<number>,
  width: number,
  height: number,
  convention: ShadoHiZDepthConvention
): { layout: ShadoHiZLayout; data: Float32Array } {
  const layout = shadoHiZLayout(width, height);
  if (depth.length !== width * height) {
    throw new Error(`Hi-Z depth has ${depth.length} samples for ${width}x${height}`);
  }
  const data = new Float32Array(layout.words);
  for (let i = 0; i < width * height; i++) {
    const value = depth[i];
    // An unknown sample is treated as open sky, never as a wall.
    data[i] = Number.isFinite(value) ? value : convention === 'normal' ? 1 : 0;
  }
  for (let l = 1; l < layout.levels.length; l++) {
    const src = layout.levels[l - 1];
    const dst = layout.levels[l];
    for (let y = 0; y < dst.height; y++) {
      for (let x = 0; x < dst.width; x++) {
        const sx0 = 2 * x;
        const sy0 = 2 * y;
        const sx1 = Math.min(sx0 + 1, src.width - 1);
        const sy1 = Math.min(sy0 + 1, src.height - 1);
        let value = data[src.offset + sy0 * src.width + sx0];
        value = farther(convention, value, data[src.offset + sy0 * src.width + sx1]);
        value = farther(convention, value, data[src.offset + sy1 * src.width + sx0]);
        value = farther(convention, value, data[src.offset + sy1 * src.width + sx1]);
        data[dst.offset + y * dst.width + x] = value;
      }
    }
  }
  return { layout, data };
}

/**
 * Projects a world AABB to a conservative pixel rectangle and its nearest
 * possible depth. Anything uncertain returns an admit reason instead.
 */
export function projectShadoHiZBounds(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
  view: ShadoHiZViewInput
): ShadoHiZProjection {
  const m = view.viewProjection;
  let ndcMinX = Infinity;
  let ndcMinY = Infinity;
  let ndcMaxX = -Infinity;
  let ndcMaxY = -Infinity;
  let nearest = view.convention === 'normal' ? Infinity : -Infinity;
  for (let corner = 0; corner < 8; corner++) {
    const x = corner & 1 ? max[0] : min[0];
    const y = corner & 2 ? max[1] : min[1];
    const z = corner & 4 ? max[2] : min[2];
    const cx = x * m[0] + y * m[4] + z * m[8] + m[12];
    const cy = x * m[1] + y * m[5] + z * m[9] + m[13];
    const cz = x * m[2] + y * m[6] + z * m[10] + m[14];
    const cw = x * m[3] + y * m[7] + z * m[11] + m[15];
    if (!Number.isFinite(cx + cy + cz + cw)) return admit(ShadoHiZAdmitReason.NonFinite);
    // Behind the eye, on the eye plane, or in front of the near plane: the
    // projected rectangle is not the bound's footprint any more.
    if (!(cw > 1e-6)) return admit(ShadoHiZAdmitReason.NearPlane);
    const ndcZ = cz / cw;
    const depth = view.ndcHalfZRange ? ndcZ : ndcZ * 0.5 + 0.5;
    if (view.convention === 'normal' ? depth < 0 : depth > 1) {
      return admit(ShadoHiZAdmitReason.NearPlane);
    }
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    ndcMinX = Math.min(ndcMinX, ndcX);
    ndcMaxX = Math.max(ndcMaxX, ndcX);
    ndcMinY = Math.min(ndcMinY, ndcY);
    ndcMaxY = Math.max(ndcMaxY, ndcY);
    nearest = view.convention === 'normal' ? Math.min(nearest, depth) : Math.max(nearest, depth);
  }
  if (ndcMaxX < -1 || ndcMinX > 1 || ndcMaxY < -1 || ndcMinY > 1) {
    return admit(ShadoHiZAdmitReason.OffScreen);
  }
  const w = view.viewportWidth;
  const h = view.viewportHeight;
  const px0 = (ndcMinX * 0.5 + 0.5) * w;
  const px1 = (ndcMaxX * 0.5 + 0.5) * w;
  // NDC +y is up; flip only when pixel row 0 is the top of the image.
  const rowA = (view.topLeftOrigin ? 0.5 - ndcMaxY * 0.5 : ndcMinY * 0.5 + 0.5) * h;
  const rowB = (view.topLeftOrigin ? 0.5 - ndcMinY * 0.5 : ndcMaxY * 0.5 + 0.5) * h;
  const e = SHADO_HIZ_PIXEL_EXPANSION;
  return {
    admit: ShadoHiZAdmitReason.None,
    x0: clampInt(Math.floor(px0) - e, 0, w - 1),
    x1: clampInt(Math.floor(px1) + e, 0, w - 1),
    y0: clampInt(Math.floor(rowA) - e, 0, h - 1),
    y1: clampInt(Math.floor(rowB) + e, 0, h - 1),
    nearest,
  };
}

export type ShadoHiZVerdict =
  | { readonly visible: true; readonly reason: ShadoHiZAdmitReason | 'depth'; readonly level?: number }
  | { readonly visible: false; readonly level: number; readonly occluderDepth: number };

/**
 * Chooses the finest level where the rectangle spans at most
 * SHADO_HIZ_MAX_TEXELS_PER_AXIS texels per axis and compares against EVERY
 * texel it touches. Equality and anything within the bias stays visible.
 */
export function testShadoHiZ(
  pyramid: { layout: ShadoHiZLayout; data: Float32Array },
  projection: ShadoHiZProjection,
  convention: ShadoHiZDepthConvention,
  bias = SHADO_HIZ_DEPTH_BIAS
): ShadoHiZVerdict {
  if (projection.admit !== ShadoHiZAdmitReason.None) {
    return { visible: true, reason: projection.admit };
  }
  const { levels } = pyramid.layout;
  let level = 0;
  while (
    level < levels.length - 1 &&
    ((projection.x1 >> level) - (projection.x0 >> level) + 1 > SHADO_HIZ_MAX_TEXELS_PER_AXIS ||
      (projection.y1 >> level) - (projection.y0 >> level) + 1 > SHADO_HIZ_MAX_TEXELS_PER_AXIS)
  ) {
    level++;
  }
  const info = levels[level];
  const tx0 = projection.x0 >> level;
  const tx1 = Math.min(projection.x1 >> level, info.width - 1);
  const ty0 = projection.y0 >> level;
  const ty1 = Math.min(projection.y1 >> level, info.height - 1);
  let occluder = convention === 'normal' ? -Infinity : Infinity;
  for (let y = ty0; y <= ty1; y++) {
    for (let x = tx0; x <= tx1; x++) {
      occluder = farther(convention, occluder, pyramid.data[info.offset + y * info.width + x]);
    }
  }
  const hidden =
    convention === 'normal'
      ? projection.nearest > occluder + bias
      : projection.nearest < occluder - bias;
  return hidden
    ? { visible: false, level, occluderDepth: occluder }
    : { visible: true, reason: 'depth', level };
}

function admit(reason: ShadoHiZAdmitReason): ShadoHiZProjection {
  return { admit: reason, x0: 0, y0: 0, x1: 0, y1: 0, nearest: 0 };
}

function clampInt(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}
