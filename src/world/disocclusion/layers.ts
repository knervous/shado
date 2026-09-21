/**
 * Capture geometry shared by the raster, the propagation and the target
 * classifier. Every stage calls these functions; none re-derives them.
 *
 * Screen convention: column increases with the capture's `right` axis, row
 * increases DOWNWARD (against `up`), matching WebGPU framebuffer coordinates
 * so the fragment shader can use @builtin(position) directly.
 */
import type { DisocclusionAxis, DisocclusionCapture, DisocclusionFrame, DisocclusionSettings, Vec3 } from './types';

const AXES: Record<DisocclusionAxis, { forward: Vec3; up: Vec3 }> = {
  '+x': { forward: [1, 0, 0], up: [0, 1, 0] },
  '-x': { forward: [-1, 0, 0], up: [0, 1, 0] },
  '+z': { forward: [0, 0, 1], up: [0, 1, 0] },
  '-z': { forward: [0, 0, -1], up: [0, 1, 0] },
  '+y': { forward: [0, 1, 0], up: [0, 0, 1] },
  '-y': { forward: [0, -1, 0], up: [0, 0, 1] },
};

export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Lowest `n` bits set, n in [0, 32]. Never shifts a u32 by 32. */
export function lowBits(n: number): number {
  if (n <= 0) return 0;
  if (n >= 32) return 0xffffffff;
  return ((1 << n) - 1) >>> 0;
}

export function validateSettings(settings: DisocclusionSettings): void {
  const { resolution, tileSize, layers } = settings;
  if (!Number.isInteger(resolution) || resolution <= 0) throw new Error('resolution must be a positive integer');
  if (!Number.isInteger(tileSize) || tileSize <= 0 || resolution % tileSize !== 0) {
    throw new Error('tileSize must divide resolution');
  }
  if (!Number.isInteger(layers) || layers < 1 || layers > 32) throw new Error('layers must be 1..32');
  if (!(settings.filterCell >= 0) || !Number.isFinite(settings.filterCell)) throw new Error('filterCell must be finite and >= 0');
}

/**
 * The paper's §3.4 support construction, adapted to a translating box.
 *
 * The camera plane passes through the box centre, perpendicular to the axis.
 * A supported ray starts anywhere in the box -- at most `halfDepth` before or
 * after that plane and within the box's lateral half extent -- and leans at
 * most `directionTan` per lateral axis. Extending or retracting such a ray to
 * the camera plane lands it inside a rectangle of half size
 * `lateralHalf + halfDepth * directionTan` per axis. Every supported ray is
 * therefore a ray from that rectangle, and the paper's lateral viewcell
 * applies unchanged. Beyond `near`, such a ray has |tan| at most
 * `directionTan + viewcellHalf / near` per axis: the raster's extended FOV.
 *
 * A ray origin can sit up to `halfDepth` BEHIND the camera plane, so geometry
 * between there and `near` is never rasterized; the classifier admits it as
 * near field. Nothing behind `-halfDepth` is reachable by a forward ray.
 */
export function captureFrame(capture: DisocclusionCapture): DisocclusionFrame {
  const { forward, up } = AXES[capture.axis];
  const right = cross(forward, up);
  const { sourceMin: min, sourceMax: max } = capture;
  for (let i = 0; i < 3; i++) {
    if (!(max[i]! > min[i]!) || !Number.isFinite(min[i]!) || !Number.isFinite(max[i]!)) {
      throw new Error('source box must be finite with max > min');
    }
  }
  if (!(capture.near > 0) || !(capture.far > capture.near) || !Number.isFinite(capture.far)) {
    throw new Error('capture needs finite 0 < near < far');
  }
  const directionTanX = capture.directionTan;
  const directionTanY = capture.directionTanUp ?? capture.directionTan;
  for (const tan of [directionTanX, directionTanY]) {
    if (!(tan > 0) || !Number.isFinite(tan)) throw new Error('direction tans must be finite and positive');
  }
  const origin: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const half: Vec3 = [(max[0] - min[0]) / 2, (max[1] - min[1]) / 2, (max[2] - min[2]) / 2];
  const extentAlong = (axis: Vec3) => Math.abs(axis[0]) * half[0] + Math.abs(axis[1]) * half[1] + Math.abs(axis[2]) * half[2];
  const sourceHalfDepth = extentAlong(forward);
  const sourceHalfRight = extentAlong(right);
  const sourceHalfUp = extentAlong(up);
  if (!(capture.near > sourceHalfDepth)) throw new Error('near must lie beyond the source box');
  const viewcellHalfX = sourceHalfRight + sourceHalfDepth * directionTanX;
  const viewcellHalfY = sourceHalfUp + sourceHalfDepth * directionTanY;
  return {
    origin,
    forward,
    right,
    up,
    viewcellHalfX,
    viewcellHalfY,
    extTanX: directionTanX + viewcellHalfX / capture.near,
    extTanY: directionTanY + viewcellHalfY / capture.near,
    near: capture.near,
    far: capture.far,
    sourceHalfDepth,
    sourceHalfRight,
    sourceHalfUp,
    directionTanX,
    directionTanY,
  };
}

/** View-space coordinates (right, up, depth) of a world point. */
export function toView(frame: DisocclusionFrame, p: Vec3): Vec3 {
  const d: Vec3 = [p[0] - frame.origin[0], p[1] - frame.origin[1], p[2] - frame.origin[2]];
  return [dot(d, frame.right), dot(d, frame.up), dot(d, frame.forward)];
}

/**
 * Paper eq. (1): layer = floor(N * log(z - zn + 1) / log(zf - zn + 1)).
 * Depth outside (near, far) has no layer; callers treat it as unknown.
 */
export function layerOfDepth(frame: DisocclusionFrame, layers: number, z: number): number {
  if (!(z >= frame.near) || !(z < frame.far)) return -1;
  const value = Math.floor((layers * Math.log(z - frame.near + 1)) / Math.log(frame.far - frame.near + 1));
  return Math.min(layers - 1, Math.max(0, value));
}

/** Front (nearest) depth of a layer; layerFront(N) is `far`. */
export function layerFront(frame: DisocclusionFrame, layers: number, layer: number): number {
  if (layer >= layers) return frame.far;
  return frame.near - 1 + Math.pow(frame.far - frame.near + 1, layer / layers);
}

/** Width of one tile in tan units along right (x) and up (y). */
export function tileTanSpan(frame: DisocclusionFrame, settings: DisocclusionSettings): [number, number] {
  const k = (2 * settings.tileSize) / settings.resolution;
  return [frame.extTanX * k, frame.extTanY * k];
}

/** Tan interval covered by tile column `tx` (along right). */
export function tileTanX(frame: DisocclusionFrame, settings: DisocclusionSettings, tx: number): [number, number] {
  const span = tileTanSpan(frame, settings)[0];
  return [-frame.extTanX + tx * span, -frame.extTanX + (tx + 1) * span];
}

/** Tan interval covered by tile row `ty` (rows grow downward). */
export function tileTanY(frame: DisocclusionFrame, settings: DisocclusionSettings, ty: number): [number, number] {
  const span = tileTanSpan(frame, settings)[1];
  return [frame.extTanY - (ty + 1) * span, frame.extTanY - ty * span];
}

/**
 * Inclusive tile column range touched by a tan-x interval, unclamped. Plain
 * floor/ceil: rounding can only widen the range by a tile, never shrink it.
 */
export function tileRangeX(frame: DisocclusionFrame, settings: DisocclusionSettings, lo: number, hi: number): [number, number] {
  const span = tileTanSpan(frame, settings)[0];
  return [Math.floor((lo + frame.extTanX) / span), Math.ceil((hi + frame.extTanX) / span) - 1];
}

/** Inclusive tile row range touched by a tan-y interval, unclamped. */
export function tileRangeY(frame: DisocclusionFrame, settings: DisocclusionSettings, lo: number, hi: number): [number, number] {
  const span = tileTanSpan(frame, settings)[1];
  return [Math.floor((frame.extTanY - hi) / span), Math.ceil((frame.extTanY - lo) / span) - 1];
}

/**
 * Tan-space growth of a disocclusion frustum along each axis: a ray from
 * anywhere on the viewcell through an opening at depth `zOpen` drifts at most
 * viewcellHalf * (1/zOpen - 1/z) in tan by depth z. Callers pass the opening
 * layer's FRONT and the target layer's BACK, the widest case. (The reference
 * source intersects at the target layer's front plane; see deviations.)
 */
export function frustumGrowth(frame: DisocclusionFrame, zOpen: number, zTarget: number): [number, number] {
  const k = Math.max(0, 1 / zOpen - 1 / zTarget);
  return [frame.viewcellHalfX * k, frame.viewcellHalfY * k];
}

/**
 * Column-major 4x4 matrix mapping world to clip for the raster pass. Clip
 * x/y are tan / extTan (WebGPU's y-up NDC puts +up at row 0); z maps
 * [near, far] to [0, 1]; w is view depth.
 */
export function captureClipMatrix(frame: DisocclusionFrame): Float32Array {
  const { right, up, forward, origin, near, far } = frame;
  const sx = 1 / frame.extTanX;
  const sy = 1 / frame.extTanY;
  const zScale = far / (far - near);
  const rows = [
    [right[0] * sx, right[1] * sx, right[2] * sx, -dot(right, origin) * sx],
    [up[0] * sy, up[1] * sy, up[2] * sy, -dot(up, origin) * sy],
    [forward[0] * zScale, forward[1] * zScale, forward[2] * zScale, (-dot(forward, origin) - near) * zScale],
    [forward[0], forward[1], forward[2], -dot(forward, origin)],
  ];
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[c * 4 + r] = rows[r]![c]!;
  return out;
}
