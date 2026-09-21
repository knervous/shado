/**
 * From tile visibility to target admission (paper §3.2 "final PVS", §3.3
 * volumetric filter), plus the application's conservative adapter.
 *
 * Raw:      a target is admitted when one of its triangle IDs is a sample in a
 *           non-empty, potentially visible tile (the paper's PVS).
 * Filtered: raw plus the paper's volumetric filter (§3.3), reported only.
 * Expanded: raw, OR any target whose full bounds touch a potentially visible
 *           cell, OR reach nearer than `near` (near field) or beyond `far`
 *           (unknown). This catches triangles that lost the per-layer depth
 *           competition, fell between samples, or were clipped.
 *
 * The filter is NOT part of `expanded`: the bounds test already admits any
 * target with a triangle inside a visible cell, which is what the filter
 * guards against, and the filter's world grid (the paper's 50 cm minimum is
 * 1.5 zone units) is thicker than a typical wall, so it admits the far face
 * of any wall whose near face is visible. `filterOnly` counts those.
 *
 * Only `expanded` may reach the runtime. `raw`/`filtered` are reported so the
 * adapter's effect is measurable.
 */
import { layerFront, tileRangeX, tileRangeY, toView } from './layers';
import { EMPTY_SAMPLE } from './reference';
import type { DisocclusionFrame, DisocclusionGeometry, DisocclusionLayers, DisocclusionMasks, DisocclusionSettings, Vec3 } from './types';

export type DisocclusionTargetReason = 'raw' | 'cell' | 'near' | 'far' | 'hidden' | 'outside';

export type DisocclusionTargetClassification = {
  /** Per target: why it was admitted, or 'hidden'/'outside' when rejected. */
  reason: DisocclusionTargetReason[];
  raw: Uint8Array;
  filtered: Uint8Array;
  expanded: Uint8Array;
  counts: { targets: number; raw: number; filtered: number; expanded: number; filterOnly: number };
};

export type TargetBounds = { min: Float64Array; max: Float64Array };

/** World AABB per target from its triangles. Targets with no triangles get an inverted box. */
export function targetBounds(geometry: DisocclusionGeometry, targetCount: number): TargetBounds {
  const min = new Float64Array(targetCount * 3).fill(Infinity);
  const max = new Float64Array(targetCount * 3).fill(-Infinity);
  const { positions, indices, triangleTarget } = geometry;
  for (let tri = 0; tri < triangleTarget.length; tri++) {
    const target = triangleTarget[tri]!;
    if (target < 0) continue;
    for (let k = 0; k < 3; k++) {
      const v = indices[tri * 3 + k]! * 3;
      for (let a = 0; a < 3; a++) {
        const value = positions[v + a]!;
        if (value < min[target * 3 + a]!) min[target * 3 + a] = value;
        if (value > max[target * 3 + a]!) max[target * 3 + a] = value;
      }
    }
  }
  return { min, max };
}

/**
 * Does an axis-aligned box touch a potentially visible cell?
 *
 * The capture axes are world axes, so the box stays axis-aligned in view
 * space. For each layer slab it overlaps, the box's tan footprint is bounded
 * exactly by its corner ratios. Returns 'near'/'far' when the box leaves the
 * captured depth range, which callers must treat as admitted.
 */
export function classifyBox(
  frame: DisocclusionFrame,
  settings: DisocclusionSettings,
  masks: DisocclusionMasks,
  min: Vec3,
  max: Vec3
): 'cell' | 'near' | 'far' | 'hidden' | 'outside' {
  const a = toView(frame, min);
  const b = toView(frame, max);
  const x0 = Math.min(a[0], b[0]);
  const x1 = Math.max(a[0], b[0]);
  const y0 = Math.min(a[1], b[1]);
  const y1 = Math.max(a[1], b[1]);
  const z0 = Math.min(a[2], b[2]);
  const z1 = Math.max(a[2], b[2]);
  // Behind every supported ray origin: no forward ray gets there.
  if (z1 <= -frame.sourceHalfDepth) return 'outside';
  // Nearer than the raster: admit only what a supported ray can reach there.
  // At depth z a ray from the box is at most viewcellHalf + tan * z sideways
  // (see captureFrame), so a box wholly outside that wedge is not near field.
  if (z0 < frame.near) {
    const zNear = Math.min(z1, frame.near);
    const reachX = frame.viewcellHalfX + frame.directionTanX * zNear;
    const reachY = frame.viewcellHalfY + frame.directionTanY * zNear;
    if (x1 >= -reachX && x0 <= reachX && y1 >= -reachY && y0 <= reachY) return 'near';
    if (z1 < frame.near) return 'outside';
  }
  if (z1 >= frame.far) return 'far';
  const tiles = masks.tilesX;
  const n = settings.layers;
  let inside = false;
  for (let layer = 0; layer < n; layer++) {
    const front = layerFront(frame, n, layer);
    const back = layerFront(frame, n, layer + 1);
    if (z1 < front || z0 >= back) continue;
    const za = Math.max(z0, front, frame.near);
    const zb = Math.min(z1, back);
    const tanLo = (lo: number) => (lo >= 0 ? lo / zb : lo / za);
    const tanHi = (hi: number) => (hi >= 0 ? hi / za : hi / zb);
    let [minX, maxX] = tileRangeX(frame, settings, tanLo(x0), tanHi(x1));
    let [minY, maxY] = tileRangeY(frame, settings, tanLo(y0), tanHi(y1));
    // Beyond the extended FOV no supported ray exists (see captureFrame).
    minX = Math.max(0, minX);
    minY = Math.max(0, minY);
    maxX = Math.min(tiles - 1, maxX);
    maxY = Math.min(tiles - 1, maxY);
    if (minX > maxX || minY > maxY) continue;
    inside = true;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (masks.visible[(layer * tiles + y) * tiles + x]) return 'cell';
      }
    }
  }
  return inside ? 'hidden' : 'outside';
}

/**
 * Paper §3.3: mark grid cells holding a vertex of any directly visible
 * triangle; then admit every other triangle with a vertex in a marked cell.
 * Returns admitted triangle flags.
 */
export function volumetricFilter(geometry: DisocclusionGeometry, visibleTriangles: Uint8Array, cell: number): Uint8Array {
  const out = Uint8Array.from(visibleTriangles);
  if (!(cell > 0)) return out;
  const { positions, indices } = geometry;
  const key = (v: number) =>
    `${Math.floor(positions[v * 3]! / cell)},${Math.floor(positions[v * 3 + 1]! / cell)},${Math.floor(positions[v * 3 + 2]! / cell)}`;
  const marked = new Set<string>();
  for (let tri = 0; tri < visibleTriangles.length; tri++) {
    if (!visibleTriangles[tri]) continue;
    for (let k = 0; k < 3; k++) marked.add(key(indices[tri * 3 + k]!));
  }
  for (let tri = 0; tri < visibleTriangles.length; tri++) {
    if (out[tri]) continue;
    for (let k = 0; k < 3; k++) {
      if (marked.has(key(indices[tri * 3 + k]!))) {
        out[tri] = 1;
        break;
      }
    }
  }
  return out;
}

/** Triangles whose ID is a sample in a non-empty, potentially visible tile. */
export function rawVisibleTriangles(settings: DisocclusionSettings, layers: DisocclusionLayers, masks: DisocclusionMasks, triangleCount: number): Uint8Array {
  const out = new Uint8Array(triangleCount);
  const r = settings.resolution;
  const t = settings.tileSize;
  const tiles = r / t;
  for (let layer = 0; layer < settings.layers; layer++) {
    for (let y = 0; y < r; y++) {
      for (let x = 0; x < r; x++) {
        const id = layers.id[(layer * r + y) * r + x]!;
        if (id === EMPTY_SAMPLE || id >= triangleCount) continue;
        if (masks.visible[(layer * tiles + Math.floor(y / t)) * tiles + Math.floor(x / t)]) out[id] = 1;
      }
    }
  }
  return out;
}

export function classifyTargets(
  frame: DisocclusionFrame,
  settings: DisocclusionSettings,
  layers: DisocclusionLayers,
  masks: DisocclusionMasks,
  geometry: DisocclusionGeometry,
  targetCount: number
): DisocclusionTargetClassification {
  const triangleCount = geometry.triangleTarget.length;
  const rawTriangles = rawVisibleTriangles(settings, layers, masks, triangleCount);
  const filteredTriangles = volumetricFilter(geometry, rawTriangles, settings.filterCell);
  const raw = new Uint8Array(targetCount);
  const filtered = new Uint8Array(targetCount);
  for (let tri = 0; tri < triangleCount; tri++) {
    const target = geometry.triangleTarget[tri]!;
    if (target < 0) continue;
    if (rawTriangles[tri]) raw[target] = 1;
    if (filteredTriangles[tri]) filtered[target] = 1;
  }
  const bounds = targetBounds(geometry, targetCount);
  const expanded = new Uint8Array(targetCount);
  const reason: DisocclusionTargetReason[] = new Array(targetCount);
  for (let target = 0; target < targetCount; target++) {
    if (raw[target]) {
      expanded[target] = 1;
      reason[target] = 'raw';
      continue;
    }
    const min: Vec3 = [bounds.min[target * 3]!, bounds.min[target * 3 + 1]!, bounds.min[target * 3 + 2]!];
    const max: Vec3 = [bounds.max[target * 3]!, bounds.max[target * 3 + 1]!, bounds.max[target * 3 + 2]!];
    if (!(min[0] <= max[0])) {
      // No triangles: nothing static to hide; admit rather than guess.
      expanded[target] = 1;
      reason[target] = 'cell';
      continue;
    }
    const verdict = classifyBox(frame, settings, masks, min, max);
    reason[target] = verdict;
    expanded[target] = verdict === 'hidden' || verdict === 'outside' ? 0 : 1;
  }
  const sum = (a: Uint8Array) => a.reduce((s, v) => s + v, 0);
  let filterOnly = 0;
  for (let target = 0; target < targetCount; target++) if (filtered[target] && !expanded[target]) filterOnly++;
  return {
    reason,
    raw,
    filtered,
    expanded,
    counts: { targets: targetCount, raw: sum(raw), filtered: sum(filtered), expanded: sum(expanded), filterOnly },
  };
}
