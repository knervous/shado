/**
 * Scalar reference for the disocclusion-buffer propagation (paper §3.2,
 * Algorithm 1; reference source disocclusion_stencil_cs / _cones_cs /
 * _vote_cs). Small, readable and independent of the GPU passes so each GPU
 * stage can be compared against it.
 */
import {
  frustumGrowth,
  layerFront,
  lowBits,
  tileRangeX,
  tileRangeY,
  tileTanX,
  tileTanY,
} from './layers';
import type { DisocclusionFrame, DisocclusionLayers, DisocclusionMasks, DisocclusionSettings } from './types';

export const EMPTY_SAMPLE = 0xffffffff;
export const TILE_EMPTY = 0;
export const TILE_OPEN = 1;
export const TILE_CLOSED = 2;

/** Samples written per tile, indexed [layer][tileY][tileX]. */
export function countTiles(data: DisocclusionLayers): Uint32Array {
  const { resolution: r, tileSize: t, layers } = data.settings;
  const tiles = r / t;
  const count = new Uint32Array(tiles * tiles * layers);
  for (let layer = 0; layer < layers; layer++) {
    for (let y = 0; y < r; y++) {
      for (let x = 0; x < r; x++) {
        if (data.depth[(layer * r + y) * r + x] === EMPTY_SAMPLE) continue;
        count[(layer * tiles + Math.floor(y / t)) * tiles + Math.floor(x / t)]! += 1;
      }
    }
  }
  return count;
}

/**
 * Algorithm 1 over per-tile sample counts.
 *
 * Stage 1 (classify): a tile is CLOSED when all tileSize^2 samples are
 * written, otherwise OPEN (empty counts as open). An open tile in any layer
 * but the last spawns disocclusion for bit `layer`: a degenerate frustum
 * (no closed 4-neighbour) marks its own tile column; otherwise a frustum
 * grows toward each closed neighbour and is clamped to the tile's own edge
 * on each open side. Neighbours outside the capture count as open.
 *
 * Stage 2 (propagate): each frustum marks bit `layer` in every tile of each
 * deeper layer its rectangle covers.
 *
 * Stage 3 (gather): a cell at layer J is potentially visible iff bits 0..J-1
 * are all set. Layer 0 is always visible.
 */
export function propagateReference(
  frame: DisocclusionFrame,
  settings: DisocclusionSettings,
  count: Uint32Array
): DisocclusionMasks {
  const tiles = settings.resolution / settings.tileSize;
  const layers = settings.layers;
  const full = settings.tileSize * settings.tileSize;
  const cells = tiles * tiles * layers;
  if (count.length !== cells) throw new Error(`count has ${count.length} entries, expected ${cells}`);
  const state = new Uint8Array(cells);
  const mask = new Uint32Array(cells);
  const column = new Uint32Array(tiles * tiles);
  const at = (layer: number, y: number, x: number) => (layer * tiles + y) * tiles + x;
  for (let i = 0; i < cells; i++) state[i] = count[i] === 0 ? TILE_EMPTY : count[i]! >= full ? TILE_CLOSED : TILE_OPEN;
  const closed = (layer: number, y: number, x: number) =>
    x >= 0 && y >= 0 && x < tiles && y < tiles && state[at(layer, y, x)] === TILE_CLOSED;

  for (let layer = 0; layer < layers - 1; layer++) {
    const zOpen = layerFront(frame, layers, layer);
    for (let ty = 0; ty < tiles; ty++) {
      for (let tx = 0; tx < tiles; tx++) {
        if (state[at(layer, ty, tx)] === TILE_CLOSED) continue;
        const left = closed(layer, ty, tx - 1);
        const right = closed(layer, ty, tx + 1);
        const top = closed(layer, ty - 1, tx);
        const bottom = closed(layer, ty + 1, tx);
        const bit = (1 << layer) >>> 0;
        if (!left && !right && !top && !bottom) {
          column[ty * tiles + tx]! |= bit;
          continue;
        }
        const [x0, x1] = tileTanX(frame, settings, tx);
        const [y0, y1] = tileTanY(frame, settings, ty);
        for (let target = layer + 1; target < layers; target++) {
          const [growX, growY] = frustumGrowth(frame, zOpen, layerFront(frame, layers, target + 1));
          let [minX, maxX] = tileRangeX(frame, settings, x0 - growX, x1 + growX);
          let [minY, maxY] = tileRangeY(frame, settings, y0 - growY, y1 + growY);
          if (!left) minX = tx;
          if (!right) maxX = tx;
          if (!top) minY = ty;
          if (!bottom) maxY = ty;
          minX = Math.max(0, minX);
          minY = Math.max(0, minY);
          maxX = Math.min(tiles - 1, maxX);
          maxY = Math.min(tiles - 1, maxY);
          for (let y = minY; y <= maxY; y++) {
            for (let x = minX; x <= maxX; x++) mask[at(target, y, x)]! |= bit;
          }
        }
      }
    }
  }

  const visible = new Uint8Array(cells);
  for (let layer = 0; layer < layers; layer++) {
    const need = lowBits(layer);
    for (let y = 0; y < tiles; y++) {
      for (let x = 0; x < tiles; x++) {
        const bits = (mask[at(layer, y, x)]! | column[y * tiles + x]!) & need;
        visible[at(layer, y, x)] = bits >>> 0 === need ? 1 : 0;
      }
    }
  }
  return { tilesX: tiles, tilesY: tiles, layers, count, state, mask, column, visible };
}

/** Human-readable dump of one layer's state/visibility, for tests and logs. */
export function describeLayer(masks: DisocclusionMasks, layer: number): string {
  const rows: string[] = [];
  for (let y = 0; y < masks.tilesY; y++) {
    let row = '';
    for (let x = 0; x < masks.tilesX; x++) {
      const i = (layer * masks.tilesY + y) * masks.tilesX + x;
      const s = masks.state[i]!;
      const v = masks.visible[i]!;
      row += s === TILE_CLOSED ? (v ? '#' : 'x') : s === TILE_OPEN ? (v ? 'o' : '-') : v ? '.' : ' ';
    }
    rows.push(row);
  }
  return rows.join('\n');
}
