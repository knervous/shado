/**
 * Segment-versus-world occlusion, for baking a real potentially-visible set.
 *
 * The visibility package has always had the shape of a PVS -- a region-to-region
 * bitset with a cell-to-region map and persistent bypass lists -- but its rows
 * were filled by a distance flood, so every region within `maxDistance` was
 * admitted and almost nothing was culled. This is the part that was missing:
 * something that can answer whether one place can see another.
 *
 * ## Why exact triangles and not a voxel grid
 *
 * A conservative voxel grid marks a whole cell solid wherever a surface passes
 * through it. That is the wrong direction of error for occlusion: it blocks
 * every ground-skimming ray and reports far more hidden than truly is, which
 * for a PVS means culling things the player can see. Measured while building
 * this, an 8-unit grid called 100% of a zone occluded from a street and placed
 * a ridge viewpoint inside a hill it demonstrably stands on. So the test is
 * exact ray-triangle, accelerated by a uniform grid it walks with a DDA rather
 * than by sampling points along the ray.
 *
 * ## The direction of error that is allowed
 *
 * Saying "visible" when something is hidden costs draw calls. Saying "hidden"
 * when something is visible is a hole in the world the player can walk up to.
 * Every approximation here therefore errs towards visible: the DDA visits every
 * cell the segment touches rather than sampling it, hits are ignored within a
 * small epsilon of either endpoint so a surface a viewpoint sits on cannot
 * occlude it, and a caller that cannot find geometry to sample is expected to
 * fall back to admitting the region.
 */
import type { ShadoWorldBounds, ShadoWorldPrimitive } from './types';

/** Hits closer than this to either end are the endpoints' own surfaces. */
const END_EPSILON = 1e-3;

/**
 * What the queries actually cost, counted rather than inferred.
 *
 * A bake time alone cannot say whether it was spent walking cells, testing
 * triangles or re-testing the same triangle out of several buckets, and the
 * three have different fixes. These are mutable on the grid so the hot path
 * stays a plain increment and no signature carries a stats argument.
 */
export type OccluderGridCounters = {
  /** Calls to {@link segmentBlocked}. */
  segmentQueries: number;
  /** Calls to {@link highestSurfaceAt}; these walk a column, not a segment. */
  columnQueries: number;
  /** Grid cells entered by a DDA walk, summed over all queries. */
  cellVisits: number;
  /** Möller-Trumbore evaluations, including the ones that miss. */
  triangleTests: number;
  /** Queries that returned true, i.e. found a blocker. */
  blockedQueries: number;
};

export type OccluderGrid = {
  /**
   * Triangle references across all buckets. Larger than `triangleCount`
   * because a triangle spanning several cells is referenced by each: the ratio
   * is how much duplication this cell size is buying its early-outs with.
   */
  readonly bucketReferences: number;
  readonly counters: OccluderGridCounters;
  readonly triangles: Float64Array;
  readonly triangleCount: number;
  readonly buckets: Map<number, number[]>;
  readonly size: number;
  readonly originX: number;
  readonly originY: number;
  readonly originZ: number;
  readonly countX: number;
  readonly countY: number;
  readonly countZ: number;
};

/**
 * Index occluder triangles into a uniform grid.
 *
 * `cellSize` trades build memory against how many triangles a segment has to
 * test per step. 32 world units is about a wall panel on these zones.
 */
export function buildOccluderGrid(
  primitives: readonly ShadoWorldPrimitive[],
  bounds: ShadoWorldBounds,
  cellSize = 32
): OccluderGrid {
  let total = 0;
  for (const primitive of primitives) total += primitive.indices.length / 3;
  const triangles = new Float64Array(total * 9);
  let write = 0;
  for (const primitive of primitives) {
    const { positions, indices } = primitive;
    for (let i = 0; i < indices.length; i += 3) {
      for (let corner = 0; corner < 3; corner++) {
        const base = Number(indices[i + corner]) * 3;
        triangles[write++] = Number(positions[base]);
        triangles[write++] = Number(positions[base + 1]);
        triangles[write++] = Number(positions[base + 2]);
      }
    }
  }
  const triangleCount = write / 9;
  const originX = bounds.min[0];
  const originY = bounds.min[1];
  const originZ = bounds.min[2];
  const countX = Math.max(1, Math.ceil((bounds.max[0] - originX) / cellSize) + 1);
  const countY = Math.max(1, Math.ceil((bounds.max[1] - originY) / cellSize) + 1);
  const countZ = Math.max(1, Math.ceil((bounds.max[2] - originZ) / cellSize) + 1);
  const buckets = new Map<number, number[]>();
  const counters: OccluderGridCounters = {
    segmentQueries: 0,
    columnQueries: 0,
    cellVisits: 0,
    triangleTests: 0,
    blockedQueries: 0,
  };
  let bucketReferences = 0;
  const grid = {
    bucketReferences,
    counters,
    triangles,
    triangleCount,
    buckets,
    size: cellSize,
    originX,
    originY,
    originZ,
    countX,
    countY,
    countZ,
  };
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const offset = triangle * 9;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let corner = 0; corner < 3; corner++) {
      const x = triangles[offset + corner * 3]!;
      const y = triangles[offset + corner * 3 + 1]!;
      const z = triangles[offset + corner * 3 + 2]!;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const x0 = clampIndex((minX - originX) / cellSize, countX);
    const x1 = clampIndex((maxX - originX) / cellSize, countX);
    const y0 = clampIndex((minY - originY) / cellSize, countY);
    const y1 = clampIndex((maxY - originY) / cellSize, countY);
    const z0 = clampIndex((minZ - originZ) / cellSize, countZ);
    const z1 = clampIndex((maxZ - originZ) / cellSize, countZ);
    for (let y = y0; y <= y1; y++) {
      for (let z = z0; z <= z1; z++) {
        for (let x = x0; x <= x1; x++) {
          const key = bucketKey(grid, x, y, z);
          const list = buckets.get(key);
          if (list) list.push(triangle);
          else buckets.set(key, [triangle]);
          bucketReferences += 1;
        }
      }
    }
  }
  return { ...grid, bucketReferences };
}

const clampIndex = (value: number, count: number): number =>
  Math.min(count - 1, Math.max(0, Math.floor(value)));

const bucketKey = (grid: OccluderGrid, x: number, y: number, z: number): number =>
  (y * grid.countZ + z) * grid.countX + x;

/**
 * Is the straight line from a to b interrupted by geometry?
 *
 * Walks the grid with a 3D DDA -- every cell the segment passes through, in
 * order, with no cell skipped. An earlier version stepped along the ray at a
 * fixed interval and checked the neighbourhood of each step, which is both
 * slower and unsound: a thin wall between two samples is missed, and a missed
 * wall is a region wrongly marked visible, which is merely wasteful, while the
 * 27-cell neighbourhood it needed to compensate made it slow enough to be
 * unusable at bake scale.
 */
export function segmentBlocked(
  grid: OccluderGrid,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number
): boolean {
  grid.counters.segmentQueries += 1;
  const dx = bx - ax, dy = by - ay, dz = bz - az;
  const length = Math.hypot(dx, dy, dz);
  if (length < END_EPSILON) return false;

  const { size, originX, originY, originZ, countX, countY, countZ } = grid;
  let cx = clampIndex((ax - originX) / size, countX);
  let cy = clampIndex((ay - originY) / size, countY);
  let cz = clampIndex((az - originZ) / size, countZ);
  const endX = clampIndex((bx - originX) / size, countX);
  const endY = clampIndex((by - originY) / size, countY);
  const endZ = clampIndex((bz - originZ) / size, countZ);

  const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0;
  const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0;
  const stepZ = dz > 0 ? 1 : dz < 0 ? -1 : 0;
  const nextBoundary = (origin: number, index: number, step: number): number =>
    origin + (step > 0 ? index + 1 : index) * size;
  const tForAxis = (target: number, start: number, delta: number): number =>
    delta === 0 ? Infinity : (target - start) / delta;

  let tMaxX = stepX === 0 ? Infinity : tForAxis(nextBoundary(originX, cx, stepX), ax, dx);
  let tMaxY = stepY === 0 ? Infinity : tForAxis(nextBoundary(originY, cy, stepY), ay, dy);
  let tMaxZ = stepZ === 0 ? Infinity : tForAxis(nextBoundary(originZ, cz, stepZ), az, dz);
  const tDeltaX = stepX === 0 ? Infinity : Math.abs(size / dx);
  const tDeltaY = stepY === 0 ? Infinity : Math.abs(size / dy);
  const tDeltaZ = stepZ === 0 ? Infinity : Math.abs(size / dz);

  // A segment crosses at most this many cells; the bound stops a degenerate
  // direction turning a bake into a hang.
  const limit = countX + countY + countZ + 3;
  for (let visited = 0; visited <= limit; visited++) {
    grid.counters.cellVisits += 1;
    const list = grid.buckets.get(bucketKey(grid, cx, cy, cz));
    if (list && hitsAny(grid, list, ax, ay, az, dx, dy, dz)) {
      grid.counters.blockedQueries += 1;
      return true;
    }
    if (cx === endX && cy === endY && cz === endZ) return false;
    if (tMaxX < tMaxY && tMaxX < tMaxZ) {
      if (tMaxX > 1) return false;
      cx += stepX; tMaxX += tDeltaX;
      if (cx < 0 || cx >= countX) return false;
    } else if (tMaxY < tMaxZ) {
      if (tMaxY > 1) return false;
      cy += stepY; tMaxY += tDeltaY;
      if (cy < 0 || cy >= countY) return false;
    } else {
      if (tMaxZ > 1) return false;
      cz += stepZ; tMaxZ += tDeltaZ;
      if (cz < 0 || cz >= countZ) return false;
    }
  }
  return false;
}

/** Möller–Trumbore against a bucket, stopping at the first real hit. */
function hitsAny(
  grid: OccluderGrid,
  list: readonly number[],
  ax: number, ay: number, az: number,
  dx: number, dy: number, dz: number
): boolean {
  const t = grid.triangles;
  for (const triangle of list) {
    // Counted here rather than by bucket length: this loop returns on the
    // first hit, so the remainder of the bucket is never evaluated.
    grid.counters.triangleTests += 1;
    const o = triangle * 9;
    const e1x = t[o + 3]! - t[o]!, e1y = t[o + 4]! - t[o + 1]!, e1z = t[o + 5]! - t[o + 2]!;
    const e2x = t[o + 6]! - t[o]!, e2y = t[o + 7]! - t[o + 1]!, e2z = t[o + 8]! - t[o + 2]!;
    const hx = dy * e2z - dz * e2y;
    const hy = dz * e2x - dx * e2z;
    const hz = dx * e2y - dy * e2x;
    const det = e1x * hx + e1y * hy + e1z * hz;
    if (det > -1e-12 && det < 1e-12) continue;
    const inv = 1 / det;
    const sx = ax - t[o]!, sy = ay - t[o + 1]!, sz = az - t[o + 2]!;
    const u = inv * (sx * hx + sy * hy + sz * hz);
    if (u < 0 || u > 1) continue;
    const qx = sy * e1z - sz * e1y;
    const qy = sz * e1x - sx * e1z;
    const qz = sx * e1y - sy * e1x;
    const v = inv * (dx * qx + dy * qy + dz * qz);
    if (v < 0 || u + v > 1) continue;
    const hit = inv * (e2x * qx + e2y * qy + e2z * qz);
    // Endpoints own their surfaces: the ground a viewpoint stands on, and the
    // facing wall of whatever is being looked at, must not count as occluders.
    if (hit > END_EPSILON && hit < 1 - END_EPSILON) return true;
  }
  return false;
}

/**
 * Highest occluder surface under a column, or null where there is none.
 *
 * Used to put sample viewpoints at plausible eye heights rather than at an
 * arbitrary Y. A column with no geometry has no floor to stand on, and the
 * caller treats that as "cannot sample" rather than "empty", because an empty
 * region that gets culled is exactly the hole this must not create.
 */
export function highestSurfaceAt(
  grid: OccluderGrid,
  x: number,
  z: number,
  bounds: ShadoWorldBounds
): number | null {
  grid.counters.columnQueries += 1;
  const top = bounds.max[1] + grid.size;
  const bottom = bounds.min[1] - grid.size;
  const dy = bottom - top;
  let best: number | null = null;
  const cx = clampIndex((x - grid.originX) / grid.size, grid.countX);
  const cz = clampIndex((z - grid.originZ) / grid.size, grid.countZ);
  for (let cy = grid.countY - 1; cy >= 0; cy--) {
    const list = grid.buckets.get(bucketKey(grid, cx, cy, cz));
    if (!list) continue;
    const t = grid.triangles;
    for (const triangle of list) {
      const o = triangle * 9;
      const e1x = t[o + 3]! - t[o]!, e1y = t[o + 4]! - t[o + 1]!, e1z = t[o + 5]! - t[o + 2]!;
      const e2x = t[o + 6]! - t[o]!, e2y = t[o + 7]! - t[o + 1]!, e2z = t[o + 8]! - t[o + 2]!;
      // d = (0, dy, 0), so d x e2 = (dy*e2z, 0, -dy*e2x). Writing that out by
      // hand once put the -dy*e2x term in y instead of z, which leaves a
      // determinant that is still non-zero on an axis-aligned quad -- so it
      // passes a test built from one and quietly misses most real ground.
      const hx = dy * e2z, hy = 0, hz = -dy * e2x;
      const det = e1x * hx + e1y * hy + e1z * hz;
      if (det > -1e-12 && det < 1e-12) continue;
      const inv = 1 / det;
      const sx = x - t[o]!, sy = top - t[o + 1]!, sz = z - t[o + 2]!;
      const u = inv * (sx * hx + sy * hy + sz * hz);
      if (u < 0 || u > 1) continue;
      const qx = sy * e1z - sz * e1y;
      const qy = sz * e1x - sx * e1z;
      const qz = sx * e1y - sy * e1x;
      const v = inv * dy * qy;
      if (v < 0 || u + v > 1) continue;
      const hit = inv * (e2x * qx + e2y * qy + e2z * qz);
      if (hit < 0 || hit > 1) continue;
      const y = top + dy * hit;
      if (best === null || y > best) best = y;
    }
    if (best !== null) return best;
  }
  return best;
}
