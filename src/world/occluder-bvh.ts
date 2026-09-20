/**
 * A bounding-volume hierarchy over occluder triangles.
 *
 * ## Why, measured rather than assumed
 *
 * The uniform grid this replaces buckets triangles into fixed 32-unit cells.
 * That is fine for a collision mesh and hopeless for the visual scene: baking
 * Crownward against its assembled opaque objects walked 8.3 cells per segment
 * and tested **640 triangles in each one**, 5,337 per query and 130 billion in
 * total, which is 16k queries a second and a bake that ran out of its budget
 * having tested 32% of its pairs. The duplication factor was only 1.15, so the
 * cost was never the indexing overhead -- it was that a cell holding part of a
 * five-million-triangle city is not a small set.
 *
 * A hierarchy spends log(n) node tests to reach a handful of triangles, and its
 * cost grows with the depth of the tree rather than with the density of the
 * world.
 *
 * ## The same direction of error
 *
 * This answers the identical question as the grid walk and errs the same way.
 * Nodes are tested with a conservative slab test against the segment, so a node
 * that might contain a blocker is always opened; hits within an epsilon of
 * either endpoint are ignored so a surface a viewpoint stands on cannot occlude
 * it; and anything it cannot decide is reported as clear, because a missed
 * blocker costs draw calls while a false blocker costs a hole.
 *
 * ## Layout
 *
 * Flattened into typed arrays rather than objects, so the same buffers can be
 * uploaded to a GPU traversal without a second build. Each node is six floats
 * of bounds plus three integers: first triangle, triangle count, right child.
 * An interior node has count 0 and its left child at `node + 1`.
 */
import type { ShadoWorldPrimitive } from './types';

/** Hits closer than this to either end are the endpoints' own surfaces. */
const END_EPSILON = 1e-3;
const LEAF_TRIANGLES = 8;

export type OccluderBvhCounters = {
  segmentQueries: number;
  columnQueries: number;
  /** Nodes whose bounds were tested; the hierarchy's equivalent of cell visits. */
  nodeVisits: number;
  triangleTests: number;
  blockedQueries: number;
};

/** Bounds on construction itself, not on the queries it will later answer. */
export type OccluderBuildLimits = {
  /** Ceiling on the structure's own allocation, checked before allocating. */
  maxBytes?: number;
  /** Polled while reading geometry; true abandons the build. */
  shouldStop?: () => boolean;
};

/**
 * Peak bytes this structure allocates for n triangles.
 *
 * Every live buffer, not only the large ones: the payload exists twice while
 * it is copied into leaf order, and the per-triangle sidedness, ordering and
 * scratch bounds are small individually and not nothing at five million
 * triangles. Partitioning is done in place on the order array, so there is no
 * recursive scratch to account for.
 */
export function estimateBvhBytes(triangleCount: number): number {
  const payload = triangleCount * 9 * 8;
  const centroids = triangleCount * 3 * 8;
  const bounds = triangleCount * 6 * 8;
  const sides = triangleCount * 2;
  const order = triangleCount * 4;
  const nodes = Math.max(4, 2 * Math.ceil(triangleCount / 4) + 1) * (6 * 8 + 3 * 4);
  return payload * 2 + centroids + bounds + sides + order + nodes;
}

/** How often construction asks whether it should stop. */
const CANCEL_STRIDE = 4096;

/** Thrown to unwind out of recursive construction; never escapes this module. */
const CANCELLED = Symbol('occluder-build-cancelled');

export type OccluderBvh = {
  /**
   * Why this structure is empty despite being asked for geometry, or null
   * when it was built. An abandoned build indexes nothing, and a caller that
   * uses it anyway simply finds no blockers -- which admits more, never less.
   */
  readonly aborted: 'cancelled' | 'over-budget' | null;
  /** Triangle corners, nine doubles each, in build order. */
  readonly triangles: Float64Array;
  /**
   * One byte per triangle: non-zero where the surface draws both faces.
   *
   * A single-sided triangle is invisible from behind, so it blocks only rays
   * that reach its front face. Without this a flat wall panel hides the world
   * from the side a player can see straight through it.
   */
  readonly doubleSided: Uint8Array;
  readonly triangleCount: number;
  /** Six doubles per node: minX, minY, minZ, maxX, maxY, maxZ. */
  readonly nodeBounds: Float64Array;
  /** Three ints per node: firstTriangle, triangleCount, rightChild. */
  readonly nodeMeta: Int32Array;
  readonly nodeCount: number;
  readonly maxDepth: number;
  readonly counters: OccluderBvhCounters;
};

/**
 * Builds the hierarchy by splitting the longest axis at the median centroid.
 *
 * Median rather than a surface-area heuristic: it is one pass per level, it
 * cannot degenerate on the axis-aligned architecture these zones are built
 * from, and the query counters say whether anything better is warranted.
 */
export function buildOccluderBvh(
  primitives: readonly ShadoWorldPrimitive[],
  limits: OccluderBuildLimits = {}
): OccluderBvh {
  let total = 0;
  for (const primitive of primitives) total += Math.floor(primitive.indices.length / 3);
  /*
   * Refuse the allocation before making it. A build that is going to exceed
   * its budget should say so while the memory is still unclaimed, not after
   * the process has already taken it: three Float64Arrays over the triangle
   * set is the dominant cost and is exactly predictable from the count.
   */
  if (limits.maxBytes !== undefined && estimateBvhBytes(total) > limits.maxBytes) {
    return emptyBvh('over-budget');
  }
  if (limits.shouldStop?.()) return emptyBvh('cancelled');
  const triangles = new Float64Array(total * 9);
  const sides = new Uint8Array(total);
  const centroids = new Float64Array(total * 3);
  const triangleBounds = new Float64Array(total * 6);
  let write = 0;
  let triangle = 0;
  const stop = limits.shouldStop;
  /*
   * Asked often enough to interrupt one enormous primitive, rarely enough to
   * cost nothing: a scene is not always many meshes, and a single
   * ten-thousand-triangle mesh used to run to completion because the only
   * check was between primitives.
   */
  let sinceCheck = 0;
  const shouldStop = (): boolean => {
    if (!stop) return false;
    if (++sinceCheck < CANCEL_STRIDE) return false;
    sinceCheck = 0;
    return stop();
  };
  let aborted: OccluderBvh['aborted'] = null;
  for (const primitive of primitives) {
    if (stop?.()) { aborted = 'cancelled'; break; }
    const { positions, indices } = primitive;
    // Unknown sidedness blocks from both sides: that is what collision
    // geometry has always done, and narrowing it silently would change every
    // historical measurement.
    const bothFaces = primitive.doubleSided !== false;
    for (let index = 0; index + 2 < indices.length; index += 3) {
      if (shouldStop()) { aborted = 'cancelled'; break; }
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let corner = 0; corner < 3; corner += 1) {
        const base = Number(indices[index + corner]) * 3;
        const x = Number(positions[base]);
        const y = Number(positions[base + 1]);
        const z = Number(positions[base + 2]);
        triangles[write++] = x;
        triangles[write++] = y;
        triangles[write++] = z;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }
      triangleBounds[triangle * 6] = minX;
      triangleBounds[triangle * 6 + 1] = minY;
      triangleBounds[triangle * 6 + 2] = minZ;
      triangleBounds[triangle * 6 + 3] = maxX;
      triangleBounds[triangle * 6 + 4] = maxY;
      triangleBounds[triangle * 6 + 5] = maxZ;
      centroids[triangle * 3] = (minX + maxX) / 2;
      centroids[triangle * 3 + 1] = (minY + maxY) / 2;
      centroids[triangle * 3 + 2] = (minZ + maxZ) / 2;
      sides[triangle] = bothFaces ? 1 : 0;
      triangle += 1;
    }
    if (aborted) break;
  }
  const triangleCount = triangle;
  const order = new Int32Array(triangleCount);
  for (let index = 0; index < triangleCount; index += 1) order[index] = index;

  /*
   * Node storage grows rather than being predicted.
   *
   * Median splitting does not produce n/LEAF_TRIANGLES leaves: halving a node
   * of nine gives leaves of four and five, so a scene of 400 triangles builds
   * 64 leaves and 127 nodes where a naive `2n/LEAF` bound allows 101. Typed
   * arrays drop out-of-range writes in silence, so that bound did not
   * overflow -- it produced a tree missing the nodes it could not store, and
   * queries that walked straight past real geometry. Growing cannot be wrong
   * by arithmetic.
   */
  let capacity = Math.max(4, 2 * Math.ceil(triangleCount / 4) + 1);
  let nodeBounds = new Float64Array(capacity * 6);
  let nodeMeta = new Int32Array(capacity * 3);
  let nodeCount = 0;
  let maxDepth = 0;
  const reserve = (): void => {
    if (nodeCount < capacity) return;
    capacity *= 2;
    const grownBounds = new Float64Array(capacity * 6);
    grownBounds.set(nodeBounds);
    nodeBounds = grownBounds;
    const grownMeta = new Int32Array(capacity * 3);
    grownMeta.set(nodeMeta);
    nodeMeta = grownMeta;
  };

  const boundsOf = (start: number, count: number, node: number): void => {
    if (stop && shouldStop()) throw CANCELLED;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let index = start; index < start + count; index += 1) {
      const base = order[index]! * 6;
      if (triangleBounds[base]! < minX) minX = triangleBounds[base]!;
      if (triangleBounds[base + 1]! < minY) minY = triangleBounds[base + 1]!;
      if (triangleBounds[base + 2]! < minZ) minZ = triangleBounds[base + 2]!;
      if (triangleBounds[base + 3]! > maxX) maxX = triangleBounds[base + 3]!;
      if (triangleBounds[base + 4]! > maxY) maxY = triangleBounds[base + 4]!;
      if (triangleBounds[base + 5]! > maxZ) maxZ = triangleBounds[base + 5]!;
    }
    nodeBounds[node * 6] = minX;
    nodeBounds[node * 6 + 1] = minY;
    nodeBounds[node * 6 + 2] = minZ;
    nodeBounds[node * 6 + 3] = maxX;
    nodeBounds[node * 6 + 4] = maxY;
    nodeBounds[node * 6 + 5] = maxZ;
  };

  /** Returns the node index; children are laid out immediately after it. */
  const build = (start: number, count: number, depth: number): number => {
    if (stop && shouldStop()) throw CANCELLED;
    reserve();
    const node = nodeCount++;
    if (depth > maxDepth) maxDepth = depth;
    boundsOf(start, count, node);
    if (count <= LEAF_TRIANGLES) {
      nodeMeta[node * 3] = start;
      nodeMeta[node * 3 + 1] = count;
      nodeMeta[node * 3 + 2] = -1;
      return node;
    }
    const extentX = nodeBounds[node * 6 + 3]! - nodeBounds[node * 6]!;
    const extentY = nodeBounds[node * 6 + 4]! - nodeBounds[node * 6 + 1]!;
    const extentZ = nodeBounds[node * 6 + 5]! - nodeBounds[node * 6 + 2]!;
    const axis = extentX >= extentY && extentX >= extentZ ? 0 : extentY >= extentZ ? 1 : 2;
    const half = count >> 1;
    /*
     * Only the median matters, so the range is partitioned around it in place
     * rather than sorted. That removes an array allocation and a full sort per
     * level -- the largest of which was over every triangle in the zone -- and
     * gives cancellation somewhere to be checked inside the work.
     */
    selectNth(order, centroids, axis, start, start + count - 1, start + half, shouldStop);
    nodeMeta[node * 3] = start;
    nodeMeta[node * 3 + 1] = 0;
    build(start, half, depth + 1);
    nodeMeta[node * 3 + 2] = build(start + half, count - half, depth + 1);
    return node;
  };
  if (aborted) return emptyBvh(aborted);
  try {
    if (triangleCount) build(0, triangleCount, 0);
  } catch (error) {
    if (error === CANCELLED) return emptyBvh('cancelled');
    throw error;
  }
  if (!triangleCount) {
    nodeCount = 1;
    nodeMeta[0] = 0;
    nodeMeta[1] = 0;
    nodeMeta[2] = -1;
  }

  // Reorder the triangle payload into leaf order so a leaf reads contiguously.
  const ordered = new Float64Array(triangleCount * 9);
  const orderedSides = new Uint8Array(triangleCount);
  for (let index = 0; index < triangleCount; index += 1) {
    // The reordering is a full pass over the payload and is interruptible too.
    if (shouldStop()) return emptyBvh('cancelled');
    ordered.set(triangles.subarray(order[index]! * 9, order[index]! * 9 + 9), index * 9);
    orderedSides[index] = sides[order[index]!]!;
  }
  return {
    aborted: null,
    triangles: ordered,
    doubleSided: orderedSides,
    triangleCount,
    nodeBounds,
    nodeMeta,
    nodeCount,
    maxDepth,
    counters: {
      segmentQueries: 0,
      columnQueries: 0,
      nodeVisits: 0,
      triangleTests: 0,
      blockedQueries: 0,
    },
  };
}

/**
 * Partitions `order[low..high]` in place so that position `nth` holds the
 * element it would hold if the range were sorted by centroid on `axis`.
 *
 * Quickselect, iterated rather than recursed. The build only needs the median,
 * and this is linear where a sort is n log n -- on the root range that is
 * every triangle in the zone. `shouldStop` is polled per partition pass so a
 * cancellation does not have to wait for the largest one to finish.
 */
function selectNth(
  order: Int32Array,
  centroids: Float64Array,
  axis: number,
  low: number,
  high: number,
  nth: number,
  shouldStop: () => boolean
): void {
  const key = (index: number): number => centroids[order[index]! * 3 + axis]!;
  const swap = (left: number, right: number): void => {
    const value = order[left]!;
    order[left] = order[right]!;
    order[right] = value;
  };
  while (low < high) {
    if (shouldStop()) throw CANCELLED;
    // Median of three, which keeps sorted and reversed input off the worst case.
    const middle = (low + high) >> 1;
    if (key(middle) < key(low)) swap(middle, low);
    if (key(high) < key(low)) swap(high, low);
    if (key(high) < key(middle)) swap(high, middle);
    const pivot = key(middle);
    let left = low;
    let right = high;
    while (left <= right) {
      while (key(left) < pivot) left += 1;
      while (key(right) > pivot) right -= 1;
      if (left <= right) {
        swap(left, right);
        left += 1;
        right -= 1;
      }
    }
    if (nth <= right) high = right;
    else if (nth >= left) low = left;
    else return;
  }
}

/** An index over nothing, which therefore blocks nothing. */
function emptyBvh(aborted: OccluderBvh['aborted']): OccluderBvh {
  const nodeMeta = new Int32Array(3);
  nodeMeta[2] = -1;
  return {
    aborted,
    triangles: new Float64Array(0),
    triangleCount: 0,
    doubleSided: new Uint8Array(0),
    nodeBounds: new Float64Array(6),
    nodeMeta,
    nodeCount: 1,
    maxDepth: 0,
    counters: {
      segmentQueries: 0,
      columnQueries: 0,
      nodeVisits: 0,
      triangleTests: 0,
      blockedQueries: 0,
    },
  };
}

/**
 * Is the straight line from a to b interrupted by geometry?
 *
 * Answers exactly what the grid walk answers, including the endpoint epsilon,
 * so the two are differentially testable against each other and against a
 * brute-force sweep.
 */
export function bvhSegmentBlocked(
  bvh: OccluderBvh,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number
): boolean {
  bvh.counters.segmentQueries += 1;
  const dx = bx - ax, dy = by - ay, dz = bz - az;
  if (Math.hypot(dx, dy, dz) < END_EPSILON) return false;
  const invX = dx === 0 ? Infinity : 1 / dx;
  const invY = dy === 0 ? Infinity : 1 / dy;
  const invZ = dz === 0 ? Infinity : 1 / dz;
  const stack = SEGMENT_STACK;
  let depth = 0;
  stack[depth++] = 0;
  while (depth > 0) {
    const node = stack[--depth]!;
    bvh.counters.nodeVisits += 1;
    if (!slabsIntersect(bvh, node, ax, ay, az, invX, invY, invZ)) continue;
    const count = bvh.nodeMeta[node * 3 + 1]!;
    if (count === 0) {
      // Interior: left is adjacent, right is recorded. Both are candidates;
      // there is no near-first ordering because any hit at all ends the query.
      if (depth + 2 >= stack.length) return false;
      stack[depth++] = node + 1;
      stack[depth++] = bvh.nodeMeta[node * 3 + 2]!;
      continue;
    }
    const first = bvh.nodeMeta[node * 3]!;
    for (let index = first; index < first + count; index += 1) {
      bvh.counters.triangleTests += 1;
      if (triangleBlocks(bvh.triangles, index * 9, ax, ay, az, dx, dy, dz, bvh.doubleSided[index] !== 0)) {
        bvh.counters.blockedQueries += 1;
        return true;
      }
    }
  }
  return false;
}

/**
 * Highest surface under a column, or null where there is none.
 *
 * The same contract as the grid's: a column with nothing under it has no floor
 * to stand on, and the caller treats that as "cannot sample" rather than
 * "empty".
 */
export function bvhHighestSurfaceAt(
  bvh: OccluderBvh,
  x: number,
  z: number
): number | null {
  bvh.counters.columnQueries += 1;
  let best: number | null = null;
  const stack = COLUMN_STACK;
  let depth = 0;
  stack[depth++] = 0;
  while (depth > 0) {
    const node = stack[--depth]!;
    bvh.counters.nodeVisits += 1;
    const base = node * 6;
    if (x < bvh.nodeBounds[base]! || x > bvh.nodeBounds[base + 3]!) continue;
    if (z < bvh.nodeBounds[base + 2]! || z > bvh.nodeBounds[base + 5]!) continue;
    if (best !== null && bvh.nodeBounds[base + 4]! <= best) continue;
    const count = bvh.nodeMeta[node * 3 + 1]!;
    if (count === 0) {
      if (depth + 2 >= stack.length) return best;
      stack[depth++] = node + 1;
      stack[depth++] = bvh.nodeMeta[node * 3 + 2]!;
      continue;
    }
    const first = bvh.nodeMeta[node * 3]!;
    for (let index = first; index < first + count; index += 1) {
      bvh.counters.triangleTests += 1;
      const height = columnHeight(bvh.triangles, index * 9, x, z);
      if (height !== null && (best === null || height > best)) best = height;
    }
  }
  return best;
}

/**
 * Every surface under a column, highest first.
 *
 * `bvhHighestSurfaceAt` answers where the roof is, which indoors is the
 * ceiling. A camera volume needs every floor a player can stand on: the
 * street under an arcade, the room under a roof, both decks of a bridge, each
 * storey of a crypt. Heights within `merge` of each other are one surface, so
 * a floor built from many triangles is not reported many times.
 */
export function bvhColumnSurfaces(
  bvh: OccluderBvh,
  x: number,
  z: number,
  limit = 8,
  merge = 0.5
): number[] {
  bvh.counters.columnQueries += 1;
  const heights: number[] = [];
  const stack = COLUMN_STACK;
  let depth = 0;
  stack[depth++] = 0;
  while (depth > 0) {
    const node = stack[--depth]!;
    bvh.counters.nodeVisits += 1;
    const base = node * 6;
    if (x < bvh.nodeBounds[base]! || x > bvh.nodeBounds[base + 3]!) continue;
    if (z < bvh.nodeBounds[base + 2]! || z > bvh.nodeBounds[base + 5]!) continue;
    const count = bvh.nodeMeta[node * 3 + 1]!;
    if (count === 0) {
      if (depth + 2 >= stack.length) break;
      stack[depth++] = node + 1;
      stack[depth++] = bvh.nodeMeta[node * 3 + 2]!;
      continue;
    }
    const first = bvh.nodeMeta[node * 3]!;
    for (let index = first; index < first + count; index += 1) {
      bvh.counters.triangleTests += 1;
      const height = columnHeight(bvh.triangles, index * 9, x, z);
      if (height !== null) heights.push(height);
    }
  }
  if (!heights.length) return heights;
  heights.sort((left, right) => right - left);
  const surfaces: number[] = [heights[0]!];
  for (const height of heights) {
    if (surfaces.length >= limit) break;
    if (surfaces[surfaces.length - 1]! - height > merge) surfaces.push(height);
  }
  return surfaces;
}

const SEGMENT_STACK = new Int32Array(256);
const COLUMN_STACK = new Int32Array(256);

/** Conservative slab test of a node against the segment's parameter range. */
function slabsIntersect(
  bvh: OccluderBvh,
  node: number,
  ax: number, ay: number, az: number,
  invX: number, invY: number, invZ: number
): boolean {
  const base = node * 6;
  let near = 0;
  let far = 1;
  for (let axis = 0; axis < 3; axis += 1) {
    const origin = axis === 0 ? ax : axis === 1 ? ay : az;
    const inverse = axis === 0 ? invX : axis === 1 ? invY : invZ;
    const min = bvh.nodeBounds[base + axis]!;
    const max = bvh.nodeBounds[base + 3 + axis]!;
    if (!Number.isFinite(inverse)) {
      // Parallel to this axis: the segment either lies within the slab or misses.
      if (origin < min || origin > max) return false;
      continue;
    }
    let low = (min - origin) * inverse;
    let high = (max - origin) * inverse;
    if (low > high) { const swap = low; low = high; high = swap; }
    if (low > near) near = low;
    if (high < far) far = high;
    if (near > far) return false;
  }
  return true;
}

/** Möller-Trumbore, with the endpoints owning their own surfaces. */
function triangleBlocks(
  triangles: Float64Array,
  offset: number,
  ax: number, ay: number, az: number,
  dx: number, dy: number, dz: number,
  doubleSided: boolean
): boolean {
  const e1x = triangles[offset + 3]! - triangles[offset]!;
  const e1y = triangles[offset + 4]! - triangles[offset + 1]!;
  const e1z = triangles[offset + 5]! - triangles[offset + 2]!;
  const e2x = triangles[offset + 6]! - triangles[offset]!;
  const e2y = triangles[offset + 7]! - triangles[offset + 1]!;
  const e2z = triangles[offset + 8]! - triangles[offset + 2]!;
  const hx = dy * e2z - dz * e2y;
  const hy = dz * e2x - dx * e2z;
  const hz = dx * e2y - dy * e2x;
  const det = e1x * hx + e1y * hy + e1z * hz;
  if (det > -1e-12 && det < 1e-12) return false;
  /*
   * glTF front faces are counter-clockwise in a right-handed frame, which is
   * a positive determinant here. A negative one means this ray reached the
   * back of the surface, and the back of a single-sided surface is not drawn.
   */
  if (!doubleSided && det < 0) return false;
  const inv = 1 / det;
  const sx = ax - triangles[offset]!;
  const sy = ay - triangles[offset + 1]!;
  const sz = az - triangles[offset + 2]!;
  const u = inv * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return false;
  const qx = sy * e1z - sz * e1y;
  const qy = sz * e1x - sx * e1z;
  const qz = sx * e1y - sy * e1x;
  const v = inv * (dx * qx + dy * qy + dz * qz);
  if (v < 0 || u + v > 1) return false;
  const hit = inv * (e2x * qx + e2y * qy + e2z * qz);
  return hit > END_EPSILON && hit < 1 - END_EPSILON;
}

/** Height where a downward line through (x, z) meets this triangle, or null. */
function columnHeight(
  triangles: Float64Array,
  offset: number,
  x: number,
  z: number
): number | null {
  const ax = triangles[offset]!, ay = triangles[offset + 1]!, az = triangles[offset + 2]!;
  const bx = triangles[offset + 3]!, by = triangles[offset + 4]!, bz = triangles[offset + 5]!;
  const cx = triangles[offset + 6]!, cy = triangles[offset + 7]!, cz = triangles[offset + 8]!;
  const area = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
  if (Math.abs(area) < 1e-12) return null;
  const w0 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) /
    ((bz - cz) * (ax - cx) + (cx - bx) * (az - cz));
  const w1 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) /
    ((bz - cz) * (ax - cx) + (cx - bx) * (az - cz));
  const w2 = 1 - w0 - w1;
  if (w0 < 0 || w1 < 0 || w2 < 0) return null;
  return w0 * ay + w1 * by + w2 * cy;
}
