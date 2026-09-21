/**
 * Occluders as prototypes placed many times, rather than as one flat soup.
 *
 * ## Why
 *
 * Crownward stamps 9,521 objects from 249 prototypes. Expanding each stamp
 * into world-space triangles turns 145,140 eligible prototype triangles into
 * **4,085,341** placed ones -- the same geometry written out twenty-eight
 * times -- and that expansion is what costs the bake its memory (1.87 GB peak)
 * and most of its time (155 s against 56 s for collision alone). The geometry
 * was never the problem; the copying was.
 *
 * Here each prototype is indexed once, in its own local space, and each stamp
 * is a transform. A segment query walks a top-level structure over instance
 * bounds, and for every instance it reaches, transforms the segment into that
 * prototype's space and asks the prototype's own hierarchy.
 *
 * ## The parameter is preserved, which is what makes this legal
 *
 * An affine transform maps the segment a→b to inv*a→inv*b and carries the
 * parameter t with it unchanged: the point at t along the local segment is the
 * image of the point at t along the world segment. So the endpoint epsilon
 * that stops a surface occluding the viewpoint standing on it means the same
 * thing in both spaces, and no distance has to be rescaled.
 *
 * ## Mirrored instances
 *
 * A prototype's hierarchy stores which face is the front in prototype space.
 * An instance with a negative determinant reverses that in world space, so a
 * ray reaching the world-space front arrives at the local-space back. Such
 * instances flip the facing test rather than silently blocking from the side
 * a player can see through.
 */
import {
  OccluderBuildGuard,
  buildOccluderBvh,
  isOccluderBuildCancelled,
  selectNth,
  type OccluderBuildLimits,
  type OccluderBvh,
} from './occluder-bvh';
import type { ShadoWorldPrimitive } from './types';

/** Hits closer than this to either end are the endpoints' own surfaces. */
const END_EPSILON = 1e-3;

export type OccluderInstanceSource = {
  /** Index into the prototype list this instance places. */
  prototype: number;
  /** Column-major world transform, as the runtime composes it. */
  matrix: readonly number[];
};

export type InstancedOccluders = {
  readonly prototypes: readonly OccluderBvh[];
  /** Per instance: which prototype, its inverse transform and world bounds. */
  readonly instancePrototype: Int32Array;
  readonly instanceInverse: Float64Array;
  readonly instanceBounds: Float64Array;
  /** Non-zero where the instance mirrors, so the facing test flips. */
  readonly instanceMirrored: Uint8Array;
  /** Non-zero where the instance places geometry; zero for an empty prototype. */
  readonly instanceValid: Uint8Array;
  readonly instanceCount: number;
  /** Top-level hierarchy over instance bounds: six doubles, three ints each. */
  readonly nodeBounds: Float64Array;
  readonly nodeMeta: Int32Array;
  readonly nodeCount: number;
  /** Instance ids in leaf order. */
  readonly order: Int32Array;
  readonly counters: {
    segmentQueries: number;
    blockedQueries: number;
    columnQueries: number;
    nodeVisits: number;
    triangleTests: number;
    instanceVisits: number;
  };
  readonly aborted: OccluderBvh['aborted'];
  /** Triangles indexed once, rather than once per placement. */
  readonly uniqueTriangles: number;
  /** Triangles a flat expansion would have written. */
  readonly placedTriangles: number;
};

const LEAF_INSTANCES = 4;

/**
 * Bytes the top level holds for `count` placements at their peak: the
 * per-instance arrays it keeps, the centroid scratch the partition reads, and
 * the node arrays. Priced BEFORE any of them is allocated.
 */
export function estimateInstancedTopBytes(count: number): number {
  const kept =
    count * 4 + // instancePrototype
    count * 16 * 8 + // instanceInverse
    count * 6 * 8 + // instanceBounds
    count + // instanceMirrored
    count + // instanceValid
    count * 4; // order
  const scratch = count * 3 * 8; // centroids
  const nodes = topNodeCapacity(count) * (6 * 8 + 3 * 4);
  return kept + scratch + nodes;
}

/** A leaf forms at LEAF_INSTANCES or fewer, and a split halves: at most ceil(n/2) leaves. */
function topNodeCapacity(count: number): number {
  return Math.max(4, 2 * Math.ceil(Math.max(1, count) / 2) + 1);
}

/**
 * Builds one hierarchy per prototype and one over the placements.
 *
 * Every build in the pass draws on ONE guard: one byte ledger and one poll
 * counter. Each prototype's finished structure stays charged for as long as it
 * lives, so the next build -- and the top level after them -- is priced
 * against what is actually still held, not against a fresh allowance. A
 * refusal or a cancellation anywhere returns a structure that indexes
 * nothing, which finds no blockers and therefore admits more.
 */
export function buildInstancedOccluders(
  prototypes: readonly (readonly ShadoWorldPrimitive[])[],
  instances: readonly OccluderInstanceSource[],
  limits: OccluderBuildLimits = {}
): InstancedOccluders {
  const counters = {
    segmentQueries: 0,
    blockedQueries: 0,
    columnQueries: 0,
    nodeVisits: 0,
    triangleTests: 0,
    instanceVisits: 0,
  };
  const guard = limits.guard ?? new OccluderBuildGuard(limits);
  let topReserved = 0;
  try {
    // On entry, even with nothing to build: a pass already stopped stays stopped.
    guard.poll();
    const built: OccluderBvh[] = [];
    let uniqueTriangles = 0;
    for (const primitives of prototypes) {
      const bvh = buildOccluderBvh(primitives, { ...limits, guard });
      if (bvh.aborted) return emptyInstanced(counters, bvh.aborted);
      built.push(bvh);
      uniqueTriangles += bvh.triangleCount;
    }

    const count = instances.length;
    /*
     * Validated before anything is derived from them. A bad prototype index
     * or a transform that cannot be inverted is corrupt input, and building a
     * root box out of it produces bounds that block the wrong space.
     */
    for (let index = 0; index < count; index += 1) {
      guard.tick(1);
      const instance = instances[index]!;
      if (!Number.isInteger(instance.prototype) || instance.prototype < 0 || instance.prototype >= built.length) {
        throw new RangeError(`Occluder instance ${index} names prototype ${instance.prototype}, which does not exist`);
      }
      const matrix = instance.matrix;
      if (matrix.length < 16 || !matrix.slice(0, 16).every(Number.isFinite)) {
        throw new RangeError(`Occluder instance ${index} has a non-finite transform`);
      }
      const det = determinant3(matrix);
      if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
        throw new RangeError(`Occluder instance ${index} has a transform that cannot be inverted`);
      }
    }

    // Priced before a single per-instance array exists.
    topReserved = estimateInstancedTopBytes(count);
    guard.reserve(topReserved);

    const instancePrototype = new Int32Array(count);
    const instanceInverse = new Float64Array(count * 16);
    const instanceBounds = new Float64Array(count * 6);
    const instanceMirrored = new Uint8Array(count);
    /*
     * An instance of an EMPTY prototype indexes no geometry and has no box to
     * bound. It stays in the arrays, so instance ids keep their meaning, and
     * is left out of the top level -- an inverted "empty" box passes the slab
     * test from every direction, which is the wrong failure.
     */
    const instanceValid = new Uint8Array(count);
    let placedTriangles = 0;
    let validCount = 0;
    for (let index = 0; index < count; index += 1) {
      guard.tick(8);
      const instance = instances[index]!;
      const prototype = built[instance.prototype]!;
      instancePrototype[index] = instance.prototype;
      if (prototype.triangleCount === 0) continue;
      instanceValid[index] = 1;
      validCount += 1;
      placedTriangles += prototype.triangleCount;
      const matrix = instance.matrix;
      instanceMirrored[index] = determinant3(matrix) < 0 ? 1 : 0;
      invertAffine(matrix, instanceInverse, index * 16);
      // Every corner, because a rotation turns a box into something no single
      // corner bounds.
      const root = prototype.nodeBounds;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let corner = 0; corner < 8; corner += 1) {
        const x = corner & 1 ? root[3]! : root[0]!;
        const y = corner & 2 ? root[4]! : root[1]!;
        const z = corner & 4 ? root[5]! : root[2]!;
        const wx = matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!;
        const wy = matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!;
        const wz = matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!;
        if (wx < minX) minX = wx; if (wx > maxX) maxX = wx;
        if (wy < minY) minY = wy; if (wy > maxY) maxY = wy;
        if (wz < minZ) minZ = wz; if (wz > maxZ) maxZ = wz;
      }
      instanceBounds[index * 6] = minX;
      instanceBounds[index * 6 + 1] = minY;
      instanceBounds[index * 6 + 2] = minZ;
      instanceBounds[index * 6 + 3] = maxX;
      instanceBounds[index * 6 + 4] = maxY;
      instanceBounds[index * 6 + 5] = maxZ;
    }

    // Top level over the valid instances only, partitioned in place.
    const order = new Int32Array(validCount);
    const centroids = new Float64Array(count * 3);
    let written = 0;
    for (let index = 0; index < count; index += 1) {
      guard.tick(1);
      if (!instanceValid[index]) continue;
      order[written++] = index;
      for (let axis = 0; axis < 3; axis += 1) {
        centroids[index * 3 + axis] =
          (instanceBounds[index * 6 + axis]! + instanceBounds[index * 6 + 3 + axis]!) / 2;
      }
    }
    const capacity = topNodeCapacity(validCount);
    const nodeBounds = new Float64Array(capacity * 6);
    const nodeMeta = new Int32Array(capacity * 3);
    let nodeCount = 0;
    const build = (start: number, span: number): number => {
      guard.tick(1);
      if (nodeCount >= capacity) {
        throw new RangeError(`Occluder top level exceeded its ${capacity}-node bound at ${validCount} instances`);
      }
      const node = nodeCount++;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let index = start; index < start + span; index += 1) {
        guard.tick(1);
        const base = order[index]! * 6;
        if (instanceBounds[base]! < minX) minX = instanceBounds[base]!;
        if (instanceBounds[base + 1]! < minY) minY = instanceBounds[base + 1]!;
        if (instanceBounds[base + 2]! < minZ) minZ = instanceBounds[base + 2]!;
        if (instanceBounds[base + 3]! > maxX) maxX = instanceBounds[base + 3]!;
        if (instanceBounds[base + 4]! > maxY) maxY = instanceBounds[base + 4]!;
        if (instanceBounds[base + 5]! > maxZ) maxZ = instanceBounds[base + 5]!;
      }
      nodeBounds[node * 6] = minX;
      nodeBounds[node * 6 + 1] = minY;
      nodeBounds[node * 6 + 2] = minZ;
      nodeBounds[node * 6 + 3] = maxX;
      nodeBounds[node * 6 + 4] = maxY;
      nodeBounds[node * 6 + 5] = maxZ;
      if (span <= LEAF_INSTANCES) {
        nodeMeta[node * 3] = start;
        nodeMeta[node * 3 + 1] = span;
        nodeMeta[node * 3 + 2] = -1;
        return node;
      }
      const axis =
        maxX - minX >= maxY - minY && maxX - minX >= maxZ - minZ ? 0 : maxY - minY >= maxZ - minZ ? 1 : 2;
      const half = span >> 1;
      /*
       * The flat builder's bounded in-place median partition. The old
       * Array.from(...).sort(...) allocated a copy and sorted it at every
       * level, unguarded -- the root level over every placement in the zone.
       */
      selectNth(order, centroids, axis, start, start + span - 1, start + half, guard);
      nodeMeta[node * 3] = start;
      nodeMeta[node * 3 + 1] = 0;
      build(start, half);
      nodeMeta[node * 3 + 2] = build(start + half, span - half);
      return node;
    };
    if (validCount) build(0, validCount);
    else {
      nodeCount = 1;
      nodeMeta[2] = -1;
    }
    guard.poll();

    // Centroids were scratch; everything else stays with the structure.
    const scratch = count * 3 * 8;
    guard.retain(topReserved - scratch);
    guard.release(scratch);
    topReserved = 0;

    return {
      prototypes: built,
      instancePrototype,
      instanceInverse,
      instanceBounds,
      instanceMirrored,
      instanceValid,
      instanceCount: count,
      nodeBounds,
      nodeMeta,
      nodeCount,
      order,
      counters,
      aborted: null,
      uniqueTriangles,
      placedTriangles,
    };
  } catch (error) {
    guard.release(topReserved);
    if (isOccluderBuildCancelled(error)) return emptyInstanced(counters, guard.reason);
    throw error;
  }
}

/** Is the straight line from a to b interrupted by any placed prototype? */
export function instancedSegmentBlocked(
  scene: InstancedOccluders,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number
): boolean {
  scene.counters.segmentQueries += 1;
  if (!scene.instanceCount) return false;
  const dx = bx - ax, dy = by - ay, dz = bz - az;
  if (Math.hypot(dx, dy, dz) < END_EPSILON) return false;
  const invX = dx === 0 ? Infinity : 1 / dx;
  const invY = dy === 0 ? Infinity : 1 / dy;
  const invZ = dz === 0 ? Infinity : 1 / dz;
  const stack = TOP_STACK;
  let depth = 0;
  stack[depth++] = 0;
  while (depth > 0) {
    const node = stack[--depth]!;
    scene.counters.nodeVisits += 1;
    if (!slabs(scene.nodeBounds, node * 6, ax, ay, az, invX, invY, invZ)) continue;
    const span = scene.nodeMeta[node * 3 + 1]!;
    if (span === 0) {
      if (depth + 2 >= stack.length) return false;
      stack[depth++] = node + 1;
      stack[depth++] = scene.nodeMeta[node * 3 + 2]!;
      continue;
    }
    const first = scene.nodeMeta[node * 3]!;
    for (let index = first; index < first + span; index += 1) {
      const instance = scene.order[index]!;
      scene.counters.instanceVisits += 1;
      if (!slabs(scene.instanceBounds, instance * 6, ax, ay, az, invX, invY, invZ)) continue;
      const base = instance * 16;
      const inverse = scene.instanceInverse;
      /*
       * Into the prototype's own space. The parameter rides along unchanged
       * under an affine map, so the endpoint epsilon keeps its meaning and
       * nothing has to be rescaled.
       */
      const lax = inverse[base]! * ax + inverse[base + 4]! * ay + inverse[base + 8]! * az + inverse[base + 12]!;
      const lay = inverse[base + 1]! * ax + inverse[base + 5]! * ay + inverse[base + 9]! * az + inverse[base + 13]!;
      const laz = inverse[base + 2]! * ax + inverse[base + 6]! * ay + inverse[base + 10]! * az + inverse[base + 14]!;
      const lbx = inverse[base]! * bx + inverse[base + 4]! * by + inverse[base + 8]! * bz + inverse[base + 12]!;
      const lby = inverse[base + 1]! * bx + inverse[base + 5]! * by + inverse[base + 9]! * bz + inverse[base + 13]!;
      const lbz = inverse[base + 2]! * bx + inverse[base + 6]! * by + inverse[base + 10]! * bz + inverse[base + 14]!;
      const prototype = scene.prototypes[scene.instancePrototype[instance]!]!;
      if (localBlocked(prototype, lax, lay, laz, lbx, lby, lbz, scene.counters)) {
        scene.counters.blockedQueries += 1;
        return true;
      }
    }
  }
  return false;
}

/** Highest surface under a column, across every placed prototype. */
export function instancedHighestSurfaceAt(
  scene: InstancedOccluders,
  x: number,
  z: number,
  top: number,
  bottom: number
): number | null {
  scene.counters.columnQueries += 1;
  let best: number | null = null;
  for (let instance = 0; instance < scene.instanceCount; instance += 1) {
    if (!scene.instanceValid[instance]) continue;
    const base = instance * 6;
    if (x < scene.instanceBounds[base]! || x > scene.instanceBounds[base + 3]!) continue;
    if (z < scene.instanceBounds[base + 2]! || z > scene.instanceBounds[base + 5]!) continue;
    if (best !== null && scene.instanceBounds[base + 4]! <= best) continue;
    // Walk the column as a segment, in this instance's space.
    const height = columnHit(scene, instance, x, z, top, bottom);
    if (height !== null && (best === null || height > best)) best = height;
  }
  return best;
}

const TOP_STACK = new Int32Array(256);
const LOCAL_STACK = new Int32Array(256);

function columnHit(
  scene: InstancedOccluders,
  instance: number,
  x: number,
  z: number,
  top: number,
  bottom: number
): number | null {
  // Bisect the vertical segment against this instance: a blocked upper half
  // means the surface is above the midpoint.
  let high = top;
  let low = bottom;
  if (!instancedSegmentBlockedOne(scene, instance, x, high, z, x, low, z)) return null;
  for (let step = 0; step < 24; step += 1) {
    const mid = (high + low) / 2;
    if (instancedSegmentBlockedOne(scene, instance, x, high, z, x, mid, z)) low = mid;
    else high = mid;
  }
  return (high + low) / 2;
}

function instancedSegmentBlockedOne(
  scene: InstancedOccluders,
  instance: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number
): boolean {
  const base = instance * 16;
  const inverse = scene.instanceInverse;
  const lax = inverse[base]! * ax + inverse[base + 4]! * ay + inverse[base + 8]! * az + inverse[base + 12]!;
  const lay = inverse[base + 1]! * ax + inverse[base + 5]! * ay + inverse[base + 9]! * az + inverse[base + 13]!;
  const laz = inverse[base + 2]! * ax + inverse[base + 6]! * ay + inverse[base + 10]! * az + inverse[base + 14]!;
  const lbx = inverse[base]! * bx + inverse[base + 4]! * by + inverse[base + 8]! * bz + inverse[base + 12]!;
  const lby = inverse[base + 1]! * bx + inverse[base + 5]! * by + inverse[base + 9]! * bz + inverse[base + 13]!;
  const lbz = inverse[base + 2]! * bx + inverse[base + 6]! * by + inverse[base + 10]! * bz + inverse[base + 14]!;
  const prototype = scene.prototypes[scene.instancePrototype[instance]!]!;
  return localBlocked(prototype, lax, lay, laz, lbx, lby, lbz, scene.counters);
}

/** The prototype hierarchy's own walk, with the facing test the instance needs. */
function localBlocked(
  bvh: OccluderBvh,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  counters: InstancedOccluders['counters']
): boolean {
  const dx = bx - ax, dy = by - ay, dz = bz - az;
  if (Math.hypot(dx, dy, dz) < END_EPSILON) return false;
  const invX = dx === 0 ? Infinity : 1 / dx;
  const invY = dy === 0 ? Infinity : 1 / dy;
  const invZ = dz === 0 ? Infinity : 1 / dz;
  const stack = LOCAL_STACK;
  let depth = 0;
  stack[depth++] = 0;
  while (depth > 0) {
    const node = stack[--depth]!;
    counters.nodeVisits += 1;
    if (!slabs(bvh.nodeBounds, node * 6, ax, ay, az, invX, invY, invZ)) continue;
    const span = bvh.nodeMeta[node * 3 + 1]!;
    if (span === 0) {
      if (depth + 2 >= stack.length) return false;
      stack[depth++] = node + 1;
      stack[depth++] = bvh.nodeMeta[node * 3 + 2]!;
      continue;
    }
    const first = bvh.nodeMeta[node * 3]!;
    for (let index = first; index < first + span; index += 1) {
      counters.triangleTests += 1;
      if (triangleBlocks(bvh, index, ax, ay, az, dx, dy, dz)) return true;
    }
  }
  return false;
}

function triangleBlocks(
  bvh: OccluderBvh,
  triangle: number,
  ax: number, ay: number, az: number,
  dx: number, dy: number, dz: number
): boolean {
  const t = bvh.triangles;
  const o = triangle * 9;
  const e1x = t[o + 3]! - t[o]!, e1y = t[o + 4]! - t[o + 1]!, e1z = t[o + 5]! - t[o + 2]!;
  const e2x = t[o + 6]! - t[o]!, e2y = t[o + 7]! - t[o + 1]!, e2z = t[o + 8]! - t[o + 2]!;
  const hx = dy * e2z - dz * e2y;
  const hy = dz * e2x - dx * e2z;
  const hz = dx * e2y - dy * e2x;
  const det = e1x * hx + e1y * hy + e1z * hz;
  if (det > -1e-12 && det < 1e-12) return false;
  /*
   * No mirror correction here, and that is deliberate.
   *
   * The segment is tested in the prototype's own space, and the inverse
   * transform that put it there has already reversed the direction a mirrored
   * placement reverses. Flipping the facing test as well corrects the same
   * thing twice, which makes a mirrored single-sided surface block from the
   * side you can see through -- exactly the bug sidedness exists to prevent.
   */
  if (bvh.doubleSided[triangle] === 0 && det < 0) return false;
  const inv = 1 / det;
  const sx = ax - t[o]!, sy = ay - t[o + 1]!, sz = az - t[o + 2]!;
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

function slabs(
  bounds: Float64Array,
  base: number,
  ax: number, ay: number, az: number,
  invX: number, invY: number, invZ: number
): boolean {
  let near = 0;
  let far = 1;
  for (let axis = 0; axis < 3; axis += 1) {
    const origin = axis === 0 ? ax : axis === 1 ? ay : az;
    const inverse = axis === 0 ? invX : axis === 1 ? invY : invZ;
    const min = bounds[base + axis]!;
    const max = bounds[base + 3 + axis]!;
    if (!Number.isFinite(inverse)) {
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

function determinant3(m: readonly number[]): number {
  return (
    m[0]! * (m[5]! * m[10]! - m[6]! * m[9]!) -
    m[4]! * (m[1]! * m[10]! - m[2]! * m[9]!) +
    m[8]! * (m[1]! * m[6]! - m[2]! * m[5]!)
  );
}

/** Inverse of an affine column-major transform, written into `out` at `at`. */
function invertAffine(m: readonly number[], out: Float64Array, at: number): void {
  const a = m[0]!, b = m[4]!, c = m[8]!;
  const d = m[1]!, e = m[5]!, f = m[9]!;
  const g = m[2]!, h = m[6]!, i = m[10]!;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const scale = det === 0 ? 0 : 1 / det;
  const r0 = (e * i - f * h) * scale, r1 = (c * h - b * i) * scale, r2 = (b * f - c * e) * scale;
  const r3 = (f * g - d * i) * scale, r4 = (a * i - c * g) * scale, r5 = (c * d - a * f) * scale;
  const r6 = (d * h - e * g) * scale, r7 = (b * g - a * h) * scale, r8 = (a * e - b * d) * scale;
  const tx = m[12]!, ty = m[13]!, tz = m[14]!;
  out[at] = r0; out[at + 1] = r3; out[at + 2] = r6; out[at + 3] = 0;
  out[at + 4] = r1; out[at + 5] = r4; out[at + 6] = r7; out[at + 7] = 0;
  out[at + 8] = r2; out[at + 9] = r5; out[at + 10] = r8; out[at + 11] = 0;
  out[at + 12] = -(r0 * tx + r1 * ty + r2 * tz);
  out[at + 13] = -(r3 * tx + r4 * ty + r5 * tz);
  out[at + 14] = -(r6 * tx + r7 * ty + r8 * tz);
  out[at + 15] = 1;
}

function emptyInstanced(
  counters: InstancedOccluders['counters'],
  aborted: OccluderBvh['aborted']
): InstancedOccluders {
  const nodeMeta = new Int32Array(3);
  nodeMeta[2] = -1;
  return {
    prototypes: [],
    instancePrototype: new Int32Array(0),
    instanceInverse: new Float64Array(0),
    instanceBounds: new Float64Array(0),
    instanceMirrored: new Uint8Array(0),
    instanceValid: new Uint8Array(0),
    instanceCount: 0,
    nodeBounds: new Float64Array(6),
    nodeMeta,
    nodeCount: 1,
    order: new Int32Array(0),
    counters,
    aborted,
    uniqueTriangles: 0,
    placedTriangles: 0,
  };
}
