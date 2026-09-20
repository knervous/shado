/**
 * Which visibility regions a thing occupies.
 *
 * ## Why one function
 *
 * Membership was the entity path's centre point: an entity belonged to the one
 * region its origin fell in, while the frustum and distance tests used its
 * radius. That is not conservative, and it is not a small error. On Crownward
 * **79.6% of placed stamps have a bounding sphere crossing their region
 * boundary** -- median radius 30 units against 64-unit regions -- so four in
 * five were admitted or rejected by a region holding a minority of their
 * volume. Under a distance flood that was invisible, because the neighbours
 * were admitted anyway. Under real occlusion rows it is a hole: a building
 * whose origin sits in a rejected region while its body reaches into an
 * admitted one simply disappears.
 *
 * Every consumer now asks this one function, so the worker, the synchronous
 * path and the reference oracle cannot drift, and a later 3D source topology
 * has exactly one place to change.
 */

/** The region grid a package defines, in world units. */
export type RegionGrid = {
  readonly originX: number;
  readonly originZ: number;
  readonly size: number;
  readonly width: number;
  readonly height: number;
};

/**
 * Regions touched by an XZ box, and which of three states the box is in.
 *
 * The three are not interchangeable and collapsing any two of them loses an
 * entity:
 *
 * - `enumerated` -- every region the box touches is listed. The box lies
 *   wholly inside the supported grid.
 * - `unknown` -- the box could not be enumerated: its bounds are not finite,
 *   it touches more regions than the cap holds, or it reaches past the edge
 *   of the supported grid. Nothing here has been proved about what can see
 *   it, so it is an always-candidate: topology admits it and the frustum,
 *   range, phase and enabled tests still decide.
 * - `whollyOutside` -- valid bounds, entirely beyond the grid. This is the
 *   only state `outsideWorldVisible` governs, because it is the only one
 *   where the box has been PROVED to be somewhere the rows do not describe.
 */
export type RegionMembershipState = 'enumerated' | 'unknown' | 'whollyOutside';

export type RegionMembership = {
  readonly regions: Uint32Array;
  readonly state: RegionMembershipState;
  /** True for `unknown`: kept so callers reading one field cannot under-admit. */
  readonly overflow: boolean;
};


/** Ceiling on stored regions per entity; beyond it the caller uses overflow. */
export const MAX_REGIONS_PER_ENTITY = 64;

/*
 * The classifier itself, as ONE self-contained function.
 *
 * It closes over nothing -- no imports, no module constants, not even the
 * epsilon -- because the entity worker runs as a standalone source string and
 * embeds this function's own text. Two hand-maintained copies of this
 * arithmetic is exactly how the worker came to answer differently from the
 * synchronous reducer for a >64-region actor, so there is one copy and the
 * worker gets it by construction rather than by review.
 *
 * Returns the number of dense region indices written to `into`, or a negative
 * code: -1 unknown, -2 wholly outside. Allocation-free on purpose; the worker
 * calls it per entity per frame.
 */
export function classifyRegionMembership(
  originX: number,
  originZ: number,
  size: number,
  minTileX: number,
  minTileZ: number,
  width: number,
  height: number,
  minX: number,
  minZ: number,
  maxX: number,
  maxZ: number,
  cap: number,
  into: Uint32Array
): number {
  if (!(size > 0) || !(width > 0) || !(height > 0)) return -1;
  if (
    !Number.isFinite(minX) || !Number.isFinite(minZ) ||
    !Number.isFinite(maxX) || !Number.isFinite(maxZ)
  ) {
    return -1;
  }
  /*
   * Boundary contact counts as membership. A bound ending exactly on a region
   * edge touches the region beyond it, and positions land on edges more often
   * than intuition suggests -- stamps sit on round numbers and regions are
   * powers of two. The epsilon widens the box, so the error is towards more
   * regions.
   */
  const epsilon = 1e-4;
  /*
   * Whether the box lies in the supported domain is asked of the box itself,
   * in world units, and NOT of the epsilon-widened one. The epsilon exists so
   * a bound resting on an interior region edge counts as touching both sides;
   * at the domain edge it would instead manufacture an overhang out of
   * nothing, and every actor standing against the first region would be
   * unknown because its bound began exactly at the origin.
   */
  const lowX = originX + minTileX * size;
  const lowZ = originZ + minTileZ * size;
  const highX = lowX + width * size;
  const highZ = lowZ + height * size;
  if (maxX < lowX - epsilon || maxZ < lowZ - epsilon || minX > highX + epsilon || minZ > highZ + epsilon) {
    // Valid bounds, no part of them inside the domain: proved outside.
    return -2;
  }
  if (minX < lowX - epsilon || minZ < lowZ - epsilon || maxX > highX + epsilon || maxZ > highZ + epsilon) {
    /*
     * Straddling the edge. Clipping to the grid and enumerating what is left
     * would claim the remaining regions describe everything that can see this
     * box, which is a claim about where a camera can be -- and third-person
     * offsets, debug flight and boundary crossings all put one outside. So it
     * is unknown until a content/camera contract says otherwise.
     */
    return -1;
  }
  const firstX = Math.max(0, Math.floor((minX - epsilon - originX) / size) - minTileX);
  const lastX = Math.min(width - 1, Math.floor((maxX + epsilon - originX) / size) - minTileX);
  const firstZ = Math.max(0, Math.floor((minZ - epsilon - originZ) / size) - minTileZ);
  const lastZ = Math.min(height - 1, Math.floor((maxZ + epsilon - originZ) / size) - minTileZ);
  if (firstX > lastX || firstZ > lastZ) return -1;
  const span = (lastX - firstX + 1) * (lastZ - firstZ + 1);
  if (span > cap || span > into.length) return -1;
  let written = 0;
  for (let z = firstZ; z <= lastZ; z += 1) {
    const row = z * width;
    for (let x = firstX; x <= lastX; x += 1) {
      into[written++] = row + x;
    }
  }
  return written;
}

/**
 * Enumerates every region an XZ box intersects, as a package-space wrapper
 * around {@link classifyRegionMembership}.
 */
export function regionsForBounds(
  grid: RegionGrid,
  minX: number,
  minZ: number,
  maxX: number,
  maxZ: number,
  into: Uint32Array,
  cap = MAX_REGIONS_PER_ENTITY
): RegionMembership {
  const written = classifyRegionMembership(
    grid.originX, grid.originZ, grid.size,
    0, 0, grid.width, grid.height,
    minX, minZ, maxX, maxZ,
    cap, into
  );
  if (written === -2) {
    return { regions: into.subarray(0, 0), state: 'whollyOutside', overflow: false };
  }
  if (written < 0) {
    return { regions: into.subarray(0, 0), state: 'unknown', overflow: true };
  }
  return { regions: into.subarray(0, written), state: 'enumerated', overflow: false };
}

/**
 * The classifier's own source, for the entity worker to embed.
 *
 * `toString()` rather than a duplicated string literal: whatever the bundler
 * did to the function is what the worker runs, and the two cannot drift
 * because there is only one of them.
 */
export const REGION_MEMBERSHIP_SOURCE = `const classifyRegionMembership = ${classifyRegionMembership.toString()};`;

/**
 * Is any single region in the membership admitted on its own?
 *
 * Every required bit must be satisfied by ONE region. Or-ing bits across
 * regions would let a loaded-but-hidden region and a visible-but-unloaded one
 * combine into a pass that neither of them is.
 */
export function anyRegionAdmits(
  regions: ArrayLike<number>,
  regionFlags: ArrayLike<number>,
  requiredBits: number
): boolean {
  for (let index = 0; index < regions.length; index += 1) {
    const region = regions[index]!;
    if (region >= regionFlags.length) continue;
    if ((regionFlags[region]! & requiredBits) === requiredBits) return true;
  }
  return false;
}
