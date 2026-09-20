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
 * Regions touched by an XZ box, and whether the list had to be truncated.
 *
 * `overflow` is never a dropped entity. A caller that sees it treats the thing
 * as an always-candidate and falls back to the tests that do not need
 * membership, which admits more and hides nothing.
 */
export type RegionMembership = {
  readonly regions: Uint32Array;
  readonly overflow: boolean;
};

/**
 * Boundary contact counts as membership.
 *
 * A bound that ends exactly on a region edge touches the region beyond it, and
 * floating-point positions land on edges more often than intuition suggests --
 * stamps are placed on round numbers and regions are powers of two. The
 * epsilon widens the box rather than narrowing it, so the error is towards
 * more regions.
 */
const BOUNDARY_EPSILON = 1e-4;

/** Ceiling on stored regions per entity; beyond it the caller uses overflow. */
export const MAX_REGIONS_PER_ENTITY = 64;

/**
 * Enumerates every region an XZ box intersects, clipped to the grid.
 *
 * The box is half-open in neither direction: a box touching the boundary
 * between two regions is in both. Out-of-grid extents are clipped rather than
 * wrapped, and a box entirely outside the grid returns nothing -- which the
 * caller must read as "unknown", not "hidden".
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
  if (!Number.isFinite(minX) || !Number.isFinite(minZ) || !Number.isFinite(maxX) || !Number.isFinite(maxZ)) {
    return { regions: into.subarray(0, 0), overflow: true };
  }
  const firstX = Math.floor((minX - BOUNDARY_EPSILON - grid.originX) / grid.size);
  const lastX = Math.floor((maxX + BOUNDARY_EPSILON - grid.originX) / grid.size);
  const firstZ = Math.floor((minZ - BOUNDARY_EPSILON - grid.originZ) / grid.size);
  const lastZ = Math.floor((maxZ + BOUNDARY_EPSILON - grid.originZ) / grid.size);
  const clampedFirstX = Math.max(0, firstX);
  const clampedLastX = Math.min(grid.width - 1, lastX);
  const clampedFirstZ = Math.max(0, firstZ);
  const clampedLastZ = Math.min(grid.height - 1, lastZ);
  if (clampedFirstX > clampedLastX || clampedFirstZ > clampedLastZ) {
    return { regions: into.subarray(0, 0), overflow: false };
  }
  /*
   * Reaching past the grid edge is not overflow. There are no rows out there
   * and no camera either -- a viewpoint outside the grid has no region to
   * look from -- so the regions that do exist describe everything that can
   * see this entity. Treating the overhang as unknown would make every
   * entity along a zone's border permanently visible.
   */
  const span = (clampedLastX - clampedFirstX + 1) * (clampedLastZ - clampedFirstZ + 1);
  if (span > cap || span > into.length) {
    return { regions: into.subarray(0, 0), overflow: true };
  }
  let written = 0;
  for (let z = clampedFirstZ; z <= clampedLastZ; z += 1) {
    const row = z * grid.width;
    for (let x = clampedFirstX; x <= clampedLastX; x += 1) {
      into[written++] = row + x;
    }
  }
  return { regions: into.subarray(0, written), overflow: false };
}

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
