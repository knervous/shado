import { buildOccluderGrid, highestSurfaceAt, segmentBlocked } from '../src/world/occlusion';
import type { ShadoWorldPrimitive } from '../src/world/types';

/** An axis-aligned quad, as two triangles, facing however you like. */
function quad(
  corners: [number, number, number][],
  name = 'wall',
): ShadoWorldPrimitive {
  const positions: number[] = [];
  for (const [x, y, z] of corners) positions.push(x, y, z);
  return {
    name,
    material: 'test',
    positions,
    indices: [0, 1, 2, 0, 2, 3],
  } as ShadoWorldPrimitive;
}

const bounds = { min: [-200, -100, -200] as [number, number, number], max: [200, 100, 200] as [number, number, number] };

/** A wall in the x=0 plane, 40 wide and 40 tall, centred on the origin. */
const wall = quad([
  [0, -20, -20],
  [0, -20, 20],
  [0, 20, 20],
  [0, 20, -20],
]);

/** A floor at y=0 spanning the test area. */
const floor = quad(
  [
    [-150, 0, -150],
    [150, 0, -150],
    [150, 0, 150],
    [-150, 0, 150],
  ],
  'floor',
);

describe('segment occlusion', () => {
  const grid = buildOccluderGrid([wall], bounds, 32);

  it('blocks a segment that crosses the wall', () => {
    expect(segmentBlocked(grid, -50, 0, 0, 50, 0, 0)).toBe(true);
  });

  it('lets a segment past the wall edge through', () => {
    // z = 60 is outside the wall's 40-unit span.
    expect(segmentBlocked(grid, -50, 0, 60, 50, 0, 60)).toBe(false);
  });

  it('lets a segment over the top through', () => {
    expect(segmentBlocked(grid, -50, 50, 0, 50, 50, 0)).toBe(false);
  });

  it('does not block a segment that stops short of the wall', () => {
    expect(segmentBlocked(grid, -50, 0, 0, -10, 0, 0)).toBe(false);
  });

  it('does not block a segment that starts past the wall', () => {
    expect(segmentBlocked(grid, 10, 0, 0, 50, 0, 0)).toBe(false);
  });

  /*
   * The property that makes a PVS safe: a thin wall standing between two grid
   * steps must still block. An earlier version sampled along the ray at a fixed
   * interval and could step straight over geometry; a DDA visits every cell the
   * segment touches, so this holds at any angle.
   */
  it('blocks at every angle, never stepping over the wall', () => {
    for (let degrees = -80; degrees <= 80; degrees += 5) {
      const radians = (degrees * Math.PI) / 180;
      const z = Math.tan(radians) * 50;
      if (Math.abs(z) > 18) continue; // still inside the wall's span
      expect(segmentBlocked(grid, -50, 0, -z, 50, 0, z)).toBe(true);
    }
  });

  it('is symmetric: what blocks one way blocks the other', () => {
    for (let z = -18; z <= 18; z += 3) {
      const forward = segmentBlocked(grid, -40, 0, z, 40, 0, z);
      const backward = segmentBlocked(grid, 40, 0, z, -40, 0, z);
      expect(forward).toBe(backward);
    }
  });

  it('never reports a hit for a degenerate segment', () => {
    expect(segmentBlocked(grid, 5, 5, 5, 5, 5, 5)).toBe(false);
  });

  it('does not let the surface a viewpoint stands on occlude it', () => {
    const floored = buildOccluderGrid([floor], bounds, 32);
    // Standing exactly on the floor, looking along it: the floor is the
    // endpoint's own surface and must not count.
    expect(segmentBlocked(floored, -100, 0, 0, 100, 0, 0)).toBe(false);
    // And a viewpoint above it can still see along.
    expect(segmentBlocked(floored, -100, 10, 0, 100, 10, 0)).toBe(false);
  });

  it('blocks through the floor, so nothing sees into the ground', () => {
    const floored = buildOccluderGrid([floor], bounds, 32);
    expect(segmentBlocked(floored, 0, 40, 0, 0, -40, 0)).toBe(true);
  });
});

describe('highest surface', () => {
  it('finds the floor under a column', () => {
    const grid = buildOccluderGrid([floor], bounds, 32);
    expect(highestSurfaceAt(grid, 10, 10, bounds)).toBeCloseTo(0, 3);
  });

  it('returns null where there is nothing to stand on', () => {
    const grid = buildOccluderGrid([wall], bounds, 32);
    expect(highestSurfaceAt(grid, 120, 120, bounds)).toBeNull();
  });

  /*
   * A floor whose triangles are NOT axis aligned. The downward-ray
   * intersection was written out by hand and had two terms of the cross
   * product transposed, which still produces a usable determinant on an
   * axis-aligned quad -- so the square-floor tests above passed while most of
   * a real zone's ground went undetected.
   */
  it('finds a floor whose triangles are not axis aligned', () => {
    const skew = quad(
      [
        [-100, 5, -40],
        [60, 5, -110],
        [110, 5, 30],
        [-30, 5, 120],
      ],
      'skew-floor',
    );
    const grid = buildOccluderGrid([skew], bounds, 32);
    expect(highestSurfaceAt(grid, 0, 0, bounds)).toBeCloseTo(5, 3);
    expect(highestSurfaceAt(grid, 20, -20, bounds)).toBeCloseTo(5, 3);
  });

  it('takes the highest of stacked surfaces', () => {
    const upper = quad(
      [
        [-50, 30, -50],
        [50, 30, -50],
        [50, 30, 50],
        [-50, 30, 50],
      ],
      'roof',
    );
    const grid = buildOccluderGrid([floor, upper], bounds, 32);
    expect(highestSurfaceAt(grid, 0, 0, bounds)).toBeCloseTo(30, 3);
    expect(highestSurfaceAt(grid, 100, 100, bounds)).toBeCloseTo(0, 3);
  });
});
