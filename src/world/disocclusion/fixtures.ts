/**
 * Deterministic synthetic scenes for the disocclusion prototype. The same
 * data feeds the offline bake and the browser proving ground; neither builds
 * its own copy.
 *
 * Units are zone units (3 per metre). Y is up.
 */
import { compileShadoWorld } from '../compiler';
import type { ShadoWorldPrimitive, ShadoWorldSpatialPackage } from '../types';
import type { DisocclusionAxis, DisocclusionCapture, Vec3 } from './types';

export const DISOCCLUSION_AXES: readonly DisocclusionAxis[] = ['+x', '-x', '+y', '-y', '+z', '-z'];

/**
 * A source volume as six axis captures that together cover every direction
 * from anywhere in the box; admission unions the faces a frustum can touch.
 *
 * Not a cube map. Side faces (+-x, +-z) lean 1 across but only `sideUp` up,
 * so their vertical tiles are finer: that is the axis a floor/wall junction
 * leaks along. The +-y faces widen to 1/sideUp on both axes to take every
 * steeper direction: a direction within `sideUp` of horizontal belongs to the
 * side face of its larger horizontal component; any steeper one has
 * |horizontal / vertical| < 1/sideUp.
 */
export function sourceVolumeCaptures(
  volume: string,
  sourceMin: Vec3,
  sourceMax: Vec3,
  range: { near: number; far: number },
  sideUp = 0.6
): DisocclusionCapture[] {
  return DISOCCLUSION_AXES.map(axis => {
    const vertical = axis === '+y' || axis === '-y';
    return {
      volume,
      sourceMin: [...sourceMin] as Vec3,
      sourceMax: [...sourceMax] as Vec3,
      axis,
      directionTan: vertical ? 1 / sideUp : 1,
      directionTanUp: vertical ? 1 / sideUp : sideUp,
      near: range.near,
      far: range.far,
    };
  });
}

export type DisocclusionRouteKey = { t: number; at: Vec3; look: Vec3 };

export type DisocclusionFixture = {
  name: string;
  primitives: ShadoWorldPrimitive[];
  /** Named targets the tests and the page talk about, with their world bounds. */
  targets: Record<string, { min: Vec3; max: Vec3; expect: 'visible' | 'hidden' }>;
  captures: DisocclusionCapture[];
  /** Compile options that keep regions small enough to separate the rooms. */
  compile: { tileSize: number; visibilityRegionSize: number; visibilityMaxDistance: number };
  route: DisocclusionRouteKey[];
  /** A free-fly starting pose. */
  start: { at: Vec3; look: Vec3 };
};

/** Largest quad edge on a fixture face. Real zone meshes are this fine or finer; one huge triangle per face would make every cluster span a whole room. */
const FACE_STEP = 2;

function box(name: string, material: string, min: Vec3, max: Vec3): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  // Each face is a grid of outward-wound quads no wider than FACE_STEP.
  const face = (origin: Vec3, u: Vec3, v: Vec3) => {
    const lu = Math.hypot(...u);
    const lv = Math.hypot(...v);
    const nu = Math.max(1, Math.ceil(lu / FACE_STEP));
    const nv = Math.max(1, Math.ceil(lv / FACE_STEP));
    const base = positions.length / 3;
    for (let j = 0; j <= nv; j++) {
      for (let i = 0; i <= nu; i++) {
        for (let a = 0; a < 3; a++) positions.push(origin[a]! + (u[a]! * i) / nu + (v[a]! * j) / nv);
      }
    }
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        const k = base + j * (nu + 1) + i;
        indices.push(k, k + 1, k + nu + 2, k, k + nu + 2, k + nu + 1);
      }
    }
  };
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const dz = z1 - z0;
  face([x0, y0, z0], [0, dy, 0], [dx, 0, 0]); // -z
  face([x0, y0, z1], [dx, 0, 0], [0, dy, 0]); // +z
  face([x0, y0, z0], [0, 0, dz], [0, dy, 0]); // -x
  face([x1, y0, z0], [0, dy, 0], [0, 0, dz]); // +x
  face([x0, y0, z0], [dx, 0, 0], [0, 0, dz]); // -y
  face([x0, y1, z0], [0, 0, dz], [dx, 0, 0]); // +y
  return { name, material, positions, indices, doubleSided: false };
}

const WALL_H = 24;

/**
 * Three rooms along +X. The source room A (x 0..24) looks east. Its east wall
 * has a doorway into room B (south); north of the doorway the wall is solid,
 * and room C behind it is sealed from both A and B. One pillar stands in B,
 * in line with the doorway; its twin stands in C, behind the solid wall.
 *
 *        z=32 +---------+-------------------------------+
 *             |         |   C  (sealed)     [sealed]    |
 *             |    A    |                               |
 *        z=16 |   src   +===============================+  <- B/C wall z 14..16
 *             |    *    :                               |
 *             |         :        [door]     B           |
 *        z=0  +---------+-------------------------------+
 *            x=0      x=24                            x=64
 *                doorway z 7..13 at x 22..24, lintel from y 10
 *
 * Walls are 24 high (8 m) so they fill whole tiles of the 128^2 capture; a
 * wall that only half-fills its tiles leaves them open and hides nothing.
 */
export function twoRoomFixture(): DisocclusionFixture {
  const p: ShadoWorldPrimitive[] = [];
  // Floor quads land on even coordinates so none straddles a wall line: a
  // cluster whose bounds cross a wall is admitted from the visible side.
  // (It stops at 66/34, not 64/32: geometry ending exactly on a region
  // boundary compiles to region -1; see the evidence doc.)
  p.push(box('floor', 'floor', [0, -1, 0], [66, 0, 34]));
  // Room A.
  p.push(box('a-west', 'stone', [0, 0, 0], [1, WALL_H, 32]));
  p.push(box('a-south', 'stone', [0, 0, 0], [24, WALL_H, 1]));
  p.push(box('a-north', 'stone', [0, 0, 31], [24, WALL_H, 32]));
  // East wall: solid north of the doorway, jamb south of it, lintel above.
  // The interior walls are 2 units (0.67 m) thick: thicker than the
  // volumetric filter's 1.5-unit grid, which admits across anything thinner.
  p.push(box('a-east-solid', 'stone', [22, 0, 13], [24, WALL_H, 32]));
  p.push(box('a-east-jamb', 'stone', [22, 0, 0], [24, WALL_H, 7]));
  p.push(box('a-east-lintel', 'stone', [22, 10, 7], [24, WALL_H, 13]));
  // Room B (south) and C (north), divided by a solid wall.
  p.push(box('bc-divide', 'stone', [24, 0, 14], [64, WALL_H, 16]));
  p.push(box('b-south', 'stone', [24, 0, 0], [64, WALL_H, 1]));
  p.push(box('bc-east', 'stone', [63, 0, 0], [64, WALL_H, 32]));
  p.push(box('c-north', 'stone', [24, 0, 31], [64, WALL_H, 32]));
  // Targets.
  p.push(box('target-door', 'target-door', [39, 0, 3], [41, 6, 5]));
  // Two units past the 16..24 region row along the B/C wall: the floor under
  // that wall is hidden but tile-visible in a deeper layer, so the volumetric
  // filter admits the room-C strip beside it (the paper's filter, kept per
  // pvs.md R3). The target stands in the interior beyond that strip.
  p.push(box('target-sealed', 'target-sealed', [55, 0, 25], [57, 6, 27]));
  return {
    name: 'two-room',
    primitives: p,
    targets: {
      'target-door': { min: [39, 0, 3], max: [41, 6, 5], expect: 'visible' },
      'target-sealed': { min: [55, 0, 25], max: [57, 6, 27], expect: 'hidden' },
    },
    // 0.75-unit half extent: the directive's starting size. Six faces, so a
    // camera anywhere in the box may look in any direction.
    captures: sourceVolumeCaptures('source', [7.25, 4.25, 15.25], [8.75, 5.75, 16.75], { near: 8, far: 256 }),
    compile: { tileSize: 8, visibilityRegionSize: 8, visibilityMaxDistance: 256 },
    route: [
      { t: 0, at: [8, 5, 16], look: [40, 5, 10] },
      { t: 2, at: [8, 5, 15.4], look: [40, 5, 8] },
      { t: 4, at: [8.6, 5.4, 16.6], look: [40, 5, 20] },
      { t: 6, at: [8, 5, 16], look: [8, 5, 48] },
      { t: 7.5, at: [8, 5, 16], look: [-24, 5, 16] },
      { t: 8.5, at: [12, 5, 16], look: [40, 5, 10] },
      { t: 10, at: [8, 5, 16], look: [40, 5, 10] },
    ],
    start: { at: [8, 5, 16], look: [40, 5, 10] },
  };
}

export const DISOCCLUSION_FIXTURES: Record<string, () => DisocclusionFixture> = {
  'two-room': twoRoomFixture,
};

/**
 * The one way a fixture becomes a world. Bake, tests and the browser all call
 * this, so their layout hashes agree and a sidecar matches the page's world.
 */
export function compileFixtureWorld(fixture: DisocclusionFixture): ShadoWorldSpatialPackage {
  return compileShadoWorld(fixture.primitives, {
    name: fixture.name,
    ...fixture.compile,
    // One chunk per cell: draw units as fine as the regions being tested.
    minRenderChunkTriangles: 1,
    maxRenderChunkExtent: fixture.compile.tileSize,
  });
}

/** Linear interpolation along a route; clamps outside its time span. */
export function sampleRoute(route: readonly DisocclusionRouteKey[], t: number): { at: Vec3; look: Vec3 } {
  if (!route.length) throw new Error('empty route');
  if (t <= route[0]!.t) return { at: [...route[0]!.at], look: [...route[0]!.look] };
  for (let i = 1; i < route.length; i++) {
    const b = route[i]!;
    if (t > b.t) continue;
    const a = route[i - 1]!;
    const f = (t - a.t) / Math.max(1e-9, b.t - a.t);
    const lerp = (u: Vec3, v: Vec3): Vec3 => [u[0] + (v[0] - u[0]) * f, u[1] + (v[1] - u[1]) * f, u[2] + (v[2] - u[2]) * f];
    return { at: lerp(a.at, b.at), look: lerp(a.look, b.look) };
  }
  const last = route[route.length - 1]!;
  return { at: [...last.at], look: [...last.look] };
}
