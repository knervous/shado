import { describe, expect, it } from '@jest/globals';

import {
  buildShadoHiZPyramid,
  projectShadoHiZBounds,
  shadoHiZLayout,
  testShadoHiZ,
} from '../src/world/hiz/reference';
import { ShadoHiZAdmitReason, type ShadoHiZViewInput } from '../src/world/hiz/types';
import { hizFixtureView } from '../src/world/hiz/fixture-math';

const W = 64;
const H = 48;

/** Depth image: far everywhere, `wall` depth inside [x0,x1)x[y0,y1). */
function wallDepth(
  wall: number,
  x0 = 0,
  x1 = W,
  y0 = 0,
  y1 = H,
  clear = 1,
  hole?: { x0: number; x1: number; y0: number; y1: number }
): Float32Array {
  const d = new Float32Array(W * H).fill(clear);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (hole && x >= hole.x0 && x < hole.x1 && y >= hole.y0 && y < hole.y1) continue;
      d[y * W + x] = wall;
    }
  }
  return d;
}

describe('Hi-Z reference', () => {
  it('lays out ceil levels down to 1x1 and keeps the far clear through odd edges', () => {
    const layout = shadoHiZLayout(5, 3);
    expect(layout.levels.map(l => [l.width, l.height])).toEqual([[5, 3], [3, 2], [2, 1], [1, 1]]);
    // One uncovered pixel in the odd last column must keep every ancestor far.
    const depth = new Float32Array(15).fill(0.2);
    depth[1 * 5 + 4] = 1;
    const pyramid = buildShadoHiZPyramid(depth, 5, 3, 'normal');
    const top = pyramid.layout.levels[3]!;
    expect(pyramid.data[top.offset]).toBe(1);
    const l1 = pyramid.layout.levels[1]!;
    expect(pyramid.data[l1.offset + 0 * l1.width + 2]).toBe(1);
    expect(pyramid.data[l1.offset + 0 * l1.width + 0]).toBeCloseTo(0.2);
    // Reversed: clear is 0, the minimum survives.
    const rev = buildShadoHiZPyramid(depth.map(v => (v === 1 ? 0 : 0.8)), 5, 3, 'reversed');
    expect(rev.data[rev.layout.levels[3]!.offset]).toBe(0);
  });

  it('rejects a box behind a sealed wall, keeps it through a doorway and a one-pixel slit', () => {
    const view = hizFixtureView(W, H, 'normal');
    const box = { min: [-0.5, -0.5, 9.5] as const, max: [0.5, 0.5, 10.5] as const };
    const projection = projectShadoHiZBounds(box.min, box.max, view);
    expect(projection.admit).toBe(ShadoHiZAdmitReason.None);
    const wall = projectShadoHiZBounds([-50, -50, 5], [50, 50, 5.01], view).nearest;

    const sealed = buildShadoHiZPyramid(wallDepth(wall), W, H, 'normal');
    expect(testShadoHiZ(sealed, projection, 'normal').visible).toBe(false);

    const door = buildShadoHiZPyramid(
      wallDepth(wall, 0, W, 0, H, 1, { x0: 28, x1: 36, y0: 18, y1: 30 }),
      W,
      H,
      'normal'
    );
    expect(testShadoHiZ(door, projection, 'normal').visible).toBe(true);

    // A single uncovered pixel anywhere inside the footprint keeps the box.
    const cx = (projection.x0 + projection.x1) >> 1;
    const slit = buildShadoHiZPyramid(
      wallDepth(wall, 0, W, 0, H, 1, { x0: cx, x1: cx + 1, y0: projection.y1 - 1, y1: projection.y1 }),
      W,
      H,
      'normal'
    );
    expect(testShadoHiZ(slit, projection, 'normal').visible).toBe(true);
  });

  it('keeps a box in front of the wall and one coplanar with it', () => {
    const view = hizFixtureView(W, H, 'normal');
    const wall = projectShadoHiZBounds([-50, -50, 5], [50, 50, 5.01], view).nearest;
    const pyramid = buildShadoHiZPyramid(wallDepth(wall), W, H, 'normal');
    const front = projectShadoHiZBounds([-0.5, -0.5, 3], [0.5, 0.5, 4], view);
    expect(testShadoHiZ(pyramid, front, 'normal').visible).toBe(true);
    const coplanar = projectShadoHiZBounds([-0.5, -0.5, 5], [0.5, 0.5, 6], view);
    expect(testShadoHiZ(pyramid, coplanar, 'normal').visible).toBe(true);
  });

  it('admits near-plane, camera-inside, off-screen and non-finite bounds without a depth test', () => {
    const view = hizFixtureView(W, H, 'normal');
    expect(projectShadoHiZBounds([-1, -1, -1], [1, 1, 1], view).admit).toBe(ShadoHiZAdmitReason.NearPlane);
    expect(projectShadoHiZBounds([-1, -1, 0.01], [1, 1, 2], view).admit).toBe(ShadoHiZAdmitReason.NearPlane);
    expect(projectShadoHiZBounds([40, -1, 5], [41, 1, 6], view).admit).toBe(ShadoHiZAdmitReason.OffScreen);
    expect(projectShadoHiZBounds([NaN, 0, 5], [1, 1, 6], view).admit).toBe(ShadoHiZAdmitReason.NonFinite);
    // Whatever the pyramid says, an admitted projection stays visible.
    const pyramid = buildShadoHiZPyramid(wallDepth(0), W, H, 'normal');
    const inside = projectShadoHiZBounds([-1, -1, -1], [1, 1, 1], view);
    expect(testShadoHiZ(pyramid, inside, 'normal').visible).toBe(true);
  });

  it('mirrors every decision under reversed depth', () => {
    const normal = hizFixtureView(W, H, 'normal');
    const reversed = hizFixtureView(W, H, 'reversed');
    const box = { min: [-0.5, -0.5, 9.5] as const, max: [0.5, 0.5, 10.5] as const };
    const pn = projectShadoHiZBounds(box.min, box.max, normal);
    const pr = projectShadoHiZBounds(box.min, box.max, reversed);
    expect([pr.x0, pr.y0, pr.x1, pr.y1]).toEqual([pn.x0, pn.y0, pn.x1, pn.y1]);
    const wallR = projectShadoHiZBounds([-50, -50, 5], [50, 50, 5.01], reversed).nearest;
    expect(wallR).toBeGreaterThan(pr.nearest);
    const sealed = buildShadoHiZPyramid(wallDepth(wallR, 0, W, 0, H, 0), W, H, 'reversed');
    expect(testShadoHiZ(sealed, pr, 'reversed').visible).toBe(false);
    const open = buildShadoHiZPyramid(
      wallDepth(wallR, 0, W, 0, H, 0, { x0: pr.x0, x1: pr.x0 + 1, y0: pr.y0, y1: pr.y0 + 1 }),
      W,
      H,
      'reversed'
    );
    expect(testShadoHiZ(open, pr, 'reversed').visible).toBe(true);
  });

  it('agrees on footprint between top-left and bottom-left pixel origins', () => {
    const top = hizFixtureView(W, H, 'normal');
    const bottom: ShadoHiZViewInput = { ...top, topLeftOrigin: false };
    const box = { min: [1, 1, 8] as const, max: [2, 3, 9] as const };
    const a = projectShadoHiZBounds(box.min, box.max, top);
    const b = projectShadoHiZBounds(box.min, box.max, bottom);
    expect(a.x0).toBe(b.x0);
    expect(a.y0 + b.y1).toBe(H - 1);
    expect(a.y1 + b.y0).toBe(H - 1);
    // +y world is up and so toward row 0 with a top-left origin.
    expect(a.y1).toBeLessThan(H / 2);
  });
});
