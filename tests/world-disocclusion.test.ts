import { describe, expect, it } from '@jest/globals';

import {
  captureFrame,
  describeLayer,
  layerFront,
  layerOfDepth,
  lowBits,
  propagateReference,
  twoRoomFixture,
  type DisocclusionFrame,
  type DisocclusionSettings,
} from '../src/world/disocclusion';

// 4x4 tiles of 8x8 samples, 4 layers with fronts 1, 2, 4, 8 (far 16), extTan 1:
// one tile spans 0.5 in tan. Small enough to read the expected masks by eye.
const settings: DisocclusionSettings = { resolution: 32, tileSize: 8, layers: 4, filterCell: 0 };
const FULL = 64;

function frame(viewcellHalf: number, layers = 4): DisocclusionFrame {
  return {
    origin: [0, 0, 0],
    forward: [0, 0, 1],
    right: [1, 0, 0],
    up: [0, 1, 0],
    viewcellHalfX: viewcellHalf,
    viewcellHalfY: viewcellHalf,
    extTanX: 1,
    extTanY: 1,
    near: 1,
    far: layers === 4 ? 16 : 1 + 2 ** layers,
    sourceHalfDepth: 0,
    sourceHalfRight: 0,
    sourceHalfUp: 0,
    directionTanX: 1,
    directionTanY: 1,
  };
}

function counts(layers = 4, tiles = 4) {
  const data = new Uint32Array(tiles * tiles * layers);
  const set = (layer: number, y: number, x: number, value: number) => {
    data[(layer * tiles + y) * tiles + x] = value;
  };
  const fill = (layer: number, value: number) => {
    for (let y = 0; y < tiles; y++) for (let x = 0; x < tiles; x++) set(layer, y, x, value);
  };
  return { data, set, fill };
}

describe('disocclusion scalar reference', () => {
  it('admits every cell of an empty view', () => {
    const masks = propagateReference(frame(0), settings, counts().data);
    expect(Array.from(masks.visible).every(v => v === 1)).toBe(true);
  });

  it('hides everything behind a solid wall and nothing in front of it', () => {
    const c = counts();
    c.fill(1, FULL);
    const masks = propagateReference(frame(0.5), settings, c.data);
    expect(describeLayer(masks, 0)).toBe('....\n....\n....\n....');
    expect(describeLayer(masks, 1)).toBe('####\n####\n####\n####');
    expect(describeLayer(masks, 2)).toBe('    \n    \n    \n    ');
    expect(describeLayer(masks, 3)).toBe('    \n    \n    \n    ');
  });

  it('sees only through an open doorway from a point viewcell', () => {
    const c = counts();
    c.fill(1, FULL);
    c.set(1, 1, 1, 0);
    const masks = propagateReference(frame(0), settings, c.data);
    expect(describeLayer(masks, 2)).toBe('    \n .  \n    \n    ');
    expect(describeLayer(masks, 3)).toBe('    \n .  \n    \n    ');
  });

  it('disoccludes the tiles beside a doorway once the camera may move sideways', () => {
    const c = counts();
    c.fill(1, FULL);
    c.set(1, 1, 1, 0);
    // Growth from the layer-1 front (z 2) to the layer-2 back (z 8) is
    // 0.5 * (1/2 - 1/8) = 0.1875 tan: under one tile, so exactly one ring.
    const masks = propagateReference(frame(0.5), settings, c.data);
    expect(describeLayer(masks, 2)).toBe('... \n... \n... \n    ');
    // The target at (0, 1) is hidden from the centre and exposed by the move.
    const at = (m: typeof masks) => m.visible[(2 * 4 + 1) * 4 + 0];
    expect(at(propagateReference(frame(0), settings, c.data))).toBe(0);
    expect(at(masks)).toBe(1);
  });

  it('widens a one-tile slit laterally but not along its own length', () => {
    const c = counts();
    c.fill(1, FULL);
    for (let y = 0; y < 4; y++) c.set(1, y, 2, 0);
    expect(describeLayer(propagateReference(frame(0), settings, c.data), 3)).toBe('  . \n  . \n  . \n  . ');
    expect(describeLayer(propagateReference(frame(0.5), settings, c.data), 3)).toBe(' ...\n ...\n ...\n ...');
  });

  it('treats a partially covered tile as open and a fully covered one as closed', () => {
    const c = counts();
    c.fill(1, FULL);
    c.set(1, 1, 1, FULL - 1);
    const open = propagateReference(frame(0), settings, c.data);
    expect(describeLayer(open, 1)).toBe('####\n#o##\n####\n####');
    expect(describeLayer(open, 2)).toBe('    \n .  \n    \n    ');
    c.set(1, 1, 1, FULL);
    expect(describeLayer(propagateReference(frame(0), settings, c.data), 2)).toBe('    \n    \n    \n    ');
  });

  it('handles the first and last of 32 layer bits without a 32-bit shift', () => {
    expect(lowBits(0)).toBe(0);
    expect(lowBits(31)).toBe(0x7fffffff);
    expect(lowBits(32)).toBe(0xffffffff);
    const s32: DisocclusionSettings = { ...settings, layers: 32 };
    const c = counts(32);
    const open = propagateReference(frame(0.5, 32), s32, c.data);
    const last = (31 * 4 + 2) * 4 + 2;
    expect(open.column[2 * 4 + 2]).toBe(0x7fffffff);
    expect(open.visible[last]).toBe(1);
    c.fill(30, FULL);
    const walled = propagateReference(frame(0.5, 32), s32, c.data);
    expect(walled.visible[(30 * 4 + 2) * 4 + 2]).toBe(1);
    expect(walled.visible[last]).toBe(0);
    expect(walled.column[2 * 4 + 2]! & (1 << 30)).toBe(0);
  });

  it('places layer boundaries on the paper curve and sizes the fixture capture', () => {
    const f = frame(0);
    expect([0, 1, 2, 3, 4].map(l => layerFront(f, 4, l))).toEqual([1, 2, 4, 8, 16]);
    expect(layerOfDepth(f, 4, 1)).toBe(0);
    expect(layerOfDepth(f, 4, 3.999)).toBe(1);
    expect(layerOfDepth(f, 4, 4)).toBe(2);
    expect(layerOfDepth(f, 4, 15.999)).toBe(3);
    expect(layerOfDepth(f, 4, 16)).toBe(-1);
    expect(layerOfDepth(f, 4, 0.5)).toBe(-1);
    const capture = captureFrame(twoRoomFixture().captures[0]!);
    // Box half 0.75, lean 1 across / 0.6 up: viewcell 0.75 + 0.75 * lean.
    expect(capture.viewcellHalfX).toBeCloseTo(1.5);
    expect(capture.viewcellHalfY).toBeCloseTo(1.2);
    expect(capture.extTanX).toBeCloseTo(1 + 1.5 / 8);
    expect(capture.extTanY).toBeCloseTo(0.6 + 1.2 / 8);
    expect(capture.origin).toEqual([8, 5, 16]);
  });
});
