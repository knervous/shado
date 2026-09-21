import { describe, expect, it } from '@jest/globals';

import { installHeadlessWebGpu } from '../src/devtools/headless-gpu';
import {
  captureFrame,
  compileFixtureWorld,
  countTiles,
  decodeDisocclusionSidecar,
  DisocclusionAdmission,
  DISOCCLUSION_DEFAULT_SETTINGS,
  propagateReference,
  twoRoomFixture,
  type Vec3,
} from '../src/world/disocclusion';
import { bakeDisocclusionCapture, bakeDisocclusionPvs } from '../src/world/disocclusion/bake-entry';
import { classifyBox } from '../src/world/disocclusion/classify';
import type { DisocclusionGeometry } from '../src/world/disocclusion';

function compileFixture() {
  const fixture = twoRoomFixture();
  return { fixture, world: compileFixtureWorld(fixture) };
}

/** Corner rays of a symmetric frustum looking from `at` toward `look`. */
function pose(at: Vec3, look: Vec3, tanX = 0.55, tanY = 0.35) {
  const f = [look[0] - at[0], look[1] - at[1], look[2] - at[2]];
  const len = Math.hypot(f[0]!, f[1]!, f[2]!);
  const fw: Vec3 = [f[0]! / len, f[1]! / len, f[2]! / len];
  const r0: Vec3 = [fw[2], 0, -fw[0]];
  const rl = Math.hypot(r0[0], r0[2]) || 1;
  const right: Vec3 = [r0[0] / rl, 0, r0[2] / rl];
  const up: Vec3 = [fw[1] * right[2] - fw[2] * right[1], fw[2] * right[0] - fw[0] * right[2], fw[0] * right[1] - fw[1] * right[0]];
  const rays: Vec3[] = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    rays.push([0, 1, 2].map(i => fw[i]! + right[i]! * sx * tanX + up[i]! * sy * tanY) as Vec3);
  }
  return { position: at, cornerRays: rays };
}

describe('disocclusion GPU bake (headless Dawn)', () => {
  it('bakes six faces that match the scalar reference, hide the sealed room from every direction, and round-trip', async () => {
    const headless = await installHeadlessWebGpu();
    let device: GPUDevice | undefined;
    try {
      const adapter = await headless.gpu.requestAdapter();
      device = (await adapter!.requestDevice()) as GPUDevice;
      const { fixture, world } = compileFixture();
      const settings = DISOCCLUSION_DEFAULT_SETTINGS;
      const captures = fixture.captures.map(c => ({ ...c, id: `${c.volume}${c.axis}` }));
      const out = await bakeDisocclusionPvs(device, world, fixture.primitives, captures, settings, {
        timeoutMs: 20_000,
        createdAt: '2026-09-20T00:00:00.000Z',
      });
      expect(out.domains.map(d => d.meta.capture.axis)).toEqual(['+x', '-x', '+y', '-y', '+z', '-z']);

      // Stage parity on every face: counts from the depth layers, propagation from counts.
      for (const [i, domain] of out.domains.entries()) {
        const frame = captureFrame(fixture.captures[i]!);
        expect(Array.from(domain.capture.masks.count)).toEqual(Array.from(countTiles(domain.capture.layers)));
        const reference = propagateReference(frame, settings, domain.capture.masks.count);
        expect(Array.from(domain.capture.masks.column)).toEqual(Array.from(reference.column));
        expect(Array.from(domain.capture.masks.mask)).toEqual(Array.from(reference.mask));
        expect(Array.from(domain.capture.masks.visible)).toEqual(Array.from(reference.visible));
        const { depth, id } = domain.capture.layers;
        for (let s = 0; s < depth.length; s++) expect(depth[s] === 0xffffffff).toBe(id[s] === 0xffffffff);
      }

      const clustersOf = (name: string) => {
        const primitive = fixture.primitives.findIndex(p => p.name === name);
        return world.clusters.primitive.flatMap((p, c) => (p === primitive ? [c] : []));
      };
      const door = clustersOf('target-door');
      const sealed = clustersOf('target-sealed');
      expect(door.length).toBeGreaterThan(0);
      expect(sealed.length).toBeGreaterThan(0);
      const east = out.domains[0]!.classification;
      expect(door.some(c => east.raw[c])).toBe(true);
      expect(door.every(c => east.expanded[c])).toBe(true);
      // No face of the volume admits the sealed room's target.
      for (const d of out.domains) expect(sealed.every(c => !d.classification.admitted[c])).toBe(true);
      // Known cost, kept on purpose (pvs.md R3): the filter admits room C's
      // floor strip along the B/C wall, via the hidden floor under that wall.
      const stripFloor = world.clusters.primitive.flatMap((p, c) =>
        fixture.primitives[p]!.name === 'floor' && world.clusters.centerZ[c]! > 16 && world.clusters.centerZ[c]! < 24 && world.clusters.centerX[c]! > 32 ? [c] : []
      );
      expect(stripFloor.some(c => east.admitted[c] && !east.expanded[c])).toBe(true);

      // Plain-data round trip, then the runtime path.
      const json = JSON.stringify(out.meta);
      const sidecar = await decodeDisocclusionSidecar(json, out.payload);
      const tampered = Uint8Array.from(out.payload);
      tampered[0] ^= 1;
      await expect(decodeDisocclusionSidecar(json, tampered)).rejects.toThrow('hash');
      const admission = new DisocclusionAdmission(world, sidecar);
      expect(admission.identityError).toBeNull();
      const cellOf = (c: number) => world.clusters.cellId[c]!;
      const at = fixture.start.at;
      const poses: Record<string, ReturnType<typeof pose>> = {
        doorway: pose(at, fixture.start.look),
        seam: pose(at, [40, 5, 48]), // 45 degrees between the +x and +z faces
        rear: pose(at, [-24, 5, 16]),
        overhead: pose(at, [8.01, 60, 16]),
        floor: pose(at, [8.01, -60, 16]),
      };
      for (const [name, p] of Object.entries(poses)) {
        const result = admission.evaluate(p);
        expect([name, result.mode]).toEqual([name, 'baked']);
        expect([name, sealed.some(c => result.cellMask![cellOf(c)])]).toEqual([name, false]);
      }
      const doorway = admission.evaluate(poses.doorway!);
      expect(door.every(c => doorway.cellMask![cellOf(c)])).toBe(true);
      expect(doorway.domains).toEqual(['source+x']);
      // The face-plane separation test is conservative: a diagonal view may
      // also pull in faces it cannot actually see into. Extra rows only admit.
      expect(admission.evaluate(poses.seam!).domains).toEqual(expect.arrayContaining(['source+x', 'source+z']));
      expect(admission.evaluate(poses.rear!).domains).toEqual(['source-x']);
      const outside = admission.evaluate(pose([12, 5, 16], fixture.start.look));
      expect(outside.mode).toBe('reference');
      expect(outside.cellMask).toBeNull();
    } finally {
      device?.destroy();
      headless.dispose();
    }
  }, 60_000);
});

describe('disocclusion blocker sidedness (headless Dawn)', () => {
  it('a single-sided wall hides what is behind it only when it faces the camera', async () => {
    const headless = await installHeadlessWebGpu();
    let device: GPUDevice | undefined;
    try {
      device = (await (await headless.gpu.requestAdapter())!.requestDevice()) as GPUDevice;
      const capture = {
        sourceMin: [-0.5, -0.5, -0.5] as Vec3,
        sourceMax: [0.5, 0.5, 0.5] as Vec3,
        axis: '+x' as const,
        directionTan: 0.5,
        near: 2,
        far: 128,
      };
      const frame = captureFrame(capture);
      // A 60x60 wall at x = 20, two triangles; `facing` winds it CCW as seen
      // from the camera (normal -x), the other way winds it away.
      const wall = (facing: boolean): DisocclusionGeometry => {
        const positions = new Float32Array([20, -30, -30, 20, -30, 30, 20, 30, 30, 20, 30, -30]);
        // Seen from -x looking +x with right = +z, up = +y: (z,y) CCW is 0,1,2.
        const front = [0, 1, 2, 0, 2, 3];
        const back = [0, 2, 1, 0, 3, 2];
        return {
          positions,
          indices: Uint32Array.from(facing ? front : back),
          triangleTarget: new Int32Array(2).fill(-1),
          blocker: new Uint8Array([1, 1]),
          doubleSided: new Uint8Array([0, 0]),
        };
      };
      const settings = { ...DISOCCLUSION_DEFAULT_SETTINGS, filterCell: 0 };
      const target = { min: [40, -2, -2] as Vec3, max: [42, 2, 2] as Vec3 };
      const verdict = async (facing: boolean) => {
        const result = await bakeDisocclusionCapture(device!, frame, settings, wall(facing), { timeoutMs: 20_000 });
        return classifyBox(frame, settings, result.masks, target.min, target.max);
      };
      expect(await verdict(true)).toBe('hidden');
      expect(await verdict(false)).toBe('cell');
    } finally {
      device?.destroy();
      headless.dispose();
    }
  }, 60_000);
});
