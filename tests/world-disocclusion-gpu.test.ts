import { describe, expect, it } from '@jest/globals';

import { installHeadlessWebGpu } from '../src/devtools/headless-gpu';
import { compileShadoWorld } from '../src/world/compiler';
import {
  captureFrame,
  countTiles,
  decodeDisocclusionSidecar,
  DisocclusionAdmission,
  DISOCCLUSION_DEFAULT_SETTINGS,
  propagateReference,
  twoRoomFixture,
  type Vec3,
} from '../src/world/disocclusion';
import { bakeDisocclusionPvs } from '../src/world/disocclusion/bake-entry';

function compileFixture() {
  const fixture = twoRoomFixture();
  const world = compileShadoWorld(fixture.primitives, {
    name: fixture.name,
    ...fixture.compile,
    minRenderChunkTriangles: 1,
    maxRenderChunkExtent: fixture.compile.tileSize,
  });
  return { fixture, world };
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
  it('matches the scalar reference, rejects the sealed target, keeps the doorway target, and round-trips', async () => {
    const headless = await installHeadlessWebGpu();
    let device: GPUDevice | undefined;
    try {
      const adapter = await headless.gpu.requestAdapter();
      device = (await adapter!.requestDevice()) as GPUDevice;
      const { fixture, world } = compileFixture();
      const settings = DISOCCLUSION_DEFAULT_SETTINGS;
      const out = await bakeDisocclusionPvs(device, world, fixture.primitives, fixture.captures, settings, {
        timeoutMs: 20_000,
        createdAt: '2026-09-20T00:00:00.000Z',
      });
      const domain = out.domains[0]!;
      const frame = captureFrame(fixture.captures[0]!);

      // Stage parity: counts from the depth layers, then propagation from counts.
      expect(Array.from(domain.capture.masks.count)).toEqual(Array.from(countTiles(domain.capture.layers)));
      const reference = propagateReference(frame, settings, domain.capture.masks.count);
      expect(Array.from(domain.capture.masks.column)).toEqual(Array.from(reference.column));
      expect(Array.from(domain.capture.masks.mask)).toEqual(Array.from(reference.mask));
      expect(Array.from(domain.capture.masks.visible)).toEqual(Array.from(reference.visible));
      // Every written sample carries an ID from the same fragment.
      const { depth, id } = domain.capture.layers;
      for (let i = 0; i < depth.length; i++) expect(depth[i] === 0xffffffff).toBe(id[i] === 0xffffffff);

      const clustersOf = (name: string) => {
        const primitive = fixture.primitives.findIndex(p => p.name === name);
        return world.clusters.primitive.flatMap((p, c) => (p === primitive ? [c] : []));
      };
      const { expanded, raw } = domain.classification;
      const door = clustersOf('target-door');
      const sealed = clustersOf('target-sealed');
      expect(door.length).toBeGreaterThan(0);
      expect(sealed.length).toBeGreaterThan(0);
      expect(door.some(c => raw[c])).toBe(true);
      expect(door.every(c => expanded[c])).toBe(true);
      expect(sealed.every(c => !expanded[c])).toBe(true);
      expect(domain.meta.counts.raw).toBeLessThanOrEqual(domain.meta.counts.expanded);

      // Plain-data round trip, then the runtime path.
      const json = JSON.stringify(out.meta);
      const sidecar = await decodeDisocclusionSidecar(json, out.payload);
      const tampered = Uint8Array.from(out.payload);
      tampered[0] ^= 1;
      await expect(decodeDisocclusionSidecar(json, tampered)).rejects.toThrow('hash');
      const admission = new DisocclusionAdmission(world, sidecar);
      expect(admission.identityError).toBeNull();
      const facing = admission.evaluate(pose(fixture.start.at, fixture.start.look));
      expect(facing.mode).toBe('baked');
      const cellOf = (c: number) => world.clusters.cellId[c]!;
      expect(door.every(c => facing.cellMask![cellOf(c)])).toBe(true);
      expect(sealed.every(c => !facing.cellMask![cellOf(c)])).toBe(true);
      const behind = admission.evaluate(pose(fixture.start.at, [-24, 5, 16]));
      expect(behind.mode).toBe('reference');
      expect(behind.cellMask).toBeNull();
      const outside = admission.evaluate(pose([12, 5, 16], fixture.start.look));
      expect(outside.mode).toBe('reference');
    } finally {
      device?.destroy();
      headless.dispose();
    }
  }, 60_000);
});
