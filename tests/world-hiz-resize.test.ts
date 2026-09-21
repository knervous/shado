import { describe, expect, it } from '@jest/globals';

import { BABYLON } from '../src/babylon';
import { createPreviewSession } from '../src/devtools/session';
import { hizFixtureView } from '../src/world/hiz/fixture-math';
import { buildShadoHiZPyramid, projectShadoHiZBounds, testShadoHiZ } from '../src/world/hiz/reference';
import { ShadoWorldHiZ } from '../src/world/hiz/ShadoWorldHiZ';
import type { ShadoHiZDepthConvention } from '../src/world/hiz/types';
import { SHADO_HIZ_FLAG_REJECTED } from '../src/world/hiz/wgsl';

/**
 * V1 regression (docs/pvs.md): resizing allocated fresh, zeroed level
 * parameter buffers and skipped writing them whenever the depth convention had
 * not changed, so the seed wrote nothing and a pyramid of depth 0 rejected
 * everything. This drives the real ShadoWorldHiZ on Babylon's WebGPUEngine
 * over Dawn through resizes in both conventions and holds each pyramid and
 * every verdict to the scalar reference.
 */

const BOXES: Array<{ min: [number, number, number]; max: [number, number, number] }> = [
  { min: [-0.5, -0.5, 9.5], max: [0.5, 0.5, 10.5] }, // behind the wall, through the hole
  { min: [-8, -1, 14], max: [-6, 1, 16] }, // behind the wall, sealed
  { min: [3, -1, 20], max: [5, 1, 22] }, // behind the wall, sealed
  { min: [-1, -1, 3], max: [1, 1, 4] }, // in front of the wall
];

function depthImage(width: number, height: number, convention: ShadoHiZDepthConvention): Float32Array {
  const view = hizFixtureView(width, height, convention);
  const wall = projectShadoHiZBounds([-80, -80, 6], [80, 80, 6.01], view).nearest;
  const clear = convention === 'normal' ? 1 : 0;
  const depth = new Float32Array(width * height).fill(wall);
  // An opening round the centre so the first box stays visible.
  const hole = projectShadoHiZBounds(BOXES[0]!.min, BOXES[0]!.max, view);
  for (let y = hole.y0; y <= hole.y1; y++) for (let x = hole.x0; x <= hole.x1; x++) depth[y * width + x] = clear;
  return depth;
}

async function pump<T>(engine: any, work: Promise<T>, label: string): Promise<T> {
  let done = false;
  const guarded = work.finally(() => (done = true));
  const deadline = Date.now() + 10_000;
  while (!done) {
    if (Date.now() > deadline) throw new Error(`${label} did not resolve`);
    engine.beginFrame();
    engine.endFrame();
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  return guarded;
}

describe('ShadoWorldHiZ on Babylon WebGPU (headless Dawn)', () => {
  it('keeps pyramid and verdicts equal to the reference across resizes, normal and reversed', async () => {
    const session = await createPreviewSession({ width: 64, height: 48 });
    const engine = session.engine;
    const hiz = new ShadoWorldHiZ(engine);
    const textures: any[] = [];
    try {
      hiz.setCandidates(
        BOXES.map((box, i) => ({ id: i, min: box.min, max: box.max, batch: i, member: 0 })),
        BOXES.map(() => ({ indexCount: 36, firstIndex: 0, capacity: 1, wholeInstances: 1 }))
      );
      // The resize that broke: same convention twice in a row, then a flip.
      const steps: Array<[number, number, ShadoHiZDepthConvention]> = [
        [64, 48, 'normal'],
        [50, 30, 'normal'],
        [50, 30, 'reversed'],
        [37, 21, 'reversed'],
        [64, 48, 'normal'],
      ];
      for (const [width, height, convention] of steps) {
        hiz.resize(width, height);
        const view = hizFixtureView(width, height, convention);
        const depth = depthImage(width, height, convention);
        const texture = BABYLON.RawTexture.CreateRTexture(
          depth, width, height, engine, false, false,
          BABYLON.Constants.TEXTURE_NEAREST_SAMPLINGMODE, BABYLON.Constants.TEXTURETYPE_FLOAT
        );
        textures.push(texture);
        let run = hiz.run(texture, view);
        for (let frame = 0; frame < 200 && !run.complete; frame++) {
          engine.beginFrame();
          run = hiz.run(texture, view);
          engine.endFrame();
          await new Promise((resolve) => setTimeout(resolve, 5));
        }
        engine.beginFrame();
        run = hiz.run(texture, view);
        engine.endFrame();
        expect({ step: `${width}x${height} ${convention}`, complete: run.complete, admitAll: run.admitAll }).toEqual({
          step: `${width}x${height} ${convention}`,
          complete: true,
          admitAll: false,
        });

        const pyramid = await pump(engine, hiz.readPyramid(), 'pyramid readback');
        const flags = await pump(engine, hiz.readFlags(), 'flag readback');
        const reference = buildShadoHiZPyramid(depth, width, height, convention);
        expect(hiz.pyramidLayout!.levels[0]).toEqual({ width, height, offset: 0 });
        // A short summary, not a 4,000-word array diff: the failure has to be
        // readable, and a slow report after Dawn teardown is where the
        // process used to die with SIGSEGV.
        const mismatches = Array.from(pyramid!).flatMap((value, i) => (value === reference.data[i] ? [] : [i]));
        expect({
          step: `${width}x${height} ${convention}`,
          mismatched: mismatches.length,
          first: mismatches.slice(0, 3).map((i) => ({ word: i, gpu: pyramid![i], reference: reference.data[i] })),
        }).toEqual({ step: `${width}x${height} ${convention}`, mismatched: 0, first: [] });
        const expected = BOXES.map((box) => testShadoHiZ(reference, projectShadoHiZBounds(box.min, box.max, view), convention).visible);
        expect(Array.from(flags).map((flag) => flag !== SHADO_HIZ_FLAG_REJECTED)).toEqual(expected);
        // The fixture must reject and keep something at every size.
        expect(expected).toContain(true);
        expect(expected).toContain(false);
      }
    } finally {
      // Let the queue go idle before destroying buffers and the device: an
      // assertion that fails mid-sequence otherwise tears Dawn down under
      // in-flight work, which exits the process with SIGSEGV (139) instead
      // of reporting the failure.
      await pump(engine, engine._device.queue.onSubmittedWorkDone(), 'queue drain').catch(() => {});
      for (const texture of textures) texture.dispose();
      hiz.dispose();
      await pump(engine, engine._device.queue.onSubmittedWorkDone(), 'queue drain').catch(() => {});
      await session.dispose();
    }
  }, 60_000);
});
