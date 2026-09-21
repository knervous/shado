import { describe, expect, it, beforeAll, afterAll } from '@jest/globals';
import { installHeadlessWebGpu } from '../src/devtools/headless-gpu';
import {
  buildInstancedOccluders,
  buildOccluderBvh,
  bvhSegmentBlocked,
  compileShadoWorldVisibility,
  compileShadoWorldVisibilityWith,
  createGpuSceneBackend,
  createGpuVisibilityBackend,
  instancedSegmentBlocked,
} from '../src/world';
import type { ShadoWorldPrimitive } from '../src/world';

/*
 * A REAL device: the headless Dawn the rest of the project renders on. The
 * stub-device tests prove refusals; only these prove the shader answers.
 */
let device: GPUDevice | null = null;
let dispose: (() => void) | null = null;
beforeAll(async () => {
  const headless = await installHeadlessWebGpu();
  dispose = headless.dispose;
  const adapter = await headless.gpu.requestAdapter();
  device = adapter ? ((await adapter.requestDevice()) as GPUDevice) : null;
});
afterAll(() => {
  device?.destroy();
  dispose?.();
});

function quads(name: string, corners: number[][], doubleSided = true): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const quad of corners) {
    const base = positions.length / 3;
    positions.push(...quad);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { name, material: 'stone', positions: new Float32Array(positions), indices: new Uint32Array(indices), doubleSided };
}

/** A wall in the YZ plane at x, facing +X when single-sided. */
const wallAt = (x: number, half = 4, doubleSided = true) =>
  quads(`wall-${x}`, [[x, -half, -half, x, -half, half, x, half, half, x, half, -half]], doubleSided);

/**
 * Segments chosen to sit on the cases f32 and f64 are most likely to
 * disagree about: grazing an edge, parallel to a face, hitting within the
 * endpoint epsilon, huge coordinates, and both faces of a one-sided wall.
 */
function edgeCases(): number[][] {
  return [
    [-10, 0, 0, 10, 0, 0], // straight through
    [-10, 4, 0, 10, 4, 0], // grazing the top edge
    [-10, 4.0001, 0, 10, 4.0001, 0], // just over it
    [-10, 0, 0, -10, 0, 5], // parallel to the face, never reaching it
    [0.0005, 0, 0, 10, 0, 0], // starting within the endpoint epsilon
    [10, 0, 0, -10, 0, 0], // the other way through a one-sided wall
    [-1e6, 0, 0, 1e6, 0, 0], // large coordinates
    [-10, 0, 3.9999, 10, 0, 3.9999], // inside the side edge
    [0, 0, 0, 0, 0, 0], // degenerate
  ];
}

describe('GPU segment backend on a real device', () => {
  it('matches the CPU on the edge cases, or errs only towards clear', async () => {
    if (!device) return;
    for (const doubleSided of [true, false]) {
      const bvh = buildOccluderBvh([wallAt(0, 4, doubleSided)]);
      const gpu = await createGpuSceneBackend(device, bvh, null);
      const cases = edgeCases();
      const got = await gpu.blockedBatch(Float32Array.from(cases.flat()), cases.length);
      let falseBlockers = 0;
      let missedBlockers = 0;
      cases.forEach((segment, index) => {
        const cpu = bvhSegmentBlocked(bvh, ...(segment as [number, number, number, number, number, number]));
        if (got[index] === 1 && !cpu) falseBlockers += 1;
        if (got[index] === 0 && cpu) missedBlockers += 1;
      });
      /*
       * Recorded apart. A false blocker would be harmful if trusted; the
       * pair backend never trusts one (it asks the CPU). A missed blocker
       * only admits. On these cases neither occurs.
       */
      expect({ doubleSided, falseBlockers, missedBlockers }).toEqual({ doubleSided, falseBlockers: 0, missedBlockers: 0 });
      gpu.dispose();
    }
  });

  it('walks mirrored and rotated placements the way the CPU does', async () => {
    if (!device) return;
    const prototype = [wallAt(0, 2, false)];
    const mirror = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 20, 0, 0, 1];
    const turn = [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 40, 0, 0, 1];
    const plain = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const instanced = buildInstancedOccluders([prototype], [
      { prototype: 0, matrix: plain },
      { prototype: 0, matrix: mirror },
      { prototype: 0, matrix: turn },
    ]);
    const gpu = await createGpuSceneBackend(device, null, instanced);
    const cases: number[][] = [];
    for (const x of [0, 20, 40]) {
      cases.push([x - 5, 0, 0, x + 5, 0, 0], [x + 5, 0, 0, x - 5, 0, 0]);
      cases.push([x, 0, -5, x, 0, 5], [x, 0, 5, x, 0, -5]);
    }
    const got = await gpu.blockedBatch(Float32Array.from(cases.flat()), cases.length);
    cases.forEach((segment, index) => {
      const cpu = instancedSegmentBlocked(instanced, ...(segment as [number, number, number, number, number, number]));
      expect({ index, gpu: got[index] === 1 }).toEqual({ index, gpu: cpu });
    });
    gpu.dispose();
  });

  it('answers a zero-segment batch without mapping anything', async () => {
    if (!device) return;
    const gpu = await createGpuSceneBackend(device, buildOccluderBvh([wallAt(0)]), null);
    expect((await gpu.blockedBatch(new Float32Array(0), 0)).length).toBe(0);
    gpu.dispose();
  });

  it('refuses a second batch while one is in flight', async () => {
    if (!device) return;
    const gpu = await createGpuSceneBackend(device, buildOccluderBvh([wallAt(0)]), null);
    const segment = Float32Array.from([-10, 0, 0, 10, 0, 0]);
    const first = gpu.blockedBatch(segment, 1);
    await expect(gpu.blockedBatch(segment, 1)).rejects.toThrow(/concurrent/);
    await first;
    gpu.dispose();
  });

  it('stops when told to, and refuses everything after', async () => {
    if (!device) return;
    let stop: string | null = null;
    const gpu = await createGpuSceneBackend(device, buildOccluderBvh([wallAt(0)]), null, {
      stopReason: () => stop,
    });
    stop = 'deadline';
    await expect(gpu.blockedBatch(Float32Array.from([-10, 0, 0, 10, 0, 0]), 1)).rejects.toThrow(/deadline/);
    expect(gpu.failure).toMatch(/deadline/);
    gpu.dispose();
  });
});

describe('an end-to-end bake on the GPU backend', () => {
  /** The walled strip every mode test uses, with a mirrored stamp row too. */
  function input() {
    const length = 160;
    const depth = 16;
    const ground: number[][] = [];
    for (let x = 0; x < length; x += 8) ground.push([x, 0, 0, x + 8, 0, 0, x + 8, 0, depth, x, 0, depth]);
    const centers: [number, number][] = [];
    for (let x = 8; x < length; x += 16) centers.push([x, depth / 2]);
    const pillar = [quads('pillar', [[0, 0, -2, 0, 0, 2, 0, 60, 2, 0, 60, -2]])];
    return {
      mode: 'sampled-occlusion' as const,
      bounds: { min: [0, 0, 0] as [number, number, number], max: [length, 200, depth] as [number, number, number] },
      regionSize: 16,
      maxDistance: 1024,
      renderCellCenters: centers,
      persistentRenderCells: new Uint8Array(centers.length),
      collisionPrimitives: [quads('ground', ground), quads('wall', [[80, 0, 0, 80, 0, depth, 80, 200, depth, 80, 200, 0]])],
      instancedOccluders: {
        prototypes: [pillar],
        instances: [
          { prototype: 0, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 120, 0, 8, 1] },
          { prototype: 0, matrix: [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 130, 0, 8, 1] },
        ],
      },
    };
  }

  it('selects the GPU backend and produces rows equal to or more conservative than the CPU', async () => {
    if (!device) return;
    const cpu = compileShadoWorldVisibility(input() as never);
    const backend = createGpuVisibilityBackend(device);
    const gpu = await compileShadoWorldVisibilityWith(input() as never, backend);
    expect(backend.stats.fellBackToCpu).toBeNull();
    expect(backend.stats.segments).toBeGreaterThan(0);
    // Every bit the CPU sets, the GPU run sets.
    for (let index = 0; index < cpu.pvs.words.length; index += 1) {
      expect((gpu.pvs.words[index]! & cpu.pvs.words[index]!) >>> 0).toBe(cpu.pvs.words[index]! >>> 0);
    }
    // And on this scene nothing was lost to f32: the rows are identical.
    expect(gpu.pvs.words).toEqual(cpu.pvs.words);
    expect(backend.stats.falseBlockers).toBe(0);
    backend.dispose();
  });

  it('falls back to the CPU, with the reason, when the scene will not fit', async () => {
    if (!device) return;
    const backend = createGpuVisibilityBackend(device, { maxDeviceBytes: 64 });
    const cpu = compileShadoWorldVisibility(input() as never);
    const gpu = await compileShadoWorldVisibilityWith(input() as never, backend);
    expect(backend.stats.fellBackToCpu).toMatch(/device bytes/);
    expect(gpu.pvs.words).toEqual(cpu.pvs.words);
    backend.dispose();
  });
});
