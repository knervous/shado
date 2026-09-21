import { describe, expect, it } from '@jest/globals';

import { installHeadlessWebGpu } from '../src/devtools/headless-gpu';
import { hizFixtureView } from '../src/world/hiz/fixture-math';
import {
  buildShadoHiZPyramid,
  projectShadoHiZBounds,
  shadoHiZLayout,
  testShadoHiZ,
} from '../src/world/hiz/reference';
import {
  SHADO_HIZ_DEPTH_BIAS,
  type ShadoHiZDepthConvention,
  type ShadoHiZViewInput,
} from '../src/world/hiz/types';
import {
  emitShadoHiZCullWGSL,
  emitShadoHiZFinalizeWGSL,
  emitShadoHiZReduceWGSL,
  emitShadoHiZResetWGSL,
  emitShadoHiZSeedWGSL,
  SHADO_HIZ_FLAG_ADMIT_BASE,
  SHADO_HIZ_FLAG_REJECTED,
  SHADO_HIZ_VIEW_WORDS,
} from '../src/world/hiz/wgsl';

// Odd on both axes so every reduction has a ragged edge.
const W = 67;
const H = 45;

function makeDepth(wall: number, clear: number, holes: Array<[number, number, number, number]>): Float32Array {
  const d = new Float32Array(W * H).fill(wall);
  for (const [x0, y0, x1, y1] of holes) {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) d[y * W + x] = clear;
  }
  // A band of uncovered sky on the ragged last column and a NaN sample.
  for (let y = 0; y < H; y += 7) d[y * W + W - 1] = clear;
  d[3 * W + 3] = NaN;
  return d;
}

function candidates(): Array<{ min: [number, number, number]; max: [number, number, number] }> {
  const out: Array<{ min: [number, number, number]; max: [number, number, number] }> = [];
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
  for (let i = 0; i < 300; i++) {
    const cx = (rnd() - 0.5) * 24;
    const cy = (rnd() - 0.5) * 16;
    const cz = rnd() < 0.1 ? rnd() * 2 - 0.5 : 1 + rnd() * 40;
    const s = 0.05 + rnd() * 3;
    out.push({ min: [cx - s, cy - s, cz - s], max: [cx + s, cy + s, cz + s] });
  }
  return out;
}

async function runGpu(
  device: GPUDevice,
  depth: Float32Array,
  view: ShadoHiZViewInput,
  boxes: ReturnType<typeof candidates>,
  batchOf: (i: number) => number,
  batchCapacities: number[]
) {
  const U = (globalThis as any).GPUBufferUsage;
  const T = (globalThis as any).GPUTextureUsage;
  const layout = shadoHiZLayout(W, H);
  const make = (data: ArrayBufferView | number, usage = U.STORAGE | U.COPY_SRC | U.COPY_DST) => {
    const size = typeof data === 'number' ? data : data.byteLength;
    const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data as any);
    return buffer;
  };
  const texture = device.createTexture({ size: [W, H], format: 'r32float', usage: T.TEXTURE_BINDING | T.COPY_DST });
  device.queue.writeTexture({ texture }, depth as any, { bytesPerRow: W * 4 }, [W, H]);
  const pyramid = make(layout.words * 4);
  const clear = view.convention === 'normal' ? 1 : 0;
  const clearBits = new Uint32Array(new Float32Array([clear]).buffer)[0]!;
  const levelParams = layout.levels.map((dst, l) => {
    const src = layout.levels[Math.max(0, l - 1)]!;
    return make(new Uint32Array([src.offset, src.width, src.height, dst.offset, dst.width, dst.height, clearBits, view.convention === 'reversed' ? 1 : 0]));
  });

  const members: number[] = new Array(batchCapacities.length).fill(0);
  const cand = new Float32Array(boxes.length * 8);
  const candBits = new Uint32Array(cand.buffer);
  boxes.forEach((b, i) => {
    const batch = batchOf(i);
    cand.set([...b.min], i * 8);
    candBits[i * 8 + 3] = batch;
    cand.set([...b.max], i * 8 + 4);
    candBits[i * 8 + 7] = members[batch]!++;
  });
  let segment = 0;
  const batchWords = new Uint32Array(batchCapacities.length * 4);
  batchCapacities.forEach((cap, b) => {
    batchWords.set([36, 6 * b, cap, segment], b * 4);
    segment += cap;
  });
  const viewWords = new Uint32Array(SHADO_HIZ_VIEW_WORDS);
  const viewFloats = new Float32Array(viewWords.buffer);
  for (let i = 0; i < 16; i++) viewFloats[i] = view.viewProjection[i]!;
  viewWords.set([W, H, view.convention === 'reversed' ? 1 : 0, 1, 1, 0], 16);
  viewFloats[22] = SHADO_HIZ_DEPTH_BIAS;
  viewWords[23] = layout.levels.length;
  viewWords[24] = boxes.length;
  layout.levels.forEach((l, i) => viewWords.set([l.width, l.height, l.offset], 28 + i * 4));

  const candBuf = make(cand);
  const batchBuf = make(batchWords);
  const viewBuf = make(viewWords);
  const args = make(batchCapacities.length * 20, U.STORAGE | U.COPY_SRC | U.COPY_DST | U.INDIRECT);
  const visible = make(Math.max(1, segment) * 4);
  const overflowSized = device.createBuffer({ size: batchCapacities.length * 4, usage: U.STORAGE | U.COPY_SRC });
  const flags = make(boxes.length * 4);

  const pipeline = (code: string) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const bind = (p: GPUComputePipeline, resources: Array<GPUBuffer | GPUTextureView>) =>
    device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: resources.map((r, binding) => ({ binding, resource: 'mapAsync' in (r as any) ? { buffer: r as GPUBuffer } : (r as GPUTextureView) })),
    });

  const encoder = device.createCommandEncoder();
  const pass = (p: GPUComputePipeline, g: GPUBindGroup, x: number, y = 1) => {
    const cp = encoder.beginComputePass();
    cp.setPipeline(p);
    cp.setBindGroup(0, g);
    cp.dispatchWorkgroups(x, y);
    cp.end();
  };
  const seed = pipeline(emitShadoHiZSeedWGSL());
  pass(seed, bind(seed, [texture.createView(), pyramid, levelParams[0]!]), Math.ceil(W / 8), Math.ceil(H / 8));
  const reduce = pipeline(emitShadoHiZReduceWGSL());
  for (let l = 1; l < layout.levels.length; l++) {
    const dst = layout.levels[l]!;
    pass(reduce, bind(reduce, [pyramid, levelParams[l]!]), Math.ceil(dst.width / 8), Math.ceil(dst.height / 8));
  }
  const reset = pipeline(emitShadoHiZResetWGSL());
  pass(reset, bind(reset, [batchBuf, args, overflowSized]), 1);
  const cull = pipeline(emitShadoHiZCullWGSL());
  pass(cull, bind(cull, [pyramid, viewBuf, candBuf, batchBuf, args, visible, flags]), Math.ceil(boxes.length / 64));
  const fin = pipeline(emitShadoHiZFinalizeWGSL());
  pass(fin, bind(fin, [batchBuf, args, overflowSized]), 1);

  const read = async (src: GPUBuffer, bytes: number) => {
    const staging = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(src, 0, staging, 0, bytes);
    device.queue.submit([e.finish()]);
    await staging.mapAsync((globalThis as any).GPUMapMode.READ);
    return staging.getMappedRange().slice(0);
  };
  device.queue.submit([encoder.finish()]);
  return {
    pyramid: new Float32Array(await read(pyramid, layout.words * 4)),
    flags: new Uint32Array(await read(flags, boxes.length * 4)),
    args: new Uint32Array(await read(args, batchCapacities.length * 20)),
    visible: new Uint32Array(await read(visible, Math.max(1, segment) * 4)),
    overflow: new Uint32Array(await read(overflowSized, batchCapacities.length * 4)),
    segments: batchCapacities.map((_, b) => batchWords[b * 4 + 3]!),
    members,
  };
}

describe('Hi-Z WGSL on headless Dawn', () => {
  for (const convention of ['normal', 'reversed'] as ShadoHiZDepthConvention[]) {
    it(`matches the scalar reference (${convention})`, async () => {
      const headless = await installHeadlessWebGpu();
      let device: GPUDevice | undefined;
      try {
        const adapter = await headless.gpu.requestAdapter();
        device = (await adapter!.requestDevice()) as GPUDevice;
        const view = hizFixtureView(W, H, convention);
        const wall = projectShadoHiZBounds([-80, -80, 12], [80, 80, 12.01], view).nearest;
        const clear = convention === 'normal' ? 1 : 0;
        const depth = makeDepth(wall, clear, [[20, 10, 30, 22], [45, 30, 46, 31]]);
        const boxes = candidates();
        const batchCaps = [0, 0, 0];
        boxes.forEach((_, i) => batchCaps[i % 3]!++);
        const gpu = await runGpu(device, depth, view, boxes, i => i % 3, batchCaps);

        const ref = buildShadoHiZPyramid(depth, W, H, convention);
        expect(Array.from(gpu.pyramid)).toEqual(Array.from(ref.data));

        let rejected = 0;
        const expectedVisible: number[][] = [[], [], []];
        boxes.forEach((b, i) => {
          const p = projectShadoHiZBounds(b.min, b.max, view);
          const verdict = testShadoHiZ(ref, p, convention);
          const flag = gpu.flags[i]!;
          if (verdict.visible) {
            expect(flag).not.toBe(SHADO_HIZ_FLAG_REJECTED);
            expectedVisible[i % 3]!.push(Math.floor(i / 3));
            if (verdict.reason !== 'depth') expect(flag).toBe(SHADO_HIZ_FLAG_ADMIT_BASE + verdict.reason);
          } else {
            rejected++;
            expect(flag).toBe(SHADO_HIZ_FLAG_REJECTED);
          }
        });
        // The fixture must exercise both outcomes to mean anything.
        expect(rejected).toBeGreaterThan(20);
        expect(rejected).toBeLessThan(boxes.length - 20);

        for (let b = 0; b < 3; b++) {
          const count = gpu.args[b * 5 + 1]!;
          expect([gpu.args[b * 5]!, gpu.args[b * 5 + 2]!, gpu.args[b * 5 + 3]!, gpu.args[b * 5 + 4]!]).toEqual([36, 6 * b, 0, 0]);
          expect(count).toBe(expectedVisible[b]!.length);
          const seg = Array.from(gpu.visible.slice(gpu.segments[b]!, gpu.segments[b]! + count)).sort((x, y) => x - y);
          expect(seg).toEqual(expectedVisible[b]);
          expect(gpu.overflow[b]).toBe(0);
        }
      } finally {
        device?.destroy();
        await headless.dispose?.();
      }
    }, 30_000);
  }
});
