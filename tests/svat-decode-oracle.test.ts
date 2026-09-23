/**
 * The optimised `.svat` decoders against the frozen pre-optimisation decoder.
 *
 * `tests/oracle/svat` is the decoder exactly as it shipped before the fused
 * kernel and the worker (byteUnshuffle → deltaDecode → scatterChunk, FNV
 * checksum byte by byte). Every path here must reproduce its output byte for
 * byte: on every real container the Eltania client ships (when the checkout
 * sits beside it), and on synthetic atlases that cover what those do not —
 * float32, the unfiltered codec path, framesX > 1, multi-tile rows and
 * multi-chunk clips.
 */
import { afterAll, describe, expect, it } from '@jest/globals';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { PackedDQVAT } from '../src/extensions/VATBuilder/VATBuilder';
import {
  configureSvatDecodeWorkers,
  decodeSvat,
  decodeSvatInWorker,
  disposeSvatDecodeWorkers,
  encodeSvat,
  svatDecodeWorkerSource,
  SvatCodec,
  SvatFilter,
  type SvatDecodeWorkerLike,
} from '../src/svat';
import { nodeGzipCompress, nodeSvatDecompressor } from '../src/svat/SvatNode';
import { decodeSvat as decodeSvatOracle } from './oracle/svat/SvatCodec';
import { svatComponentIndex } from './oracle/svat/SvatFormat';

const here = path.dirname(fileURLToPath(import.meta.url));
const VAT_DIR = path.resolve(here, '../../client/public/eqrequiem/vat');

function pixelBytes(packed: PackedDQVAT): Buffer {
  return Buffer.from(packed.pixels.buffer, packed.pixels.byteOffset, packed.pixels.byteLength);
}

function expectIdentical(actual: PackedDQVAT, expected: PackedDQVAT, label: string): void {
  const { pixels: actualPixels, ...actualMeta } = actual;
  const { pixels: expectedPixels, ...expectedMeta } = expected;
  expect(actualMeta).toEqual(expectedMeta);
  expect(actualPixels.constructor).toBe(expectedPixels.constructor);
  expect(actualPixels.length).toBe(expectedPixels.length);
  const same = pixelBytes(actual).equals(pixelBytes(expected));
  if (!same) throw new Error(`${label}: decoded atlas differs from the oracle`);
}

/**
 * A worker that runs the real worker script in-process: the same text a
 * browser would load from the blob URL, with structuredClone standing in for
 * the thread boundary (transfer lists included).
 */
function inProcessWorker(source: string): SvatDecodeWorkerLike {
  const handle: SvatDecodeWorkerLike = {
    onmessage: null,
    onerror: null,
    postMessage(message, transfer) {
      const data = structuredClone(message, { transfer });
      queueMicrotask(() => (scope.onmessage as (event: { data: unknown }) => void)({ data }));
    },
    terminate() {},
  };
  const scope: Record<string, unknown> = {
    postMessage(message: unknown, transfer: Transferable[] = []) {
      const data = structuredClone(message, { transfer });
      queueMicrotask(() => handle.onmessage?.({ data }));
    },
  };
  new Function('self', source)(scope);
  return handle;
}

function synthetic(options: {
  componentType: 'float16' | 'float32';
  bones: number;
  widthBones: number;
  strideTexels: number;
  framesX: number;
  clips: number[];
}): PackedDQVAT {
  const { componentType, bones, widthBones, strideTexels, framesX } = options;
  const tilesX = Math.ceil(bones / widthBones);
  const framesTotal = options.clips.reduce((sum, frames) => sum + frames, 0);
  const widthTexels = framesX * widthBones * strideTexels;
  const heightTexels = Math.ceil(framesTotal / framesX) * tilesX;
  const count = widthTexels * heightTexels * 4;
  let seed = 0x1234567;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0);
  const layout = { bones, framesTotal, widthBones, tilesX, framesX, strideTexels, widthTexels, heightTexels, hasScale: strideTexels === 3, componentType };
  // Only addressed texels carry data; padding stays zero, as a bake leaves it.
  const pixels = componentType === 'float16' ? new Uint16Array(count) : new Float32Array(count);
  for (let frame = 0; frame < framesTotal; frame++) {
    for (let bone = 0; bone < bones; bone++) {
      for (let slot = 0; slot < strideTexels; slot++) {
        for (let c = 0; c < 4; c++) {
          const i = svatComponentIndex(layout, frame, bone, slot, c);
          pixels[i] =
            componentType === 'float16'
              ? // Smooth-ish streams with noise in the low bits, like real tracks.
                (0x3800 + ((frame + bone * 7 + slot * 3 + c) & 0x3ff)) ^ (next() & 0x1f)
              : Math.sin(frame * 0.05 + bone + c) + (next() & 0xff) * 1e-6;
        }
      }
    }
  }
  return {
    componentType,
    widthTexels,
    heightTexels,
    framesTotal,
    bones,
    dqWidthBones: widthBones,
    dqTilesX: tilesX,
    dqFramesX: framesX,
    dqStrideTexels: strideTexels,
    dqHasScale: strideTexels === 3,
    clips: options.clips.map((frames, index) => ({
      name: `clip${index}`,
      from: index * 10,
      to: index * 10 + frames,
      frames,
      fps: 30,
    })),
    pixels,
  };
}

afterAll(() => {
  disposeSvatDecodeWorkers();
  configureSvatDecodeWorkers({ workerFactory: null });
});

const realContainers = existsSync(VAT_DIR)
  ? readdirSync(VAT_DIR).filter(file => file.endsWith('.svat')).sort()
  : [];

describe('svat decode matches the pre-optimisation oracle', () => {
  const decompress = nodeSvatDecompressor();

  (realContainers.length ? it : it.skip)(
    `every shipped container (${realContainers.length}) — main thread and worker`,
    async () => {
      configureSvatDecodeWorkers({ workerFactory: inProcessWorker, maxWorkers: 2 });
      for (const file of realContainers) {
        const bytes = new Uint8Array(readFileSync(path.join(VAT_DIR, file)));
        const oracle = await decodeSvatOracle(bytes, { decompress });
        expectIdentical(await decodeSvat(bytes, { decompress }), oracle, `${file} main`);
        expectIdentical(await decodeSvatInWorker(bytes), oracle, `${file} worker`);
      }
    },
    300_000
  );

  const cases = [
    { componentType: 'float16', bones: 21, widthBones: 21, strideTexels: 2, framesX: 1, clips: [37, 5, 1] },
    { componentType: 'float16', bones: 33, widthBones: 8, strideTexels: 3, framesX: 3, clips: [20, 11] },
    { componentType: 'float32', bones: 17, widthBones: 17, strideTexels: 2, framesX: 1, clips: [40, 3] },
    { componentType: 'float32', bones: 29, widthBones: 7, strideTexels: 3, framesX: 2, clips: [13, 9, 2] },
  ] as const;

  for (const shape of cases) {
    for (const filter of [SvatFilter.DeltaXorShuffle, SvatFilter.None]) {
      for (const codec of [SvatCodec.None, SvatCodec.Gzip]) {
        it(`${shape.componentType} ${shape.bones}b/${shape.widthBones}w stride${shape.strideTexels} framesX${shape.framesX} filter${filter} codec${codec}`, async () => {
          const packed = synthetic({ ...shape, clips: [...shape.clips] });
          const encoded = await encodeSvat(packed, {
            codec,
            compress: codec === SvatCodec.Gzip ? nodeGzipCompress(6) : bytes => bytes,
            filter,
            continuity: false,
            // Several chunks per clip.
            targetChunkBytes: 2048,
          });
          const oracle = await decodeSvatOracle(encoded, { decompress });
          // The oracle itself round-trips (finite values, so float32 is exact).
          expect(pixelBytes(oracle).equals(pixelBytes(packed))).toBe(true);
          expectIdentical(await decodeSvat(encoded, { decompress }), oracle, 'main');
          configureSvatDecodeWorkers({ workerFactory: inProcessWorker });
          expectIdentical(await decodeSvatInWorker(encoded), oracle, 'worker');
        });
      }
    }
  }
});

describe('svat worker decode', () => {
  const packed = synthetic({
    componentType: 'float16',
    bones: 9,
    widthBones: 9,
    strideTexels: 2,
    framesX: 1,
    clips: [12],
  });

  it('rejects a corrupt chunk exactly like decodeSvat, without falling back', async () => {
    const encoded = await encodeSvat(packed, { codec: SvatCodec.None, compress: bytes => bytes });
    encoded[encoded.length - 1] ^= 0xff;
    configureSvatDecodeWorkers({ workerFactory: inProcessWorker });
    await expect(decodeSvatInWorker(encoded)).rejects.toThrow(/checksum mismatch/);
    await expect(
      decodeSvat(encoded, { decompress: nodeSvatDecompressor() })
    ).rejects.toThrow(/checksum mismatch/);
  });

  it('decodes on the main thread when no worker can be created', async () => {
    const encoded = await encodeSvat(packed, { codec: SvatCodec.None, compress: bytes => bytes });
    configureSvatDecodeWorkers({
      workerFactory: () => {
        throw new Error('blocked by CSP');
      },
    });
    const decoded = await decodeSvatInWorker(encoded);
    expect(pixelBytes(decoded).equals(pixelBytes(packed))).toBe(true);
    configureSvatDecodeWorkers({ workerFactory: null });
  });

  it('decodes on the main thread when the worker dies', async () => {
    const encoded = await encodeSvat(packed, { codec: SvatCodec.None, compress: bytes => bytes });
    configureSvatDecodeWorkers({
      workerFactory: () => {
        const worker: SvatDecodeWorkerLike = {
          onmessage: null,
          onerror: null,
          postMessage() {
            queueMicrotask(() => worker.onerror?.({ message: 'script failed to load' }));
          },
          terminate() {},
        };
        return worker;
      },
    });
    const decoded = await decodeSvatInWorker(encoded);
    expect(pixelBytes(decoded).equals(pixelBytes(packed))).toBe(true);
    configureSvatDecodeWorkers({ workerFactory: null });
  });

  it('leaves the caller bytes intact', async () => {
    const encoded = await encodeSvat(packed, { codec: SvatCodec.None, compress: bytes => bytes });
    const copy = encoded.slice();
    configureSvatDecodeWorkers({ workerFactory: inProcessWorker });
    await decodeSvatInWorker(encoded);
    expect(Buffer.from(encoded).equals(Buffer.from(copy))).toBe(true);
  });

  it('ships a self-contained script', () => {
    // Evaluating it must not reference anything outside itself.
    expect(() => new Function('self', svatDecodeWorkerSource())({})).not.toThrow();
  });
});
