import { afterEach, describe, expect, it } from '@jest/globals';

import { adaptQueue, installImageDecoder } from '../src/devtools/headless-gpu';

const RED = [255, 0, 0, 255];
const GREEN = [0, 255, 0, 255];
const BLUE = [0, 0, 255, 255];
const WHITE = [255, 255, 255, 255];
const PIXELS_2X2 = new Uint8Array([...RED, ...GREEN, ...BLUE, ...WHITE]);

type Image = Uint8Array & { width: number; height: number };

function image(width: number, height: number, pixels: number[] | Uint8Array): Image {
  const result = new Uint8Array(pixels) as Image;
  result.width = width;
  result.height = height;
  return result;
}

function component(value: unknown, key: string, index: number, fallback: number): number {
  if (Array.isArray(value)) return value[index] ?? fallback;
  if (typeof value === 'object' && value !== null) {
    return (value as Record<string, number>)[key] ?? fallback;
  }
  return fallback;
}

function createQueue(width: number, height: number, format = 'rgba8unorm') {
  const pixels = new Uint8Array(width * height * 4);
  const texture = { format };
  let writes = 0;
  const queue = adaptQueue({
    writeTexture(destination: unknown, data: ArrayBufferView, layout: unknown, size: unknown) {
      writes++;
      const destinationDescriptor = destination as Record<string, unknown>;
      const origin = destinationDescriptor.origin;
      const originX = component(origin, 'x', 0, 0);
      const originY = component(origin, 'y', 1, 0);
      const copyWidth = component(size, 'width', 0, 0);
      const copyHeight = component(size, 'height', 1, 1);
      const offset = component(layout, 'offset', 0, 0);
      const bytesPerRow = component(layout, 'bytesPerRow', 1, copyWidth * 4);
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

      for (let y = 0; y < copyHeight; y++) {
        const sourceStart = offset + y * bytesPerRow;
        const destinationStart = ((originY + y) * width + originX) * 4;
        pixels.set(bytes.subarray(sourceStart, sourceStart + copyWidth * 4), destinationStart);
      }
    },
    copyExternalImageToTexture() {
      throw new Error('Synthetic images must use writeTexture');
    },
  });

  return { pixels, queue, texture, writes: () => writes };
}

function upload(
  target: ReturnType<typeof createQueue>,
  source: Record<string, unknown>,
  size: Record<string, number> = { width: 2, height: 2 },
  destination: Record<string, unknown> = {},
): void {
  target.queue.copyExternalImageToTexture(
    { source: image(2, 2, PIXELS_2X2), ...source },
    { texture: target.texture, ...destination },
    size,
  );
}

describe('headless WebGPU external-image uploads', () => {
  it('uploads a 2x2 image without flipping', () => {
    const target = createQueue(2, 2);
    upload(target, { flipY: false });
    expect([...target.pixels]).toEqual([...PIXELS_2X2]);
  });

  it('flips a 2x2 image vertically', () => {
    const target = createQueue(2, 2);
    upload(target, { flipY: true });
    expect([...target.pixels]).toEqual([...BLUE, ...WHITE, ...RED, ...GREEN]);
  });

  it('defaults flipY to false', () => {
    const target = createQueue(2, 2);
    upload(target, {});
    expect([...target.pixels]).toEqual([...PIXELS_2X2]);
  });

  it('uses the full source-image stride for partial-width copies', () => {
    const target = createQueue(1, 2);
    upload(target, {}, { width: 1, height: 2 });
    expect([...target.pixels]).toEqual([...RED, ...BLUE]);
  });

  it('honors non-zero X and Y source origins', () => {
    const source = image(3, 3, [
      ...RED, ...GREEN, ...BLUE,
      ...WHITE, 1, 2, 3, 4, 5, 6, 7, 8,
      9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ]);
    const target = createQueue(2, 2);
    target.queue.copyExternalImageToTexture(
      { source, origin: { x: 1, y: 1 } },
      { texture: target.texture },
      { width: 2, height: 2 },
    );
    expect([...target.pixels]).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20]);
  });

  it('crops at the source origin before applying flipY', () => {
    const source = image(3, 3, [
      ...RED, ...GREEN, ...BLUE,
      ...WHITE, 1, 2, 3, 4, 5, 6, 7, 8,
      9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ]);
    const target = createQueue(2, 2);
    target.queue.copyExternalImageToTexture(
      { source, origin: [1, 1], flipY: true },
      { texture: target.texture },
      [2, 2],
    );
    expect([...target.pixels]).toEqual([13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('preserves the destination origin', () => {
    const target = createQueue(4, 4);
    upload(target, {}, { width: 2, height: 2 }, { origin: { x: 1, y: 1, z: 0 } });
    expect([...target.pixels]).toEqual([
      ...new Array(20).fill(0),
      ...RED, ...GREEN, ...new Array(8).fill(0),
      ...BLUE, ...WHITE, ...new Array(20).fill(0),
    ]);
  });

  it('rejects multiple layers and treats zero layers as a no-op', () => {
    const target = createQueue(2, 2);
    expect(() => upload(target, {}, { width: 2, height: 2, depthOrArrayLayers: 2 })).toThrow(
      'depthOrArrayLayers to be at most 1',
    );
    upload(target, {}, { width: 2, height: 2, depthOrArrayLayers: 0 });
    expect(target.writes()).toBe(0);
  });

  it('explicitly rejects unsupported premultiplied-alpha conversion', () => {
    const target = createQueue(2, 2);
    expect(() => upload(target, {}, undefined, { premultipliedAlpha: true })).toThrow(
      'premultipliedAlpha: true',
    );
  });

  it('writes destination-native bytes for BGRA textures', () => {
    const target = createQueue(2, 2, 'bgra8unorm');
    upload(target, {});
    expect([...target.pixels]).toEqual([
      0, 0, 255, 255,
      0, 255, 0, 255,
      255, 0, 0, 255,
      255, 255, 255, 255,
    ]);
  });
});

describe('createImageBitmap shim options', () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'createImageBitmap');

  afterEach(() => {
    if (original) Object.defineProperty(globalThis, 'createImageBitmap', original);
    else delete (globalThis as Record<string, unknown>).createImageBitmap;
  });

  it('accepts Babylon’s no-conversion options and rejects unsupported behavior', async () => {
    delete (globalThis as Record<string, unknown>).createImageBitmap;
    installImageDecoder(async () => ({ width: 1, height: 1, data: new Uint8Array(RED) }));
    const createImageBitmap = (globalThis as Record<string, unknown>).createImageBitmap as (
      source: Uint8Array,
      ...arguments_: unknown[]
    ) => Promise<unknown>;

    const bitmap = await createImageBitmap(new Uint8Array(), {
      premultiplyAlpha: 'none',
      colorSpaceConversion: 'none',
    }) as Uint8Array;
    expect([...bitmap]).toEqual(RED);
    await expect(createImageBitmap(new Uint8Array(), { premultiplyAlpha: 'premultiply' })).rejects.toThrow(
      'premultiplyAlpha option',
    );
    await expect(createImageBitmap(new Uint8Array(), 0, 0, 1, 1)).rejects.toThrow('crop overload');
  });
});
