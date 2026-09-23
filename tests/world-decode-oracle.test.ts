/**
 * The optimised world collision decode and spatial-package validation against
 * their frozen pre-optimisation implementations (tests/oracle/world).
 *
 * Same results on the real Crownward package when the client checkout sits
 * beside this repo, and the same rejections for corrupt input everywhere.
 */
import { describe, expect, it } from '@jest/globals';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import {
  decodeShadoWorldCollision,
  encodeShadoWorldCollision,
  fnv1a32Bytes,
} from '../src/world/collision';
import { computeShadoWorldLayoutHash, validateShadoWorldPackage } from '../src/world/validation';
import type { ShadoWorldCollisionDescriptor, ShadoWorldPrimitive } from '../src/world/types';
import {
  decodeShadoWorldCollision as decodeOracle,
  fnv1a32Bytes as fnvOracle,
} from './oracle/world/collision';
import { computeShadoWorldLayoutHash as layoutHashOracle } from './oracle/world/validation';

const here = path.dirname(fileURLToPath(import.meta.url));
const WORLD = path.resolve(here, '../../client/public/eqrequiem/worlds/talioscrownward');
const haveWorld =
  existsSync(`${WORLD}.collision.bin.gz`) && existsSync(`${WORLD}.spatial.json.gz`);

function bytesOf(view: ArrayBufferView): Buffer {
  return Buffer.from(view.buffer, view.byteOffset, view.byteLength);
}

function expectSameCollision(
  actual: ReturnType<typeof decodeShadoWorldCollision>,
  expected: ReturnType<typeof decodeOracle>
): void {
  const strip = (data: typeof actual) => ({
    ...data,
    chunks: data.chunks.map(({ positions, indices, ...rest }) => rest),
  });
  expect(strip(actual)).toEqual(strip(expected));
  actual.chunks.forEach((chunk, index) => {
    const other = expected.chunks[index]!;
    expect(bytesOf(chunk.positions).equals(bytesOf(other.positions))).toBe(true);
    expect(bytesOf(chunk.indices).equals(bytesOf(other.indices))).toBe(true);
  });
}

function grid(name: string, offsetX: number, size: number): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let z = 0; z <= size; z++) {
    for (let x = 0; x <= size; x++) positions.push(offsetX + x * 37.5, Math.sin(x + z), z * 41.25);
  }
  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      const a = z * (size + 1) + x;
      indices.push(a, a + 1, a + size + 1, a + 1, a + size + 2, a + size + 1);
    }
  }
  return { name, material: 'm', positions, indices } as unknown as ShadoWorldPrimitive;
}

describe('world collision decode matches the oracle', () => {
  const artifact = encodeShadoWorldCollision([grid('a', 0, 12), grid('b', 300, 9)], {
    chunkSize: 128,
  });
  const descriptor = { ...artifact.descriptor, source: 'x.bin' } as ShadoWorldCollisionDescriptor;

  it('hashes identically at every length, including unaligned tails', () => {
    for (const length of [0, 1, 2, 3, 4, 5, 7, 8, 1023, 4096 + 3]) {
      const bytes = artifact.bytes.subarray(0, Math.min(length, artifact.bytes.length));
      expect(fnv1a32Bytes(bytes)).toBe(fnvOracle(bytes));
    }
    expect(fnv1a32Bytes(artifact.bytes)).toBe(artifact.descriptor.contentHash);
  });

  it('decodes a synthetic artifact identically, including from an unaligned view', () => {
    expectSameCollision(
      decodeShadoWorldCollision(artifact.bytes, descriptor),
      decodeOracle(artifact.bytes, descriptor)
    );
    const padded = new Uint8Array(artifact.bytes.length + 3);
    padded.set(artifact.bytes, 3);
    const unaligned = padded.subarray(3);
    expectSameCollision(
      decodeShadoWorldCollision(unaligned, descriptor),
      decodeOracle(unaligned, descriptor)
    );
  });

  it('rejects the same corruption the oracle rejects', () => {
    // Integrity: any flipped byte.
    const flipped = artifact.bytes.slice();
    flipped[flipped.length - 5] ^= 1;
    expect(() => decodeShadoWorldCollision(flipped, descriptor)).toThrow(/integrity/);

    // An out-of-range index that still hashes to its descriptor.
    const badIndex = artifact.bytes.slice();
    const view = new DataView(badIndex.buffer);
    const vertexCount = view.getUint32(56 + 12, true);
    const payloadOffset = view.getUint32(56 + 20, true);
    view.setUint32(payloadOffset + vertexCount * 12, vertexCount, true);
    const rehashed = { ...descriptor, contentHash: fnvOracle(badIndex) };
    expect(() => decodeOracle(badIndex, rehashed)).toThrow(/invalid chunk index/);
    expect(() => decodeShadoWorldCollision(badIndex, rehashed)).toThrow(/invalid chunk index/);

    // A moved vertex: chunk bounds no longer match the geometry.
    const moved = artifact.bytes.slice();
    new DataView(moved.buffer).setFloat32(payloadOffset, 1e6, true);
    const movedDescriptor = { ...descriptor, contentHash: fnvOracle(moved) };
    expect(() => decodeOracle(moved, movedDescriptor)).toThrow(/chunk bounds/);
    expect(() => decodeShadoWorldCollision(moved, movedDescriptor)).toThrow(/chunk bounds/);

    // A NaN position poisons the bounds in both.
    const nan = artifact.bytes.slice();
    new DataView(nan.buffer).setFloat32(payloadOffset + 4, Number.NaN, true);
    const nanDescriptor = { ...descriptor, contentHash: fnvOracle(nan) };
    expect(() => decodeOracle(nan, nanDescriptor)).toThrow(/chunk bounds/);
    expect(() => decodeShadoWorldCollision(nan, nanDescriptor)).toThrow(/chunk bounds/);
  });

  (haveWorld ? it : it.skip)(
    'decodes and validates the shipped Crownward package identically',
    () => {
      const world = JSON.parse(gunzipSync(readFileSync(`${WORLD}.spatial.json.gz`)).toString('utf8'));
      const bytes = new Uint8Array(gunzipSync(readFileSync(`${WORLD}.collision.bin.gz`)));
      expectSameCollision(
        decodeShadoWorldCollision(bytes, world.collision),
        decodeOracle(bytes, world.collision)
      );
      const layoutHash = computeShadoWorldLayoutHash(world);
      expect(layoutHash).toBe(layoutHashOracle(world));
      expect(layoutHash).toBe(world.integrity.layoutHash);
      expect(() => validateShadoWorldPackage(world)).not.toThrow();

      // A single changed topology value must still fail validation.
      const firstIndex = world.clusters.firstIndex as number[];
      firstIndex[0] = (firstIndex[0] ?? 0) + 1;
      expect(computeShadoWorldLayoutHash(world)).toBe(layoutHashOracle(world));
      expect(() => validateShadoWorldPackage(world)).toThrow(/integrity mismatch/);
    },
    120_000
  );
});
