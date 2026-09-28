/**
 * The runtime physics pack format: a round trip keeps every mesh, instance and
 * chunk reference, and a damaged file is refused rather than half-read (the
 * client falls back to the collision artifact on any throw).
 */
import { describe, expect, it } from '@jest/globals';
import {
  decodeShadoPhysicsPack,
  encodeShadoPhysicsPack,
  type ShadoPhysicsPack,
} from '../src/world/physics-pack';

const box = {
  positions: Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
  indices: Uint32Array.from([0, 1, 2, 0, 2, 3]),
};
const soup = {
  positions: Float32Array.from([-5, 0, -5, 5, 0, -5, 5, 0, 5]),
  indices: Uint32Array.from([0, 1, 2]),
};

const pack: ShadoPhysicsPack = {
  chunkSize: 256,
  sourceHash: '85650dd6',
  meshes: [box, soup],
  instances: [
    { mesh: 0, translation: [10, 1, -3], rotation: [0, 0.3826834, 0, 0.9238795], scale: [3, 1, 2] },
    { mesh: 0, translation: [-300, 0, 12], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  ],
  chunks: [
    { x: -2, z: 0, flags: 12, soupMesh: -1, instances: Uint32Array.from([1]) },
    { x: 0, z: -1, flags: 15, soupMesh: 1, instances: Uint32Array.from([0, 1]) },
  ],
};

describe('physics pack', () => {
  it('round-trips meshes, instances and chunk references', () => {
    const decoded = decodeShadoPhysicsPack(encodeShadoPhysicsPack(pack));
    expect(decoded.chunkSize).toBe(256);
    expect(decoded.sourceHash).toBe('85650dd6');
    expect(decoded.meshes.map(mesh => [...mesh.positions])).toEqual(pack.meshes.map(mesh => [...mesh.positions]));
    expect(decoded.meshes.map(mesh => [...mesh.indices])).toEqual(pack.meshes.map(mesh => [...mesh.indices]));
    expect(decoded.instances[0]!.translation).toEqual([10, 1, -3]);
    expect(decoded.instances[0]!.scale).toEqual([3, 1, 2]);
    expect(decoded.instances[0]!.rotation[1]).toBeCloseTo(0.3826834, 6);
    expect(decoded.chunks.map(chunk => [chunk.x, chunk.z, chunk.flags, chunk.soupMesh, [...chunk.instances]])).toEqual([
      [-2, 0, 12, -1, [1]],
      [0, -1, 15, 1, [0, 1]],
    ]);
  });

  it('refuses a truncated file and an out-of-range index', () => {
    const bytes = encodeShadoPhysicsPack(pack);
    expect(() => decodeShadoPhysicsPack(bytes.slice(0, bytes.length - 8))).toThrow();
    const corrupt = bytes.slice();
    // The first mesh's first index lives at its index word; point it past the vertices.
    const words = new Uint32Array(corrupt.buffer);
    words[words[8 + 3]!] = 99;
    expect(() => decodeShadoPhysicsPack(corrupt)).toThrow(/invalid index/);
  });

  it('refuses a source hash that is not eight hex digits', () => {
    expect(() => encodeShadoPhysicsPack({ ...pack, sourceHash: 'nope' })).toThrow();
  });
});
