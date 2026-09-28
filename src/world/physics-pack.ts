/**
 * The runtime physics pack: a zone's player collision as INSTANCES of shared
 * meshes, for the client's Havok streamer.
 *
 * The collision artifact (`collision.ts`) flattens every stamped object into
 * world-space triangle soup, and that is the right shape for the readers that
 * query it -- line of sight, navmesh, seating, surveys. It is the wrong shape to
 * hand Havok. Crownward's artifact is 1.04M triangles of which 870k are stamps,
 * and those are a few hundred distinct modules repeated: one 360-triangle wall
 * bay is stamped 588 times. Havok has no shape serialisation, so the only thing
 * a bake can do about its BVH build time is give it less to build. Built once
 * per module and placed as container children, the same walls cost one BVH.
 *
 * So this pack sits BESIDE the collision artifact rather than replacing it:
 *
 *   - `meshes`: local-space prototype meshes (each built once) and world-space
 *     soup meshes (the non-stamp geometry, one per chunk);
 *   - `instances`: a mesh placed by translation, rotation and scale, which is
 *     exactly the QS transform a Havok container child carries;
 *   - `chunks`: the same XZ grid and keys as the collision artifact, each with
 *     an optional soup mesh and a list of instances overlapping it, in Morton
 *     order so that consecutive instances are spatial neighbours.
 *
 * `sourceHash` is the collision artifact's content hash. The pack is derived
 * from the same inputs, and a zone republished without regenerating its pack
 * would otherwise hand the player last week's walls; a reader compares the two
 * and falls back to the artifact on any mismatch.
 *
 * Layout (little-endian, 4-byte aligned):
 *   header      8 x u32  magic, version, chunkSize (f32), sourceHash,
 *                        meshCount, chunkCount, instanceCount, refCount
 *   meshes      meshCount x 4 u32   vertexCount, indexCount, positionWord, indexWord
 *   chunks      chunkCount x 6 u32  x (i32), z (i32), flags, soupMesh+1, firstRef, refCount
 *   refs        refCount x u32      instance ids, per chunk
 *   instances   instanceCount x 12  mesh (u32), t xyz, q xyzw, s xyz (f32)
 *   data        f32 positions / u32 indices, addressed in 4-byte words
 */

const MAGIC = 0x59485053; // 'SPHY'
const VERSION = 1;
const HEADER_WORDS = 8;
const MESH_WORDS = 4;
const CHUNK_WORDS = 6;
const INSTANCE_WORDS = 12;

export type ShadoPhysicsPackMesh = {
  positions: Float32Array;
  indices: Uint32Array;
};

export type ShadoPhysicsPackInstance = {
  mesh: number;
  translation: [number, number, number];
  /** Quaternion x, y, z, w. */
  rotation: [number, number, number, number];
  scale: [number, number, number];
};

export type ShadoPhysicsPackChunk = {
  x: number;
  z: number;
  flags: number;
  /** World-space soup for the chunk's non-instanced geometry, or -1. */
  soupMesh: number;
  instances: Uint32Array;
};

export type ShadoPhysicsPack = {
  chunkSize: number;
  /** The collision artifact content hash this pack was derived alongside. */
  sourceHash: string;
  meshes: ShadoPhysicsPackMesh[];
  instances: ShadoPhysicsPackInstance[];
  chunks: ShadoPhysicsPackChunk[];
};

export function encodeShadoPhysicsPack(pack: ShadoPhysicsPack): Uint8Array {
  const refCount = pack.chunks.reduce((total, chunk) => total + chunk.instances.length, 0);
  let words =
    HEADER_WORDS +
    pack.meshes.length * MESH_WORDS +
    pack.chunks.length * CHUNK_WORDS +
    refCount +
    pack.instances.length * INSTANCE_WORDS;
  const dataStart = words;
  for (const mesh of pack.meshes) words += mesh.positions.length + mesh.indices.length;

  const buffer = new ArrayBuffer(words * 4);
  const u32 = new Uint32Array(buffer);
  const i32 = new Int32Array(buffer);
  const f32 = new Float32Array(buffer);
  u32[0] = MAGIC;
  u32[1] = VERSION;
  f32[2] = pack.chunkSize;
  u32[3] = parseSourceHash(pack.sourceHash);
  u32[4] = pack.meshes.length;
  u32[5] = pack.chunks.length;
  u32[6] = pack.instances.length;
  u32[7] = refCount;

  let cursor = HEADER_WORDS;
  let data = dataStart;
  for (const mesh of pack.meshes) {
    if (mesh.positions.length % 3 !== 0 || mesh.indices.length % 3 !== 0) {
      throw new Error('Physics pack mesh is not indexed triangle geometry');
    }
    u32[cursor++] = mesh.positions.length / 3;
    u32[cursor++] = mesh.indices.length;
    u32[cursor++] = data;
    f32.set(mesh.positions, data);
    data += mesh.positions.length;
    u32[cursor++] = data;
    u32.set(mesh.indices, data);
    data += mesh.indices.length;
  }
  let ref = 0;
  const refsStart = cursor + pack.chunks.length * CHUNK_WORDS;
  for (const chunk of pack.chunks) {
    i32[cursor++] = chunk.x;
    i32[cursor++] = chunk.z;
    u32[cursor++] = chunk.flags;
    u32[cursor++] = chunk.soupMesh + 1;
    u32[cursor++] = ref;
    u32[cursor++] = chunk.instances.length;
    u32.set(chunk.instances, refsStart + ref);
    ref += chunk.instances.length;
  }
  cursor = refsStart + refCount;
  for (const instance of pack.instances) {
    u32[cursor++] = instance.mesh;
    f32.set(instance.translation, cursor);
    f32.set(instance.rotation, cursor + 3);
    f32.set(instance.scale, cursor + 7);
    cursor += 11;
  }
  return new Uint8Array(buffer);
}

/**
 * Views over `bytes`, no copies: a pack is decoded once per zone entry on the
 * main thread. `bytes` must be 4-byte aligned (a fresh ArrayBuffer is).
 * Throws on anything structurally wrong; callers fall back to the artifact.
 */
export function decodeShadoPhysicsPack(bytes: Uint8Array): ShadoPhysicsPack {
  if (bytes.byteOffset % 4 !== 0 || bytes.byteLength % 4 !== 0 || bytes.byteLength < HEADER_WORDS * 4) {
    throw new Error('Physics pack is truncated or misaligned');
  }
  const words = bytes.byteLength / 4;
  const u32 = new Uint32Array(bytes.buffer, bytes.byteOffset, words);
  const i32 = new Int32Array(bytes.buffer, bytes.byteOffset, words);
  const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, words);
  if (u32[0] !== MAGIC || u32[1] !== VERSION) throw new Error('Not a version 1 physics pack');
  const meshCount = u32[4]!;
  const chunkCount = u32[5]!;
  const instanceCount = u32[6]!;
  const refCount = u32[7]!;
  const refsStart = HEADER_WORDS + meshCount * MESH_WORDS + chunkCount * CHUNK_WORDS;
  const instancesStart = refsStart + refCount;
  if (instancesStart + instanceCount * INSTANCE_WORDS > words) {
    throw new Error('Physics pack tables overrun the file');
  }
  const inRange = (start: number, length: number) => start >= instancesStart && start + length <= words;

  const meshes: ShadoPhysicsPackMesh[] = [];
  for (let mesh = 0; mesh < meshCount; mesh++) {
    const base = HEADER_WORDS + mesh * MESH_WORDS;
    const vertexCount = u32[base]!;
    const indexCount = u32[base + 1]!;
    const positionWord = u32[base + 2]!;
    const indexWord = u32[base + 3]!;
    if (!inRange(positionWord, vertexCount * 3) || !inRange(indexWord, indexCount) || indexCount % 3 !== 0) {
      throw new Error(`Physics pack mesh ${mesh} is out of range`);
    }
    const indices = u32.subarray(indexWord, indexWord + indexCount);
    for (let index = 0; index < indices.length; index++) {
      if (indices[index]! >= vertexCount) throw new Error(`Physics pack mesh ${mesh} has an invalid index`);
    }
    meshes.push({ positions: f32.subarray(positionWord, positionWord + vertexCount * 3), indices });
  }

  const instances: ShadoPhysicsPackInstance[] = [];
  for (let instance = 0; instance < instanceCount; instance++) {
    const base = instancesStart + instance * INSTANCE_WORDS;
    const mesh = u32[base]!;
    if (mesh >= meshCount) throw new Error(`Physics pack instance ${instance} names no mesh`);
    instances.push({
      mesh,
      translation: [f32[base + 1]!, f32[base + 2]!, f32[base + 3]!],
      rotation: [f32[base + 4]!, f32[base + 5]!, f32[base + 6]!, f32[base + 7]!],
      scale: [f32[base + 8]!, f32[base + 9]!, f32[base + 10]!],
    });
  }

  const chunks: ShadoPhysicsPackChunk[] = [];
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const base = HEADER_WORDS + meshCount * MESH_WORDS + chunk * CHUNK_WORDS;
    const soupMesh = u32[base + 3]! - 1;
    const firstRef = u32[base + 4]!;
    const count = u32[base + 5]!;
    if (soupMesh >= meshCount || firstRef + count > refCount) {
      throw new Error(`Physics pack chunk ${chunk} is out of range`);
    }
    const refs = u32.subarray(refsStart + firstRef, refsStart + firstRef + count);
    for (let ref = 0; ref < refs.length; ref++) {
      if (refs[ref]! >= instanceCount) throw new Error(`Physics pack chunk ${chunk} names no instance`);
    }
    chunks.push({ x: i32[base]!, z: i32[base + 1]!, flags: u32[base + 2]!, soupMesh, instances: refs });
  }

  return {
    chunkSize: f32[2]!,
    sourceHash: u32[3]!.toString(16).padStart(8, '0'),
    meshes,
    instances,
    chunks,
  };
}

function parseSourceHash(hash: string): number {
  if (!/^[0-9a-f]{8}$/i.test(hash)) throw new Error(`Physics pack source hash '${hash}' is not 8 hex digits`);
  return Number.parseInt(hash, 16) >>> 0;
}
