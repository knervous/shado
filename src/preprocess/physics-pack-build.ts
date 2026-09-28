import {
  encodeShadoWorldCollision,
  ShadoCollisionFlags,
  type ShadoWorldCollisionChunk,
} from '../world/collision';
import type {
  ShadoPhysicsPack,
  ShadoPhysicsPackChunk,
  ShadoPhysicsPackInstance,
  ShadoPhysicsPackMesh,
} from '../world/physics-pack';
import type { ShadoWorldCompileOptions, ShadoWorldPrimitive } from '../world/types';
import {
  importStampedObjectGeometry,
  importWorldPrimitives,
  worldGlbPrimitivePolicies,
  type StampedCollisionInstance,
  type WorldObjectAssetLoader,
} from './world-core';

export type BuildShadoPhysicsPackOptions = {
  /** The runtime world GLB, as the collision artifact was baked from. */
  glb: Uint8Array;
  authoring: ShadoWorldCompileOptions['authoring'];
  loadObjectAsset: WorldObjectAssetLoader;
  /** The collision artifact's content hash; the pack is only valid beside it. */
  sourceHash: string;
  chunkSize: number;
  inputTransform?: ShadoWorldCompileOptions['sourceTransform'];
  inputRightHanded?: boolean;
};

export type ShadoPhysicsPackStats = {
  sourceTriangles: number;
  soupTriangles: number;
  prototypeTriangles: number;
  prototypes: number;
  instances: number;
  flattenedInstances: number;
  /** Triangles Havok builds if every chunk is resident once. */
  builtTriangles: number;
};

const STAMP_FLAGS = ShadoCollisionFlags.StaticObject | ShadoCollisionFlags.PlayerSolid;

/**
 * Derive a zone's physics pack (see `world/physics-pack.ts`) from the same
 * inputs its collision artifact is baked from. Deterministic: stamps and
 * prototypes are visited in authoring order.
 */
export async function buildShadoPhysicsPack(
  options: BuildShadoPhysicsPackOptions
): Promise<{ pack: ShadoPhysicsPack; stats: ShadoPhysicsPackStats }> {
  const world = await importWorldPrimitives(
    options.glb,
    worldGlbPrimitivePolicies(options.glb),
    options.inputTransform ?? 'identity',
    options.inputRightHanded ?? true,
    options.authoring,
    false
  );
  const stamped: StampedCollisionInstance[] = [];
  await importStampedObjectGeometry(options.authoring, options.loadObjectAsset, stamped);
  const BABYLON = await import('@babylonjs/core');

  const triangles = (primitives: readonly ShadoWorldPrimitive[]) =>
    primitives.reduce((total, primitive) => total + primitive.indices.length / 3, 0);
  let sourceTriangles = triangles(world.collision);
  for (const instance of stamped) sourceTriangles += triangles(instance.primitives);

  const meshes: ShadoPhysicsPackMesh[] = [];
  const prototypeMesh = new Map<string, { mesh: number; min: number[]; max: number[] } | null>();
  const meshFor = (instance: StampedCollisionInstance) => {
    if (prototypeMesh.has(instance.prototypeId)) return prototypeMesh.get(instance.prototypeId)!;
    const merged = mergePrimitives(instance.primitives);
    let entry: { mesh: number; min: number[]; max: number[] } | null = null;
    if (merged.indices.length) {
      meshes.push(merged);
      entry = { mesh: meshes.length - 1, ...boundsOf(merged.positions) };
    }
    prototypeMesh.set(instance.prototypeId, entry);
    return entry;
  };

  const instances: ShadoPhysicsPackInstance[] = [];
  const instanceChunks = new Map<string, number[]>();
  const flattened: ShadoWorldPrimitive[] = [];
  let flattenedInstances = 0;
  for (const stamp of stamped) {
    const entry = meshFor(stamp);
    if (!entry) continue;
    const rotation = BABYLON.Quaternion.RotationYawPitchRoll(
      BABYLON.Tools.ToRadians(stamp.rotationDegrees[1]),
      BABYLON.Tools.ToRadians(stamp.rotationDegrees[0]),
      BABYLON.Tools.ToRadians(stamp.rotationDegrees[2])
    );
    const matrix = BABYLON.Matrix.Compose(
      BABYLON.Vector3.FromArray(stamp.scale),
      rotation,
      BABYLON.Vector3.FromArray(stamp.position)
    ).m;
    // A Havok container child cannot mirror: negative or zero scale would
    // turn the shape inside out. Those few stamps join the world soup instead.
    if (!stamp.scale.every(component => component > 0)) {
      const mesh = meshes[entry.mesh]!;
      flattened.push({
        name: `physics-flattened:${flattenedInstances}`,
        material: '',
        collisionFlags: STAMP_FLAGS,
        positions: transformPositions(mesh.positions, matrix),
        indices: mesh.indices,
      });
      flattenedInstances++;
      continue;
    }
    const id = instances.length;
    instances.push({
      mesh: entry.mesh,
      translation: [...stamp.position],
      rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
      scale: [...stamp.scale],
    });
    // Every chunk the placed bounds overlap, matching the artifact's rule for
    // triangles: a stamp across a seam is solid from either side.
    const worldBounds = boundsOf(transformPositions(corners(entry.min, entry.max), matrix));
    for (const key of chunkKeysOver(worldBounds, options.chunkSize)) {
      const list = instanceChunks.get(key) ?? [];
      list.push(id);
      instanceChunks.set(key, list);
    }
  }

  const soupPrimitives = [...world.collision, ...flattened];
  const soupChunks: ShadoWorldCollisionChunk[] = soupPrimitives.some(primitive => primitive.indices.length)
    ? encodeShadoWorldCollision(soupPrimitives, { chunkSize: options.chunkSize }).chunks
    : [];
  const soupByKey = new Map(soupChunks.map(chunk => [`${chunk.x},${chunk.z}`, chunk]));

  const keys = [...new Set([...soupByKey.keys(), ...instanceChunks.keys()])].sort(compareChunkKeys);
  const chunks: ShadoPhysicsPackChunk[] = [];
  let soupTriangles = 0;
  for (const key of keys) {
    const [x, z] = key.split(',').map(Number) as [number, number];
    const soup = soupByKey.get(key);
    let soupMesh = -1;
    if (soup) {
      meshes.push({ positions: soup.positions, indices: soup.indices });
      soupMesh = meshes.length - 1;
      soupTriangles += soup.indices.length / 3;
    }
    // Morton order within the chunk: the runtime groups consecutive
    // instances into small containers (Havok's container build is quadratic
    // in its child count), and neighbours in this order are neighbours in
    // space, so each group's bounds stay tight for the broadphase.
    const refs = (instanceChunks.get(key) ?? []).sort(
      (left, right) =>
        morton(instances[left]!.translation, x, z, options.chunkSize) -
          morton(instances[right]!.translation, x, z, options.chunkSize) || left - right
    );
    chunks.push({
      x,
      z,
      flags: (soup?.flags ?? 0) | (refs.length ? STAMP_FLAGS : 0),
      soupMesh,
      instances: Uint32Array.from(refs),
    });
  }

  let prototypeTriangles = 0;
  let prototypes = 0;
  for (const entry of prototypeMesh.values()) {
    if (!entry) continue;
    prototypes++;
    prototypeTriangles += meshes[entry.mesh]!.indices.length / 3;
  }
  return {
    pack: { chunkSize: options.chunkSize, sourceHash: options.sourceHash, meshes, instances, chunks },
    stats: {
      sourceTriangles,
      soupTriangles,
      prototypeTriangles,
      prototypes,
      instances: instances.length,
      flattenedInstances,
      builtTriangles: soupTriangles + prototypeTriangles,
    },
  };
}

function mergePrimitives(primitives: readonly ShadoWorldPrimitive[]): ShadoPhysicsPackMesh {
  let vertexCount = 0;
  let indexCount = 0;
  for (const primitive of primitives) {
    vertexCount += primitive.positions.length / 3;
    indexCount += primitive.indices.length;
  }
  const positions = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(indexCount);
  let vertex = 0;
  let index = 0;
  for (const primitive of primitives) {
    positions.set(primitive.positions as ArrayLike<number>, vertex * 3);
    for (let i = 0; i < primitive.indices.length; i++) {
      indices[index++] = Number(primitive.indices[i]) + vertex;
    }
    vertex += primitive.positions.length / 3;
  }
  return { positions, indices };
}

function transformPositions(positions: ArrayLike<number>, m: ArrayLike<number>): Float32Array {
  const out = new Float32Array(positions.length);
  for (let offset = 0; offset < positions.length; offset += 3) {
    const x = positions[offset]!;
    const y = positions[offset + 1]!;
    const z = positions[offset + 2]!;
    out[offset] = x * m[0]! + y * m[4]! + z * m[8]! + m[12]!;
    out[offset + 1] = x * m[1]! + y * m[5]! + z * m[9]! + m[13]!;
    out[offset + 2] = x * m[2]! + y * m[6]! + z * m[10]! + m[14]!;
  }
  return out;
}

function boundsOf(positions: ArrayLike<number>): { min: number[]; max: number[] } {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let offset = 0; offset < positions.length; offset += 3) {
    for (let axis = 0; axis < 3; axis++) {
      const value = positions[offset + axis]!;
      if (value < min[axis]!) min[axis] = value;
      if (value > max[axis]!) max[axis] = value;
    }
  }
  return { min, max };
}

function corners(min: number[], max: number[]): Float32Array {
  const out: number[] = [];
  for (const x of [min[0]!, max[0]!]) {
    for (const y of [min[1]!, max[1]!]) for (const z of [min[2]!, max[2]!]) out.push(x, y, z);
  }
  return Float32Array.from(out);
}

function chunkKeysOver(bounds: { min: number[]; max: number[] }, chunkSize: number): string[] {
  const keys: string[] = [];
  for (let z = Math.floor(bounds.min[2]! / chunkSize); z <= Math.floor(bounds.max[2]! / chunkSize); z++) {
    for (let x = Math.floor(bounds.min[0]! / chunkSize); x <= Math.floor(bounds.max[0]! / chunkSize); x++) {
      keys.push(`${x},${z}`);
    }
  }
  return keys;
}

/** 16-bit-per-axis Z-order of a point's XZ position inside its chunk. */
function morton(translation: readonly number[], chunkX: number, chunkZ: number, chunkSize: number): number {
  const cell = (value: number, origin: number) =>
    Math.max(0, Math.min(0xffff, Math.floor(((value - origin * chunkSize) / chunkSize) * 0xffff)));
  const spread = (value: number) => {
    let v = value;
    v = (v | (v << 8)) & 0x00ff00ff;
    v = (v | (v << 4)) & 0x0f0f0f0f;
    v = (v | (v << 2)) & 0x33333333;
    return (v | (v << 1)) & 0x55555555;
  };
  return (spread(cell(translation[0]!, chunkX)) | (spread(cell(translation[2]!, chunkZ)) << 1)) >>> 0;
}

function compareChunkKeys(left: string, right: string): number {
  const [lx, lz] = left.split(',').map(Number) as [number, number];
  const [rx, rz] = right.split(',').map(Number) as [number, number];
  return lz - rz || lx - rx;
}
