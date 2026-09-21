/**
 * Bake input from a compiled world: every render cluster's triangles, tagged
 * with the cluster as their target. Cluster -> cell -> region is exactly how
 * the reducer decides whether a cluster draws, so admitting a cluster's
 * region is admitting that cluster.
 */
import type { ShadoWorldPrimitive, ShadoWorldSpatialPackage } from '../types';
import type { DisocclusionGeometry } from './types';

export function geometryFromWorld(
  world: ShadoWorldSpatialPackage,
  primitives: readonly ShadoWorldPrimitive[]
): DisocclusionGeometry {
  if (primitives.length !== world.primitives.length) {
    throw new Error(`world has ${world.primitives.length} primitives, ${primitives.length} supplied`);
  }
  const offsets: number[] = [];
  let vertices = 0;
  primitives.forEach((primitive, i) => {
    const count = primitive.positions.length / 3;
    if (count !== world.primitives[i]!.vertexCount) {
      throw new Error(`primitive ${i} '${primitive.name}' has ${count} vertices, package says ${world.primitives[i]!.vertexCount}`);
    }
    offsets.push(vertices);
    vertices += count;
  });
  const positions = new Float32Array(vertices * 3);
  primitives.forEach((primitive, i) => {
    for (let k = 0; k < primitive.positions.length; k++) positions[offsets[i]! * 3 + k] = Number(primitive.positions[k]);
  });
  const clusterCount = world.clusters.firstIndex.length;
  let indexCount = 0;
  for (let c = 0; c < clusterCount; c++) indexCount += world.clusters.indexCount[c]!;
  const indices = new Uint32Array(indexCount);
  const triangleTarget = new Int32Array(indexCount / 3);
  let at = 0;
  for (let c = 0; c < clusterCount; c++) {
    const base = offsets[world.clusters.primitive[c]!]!;
    const first = world.clusters.firstIndex[c]!;
    const count = world.clusters.indexCount[c]!;
    for (let k = 0; k < count; k++) indices[at + k] = base + world.clusterIndices[first + k]!;
    triangleTarget.fill(c, at / 3, (at + count) / 3);
    at += count;
  }
  return { positions, indices, triangleTarget };
}

/** Region owning each cluster, or -1 when the package has no dense regions. */
export function clusterRegions(world: ShadoWorldSpatialPackage): Int32Array {
  const out = new Int32Array(world.clusters.cellId.length).fill(-1);
  const cellRegion = world.visibility?.cellRegion;
  if (!cellRegion) return out;
  world.clusters.cellId.forEach((cell, c) => {
    out[c] = cellRegion[cell] ?? -1;
  });
  return out;
}
