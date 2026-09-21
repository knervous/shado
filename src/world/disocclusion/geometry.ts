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
  // Sidedness as the primitive declares it; undeclared means double-sided.
  const doubleSided = new Uint8Array(indexCount / 3);
  let at = 0;
  for (let c = 0; c < clusterCount; c++) {
    const primitive = world.clusters.primitive[c]!;
    const base = offsets[primitive]!;
    const first = world.clusters.firstIndex[c]!;
    const count = world.clusters.indexCount[c]!;
    for (let k = 0; k < count; k++) indices[at + k] = base + world.clusterIndices[first + k]!;
    triangleTarget.fill(c, at / 3, (at + count) / 3);
    doubleSided.fill(primitives[primitive]!.doubleSided === false ? 0 : 1, at / 3, (at + count) / 3);
    at += count;
  }
  return { positions, indices, triangleTarget, doubleSided };
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

/** What a zone bake put in the raster and why the rest stayed out. */
export type ZoneBakeGeometryManifest = {
  targets: { clusters: number; mappedPrimitives: number; primitives: number; triangles: number; blockerTriangles: number };
  blockers: { baseExtraTriangles: number; objectTriangles: number; total: number };
  sidedness: { doubleSided: number; singleSided: number };
  /** Base GLB parts left out of the raster, by reason (they may still be targets). */
  baseExclusions: Record<string, number>;
  /** Clusters whose recovered triangles do not sit inside the package's cluster sphere. */
  frameMismatches: number;
  /**
   * Those clusters, named (pvs.md V3). Their triangles are NOT blockers (the
   * recovered geometry disagrees with the package, so it cannot be trusted to
   * hide anything) and the bake admits them in every row.
   */
  suspectClusters: Array<{
    cluster: number;
    primitive: string;
    triangles: number;
    center: [number, number, number];
    radius: number;
    /** Farthest recovered vertex from the package centre, in radii. */
    worstDistanceRadii: number;
  }>;
};

type GlbPart = {
  node: string;
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
  doubleSided: boolean;
  /** Null when the part may block (see occluderEligibility). */
  exclusion: string | null;
};

/**
 * Bake input for a real zone.
 *
 * Targets are the package's render clusters, exactly as the scene layer
 * draws them: package primitive `N#k` is primitive k of GLB node `N` (the
 * runtime strips `#k` to find the same mesh), and cluster indices address its
 * vertices. Blockers are every eligible static opaque surface: cluster
 * triangles of eligible parts, the GLB's other eligible parts (drawn through
 * merged geometry, never PVS targets) and eligible placed objects. Placed
 * objects and extra parts are blockers only (target -1): objects keep
 * reference admission.
 */
export function zoneBakeGeometry(
  world: ShadoWorldSpatialPackage,
  glbParts: readonly GlbPart[],
  objectPrimitives: readonly ShadoWorldPrimitive[]
): { geometry: DisocclusionGeometry; manifest: ZoneBakeGeometryManifest } {
  const byNode = new Map<string, GlbPart>();
  for (const part of glbParts) byNode.set(part.node.slice(part.node.lastIndexOf('/') + 1), part);
  const matched = world.primitives.map(primitive => {
    // `N#k`: the runtime strips `#k` to name the Babylon mesh. Babylon splits a
    // multi-primitive glTF mesh into `N_primitiveI` meshes, which
    // readGlbPrimitives names `N#I`.
    const hash = primitive.name.lastIndexOf('#');
    const mesh = hash >= 0 ? primitive.name.slice(0, hash) : primitive.name;
    const split = /^(.*)_primitive(\d+)$/.exec(mesh);
    const part =
      byNode.get(mesh) ?? (split ? byNode.get(`${split[1]}#${split[2]}`) : undefined) ?? byNode.get(`${mesh}#0`) ?? null;
    if (part && part.positions.length / 3 !== primitive.vertexCount) {
      throw new Error(`GLB part ${part.node} has ${part.positions.length / 3} vertices, package primitive ${primitive.name} ${primitive.vertexCount}`);
    }
    return part;
  });
  const matchedParts = new Set(matched.filter((p): p is GlbPart => p !== null));
  const unmapped = world.primitives.filter((_, i) => !matched[i]).map(p => p.name);
  if (unmapped.length) throw new Error(`package primitives with no GLB part: ${unmapped.slice(0, 5).join(', ')}`);

  const clusterCount = world.clusters.firstIndex.length;
  let targetTriangles = 0;
  for (let c = 0; c < clusterCount; c++) targetTriangles += world.clusters.indexCount[c]! / 3;
  const baseExtra = glbParts.filter(p => !matchedParts.has(p) && p.exclusion === null);
  const baseExclusions: Record<string, number> = {};
  for (const part of glbParts) {
    if (part.exclusion) baseExclusions[part.exclusion] = (baseExclusions[part.exclusion] ?? 0) + part.indices.length / 3;
  }
  const extraTriangles = baseExtra.reduce((s, p) => s + p.indices.length / 3, 0);
  const objectTriangles = objectPrimitives.reduce((s, p) => s + p.indices.length / 3, 0);
  const vertexCount =
    [...matchedParts].reduce((s, p) => s + p.positions.length / 3, 0) +
    baseExtra.reduce((s, p) => s + p.positions.length / 3, 0) +
    objectPrimitives.reduce((s, p) => s + p.positions.length / 3, 0);
  const triangles = targetTriangles + extraTriangles + objectTriangles;
  const positions = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(triangles * 3);
  const triangleTarget = new Int32Array(triangles).fill(-1);
  const blocker = new Uint8Array(triangles);
  const doubleSided = new Uint8Array(triangles);
  let vertexAt = 0;
  let triangleAt = 0;
  const partBase = new Map<GlbPart, number>();
  const pushVertices = (source: ArrayLike<number>) => {
    const base = vertexAt;
    for (let i = 0; i < source.length; i++) positions[base * 3 + i] = Number(source[i]);
    vertexAt += source.length / 3;
    return base;
  };
  for (const part of matchedParts) partBase.set(part, pushVertices(part.positions));

  let blockerTargetTriangles = 0;
  let frameMismatches = 0;
  const suspectClusters: ZoneBakeGeometryManifest['suspectClusters'] = [];
  // Recover and check every cluster's frame first: a suspect cluster must not
  // contribute a single blocker triangle.
  const suspect = new Uint8Array(clusterCount);
  for (let c = 0; c < clusterCount; c++) {
    const part = matched[world.clusters.primitive[c]!]!;
    const base = partBase.get(part)!;
    const first = world.clusters.firstIndex[c]!;
    const count = world.clusters.indexCount[c]!;
    const cx = world.clusters.centerX[c]!;
    const cy = world.clusters.centerY[c]!;
    const cz = world.clusters.centerZ[c]!;
    const radius = world.clusters.radius[c]!;
    const r2 = (radius * 1.01 + 0.01) ** 2;
    let worst2 = 0;
    for (let k = 0; k < count; k++) {
      const v = base + world.clusterIndices[first + k]!;
      const dx = positions[v * 3]! - cx;
      const dy = positions[v * 3 + 1]! - cy;
      const dz = positions[v * 3 + 2]! - cz;
      worst2 = Math.max(worst2, dx * dx + dy * dy + dz * dz);
    }
    if (worst2 > r2) {
      suspect[c] = 1;
      frameMismatches++;
      suspectClusters.push({
        cluster: c,
        primitive: world.primitives[world.clusters.primitive[c]!]?.name ?? '?',
        triangles: count / 3,
        center: [cx, cy, cz],
        radius,
        worstDistanceRadii: Math.sqrt(worst2) / Math.max(radius, 1e-6),
      });
    }
  }
  for (let c = 0; c < clusterCount; c++) {
    const part = matched[world.clusters.primitive[c]!]!;
    const base = partBase.get(part)!;
    const first = world.clusters.firstIndex[c]!;
    const count = world.clusters.indexCount[c]!;
    const blocks = part.exclusion === null && !suspect[c];
    for (let k = 0; k < count; k++) {
      indices[triangleAt * 3 + (k % 3)] = base + world.clusterIndices[first + k]!;
      if (k % 3 === 2) {
        triangleTarget[triangleAt] = c;
        blocker[triangleAt] = blocks ? 1 : 0;
        doubleSided[triangleAt] = part.doubleSided ? 1 : 0;
        if (blocks) blockerTargetTriangles++;
        triangleAt++;
      }
    }
  }
  const pushBlocker = (source: { positions: ArrayLike<number>; indices: ArrayLike<number>; doubleSided?: boolean }) => {
    const base = pushVertices(source.positions);
    const two = source.doubleSided === false ? 0 : 1;
    for (let i = 0; i < source.indices.length; i += 3) {
      indices[triangleAt * 3] = base + Number(source.indices[i]);
      indices[triangleAt * 3 + 1] = base + Number(source.indices[i + 1]);
      indices[triangleAt * 3 + 2] = base + Number(source.indices[i + 2]);
      blocker[triangleAt] = 1;
      doubleSided[triangleAt] = two;
      triangleAt++;
    }
  };
  for (const part of baseExtra) pushBlocker(part);
  for (const primitive of objectPrimitives) pushBlocker(primitive);

  let two = 0;
  for (let t = 0; t < triangles; t++) if (blocker[t]) two += doubleSided[t]!;
  const blockerTotal = blockerTargetTriangles + extraTriangles + objectTriangles;
  // GLB parts carry vertices no cluster references; keep only used ones.
  const remap = new Int32Array(vertexCount).fill(-1);
  let used = 0;
  for (let i = 0; i < indices.length; i++) {
    const v = indices[i]!;
    if (remap[v] < 0) remap[v] = used++;
    indices[i] = remap[v]!;
  }
  const compact = new Float32Array(used * 3);
  for (let v = 0; v < vertexCount; v++) {
    const to = remap[v]!;
    if (to < 0) continue;
    compact[to * 3] = positions[v * 3]!;
    compact[to * 3 + 1] = positions[v * 3 + 1]!;
    compact[to * 3 + 2] = positions[v * 3 + 2]!;
  }
  return {
    geometry: { positions: compact, indices, triangleTarget, blocker, doubleSided },
    manifest: {
      targets: {
        clusters: clusterCount,
        mappedPrimitives: matchedParts.size,
        primitives: world.primitives.length,
        triangles: targetTriangles,
        blockerTriangles: blockerTargetTriangles,
      },
      blockers: { baseExtraTriangles: extraTriangles, objectTriangles, total: blockerTotal },
      sidedness: { doubleSided: two, singleSided: blockerTotal - two },
      baseExclusions,
      frameMismatches,
      suspectClusters,
    },
  };
}
