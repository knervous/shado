import { BABYLON, type Mesh } from '../../babylon';

/**
 * The slice of Babylon's `Mesh` this API touches, declared structurally.
 *
 * Deliberately not `Mesh`. A consumer resolves `@babylonjs/core` from its own
 * node_modules, and TypeScript compares those class types nominally, so a
 * published signature naming Babylon's `Mesh` fails to accept the caller's own
 * meshes whenever the two installs are distinct copies. Structural typing keeps
 * the surface checked without dragging this package's Babylon into the
 * consumer's type graph.
 */
export type ShadoModuleMesh = {
  name: string;
  isVisible: boolean;
  alwaysSelectAsActiveMesh: boolean;
  material: any;
  thinInstanceCount: number;
  getTotalVertices(): number;
  getVerticesData(kind: string): Float32Array | number[] | null;
  setVerticesData(kind: string, data: any, updatable?: boolean, stride?: number): void;
  computeWorldMatrix(force?: boolean): { determinant(): number };
  flipFaces(flipNormals?: boolean): void;
  thinInstanceRegisterAttribute(kind: string, stride: number): void;
  thinInstanceAdd(matrix: any, refresh?: boolean): number;
  thinInstanceSetMatrixAt(index: number, matrix: any, refresh?: boolean): void;
  dispose(): void;
};

/**
 * A custom per-vertex stream that must survive a merge.
 *
 * `Mesh.MergeMeshes` rebuilds vertex data through `VertexData.ExtractFromMesh`,
 * which only knows Babylon's own vertex kinds. Anything an application stamps
 * itself - an atlas page, a submesh ordinal, an equipment slot - is silently
 * dropped, so it has to be concatenated back by hand in merge order.
 */
export type ShadoPreservedAttribute = {
  /** Vertex-buffer kind, as passed to `setVerticesData`. */
  kind: string;
  /** Floats per vertex. */
  stride: number;
};

/** One draw owner: the meshes of a single group, merged. */
export type ShadoModuleGeometry = {
  /** Group key this module was produced from. */
  key: string;
  /** Indices into the source array that were merged, in merge order. */
  sourceIndices: number[];
  mesh: ShadoModuleMesh;
};

export type ShadoModuleSplitOptions = {
  /**
   * Group key for a source mesh. Meshes sharing a key are merged into one
   * module; returning a constant collapses the whole model into the single
   * merge a supermesh already does, which is the useful no-op baseline.
   */
  groupKey: (mesh: ShadoModuleMesh, index: number) => string;
  /** Custom vertex streams to concatenate back onto each merged module. */
  preserveAttributes?: readonly ShadoPreservedAttribute[];
  /** Passed through to MergeMeshes. Leave false to keep 16-bit indices. */
  allow32BitsIndices?: boolean;
  /** Names the merged mesh. Defaults to the group key. */
  name?: (key: string) => string;
  /** Called instead of throwing when a group fails to merge. */
  onMergeFailed?: (key: string, sources: ShadoModuleMesh[]) => void;
};

/**
 * Splits a variant supermesh into per-group module meshes.
 *
 * The point of the split is draw ownership: a model that ships every equipment
 * variant as its own submesh and hides the unworn ones per instance still
 * skins every variant for every instance. Merging per group instead lets each
 * module draw only the actors that actually show it - see
 * `ShadoModuleDrawSet`, which owns that half.
 *
 * Source meshes are disposed by the merge, so read anything you still need
 * from them before calling this.
 */
export function splitMeshesIntoModules(
  meshes: readonly ShadoModuleMesh[],
  options: ShadoModuleSplitOptions
): ShadoModuleGeometry[] {
  type Group = { key: string; sourceIndices: number[]; sources: ShadoModuleMesh[] };
  const groups: Group[] = [];
  const byKey = new Map<string, Group>();
  for (let index = 0; index < meshes.length; index++) {
    const mesh = meshes[index];
    if (!mesh || mesh.getTotalVertices() <= 0) continue;
    const key = options.groupKey(mesh, index);
    let group = byKey.get(key);
    if (!group) {
      group = { key, sourceIndices: [], sources: [] };
      byKey.set(key, group);
      groups.push(group);
    }
    group.sourceIndices.push(index);
    group.sources.push(mesh);
  }

  const preserved = options.preserveAttributes ?? [];
  const modules: ShadoModuleGeometry[] = [];
  for (const group of groups) {
    // Read the custom streams and the reflection state before merging: the
    // merge disposes its sources.
    const captured = preserved.map((attribute) => ({
      attribute,
      buffers: group.sources.map(
        (source) =>
          (source.getVerticesData(attribute.kind) as Float32Array | null) ??
          new Float32Array((source.getTotalVertices() || 0) * attribute.stride)
      ),
    }));
    const reflected = group.sources[0].computeWorldMatrix(true).determinant() < 0;

    const merged = BABYLON.Mesh.MergeMeshes(
      group.sources as unknown as Mesh[],
      true,
      options.allow32BitsIndices ?? false,
      undefined,
      false,
      false
    );
    if (!merged) {
      options.onMergeFailed?.(group.key, group.sources);
      continue;
    }
    repairSingleSourceMergeWinding(group.sources.length, reflected, merged);
    merged.name = options.name?.(group.key) ?? group.key;
    for (const { attribute, buffers } of captured) {
      concatenateVertexAttribute(merged, attribute, buffers);
    }
    modules.push({
      key: group.key,
      sourceIndices: group.sourceIndices,
      mesh: merged,
    });
  }
  return modules;
}

/**
 * Restores the winding flip `MergeMeshes` skips for a one-mesh group.
 *
 * MergeMeshes reverses the winding of a reflected source, but for its *root*
 * mesh it only does so inside the branch that resizes the index buffer to fit
 * the meshes being appended:
 *
 * ```js
 * if (indices.length !== totalIndices) {   // false when there are no others
 *   ...resize...
 *   if (transform.determinant() < 0) _FlipFaces(indices, 0, indicesOffset);
 * }
 * for (const { transform } of others) {    // every other mesh, unconditional
 *   if (transform.determinant() < 0) _FlipFaces(indices, offset, count);
 * }
 * ```
 *
 * A group of exactly one mesh therefore keeps the wrong winding. This is
 * invisible while a whole model merges as one supermesh - there are always
 * other meshes to force the branch - and appears the moment a split produces a
 * single-submesh module. Backface culling then draws the inside of the part,
 * whose normals face away from the light, so it reads as inside-out *and*
 * unlit. Winding only: the multi-mesh path leaves normals alone too.
 */
export function repairSingleSourceMergeWinding(
  sourceCount: number,
  reflected: boolean,
  merged: ShadoModuleMesh
): void {
  if (sourceCount !== 1 || !reflected) return;
  merged.flipFaces(false);
}

/** Concatenates one custom stream from the sources onto the merged mesh. */
function concatenateVertexAttribute(
  merged: ShadoModuleMesh,
  attribute: ShadoPreservedAttribute,
  buffers: readonly Float32Array[]
): void {
  const stride = attribute.stride;
  const vertexCount = buffers.reduce((total, buffer) => total + buffer.length / stride, 0);
  if (!vertexCount) return;
  const data = new Float32Array(vertexCount * stride);
  let offset = 0;
  for (const buffer of buffers) {
    data.set(buffer, offset * stride);
    offset += buffer.length / stride;
  }
  merged.setVerticesData(attribute.kind, data, false, stride);
}
