import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { NullEngine, Scene, MeshBuilder, Mesh } from '@babylonjs/core';

const { MergeMeshes } = Mesh;
import {
  ShadoModuleDrawSet,
  splitMeshesIntoModules,
} from '../src/extensions/ShadoModuleDraw';

function wardrobe(scene: Scene, pieces: string[], variants: number) {
  const parts: { piece: string; variant: string; mesh: any }[] = [];
  for (const piece of pieces) {
    for (let variant = 0; variant < variants; variant++) {
      const mesh = MeshBuilder.CreateBox(`${piece}_${variant}`, { size: 1 }, scene);
      // A per-vertex stream Babylon's merge does not know about.
      const count = mesh.getTotalVertices();
      const stamp = new Float32Array(count * 2);
      for (let v = 0; v < count; v++) {
        stamp[v * 2] = parts.length;
        stamp[v * 2 + 1] = variant;
      }
      mesh.setVerticesData('submeshData', stamp, false, 2);
      parts.push({ piece, variant: String(variant), mesh });
    }
  }
  return parts;
}

describe('variant supermesh module draws', () => {
  let engine: NullEngine;
  let scene: Scene;

  beforeEach(() => {
    engine = new NullEngine();
    scene = new Scene(engine);
  });

  afterEach(() => {
    scene.dispose();
    engine.dispose();
  });

  it('groups submeshes into one module per key and preserves custom streams', () => {
    const parts = wardrobe(scene, ['chest', 'legs'], 3);
    const perMeshVertices = parts[0].mesh.getTotalVertices();

    const modules = splitMeshesIntoModules(
      parts.map(p => p.mesh),
      {
        groupKey: (_mesh, index) => `${parts[index].piece}:${parts[index].variant}`,
        preserveAttributes: [{ kind: 'submeshData', stride: 2 }],
      }
    );

    expect(modules).toHaveLength(6);
    expect(modules.map(m => m.key)).toEqual([
      'chest:0', 'chest:1', 'chest:2', 'legs:0', 'legs:1', 'legs:2',
    ]);
    for (const module of modules) {
      expect(module.sourceIndices).toHaveLength(1);
      expect(module.mesh.getTotalVertices()).toBe(perMeshVertices);
      // The stream survived the merge, at the right stride and length.
      const stream = module.mesh.getVerticesData('submeshData') as Float32Array;
      expect(stream).toBeTruthy();
      expect(stream.length).toBe(perMeshVertices * 2);
      expect(stream[0]).toBe(module.sourceIndices[0]);
    }
  });

  it('repairs the winding MergeMeshes skips for a reflected one-mesh group', () => {
    // Every humanoid rig node measures a negative determinant. MergeMeshes
    // flips a reflected source's winding for its root mesh only while resizing
    // the index buffer for the meshes being appended, so a one-mesh group keeps
    // the wrong winding - invisible in a whole-model merge, and the reason a
    // single-submesh module drew inside-out and unlit.
    const windingMatchesNormals = (mesh: any) => {
      const positions = mesh.getVerticesData('position') as Float32Array;
      const normals = mesh.getVerticesData('normal') as Float32Array;
      const indices = mesh.getIndices() as number[];
      let match = 0;
      let total = 0;
      for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i] * 3;
        const b = indices[i + 1] * 3;
        const c = indices[i + 2] * 3;
        const e1 = [
          positions[b] - positions[a],
          positions[b + 1] - positions[a + 1],
          positions[b + 2] - positions[a + 2],
        ];
        const e2 = [
          positions[c] - positions[a],
          positions[c + 1] - positions[a + 1],
          positions[c + 2] - positions[a + 2],
        ];
        const face = [
          e1[1] * e2[2] - e1[2] * e2[1],
          e1[2] * e2[0] - e1[0] * e2[2],
          e1[0] * e2[1] - e1[1] * e2[0],
        ];
        const vertex = [
          (normals[a] + normals[b] + normals[c]) / 3,
          (normals[a + 1] + normals[b + 1] + normals[c + 1]) / 3,
          (normals[a + 2] + normals[b + 2] + normals[c + 2]) / 3,
        ];
        const dot = face[0] * vertex[0] + face[1] * vertex[1] + face[2] * vertex[2];
        if (Math.abs(dot) > 1e-12) {
          total++;
          if (dot > 0) match++;
        }
      }
      return match / total;
    };

    const reflect = (mesh: any) => {
      mesh.scaling.x = -1;
      mesh.computeWorldMatrix(true);
      return mesh;
    };

    // Control: what Babylon does to a lone reflected mesh with no repair.
    const control = MergeMeshes(
      [reflect(MeshBuilder.CreateBox('control', { size: 1 }, scene))],
      true,
      false,
      undefined,
      false,
      false
    )!;
    const unrepaired = windingMatchesNormals(control);

    const lone = reflect(MeshBuilder.CreateBox('lone', { size: 1 }, scene));
    const pairA = reflect(MeshBuilder.CreateBox('pairA', { size: 1 }, scene));
    const pairB = reflect(MeshBuilder.CreateBox('pairB', { size: 1 }, scene));

    const modules = splitMeshesIntoModules([lone, pairA, pairB], {
      groupKey: (_mesh, index) => (index === 0 ? 'lone' : 'pair'),
    });

    expect(modules).toHaveLength(2);
    const single = windingMatchesNormals(modules[0].mesh);
    const multi = windingMatchesNormals(modules[1].mesh);

    // The bug: an unrepaired one-mesh merge is the exact inverse of what the
    // multi-mesh merge produces. Every triangle disagrees, not merely some.
    expect(unrepaired).toBeCloseTo(1 - multi, 5);
    // The repair: the one-mesh module now agrees with the multi-mesh one.
    expect(single).toBeCloseTo(multi, 5);
  });

  it('collapses to the original supermesh when the group key is constant', () => {
    const parts = wardrobe(scene, ['chest', 'legs'], 3);
    const total = parts.reduce((n, p) => n + p.mesh.getTotalVertices(), 0);

    const modules = splitMeshesIntoModules(
      parts.map(p => p.mesh),
      { groupKey: () => 'all', preserveAttributes: [{ kind: 'submeshData', stride: 2 }] }
    );

    expect(modules).toHaveLength(1);
    expect(modules[0].mesh.getTotalVertices()).toBe(total);
    expect(modules[0].sourceIndices).toHaveLength(6);
    expect(
      (modules[0].mesh.getVerticesData('submeshData') as Float32Array).length
    ).toBe(total * 2);
  });

  it('draws only the actors that wear each module, and nothing when unworn', () => {
    const pieces = ['chest', 'legs'];
    const variants = 3;
    const parts = wardrobe(scene, pieces, variants);
    const modules = splitMeshesIntoModules(
      parts.map(p => p.mesh),
      { groupKey: (_m, i) => `${parts[i].piece}:${parts[i].variant}` }
    );
    const moduleParts = modules.map(m => parts[m.sourceIndices[0]]);
    const draws = new ShadoModuleDrawSet(engine, modules);

    const actorCount = 9;
    const visible = new Uint32Array(actorCount);
    for (let actor = 0; actor < actorCount; actor++) visible[actor] = actor;
    // Actor n wears variant n % 3 of every piece.
    const worn = (actor: number, moduleIndex: number) =>
      moduleParts[moduleIndex].variant === String(actor % variants);

    const stats = draws.refresh(visible, worn);

    // The invariant: per piece, the module draw counts sum to the visible count.
    for (const piece of pieces) {
      const drawn = draws.modules
        .filter((_m, i) => moduleParts[i].piece === piece)
        .reduce((sum, m) => sum + m.drawnCount, 0);
      expect(drawn).toBe(actorCount);
    }
    expect(stats.visibleActors).toBe(actorCount);
    expect(stats.populatedModules).toBe(pieces.length * variants);
    // Each actor shows 1 of 3 variants per piece, so a third of the work.
    expect(stats.submittedVertices * 3).toBe(stats.baselineVertices);
    expect(stats.vertexWorkReduction).toBeCloseTo(3, 5);

    // Nobody wears variant 2 any more: that bucket empties and switches off.
    const narrowed = draws.refresh(
      visible,
      (actor, moduleIndex) =>
        moduleParts[moduleIndex].variant === String(actor % 2)
    );
    for (const [index, module] of draws.modules.entries()) {
      if (moduleParts[index].variant !== '2') continue;
      expect(module.drawnCount).toBe(0);
      expect(module.mesh.isVisible).toBe(false);
    }
    expect(narrowed.populatedModules).toBe(pieces.length * 2);
  });

  it('does not re-upload a bucket whose actor list is unchanged', () => {
    const parts = wardrobe(scene, ['chest'], 2);
    const modules = splitMeshesIntoModules(
      parts.map(p => p.mesh),
      { groupKey: (_m, i) => parts[i].variant }
    );
    const draws = new ShadoModuleDrawSet(engine, modules);
    const visible = new Uint32Array([0, 1, 2]);

    const selection = draws.modules[0].selection!;
    const setActorIndices = jest.spyOn(selection, 'setActorIndices');
    const member = (_actor: number, moduleIndex: number) => moduleIndex === 0;

    draws.refresh(visible, member);
    expect(setActorIndices).toHaveBeenCalledTimes(1);
    draws.refresh(visible, member);
    draws.refresh(visible, member);
    expect(setActorIndices).toHaveBeenCalledTimes(1);

    // A changed bucket does upload again.
    draws.refresh(visible, (actor, moduleIndex) => moduleIndex === 0 && actor !== 1);
    expect(setActorIndices).toHaveBeenCalledTimes(2);
  });

  it('leaves a single module unsplit, drawing every visible actor', () => {
    const parts = wardrobe(scene, ['chest'], 2);
    const modules = splitMeshesIntoModules(
      parts.map(p => p.mesh),
      { groupKey: () => 'all' }
    );
    const draws = new ShadoModuleDrawSet(engine, modules);

    expect(draws.isSplit).toBe(false);
    expect(draws.modules[0].selection).toBeNull();

    const stats = draws.refresh(new Uint32Array([0, 1, 2, 3]), () => false);
    // Membership is not consulted: an unsplit model has nothing to select on.
    expect(draws.modules[0].drawnCount).toBe(4);
    expect(stats.vertexWorkReduction).toBe(1);
    expect(stats.submittedVertices).toBe(stats.baselineVertices);
  });
});
