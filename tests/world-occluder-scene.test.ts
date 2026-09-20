import {
  assembleOccluderScene,
  compileShadoWorldVisibility,
  occluderEligibility,
  readGlbPrimitives,
} from '../src/world';
import type { ShadoWorldSpatialPackage } from '../src/world';

/** A GLB holding one axis-aligned quad, built by hand so the test owns every byte. */
function glb(options: {
  alphaMode?: string;
  doubleSided?: boolean;
  node?: { translation?: number[]; scale?: number[] };
  corners?: number[];
  /** Explicit triangle list; defaults to one quad over the first four corners. */
  indices?: number[];
  /** Drive the node with an animation channel. */
  animated?: boolean;
  /** Give the primitive a morph target. */
  morphed?: boolean;
  /** Attach the node to a skin. */
  skinned?: boolean;
}): Uint8Array {
  const corners = options.corners ?? [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
  const positions = new Float32Array(corners);
  const indices = new Uint16Array(
    options.indices ?? [0, 1, 2, 0, 2, 3]
  );
  const indexBytes = new Uint8Array(indices.buffer);
  const padding = (4 - (indexBytes.byteLength % 4)) % 4;
  const binary = new Uint8Array(positions.byteLength + indexBytes.byteLength + padding);
  binary.set(new Uint8Array(positions.buffer), 0);
  binary.set(indexBytes, positions.byteLength);
  const gltf = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{
      name: 'root',
      mesh: 0,
      ...(options.skinned ? { skin: 0 } : {}),
      ...(options.node ?? {}),
    }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0 },
        indices: 1,
        material: 0,
        mode: 4,
        ...(options.morphed ? { targets: [{ POSITION: 0 }] } : {}),
      }],
    }],
    materials: [{
      name: 'stone',
      ...(options.alphaMode ? { alphaMode: options.alphaMode } : {}),
      ...(options.doubleSided === undefined ? {} : { doubleSided: options.doubleSided }),
    }],
    ...(options.animated
      ? { animations: [{ channels: [{ target: { node: 0, path: 'translation' }, sampler: 0 }], samplers: [] }] }
      : {}),
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: indices.length, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
      { buffer: 0, byteOffset: positions.byteLength, byteLength: indexBytes.byteLength },
    ],
    buffers: [{ byteLength: binary.byteLength }],
  };
  const json = new TextEncoder().encode(JSON.stringify(gltf));
  const jsonPadding = (4 - (json.byteLength % 4)) % 4;
  const jsonLength = json.byteLength + jsonPadding;
  const total = 12 + 8 + jsonLength + 8 + binary.byteLength;
  const bytes = new Uint8Array(total);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.set(json, 20);
  bytes.fill(0x20, 20 + json.byteLength, 20 + jsonLength);
  view.setUint32(20 + jsonLength, binary.byteLength, true);
  view.setUint32(24 + jsonLength, 0x004e4942, true);
  bytes.set(binary, 28 + jsonLength);
  return bytes;
}

/** A package holding only what the occluder assembler reads. */
function packageWith(stamps: {
  prototype: number[];
  enabled: number[];
  phaseMask: number[];
  position: [number, number, number][];
  rotationY?: number[];
  scale?: [number, number, number][];
  radius?: number[];
}, prototypes: { id: string; source: string }[]): ShadoWorldSpatialPackage {
  const count = stamps.prototype.length;
  return {
    objects: {
      prototypes: {
        id: prototypes.map((p) => p.id),
        source: prototypes.map((p) => p.source),
        boundsRadius: prototypes.map(() => 1),
        firstStampRef: prototypes.map(() => 0),
        stampRefCount: prototypes.map(() => count),
        metadata: prototypes.map(() => ({})),
      },
      prototypeStampRefs: [],
      stamps: {
        id: Array.from({ length: count }, (_, i) => `stamp-${i}`),
        prototype: stamps.prototype,
        enabled: stamps.enabled,
        positionX: stamps.position.map((p) => p[0]),
        positionY: stamps.position.map((p) => p[1]),
        positionZ: stamps.position.map((p) => p[2]),
        rotationX: stamps.position.map(() => 0),
        rotationY: stamps.rotationY ?? stamps.position.map(() => 0),
        rotationZ: stamps.position.map(() => 0),
        scaleX: (stamps.scale ?? stamps.position.map(() => [1, 1, 1] as [number, number, number])).map((s) => s[0]),
        scaleY: (stamps.scale ?? stamps.position.map(() => [1, 1, 1] as [number, number, number])).map((s) => s[1]),
        scaleZ: (stamps.scale ?? stamps.position.map(() => [1, 1, 1] as [number, number, number])).map((s) => s[2]),
        radius: stamps.radius ?? stamps.position.map(() => 1),
        cellId: stamps.position.map(() => 0),
        phaseMask: stamps.phaseMask,
        tags: stamps.position.map(() => []),
        metadata: stamps.position.map(() => ({})),
      },
    },
  } as unknown as ShadoWorldSpatialPackage;
}

describe('reading a GLB without a renderer', () => {
  it('reads positions, indices and the material opacity that decides eligibility', () => {
    const parts = readGlbPrimitives(glb({}));
    expect(parts).toHaveLength(1);
    expect(parts[0]!.material).toBe('stone');
    expect(parts[0]!.alphaMode).toBe('OPAQUE');
    expect(Array.from(parts[0]!.indices)).toEqual([0, 1, 2, 0, 2, 3]);
    expect(Array.from(parts[0]!.positions.slice(0, 3))).toEqual([0, 0, 0]);
    expect(occluderEligibility(parts[0]!)).toBeNull();
  });

  it('applies the node transform, including a negative scale', () => {
    const parts = readGlbPrimitives(glb({ node: { translation: [10, 0, 0], scale: [-1, 1, 1] } }));
    const positions = Array.from(parts[0]!.positions);
    expect(positions.slice(0, 3)).toEqual([10, 0, 0]);
    // x = 1 mirrored to -1, then translated.
    expect(positions.slice(3, 6)).toEqual([9, 0, 0]);
  });

  it('refuses to let anything that light passes through hide the world', () => {
    expect(occluderEligibility(readGlbPrimitives(glb({ alphaMode: 'BLEND' }))[0]!))
      .toBe('alpha-blended-material');
    expect(occluderEligibility(readGlbPrimitives(glb({ alphaMode: 'MASK' }))[0]!))
      .toBe('alpha-tested-material');
  });
});

describe('assembling placed objects', () => {
  const opaque = glb({});
  const glass = glb({ alphaMode: 'BLEND' });

  it('places each stamp with the runtime transform and reports full coverage', () => {
    const world = packageWith(
      {
        prototype: [0, 0],
        enabled: [1, 1],
        phaseMask: [0xffffffff, 0xffffffff],
        position: [[0, 0, 0], [100, 0, 0]],
      },
      [{ id: 'wall', source: '/eqrequiem/objects/wall/final.glb.gz' }],
    );
    const { primitives, manifest } = assembleOccluderScene({
      world,
      loadPrototype: () => opaque,
    });
    expect(primitives).toHaveLength(2);
    expect(manifest.stamps.included).toBe(2);
    expect(manifest.triangles.placed).toBe(4);
    expect(manifest.prototypes.resolved).toBe(1);
    // The second stamp stands 100 units along x from the first.
    const first = primitives.find((p) => p.name.startsWith('stamp-0'))!;
    const second = primitives.find((p) => p.name.startsWith('stamp-1'))!;
    expect(second.positions[0]! - first.positions[0]!).toBeCloseTo(100, 5);
  });

  it('records why each thing was left out instead of dropping it silently', () => {
    const world = packageWith(
      {
        // The last stamp is the only one that reaches the missing asset: a
        // prototype whose every stamp is disabled or out of phase is not a
        // coverage gap, and the manifest should not invent one.
        prototype: [0, 1, 2, 2, 2],
        enabled: [1, 1, 0, 1, 1],
        phaseMask: [0xffffffff, 0xffffffff, 0xffffffff, 0b10, 0xffffffff],
        position: [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]],
      },
      [
        { id: 'wall', source: 'wall.glb' },
        { id: 'window', source: 'window.glb' },
        { id: 'missing', source: 'missing.glb' },
      ],
    );
    const { primitives, manifest } = assembleOccluderScene({
      world,
      activePhaseMask: 0b01,
      loadPrototype: (source) =>
        source === 'wall.glb' ? opaque : source === 'window.glb' ? glass : null,
    });
    expect(primitives).toHaveLength(1);
    expect(manifest.stamps.excluded['no-eligible-submesh']).toBe(2);
    expect(manifest.stamps.excluded['stamp-disabled']).toBe(1);
    expect(manifest.stamps.excluded['stamp-out-of-phase']).toBe(1);
    expect(manifest.prototypes.unresolved).toEqual([
      { id: 'missing', source: 'missing.glb', reason: 'missing-prototype-asset' },
    ]);
    const glassSubmesh = manifest.submeshes.find((s) => s.id === 'window')!;
    expect(glassSubmesh.eligible).toBe(false);
    expect(glassSubmesh.reason).toBe('alpha-blended-material');
  });

  it('spends a triangle budget on the largest objects and says what it dropped', () => {
    const world = packageWith(
      {
        prototype: [0, 0, 0],
        enabled: [1, 1, 1],
        phaseMask: [0xffffffff, 0xffffffff, 0xffffffff],
        position: [[0, 0, 0], [10, 0, 0], [20, 0, 0]],
        radius: [1, 50, 10],
      },
      [{ id: 'wall', source: 'wall.glb' }],
    );
    const { primitives, manifest } = assembleOccluderScene({
      world,
      loadPrototype: () => opaque,
      maxTriangles: 4,
    });
    expect(manifest.stamps.included).toBe(2);
    expect(manifest.stamps.excluded['triangle-budget']).toBe(1);
    // The two biggest survived; the one-unit prop is what went.
    const names = primitives.map((p) => p.name.split(':')[0]);
    expect(names).toContain('stamp-1');
    expect(names).toContain('stamp-2');
    expect(names).not.toContain('stamp-0');
  });
});

describe('a placed object is why something is hidden', () => {
  /** A 200-unit wall as a prototype, so a stamp of it can block a street. */
  const wallGlb = glb({
    // Two quads back to back: a thin wall drawn on both of its faces, which is
    // what a building is and what a single unpaired quad is not.
    corners: [
      0, 0, -8, 0, 0, 8, 0, 200, 8, 0, 200, -8,
      1, 0, -8, 1, 0, 8, 1, 200, 8, 1, 200, -8,
    ],
    indices: [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6],
  });
  const REGION = 16;
  const LENGTH = 160;

  /** Ground for the viewers to stand on; the only collision in the scene. */
  const ground = () => {
    const positions: number[] = [];
    const indices: number[] = [];
    for (let x = 0; x < LENGTH; x += 8) {
      const v = positions.length / 3;
      positions.push(x, 0, -8, x + 8, 0, -8, x + 8, 0, 8, x, 0, 8);
      indices.push(v, v + 1, v + 2, v, v + 2, v + 3);
    }
    return {
      name: 'ground',
      material: 'stone',
      positions: new Float32Array(positions),
      indices: new Uint32Array(indices),
    };
  };

  /** The same wall drawn on one face only: visible from in front, not behind. */
  const oneSidedWallGlb = glb({
    corners: [0, 0, -8, 0, 0, 8, 0, 200, 8, 0, 200, -8],
    doubleSided: false,
  });

  function rowsWithWallAt(x: number | null, asset: Uint8Array = wallGlb) {
    const world = packageWith(
      x === null
        ? { prototype: [], enabled: [], phaseMask: [], position: [] }
        : {
            prototype: [0],
            enabled: [1],
            phaseMask: [0xffffffff],
            position: [[x, 0, 0]],
            radius: [100],
          },
      [{ id: 'wall', source: 'wall.glb' }],
    );
    const { primitives } = assembleOccluderScene({ world, loadPrototype: () => asset });
    const centers: [number, number][] = [];
    for (let cx = REGION / 2; cx < LENGTH; cx += REGION) centers.push([cx, 0]);
    const visibility = compileShadoWorldVisibility({
      mode: 'sampled-occlusion',
      bounds: { min: [0, 0, -8], max: [LENGTH, 200, 8] },
      regionSize: REGION,
      maxDistance: 1024,
      renderCellCenters: centers,
      persistentRenderCells: new Uint8Array(centers.length),
      collisionPrimitives: [ground(), ...primitives],
    });
    return (from: number, to: number) =>
      ((visibility.pvs.words[from * visibility.pvs.wordsPerRow + (to >>> 5)]! >>> 0) &
        (1 << (to & 31))) !== 0;
  }

  it('hides across a stamped wall, and stops hiding when the stamp moves away', () => {
    const open = rowsWithWallAt(null);
    const walled = rowsWithWallAt(80);
    const moved = rowsWithWallAt(158);

    // With nothing placed, the street is open end to end.
    expect(open(0, 9)).toBe(true);
    // Placing one opaque object is the entire difference.
    expect(walled(0, 9)).toBe(false);
    // Moving that same object out of the way restores the view, so the
    // rejection was caused by where the stamp stands and not by the ground.
    expect(moved(0, 9)).toBe(true);
  });

  it('does not hide anything behind a surface that is only drawn from one side', () => {
    /*
     * A row has to hold for a camera on either side of the blocker. A
     * single-sided panel is invisible from behind, so the far region is
     * genuinely reachable by sight from there, and admitting the pair is the
     * only correct answer -- even though the same panel, from in front, looks
     * like a perfectly good wall. Recovering that occlusion needs directed
     * rows, not a bolder ray test.
     */
    const oneSided = rowsWithWallAt(80, oneSidedWallGlb);
    expect(oneSided(0, 9)).toBe(true);
    expect(oneSided(9, 0)).toBe(true);
    // The two-faced wall in the same place still hides, so the difference is
    // sidedness and not the fixture.
    expect(rowsWithWallAt(80)(0, 9)).toBe(false);
  });
});

describe('what may be trusted as a permanent blocker', () => {
  const opaque = glb({});

  function world(stamps: { phaseMask: number[]; prototype?: number[] }, sources: string[]) {
    const count = stamps.phaseMask.length;
    return packageWith(
      {
        prototype: stamps.prototype ?? stamps.phaseMask.map(() => 0),
        enabled: stamps.phaseMask.map(() => 1),
        phaseMask: stamps.phaseMask,
        position: stamps.phaseMask.map((_, index) => [index * 10, 0, 0] as [number, number, number]),
      },
      sources.map((source, index) => ({ id: `proto-${index}`, source })),
    );
  }

  it('refuses a blocker that exists in one phase of the row and not another', () => {
    // Two phases share this bake. The first stamp is in both; the second is
    // only in phase A, so from phase B it is not there to hide anything.
    const scene = world({ phaseMask: [0b11, 0b01] }, ['wall.glb']);
    const { manifest, primitives } = assembleOccluderScene({
      world: scene,
      loadPrototype: () => opaque,
    });
    expect(manifest.phases).toMatchObject({ policy: 'invariant', worldMask: 0b11 });
    expect(manifest.stamps.included).toBe(1);
    expect(manifest.stamps.excluded['phase-variant-blocker']).toBe(1);
    expect(primitives).toHaveLength(1);
  });

  it('does not let an explicit mask talk its way out of invariance', () => {
    /*
     * Asking for one phase used to narrow the set invariance was measured
     * against, so a stamp present only in that phase looked invariant across
     * it -- and the row that came out had no phase restriction a runtime
     * could enforce. Invariance is a property of the world, not of the
     * request, so selecting phase 1 must still refuse a blocker that phase 2
     * does not have.
     */
    const scene = world({ phaseMask: [0b11, 0b01] }, ['wall.glb']);
    const { manifest, primitives } = assembleOccluderScene({
      world: scene,
      activePhaseMask: 0b01,
      loadPrototype: () => opaque,
    });
    expect(manifest.phases).toMatchObject({ policy: 'invariant', activeMask: 0b01, worldMask: 0b11 });
    expect(manifest.stamps.included).toBe(1);
    expect(manifest.stamps.excluded['phase-variant-blocker']).toBe(1);
    expect(primitives).toHaveLength(1);
  });

  it('admits the phase-bound blocker only when the caller binds the row to that phase', () => {
    const scene = world({ phaseMask: [0b11, 0b01] }, ['wall.glb']);
    const { manifest } = assembleOccluderScene({
      world: scene,
      activePhaseMask: 0b01,
      phasePolicy: 'active-phase',
      loadPrototype: () => opaque,
    });
    expect(manifest.stamps.included).toBe(2);
    expect(manifest.stamps.excluded['phase-variant-blocker']).toBe(0);
  });

  it('refuses geometry that is not where its buffer says it is', () => {
    for (const [reason, asset] of [
      ['animated-node', glb({ animated: true })],
      ['morph-targets', glb({ morphed: true })],
      ['skinned-geometry', glb({ skinned: true })],
    ] as const) {
      const parts = readGlbPrimitives(asset);
      expect(parts[0]!.dynamic).toBe(reason);
      expect(occluderEligibility(parts[0]!)).toBe(reason);
      const { primitives, manifest } = assembleOccluderScene({
        world: world({ phaseMask: [0xffffffff] }, ['thing.glb']),
        loadPrototype: () => asset,
      });
      expect(primitives).toHaveLength(0);
      expect(manifest.submeshes[0]!.reason).toBe(reason);
    }
  });

  it('carries sidedness through, so a one-sided surface cannot hide from behind', () => {
    const oneSided = glb({ doubleSided: false });
    const bothSides = glb({ doubleSided: true });
    expect(readGlbPrimitives(oneSided)[0]!.doubleSided).toBe(false);
    expect(readGlbPrimitives(bothSides)[0]!.doubleSided).toBe(true);
    const { primitives } = assembleOccluderScene({
      world: world({ phaseMask: [0xffffffff, 0xffffffff], prototype: [0, 1] }, ['one.glb', 'both.glb']),
      loadPrototype: (source) => (source === 'one.glb' ? oneSided : bothSides),
    });
    expect(primitives.map((primitive) => primitive.doubleSided)).toEqual([false, true]);
  });

  it('agrees on the blocking face however the mirror is expressed', () => {
    /*
     * The same placement written two ways: a mirrored glTF node with an
     * unmirrored stamp, and an unmirrored node with a mirrored stamp. They
     * put identical vertices in identical places, so they must agree on which
     * side of the surface blocks. Correcting winding at only one of the two
     * levels made them disagree.
     */
    const mirroredNode = glb({ node: { scale: [-1, 1, 1] } });
    const plainNode = glb({});
    const stamp = (scale: [number, number, number]) =>
      packageWith(
        { prototype: [0], enabled: [1], phaseMask: [0xffffffff], position: [[0, 0, 0]], scale: [scale] },
        [{ id: 'wall', source: 'wall.glb' }],
      );
    const viaNode = assembleOccluderScene({
      world: stamp([1, 1, 1]),
      loadPrototype: () => mirroredNode,
    }).primitives[0]!;
    const viaStamp = assembleOccluderScene({
      world: stamp([-1, 1, 1]),
      loadPrototype: () => plainNode,
    }).primitives[0]!;
    // Same geometry either way.
    expect(Array.from(viaNode.positions as Float32Array)).toEqual(
      Array.from(viaStamp.positions as Float32Array)
    );
    // And the same front face either way.
    expect(Array.from(viaNode.indices as Uint32Array)).toEqual(
      Array.from(viaStamp.indices as Uint32Array)
    );

    // Two mirrors compose back to no mirror, and must not be corrected twice.
    const doubleMirrored = assembleOccluderScene({
      world: stamp([-1, 1, 1]),
      loadPrototype: () => mirroredNode,
    }).primitives[0]!;
    const plain = assembleOccluderScene({
      world: stamp([1, 1, 1]),
      loadPrototype: () => plainNode,
    }).primitives[0]!;
    expect(Array.from(doubleMirrored.indices as Uint32Array)).toEqual(
      Array.from(plain.indices as Uint32Array)
    );
  });

  it('reverses winding for a mirrored stamp, keeping the front face in front', () => {
    const base = packageWith(
      {
        prototype: [0],
        enabled: [1],
        phaseMask: [0xffffffff],
        position: [[0, 0, 0]],
      },
      [{ id: 'wall', source: 'wall.glb' }],
    );
    const mirrored = packageWith(
      {
        prototype: [0],
        enabled: [1],
        phaseMask: [0xffffffff],
        position: [[0, 0, 0]],
        scale: [[-1, 1, 1]],
      },
      [{ id: 'wall', source: 'wall.glb' }],
    );
    const one = assembleOccluderScene({ world: base, loadPrototype: () => opaque }).primitives[0]!;
    const other = assembleOccluderScene({ world: mirrored, loadPrototype: () => opaque }).primitives[0]!;
    // Same triangles, opposite corner order: the mirror flipped the geometry,
    // so the index order flips back to keep the drawn face the front face.
    expect(Array.from(other.indices as Uint32Array).slice(0, 3)).toEqual([0, 2, 1]);
    expect(Array.from(one.indices as Uint32Array).slice(0, 3)).toEqual([0, 1, 2]);
  });
});
