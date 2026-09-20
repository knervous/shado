import {
  assembleOccluderScene,
  occluderEligibility,
  readGlbPrimitives,
} from '../src/world';
import type { ShadoWorldSpatialPackage } from '../src/world';

/** A GLB holding one axis-aligned quad, built by hand so the test owns every byte. */
function glb(options: {
  alphaMode?: string;
  node?: { translation?: number[]; scale?: number[] };
  corners?: number[];
}): Uint8Array {
  const corners = options.corners ?? [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
  const positions = new Float32Array(corners);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  const indexBytes = new Uint8Array(indices.buffer);
  const padding = (4 - (indexBytes.byteLength % 4)) % 4;
  const binary = new Uint8Array(positions.byteLength + indexBytes.byteLength + padding);
  binary.set(new Uint8Array(positions.buffer), 0);
  binary.set(indexBytes, positions.byteLength);
  const gltf = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ name: 'root', mesh: 0, ...(options.node ?? {}) }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0, mode: 4 }] }],
    materials: [{ name: 'stone', ...(options.alphaMode ? { alphaMode: options.alphaMode } : {}) }],
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
