import {
  buildInstancedOccluders,
  buildOccluderBvh,
  bvhSegmentBlocked,
  instancedSegmentBlocked,
} from '../src/world';
import type { ShadoWorldPrimitive } from '../src/world';

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** A closed-ish box in prototype space, so it blocks from any side. */
function box(size = 1): ShadoWorldPrimitive {
  const h = size / 2;
  const positions: number[] = [];
  const indices: number[] = [];
  const quad = (corners: number[][]) => {
    const v = positions.length / 3;
    for (const corner of corners) positions.push(...corner);
    indices.push(v, v + 1, v + 2, v, v + 2, v + 3);
  };
  quad([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h]]);
  quad([[h, -h, h], [-h, -h, h], [-h, h, h], [h, h, h]]);
  quad([[-h, -h, h], [-h, -h, -h], [-h, h, -h], [-h, h, h]]);
  quad([[h, -h, -h], [h, -h, h], [h, h, h], [h, h, -h]]);
  quad([[-h, h, -h], [h, h, -h], [h, h, h], [-h, h, h]]);
  quad([[-h, -h, h], [h, -h, h], [h, -h, -h], [-h, -h, -h]]);
  return {
    name: 'box',
    material: 'stone',
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
  };
}

/** Column-major translate/scale, as the runtime composes a stamp. */
function transform(tx: number, ty: number, tz: number, sx = 1, sy = 1, sz = 1): number[] {
  return [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, tx, ty, tz, 1];
}

/** The flat expansion this structure replaces, for differential comparison. */
function expand(prototype: ShadoWorldPrimitive, matrix: readonly number[]): ShadoWorldPrimitive {
  const positions = new Float32Array(prototype.positions.length);
  for (let offset = 0; offset < prototype.positions.length; offset += 3) {
    const x = Number(prototype.positions[offset]);
    const y = Number(prototype.positions[offset + 1]);
    const z = Number(prototype.positions[offset + 2]);
    positions[offset] = matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!;
    positions[offset + 1] = matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!;
    positions[offset + 2] = matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!;
  }
  const mirrored =
    matrix[0]! * (matrix[5]! * matrix[10]! - matrix[6]! * matrix[9]!) -
      matrix[4]! * (matrix[1]! * matrix[10]! - matrix[2]! * matrix[9]!) +
      matrix[8]! * (matrix[1]! * matrix[6]! - matrix[2]! * matrix[5]!) <
    0;
  const indices = Uint32Array.from(prototype.indices as ArrayLike<number>);
  if (mirrored) {
    for (let index = 0; index + 2 < indices.length; index += 3) {
      const swap = indices[index + 1]!;
      indices[index + 1] = indices[index + 2]!;
      indices[index + 2] = swap;
    }
  }
  return { name: 'placed', material: prototype.material, positions, indices };
}

describe('placed prototypes answer what expanded geometry answers', () => {
  const prototype = box(2);
  /** A single-sided wall, so the random comparison exercises facing too. */
  const wall: ShadoWorldPrimitive = {
    name: 'wall',
    material: 'stone',
    doubleSided: false,
    positions: new Float32Array([-6, -6, 0, 6, -6, 0, 6, 6, 0, -6, 6, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
  const placements = [
    transform(0, 0, 0),
    transform(20, 0, 0, 2, 2, 2),
    transform(-15, 5, 8),
    transform(30, 0, -12, -1, 1, 1), // mirrored
    transform(5, 0, 25, 1, 3, 1),
  ];

  it('agrees with the flat expansion on thousands of random segments', () => {
    const wallPlacements = [transform(10, 0, 10), transform(-8, 0, -8, -1, 1, 1)];
    const instanced = buildInstancedOccluders(
      [[prototype], [wall]],
      [
        ...placements.map((matrix) => ({ prototype: 0, matrix })),
        ...wallPlacements.map((matrix) => ({ prototype: 1, matrix })),
      ],
    );
    const flat = buildOccluderBvh([
      ...placements.map((matrix) => expand(prototype, matrix)),
      ...wallPlacements.map((matrix) => ({ ...expand(wall, matrix), doubleSided: false })),
    ]);
    // Twelve for the box, two for the wall: each indexed once.
    expect(instanced.uniqueTriangles).toBe(14);
    expect(instanced.placedTriangles).toBe(64);
    expect(flat.triangleCount).toBe(64);

    const next = random(90210);
    let blocked = 0;
    for (let trial = 0; trial < 4000; trial += 1) {
      const a = [(next() - 0.5) * 100, (next() - 0.5) * 40, (next() - 0.5) * 100] as const;
      const b = [(next() - 0.5) * 100, (next() - 0.5) * 40, (next() - 0.5) * 100] as const;
      const viaInstances = instancedSegmentBlocked(instanced, a[0], a[1], a[2], b[0], b[1], b[2]);
      const viaFlat = bvhSegmentBlocked(flat, a[0], a[1], a[2], b[0], b[1], b[2]);
      expect(viaInstances).toBe(viaFlat);
      if (viaFlat) blocked += 1;
    }
    // A comparison where nothing is ever blocked proves nothing about blocking.
    expect(blocked).toBeGreaterThan(50);
  });

  it('indexes each prototype once however many times it is placed', () => {
    const many = Array.from({ length: 500 }, (_, index) => ({
      prototype: 0,
      matrix: transform((index % 25) * 6, 0, Math.floor(index / 25) * 6),
    }));
    const instanced = buildInstancedOccluders([[prototype]], many);
    expect(instanced.uniqueTriangles).toBe(12);
    expect(instanced.placedTriangles).toBe(6000);
    // The geometry is stored once; only transforms repeat.
    expect(instanced.prototypes[0]!.triangles.length).toBe(12 * 9);
  });

  it('keeps the endpoint rule under the transform', () => {
    /*
     * The parameter rides along unchanged under an affine map, so a segment
     * that starts exactly on a placed surface is still standing on it rather
     * than being blocked by it -- whatever scale the placement applies.
     */
    const instanced = buildInstancedOccluders(
      [[prototype]],
      [{ prototype: 0, matrix: transform(0, 0, 0, 4, 4, 4) }],
    );
    // From the box's own top face, straight up: nothing in the way.
    expect(instancedSegmentBlocked(instanced, 0, 4, 0, 0, 40, 0)).toBe(false);
    // From above, down through it: blocked.
    expect(instancedSegmentBlocked(instanced, 0, 40, 0, 0, -40, 0)).toBe(true);
  });

  it('flips facing for a mirrored placement rather than blocking the wrong side', () => {
    // A single-sided plane facing +z at the origin.
    const plane: ShadoWorldPrimitive = {
      name: 'plane',
      material: 'stone',
      doubleSided: false,
      positions: new Float32Array([-5, -5, 0, 5, -5, 0, 5, 5, 0, -5, 5, 0]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
    const plain = buildInstancedOccluders([[plane]], [{ prototype: 0, matrix: transform(0, 0, 0) }]);
    const mirrored = buildInstancedOccluders(
      [[plane]],
      [{ prototype: 0, matrix: transform(0, 0, 0, 1, 1, -1) }],
    );
    const front = (scene: typeof plain) => instancedSegmentBlocked(scene, 0, 0, 20, 0, 0, -20);
    const back = (scene: typeof plain) => instancedSegmentBlocked(scene, 0, 0, -20, 0, 0, 20);
    // The unmirrored plane blocks from one side only...
    expect(front(plain)).not.toBe(back(plain));
    // ...and mirroring it swaps which side that is, rather than changing how
    // many sides block.
    expect(front(mirrored)).toBe(back(plain));
    expect(back(mirrored)).toBe(front(plain));
  });

  it('is empty-scene safe', () => {
    const empty = buildInstancedOccluders([], []);
    expect(empty.instanceCount).toBe(0);
    expect(instancedSegmentBlocked(empty, 0, 0, 0, 1, 1, 1)).toBe(false);
  });
});
