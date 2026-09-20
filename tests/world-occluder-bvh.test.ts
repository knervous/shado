import {
  buildOccluderBvh,
  estimateBvhBytes,
  estimateGridPayloadBytes,
  buildOccluderGrid,
  bvhHighestSurfaceAt,
  bvhSegmentBlocked,
  highestSurfaceAt,
  segmentBlocked,
} from '../src/world';
import type { ShadoWorldPrimitive } from '../src/world';

/** Deterministic pseudo-random, so a failure is reproducible from its seed. */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** A scene of randomly placed, randomly oriented triangles. */
function scatter(count: number, seed: number, extent = 200): ShadoWorldPrimitive {
  const next = random(seed);
  const positions = new Float32Array(count * 9);
  const indices = new Uint32Array(count * 3);
  for (let triangle = 0; triangle < count; triangle += 1) {
    const cx = (next() - 0.5) * extent;
    const cy = (next() - 0.5) * extent;
    const cz = (next() - 0.5) * extent;
    for (let corner = 0; corner < 3; corner += 1) {
      const offset = triangle * 9 + corner * 3;
      positions[offset] = cx + (next() - 0.5) * 20;
      positions[offset + 1] = cy + (next() - 0.5) * 20;
      positions[offset + 2] = cz + (next() - 0.5) * 20;
      indices[triangle * 3 + corner] = triangle * 3 + corner;
    }
  }
  return { name: 'scatter', material: 'stone', positions, indices };
}

/** The answer with no acceleration at all: every triangle, every time. */
function bruteForceBlocked(
  primitive: ShadoWorldPrimitive,
  a: readonly [number, number, number],
  b: readonly [number, number, number]
): boolean {
  const one = { ...primitive };
  const bvh = buildOccluderBvh([one]);
  // A single leaf per triangle is still a hierarchy, so brute force is done
  // directly here rather than by degenerating the structure under test.
  const positions = primitive.positions;
  const indices = primitive.indices;
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  for (let index = 0; index + 2 < indices.length; index += 3) {
    const p = [0, 1, 2].map((corner) => {
      const base = Number(indices[index + corner]) * 3;
      return [Number(positions[base]), Number(positions[base + 1]), Number(positions[base + 2])];
    });
    const e1 = [p[1]![0]! - p[0]![0]!, p[1]![1]! - p[0]![1]!, p[1]![2]! - p[0]![2]!];
    const e2 = [p[2]![0]! - p[0]![0]!, p[2]![1]! - p[0]![1]!, p[2]![2]! - p[0]![2]!];
    const h = [dy * e2[2]! - dz * e2[1]!, dz * e2[0]! - dx * e2[2]!, dx * e2[1]! - dy * e2[0]!];
    const det = e1[0]! * h[0]! + e1[1]! * h[1]! + e1[2]! * h[2]!;
    if (det > -1e-12 && det < 1e-12) continue;
    const inv = 1 / det;
    const s = [a[0] - p[0]![0]!, a[1] - p[0]![1]!, a[2] - p[0]![2]!];
    const u = inv * (s[0]! * h[0]! + s[1]! * h[1]! + s[2]! * h[2]!);
    if (u < 0 || u > 1) continue;
    const q = [
      s[1]! * e1[2]! - s[2]! * e1[1]!,
      s[2]! * e1[0]! - s[0]! * e1[2]!,
      s[0]! * e1[1]! - s[1]! * e1[0]!,
    ];
    const v = inv * (dx * q[0]! + dy * q[1]! + dz * q[2]!);
    if (v < 0 || u + v > 1) continue;
    const hit = inv * (e2[0]! * q[0]! + e2[1]! * q[1]! + e2[2]! * q[2]!);
    if (hit > 1e-3 && hit < 1 - 1e-3) return true;
  }
  void bvh;
  return false;
}

const BOUNDS = { min: [-200, -200, -200] as [number, number, number], max: [200, 200, 200] as [number, number, number] };

describe('the hierarchy answers what the grid and a brute-force sweep answer', () => {
  const scene = scatter(400, 12345);
  const grid = buildOccluderGrid([scene], BOUNDS, 32);
  const bvh = buildOccluderBvh([scene]);
  const next = random(999);

  it('agrees on randomized segments', () => {
    let blocked = 0;
    for (let trial = 0; trial < 600; trial += 1) {
      const a: [number, number, number] = [
        (next() - 0.5) * 400, (next() - 0.5) * 400, (next() - 0.5) * 400,
      ];
      const b: [number, number, number] = [
        (next() - 0.5) * 400, (next() - 0.5) * 400, (next() - 0.5) * 400,
      ];
      const truth = bruteForceBlocked(scene, a, b);
      expect(bvhSegmentBlocked(bvh, a[0], a[1], a[2], b[0], b[1], b[2])).toBe(truth);
      expect(segmentBlocked(grid, a[0], a[1], a[2], b[0], b[1], b[2])).toBe(truth);
      if (truth) blocked += 1;
    }
    // A test where nothing is ever blocked proves nothing about blocking.
    expect(blocked).toBeGreaterThan(30);
  });

  it('agrees on the degenerate and axis-aligned cases', () => {
    const cases: [number, number, number, number, number, number][] = [
      [0, 0, 0, 0, 0, 0],
      [-500, 0, 0, 500, 0, 0],
      [0, -500, 0, 0, 500, 0],
      [0, 0, -500, 0, 0, 500],
      [-500, -500, -500, 500, 500, 500],
      [1e6, 1e6, 1e6, -1e6, -1e6, -1e6],
      [-500, 0.5, 0.25, -499, 0.5, 0.25],
    ];
    for (const [ax, ay, az, bx, by, bz] of cases) {
      expect(bvhSegmentBlocked(bvh, ax, ay, az, bx, by, bz)).toBe(
        segmentBlocked(grid, ax, ay, az, bx, by, bz)
      );
    }
  });

  it('finds the same floor under a column', () => {
    const floor: ShadoWorldPrimitive = {
      name: 'stack',
      material: 'stone',
      positions: new Float32Array([
        0, 0, 0, 10, 0, 0, 10, 0, 10, 0, 0, 10,
        0, 8, 0, 10, 8, 0, 10, 8, 10, 0, 8, 10,
      ]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]),
    };
    const stackedGrid = buildOccluderGrid([floor], { min: [0, 0, 0], max: [10, 8, 10] }, 4);
    const stackedBvh = buildOccluderBvh([floor]);
    for (const [x, z] of [[5, 5], [0.5, 0.5], [9.5, 9.5], [-1, 5], [5, 20]]) {
      expect(bvhHighestSurfaceAt(stackedBvh, x!, z!)).toEqual(
        highestSurfaceAt(stackedGrid, x!, z!, { min: [0, 0, 0], max: [10, 8, 10] })
      );
    }
  });

  it('is empty-scene safe', () => {
    const empty = buildOccluderBvh([]);
    expect(empty.triangleCount).toBe(0);
    expect(bvhSegmentBlocked(empty, 0, 0, 0, 1, 1, 1)).toBe(false);
    expect(bvhHighestSurfaceAt(empty, 0, 0)).toBeNull();
  });

  it('covers every triangle in exactly one leaf', () => {
    /*
     * The invariant a silently truncated node array violates. Typed arrays
     * drop out-of-range writes without complaint, so an undersized node budget
     * does not throw -- it produces a tree whose leaves do not add up, and
     * queries that walk past real geometry and report a clear line.
     */
    for (const scene of [scatter(1, 1), scatter(9, 2), scatter(400, 3), scatter(1500, 4)]) {
      const built = buildOccluderBvh([scene]);
      const covered = new Uint8Array(built.triangleCount);
      let leaves = 0;
      for (let node = 0; node < built.nodeCount; node += 1) {
        const count = built.nodeMeta[node * 3 + 1]!;
        if (count === 0) continue;
        leaves += 1;
        const first = built.nodeMeta[node * 3]!;
        for (let index = first; index < first + count; index += 1) {
          expect(covered[index]).toBe(0);
          covered[index] = 1;
        }
      }
      expect(Array.from(covered).every((seen) => seen === 1)).toBe(true);
      expect(leaves).toBeGreaterThan(0);
      // A node's bounds must contain its children's, or the slab test prunes
      // away geometry that is really there.
      for (let node = 0; node < built.nodeCount; node += 1) {
        if (built.nodeMeta[node * 3 + 1]! !== 0) continue;
        for (const child of [node + 1, built.nodeMeta[node * 3 + 2]!]) {
          for (let axis = 0; axis < 3; axis += 1) {
            expect(built.nodeBounds[child * 6 + axis]).toBeGreaterThanOrEqual(
              built.nodeBounds[node * 6 + axis]! - 1e-9
            );
            expect(built.nodeBounds[child * 6 + 3 + axis]).toBeLessThanOrEqual(
              built.nodeBounds[node * 6 + 3 + axis]! + 1e-9
            );
          }
        }
      }
    }
  });
});

describe('construction can be stopped while it is working', () => {
  /** One primitive big enough that stopping between primitives is no use. */
  const huge = scatter(10_000, 31337, 800);

  /** How many times a never-stopping guard is consulted for n triangles. */
  function pollCount(triangles: number): number {
    let polls = 0;
    buildOccluderBvh([scatter(triangles, 4242, 800)], {
      stopReason: () => { polls += 1; return null; },
    });
    return polls;
  }

  it('counts work in elementary iterations, not in calls', () => {
    /*
     * The distinction the review turned on. A guard polled once per call lets
     * one call over every triangle in the zone -- a root bounds scan or a root
     * partition -- run to completion. Polling charged per element scales with
     * the geometry, so four times the triangles is about four times the polls.
     */
    const small = pollCount(4_000);
    const large = pollCount(16_000);
    expect(small).toBeGreaterThan(2);
    expect(large / small).toBeGreaterThan(3);
  });

  it('aborts wherever it happens to be when the answer changes', () => {
    // Spread across ingestion, the order fill, bounds scans, partitioning and
    // the payload reorder; every one of them must unwind.
    for (const after of [1, 2, 4, 8, 16, 32]) {
      let calls = 0;
      const built = buildOccluderBvh([huge], {
        stopReason: () => (++calls > after ? 'cancelled' : null),
      });
      expect(built.aborted).toBe('cancelled');
      expect(built.triangleCount).toBe(0);
      expect(built.nodeCount).toBe(1);
    }
  });

  it('reports the reason it was given, not a generic cancellation', () => {
    for (const reason of ['seconds', 'memory', 'segment-queries', 'cancelled'] as const) {
      let calls = 0;
      const built = buildOccluderBvh([huge], {
        stopReason: () => (++calls > 3 ? reason : null),
      });
      expect(built.aborted).toBe(reason);
    }
  });

  it('runs to completion when nothing asks it to stop', () => {
    let calls = 0;
    const built = buildOccluderBvh([huge], {
      stopReason: () => { calls += 1; return null; },
    });
    expect(built.aborted).toBeNull();
    expect(built.triangleCount).toBe(10_000);
    expect(calls).toBeGreaterThan(3);
  });

  it('refuses an allocation it can predict will not fit', () => {
    const built = buildOccluderBvh([huge], { maxBytes: 1024 });
    expect(built.aborted).toBe('memory');
    expect(built.triangleCount).toBe(0);
    // And the prediction covers every buffer it would have taken.
    expect(estimateBvhBytes(10_000)).toBeGreaterThan(10_000 * 9 * 8 * 2);
  });

  it('selects the same tree a full sort would have', () => {
    const built = buildOccluderBvh([huge]);
    const covered = new Uint8Array(built.triangleCount);
    for (let node = 0; node < built.nodeCount; node += 1) {
      const count = built.nodeMeta[node * 3 + 1]!;
      if (count === 0) continue;
      const first = built.nodeMeta[node * 3]!;
      for (let index = first; index < first + count; index += 1) covered[index] = 1;
    }
    expect(Array.from(covered).every((seen) => seen === 1)).toBe(true);
    const grid = buildOccluderGrid([huge], { min: [-800, -800, -800], max: [800, 800, 800] }, 32);
    const pick = random(24680);
    for (let trial = 0; trial < 300; trial += 1) {
      const a = [(pick() - 0.5) * 1600, (pick() - 0.5) * 1600, (pick() - 0.5) * 1600] as const;
      const b = [(pick() - 0.5) * 1600, (pick() - 0.5) * 1600, (pick() - 0.5) * 1600] as const;
      expect(bvhSegmentBlocked(built, a[0], a[1], a[2], b[0], b[1], b[2])).toBe(
        segmentBlocked(grid, a[0], a[1], a[2], b[0], b[1], b[2])
      );
    }
  });
});

describe('the grid refuses bounded work instead of pretending', () => {
  /**
   * The review's reproduction: one triangle spanning the world diagonally.
   * Its payload estimate is 153 bytes and it produces 35,937 buckets, so a
   * triangle-count estimate cannot bound this backend at all.
   */
  const sprawling: ShadoWorldPrimitive = {
    name: 'diagonal',
    material: 'stone',
    positions: new Float32Array([0, 0, 0, 1024, 0, 1024, 0, 1024, 1024]),
    indices: new Uint32Array([0, 1, 2]),
  };
  const bounds = { min: [0, 0, 0] as [number, number, number], max: [1024, 1024, 1024] as [number, number, number] };

  it('refuses a memory ceiling before inserting anything', () => {
    expect(() => buildOccluderGrid([sprawling], bounds, 32, { maxBytes: 1024 }))
      .toThrow(/cannot honour memory or cancellation limits/);
  });

  it('refuses a cancellation contract it cannot honour', () => {
    expect(() => buildOccluderGrid([sprawling], bounds, 32, { stopReason: () => null }))
      .toThrow(/bvh/);
  });

  it('still builds, unbounded, as the diagnostic it is', () => {
    const grid = buildOccluderGrid([sprawling], bounds, 32);
    expect(grid.aborted).toBeNull();
    expect(grid.triangleCount).toBe(1);
    // One triangle, tens of thousands of bucket references: the number the
    // payload estimate never saw.
    expect(grid.bucketReferences).toBeGreaterThan(30_000);
    expect(estimateGridPayloadBytes(1)).toBeLessThan(1024);
  });
});
