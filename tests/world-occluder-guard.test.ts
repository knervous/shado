import {
  OccluderBuildGuard,
  buildInstancedOccluders,
  buildOccluderBvh,
  estimateBvhBytes,
  estimateInstancedTopBytes,
  instancedSegmentBlocked,
} from '../src/world';
import type { OccluderBuildStop, ShadoWorldPrimitive } from '../src/world';

/** Two triangles: the smallest prototype that indexes anything. */
function square(): ShadoWorldPrimitive {
  return {
    name: 'square',
    material: 'stone',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

/** n triangles, spread out so the hierarchy has real work to do. */
function soup(count: number): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const x = index * 3;
    const v = positions.length / 3;
    positions.push(x, 0, 0, x + 1, 0, 0, x, 1, 0);
    indices.push(v, v + 1, v + 2);
  }
  return { name: 'soup', material: 'stone', positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const at = (x: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1];

describe('one guard across every BLAS and the TLAS', () => {
  it('refuses 1,000 placements under 4 KiB before allocating the instances', () => {
    /*
     * The audit's reproduction: one valid two-triangle prototype, 1,000
     * identity placements, maxBytes 4096. The builder returned aborted:null
     * while the instance and TLAS arrays alone held 245,060 bytes.
     */
    const instances = Array.from({ length: 1000 }, () => ({ prototype: 0, matrix: IDENTITY }));
    expect(estimateInstancedTopBytes(1000)).toBeGreaterThan(4096);
    const scene = buildInstancedOccluders([[square()]], instances, { maxBytes: 4096 });
    expect(scene.aborted).toBe('memory');
    // Refused, so it indexes nothing and blocks nothing: it admits.
    expect(scene.instanceCount).toBe(0);
    expect(instancedSegmentBlocked(scene, 0, 0, -5, 0, 0, 5)).toBe(false);
  });

  it('stops the TLAS when cancellation arrives right after the BLAS', () => {
    /*
     * The stop callback answers "cancelled" from the first poll after the
     * prototype is built. The TLAS used to make NO polls at all, so this
     * cancellation had nowhere to land and the build completed.
     */
    let blasDone = false;
    const guard = new OccluderBuildGuard({
      stopReason: (): OccluderBuildStop | null => (blasDone ? 'cancelled' : null),
    });
    const bvh = buildOccluderBvh([square()], { guard });
    expect(bvh.aborted).toBeNull();
    blasDone = true;
    const instances = Array.from({ length: 10 }, (_, index) => ({ prototype: 0, matrix: at(index * 4) }));
    const scene = buildInstancedOccluders([[square()]], instances, { guard });
    expect(scene.aborted).toBe('cancelled');
  });

  it('interrupts a large TLAS partway, not only on entry', () => {
    let polls = 0;
    const scene = buildInstancedOccluders(
      [[square()]],
      Array.from({ length: 50_000 }, (_, index) => ({ prototype: 0, matrix: at(index * 3) })),
      {
        stopReason: () => {
          polls += 1;
          // Past the BLAS and the entry checks: inside the TLAS's own loops.
          return polls > 6 ? 'cancelled' : null;
        },
      }
    );
    expect(scene.aborted).toBe('cancelled');
    expect(polls).toBeGreaterThan(6);
  });

  it('charges two structures that each fit against one combined limit', () => {
    /*
     * Each of these fits on its own. Together they do not -- and separate
     * guards, each with the full allowance, built both.
     */
    const one = estimateBvhBytes(400);
    // What a finished one keeps, measured rather than guessed.
    const probe = new OccluderBuildGuard({});
    buildOccluderBvh([soup(400)], { guard: probe });
    const kept = probe.ledger.retainedBytes;
    // Room for one build at its peak, but not for that plus a finished one.
    const limit = one + Math.floor(kept / 2);
    expect(buildOccluderBvh([soup(400)], { maxBytes: limit }).aborted).toBeNull();

    const guard = new OccluderBuildGuard({ maxBytes: limit });
    const first = buildOccluderBvh([soup(400)], { guard });
    expect(first.aborted).toBeNull();
    const second = buildOccluderBvh([soup(400)], { guard });
    expect(second.aborted).toBe('memory');
    expect(guard.ledger.retainedBytes).toBeGreaterThan(0);
  });

  it('keeps a finished BLAS charged against the TLAS that follows it', () => {
    const guard = new OccluderBuildGuard({});
    buildOccluderBvh([soup(400)], { guard });
    const held = guard.ledger.retainedBytes;
    // Scratch is given back; the leaf payload and nodes are not.
    expect(held).toBeGreaterThan(400 * 9 * 8);
    expect(held).toBeLessThan(estimateBvhBytes(400));
    expect(guard.ledger.reservedBytes).toBe(0);
  });

  it('checks on entry even when there are no prototypes', () => {
    const scene = buildInstancedOccluders([], [], { stopReason: () => 'seconds' });
    expect(scene.aborted).toBe('seconds');
  });

  it('refuses a placement that names a prototype that does not exist', () => {
    expect(() => buildInstancedOccluders([[square()]], [{ prototype: 3, matrix: IDENTITY }])).toThrow(
      /prototype 3/
    );
  });

  it('refuses a transform that is not finite or cannot be inverted', () => {
    const nan = [...IDENTITY];
    nan[12] = Number.NaN;
    expect(() => buildInstancedOccluders([[square()]], [{ prototype: 0, matrix: nan }])).toThrow(/non-finite/);
    const flat = [...IDENTITY];
    flat[10] = 0;
    expect(() => buildInstancedOccluders([[square()]], [{ prototype: 0, matrix: flat }])).toThrow(/inverted/);
  });

  it('leaves an empty prototype out of the top level instead of bounding it', () => {
    /*
     * An empty prototype has no box. Carried through a transform its
     * infinities make an inverted box that passes the slab test from every
     * direction, so it is excluded -- and the placement that IS real still
     * blocks.
     */
    const empty: ShadoWorldPrimitive = {
      name: 'empty',
      material: 'stone',
      positions: new Float32Array(0),
      indices: new Uint32Array(0),
    };
    const scene = buildInstancedOccluders([[empty], [square()]], [
      { prototype: 0, matrix: IDENTITY },
      { prototype: 1, matrix: at(10) },
    ]);
    expect(scene.aborted).toBeNull();
    expect(Array.from(scene.instanceValid)).toEqual([0, 1]);
    expect(scene.order.length).toBe(1);
    expect(instancedSegmentBlocked(scene, 0, 0, -5, 0, 0, 5)).toBe(false);
    expect(instancedSegmentBlocked(scene, 10, 0, -5, 10, 0, 5)).toBe(true);
  });
});
