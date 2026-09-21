import { createGpuOccluderBackend, ShadoGpuBackendUnsupported } from '../src/world';
import type { OccluderBvh } from '../src/world';

/** A hierarchy of the given size, with no geometry worth speaking of. */
function bvh(triangleCount: number, nodeCount: number, maxDepth = 8): OccluderBvh {
  return {
    aborted: null,
    triangles: new Float64Array(triangleCount * 9),
    doubleSided: new Uint8Array(triangleCount),
    triangleCount,
    nodeBounds: new Float64Array(nodeCount * 6),
    nodeMeta: new Int32Array(nodeCount * 3),
    nodeCount,
    maxDepth,
    counters: {
      segmentQueries: 0,
      columnQueries: 0,
      nodeVisits: 0,
      triangleTests: 0,
      blockedQueries: 0,
    },
  };
}

/**
 * A device that only reports limits.
 *
 * Every refusal has to happen before a buffer is created, so a stub with no
 * buffer methods is exactly the right shape: if the backend ever gets past
 * the limit checks it throws a TypeError instead, and the test says so.
 */
const device = (limits: Partial<GPUSupportedLimits>): GPUDevice =>
  ({
    limits: {
      maxStorageBufferBindingSize: 1 << 27,
      maxBufferSize: 1 << 28,
      ...limits,
    },
  }) as unknown as GPUDevice;

describe('GPU occluder backend limits', () => {
  it('refuses a hierarchy no buffer on the device can hold', async () => {
    await expect(
      createGpuOccluderBackend(device({ maxStorageBufferBindingSize: 1 << 16 }), bvh(100_000, 20_000))
    ).rejects.toBeInstanceOf(ShadoGpuBackendUnsupported);
  });

  it('takes the smaller of the binding and the buffer limit', async () => {
    /*
     * A device can bind more than it can allocate. Reading only the binding
     * limit would send an allocation the device cannot make, and an
     * over-large write aborts Dawn outright rather than raising.
     */
    await expect(
      createGpuOccluderBackend(device({ maxBufferSize: 1 << 16 }), bvh(100_000, 20_000))
    ).rejects.toBeInstanceOf(ShadoGpuBackendUnsupported);
  });

  it('refuses a hierarchy deeper than the shader stack', async () => {
    await expect(
      createGpuOccluderBackend(device({}), bvh(8, 3, 4096))
    ).rejects.toBeInstanceOf(ShadoGpuBackendUnsupported);
  });

  it('refuses an empty hierarchy rather than reporting nothing blocks', async () => {
    await expect(createGpuOccluderBackend(device({}), bvh(0, 0))).rejects.toBeInstanceOf(
      ShadoGpuBackendUnsupported
    );
  });

  it('names what did not fit, so the fallback is reportable', async () => {
    await expect(
      createGpuOccluderBackend(device({ maxStorageBufferBindingSize: 1 << 16 }), bvh(100_000, 20_000))
    ).rejects.toThrow(/\w+ need \d+ bytes and the device binds at most 65536/);
  });
});
