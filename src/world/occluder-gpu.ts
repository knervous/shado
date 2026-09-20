/**
 * A WebGPU compute backend for the segment queries the bake spends its time
 * in, over the same hierarchy the CPU walks.
 *
 * What it is allowed to decide: nothing, on its own. The shader runs in f32
 * where the CPU runs in f64, so the two disagree on segments that graze a
 * surface -- and a bake that cleared a PVS bit on an f32 near-miss would hide
 * geometry a player can see. So this proposes: a segment it calls CLEAR is
 * clear (admitting is always safe), and a segment it calls BLOCKED is handed
 * back to the CPU to certify before anything is rejected on the strength of
 * it. See `certifyBlocked` in the caller.
 */
import type { OccluderBvh } from './occluder-bvh';

/** Raised when a device cannot host this hierarchy. Always a fallback, never a failure. */
export class ShadoGpuBackendUnsupported extends Error {
  constructor(reason: string) {
    super(`Shado GPU occluder backend unsupported: ${reason}`);
    this.name = 'ShadoGpuBackendUnsupported';
  }
}

export type GpuBackendStats = {
  /** Bytes resident on the device for the immutable hierarchy. */
  readonly residentBytes: number;
  /** Milliseconds spent building and uploading, once, before any query. */
  uploadMs: number;
  /** Milliseconds inside submit-and-wait, summed over batches. */
  dispatchMs: number;
  /** Milliseconds spent mapping and copying results back. */
  readbackMs: number;
  segments: number;
  batches: number;
  /** Segments the shader proposed as blocked, before CPU certification. */
  proposedBlocked: number;
};

export type GpuOccluderBackend = {
  /**
   * One byte per segment: 1 where the shader found a blocker. Reused between
   * batches, so the caller reads it before the next call.
   */
  blockedBatch(segments: Float32Array, count: number): Promise<Uint8Array>;
  readonly stats: GpuBackendStats;
  readonly maxSegmentsPerBatch: number;
  dispose(): void;
};

/*
 * The CPU's own constants, restated for the shader. Keeping them literal in
 * the source rather than uniform-fed means a change on one side shows up as a
 * differential failure rather than as a silently divergent run.
 */
const END_EPSILON = 1e-3;
const DET_EPSILON = 1e-12;
const STACK_DEPTH = 64;
const WORKGROUP = 64;

const SHADER = /* wgsl */ `
struct Meta { first: i32, count: i32, right: i32, pad: i32 };

@group(0) @binding(0) var<storage, read> nodeBounds: array<f32>;
@group(0) @binding(1) var<storage, read> nodeMeta: array<Meta>;
@group(0) @binding(2) var<storage, read> triangles: array<f32>;
@group(0) @binding(3) var<storage, read> doubleSided: array<u32>;
@group(0) @binding(4) var<storage, read> segments: array<f32>;
@group(0) @binding(5) var<storage, read_write> blocked: array<u32>;
@group(0) @binding(6) var<uniform> counts: vec4<u32>;

fn slabs(node: u32, origin: vec3<f32>, inverse: vec3<f32>) -> bool {
  let base = node * 6u;
  var near = 0.0;
  var far = 1.0;
  for (var axis = 0u; axis < 3u; axis = axis + 1u) {
    let o = origin[axis];
    let inv = inverse[axis];
    let lo = nodeBounds[base + axis];
    let hi = nodeBounds[base + 3u + axis];
    if (inv == 0.0) {
      // Parallel to this axis: inside the slab or missing it entirely.
      if (o < lo || o > hi) { return false; }
      continue;
    }
    var low = (lo - o) * inv;
    var high = (hi - o) * inv;
    if (low > high) { let swap = low; low = high; high = swap; }
    near = max(near, low);
    far = min(far, high);
    if (near > far) { return false; }
  }
  return true;
}

fn triangleBlocks(index: u32, origin: vec3<f32>, direction: vec3<f32>) -> bool {
  let offset = index * 9u;
  let v0 = vec3<f32>(triangles[offset], triangles[offset + 1u], triangles[offset + 2u]);
  let e1 = vec3<f32>(triangles[offset + 3u], triangles[offset + 4u], triangles[offset + 5u]) - v0;
  let e2 = vec3<f32>(triangles[offset + 6u], triangles[offset + 7u], triangles[offset + 8u]) - v0;
  let h = cross(direction, e2);
  let det = dot(e1, h);
  if (det > -${DET_EPSILON} && det < ${DET_EPSILON}) { return false; }
  // The back of a single-sided surface is not drawn, so it blocks nothing.
  if (doubleSided[index] == 0u && det < 0.0) { return false; }
  let inv = 1.0 / det;
  let s = origin - v0;
  let u = inv * dot(s, h);
  if (u < 0.0 || u > 1.0) { return false; }
  let q = cross(s, e1);
  let v = inv * dot(direction, q);
  if (v < 0.0 || u + v > 1.0) { return false; }
  let hit = inv * dot(e2, q);
  return hit > ${END_EPSILON} && hit < ${1 - END_EPSILON};
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let segment = id.x;
  if (segment >= counts.x) { return; }
  let base = segment * 6u;
  let origin = vec3<f32>(segments[base], segments[base + 1u], segments[base + 2u]);
  let destination = vec3<f32>(segments[base + 3u], segments[base + 4u], segments[base + 5u]);
  let direction = destination - origin;
  blocked[segment] = 0u;
  if (length(direction) < ${END_EPSILON}) { return; }
  // Zero stands for "parallel" here; the CPU carries an infinity instead and
  // both take the same branch on it.
  let inverse = vec3<f32>(
    select(1.0 / direction.x, 0.0, direction.x == 0.0),
    select(1.0 / direction.y, 0.0, direction.y == 0.0),
    select(1.0 / direction.z, 0.0, direction.z == 0.0),
  );
  var stack: array<u32, ${STACK_DEPTH}>;
  var depth = 0u;
  stack[depth] = 0u;
  depth = depth + 1u;
  while (depth > 0u) {
    depth = depth - 1u;
    let node = stack[depth];
    if (!slabs(node, origin, inverse)) { continue; }
    let leaf = nodeMeta[node];
    if (leaf.count == 0) {
      // Interior: left is adjacent, right is recorded. An overflowing stack
      // gives up and reports clear, which admits and never hides.
      if (depth + 2u >= ${STACK_DEPTH}u) { return; }
      stack[depth] = node + 1u;
      stack[depth + 1u] = u32(leaf.right);
      depth = depth + 2u;
      continue;
    }
    let first = u32(leaf.first);
    for (var t = 0u; t < u32(leaf.count); t = t + 1u) {
      if (triangleBlocks(first + t, origin, direction)) {
        blocked[segment] = 1u;
        return;
      }
    }
  }
}
`;

/*
 * Usage flags as the WebGPU specification fixes them, rather than read off
 * the runtime's globals. A device can be injected from Node, where those
 * globals do not exist -- and a backend that needed browser globals could not
 * be handed a headless Dawn device, which is the whole point of injecting
 * one.
 */
const USAGE_MAP_READ = 0x0001;
const USAGE_COPY_SRC = 0x0004;
const USAGE_COPY_DST = 0x0008;
const USAGE_UNIFORM = 0x0040;
const USAGE_STORAGE = 0x0080;
const MAP_MODE_READ = 0x0001;

/** Rounds up to the 4-byte-per-element stride a storage buffer needs. */
function storageBytes(elements: number): number {
  return Math.max(4, elements * 4);
}

/**
 * Uploads the hierarchy once and keeps it resident.
 *
 * The device is injected rather than requested here: this package has no
 * WebGPU dependency, and the caller -- the bake, a browser, a test -- owns
 * which device it is and when it goes away.
 */
export async function createGpuOccluderBackend(
  device: GPUDevice,
  bvh: OccluderBvh,
  options: { maxSegmentsPerBatch?: number } = {}
): Promise<GpuOccluderBackend> {
  if (bvh.triangleCount === 0 || bvh.nodeCount === 0) {
    throw new ShadoGpuBackendUnsupported('hierarchy is empty');
  }
  if (bvh.maxDepth >= STACK_DEPTH) {
    throw new ShadoGpuBackendUnsupported(
      `hierarchy is ${bvh.maxDepth} deep and the shader stack holds ${STACK_DEPTH}`
    );
  }
  const limit = Math.min(
    device.limits.maxStorageBufferBindingSize,
    device.limits.maxBufferSize
  );
  const triangleBytes = storageBytes(bvh.triangleCount * 9);
  const boundsBytes = storageBytes(bvh.nodeCount * 6);
  const metaBytes = storageBytes(bvh.nodeCount * 4);
  const sidedBytes = storageBytes(bvh.triangleCount);
  for (const [name, bytes] of [
    ['triangles', triangleBytes],
    ['node bounds', boundsBytes],
    ['node meta', metaBytes],
    ['sidedness', sidedBytes],
  ] as const) {
    if (bytes > limit) {
      throw new ShadoGpuBackendUnsupported(
        `${name} need ${bytes} bytes and the device binds at most ${limit}`
      );
    }
  }

  const uploadStarted = performance.now();
  /*
   * f64 to f32 on the way in. This is the whole reason the backend proposes
   * rather than decides: a corner rounded here is a corner the CPU does not
   * have, and the disagreement shows up on grazing segments.
   */
  const triangles = Float32Array.from(bvh.triangles.subarray(0, bvh.triangleCount * 9));
  const bounds = Float32Array.from(bvh.nodeBounds.subarray(0, bvh.nodeCount * 6));
  /*
   * Bounds are widened to the f32 box that contains the f64 one. A node whose
   * bounds rounded inward would let a segment skip a subtree the CPU walks,
   * which is a missed blocker and the one rounding error that is not merely
   * a disagreement.
   */
  for (let node = 0; node < bvh.nodeCount; node += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const min = bvh.nodeBounds[node * 6 + axis]!;
      const max = bvh.nodeBounds[node * 6 + 3 + axis]!;
      if (bounds[node * 6 + axis]! > min) bounds[node * 6 + axis] = Math.fround(min - Math.abs(min) * 1e-6 - 1e-6);
      if (bounds[node * 6 + 3 + axis]! < max) bounds[node * 6 + 3 + axis] = Math.fround(max + Math.abs(max) * 1e-6 + 1e-6);
    }
  }
  const meta = new Int32Array(bvh.nodeCount * 4);
  for (let node = 0; node < bvh.nodeCount; node += 1) {
    meta[node * 4] = bvh.nodeMeta[node * 3]!;
    meta[node * 4 + 1] = bvh.nodeMeta[node * 3 + 1]!;
    meta[node * 4 + 2] = bvh.nodeMeta[node * 3 + 2]!;
  }
  const sided = Uint32Array.from(bvh.doubleSided.subarray(0, bvh.triangleCount));

  /*
   * Written in chunks. A single multi-megabyte `writeBuffer` aborts the
   * process on Dawn rather than raising -- the staging allocation is not a
   * validation error, so there is nothing to catch. A zone's triangle buffer
   * is tens of megabytes, which is well past where that starts.
   */
  const WRITE_CHUNK_BYTES = 4 << 20;
  const storage = (data: ArrayBufferView, label: string): GPUBuffer => {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const buffer = device.createBuffer({
      label,
      size: Math.max(4, data.byteLength),
      usage: USAGE_STORAGE | USAGE_COPY_DST,
    });
    for (let offset = 0; offset < bytes.byteLength; offset += WRITE_CHUNK_BYTES) {
      const chunk = bytes.subarray(offset, Math.min(offset + WRITE_CHUNK_BYTES, bytes.byteLength));
      device.queue.writeBuffer(buffer, offset, chunk as never, 0, chunk.byteLength);
    }
    return buffer;
  };
  const boundsBuffer = storage(bounds, 'pvs-node-bounds');
  const metaBuffer = storage(meta, 'pvs-node-meta');
  const triangleBuffer = storage(triangles, 'pvs-triangles');
  const sidedBuffer = storage(sided, 'pvs-sidedness');

  const maxSegmentsPerBatch = Math.max(
    WORKGROUP,
    Math.min(
      options.maxSegmentsPerBatch ?? 1 << 16,
      Math.floor(limit / (6 * 4)),
      Math.floor(limit / 4)
    )
  );
  const segmentBuffer = device.createBuffer({
    label: 'pvs-segments',
    size: maxSegmentsPerBatch * 6 * 4,
    usage: USAGE_STORAGE | USAGE_COPY_DST,
  });
  const resultBuffer = device.createBuffer({
    label: 'pvs-blocked',
    size: maxSegmentsPerBatch * 4,
    usage: USAGE_STORAGE | USAGE_COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    label: 'pvs-readback',
    size: maxSegmentsPerBatch * 4,
    usage: USAGE_MAP_READ | USAGE_COPY_DST,
  });
  const countBuffer = device.createBuffer({
    label: 'pvs-counts',
    size: 16,
    usage: USAGE_UNIFORM | USAGE_COPY_DST,
  });

  const pipeline = device.createComputePipeline({
    label: 'pvs-segment-occlusion',
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: 'main' },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: boundsBuffer } },
      { binding: 1, resource: { buffer: metaBuffer } },
      { binding: 2, resource: { buffer: triangleBuffer } },
      { binding: 3, resource: { buffer: sidedBuffer } },
      { binding: 4, resource: { buffer: segmentBuffer } },
      { binding: 5, resource: { buffer: resultBuffer } },
      { binding: 6, resource: { buffer: countBuffer } },
    ],
  });

  const stats: GpuBackendStats = {
    residentBytes: boundsBytes + metaBytes + triangleBytes + sidedBytes,
    uploadMs: performance.now() - uploadStarted,
    dispatchMs: 0,
    readbackMs: 0,
    segments: 0,
    batches: 0,
    proposedBlocked: 0,
  };
  const results = new Uint8Array(maxSegmentsPerBatch);
  const counts = new Uint32Array(4);
  let disposed = false;

  return {
    stats,
    maxSegmentsPerBatch,
    async blockedBatch(segmentData: Float32Array, count: number): Promise<Uint8Array> {
      if (disposed) throw new Error('Shado GPU occluder backend was disposed');
      if (count > maxSegmentsPerBatch) {
        throw new ShadoGpuBackendUnsupported(
          `batch of ${count} exceeds the ${maxSegmentsPerBatch} this device holds`
        );
      }
      const dispatchStarted = performance.now();
      counts[0] = count;
      device.queue.writeBuffer(countBuffer, 0, counts as never);
      device.queue.writeBuffer(segmentBuffer, 0, segmentData as never, 0, count * 6);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
      pass.end();
      encoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, count * 4);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      stats.dispatchMs += performance.now() - dispatchStarted;

      const readbackStarted = performance.now();
      await readbackBuffer.mapAsync(MAP_MODE_READ, 0, count * 4);
      const mapped = new Uint32Array(readbackBuffer.getMappedRange(0, count * 4));
      let proposed = 0;
      for (let index = 0; index < count; index += 1) {
        const blocked = mapped[index] === 1 ? 1 : 0;
        results[index] = blocked;
        proposed += blocked;
      }
      readbackBuffer.unmap();
      stats.readbackMs += performance.now() - readbackStarted;
      stats.segments += count;
      stats.batches += 1;
      stats.proposedBlocked += proposed;
      return results;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      for (const buffer of [
        boundsBuffer,
        metaBuffer,
        triangleBuffer,
        sidedBuffer,
        segmentBuffer,
        resultBuffer,
        readbackBuffer,
        countBuffer,
      ]) {
        buffer.destroy();
      }
    },
  };
}
