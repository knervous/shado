/**
 * A WebGPU compute backend for the bake's segment queries, over the same
 * structures the CPU walks: the flat occluder hierarchy, every prototype's own
 * hierarchy, and the top level over their placements.
 *
 * What it is allowed to decide: nothing that hides geometry, on its own. The
 * shader runs in f32 where the CPU runs in f64, and applies each placement's
 * inverse transform in f32 too, so the two can disagree on a segment that
 * grazes a surface. So it PROPOSES: a pair it finds a clear segment for is
 * admitted outright -- admitting is always safe -- and a pair it calls
 * occluded is handed to the CPU, whose answer is the one used. GPU
 * uncertainty admits.
 *
 * The device is injected: this package has no WebGPU dependency, and the
 * caller -- the bake, a browser, a test -- owns which device it is and when it
 * goes away.
 */
import type { OccluderBvh } from './occluder-bvh';
import type { InstancedOccluders } from './occluder-instances';
import type { ShadoVisibilityPairBatch, ShadoVisibilitySegmentBackend } from './visibility';

/** Raised when a device cannot host this scene. Always a fallback, never a failure. */
export class ShadoGpuBackendUnsupported extends Error {
  constructor(reason: string) {
    super(`Shado GPU occluder backend unsupported: ${reason}`);
    this.name = 'ShadoGpuBackendUnsupported';
  }
}

export type GpuBackendStats = {
  /** Bytes resident on the device for the immutable scene. */
  readonly residentBytes: number;
  /** Milliseconds spent converting and uploading, once, before any query. */
  uploadMs: number;
  /** Milliseconds inside submit-and-wait, summed over dispatches. */
  dispatchMs: number;
  /** Milliseconds spent mapping and copying results back. */
  readbackMs: number;
  segments: number;
  dispatches: number;
  /** Segments the shader proposed as blocked. */
  proposedBlocked: number;
};

export type GpuOccluderBackend = {
  /**
   * One byte per segment: 1 where the shader found a blocker. The returned
   * array is the caller's own.
   */
  blockedBatch(segments: Float32Array, count: number): Promise<Uint8Array>;
  readonly stats: GpuBackendStats;
  readonly maxSegmentsPerBatch: number;
  /** Set once the device is lost or the backend stopped; every later call refuses. */
  readonly failure: string | null;
  dispose(): void;
};

export type GpuBackendOptions = {
  /** Upper bound on segments per dispatch; clamped to what the device allows. */
  maxSegmentsPerBatch?: number;
  /** Ceiling on bytes this backend may place on the device. */
  maxDeviceBytes?: number;
  /**
   * Polled before upload, before every dispatch and after every readback.
   * A reason stops the backend: the current call answers conservatively and
   * every later one refuses.
   */
  stopReason?: () => string | null;
};

/*
 * The CPU's own constants, restated for the shader. Literal in the source
 * rather than uniform-fed, so a change on one side shows up as a differential
 * failure rather than as a silently divergent run.
 */
const END_EPSILON = 1e-3;
const DET_EPSILON = 1e-12;
const STACK_DEPTH = 64;
const WORKGROUP = 64;
/** Floats per instance record: a 3x4 affine inverse, then four u32 fields. */
const INSTANCE_STRIDE = 16;

/*
 * Usage flags as the WebGPU specification fixes them, rather than read off
 * the runtime's globals: a device injected from Node has none.
 */
const USAGE_MAP_READ = 0x0001;
const USAGE_COPY_SRC = 0x0004;
const USAGE_COPY_DST = 0x0008;
const USAGE_UNIFORM = 0x0040;
const USAGE_STORAGE = 0x0080;
const MAP_MODE_READ = 0x0001;

/*
 * A single multi-megabyte writeBuffer aborts the process on Dawn rather than
 * raising -- the staging allocation is not a validation error, so there is
 * nothing to catch. Writes are chunked.
 */
const WRITE_CHUNK_BYTES = 4 << 20;

const SHADER = /* wgsl */ `
struct Meta { first: i32, count: i32, right: i32, pad: i32 };
struct Instance {
  row0: vec4<f32>,
  row1: vec4<f32>,
  row2: vec4<f32>,
  // root node, node offset, triangle offset, unused
  refs: vec4<u32>,
};
/*
 * Seven storage bindings, not eleven: devices commonly allow eight per stage.
 * Every box -- flat nodes, prototype nodes, top-level nodes, instance bounds
 * -- lives in one bounds buffer, every node record in one meta buffer, and
 * the instance order rides at the tail of the instance buffer. Offsets say
 * where each part starts.
 */
struct Counts {
  segments: u32,
  hasFlat: u32,
  hasInstanced: u32,
  topNodeOffset: u32,
  instanceBoundsOffset: u32,
  orderOffset: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read> bounds: array<f32>;
@group(0) @binding(1) var<storage, read> nodeMeta: array<Meta>;
@group(0) @binding(2) var<storage, read> triangles: array<f32>;
@group(0) @binding(3) var<storage, read> doubleSided: array<u32>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> segments: array<f32>;
@group(0) @binding(6) var<storage, read_write> blocked: array<u32>;
@group(0) @binding(7) var<uniform> counts: Counts;

fn slabs(base: u32, origin: vec3<f32>, inverse: vec3<f32>) -> bool {
  var near = 0.0;
  var far = 1.0;
  for (var axis = 0u; axis < 3u; axis = axis + 1u) {
    let o = origin[axis];
    let inv = inverse[axis];
    let lo = bounds[base + axis];
    let hi = bounds[base + 3u + axis];
    if (inv == 0.0) {
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

/** The instance order, stored as u32 bits at the tail of the instance buffer. */
fn orderAt(index: u32) -> u32 {
  let word = counts.orderOffset + index;
  let record = instances[word / 16u];
  let lane = word % 16u;
  if (lane < 4u) { return bitcast<u32>(record.row0[lane]); }
  if (lane < 8u) { return bitcast<u32>(record.row1[lane - 4u]); }
  if (lane < 12u) { return bitcast<u32>(record.row2[lane - 8u]); }
  return record.refs[lane - 12u];
}

fn inverseOf(direction: vec3<f32>) -> vec3<f32> {
  // Zero stands for "parallel"; the CPU carries an infinity and both branch alike.
  return vec3<f32>(
    select(1.0 / direction.x, 0.0, direction.x == 0.0),
    select(1.0 / direction.y, 0.0, direction.y == 0.0),
    select(1.0 / direction.z, 0.0, direction.z == 0.0),
  );
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

/*
 * Walks one hierarchy stored at an offset in the shared node and triangle
 * buffers. Child and triangle indices inside it are LOCAL, so both offsets
 * are added. Returns 1 blocked, 0 clear, 2 gave up (stack) -- and giving up
 * reads as clear, which admits.
 */
fn walk(root: u32, nodeOffset: u32, triOffset: u32, origin: vec3<f32>, direction: vec3<f32>) -> u32 {
  if (length(direction) < ${END_EPSILON}) { return 0u; }
  let inverse = inverseOf(direction);
  var stack: array<u32, ${STACK_DEPTH}>;
  var depth = 1u;
  stack[0] = root;
  while (depth > 0u) {
    depth = depth - 1u;
    let node = stack[depth];
    if (!slabs(node * 6u, origin, inverse)) { continue; }
    let entry = nodeMeta[node];
    if (entry.count == 0) {
      if (depth + 2u >= ${STACK_DEPTH}u) { return 2u; }
      stack[depth] = node + 1u;
      stack[depth + 1u] = u32(entry.right) + nodeOffset;
      depth = depth + 2u;
      continue;
    }
    let first = u32(entry.first) + triOffset;
    for (var t = 0u; t < u32(entry.count); t = t + 1u) {
      if (triangleBlocks(first + t, origin, direction)) { return 1u; }
    }
  }
  return 0u;
}

fn instancedBlocked(origin: vec3<f32>, destination: vec3<f32>) -> bool {
  let direction = destination - origin;
  if (length(direction) < ${END_EPSILON}) { return false; }
  let inverse = inverseOf(direction);
  var stack: array<u32, ${STACK_DEPTH}>;
  var depth = 1u;
  stack[0] = 0u;
  while (depth > 0u) {
    depth = depth - 1u;
    let node = stack[depth];
    let topNode = counts.topNodeOffset + node;
    if (!slabs(topNode * 6u, origin, inverse)) { continue; }
    let entry = nodeMeta[topNode];
    if (entry.count == 0) {
      if (depth + 2u >= ${STACK_DEPTH}u) { return false; }
      stack[depth] = node + 1u;
      stack[depth + 1u] = u32(entry.right);
      depth = depth + 2u;
      continue;
    }
    for (var k = 0u; k < u32(entry.count); k = k + 1u) {
      let id = orderAt(u32(entry.first) + k);
      if (!slabs(counts.instanceBoundsOffset + id * 6u, origin, inverse)) { continue; }
      let placed = instances[id];
      /*
       * Into the prototype's own space. The parameter rides along unchanged
       * under an affine map, so the endpoint epsilon keeps its meaning --
       * and the inverse has already undone any mirroring, so the facing
       * test is NOT flipped again.
       */
      let a = vec3<f32>(
        dot(placed.row0.xyz, origin) + placed.row0.w,
        dot(placed.row1.xyz, origin) + placed.row1.w,
        dot(placed.row2.xyz, origin) + placed.row2.w);
      let b = vec3<f32>(
        dot(placed.row0.xyz, destination) + placed.row0.w,
        dot(placed.row1.xyz, destination) + placed.row1.w,
        dot(placed.row2.xyz, destination) + placed.row2.w);
      if (walk(placed.refs.x, placed.refs.y, placed.refs.z, a, b - a) == 1u) { return true; }
    }
  }
  return false;
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  // Two-dimensional dispatch, so a batch is not capped at one row of workgroups.
  let segment = id.x + id.y * groups.x * ${WORKGROUP}u;
  if (segment >= counts.segments) { return; }
  let base = segment * 6u;
  let origin = vec3<f32>(segments[base], segments[base + 1u], segments[base + 2u]);
  let destination = vec3<f32>(segments[base + 3u], segments[base + 4u], segments[base + 5u]);
  var result = 0u;
  if (counts.hasFlat == 1u && walk(0u, 0u, 0u, origin, destination - origin) == 1u) { result = 1u; }
  if (result == 0u && counts.hasInstanced == 1u && instancedBlocked(origin, destination)) { result = 1u; }
  blocked[segment] = result;
}
`;

/** Floats rounded OUTWARD, so an f32 box always contains the f64 one it came from. */
function widenInto(source: ArrayLike<number>, sourceOffset: number, out: Float32Array, outOffset: number): void {
  for (let axis = 0; axis < 3; axis += 1) {
    const min = source[sourceOffset + axis]!;
    const max = source[sourceOffset + 3 + axis]!;
    let low = Math.fround(min);
    let high = Math.fround(max);
    /*
     * A box that rounded inward would let a segment skip a subtree the CPU
     * walks -- a missed blocker, and the one rounding error that is not just
     * a disagreement. Widened by a relative and an absolute step.
     */
    if (low > min) low = Math.fround(min - Math.abs(min) * 1e-6 - 1e-6);
    if (high < max) high = Math.fround(max + Math.abs(max) * 1e-6 + 1e-6);
    out[outOffset + axis] = low;
    out[outOffset + 3 + axis] = high;
  }
}

type SceneArrays = {
  bounds: Float32Array;
  meta: Int32Array;
  triangles: Float32Array;
  doubleSided: Uint32Array;
  instances: Float32Array;
  topNodeOffset: number;
  instanceBoundsOffset: number;
  orderOffset: number;
  hasFlat: boolean;
  hasInstanced: boolean;
};

/** Measures, without allocating, what a scene would need on the device. */
function measureScene(flat: OccluderBvh | null, instanced: InstancedOccluders | null) {
  let nodes = flat ? flat.nodeCount : 0;
  let tris = flat ? flat.triangleCount : 0;
  let maxDepth = flat ? flat.maxDepth : 0;
  if (instanced) {
    for (const prototype of instanced.prototypes) {
      nodes += prototype.nodeCount;
      tris += prototype.triangleCount;
      maxDepth = Math.max(maxDepth, prototype.maxDepth);
    }
  }
  const topNodes = instanced ? instanced.nodeCount : 0;
  const instanceCount = instanced ? instanced.instanceCount : 0;
  const orderLength = instanced ? instanced.order.length : 0;
  const bytes = {
    bounds: (nodes + topNodes + instanceCount) * 6 * 4,
    meta: (nodes + topNodes) * 16,
    triangles: tris * 9 * 4,
    doubleSided: tris * 4,
    instances: (instanceCount * INSTANCE_STRIDE + orderLength) * 4 + 64,
  };
  return { nodes, tris, maxDepth, topNodes, instanceCount, bytes };
}

/** Converts the CPU structures into the flat buffers the shader reads. */
function convertScene(flat: OccluderBvh | null, instanced: InstancedOccluders | null): SceneArrays {
  const measured = measureScene(flat, instanced);
  const topNodeOffset = measured.nodes;
  const instanceBoundsOffset = (measured.nodes + measured.topNodes) * 6;
  const bounds = new Float32Array(Math.max(6, (measured.nodes + measured.topNodes + measured.instanceCount) * 6));
  const meta = new Int32Array(Math.max(4, (measured.nodes + measured.topNodes) * 4));
  const triangles = new Float32Array(Math.max(9, measured.tris * 9));
  const doubleSided = new Uint32Array(Math.max(1, measured.tris));
  let nodeCursor = 0;
  let triCursor = 0;
  const append = (bvh: OccluderBvh): { root: number; nodeOffset: number; triOffset: number } => {
    const nodeOffset = nodeCursor;
    const triOffset = triCursor;
    for (let node = 0; node < bvh.nodeCount; node += 1) {
      widenInto(bvh.nodeBounds, node * 6, bounds, (nodeOffset + node) * 6);
      meta[(nodeOffset + node) * 4] = bvh.nodeMeta[node * 3]!;
      meta[(nodeOffset + node) * 4 + 1] = bvh.nodeMeta[node * 3 + 1]!;
      meta[(nodeOffset + node) * 4 + 2] = bvh.nodeMeta[node * 3 + 2]!;
    }
    triangles.set(bvh.triangles.subarray(0, bvh.triangleCount * 9), triOffset * 9);
    for (let index = 0; index < bvh.triangleCount; index += 1) {
      doubleSided[triOffset + index] = bvh.doubleSided[index]!;
    }
    nodeCursor += bvh.nodeCount;
    triCursor += bvh.triangleCount;
    return { root: nodeOffset, nodeOffset, triOffset };
  };
  // The flat hierarchy goes first, so its root is node 0 and its offsets zero.
  const hasFlat = flat !== null && flat.triangleCount > 0;
  if (flat && hasFlat) append(flat);
  const hasInstanced = instanced !== null && instanced.order.length > 0;
  const instanceCount = measured.instanceCount;
  const orderOffset = instanceCount * INSTANCE_STRIDE;
  const orderLength = instanced ? instanced.order.length : 0;
  // Rounded up to whole 16-float records, so the shader reads full structs.
  const instanceFloats = Math.max(INSTANCE_STRIDE, Math.ceil((orderOffset + orderLength) / INSTANCE_STRIDE) * INSTANCE_STRIDE);
  const instances = new Float32Array(instanceFloats);
  if (instanced && hasInstanced) {
    const placedAt = instanced.prototypes.map((prototype) => append(prototype));
    for (let node = 0; node < instanced.nodeCount; node += 1) {
      widenInto(instanced.nodeBounds, node * 6, bounds, (topNodeOffset + node) * 6);
      meta[(topNodeOffset + node) * 4] = instanced.nodeMeta[node * 3]!;
      meta[(topNodeOffset + node) * 4 + 1] = instanced.nodeMeta[node * 3 + 1]!;
      meta[(topNodeOffset + node) * 4 + 2] = instanced.nodeMeta[node * 3 + 2]!;
    }
    const words = new Uint32Array(instances.buffer);
    words.set(instanced.order, orderOffset);
    for (let id = 0; id < instanceCount; id += 1) {
      if (!instanced.instanceValid[id]) continue;
      widenInto(instanced.instanceBounds, id * 6, bounds, instanceBoundsOffset + id * 6);
      const inverse = instanced.instanceInverse;
      const base = id * 16;
      const out = id * INSTANCE_STRIDE;
      // Rows of the column-major inverse: local = row . p + row.w.
      for (let row = 0; row < 3; row += 1) {
        instances[out + row * 4] = inverse[base + row]!;
        instances[out + row * 4 + 1] = inverse[base + 4 + row]!;
        instances[out + row * 4 + 2] = inverse[base + 8 + row]!;
        instances[out + row * 4 + 3] = inverse[base + 12 + row]!;
      }
      const at = placedAt[instanced.instancePrototype[id]!]!;
      words[out + 12] = at.root;
      words[out + 13] = at.nodeOffset;
      words[out + 14] = at.triOffset;
      words[out + 15] = 0;
    }
  }
  return {
    bounds, meta, triangles, doubleSided, instances,
    topNodeOffset, instanceBoundsOffset, orderOffset,
    hasFlat, hasInstanced,
  };
}

/**
 * Uploads a scene once and answers batches of segments against it.
 *
 * Every limit is checked before anything is allocated on the device. Every
 * resource created is destroyed on dispose, on a failed initialisation, and
 * when the device is lost.
 */
export async function createGpuSceneBackend(
  device: GPUDevice,
  flat: OccluderBvh | null,
  instanced: InstancedOccluders | null,
  options: GpuBackendOptions = {}
): Promise<GpuOccluderBackend> {
  const measured = measureScene(flat, instanced);
  const hasWork = (flat?.triangleCount ?? 0) > 0 || (instanced?.order.length ?? 0) > 0;
  if (!hasWork) throw new ShadoGpuBackendUnsupported('scene is empty');
  if (measured.maxDepth >= STACK_DEPTH) {
    throw new ShadoGpuBackendUnsupported(
      `hierarchy is ${measured.maxDepth} deep and the shader stack holds ${STACK_DEPTH}`
    );
  }
  const limits = device.limits;
  const bindLimit = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  for (const [name, bytes] of Object.entries(measured.bytes)) {
    if (bytes > bindLimit) {
      throw new ShadoGpuBackendUnsupported(`${name} need ${bytes} bytes and the device binds at most ${bindLimit}`);
    }
  }
  if (limits.maxStorageBuffersPerShaderStage !== undefined && limits.maxStorageBuffersPerShaderStage < 7) {
    throw new ShadoGpuBackendUnsupported(
      `the shader binds 7 storage buffers and the device allows ${limits.maxStorageBuffersPerShaderStage}`
    );
  }
  if (limits.maxComputeWorkgroupSizeX < WORKGROUP) {
    throw new ShadoGpuBackendUnsupported(`workgroup size ${WORKGROUP} exceeds ${limits.maxComputeWorkgroupSizeX}`);
  }
  const perDimension = limits.maxComputeWorkgroupsPerDimension;
  const maxSegmentsPerBatch = Math.max(
    WORKGROUP,
    Math.min(
      Math.floor(options.maxSegmentsPerBatch ?? 1 << 16),
      Math.floor(bindLimit / (6 * 4)),
      WORKGROUP * perDimension * perDimension
    )
  );
  const residentBytes = Object.values(measured.bytes).reduce((sum, bytes) => sum + Math.max(bytes, 16), 0);
  const perBatchBytes = maxSegmentsPerBatch * (6 * 4 + 4 + 4) + 16;
  if (options.maxDeviceBytes !== undefined && residentBytes + perBatchBytes > options.maxDeviceBytes) {
    throw new ShadoGpuBackendUnsupported(
      `the scene needs ${residentBytes + perBatchBytes} device bytes and the budget is ${options.maxDeviceBytes}`
    );
  }
  const stopped = (): string | null => options.stopReason?.() ?? null;
  const beforeUpload = stopped();
  if (beforeUpload) throw new ShadoGpuBackendUnsupported(`stopped before upload: ${beforeUpload}`);

  const created: GPUBuffer[] = [];
  let failure: string | null = null;
  void device.lost?.then((info) => {
    failure = `device lost: ${info.message || info.reason}`;
  });
  const destroyAll = (): void => {
    for (const buffer of created.splice(0)) buffer.destroy();
  };

  try {
    const uploadStarted = performance.now();
    const scene = convertScene(flat, instanced);
    const storage = (data: ArrayBufferView, label: string): GPUBuffer => {
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const buffer = device.createBuffer({ label, size: Math.max(16, bytes.byteLength), usage: USAGE_STORAGE | USAGE_COPY_DST });
      created.push(buffer);
      for (let offset = 0; offset < bytes.byteLength; offset += WRITE_CHUNK_BYTES) {
        const chunk = bytes.subarray(offset, Math.min(offset + WRITE_CHUNK_BYTES, bytes.byteLength));
        device.queue.writeBuffer(buffer, offset, chunk as never, 0, chunk.byteLength);
      }
      return buffer;
    };
    const boundsBuffer = storage(scene.bounds, 'pvs-bounds');
    const metaBuffer = storage(scene.meta, 'pvs-meta');
    const triangleBuffer = storage(scene.triangles, 'pvs-triangles');
    const sidedBuffer = storage(scene.doubleSided, 'pvs-sidedness');
    const instanceBuffer = storage(scene.instances, 'pvs-instances');
    const segmentBuffer = device.createBuffer({ label: 'pvs-segments', size: maxSegmentsPerBatch * 6 * 4, usage: USAGE_STORAGE | USAGE_COPY_DST });
    created.push(segmentBuffer);
    const resultBuffer = device.createBuffer({ label: 'pvs-blocked', size: maxSegmentsPerBatch * 4, usage: USAGE_STORAGE | USAGE_COPY_SRC });
    created.push(resultBuffer);
    const readbackBuffer = device.createBuffer({ label: 'pvs-readback', size: maxSegmentsPerBatch * 4, usage: USAGE_MAP_READ | USAGE_COPY_DST });
    created.push(readbackBuffer);
    const countBuffer = device.createBuffer({ label: 'pvs-counts', size: 32, usage: USAGE_UNIFORM | USAGE_COPY_DST });
    created.push(countBuffer);

    /*
     * Shader and pipeline errors are caught here, where they happen, instead
     * of surfacing three errors downstream through a command encoder.
     */
    device.pushErrorScope('validation');
    const module = device.createShaderModule({ label: 'pvs-segment-occlusion', code: SHADER });
    const pipeline = device.createComputePipeline({
      label: 'pvs-segment-occlusion',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        boundsBuffer, metaBuffer, triangleBuffer, sidedBuffer, instanceBuffer,
        segmentBuffer, resultBuffer, countBuffer,
      ].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const validation = await device.popErrorScope();
    if (validation) throw new ShadoGpuBackendUnsupported(`pipeline did not validate: ${validation.message}`);

    const stats: GpuBackendStats = {
      residentBytes,
      uploadMs: performance.now() - uploadStarted,
      dispatchMs: 0,
      readbackMs: 0,
      segments: 0,
      dispatches: 0,
      proposedBlocked: 0,
    };
    const counts = new Uint32Array([
      0,
      scene.hasFlat ? 1 : 0,
      scene.hasInstanced ? 1 : 0,
      scene.topNodeOffset,
      scene.instanceBoundsOffset,
      scene.orderOffset,
      0,
      0,
    ]);
    let busy = false;
    let disposed = false;

    const refuse = (): void => {
      if (disposed) throw new Error('Shado GPU occluder backend was disposed');
      if (failure) throw new Error(`Shado GPU occluder backend failed: ${failure}`);
      if (busy) {
        // Two calls would share one segment and one readback buffer.
        throw new Error('Shado GPU occluder backend does not accept concurrent batches');
      }
    };

    const backend: GpuOccluderBackend = {
      stats,
      maxSegmentsPerBatch,
      get failure() {
        return failure;
      },
      async blockedBatch(segmentData: Float32Array, count: number): Promise<Uint8Array> {
        refuse();
        if (!Number.isInteger(count) || count < 0) throw new RangeError(`segment count ${count} is not a count`);
        if (count > maxSegmentsPerBatch) {
          throw new RangeError(`batch of ${count} exceeds the ${maxSegmentsPerBatch} this device holds`);
        }
        if (segmentData.length < count * 6) {
          throw new RangeError(`${count} segments need ${count * 6} floats; got ${segmentData.length}`);
        }
        // Nothing to do is not a zero-size map.
        if (count === 0) return new Uint8Array(0);
        const reason = stopped();
        if (reason) {
          failure = `stopped: ${reason}`;
          throw new Error(`Shado GPU occluder backend stopped: ${reason}`);
        }
        busy = true;
        try {
          const dispatchStarted = performance.now();
          counts[0] = count;
          device.queue.writeBuffer(countBuffer, 0, counts as never);
          device.queue.writeBuffer(segmentBuffer, 0, segmentData as never, 0, count * 6);
          const groups = Math.ceil(count / WORKGROUP);
          const groupsX = Math.min(groups, perDimension);
          const groupsY = Math.ceil(groups / groupsX);
          device.pushErrorScope('validation');
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(groupsX, groupsY);
          pass.end();
          encoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, count * 4);
          device.queue.submit([encoder.finish()]);
          const submitError = await device.popErrorScope();
          if (submitError) {
            failure = `dispatch did not validate: ${submitError.message}`;
            throw new Error(failure);
          }
          await device.queue.onSubmittedWorkDone();
          stats.dispatchMs += performance.now() - dispatchStarted;

          const readbackStarted = performance.now();
          await readbackBuffer.mapAsync(MAP_MODE_READ, 0, count * 4);
          const out = new Uint8Array(count);
          try {
            const mapped = new Uint32Array(readbackBuffer.getMappedRange(0, count * 4));
            let proposed = 0;
            for (let index = 0; index < count; index += 1) {
              const value = mapped[index] === 1 ? 1 : 0;
              out[index] = value;
              proposed += value;
            }
            stats.proposedBlocked += proposed;
          } finally {
            readbackBuffer.unmap();
          }
          stats.readbackMs += performance.now() - readbackStarted;
          stats.segments += count;
          stats.dispatches += 1;
          const after = stopped();
          if (after) failure = `stopped: ${after}`;
          if (failure) throw new Error(`Shado GPU occluder backend failed: ${failure}`);
          return out;
        } finally {
          busy = false;
        }
      },
      dispose(): void {
        if (disposed) return;
        disposed = true;
        destroyAll();
      },
    };
    return backend;
  } catch (error) {
    // Partial initialisation leaves nothing behind.
    destroyAll();
    if (error instanceof ShadoGpuBackendUnsupported) throw error;
    throw new ShadoGpuBackendUnsupported(error instanceof Error ? error.message : String(error));
  }
}

/** The single-hierarchy form, kept for the segment benchmark and its tests. */
export function createGpuOccluderBackend(
  device: GPUDevice,
  bvh: OccluderBvh,
  options: GpuBackendOptions = {}
): Promise<GpuOccluderBackend> {
  if (bvh.triangleCount === 0 || bvh.nodeCount === 0) {
    return Promise.reject(new ShadoGpuBackendUnsupported('hierarchy is empty'));
  }
  return createGpuSceneBackend(device, bvh, null, options);
}

/** What a pair-level GPU run did, for the bake report. */
export type GpuVisibilityBackendStats = {
  pairs: number;
  /** Pairs the GPU found a clear segment for, admitted with no CPU work. */
  gpuAdmitted: number;
  /** Pairs the GPU proposed as occluded, each re-tested on the CPU. */
  confirmed: number;
  /** Of those, how many the CPU found a clear segment for: admitted. */
  falseBlockers: number;
  /** Segments evaluated on the GPU. */
  segments: number;
  /** Segments the CPU's early exit would not have evaluated. */
  speculativeSegments: number;
  /** Set when the backend stopped or failed; later pairs were answered on the CPU. */
  fellBackToCpu: string | null;
};

/**
 * The bake's pair backend on a GPU scene.
 *
 * Every segment of every pair is evaluated in one pass -- no early exit, so
 * speculative work is counted. A pair is admitted as soon as any of its
 * segments comes back clear. A pair the GPU calls occluded is re-tested on
 * the CPU, and only the CPU's "occluded" clears a bit. If the GPU stops or
 * fails mid-bake, the rest of the bake is answered on the CPU: the rows stay
 * exact, only the speed changes.
 */
export function createGpuVisibilityBackend(
  device: GPUDevice,
  options: GpuBackendOptions = {}
): ShadoVisibilitySegmentBackend & {
  readonly stats: GpuVisibilityBackendStats;
  readonly gpu: GpuOccluderBackend | null;
  dispose(): void;
} {
  const stats: GpuVisibilityBackendStats = {
    pairs: 0,
    gpuAdmitted: 0,
    confirmed: 0,
    falseBlockers: 0,
    segments: 0,
    speculativeSegments: 0,
    fellBackToCpu: null,
  };
  let gpu: GpuOccluderBackend | null = null;
  let sceneKey: unknown = null;

  const cpuOnly = (batch: ShadoVisibilityPairBatch) => {
    const occluded = new Uint8Array(batch.pairs.length);
    for (let index = 0; index < occluded.length; index += 1) occluded[index] = batch.cpuOccluded(index) ? 1 : 0;
    return { occluded, segments: 0 };
  };

  return {
    kind: 'gpu',
    stats,
    get gpu() {
      return gpu;
    },
    async resolve(batch) {
      stats.pairs += batch.pairs.length;
      if (stats.fellBackToCpu) return cpuOnly(batch);
      if (!gpu || sceneKey !== batch.flat) {
        gpu?.dispose();
        try {
          gpu = await createGpuSceneBackend(device, batch.flat, batch.instanced, options);
          sceneKey = batch.flat;
        } catch (error) {
          stats.fellBackToCpu = error instanceof Error ? error.message : String(error);
          return cpuOnly(batch);
        }
      }
      // Every segment of every pair, forward then back, laid end to end.
      const ranges: { forward: [number, number]; back: [number, number] }[] = [];
      let total = 0;
      for (const pair of batch.pairs) {
        const forward = pair.forward ? (pair.forward[0].length / 3) * (pair.forward[1].length / 3) : 0;
        const back = pair.back ? (pair.back[0].length / 3) * (pair.back[1].length / 3) : 0;
        ranges.push({ forward: [total, total + forward], back: [total + forward, total + forward + back] });
        total += forward + back;
      }
      const segments = new Float32Array(total * 6);
      let write = 0;
      const emit = (from: Float64Array, to: Float64Array): void => {
        // Target-major, eye-minor: the order the CPU's anyClearSegment walks.
        for (let t = to.length - 3; t >= 0; t -= 3) {
          for (let e = 0; e < from.length; e += 3) {
            segments[write++] = from[e]!; segments[write++] = from[e + 1]!; segments[write++] = from[e + 2]!;
            segments[write++] = to[t]!; segments[write++] = to[t + 1]!; segments[write++] = to[t + 2]!;
          }
        }
      };
      for (const pair of batch.pairs) {
        if (pair.forward) emit(pair.forward[0], pair.forward[1]);
        if (pair.back) emit(pair.back[0], pair.back[1]);
      }
      const blocked = new Uint8Array(total);
      try {
        for (let first = 0; first < total; first += gpu.maxSegmentsPerBatch) {
          const count = Math.min(gpu.maxSegmentsPerBatch, total - first);
          blocked.set(await gpu.blockedBatch(segments.subarray(first * 6, (first + count) * 6), count), first);
        }
      } catch (error) {
        stats.fellBackToCpu = error instanceof Error ? error.message : String(error);
        return cpuOnly(batch);
      }
      stats.segments += total;
      const occluded = new Uint8Array(batch.pairs.length);
      for (let index = 0; index < batch.pairs.length; index += 1) {
        const { forward, back } = ranges[index]!;
        const allBlocked = (range: [number, number]): boolean => {
          for (let s = range[0]; s < range[1]; s += 1) if (!blocked[s]) return false;
          return true;
        };
        const firstClear = (range: [number, number]): number => {
          for (let s = range[0]; s < range[1]; s += 1) if (!blocked[s]) return s - range[0] + 1;
          return range[1] - range[0];
        };
        // What the CPU's early exit would have spent on this pair.
        const cpuWouldSpend = allBlocked(forward)
          ? forward[1] - forward[0] + firstClear(back)
          : firstClear(forward);
        stats.speculativeSegments += Math.max(0, back[1] - forward[0] - cpuWouldSpend);
        if (!allBlocked(forward) || !allBlocked(back)) {
          stats.gpuAdmitted += 1;
          continue;
        }
        // Proposed occluded: only the CPU may clear the bit.
        stats.confirmed += 1;
        if (batch.cpuOccluded(index)) occluded[index] = 1;
        else stats.falseBlockers += 1;
      }
      return { occluded, segments: total };
    },
    dispose() {
      gpu?.dispose();
      gpu = null;
    },
  };
}
