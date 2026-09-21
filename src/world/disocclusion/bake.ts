/**
 * Bounded offline WebGPU stages of the disocclusion buffer (paper §3.1-3.2,
 * §4.1-4.2). The device is injected: Dawn in Node, the browser's adapter in
 * the proving ground. Nothing here is imported by the runtime.
 *
 * Deviations from the paper/reference source (see evidence doc):
 * - Dense [layer][y][x] buffers, not the sparse page table.
 * - The 64-bit depth|ID atomicMin is split into two complete raster passes:
 *   pass 1 atomicMin's an ordered depth key, pass 2 re-rasterizes identically
 *   and atomicMin's the triangle ID only where its depth equals the winner.
 * - Frustum rectangles are tabulated on the CPU (same functions the scalar
 *   reference uses) and one thread per open tile walks them, dispatched one
 *   source layer at a time, instead of prefix-sum load balancing.
 */
import { captureClipMatrix, frustumGrowth, layerFront, tileRangeX, tileRangeY, tileTanX, tileTanY, validateSettings } from './layers';
import { EMPTY_SAMPLE } from './reference';
import type {
  DisocclusionFrame,
  DisocclusionGeometry,
  DisocclusionLayers,
  DisocclusionMasks,
  DisocclusionSettings,
  DisocclusionStageTimings,
} from './types';

/* WebGPU usage/stage flags as numbers so this module needs no DOM globals. */
const BUF = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 } as const;
const STAGE = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } as const;
const TEX_RENDER_ATTACHMENT = 16;

export const DISOCCLUSION_MAX_BYTES = 64 * 1024 * 1024;

export class DisocclusionBakeError extends Error {
  constructor(
    message: string,
    public readonly stage: string,
    public readonly timings: DisocclusionStageTimings
  ) {
    super(message);
  }
}

export type DisocclusionBakeOptions = {
  maxBytes?: number;
  /** Whole-capture watchdog in ms. A timeout fails the bake; it never retries. */
  timeoutMs?: number;
  /** Polled between stages and source layers. */
  shouldStop?: () => boolean;
  now?: () => number;
};

export type DisocclusionCaptureResult = {
  layers: DisocclusionLayers;
  masks: DisocclusionMasks;
  timings: DisocclusionStageTimings;
  bytes: number;
};

type Tables = { rangeX: Int32Array; rangeY: Int32Array };

/** Unclamped frustum tile ranges for every (opening layer, target layer, tile). */
export function frustumTables(frame: DisocclusionFrame, settings: DisocclusionSettings): Tables {
  const tiles = settings.resolution / settings.tileSize;
  const n = settings.layers;
  const rangeX = new Int32Array(n * n * tiles * 2);
  const rangeY = new Int32Array(n * n * tiles * 2);
  for (let open = 0; open < n; open++) {
    const zOpen = layerFront(frame, n, open);
    for (let target = open + 1; target < n; target++) {
      const [growX, growY] = frustumGrowth(frame, zOpen, layerFront(frame, n, target + 1));
      for (let t = 0; t < tiles; t++) {
        const at = ((open * n + target) * tiles + t) * 2;
        const [x0, x1] = tileTanX(frame, settings, t);
        const [y0, y1] = tileTanY(frame, settings, t);
        const rx = tileRangeX(frame, settings, x0 - growX, x1 + growX);
        const ry = tileRangeY(frame, settings, y0 - growY, y1 + growY);
        rangeX[at] = rx[0];
        rangeX[at + 1] = rx[1];
        rangeY[at] = ry[0];
        rangeY[at + 1] = ry[1];
      }
    }
  }
  return { rangeX, rangeY };
}

const RASTER_WGSL = /* wgsl */ `
struct Params {
  clip: mat4x4<f32>,
  resolution: u32,
  layers: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> fronts: array<f32>;
@group(0) @binding(4) var<storage, read_write> depthKeys: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> ids: array<atomic<u32>>;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) depth: f32,
  @location(1) @interpolate(flat) tri: u32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let index = indices[vi];
  let p = vec3<f32>(positions[index * 3u], positions[index * 3u + 1u], positions[index * 3u + 2u]);
  let c = params.clip * vec4<f32>(p, 1.0);
  var o: VOut;
  o.pos = c;
  o.depth = c.w;
  o.tri = vi / 3u;
  return o;
}

// Layer by comparison against the CPU's layer fronts, so the raster and the
// CPU classifier agree on every boundary. -1 outside [near, far).
fn sampleIndex(pos: vec4<f32>, depth: f32) -> i32 {
  if (!(depth >= fronts[0]) || !(depth < fronts[params.layers])) { return -1; }
  var layer = 0u;
  loop {
    if (layer + 1u >= params.layers || depth < fronts[layer + 1u]) { break; }
    layer = layer + 1u;
  }
  let x = min(u32(pos.x), params.resolution - 1u);
  let y = min(u32(pos.y), params.resolution - 1u);
  return i32((layer * params.resolution + y) * params.resolution + x);
}

@fragment fn fsDepth(v: VOut) -> @location(0) vec4<f32> {
  let i = sampleIndex(v.pos, v.depth);
  if (i >= 0) { atomicMin(&depthKeys[u32(i)], bitcast<u32>(v.depth)); }
  return vec4<f32>(0.0);
}

@fragment fn fsId(v: VOut) -> @location(0) vec4<f32> {
  let i = sampleIndex(v.pos, v.depth);
  if (i >= 0 && atomicLoad(&depthKeys[u32(i)]) == bitcast<u32>(v.depth)) {
    atomicMin(&ids[u32(i)], v.tri);
  }
  return vec4<f32>(0.0);
}
`;

const TILE_WGSL = /* wgsl */ `
struct Grid {
  resolution: u32,
  tiles: u32,
  tileSize: u32,
  layers: u32,
};
struct Source { layer: u32, };
@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<storage, read> depthKeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(3) var<storage, read> rangeX: array<i32>;
@group(0) @binding(4) var<storage, read> rangeY: array<i32>;
@group(0) @binding(5) var<storage, read_write> mask: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> column: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> visible: array<u32>;
@group(1) @binding(0) var<uniform> source: Source;

@compute @workgroup_size(64) fn countTiles(@builtin(global_invocation_id) g: vec3<u32>) {
  let cells = grid.tiles * grid.tiles * grid.layers;
  if (g.x >= cells) { return; }
  let layer = g.x / (grid.tiles * grid.tiles);
  let rem = g.x % (grid.tiles * grid.tiles);
  let ty = rem / grid.tiles;
  let tx = rem % grid.tiles;
  var n = 0u;
  for (var sy = 0u; sy < grid.tileSize; sy++) {
    for (var sx = 0u; sx < grid.tileSize; sx++) {
      let y = ty * grid.tileSize + sy;
      let x = tx * grid.tileSize + sx;
      if (depthKeys[(layer * grid.resolution + y) * grid.resolution + x] != 0xffffffffu) { n = n + 1u; }
    }
  }
  counts[g.x] = n;
}

fn closed(layer: u32, y: i32, x: i32) -> bool {
  let t = i32(grid.tiles);
  if (x < 0 || y < 0 || x >= t || y >= t) { return false; }
  return counts[(layer * grid.tiles + u32(y)) * grid.tiles + u32(x)] >= grid.tileSize * grid.tileSize;
}

@compute @workgroup_size(64) fn propagate(@builtin(global_invocation_id) g: vec3<u32>) {
  let L = source.layer;
  if (g.x >= grid.tiles * grid.tiles || L + 1u >= grid.layers) { return; }
  let ty = i32(g.x / grid.tiles);
  let tx = i32(g.x % grid.tiles);
  if (closed(L, ty, tx)) { return; }
  let left = closed(L, ty, tx - 1);
  let right = closed(L, ty, tx + 1);
  let top = closed(L, ty - 1, tx);
  let bottom = closed(L, ty + 1, tx);
  let bit = 1u << L;
  if (!left && !right && !top && !bottom) {
    atomicOr(&column[g.x], bit);
    return;
  }
  let last = i32(grid.tiles) - 1;
  for (var J = L + 1u; J < grid.layers; J++) {
    let bx = ((L * grid.layers + J) * grid.tiles + u32(tx)) * 2u;
    let by = ((L * grid.layers + J) * grid.tiles + u32(ty)) * 2u;
    var minX = select(tx, rangeX[bx], left);
    var maxX = select(tx, rangeX[bx + 1u], right);
    var minY = select(ty, rangeY[by], top);
    var maxY = select(ty, rangeY[by + 1u], bottom);
    minX = max(minX, 0); minY = max(minY, 0);
    maxX = min(maxX, last); maxY = min(maxY, last);
    for (var y = minY; y <= maxY; y++) {
      for (var x = minX; x <= maxX; x++) {
        atomicOr(&mask[(J * grid.tiles + u32(y)) * grid.tiles + u32(x)], bit);
      }
    }
  }
}

fn lowBits(n: u32) -> u32 {
  if (n == 0u) { return 0u; }
  if (n >= 32u) { return 0xffffffffu; }
  return (1u << n) - 1u;
}

@compute @workgroup_size(64) fn gather(@builtin(global_invocation_id) g: vec3<u32>) {
  let per = grid.tiles * grid.tiles;
  if (g.x >= per * grid.layers) { return; }
  let layer = g.x / per;
  let need = lowBits(layer);
  let bits = (atomicLoad(&mask[g.x]) | atomicLoad(&column[g.x % per])) & need;
  visible[g.x] = select(0u, 1u, bits == need);
}
`;

/** Bytes this capture will allocate, before allocating any of it. */
export function captureBytes(settings: DisocclusionSettings, geometry: DisocclusionGeometry): number {
  const r = settings.resolution;
  const tiles = r / settings.tileSize;
  const n = settings.layers;
  const samples = r * r * n * 4;
  const cells = tiles * tiles * n * 4;
  const tables = n * n * tiles * 2 * 4 * 2;
  const geometryBytes = geometry.positions.byteLength + geometry.indices.byteLength;
  // depth + id, their readback copies, count/mask/visible + readback, column, tables, fronts, uniforms, target.
  return samples * 4 + cells * 6 + tiles * tiles * 4 * 2 + tables + (n + 1) * 4 + 64 * 256 + r * r + geometryBytes;
}

type GpuDevice = GPUDevice;

/**
 * Rasterize one directional capture and run the tile stages. Every resource
 * is destroyed in `finally`. Any failure throws DisocclusionBakeError with the
 * stage it reached and the timings so far; it never returns an empty result.
 */
export async function bakeDisocclusionCapture(
  device: GpuDevice,
  frame: DisocclusionFrame,
  settings: DisocclusionSettings,
  geometry: DisocclusionGeometry,
  options: DisocclusionBakeOptions = {}
): Promise<DisocclusionCaptureResult> {
  validateSettings(settings);
  const now = options.now ?? (() => performance.now());
  const timings: DisocclusionStageTimings = {};
  let stage = 'account';
  const started = now();
  const deadline = started + (options.timeoutMs ?? 10_000);
  const bytes = captureBytes(settings, geometry);
  const maxBytes = options.maxBytes ?? DISOCCLUSION_MAX_BYTES;
  const fail = (message: string): never => {
    timings.totalMs = now() - started;
    throw new DisocclusionBakeError(message, stage, { ...timings });
  };
  if (bytes > maxBytes) fail(`capture needs ${bytes} bytes, cap is ${maxBytes}`);
  if (geometry.indices.length % 3 || geometry.positions.length % 3) fail('geometry is not triangles');
  const triangles = geometry.indices.length / 3;
  if (triangles === 0) fail('capture has no occluder triangles; refusing to emit an empty PVS');
  const vertexCount = geometry.positions.length / 3;
  for (let i = 0; i < geometry.indices.length; i++) {
    if (geometry.indices[i]! >= vertexCount) fail(`index ${i} out of range`);
  }

  const r = settings.resolution;
  const tiles = r / settings.tileSize;
  const n = settings.layers;
  const sampleCount = r * r * n;
  const cellCount = tiles * tiles * n;
  const buffers: GPUBuffer[] = [];
  const textures: GPUTexture[] = [];
  let lost: string | null = null;
  void device.lost?.then(info => {
    lost = `device lost: ${info.reason} ${info.message}`;
  });
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  let scopes = 2;

  const checkpoint = async (name: string, begin: number) => {
    const pending = device.queue.onSubmittedWorkDone();
    const remaining = deadline - now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<'timeout'>(resolve => {
      timer = setTimeout(() => resolve('timeout'), Math.max(0, remaining));
    });
    const outcome = await Promise.race([pending.then(() => 'done' as const), timeout]);
    if (timer) clearTimeout(timer);
    timings[`${name}Ms`] = now() - begin;
    if (outcome === 'timeout') fail(`watchdog expired during ${name}`);
    if (lost) fail(lost);
    if (options.shouldStop?.()) fail(`cancelled after ${name}`);
  };
  const buffer = (size: number, usage: number, label: string) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label });
    buffers.push(b);
    return b;
  };
  const upload = (data: ArrayBufferView, usage: number, label: string) => {
    const b = buffer(data.byteLength, usage | BUF.COPY_DST, label);
    device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  };

  try {
    stage = 'upload';
    let begin = now();
    const fronts = new Float32Array(n + 1);
    for (let i = 0; i <= n; i++) fronts[i] = layerFront(frame, n, i);
    const params = new ArrayBuffer(80);
    new Float32Array(params, 0, 16).set(captureClipMatrix(frame));
    new Uint32Array(params, 64, 4).set([r, n, 0, 0]);
    const paramsBuffer = upload(new Uint8Array(params), BUF.UNIFORM, 'dpvs-params');
    const positions = upload(Float32Array.from(geometry.positions), BUF.STORAGE, 'dpvs-positions');
    const indices = upload(Uint32Array.from(geometry.indices), BUF.STORAGE, 'dpvs-indices');
    const frontsBuffer = upload(fronts, BUF.STORAGE, 'dpvs-fronts');
    const sentinel = new Uint32Array(sampleCount).fill(EMPTY_SAMPLE);
    const depth = upload(sentinel, BUF.STORAGE | BUF.COPY_SRC, 'dpvs-depth');
    const ids = upload(sentinel, BUF.STORAGE | BUF.COPY_SRC, 'dpvs-ids');
    const tables = frustumTables(frame, settings);
    const rangeX = upload(tables.rangeX, BUF.STORAGE, 'dpvs-range-x');
    const rangeY = upload(tables.rangeY, BUF.STORAGE, 'dpvs-range-y');
    const zeroCells = new Uint32Array(cellCount);
    const counts = upload(zeroCells, BUF.STORAGE | BUF.COPY_SRC, 'dpvs-counts');
    const mask = upload(zeroCells, BUF.STORAGE | BUF.COPY_SRC, 'dpvs-mask');
    const visible = upload(zeroCells, BUF.STORAGE | BUF.COPY_SRC, 'dpvs-visible');
    const column = upload(new Uint32Array(tiles * tiles), BUF.STORAGE | BUF.COPY_SRC, 'dpvs-column');
    const grid = upload(Uint32Array.from([r, tiles, settings.tileSize, n]), BUF.UNIFORM, 'dpvs-grid');
    const sourceStride = 256;
    const sourceData = new Uint32Array((sourceStride / 4) * n);
    for (let l = 0; l < n; l++) sourceData[(l * sourceStride) / 4] = l;
    const sourceBuffer = upload(sourceData, BUF.UNIFORM, 'dpvs-source-layer');
    await checkpoint('upload', begin);

    stage = 'raster';
    begin = now();
    const target = device.createTexture({
      size: [r, r],
      format: 'r8unorm',
      usage: TEX_RENDER_ATTACHMENT,
      label: 'dpvs-target',
    });
    textures.push(target);
    const rasterModule = device.createShaderModule({ code: RASTER_WGSL, label: 'dpvs-raster' });
    const rasterLayout = device.createBindGroupLayout({
      label: 'dpvs-raster',
      entries: [
        { binding: 0, visibility: STAGE.VERTEX | STAGE.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: STAGE.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: STAGE.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: STAGE.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: STAGE.FRAGMENT, buffer: { type: 'storage' } },
        { binding: 5, visibility: STAGE.FRAGMENT, buffer: { type: 'storage' } },
      ],
    });
    const rasterGroup = device.createBindGroup({
      layout: rasterLayout,
      entries: [paramsBuffer, positions, indices, frontsBuffer, depth, ids].map((b, binding) => ({
        binding,
        resource: { buffer: b },
      })),
    });
    const rasterPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [rasterLayout] });
    const pipeline = (entryPoint: string) =>
      device.createRenderPipeline({
        label: `dpvs-${entryPoint}`,
        layout: rasterPipelineLayout,
        vertex: { module: rasterModule, entryPoint: 'vs' },
        fragment: { module: rasterModule, entryPoint, targets: [{ format: 'r8unorm', writeMask: 0 }] },
        // No culling and no depth attachment: every layer keeps its own nearest sample.
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      });
    const rasterPass = (entryPoint: string) => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] }],
      });
      pass.setPipeline(pipeline(entryPoint));
      pass.setBindGroup(0, rasterGroup);
      pass.draw(triangles * 3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    };
    rasterPass('fsDepth');
    await checkpoint('rasterDepth', begin);
    begin = now();
    rasterPass('fsId');
    await checkpoint('rasterId', begin);

    stage = 'tiles';
    begin = now();
    const tileModule = device.createShaderModule({ code: TILE_WGSL, label: 'dpvs-tiles' });
    const storage = (binding: number, readOnly: boolean) => ({
      binding,
      visibility: STAGE.COMPUTE,
      buffer: { type: readOnly ? ('read-only-storage' as const) : ('storage' as const) },
    });
    const tileLayout = device.createBindGroupLayout({
      label: 'dpvs-tiles',
      entries: [
        { binding: 0, visibility: STAGE.COMPUTE, buffer: { type: 'uniform' } },
        storage(1, true),
        storage(2, false),
        storage(3, true),
        storage(4, true),
        storage(5, false),
        storage(6, false),
        storage(7, false),
      ],
    });
    const sourceLayout = device.createBindGroupLayout({
      label: 'dpvs-source',
      entries: [
        { binding: 0, visibility: STAGE.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 } },
      ],
    });
    const tileGroup = device.createBindGroup({
      layout: tileLayout,
      entries: [grid, depth, counts, rangeX, rangeY, mask, column, visible].map((b, binding) => ({
        binding,
        resource: { buffer: b },
      })),
    });
    const sourceGroup = device.createBindGroup({
      layout: sourceLayout,
      entries: [{ binding: 0, resource: { buffer: sourceBuffer, size: 16 } }],
    });
    const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [tileLayout, sourceLayout] });
    const compute = (entryPoint: string) =>
      device.createComputePipeline({ label: `dpvs-${entryPoint}`, layout: computeLayout, compute: { module: tileModule, entryPoint } });
    const dispatch = (pipelineEntry: GPUComputePipeline, threads: number, layer: number) => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipelineEntry);
      pass.setBindGroup(0, tileGroup);
      pass.setBindGroup(1, sourceGroup, [layer * sourceStride]);
      pass.dispatchWorkgroups(Math.ceil(threads / 64));
      pass.end();
      device.queue.submit([encoder.finish()]);
    };
    dispatch(compute('countTiles'), cellCount, 0);
    await checkpoint('count', begin);
    begin = now();
    const propagate = compute('propagate');
    for (let layer = 0; layer < n - 1; layer++) {
      dispatch(propagate, tiles * tiles, layer);
      // Bounded work per submit: every source layer is its own checkpoint.
      if ((layer & 7) === 7) await checkpoint(`propagate${layer}`, begin);
    }
    await checkpoint('propagate', begin);
    begin = now();
    dispatch(compute('gather'), cellCount, 0);
    await checkpoint('gather', begin);

    stage = 'readback';
    begin = now();
    const read = async (source: GPUBuffer, size: number) => {
      const staging = buffer(size, BUF.MAP_READ | BUF.COPY_DST, 'dpvs-readback');
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, staging, 0, size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(1 /* GPUMapMode.READ */);
      const out = new Uint32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return out;
    };
    const [depthOut, idOut, countOut, maskOut, columnOut, visibleOut] = await Promise.all([
      read(depth, sampleCount * 4),
      read(ids, sampleCount * 4),
      read(counts, cellCount * 4),
      read(mask, cellCount * 4),
      read(column, tiles * tiles * 4),
      read(visible, cellCount * 4),
    ]);
    scopes = 0;
    const oom = await device.popErrorScope();
    const validation = await device.popErrorScope();
    if (oom) fail(`out of memory: ${oom.message}`);
    if (validation) fail(`validation: ${validation.message}`);
    await checkpoint('readback', begin);
    timings.totalMs = now() - started;

    const state = new Uint8Array(cellCount);
    const full = settings.tileSize * settings.tileSize;
    for (let i = 0; i < cellCount; i++) state[i] = countOut[i] === 0 ? 0 : countOut[i]! >= full ? 2 : 1;
    return {
      layers: { settings, depth: depthOut, id: idOut },
      masks: {
        tilesX: tiles,
        tilesY: tiles,
        layers: n,
        count: countOut,
        state,
        mask: maskOut,
        column: columnOut,
        visible: Uint8Array.from(visibleOut),
      },
      timings,
      bytes,
    };
  } catch (error) {
    if (error instanceof DisocclusionBakeError) throw error;
    timings.totalMs = now() - started;
    throw new DisocclusionBakeError(error instanceof Error ? error.message : String(error), stage, { ...timings });
  } finally {
    while (scopes-- > 0) await device.popErrorScope().catch(() => null);
    for (const b of buffers) b.destroy();
    for (const t of textures) t.destroy();
  }
}
