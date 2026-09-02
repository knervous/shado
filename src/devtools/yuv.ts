/**
 * RGBA render target -> yuv420p, on the GPU.
 *
 * Measured at 2560x1440, feeding ffmpeg 60 frames:
 *
 *     rgba in  -> x264 (swscale converts)   23.5 ms/frame   14.7 MB/frame
 *     yuv420p in -> x264 (no conversion)     6.7 ms/frame    5.5 MB/frame
 *
 * So ~71% of what looked like "encoding" was actually colour conversion on the
 * CPU inside ffmpeg. Doing it in a compute shader removes that entirely, and
 * cuts the readback to 1.5 bytes per pixel instead of 4 — the same pass folds
 * in the vertical flip, so `-vf vflip` goes too.
 *
 * Note this is *not* GPU encoding. WebGPU has no encoder. It is the conversion
 * around the encoder, which turned out to be where the time was.
 *
 * Nothing here is platform-specific: it is WGSL and core WebGPU, so it behaves
 * the same on any backend Dawn targets.
 */

/** BT.709, limited range — the correct matrix for HD, and what the stream is tagged as. */
const SHADER = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  yWords: u32,      // u32s per Y row
  cWords: u32,      // u32s per chroma row
  uOffset: u32,     // word offset of the U plane
  vOffset: u32,     // word offset of the V plane
  flip: u32,
  _pad: u32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> planes: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn srcRow(y: u32) -> i32 {
  // The readback convention is bottom-up; folding the flip in here costs
  // nothing and saves a full-resolution pass downstream.
  if (params.flip == 1u) { return i32(params.height - 1u - y); }
  return i32(y);
}

fn texel(x: u32, y: u32) -> vec3<f32> {
  return textureLoad(src, vec2<i32>(i32(x), srcRow(y)), 0).rgb;
}

fn luma(c: vec3<f32>) -> f32 {
  return 16.0 + (0.1826 * c.r + 0.6142 * c.g + 0.0620 * c.b) * 255.0;
}
fn chromaB(c: vec3<f32>) -> f32 {
  return 128.0 + (-0.1006 * c.r - 0.3386 * c.g + 0.4392 * c.b) * 255.0;
}
fn chromaR(c: vec3<f32>) -> f32 {
  return 128.0 + (0.4392 * c.r - 0.3989 * c.g - 0.0403 * c.b) * 255.0;
}
fn pack(a: f32, b: f32, c: f32, d: f32) -> u32 {
  return (u32(clamp(a, 0.0, 255.0) + 0.5))
       | (u32(clamp(b, 0.0, 255.0) + 0.5) << 8u)
       | (u32(clamp(c, 0.0, 255.0) + 0.5) << 16u)
       | (u32(clamp(d, 0.0, 255.0) + 0.5) << 24u);
}

// One invocation writes one u32 = four horizontally adjacent luma samples.
// Storage buffers cannot address single bytes, which is what dictates this
// shape (and r8unorm storage textures are not core WebGPU).
@compute @workgroup_size(8, 8)
fn yPlane(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.yWords || id.y >= params.height) { return; }
  let x = id.x * 4u;
  planes[id.y * params.yWords + id.x] = pack(
    luma(texel(x, id.y)), luma(texel(x + 1u, id.y)),
    luma(texel(x + 2u, id.y)), luma(texel(x + 3u, id.y)),
  );
}

// One invocation writes four chroma samples, each averaged over a 2x2 source
// block — eight source columns and two source rows per invocation.
fn chroma(id: vec3<u32>, isU: bool) -> u32 {
  var out: array<f32, 4>;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let x = id.x * 8u + i * 2u;
    let y = id.y * 2u;
    let sum = texel(x, y) + texel(x + 1u, y) + texel(x, y + 1u) + texel(x + 1u, y + 1u);
    let avg = sum * 0.25;
    out[i] = select(chromaR(avg), chromaB(avg), isU);
  }
  return pack(out[0], out[1], out[2], out[3]);
}

@compute @workgroup_size(8, 8)
fn uPlane(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.cWords || id.y >= params.height / 2u) { return; }
  planes[params.uOffset + id.y * params.cWords + id.x] = chroma(id, true);
}

@compute @workgroup_size(8, 8)
fn vPlane(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.cWords || id.y >= params.height / 2u) { return; }
  planes[params.vOffset + id.y * params.cWords + id.x] = chroma(id, false);
}
`;

/**
 * WebGPU's constant objects are globals in a browser. Using the numbers
 * directly keeps this module independent of when the native globals are
 * installed.
 */
const USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;
const SHADER_STAGE_COMPUTE = 0x4;
const MAP_MODE_READ = 0x1;

/**
 * The slice of WebGPU this file uses, declared structurally.
 *
 * The ambient WebGPU globals resolve when the whole program is type-checked
 * together — Babylon's types pull them in — but not when the declaration build
 * compiles files in isolation, where they fail as "Cannot find name". Declaring
 * what we call keeps this file independent of which tsconfig is in play, the
 * same way `video/types.ts` avoids DOM types.
 */
export interface GpuTextureLike {
  createView(): unknown;
}
interface GpuBufferLike {
  destroy(): void;
  unmap(): void;
  getMappedRange(): ArrayBuffer;
  mapAsync(mode: number): Promise<void>;
}
interface GpuComputePassLike {
  setBindGroup(index: number, group: unknown): void;
  setPipeline(pipeline: unknown): void;
  dispatchWorkgroups(x: number, y: number): void;
  end(): void;
}
interface GpuCommandEncoderLike {
  beginComputePass(): GpuComputePassLike;
  copyBufferToBuffer(src: unknown, so: number, dst: unknown, dof: number, size: number): void;
  finish(): unknown;
}
export interface GpuDeviceLike {
  createShaderModule(descriptor: { code: string; label?: string }): unknown;
  createBuffer(descriptor: { size: number; usage: number }): GpuBufferLike;
  createBindGroupLayout(descriptor: unknown): unknown;
  createPipelineLayout(descriptor: unknown): unknown;
  createComputePipeline(descriptor: unknown): unknown;
  createBindGroup(descriptor: unknown): unknown;
  createCommandEncoder(descriptor?: { label?: string }): GpuCommandEncoderLike;
  queue: { writeBuffer(buffer: unknown, offset: number, data: ArrayBufferView): void; submit(buffers: unknown[]): void };
}

export interface YuvConverterOptions {
  width: number;
  height: number;
  /** Source rows run bottom-up and should be flipped. Defaults to true. */
  flip?: boolean;
  /** Conversions that may be in flight at once; sizes the staging ring. */
  depth?: number;
}

export interface YuvConverter {
  readonly width: number;
  readonly height: number;
  /** Bytes per converted frame: `width * height * 1.5`. */
  readonly byteLength: number;
  /**
   * Converts one texture. Safe to call again before the previous promise
   * settles, up to `depth` times — the staging buffers are a ring.
   *
   * The returned view is borrowed and valid until `depth` further calls.
   */
  convert(texture: GpuTextureLike): Promise<Uint8Array>;
  dispose(): void;
}

/**
 * Sizes that the planar layout can address with whole 32-bit words.
 *
 * Width must be a multiple of 8 (four luma samples per word, and eight source
 * columns per chroma word) and height a multiple of 2 (4:2:0 halves it).
 * H.264 already requires even dimensions, so this only adds the width rule.
 */
export function supportsGpuYuv(width: number, height: number): boolean {
  return width % 8 === 0 && height % 2 === 0;
}

export function createYuvConverter(device: GpuDeviceLike, options: YuvConverterOptions): YuvConverter {
  const { width, height } = options;
  if (!supportsGpuYuv(width, height)) {
    throw new Error(`GPU yuv420p needs width % 8 == 0 and height % 2 == 0; got ${width}x${height}`);
  }
  const depth = Math.max(1, Math.floor(options.depth ?? 1));
  const yWords = width / 4;
  const cWords = width / 8;
  const uOffset = yWords * height;
  const vOffset = uOffset + cWords * (height / 2);
  const words = vOffset + cWords * (height / 2);
  const byteLength = words * 4;

  const module = device.createShaderModule({ code: SHADER, label: 'shado-rgba-to-yuv420p' });
  const params = device.createBuffer({ size: 32, usage: USAGE.UNIFORM | USAGE.COPY_DST });
  device.queue.writeBuffer(
    params,
    0,
    new Uint32Array([width, height, yWords, cWords, uOffset, vOffset, options.flip === false ? 0 : 1, 0]),
  );

  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipelines = (['yPlane', 'uPlane', 'vPlane'] as const).map((entryPoint) =>
    device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } }),
  );

  // One plane buffer per in-flight conversion, plus its own mappable staging
  // buffer — the same ring discipline the RGBA readback uses.
  const slots = Array.from({ length: depth }, () => ({
    planes: device.createBuffer({ size: byteLength, usage: USAGE.STORAGE | USAGE.COPY_SRC }),
    staging: device.createBuffer({ size: byteLength, usage: USAGE.COPY_DST | USAGE.MAP_READ }),
    out: new Uint8Array(byteLength),
    bind: null as unknown,
    view: null as GpuTextureLike | null,
  }));
  let cursor = 0;
  let disposed = false;

  return {
    width,
    height,
    byteLength,
    async convert(texture: GpuTextureLike): Promise<Uint8Array> {
      if (disposed) throw new Error('This YUV converter is disposed');
      const slot = slots[cursor++ % depth]!;
      // The bind group pins the source view, so rebuild it whenever Babylon
      // hands back a different texture (a resize disposes the old one).
      if (slot.bind === null || slot.view !== texture) {
        slot.view = texture;
        slot.bind = device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: { buffer: slot.planes } },
            { binding: 2, resource: { buffer: params } },
          ],
        });
      }

      const encoder = device.createCommandEncoder({ label: 'rgba-to-yuv' });
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, slot.bind);
      const dispatches: Array<[number, number]> = [
        [Math.ceil(yWords / 8), Math.ceil(height / 8)],
        [Math.ceil(cWords / 8), Math.ceil(height / 2 / 8)],
        [Math.ceil(cWords / 8), Math.ceil(height / 2 / 8)],
      ];
      for (let i = 0; i < pipelines.length; i++) {
        pass.setPipeline(pipelines[i]!);
        pass.dispatchWorkgroups(dispatches[i]![0], dispatches[i]![1]);
      }
      pass.end();
      encoder.copyBufferToBuffer(slot.planes, 0, slot.staging, 0, byteLength);
      device.queue.submit([encoder.finish()]);

      await slot.staging.mapAsync(MAP_MODE_READ);
      slot.out.set(new Uint8Array(slot.staging.getMappedRange()));
      slot.staging.unmap();
      return slot.out;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      params.destroy();
      for (const slot of slots) { slot.planes.destroy(); slot.staging.destroy(); }
    },
  };
}
