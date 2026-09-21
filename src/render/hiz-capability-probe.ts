/**
 * Hi-Z capability probe (docs/pvs-hiz-prototype.md H0.5).
 *
 * `navigator.gpu` existing is not a working device. Before Hi-Z may reject
 * anything this proves, once, on the engine's own device:
 *   - the limits the passes need (storage buffers per stage, workgroup size,
 *     binding size for a 4K pyramid);
 *   - a compute shader can write indexed-indirect style arguments that a
 *     render pass then consumes with drawIndirect, and the result is visible
 *     (a 1x1 target turns red, read back).
 * Optional features (timestamp-query, non-zero firstInstance) are reported,
 * never required. Runs on its own command encoder, off the frame.
 */

export interface HiZCapabilityReport {
  readonly ok: boolean;
  readonly reason: string;
  readonly limits: Record<string, number>;
  readonly optional: { timestampQuery: boolean; indirectFirstInstance: boolean };
  readonly ms: number;
}

/** Storage buffers bound by the cull pass (the widest one). */
const REQUIRED_STORAGE_BUFFERS = 7;
const REQUIRED_WORKGROUP_X = 64;
/** A 3840x2160 pyramid of f32 (x ~1.34 for the chain). */
const REQUIRED_BINDING_BYTES = Math.ceil(3840 * 2160 * 4 * 1.34);

const COMPUTE = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
  args[0] = 3u; // vertexCount
  args[1] = 1u; // instanceCount
  args[2] = 0u; // firstVertex
  args[3] = 0u; // firstInstance
}`;

const RENDER = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }`;

export async function probeHiZCapability(device: GPUDevice, timeoutMs = 3000): Promise<HiZCapabilityReport> {
  const started = performance.now();
  const limits = {
    maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
  };
  const optional = {
    timestampQuery: device.features.has('timestamp-query' as GPUFeatureName),
    indirectFirstInstance: device.features.has('indirect-first-instance' as GPUFeatureName),
  };
  const fail = (reason: string): HiZCapabilityReport => ({ ok: false, reason, limits, optional, ms: performance.now() - started });
  if (limits.maxStorageBuffersPerShaderStage < REQUIRED_STORAGE_BUFFERS) {
    return fail(`maxStorageBuffersPerShaderStage ${limits.maxStorageBuffersPerShaderStage} < ${REQUIRED_STORAGE_BUFFERS}`);
  }
  if (limits.maxComputeWorkgroupSizeX < REQUIRED_WORKGROUP_X) {
    return fail(`maxComputeWorkgroupSizeX ${limits.maxComputeWorkgroupSizeX} < ${REQUIRED_WORKGROUP_X}`);
  }
  if (limits.maxStorageBufferBindingSize < REQUIRED_BINDING_BYTES) {
    return fail(`maxStorageBufferBindingSize ${limits.maxStorageBufferBindingSize} < ${REQUIRED_BINDING_BYTES} (4K pyramid)`);
  }

  const B = (globalThis as any).GPUBufferUsage;
  const T = (globalThis as any).GPUTextureUsage;
  const created: Array<{ destroy(): void }> = [];
  try {
    device.pushErrorScope('validation');
    const args = device.createBuffer({ size: 16, usage: B.STORAGE | B.INDIRECT });
    const target = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: T.RENDER_ATTACHMENT | T.COPY_SRC });
    const readback = device.createBuffer({ size: 256, usage: B.COPY_DST | B.MAP_READ });
    created.push(args, target, readback);
    const compute = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: COMPUTE }), entryPoint: 'main' },
    });
    const module = device.createShaderModule({ code: RENDER });
    const render = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
    const encoder = device.createCommandEncoder({ label: 'Hi-Z capability probe' });
    const cp = encoder.beginComputePass();
    cp.setPipeline(compute);
    cp.setBindGroup(0, device.createBindGroup({ layout: compute.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: args } }] }));
    cp.dispatchWorkgroups(1);
    cp.end();
    const rp = encoder.beginRenderPass({
      colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    rp.setPipeline(render);
    rp.drawIndirect(args, 0);
    rp.end();
    encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: 256 }, [1, 1]);
    device.queue.submit([encoder.finish()]);
    const validation = await device.popErrorScope();
    if (validation) return fail(`validation: ${validation.message}`);
    const mapped = await Promise.race([
      readback.mapAsync((globalThis as any).GPUMapMode.READ).then(() => true),
      new Promise<false>((resolve) => setTimeout(() => resolve(false), timeoutMs)),
    ]);
    if (!mapped) return fail(`compute -> indirect draw readback did not finish in ${timeoutMs} ms`);
    const pixel = new Uint8Array(readback.getMappedRange().slice(0, 4));
    readback.unmap();
    if (pixel[0] !== 255 || pixel[1] !== 0) return fail(`compute -> indirect draw produced ${Array.from(pixel).join(',')}, expected 255,0,0,255`);
    return { ok: true, reason: 'ok', limits, optional, ms: performance.now() - started };
  } catch (error) {
    return fail(error instanceof Error ? error.message : String(error));
  } finally {
    for (const resource of created) resource.destroy();
  }
}
