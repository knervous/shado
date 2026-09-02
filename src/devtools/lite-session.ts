/**
 * A preview session backed by Babylon Lite.
 *
 * Lite is a different library, not a smaller build of the same one: its API is
 * functional rather than class-based (`createEngine` / `stepScene`, no `Scene`
 * class), so this is a second implementation of the same contract rather than
 * a configuration of the first. What it shares with `session.ts` is the shape
 * the capture pipeline consumes — `scene`, `loadGlb`, `frameCamera`,
 * `captureRaw` — which is all `createSessionFrameSource` asks for.
 *
 * Three things about Lite's model are easy to get wrong, and each fails by
 * rendering the clear colour and nothing else — no error, no warning:
 *
 * 1. **`stepScene` does not draw.** Its own docs say it "does not record or
 *    submit any GPU work"; it is update-only, for null engines. Lite renders
 *    from a loop it owns: `registerScene` then `startEngine`.
 * 2. **Register the scene only once it is populated.** A scene registered while
 *    still empty does not pick up meshes added afterwards, so registration is
 *    deferred to the first capture.
 * 3. **`Vec3` is `{ x, y, z }`, not a tuple** — while `createDirectionalLight`
 *    *does* take a tuple, which makes the inconsistency easy to miss. An array
 *    camera target leaves `target.x` undefined and the view matrix NaN.
 *
 * `captureScreenshot` is also avoided: it never settles on a headless surface.
 * Reading the swapchain texture directly is both the fix and the better path,
 * since it keeps the GPU texture reachable for the yuv converter.
 *
 * It runs on exactly the same headless Dawn shims as the core session. Two of
 * those shims exist because of Lite specifically: the WebGPU constant globals
 * (`GPUShaderStage` and friends, which core Babylon carries its own copies of)
 * and the `copyExternalImageToTexture` translation (which core Babylon avoids
 * by branching on `byteLength`). See `headless-gpu.ts`.
 */

import {
  createHeadlessCanvas,
  installHeadlessWebGpu,
  installImageDecoder,
  type HeadlessGpu,
  type ImageDecoder,
} from './headless-gpu';
import type { CaptureFormat, CaptureStream, RawFrame } from './session';
import { createYuvConverter, supportsGpuYuv, type YuvConverter } from './yuv';

export interface LiteSessionOptions {
  width?: number;
  height?: number;
  /** Enables texture decoding. `sharp` satisfies this. */
  decodeImage?: ImageDecoder;
  /** Overrides the Lite module specifier, for hosts that own their own copy. */
  liteModule?: string;
}

export interface LiteSceneOptions {
  clearColor?: [number, number, number];
  /** Add a directional key and hemispheric fill. Off if you light it yourself. */
  defaultLights?: boolean;
}

export interface LiteCameraFraming {
  alpha?: number;
  beta?: number;
  /** Multiplier on the framed radius; 1 fits the bounds. */
  zoom?: number;
  target?: [number, number, number];
}

export interface LitePreviewSession {
  /** Declared for the capture pipeline: Lite's swapchain is bgra8, top-down. */
  readonly captureFormat: 'bgra8';
  readonly captureFlipped: false;
  /** Lite's `EngineContext`. */
  readonly engine: any;
  /** Lite's `SceneContext` from the most recent `newScene()`. */
  readonly scene: any;
  /** The Lite module, so a caller can reach the rest of its API. */
  readonly lite: any;
  newScene(options?: LiteSceneOptions): Promise<any>;
  /** Loads a GLB and adds it to the scene; returns Lite's `AssetContainer`. */
  loadGlb(glb: Uint8Array): Promise<any>;
  frameCamera(framing?: LiteCameraFraming): Promise<any>;
  /** Renders one frame and reads the surface back. */
  captureRaw(options?: { width?: number; height?: number }): Promise<RawFrame>;
  /** Pipelined capture, matching the core session's contract. */
  createCaptureStream(options?: { width?: number; height?: number; depth?: number; format?: CaptureFormat }): Promise<CaptureStream>;
  dispose(): Promise<void>;
}

const FRAME_MS = 1000 / 60;
/** WebGPU requires each copied row to start on a 256-byte boundary. */
const ROW_ALIGNMENT = 256;
/** Frames rendered before capture, to let asynchronous pipeline builds finish. */
const WARMUP_FRAMES = 12;
const BUFFER_USAGE_COPY_DST = 0x0008;
const BUFFER_USAGE_MAP_READ = 0x0001;
const MAP_MODE_READ = 0x1;

export async function createLitePreviewSession(
  options: LiteSessionOptions = {},
): Promise<LitePreviewSession> {
  const width = options.width ?? 1024;
  const height = options.height ?? 768;
  const gpu: HeadlessGpu = await installHeadlessWebGpu();
  if (options.decodeImage) installImageDecoder(options.decodeImage);

  const lite: any = await import(/* @vite-ignore */ options.liteModule ?? '@babylonjs/lite');
  const canvas = createHeadlessCanvas(width, height);
  const engine = await lite.createEngine(canvas as never, {});

  let scene: any = null;
  let camera: any = null;
  let readback: any = null;
  let pixels: Uint8Array | null = null;
  let registered = false;
  let warmed = false;

  /**
   * A scene registered while still empty never picks up meshes added later, so
   * this is deferred until the first frame is actually wanted.
   */
  const ensureRegistered = (): void => {
    if (registered || !scene) return;
    lite.registerScene(scene);
    registered = true;
  };

  /**
   * Lite builds pipelines asynchronously, so the first several frames draw the
   * clear colour and nothing else — measured at ~7 before the subject appeared.
   * Rendering with a yield between each lets those builds land. Without it,
   * every capture silently begins with a run of blank frames.
   */
  const warmUp = async (): Promise<void> => {
    if (warmed) return;
    for (let frame = 0; frame < WARMUP_FRAMES; frame++) {
      lite.renderFrame(engine, FRAME_MS);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    warmed = true;
  };

  const surfaceTexture = (): any => {
    const texture = (canvas as any)._context?.texture;
    if (!texture) throw new Error('The headless surface has no swapchain texture');
    return texture;
  };
  // The swapchain is configured bgra8unorm, and a screenshot of it comes back
  // in that order; the encoders take `bgra8` directly rather than paying for a
  // channel swap here.
  const format = 'bgra8' as const;

  const session: LitePreviewSession = {
    captureFormat: format,
    captureFlipped: false,
    get engine() { return engine; },
    get scene() { return scene; },
    get lite() { return lite; },

    async newScene(sceneOptions: LiteSceneOptions = {}) {
      scene = lite.createSceneContext(engine);
      camera = null;
      const clear = sceneOptions.clearColor ?? [0.02, 0.02, 0.03];
      // Lite's clearColor is a plain tuple rather than a Color4 instance.
      scene.clearColor = [clear[0], clear[1], clear[2], 1];
      if (sceneOptions.defaultLights !== false) {
        lite.addToScene(scene, lite.createDirectionalLight([-0.4, -1, -0.6], 1.1));
        lite.addToScene(scene, lite.createHemisphericLight([0.3, 1, 0.2], 0.35));
      }
      // Registration is deferred to the first capture, not done here: a scene
      // registered while still empty does not pick up meshes added afterwards.
      return scene;
    },

    async loadGlb(glb: Uint8Array) {
      if (!scene) await session.newScene();
      // Lite's loader takes an ArrayBuffer, and a Uint8Array view is usually a
      // window onto a larger buffer — hand over exactly this frame's bytes.
      const buffer = glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength) as ArrayBuffer;
      const container = await lite.loadGltf(engine, buffer);
      lite.addToScene(scene, container);
      return container;
    },

    async frameCamera(framing: LiteCameraFraming = {}) {
      const meshes: any[] = scene?.meshes ? [...scene.meshes] : [];
      let radius = 1;
      let centre: [number, number, number] = [0, 0, 0];
      const extents = meshes.length ? lite.computeMaxExtents(meshes) : [];
      if (extents.length) {
        let min = [Infinity, Infinity, Infinity];
        let max = [-Infinity, -Infinity, -Infinity];
        for (const extent of extents) {
          const lo = extent.min ?? extent.minimum ?? extent.aabb?.min;
          const hi = extent.max ?? extent.maximum ?? extent.aabb?.max;
          if (!lo || !hi) continue;
          for (let i = 0; i < 3; i++) {
            min[i] = Math.min(min[i]!, lo[i] ?? lo['xyz'[i] as 'x']);
            max[i] = Math.max(max[i]!, hi[i] ?? hi['xyz'[i] as 'x']);
          }
        }
        if (Number.isFinite(min[0])) {
          centre = [(min[0]! + max[0]!) / 2, (min[1]! + max[1]!) / 2, (min[2]! + max[2]!) / 2];
          const span = Math.hypot(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!);
          radius = Math.max(span / 2, 0.001);
        }
      }
      const target = framing.target ?? centre;
      camera = lite.createArcRotateCamera(
        framing.alpha ?? Math.PI / 4,
        framing.beta ?? Math.PI / 3,
        radius * (framing.zoom ?? 2.4),
        // A `Vec3` here is a plain `{ x, y, z }`, not a tuple — Lite's lights
        // take tuples, which makes the inconsistency easy to miss. Passing an
        // array leaves `target.x` undefined, the view matrix NaN, and the pass
        // clearing correctly while drawing nothing at all.
        { x: target[0], y: target[1], z: target[2] },
      );
      // Sized to the subject. The defaults are tuned for a scene a few units
      // across, so a model framed at radius 175 falls entirely beyond the far
      // plane and renders as nothing but the clear colour.
      camera.nearPlane = Math.max(0.01, radius / 1000);
      camera.farPlane = radius * 100;
      lite.addToScene(scene, camera);
      scene.camera = camera;
      return camera;
    },

    async captureRaw() {
      if (!scene) throw new Error('captureRaw before newScene');
      if (!camera) await session.frameCamera();
      ensureRegistered();
      await warmUp();
      // `renderFrame` records and submits one frame synchronously. Driving it
      // directly rather than running `startEngine`'s loop is what makes capture
      // deterministic: the frame is drawn when we say, not when a timer fires.
      lite.renderFrame(engine, FRAME_MS);

      // Read that texture directly rather than through Lite's
      // `captureScreenshot`, which never settles on this headless surface. This
      // is the same copy-and-map the core session's readback uses, and it keeps
      // the GPU texture reachable for the yuv converter.
      const texture = surfaceTexture();
      const device = engine._device;
      if (!device) throw new Error('Lite engine exposes no GPUDevice');

      const bytesPerRow = Math.ceil((width * 4) / ROW_ALIGNMENT) * ROW_ALIGNMENT;
      readback ??= device.createBuffer({
        size: bytesPerRow * height,
        usage: BUFFER_USAGE_COPY_DST | BUFFER_USAGE_MAP_READ,
      });
      const encoder = device.createCommandEncoder({ label: 'lite-capture' });
      encoder.copyTextureToBuffer(
        { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
        { buffer: readback, offset: 0, bytesPerRow, rowsPerImage: height },
        { width, height, depthOrArrayLayers: 1 },
      );
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(MAP_MODE_READ);
      const mapped = new Uint8Array(readback.getMappedRange());

      // Rows arrive padded to the alignment; hand back a tightly packed frame.
      pixels ??= new Uint8Array(width * height * 4);
      const stride = width * 4;
      for (let y = 0; y < height; y++) {
        pixels.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + stride), y * stride);
      }
      readback.unmap();

      return {
        data: pixels,
        width,
        height,
        // The surface is drawn top-down, unlike the core session's readback.
        flipped: false,
        format,
      } as RawFrame;
    },

    async createCaptureStream(streamOptions = {}) {
      if (!scene) throw new Error('createCaptureStream before newScene');
      // The surface is sized at engine creation, so a stream cannot ask for a
      // different size — say so rather than silently returning wrong pixels.
      const streamWidth = streamOptions.width ?? width;
      const streamHeight = streamOptions.height ?? height;
      if (streamWidth !== width || streamHeight !== height) {
        throw new Error(
          `A Lite capture stream must match the session surface (${width}x${height}); got ${streamWidth}x${streamHeight}`,
        );
      }
      const streamFormat: CaptureFormat = streamOptions.format === 'yuv420p' ? 'yuv420p' : 'rgba8';
      const useYuv = streamFormat === 'yuv420p';
      if (useYuv && !supportsGpuYuv(width, height)) {
        throw new Error(`yuv420p capture needs width % 8 == 0 and height % 2 == 0; got ${width}x${height}`);
      }
      const depth = Math.max(1, Math.floor(streamOptions.depth ?? (useYuv ? 8 : 4)));
      const device = engine._device;
      if (!device) throw new Error('Lite engine exposes no GPUDevice');

      // The surface carries TEXTURE_BINDING, so the converter samples the drawn
      // frame directly — no CPU readback in front of it.
      let converter: YuvConverter | null = useYuv
        ? createYuvConverter(device, { width, height, flip: false, depth })
        : null;

      const bytesPerRow = Math.ceil((width * 4) / ROW_ALIGNMENT) * ROW_ALIGNMENT;
      const slots = converter ? [] : Array.from({ length: depth }, () => ({
        buffer: device.createBuffer({
          size: bytesPerRow * height,
          usage: BUFFER_USAGE_COPY_DST | BUFFER_USAGE_MAP_READ,
        }),
        pixels: new Uint8Array(width * height * 4),
      }));
      let cursor = 0;
      let disposed = false;

      ensureRegistered();
      await warmUp();

      return {
        width,
        height,
        depth,
        // Without the converter this is the raw surface, which is bgra8.
        format: (useYuv ? 'yuv420p' : 'bgra8') as CaptureFormat,
        // A swapchain read is top-down either way.
        flipped: false,
        capture(): Promise<RawFrame> {
          if (disposed) throw new Error('This capture stream is disposed');
          // Synchronous through the draw and the copy submission, so several
          // of these may be outstanding without racing each other.
          lite.renderFrame(engine, FRAME_MS);
          const texture = surfaceTexture();

          if (converter) {
            return converter.convert(texture).then((data) => ({
              data, width, height, flipped: false, format: 'yuv420p' as const,
            }));
          }

          const slot = slots[cursor++ % depth]!;
          const encoder = device.createCommandEncoder({ label: 'lite-capture-stream' });
          encoder.copyTextureToBuffer(
            { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
            { buffer: slot.buffer, offset: 0, bytesPerRow, rowsPerImage: height },
            { width, height, depthOrArrayLayers: 1 },
          );
          device.queue.submit([encoder.finish()]);
          return slot.buffer.mapAsync(MAP_MODE_READ).then(() => {
            const mapped = new Uint8Array(slot.buffer.getMappedRange());
            const stride = width * 4;
            for (let y = 0; y < height; y++) {
              slot.pixels.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + stride), y * stride);
            }
            slot.buffer.unmap();
            return { data: slot.pixels, width, height, flipped: false, format: 'bgra8' as const };
          });
        },
        dispose(): void {
          if (disposed) return;
          disposed = true;
          converter?.dispose();
          converter = null;
          for (const slot of slots) slot.buffer.destroy?.();
        },
      } as CaptureStream;
    },

    async dispose() {
      try {
        readback?.destroy?.();
        if (scene) lite.disposeScene?.(scene);
        lite.disposeEngine?.(engine);
      } finally {
        gpu.dispose();
      }
    },
  };
  return session;
}
