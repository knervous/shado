/**
 * The scripting surface for headless shado previews.
 *
 * `createHeadlessPreview` covers "render this GLB from these angles". Anything
 * beyond that — stepping an animation to a keyframe, posing a rig, driving the
 * shado runtime, compositing a scene by hand — wants the engine itself rather
 * than another CLI flag. This hands the caller the real `WebGPUEngine` and
 * `Scene` and keeps only the awkward parts: device setup, the browser shims,
 * offscreen capture, and a stall watchdog.
 *
 * Everything Babylon exposes is fair game: `container.animationGroups`,
 * `scene.beginAnimation`, custom materials, post-processes, the lot.
 */

import { babylonImport } from './babylon';
import { createHeadlessCanvas, installHeadlessWebGpu, installImageDecoder, type HeadlessGpu, type ImageDecoder } from './headless-gpu';
import { installEnvironmentBrdf } from './brdf';
import { encodePng } from './png';
import { viewAngles } from './views';
import { installHeadlessKtx2Transcoder } from './ktx2';
import { resolveSharedTextureUri } from './shared-textures';
import { createYuvConverter, supportsGpuYuv, type GpuDeviceLike, type YuvConverter } from './yuv';

export interface PreviewImage {
  pixels: Uint8Array;
  width: number;
  height: number;
}

/**
 * A frame exactly as the GPU handed it back: RGBA8, rows bottom-up, alpha as
 * rendered. {@link PreviewImage} is this after a full-resolution CPU pass that
 * flips the rows and forces alpha opaque.
 *
 * That pass is free for one still and is the whole frame budget for video, so
 * the video pipeline takes the raw bytes and lets its encoder flip them — which
 * ffmpeg does as part of a scale it was already doing.
 */
export interface RawFrame {
  data: Uint8Array;
  width: number;
  height: number;
  /** Bottom-up rows. False for GPU-converted yuv420p, which flips as it converts. */
  flipped: boolean;
  /** Defaults to rgba8 when absent. */
  format?: 'rgba8' | 'yuv420p';
}

/**
 * Rewrites a URI an asset references, before the loader fetches it.
 *
 * This is the seam for every "the bytes are not where the glTF says they are"
 * problem: an external texture store, a `.gltf` with sibling `.bin` and images,
 * an asset served out of a zip or a database. Node's fetch cannot open `file:`
 * URLs, so the usual answer is to read the bytes and return a `data:` URL.
 *
 * Return the input unchanged to pass a URI on to the next resolver.
 */
export type UriResolver = (url: string) => string | Promise<string>;

export interface LoadOptions {
  id?: string;
  /**
   * Which Babylon loader to use. Defaults to `.glb`; `.gltf` works too, given
   * a `rootUrl` or a `resolveUri` that can reach the sibling files.
   */
  pluginExtension?: string;
  /** Base URL for formats with external references. */
  rootUrl?: string;
  /**
   * Runs ahead of the session's resolver and of the built-in shared-store one.
   * Resolvers are composed rather than replaced, so adding one never costs you
   * the others.
   */
  resolveUri?: UriResolver;
  /**
   * Merged over the loader defaults — Draco and meshopt settings, extension
   * toggles, `compileMaterials`, anything the glTF loader accepts.
   * `gltf.preprocessUrlAsync` is composed with the resolver chain rather than
   * overwriting it.
   */
  pluginOptions?: Record<string, any>;
}

export interface SessionOptions {
  width?: number;
  height?: number;
  /** Enables Babylon's real texture pipeline. `sharp` satisfies this. */
  decodeImage?: ImageDecoder;
  /** A healthy render is ~1s; anything far past this is a stall, not slow work. */
  phaseTimeoutMs?: number;
  /** Dawn binding to load; see {@link DEFAULT_DAWN_MODULE}. */
  dawnModule?: string;
  /** Applied to every load in this session. Per-load resolvers run first. */
  resolveUri?: UriResolver;
  /** Loader options for every load in this session; per-load ones merge over these. */
  pluginOptions?: Record<string, any>;
}

export interface SceneOptions {
  clearColor?: [number, number, number];
  /** Load glTF materials and textures. Requires `decodeImage` on the session. */
  materials?: boolean;
  /** Add a directional key and hemispheric fill. Off if you light it yourself. */
  defaultLights?: boolean;
  /**
   * Dither before the 8-bit quantization, to break up banding in smooth
   * gradients — skies, fog, soft-lit curves.
   *
   * It works: on a smooth-shaded sphere, a 96x96 patch of the falloff goes from
   * 216 value changes along its rows to 3030, which is banding being replaced
   * with noise.
   *
   * **But do not turn it on for video.** Measured through h264 at crf 20 those
   * 3030 changes come back down to 322 — the encoder removes exactly the
   * low-amplitude high-frequency signal dither adds — and the dithered result
   * ends up with *fewer* distinct levels than the undithered one (9 vs 12) for
   * slightly more bitrate. It is worth having for PNG stills and for
   * near-lossless encodes, and is a small loss for anything else.
   *
   * Off by default regardless, because Libra's review stills are compared
   * against images already on file and this changes every pixel slightly.
   */
  dithering?: boolean;
}

export interface CaptureOptions {
  width?: number;
  height?: number;
  camera?: unknown;
}

/**
 * Pixel layout a capture stream produces.
 *
 * `yuv420p` converts on the GPU and reads back 1.5 bytes per pixel instead of
 * 4, already flipped — which removes ~71% of what an ffmpeg encode was
 * spending its time on. `rgba8` is the portable fallback.
 */
export type CaptureFormat = 'rgba8' | 'yuv420p';

export interface CaptureStreamOptions extends CaptureOptions {
  /**
   * Defaults to `rgba8`. `yuv420p` needs width % 8 == 0 and height % 2 == 0;
   * ask for it at another size and it throws rather than silently falling
   * back, because the difference shows up as a speed regression nobody
   * attributes to the capture.
   */
  format?: CaptureFormat;
  /**
   * How many GPU readbacks may be in flight at once.
   *
   * On `@kmamal/gpu` this is the whole performance story: that binding
   * dispatches map callbacks on a shared ~100ms tick, so one readback and
   * sixteen cost the same wall clock — depth 1 is 100ms/frame, depth 8 is
   * 16.7ms, depth 16 is 8.3ms. It helps on `webgpu` too, where readback is
   * genuinely fast (~7ms) but still worth overlapping.
   *
   * Costs `depth * width * height * 4` bytes of resident buffers, which is why
   * the default is derived from a memory budget rather than fixed.
   */
  depth?: number;
}

/**
 * A render target and a ring of readback buffers, set up once and driven frame
 * after frame.
 *
 * The reason this exists rather than a loop over {@link PreviewSession.captureRaw}
 * is ordering. Pipelining requires issuing frame N+1's render before frame N's
 * readback resolves, which is only safe if applying camera state, rendering and
 * submitting the readback happen with no `await` between them. Everything that
 * needs awaiting is hoisted into construction so that `capture()` can be
 * synchronous up to the point the GPU work is queued.
 */
export interface CaptureStream {
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly format: CaptureFormat;
  /**
   * Renders one frame and returns its readback.
   *
   * Call it again before the previous promise settles — that is the point. Do
   * not exceed `depth` outstanding calls: the buffer ring wraps and a frame
   * still in flight would be overwritten.
   */
  capture(): Promise<RawFrame>;
  dispose(): void;
}

export interface PreviewSession {
  /** The real Babylon WebGPU engine, running on Dawn. */
  readonly engine: any;
  /** The scene created by the most recent `newScene()`. */
  readonly scene: any;
  newScene(options?: SceneOptions): Promise<any>;
  /**
   * Loads a model into the current scene and returns Babylon's AssetContainer,
   * so `animationGroups`, `skeletons` and `meshes` are all reachable.
   *
   * Bytes work for binary formats. Text formats — `.gltf` above all — must be
   * given a string instead: Babylon's JSON plugin refuses an ArrayBufferView
   * outright ("plugins that don't support binary loading"). A `data:` URL of
   * the JSON is the portable way to pass one, since Node's fetch cannot open
   * `file:`.
   */
  loadGlb(source: Uint8Array | string, options?: LoadOptions): Promise<any>;
  /** ArcRotate camera framed on the current scene; accepts a named review view. */
  frameCamera(framing?: { view?: string; raised?: boolean; alpha?: number; beta?: number; zoom?: number; target?: [number, number, number] }): Promise<any>;
  /** Renders the scene as it stands right now. */
  capture(options?: CaptureOptions): Promise<PreviewImage>;
  /**
   * Renders one frame and returns the GPU's bytes untouched.
   *
   * Reuses one render target and one readback buffer across calls, and skips
   * the shader warm-up after the first frame of a scene, so a capture loop
   * costs one render rather than two renders, two sleeps and two allocations.
   *
   * The returned `data` IS that shared buffer. It is valid until the next
   * `captureRaw`, which is all a streaming encoder needs; anything that keeps a
   * frame must copy it.
   */
  captureRaw(options?: CaptureOptions): Promise<RawFrame>;
  /** Sets up a pipelined capture loop. See {@link CaptureStream}. */
  createCaptureStream(options?: CaptureStreamOptions): Promise<CaptureStream>;
  /** Convenience: capture and write a PNG. */
  captureToFile(path: string, options?: CaptureOptions): Promise<PreviewImage>;
  dispose(): Promise<void>;
}

const DEFAULT_PHASE_TIMEOUT_MS = 45_000;

/**
 * Resident bytes the readback ring may occupy. Depth is derived from this
 * rather than fixed, because the useful depth depends entirely on frame size:
 * 16 is comfortable at 960x540 and 530MB at 4K.
 */
const CAPTURE_RING_BUDGET_BYTES = 64 * 1024 * 1024;
const MIN_CAPTURE_DEPTH = 2;
/**
 * Measured ceiling. Depth 8 is 16.7ms/frame at 960x540 and depth 16 is
 * 8.3ms/frame; past that the readback stops being what limits throughput.
 */
const MAX_CAPTURE_DEPTH = 16;

/**
 * Deepest ring that fits the budget at this frame size.
 *
 * Format matters: a yuv420p frame is 1.5 bytes per pixel against RGBA's 4, so
 * the same memory buys a ring 2.7x deeper. Getting this wrong is expensive and
 * silent — sizing the YUV ring by RGBA's footprint left it at depth 4 where 16
 * fit, and cost 7.3ms/frame against 1.7ms at 1440p.
 */
function captureDepthFor(width: number, height: number, format: CaptureFormat = 'rgba8'): number {
  const bytesPerPixel = format === 'yuv420p' ? 1.5 : 4;
  const frameBytes = Math.max(1, width * height * bytesPerPixel);
  return Math.min(MAX_CAPTURE_DEPTH, Math.max(MIN_CAPTURE_DEPTH, Math.floor(CAPTURE_RING_BUDGET_BYTES / frameBytes)));
}

async function withDeadline<T>(work: Promise<T>, label: string, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work,
      new Promise<never>((_r, reject) => {
        timer = setTimeout(() => reject(new Error(`Preview stalled in '${label}' after ${(ms / 1000).toFixed(0)}s`)), ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export async function createPreviewSession(options: SessionOptions = {}): Promise<PreviewSession> {
  const width = options.width ?? 1024;
  const height = options.height ?? 768;
  const phaseTimeout = options.phaseTimeoutMs ?? DEFAULT_PHASE_TIMEOUT_MS;
  const gpu: HeadlessGpu = await installHeadlessWebGpu(options.dawnModule);
  if (options.decodeImage) installImageDecoder(options.decodeImage);

  const { WebGPUEngine } = await babylonImport('@babylonjs/core/Engines/webgpuEngine.js');
  await babylonImport('@babylonjs/loaders/glTF/index.js');
  const engine = await WebGPUEngine.CreateAsync(createHeadlessCanvas(width, height) as never, {
    antialias: true, stencil: false, adaptToDeviceRatio: false,
  });

  let scene: any = null;
  let camera: any = null;
  // One render target and one readback buffer, held for as long as the scene
  // and the requested size stay put. Rebuilding them per frame was invisible
  // for stills and dominated the cost of a capture loop.
  let target: any = null;
  let targetScene: any = null;
  let targetWidth = 0;
  let targetHeight = 0;
  let readback: Uint8Array | null = null;

  /**
   * Babylon compiles shaders asynchronously and the first frame renders
   * nothing, but that is true only until the scene's shaders exist. Paying it
   * per frame would cost a capture loop two renders and 32ms of sleep each.
   */
  const warmUp = async (): Promise<void> => {
    for (let frame = 0; frame < 2; frame++) {
      scene.render();
      await new Promise((resolve) => setTimeout(resolve, 16));
    }
    (scene as any).__captureWarm = true;
  };

  const releaseTarget = (): void => {
    // A disposed scene has already taken its render targets with it.
    if (target && targetScene === scene) target.dispose();
    target = null;
    targetScene = null;
    targetWidth = 0;
    targetHeight = 0;
    readback = null;
  };

  const session: PreviewSession = {
    get engine() { return engine; },
    get scene() { return scene; },

    async newScene(sceneOptions: SceneOptions = {}) {
      releaseTarget();
      scene?.dispose();
      camera = null;
      const { Scene } = await babylonImport('@babylonjs/core/scene.js');
      const { Color4 } = await babylonImport('@babylonjs/core/Maths/math.js');
      scene = new Scene(engine);
      const clear = sceneOptions.clearColor ?? [0.02, 0.02, 0.03];
      scene.clearColor = new Color4(clear[0], clear[1], clear[2], 1);
      // Must precede any PBR material — Babylon only loads its own browser-only
      // BRDF texture while this property is unset.
      if (sceneOptions.materials) await installEnvironmentBrdf(scene);
      (scene as any).__materials = sceneOptions.materials === true;
      if (sceneOptions.defaultLights !== false) {
        const { DirectionalLight } = await babylonImport('@babylonjs/core/Lights/directionalLight.js');
        const { HemisphericLight } = await babylonImport('@babylonjs/core/Lights/hemisphericLight.js');
        const { Vector3 } = await babylonImport('@babylonjs/core/Maths/math.js');
        // PBR is calibrated for physical light values; the intensities that
        // look right on StandardMaterial blow it out. 2.2/1.1 clipped 30% of
        // lit pixels to pure white and flattened every texture.
        const key = new DirectionalLight('key', new Vector3(-0.4, -1, -0.6), scene);
        key.intensity = 1.1;
        const fill = new HemisphericLight('fill', new Vector3(0.3, 1, 0.2), scene);
        fill.intensity = 0.35;
        // Roll highlights off instead of clipping them. Without tone mapping a
        // specular hit lands at pure white and takes the albedo with it.
        scene.imageProcessingConfiguration.toneMappingEnabled = true;
        scene.imageProcessingConfiguration.toneMappingType = 1; // ACES
        scene.imageProcessingConfiguration.exposure = 1.0;
      }
      if (sceneOptions.dithering) {
        // Applied in the shader before quantization, so unlike a CPU dither it
        // needs no HDR readback and costs nothing per frame. The default
        // intensity is one quantization step, which is the whole point.
        scene.imageProcessingConfiguration.ditheringEnabled = true;
      }
      return scene;
    },

    async loadGlb(source, loadOptions = {}) {
      if (!scene) await session.newScene();
      const extension = loadOptions.pluginExtension ?? '.glb';
      // Babylon's failure here is "plugins that don't support binary loading",
      // which does not tell you what to do about it.
      if (extension === '.gltf' && typeof source !== 'string') {
        throw new Error(
          "The .gltf plugin cannot take bytes; pass a string. A data: URL of the JSON works everywhere "
          + "(`data:model/gltf+json;base64,...`), and a resolveUri handles its .bin and image siblings.",
        );
      }
      const { SceneLoader, LoadAssetContainerAsync } = await babylonImport('@babylonjs/core/Loading/sceneLoader.js');
      if (!(scene as any).__materials) {
        // addOnce fires for one plugin activation, so re-arm per load or the
        // second model loads with materials and stalls on an undecodable image.
        SceneLoader.OnPluginActivatedObservable.addOnce((plugin: any) => {
          if (plugin.name === 'gltf') plugin.skipMaterials = true;
        });
      }
      // Promoted objects reference their textures out of a shared KTX2 store
      // rather than embedding them. Without the decoder AND a way to resolve
      // that reference, the load never settles and the preview stalls.
      if ((scene as any).__materials) await installHeadlessKtx2Transcoder();

      // Composed, not replaced: a caller adding a resolver for its own store
      // still gets the built-in one, and the loader still gets exactly one
      // `preprocessUrlAsync`. Most-specific first.
      const supplied = loadOptions.pluginOptions?.gltf?.preprocessUrlAsync as UriResolver | undefined;
      const chain: UriResolver[] = [
        loadOptions.resolveUri,
        options.resolveUri,
        supplied,
        resolveSharedTextureUri,
      ].filter(Boolean) as UriResolver[];
      const preprocessUrlAsync = async (url: string): Promise<string> => {
        for (const resolve of chain) {
          const next = await resolve(url);
          if (next !== url) return next;
        }
        return url;
      };

      const pluginOptions = {
        ...options.pluginOptions,
        ...loadOptions.pluginOptions,
        gltf: {
          ...(options.pluginOptions?.gltf as object | undefined),
          ...(loadOptions.pluginOptions?.gltf as object | undefined),
          preprocessUrlAsync,
        },
      };

      // Annotated because the loader now arrives through `babylonImport`, whose
      // return is deliberately untyped.
      const container: any = await withDeadline<any>(
        LoadAssetContainerAsync(source as never, scene, {
          pluginExtension: extension,
          ...(loadOptions.rootUrl ? { rootUrl: loadOptions.rootUrl } : {}),
          pluginOptions,
        } as never),
        `load ${loadOptions.id ?? 'model'}`,
        phaseTimeout,
      );
      container.addAllToScene();
      if (!(scene as any).__materials) {
        const { StandardMaterial } = await babylonImport('@babylonjs/core/Materials/standardMaterial.js');
        const { Color3 } = await babylonImport('@babylonjs/core/Maths/math.color.js');
        const neutral = new StandardMaterial('preview-neutral', scene);
        neutral.diffuseColor = new Color3(0.82, 0.82, 0.84);
        neutral.specularColor = new Color3(0.05, 0.05, 0.05);
        for (const mesh of scene.meshes) if (!mesh.material) mesh.material = neutral;
      }
      return container;
    },

    async frameCamera(framing = {}) {
      const { ArcRotateCamera } = await babylonImport('@babylonjs/core/Cameras/arcRotateCamera.js');
      const { Vector3 } = await babylonImport('@babylonjs/core/Maths/math.js');
      let min = new Vector3(Infinity, Infinity, Infinity);
      let max = new Vector3(-Infinity, -Infinity, -Infinity);
      for (const mesh of scene.meshes) {
        const info = mesh.getBoundingInfo?.();
        if (!info) continue;
        min = Vector3.Minimize(min, info.boundingBox.minimumWorld);
        max = Vector3.Maximize(max, info.boundingBox.maximumWorld);
      }
      if (!Number.isFinite(min.x)) { min = new Vector3(-1, -1, -1); max = new Vector3(1, 1, 1); }
      const centre = Vector3.Center(min, max);
      const radius = Math.max(max.subtract(min).length() / 2, 0.001);
      const [namedAlpha, namedBeta] = framing.view ? viewAngles(framing.view, framing.raised === true) : [Math.PI / 4, Math.PI / 3];
      camera = new ArcRotateCamera(
        'preview',
        framing.alpha ?? namedAlpha,
        framing.beta ?? namedBeta,
        radius * (framing.zoom ?? 2.4),
        framing.target ? new Vector3(...framing.target) : centre,
        scene,
      );
      camera.minZ = Math.max(0.01, radius / 1000);
      camera.maxZ = radius * 100;
      return camera;
    },

    async captureRaw(captureOptions = {}) {
      const { RenderTargetTexture } = await babylonImport('@babylonjs/core/Materials/Textures/renderTargetTexture.js');
      const captureWidth = captureOptions.width ?? width;
      const captureHeight = captureOptions.height ?? height;
      // A caller that brings its own camera gets it used as-is; only frame a
      // default one when nothing has been framed at all.
      const captureCamera = (captureOptions.camera as any) ?? camera ?? await session.frameCamera();
      await withDeadline(
        new Promise<void>((resolve) => scene.executeWhenReady(() => resolve())),
        'scene ready (a texture or shader never finished)',
        phaseTimeout,
      );
      if (!target || targetScene !== scene || targetWidth !== captureWidth || targetHeight !== captureHeight) {
        releaseTarget();
        target = new RenderTargetTexture('capture', { width: captureWidth, height: captureHeight }, scene, false);
        targetScene = scene;
        targetWidth = captureWidth;
        targetHeight = captureHeight;
        readback = new Uint8Array(captureWidth * captureHeight * 4);
        scene.customRenderTargets.push(target);
      }
      // Meshes can arrive between frames, so the list is refreshed every time;
      // the texture behind it is not.
      target.renderList = scene.meshes;
      target.activeCamera = captureCamera;
      target.clearColor = scene.clearColor;
      if (!(scene as any).__captureWarm) await warmUp();
      else scene.render();
      const data = await withDeadline(target.readPixels(0, 0, readback) as Promise<unknown>, 'readback', phaseTimeout);
      const raw = new Uint8Array((data as any)?.buffer ?? (data as never));
      return { data: raw, width: captureWidth, height: captureHeight, flipped: true };
    },

    async createCaptureStream(streamOptions = {}) {
      const { RenderTargetTexture } = await babylonImport('@babylonjs/core/Materials/Textures/renderTargetTexture.js');
      const streamWidth = streamOptions.width ?? width;
      const streamHeight = streamOptions.height ?? height;
      const requestedFormat: CaptureFormat = streamOptions.format ?? 'rgba8';
      const streamDepth = Math.max(
        1,
        Math.floor(streamOptions.depth ?? captureDepthFor(streamWidth, streamHeight, requestedFormat)),
      );
      const streamCamera = (streamOptions.camera as any) ?? camera ?? await session.frameCamera();
      const streamScene = scene;
      await withDeadline(
        new Promise<void>((resolve) => streamScene.executeWhenReady(() => resolve())),
        'scene ready (a texture or shader never finished)',
        phaseTimeout,
      );

      const streamTarget = new RenderTargetTexture(
        'capture-stream',
        { width: streamWidth, height: streamHeight },
        streamScene,
        false,
      );
      streamTarget.activeCamera = streamCamera;
      streamTarget.clearColor = streamScene.clearColor;
      streamScene.customRenderTargets.push(streamTarget);
      if (!(streamScene as any).__captureWarm) await warmUp();

      // One CPU buffer per slot. Babylon allocates its own GPU staging buffer
      // per readback, so this is the only piece that has to be per-frame.
      const streamFormat = requestedFormat;
      if (streamFormat === 'yuv420p' && !supportsGpuYuv(streamWidth, streamHeight)) {
        throw new Error(
          `yuv420p capture needs width % 8 == 0 and height % 2 == 0; got ${streamWidth}x${streamHeight}`,
        );
      }
      let converter: YuvConverter | null = null;
      if (streamFormat === 'yuv420p') {
        converter = createYuvConverter(engine._device as GpuDeviceLike, {
          width: streamWidth,
          height: streamHeight,
          flip: true,
          depth: streamDepth,
        });
      }

      const ring = converter
        ? []
        : Array.from({ length: streamDepth }, () => new Uint8Array(streamWidth * streamHeight * 4));
      let cursor = 0;
      let disposed = false;

      return {
        width: streamWidth,
        height: streamHeight,
        depth: streamDepth,
        format: streamFormat,
        capture(): Promise<RawFrame> {
          if (disposed) throw new Error('This capture stream is disposed');
          const slot = cursor++ % streamDepth;
          // Meshes can arrive between frames; the target behind them does not.
          streamTarget.renderList = streamScene.meshes;
          streamScene.render();

          if (converter) {
            // Babylon may not have submitted the render yet; the compute pass
            // reads the texture, so its commands must already be queued. This
            // is the same flush readPixels performs internally.
            engine.flushFramebuffer();
            const gpuTexture = streamTarget.getInternalTexture()?._hardwareTexture?.underlyingResource;
            if (!gpuTexture) throw new Error('The capture target has no underlying GPU texture');
            return withDeadline(converter.convert(gpuTexture), `yuv convert slot ${slot}`, phaseTimeout).then(
              (data) => ({
                data,
                width: streamWidth,
                height: streamHeight,
                // The shader already flipped it.
                flipped: false,
                format: 'yuv420p' as const,
              }),
            );
          }

          // readPixels flushes Babylon's pending commands before queueing the
          // texture copy, so the copy sees this frame and not the next one.
          const pending = streamTarget.readPixels(0, 0, ring[slot]!) as Promise<unknown>;
          return withDeadline(pending, `readback slot ${slot}`, phaseTimeout).then((data) => ({
            data: new Uint8Array((data as any)?.buffer ?? (data as never)),
            width: streamWidth,
            height: streamHeight,
            flipped: true,
            format: 'rgba8' as const,
          }));
        },
        dispose(): void {
          if (disposed) return;
          disposed = true;
          converter?.dispose();
          const index = streamScene.customRenderTargets.indexOf(streamTarget);
          if (index >= 0) streamScene.customRenderTargets.splice(index, 1);
          streamTarget.dispose();
        },
      };
    },

    async capture(captureOptions = {}) {
      const frame = await session.captureRaw(captureOptions);
      // readPixels hands back bottom-up rows (GL convention); PNG and every
      // consumer here expect top-down, so flip while forcing alpha opaque.
      const pixels = new Uint8Array(frame.data.length);
      const stride = frame.width * 4;
      for (let y = 0; y < frame.height; y++) {
        const from = (frame.height - 1 - y) * stride;
        for (let x = 0; x < stride; x += 4) {
          const to = y * stride + x;
          pixels[to] = frame.data[from + x]!;
          pixels[to + 1] = frame.data[from + x + 1]!;
          pixels[to + 2] = frame.data[from + x + 2]!;
          pixels[to + 3] = 255;
        }
      }
      return { pixels, width: frame.width, height: frame.height };
    },

    async captureToFile(path, captureOptions = {}) {
      const image = await session.capture(captureOptions);
      const { writeFile, mkdir } = await import('node:fs/promises');
      const { dirname } = await import('node:path');
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, encodePng(image.pixels, image.width, image.height));
      return image;
    },

    async dispose() {
      releaseTarget();
      scene?.dispose();
      engine.dispose();
      gpu.dispose();
    },
  };
  return session;
}
