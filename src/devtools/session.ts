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

export interface PreviewImage {
  pixels: Uint8Array;
  width: number;
  height: number;
}

export interface SessionOptions {
  width?: number;
  height?: number;
  /** Enables Babylon's real texture pipeline. `sharp` satisfies this. */
  decodeImage?: ImageDecoder;
  /** A healthy render is ~1s; anything far past this is a stall, not slow work. */
  phaseTimeoutMs?: number;
}

export interface SceneOptions {
  clearColor?: [number, number, number];
  /** Load glTF materials and textures. Requires `decodeImage` on the session. */
  materials?: boolean;
  /** Add a directional key and hemispheric fill. Off if you light it yourself. */
  defaultLights?: boolean;
}

export interface CaptureOptions {
  width?: number;
  height?: number;
  camera?: unknown;
}

export interface PreviewSession {
  /** The real Babylon WebGPU engine, running on Dawn. */
  readonly engine: any;
  /** The scene created by the most recent `newScene()`. */
  readonly scene: any;
  newScene(options?: SceneOptions): Promise<any>;
  /** Loads a GLB into the current scene and returns Babylon's AssetContainer. */
  loadGlb(glb: Uint8Array, options?: { id?: string }): Promise<any>;
  /** ArcRotate camera framed on the current scene; accepts a named review view. */
  frameCamera(framing?: { view?: string; raised?: boolean; alpha?: number; beta?: number; zoom?: number; target?: [number, number, number] }): Promise<any>;
  /** Renders the scene as it stands right now. */
  capture(options?: CaptureOptions): Promise<PreviewImage>;
  /** Convenience: capture and write a PNG. */
  captureToFile(path: string, options?: CaptureOptions): Promise<PreviewImage>;
  dispose(): Promise<void>;
}

const DEFAULT_PHASE_TIMEOUT_MS = 45_000;

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
  const gpu: HeadlessGpu = await installHeadlessWebGpu();
  if (options.decodeImage) installImageDecoder(options.decodeImage);

  const { WebGPUEngine } = await babylonImport('@babylonjs/core/Engines/webgpuEngine.js');
  await babylonImport('@babylonjs/loaders/glTF/index.js');
  const engine = await WebGPUEngine.CreateAsync(createHeadlessCanvas(width, height) as never, {
    antialias: true, stencil: false, adaptToDeviceRatio: false,
  });

  let scene: any = null;
  let camera: any = null;

  const session: PreviewSession = {
    get engine() { return engine; },
    get scene() { return scene; },

    async newScene(sceneOptions: SceneOptions = {}) {
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
      return scene;
    },

    async loadGlb(glb, loadOptions = {}) {
      if (!scene) await session.newScene();
      const { SceneLoader, LoadAssetContainerAsync } = await babylonImport('@babylonjs/core/Loading/sceneLoader.js');
      if (!(scene as any).__materials) {
        // addOnce fires for one plugin activation, so re-arm per load or the
        // second model loads with materials and stalls on an undecodable image.
        SceneLoader.OnPluginActivatedObservable.addOnce((plugin: any) => {
          if (plugin.name === 'gltf') plugin.skipMaterials = true;
        });
      }
      // Annotated because the loader now arrives through `babylonImport`, whose
      // return is deliberately untyped.
      const container: any = await withDeadline<any>(
        LoadAssetContainerAsync(glb as never, scene, { pluginExtension: '.glb' }),
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

    async capture(captureOptions = {}) {
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
      const target = new RenderTargetTexture('capture', { width: captureWidth, height: captureHeight }, scene, false);
      target.renderList = scene.meshes;
      target.activeCamera = captureCamera;
      target.clearColor = scene.clearColor;
      scene.customRenderTargets.push(target);
      try {
        // Babylon compiles shaders asynchronously; one frame renders nothing.
        for (let frame = 0; frame < 2; frame++) {
          scene.render();
          await new Promise((resolve) => setTimeout(resolve, 16));
        }
        const data = await withDeadline(target.readPixels() as Promise<unknown>, 'readback', phaseTimeout);
        const raw = new Uint8Array((data as any)?.buffer ?? (data as never));
        // readPixels hands back bottom-up rows (GL convention); PNG and every
        // consumer here expect top-down, so flip while forcing alpha opaque.
        const pixels = new Uint8Array(raw.length);
        const stride = captureWidth * 4;
        for (let y = 0; y < captureHeight; y++) {
          const from = (captureHeight - 1 - y) * stride;
          for (let x = 0; x < stride; x += 4) {
            const to = y * stride + x;
            pixels[to] = raw[from + x]!;
            pixels[to + 1] = raw[from + x + 1]!;
            pixels[to + 2] = raw[from + x + 2]!;
            pixels[to + 3] = 255;
          }
        }
        return { pixels, width: captureWidth, height: captureHeight };
      } finally {
        scene.customRenderTargets.pop();
        target.dispose();
      }
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
      scene?.dispose();
      engine.dispose();
      gpu.dispose();
    },
  };
  return session;
}
