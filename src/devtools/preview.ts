/**
 * Headless preview rendering for shado development.
 *
 * Drives Babylon's real WebGPU engine on Dawn (see headless-gpu.ts), so what
 * you see is the engine the game uses rather than an approximation. Built for
 * quick iteration on models and scenes, and for proving a pipeline end to end:
 * the same camera renders the raw GLB, the bake's output, and the finalized
 * runtime artifact, so the three are directly comparable.
 */

import { babylonImport } from './babylon';
import { createHeadlessCanvas, installHeadlessWebGpu, type HeadlessGpu } from './headless-gpu';
import { MULTIVIEWS, coverage, viewAngles } from './views';
import { createPreviewSession, type PreviewImage } from './session';
import { type ImageDecoder } from './headless-gpu';

export type { PreviewImage };

export interface CameraFraming {
  /** Orbit angle around the vertical axis, radians. */
  alpha?: number;
  /** Elevation, radians. */
  beta?: number;
  /** Multiplier on the framed radius; 1 fits the bounds. */
  zoom?: number;
  target?: [number, number, number];
}

export interface PreviewOptions {
  width?: number;
  height?: number;
  camera?: CameraFraming;
  clearColor?: [number, number, number];
  /**
   * Per-mesh RGBA vertex colours keyed by mesh name, as a vertex-lighting field
   * stores them. Applied over whatever the GLB carries, which is how a bake's
   * output is previewed against the raw asset it came from.
   */
  vertexColors?: Record<string, ArrayLike<number>>;
  /**
   * Whether to load glTF materials and their textures through Babylon's own
   * pipeline. Requires `decodeImage` on the renderer so images can be decoded;
   * with it off, geometry, shading and vertex colours still render, which is
   * what proving a bake needs.
   */
  materials?: boolean;
}

/** One transformed model in a composition — a prefab placement, typically. */
export interface GlbPlacement {
  id?: string;
  glb: Uint8Array;
  position?: [number, number, number];
  rotationDegrees?: [number, number, number];
  scale?: [number, number, number];
}

export interface RenderedView {
  view: string;
  image: PreviewImage;
  /** Fraction of non-background pixels; a framing sanity check. */
  coverage: number;
}

export interface ViewRenderOptions extends PreviewOptions {
  /** Defaults to left/front/back/right. */
  views?: readonly string[];
  /** Tilt every view except `top` down to the raised elevation. */
  raised?: boolean;
}

export interface HeadlessPreview {
  /** Renders a single GLB and returns raw RGBA pixels. */
  renderGlb(glb: Uint8Array, options?: PreviewOptions): Promise<PreviewImage>;
  /** Renders several transformed models composed into one scene. */
  renderComposition(placements: readonly GlbPlacement[], options?: PreviewOptions): Promise<PreviewImage>;
  /** Renders the named review views, reporting coverage for each. */
  renderViews(subject: Uint8Array | readonly GlbPlacement[], options?: ViewRenderOptions): Promise<RenderedView[]>;
  dispose(): Promise<void>;
}

const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 768;

/**
 * Convenience layer over {@link createPreviewSession}.
 *
 * This used to carry its own engine setup, lighting and capture code, which
 * silently drifted from the session's — a lighting fix applied here had no
 * effect on callers using the session, and vice versa. There is now exactly one
 * implementation; this only arranges scenes and cameras on top of it.
 */
export async function createHeadlessPreview(
  options: { width?: number; height?: number; decodeImage?: ImageDecoder; phaseTimeoutMs?: number } = {},
): Promise<HeadlessPreview> {
  const width = options.width ?? DEFAULT_WIDTH;
  const height = options.height ?? DEFAULT_HEIGHT;
  const session = await createPreviewSession({ ...options, width, height });

  const renderComposition = async (
    placements: readonly GlbPlacement[],
    previewOptions: PreviewOptions = {},
  ): Promise<PreviewImage> => {
    const { Vector3 } = await babylonImport('@babylonjs/core/Maths/math.js');
    const { VertexBuffer } = await babylonImport('@babylonjs/core/Buffers/buffer.js');
    await session.newScene({
      ...(previewOptions.clearColor ? { clearColor: previewOptions.clearColor } : {}),
      materials: previewOptions.materials === true,
    });

    for (const placement of placements) {
      const container = await session.loadGlb(placement.glb, ...(placement.id ? [{ id: placement.id }] : []));
      for (const root of container.rootNodes as any[]) {
        if (placement.position) root.position = new Vector3(...placement.position);
        if (placement.scale) root.scaling = new Vector3(...placement.scale);
        if (placement.rotationDegrees) {
          const [pitch, yaw, roll] = placement.rotationDegrees;
          root.rotation = new Vector3((pitch * Math.PI) / 180, (yaw * Math.PI) / 180, (roll * Math.PI) / 180);
        }
        root.computeWorldMatrix?.(true);
      }
    }
    session.scene.meshes.forEach((mesh: any) => mesh.computeWorldMatrix(true));

    if (previewOptions.vertexColors) {
      for (const mesh of session.scene.meshes as any[]) {
        const colors = previewOptions.vertexColors[mesh.name];
        if (!colors) continue;
        mesh.setVerticesData(VertexBuffer.ColorKind, Array.from(colors), false, 4);
        mesh.useVertexColors = true;
      }
    }

    await session.frameCamera(previewOptions.camera ?? {});
    return session.capture({ width: previewOptions.width ?? width, height: previewOptions.height ?? height });
  };

  return {
    renderComposition,
    async renderGlb(glb, previewOptions = {}) {
      return renderComposition([{ glb }], previewOptions);
    },
    async renderViews(subject, viewOptions = {}) {
      const placements = subject instanceof Uint8Array ? [{ glb: subject }] : subject;
      const background = viewOptions.clearColor ?? [0.02, 0.02, 0.03];
      const wanted = viewOptions.views ?? MULTIVIEWS;
      const results: RenderedView[] = [];
      for (const view of wanted) {
        const [alpha, beta] = viewAngles(view, viewOptions.raised === true);
        const image = await renderComposition(placements, {
          ...viewOptions,
          clearColor: background,
          camera: { ...(viewOptions.camera ?? {}), alpha, beta },
        });
        results.push({ view, image, coverage: coverage(image.pixels, background) });
      }
      return results;
    },
    async dispose() {
      await session.dispose();
    },
  };
}

/** Absolute per-pixel difference, amplified so small deltas are visible. */
export function differenceImage(a: PreviewImage, b: PreviewImage, amplify = 8): PreviewImage {
  if (a.pixels.length !== b.pixels.length) throw new Error('Preview images differ in size');
  const pixels = new Uint8Array(a.pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    const delta = Math.max(
      Math.abs(a.pixels[i]! - b.pixels[i]!),
      Math.abs(a.pixels[i + 1]! - b.pixels[i + 1]!),
      Math.abs(a.pixels[i + 2]! - b.pixels[i + 2]!),
    );
    const value = Math.min(255, delta * amplify);
    pixels[i] = value; pixels[i + 1] = value; pixels[i + 2] = value; pixels[i + 3] = 255;
  }
  return { pixels, width: a.width, height: a.height };
}
