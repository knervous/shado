/**
 * Drop-in replacement for `@eltania/glb-render`.
 *
 * That renderer drives Babylon inside Playwright and encodes with sharp; this
 * runs the same Babylon engine headless on Dawn, with no browser process. The
 * surface, the file naming (`<view>.png`), the default white background, the
 * named view angles and the reported `coverage` are all preserved, because
 * Libra's review renders are read as evidence and compared against images
 * already on file.
 *
 * Images are decoded with sharp, loaded lazily so nothing pays for it until a
 * textured render is actually requested.
 */

import { babylonImport } from './babylon';
import { join } from 'node:path';
import { mkdir, readFile, writeFile } from 'node:fs/promises';

import { encodePng } from './png';
import { createPreviewSession, type PreviewSession } from './session';
import { MULTIVIEWS, coverage, viewAngles } from './views';

export { MULTIVIEWS };

export interface GlbRenderOptions {
  width?: number;
  height?: number;
  /** Clear colour as linear RGB in 0..1. Defaults to white, as before. */
  background?: [number, number, number];
}

export interface GlbRenderRequest extends GlbRenderOptions {
  views?: string[];
  /** Tilt the camera down slightly, matching the old `${view}Top` modes. */
  raised?: boolean;
}

export interface GlbRenderedView {
  view: string;
  path: string;
  /** Fraction of non-background pixels — a framing sanity check. */
  coverage: number;
}

export interface GlbRenderResult {
  meshes: number;
  radius: number;
  engine: string;
  views: GlbRenderedView[];
}

export interface GlbAssemblyEntry {
  id: string;
  glb: string | Uint8Array;
  position?: [number, number, number];
  rotationDegrees?: [number, number, number];
  scale?: [number, number, number];
}

export interface GlbAssemblyRenderResult extends GlbRenderResult {
  instances: number;
}

async function decodeImage(bytes: Uint8Array): Promise<{ width: number; height: number; data: Uint8Array }> {
  const sharp = (await import('sharp')).default;
  const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data) };
}

async function toBytes(glb: string | Uint8Array): Promise<Uint8Array> {
  if (typeof glb !== 'string') return glb;
  const raw = await readFile(glb);
  if (!glb.endsWith('.gz')) return new Uint8Array(raw);
  const { gunzipSync } = await import('node:zlib');
  return new Uint8Array(gunzipSync(raw));
}

/** Offscreen GLB renderer: Babylon on Dawn, no browser and no dev server. */
export class GlbRenderer {
  private session: PreviewSession | null = null;
  private readonly width: number;
  private readonly height: number;
  private readonly background: [number, number, number];

  constructor(options: GlbRenderOptions = {}) {
    this.width = options.width ?? 1024;
    this.height = options.height ?? 1024;
    this.background = options.background ?? [1, 1, 1];
  }

  async start(): Promise<void> {
    this.session ??= await createPreviewSession({ width: this.width, height: this.height, decodeImage });
  }

  private async shoot(
    entries: readonly GlbAssemblyEntry[],
    outputDir: string,
    options: GlbRenderRequest,
  ): Promise<GlbRenderResult & { instances: number }> {
    await this.start();
    const session = this.session!;
    const background = options.background ?? this.background;
    const width = options.width ?? this.width;
    const height = options.height ?? this.height;

    await session.newScene({ clearColor: background, materials: true });
    const { Vector3 } = await babylonImport('@babylonjs/core/Maths/math.js');
    let instances = 0;
    for (const entry of entries) {
      const container = await session.loadGlb(await toBytes(entry.glb), { id: entry.id });
      for (const root of container.rootNodes as any[]) {
        if (entry.position) root.position = new Vector3(...entry.position);
        if (entry.scale) root.scaling = new Vector3(...entry.scale);
        if (entry.rotationDegrees) {
          const [pitch, yaw, roll] = entry.rotationDegrees;
          root.rotation = new Vector3((pitch * Math.PI) / 180, (yaw * Math.PI) / 180, (roll * Math.PI) / 180);
        }
        root.computeWorldMatrix?.(true);
      }
      instances++;
    }
    session.scene.meshes.forEach((mesh: any) => mesh.computeWorldMatrix(true));

    await mkdir(outputDir, { recursive: true });
    const views: GlbRenderedView[] = [];
    let radius = 0;
    for (const view of options.views ?? MULTIVIEWS) {
      const [alpha, beta] = viewAngles(view, options.raised === true);
      const camera = await session.frameCamera({ alpha, beta });
      radius = camera.radius / 2.4;
      const image = await session.capture({ width, height });
      const path = join(outputDir, `${view}.png`);
      await writeFile(path, encodePng(image.pixels, image.width, image.height));
      views.push({ view, path, coverage: coverage(image.pixels, background) });
    }
    return {
      meshes: session.scene.meshes.length,
      radius,
      engine: 'WebGPU (Dawn, headless)',
      views,
      instances,
    };
  }

  async render(glb: string | Uint8Array, outputDir: string, options: GlbRenderRequest = {}): Promise<GlbRenderResult> {
    const { instances: _instances, ...result } = await this.shoot([{ id: 'model', glb }], outputDir, options);
    return result;
  }

  async renderAssembly(
    models: GlbAssemblyEntry[],
    outputDir: string,
    options: GlbRenderRequest = {},
  ): Promise<GlbAssemblyRenderResult> {
    return this.shoot(models, outputDir, options);
  }

  async close(): Promise<void> {
    await this.session?.dispose();
    this.session = null;
  }
}

/** Convenience wrapper that starts and closes a renderer around one object. */
export async function renderGlb(
  glb: string | Uint8Array,
  outputDir: string,
  options: GlbRenderRequest = {},
): Promise<GlbRenderResult> {
  const renderer = new GlbRenderer(options);
  try {
    return await renderer.render(glb, outputDir, options);
  } finally {
    await renderer.close();
  }
}

/** Convenience wrapper that renders a transformed object/prefab composition. */
export async function renderGlbAssembly(
  models: GlbAssemblyEntry[],
  outputDir: string,
  options: GlbRenderRequest = {},
): Promise<GlbAssemblyRenderResult> {
  const renderer = new GlbRenderer(options);
  try {
    return await renderer.renderAssembly(models, outputDir, options);
  } finally {
    await renderer.close();
  }
}
