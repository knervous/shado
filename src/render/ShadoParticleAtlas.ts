import { BABYLON, type RawTexture2DArray, type Scene } from '../babylon';
import type { ShadoAtlasEntry, ShadoTextureAtlas } from './ShadoTextureAtlas';

/** RGBA8 pixels, rows top to bottom. */
export interface ShadoParticleImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array | Uint8ClampedArray;
}

/**
 * Bilinear resample of `image` into a `size` x `size` RGBA8 cell.
 *
 * Particle sprites are small and soft, so a fixed cell per image (rather than packing
 * rectangles) costs little and keeps every UV rectangle the whole layer, which removes
 * mip bleed between neighbours entirely.
 */
export function resampleShadoParticleImage(image: ShadoParticleImage, size: number): Uint8Array {
  const out = new Uint8Array(size * size * 4);
  const { width, height, data } = image;
  if (width <= 0 || height <= 0) return out;
  for (let y = 0; y < size; y++) {
    const sy = Math.min(height - 1, Math.max(0, ((y + 0.5) * height) / size - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < size; x++) {
      const sx = Math.min(width - 1, Math.max(0, ((x + 0.5) * width) / size - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = sx - x0;
      const o = (y * size + x) * 4;
      for (let c = 0; c < 4; c++) {
        const a = data[(y0 * width + x0) * 4 + c] * (1 - fx) + data[(y0 * width + x1) * 4 + c] * fx;
        const b = data[(y1 * width + x0) * 4 + c] * (1 - fx) + data[(y1 * width + x1) * 4 + c] * fx;
        out[o + c] = Math.round(a * (1 - fy) + b * fy);
      }
    }
  }
  return out;
}

/**
 * Every particle texture in one 2D array texture, one layer each, added on demand.
 *
 * One texture for every effect is what lets all particles share one draw. The array
 * doubles its layer count when full, which re-uploads everything once; with layers added
 * only as effects are first played that happens a handful of times per session.
 */
export class ShadoParticleAtlas implements ShadoTextureAtlas {
  public texture: RawTexture2DArray;
  public readonly entries: Record<string, ShadoAtlasEntry> = {};
  private layers: number;
  private used = 0;
  private data: Uint8Array;
  private dirty = false;

  public constructor(
    private readonly scene: Scene,
    public readonly cellSize = 128,
    initialLayers = 16
  ) {
    if (cellSize < 1 || (cellSize & (cellSize - 1)) !== 0) {
      throw new Error(`ShadoParticleAtlas: cellSize must be a power of two, got ${cellSize}`);
    }
    this.layers = Math.max(2, initialLayers);
    this.data = new Uint8Array(this.cellSize * this.cellSize * 4 * this.layers);
    // Layer 0 is a soft white disc, for particles with no texture of their own.
    this.writeLayer(0, softDisc(this.cellSize));
    this.entries.default = { layer: 0, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } };
    // Layer 1 is a soft ring, for flat shockwaves.
    this.writeLayer(1, softRing(this.cellSize));
    this.entries.ring = { layer: 1, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } };
    this.used = 2;
    this.texture = this.createTexture();
    this.dirty = true;
  }

  public get(key: string): ShadoAtlasEntry {
    return this.entries[key] ?? this.entries.default;
  }

  public has(key: string): boolean {
    return key in this.entries;
  }

  public get layerCount(): number {
    return this.used;
  }

  /** Adds `image` under `key` (no-op if present) and returns its entry. */
  public add(key: string, image: ShadoParticleImage): ShadoAtlasEntry {
    const existing = this.entries[key];
    if (existing) return existing;
    if (this.used >= this.layers) this.grow();
    const layer = this.used++;
    this.writeLayer(layer, resampleShadoParticleImage(image, this.cellSize));
    const entry: ShadoAtlasEntry = { layer, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } };
    this.entries[key] = entry;
    this.dirty = true;
    return entry;
  }

  /** Uploads pending layers. Called by the renderer before drawing. */
  public flush(): void {
    if (!this.dirty) return;
    this.dirty = false;
    // `update` writes only mip 0; small particles sample lower levels, so the whole chain
    // is rebuilt here (cells are square powers of two, so a 2x2 box filter per level).
    let level = this.data;
    let size = this.cellSize;
    let mip = 0;
    for (;;) {
      (this.texture as unknown as { updateMipLevel(data: ArrayBufferView, level: number): void }).updateMipLevel(level, mip);
      if (size <= 1) break;
      level = downsampleLayers(level, size, this.layers);
      size >>= 1;
      mip++;
    }
  }

  public dispose(): void {
    this.texture.dispose();
  }

  private writeLayer(layer: number, pixels: Uint8Array): void {
    this.data.set(pixels, layer * this.cellSize * this.cellSize * 4);
  }

  private grow(): void {
    const next = new Uint8Array(this.data.length * 2);
    next.set(this.data);
    this.data = next;
    this.layers *= 2;
    this.texture.dispose();
    this.texture = this.createTexture();
  }

  private createTexture(): RawTexture2DArray {
    const texture = new BABYLON.RawTexture2DArray(
      this.data,
      this.cellSize,
      this.cellSize,
      this.layers,
      BABYLON.Constants.TEXTUREFORMAT_RGBA,
      this.scene,
      true,
      false,
      BABYLON.Texture.TRILINEAR_SAMPLINGMODE
    );
    texture.wrapU = BABYLON.Texture.CLAMP_ADDRESSMODE;
    texture.wrapV = BABYLON.Texture.CLAMP_ADDRESSMODE;
    return texture;
  }
}

/** Halves every layer of a stacked RGBA8 array with a 2x2 box filter. */
function downsampleLayers(source: Uint8Array, size: number, layers: number): Uint8Array {
  const half = size >> 1;
  const out = new Uint8Array(half * half * layers * 4);
  for (let layer = 0; layer < layers; layer++) {
    const from = layer * size * size * 4;
    const to = layer * half * half * 4;
    for (let y = 0; y < half; y++) {
      for (let x = 0; x < half; x++) {
        const a = from + (y * 2 * size + x * 2) * 4;
        const b = a + 4;
        const c = a + size * 4;
        const d = c + 4;
        const o = to + (y * half + x) * 4;
        for (let ch = 0; ch < 4; ch++) {
          out[o + ch] = (source[a + ch]! + source[b + ch]! + source[c + ch]! + source[d + ch]! + 2) >> 2;
        }
      }
    }
  }
  return out;
}

/** Annulus peaking at 80% of the radius, fading to both edges (premultiplied, like the disc). */
function softRing(size: number): Uint8Array {
  const out = new Uint8Array(size * size * 4);
  const half = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r = Math.hypot(x + 0.5 - half, y + 0.5 - half) / half;
      const a = Math.max(0, 1 - Math.abs(r - 0.8) / 0.2);
      const o = (y * size + x) * 4;
      out[o] = out[o + 1] = out[o + 2] = out[o + 3] = Math.round(a * a * 255);
    }
  }
  return out;
}

/**
 * The built-in sprites carry their shape in colour as well as alpha (premultiplied): an
 * additive particle ignores alpha, as a ONE/ONE blend does, so a white square with a round
 * alpha would draw as a square.
 */
function softDisc(size: number): Uint8Array {
  const out = new Uint8Array(size * size * 4);
  const r = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x + 0.5 - r, y + 0.5 - r) / r;
      const a = Math.max(0, 1 - d);
      const o = (y * size + x) * 4;
      const v = Math.round(a * a * 255);
      out[o] = out[o + 1] = out[o + 2] = out[o + 3] = v;
    }
  }
  return out;
}
