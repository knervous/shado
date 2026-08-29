/**
 * The split-sum environment BRDF lookup, computed on the CPU.
 *
 * Babylon normally loads this as an RGBD-encoded PNG and expands it to float on
 * the GPU. That path needs a browser image upload — Dawn rejects synthetic
 * image sources, and the raw-upload fallback supplies 8-bit data to an
 * rgba16float texture, which fails as "Required size for texture data layout
 * (524288) exceeds the linear data size (262144)" and leaves PBR unlit.
 *
 * The LUT is a pure function of (NdotV, roughness), so computing it is both
 * exact and cheaper than arguing with the loader. Assigning it to
 * `scene.environmentBRDFTexture` before any PBR material exists also stops
 * Babylon loading its own copy — `loadBRDFTexture` is guarded on the property
 * already being set.
 */
// `babylonImport` was used below without ever being imported, so any scene
// created with `materials: true` threw `babylonImport is not defined` before a
// single textured pixel was drawn. Nothing caught it because every other
// devtools path renders untextured, where the BRDF LUT is never built.
import { babylonImport } from './babylon';

const SIZE = 128;
const SAMPLES = 128;

/** Van der Corput radical inverse — the Hammersley sequence's second term. */
function radicalInverse(bits: number): number {
  let value = bits;
  value = ((value << 16) | (value >>> 16)) >>> 0;
  value = (((value & 0x55555555) << 1) | ((value & 0xaaaaaaaa) >>> 1)) >>> 0;
  value = (((value & 0x33333333) << 2) | ((value & 0xcccccccc) >>> 2)) >>> 0;
  value = (((value & 0x0f0f0f0f) << 4) | ((value & 0xf0f0f0f0) >>> 4)) >>> 0;
  value = (((value & 0x00ff00ff) << 8) | ((value & 0xff00ff00) >>> 8)) >>> 0;
  return value * 2.3283064365386963e-10;
}

/** Smith geometry term, IBL parameterisation. */
function geometrySmith(nDotV: number, nDotL: number, roughness: number): number {
  const k = (roughness * roughness) / 2;
  const gv = nDotV / (nDotV * (1 - k) + k);
  const gl = nDotL / (nDotL * (1 - k) + k);
  return gv * gl;
}

/** Integrates the GGX split-sum terms for one (NdotV, roughness) pair. */
function integrate(nDotV: number, roughness: number): [number, number] {
  const v = [Math.sqrt(1 - nDotV * nDotV), 0, nDotV];
  let scale = 0;
  let bias = 0;
  const alpha = roughness * roughness;
  for (let i = 0; i < SAMPLES; i++) {
    const u1 = i / SAMPLES;
    const u2 = radicalInverse(i);
    const phi = 2 * Math.PI * u1;
    const cosTheta = Math.sqrt((1 - u2) / (1 + (alpha * alpha - 1) * u2));
    const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
    const h = [sinTheta * Math.cos(phi), sinTheta * Math.sin(phi), cosTheta];
    const vDotH = v[0]! * h[0]! + v[1]! * h[1]! + v[2]! * h[2]!;
    const l = [2 * vDotH * h[0]! - v[0]!, 2 * vDotH * h[1]! - v[1]!, 2 * vDotH * h[2]! - v[2]!];
    const nDotL = l[2]!;
    if (nDotL <= 0) continue;
    const nDotH = Math.max(h[2]!, 0);
    const clampedVdotH = Math.max(vDotH, 0);
    const g = geometrySmith(nDotV, nDotL, roughness);
    const gVis = (g * clampedVdotH) / (nDotH * nDotV || 1e-6);
    const fc = Math.pow(1 - clampedVdotH, 5);
    scale += (1 - fc) * gVis;
    bias += fc * gVis;
  }
  return [scale / SAMPLES, bias / SAMPLES];
}

/**
 * Builds the LUT and assigns it to the scene. Call before any PBR material is
 * created; returns the texture so callers can dispose it with the scene.
 */
export async function installEnvironmentBrdf(scene: any): Promise<unknown> {
  if (scene.environmentBRDFTexture) return scene.environmentBRDFTexture;
  const [{ RawTexture }, { Constants }, { Texture }] = await Promise.all([
    babylonImport('@babylonjs/core/Materials/Textures/rawTexture.js'),
    babylonImport('@babylonjs/core/Engines/constants.js'),
    babylonImport('@babylonjs/core/Materials/Textures/texture.js'),
  ]);
  const data = new Float32Array(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    // Row 0 is the roughest; Babylon samples with roughness on V.
    const roughness = (y + 0.5) / SIZE;
    for (let x = 0; x < SIZE; x++) {
      const nDotV = Math.max((x + 0.5) / SIZE, 1e-4);
      const [scale, bias] = integrate(nDotV, roughness);
      const offset = (y * SIZE + x) * 4;
      data[offset] = scale;
      data[offset + 1] = bias;
      data[offset + 2] = 0;
      data[offset + 3] = 1;
    }
  }
  const texture = RawTexture.CreateRGBATexture(
    data, SIZE, SIZE, scene, false, false, Texture.BILINEAR_SAMPLINGMODE, Constants.TEXTURETYPE_FLOAT,
  );
  texture.name = 'headless-environment-brdf';
  texture.wrapU = Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = Texture.CLAMP_ADDRESSMODE;
  scene.environmentBRDFTexture = texture;
  return texture;
}
