/**
 * Compiles an authored terrain document into the spec the runtime binds.
 *
 * This is the step that did not exist. `ShadoWorldTerrainMaterialAuthoring` was
 * stored, budget-checked and cloned into the compiled world, and then nothing
 * read it: the client took its terrain settings from `eltania.terrain` extras
 * baked into zone GLB materials by one-off Python authoring scripts, so a layer
 * added in Libra changed nothing and a layer removed in Libra changed nothing.
 * Two authorities, one of which no author could reach.
 *
 * With this, the authored document is the only authority. Libra edits it, the
 * compiler resolves it against the shared palette, and both the client and the
 * Libra viewer bind the identical spec — which is also what makes painting
 * WYSIWYG, since the editor is no longer approximating what the game will do.
 */

import {
  ELTANIA_TERRAIN_MAX_LAYERS,
  resolveTerrainLayer,
  steepnessFromDegrees,
  type ResolvedTerrainLayer,
} from './terrain-palette';
import type { ShadoWorldTerrainMaterialAuthoring } from './types';

/** One layer, flattened to the constants a shader wants and nothing else. */
export type EltaniaTerrainSpecLayer = {
  id: string;
  name: string;
  material: string;
  textures: { albedo: string; normal: string; hrm: string };
  projection: 'world-xz' | 'triplanar' | 'hybrid';
  weight: number;
  /** Reciprocal tile width; multiply world XZ by this to get UV. */
  tileScale: number;
  slope: [number, number];
  altitude: [number, number];
  noiseScale: number;
  noiseAmount: number;
  normalScale: number;
  /** Multiplied onto the sampled roughness, so a surface can be pushed wet or dry without a new scan. */
  roughness: number;
  heightContrast: number;
  /** RGBA component of control map 0 this layer answers to, or -1. */
  controlComponent: number;
  controlSign: number;
  controlGain: number;
};

export type EltaniaTerrainSurfaceSpec = {
  schema: 'eltania.terrain.surface';
  version: 2;
  layers: EltaniaTerrainSpecLayer[];
  controlMaps: string[];
  worldMin: [number, number];
  worldMax: [number, number];
  /** Steepness window over which a hybrid layer crosses to triplanar. */
  triplanarSlope: [number, number];
  /**
   * How decisively the tallest layer wins where two overlap. Low values cross-
   * fade and read as fog on the ground; high values cut and read as a decal.
   */
  heightBlendSharpness: number;
  /** Reciprocal world size of the shared macro colour variation. */
  macroScale: number;
  macroStrength: number;
  /**
   * How completely painted path wear clears everything that is not itself
   * path-driven. A road is a road: at 1 nothing else survives under it.
   */
  pathSuppression: number;
  /** Multiplied into every layer's albedo; where a zone's mood lives. */
  biomeTint: [number, number, number];
};

export type TerrainCompileOptions = {
  worldMin: [number, number];
  worldMax: [number, number];
  /** Degrees of slope over which hybrid layers cross to triplanar projection. */
  triplanarDegrees?: [number, number];
  heightBlendSharpness?: number;
  macroMetres?: number;
  macroStrength?: number;
  pathSuppression?: number;
  biomeTint?: [number, number, number];
};

/**
 * Control gain per layer.
 *
 * A painted channel has to be able to overrule the procedural rules outright —
 * an author painting a road across a meadow means a road, not a suggestion —
 * so a fully painted control contributes several times any layer's own weight.
 *
 * The gain is absolute, not a multiple of the layer's base weight. Scaling it
 * by that weight meant the quieter a layer was by default, the less painting it
 * could do: a path layer sitting at 0.4 could only ever add 1.04, which lost
 * outright to grass that the growth channel had already lifted to 2.87. The
 * brush felt inert precisely where an author most wanted it to bite.
 */
const CONTROL_GAIN = 2.6;

/** Fraction by which macro noise may swing a layer's weight. */
const LAYER_NOISE_AMOUNT = 0.42;

export function compileTerrainSurface(
  terrain: ShadoWorldTerrainMaterialAuthoring,
  options: TerrainCompileOptions,
): EltaniaTerrainSurfaceSpec | null {
  if (!terrain?.enabled) return null;
  const enabled = terrain.layers.filter((layer) => layer.enabled);
  if (enabled.length === 0) return null;
  if (enabled.length > ELTANIA_TERRAIN_MAX_LAYERS) {
    throw new Error(
      `Terrain has ${enabled.length} enabled layers; the runtime binds at most ${ELTANIA_TERRAIN_MAX_LAYERS}. ` +
      'Disable a layer or fold two together.',
    );
  }
  /**
   * A control map is optional.
   *
   * The whole design is procedural first and painted second: slope, altitude
   * and macro noise decide the predictable behaviour, and an author paints only
   * the exceptions. Terrain that has never been painted is therefore a complete
   * and valid surface, and demanding a published control map before anything
   * could compile inverted that — it broke the exact order an author works in,
   * enabling terrain and choosing layers before deciding where the paths go.
   * The runtime binds a neutral map when none is published.
   */

  const resolved: ResolvedTerrainLayer[] = enabled.map((layer) => resolveTerrainLayer(layer));
  const triplanarDegrees = options.triplanarDegrees ?? [24, 46];
  const macroMetres = options.macroMetres ?? 60;

  return {
    schema: 'eltania.terrain.surface',
    version: 2,
    layers: resolved.map((layer) => ({
      id: layer.id,
      name: layer.name,
      material: layer.material.id,
      textures: layer.material.textures,
      projection: layer.projection,
      weight: layer.weight,
      tileScale: layer.tileScale,
      slope: layer.slope,
      altitude: layer.altitude,
      noiseScale: layer.noiseScale,
      noiseAmount: layer.noiseScale > 0 ? LAYER_NOISE_AMOUNT : 0,
      normalScale: layer.normalScale,
      roughness: layer.roughness,
      heightContrast: layer.heightContrast,
      controlComponent: layer.control ? layer.control.component : -1,
      controlSign: layer.controlSign,
      controlGain: CONTROL_GAIN,
    })),
    controlMaps: [...terrain.controlMaps],
    worldMin: options.worldMin,
    worldMax: options.worldMax,
    triplanarSlope: [steepnessFromDegrees(triplanarDegrees[0]), steepnessFromDegrees(triplanarDegrees[1])],
    heightBlendSharpness: options.heightBlendSharpness ?? 0.14,
    macroScale: 1 / Math.max(1, macroMetres),
    macroStrength: options.macroStrength ?? 0.22,
    pathSuppression: options.pathSuppression ?? 0.92,
    biomeTint: options.biomeTint ?? [1, 1, 1],
  };
}

/**
 * Every distinct texture the spec needs, in the order the runtime packs its
 * array slices. Layer `i` is slice `i` in all three arrays, so this is the
 * loader's whole contract.
 */
export function terrainSpecTextures(spec: EltaniaTerrainSurfaceSpec): {
  albedo: string[];
  normal: string[];
  hrm: string[];
} {
  return {
    albedo: spec.layers.map((layer) => layer.textures.albedo),
    normal: spec.layers.map((layer) => layer.textures.normal),
    hrm: spec.layers.map((layer) => layer.textures.hrm),
  };
}
