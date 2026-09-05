/**
 * The Eltania terrain palette: the shared list of ground materials a terrain
 * layer may name, and the rules for turning an authored layer into something a
 * shader can bind.
 *
 * This module is the contract between three places that previously had no way
 * to agree with each other:
 *
 * - Libra's terrain panel, which offers these materials in a dropdown instead
 *   of asking an author to type a key no validator ever checked;
 * - the world compiler, which resolves an authored layer into concrete texture
 *   URLs and shader constants at publish time;
 * - the client's terrain shader, which binds those URLs.
 *
 * Everything here is deliberately expressed in units an author can hold in
 * their head. A layer says a material repeats every six metres, that it takes
 * over on slopes between twenty and forty degrees, and that it appears where
 * the author painted "path wear". The reciprocal scales, normalised steepness
 * and channel indices the shader actually wants are derived at the boundary, in
 * one place, by `resolveTerrainLayer`.
 */

import { ELTANIA_TERRAIN_PALETTE_MATERIALS, ELTANIA_TERRAIN_ROLE_LABELS } from './terrain-palette.generated';
import type { ShadoWorldTerrainLayer, ShadoWorldTerrainProjection } from './types';

export type EltaniaTerrainRole =
  | 'terrain.grass'
  | 'terrain.path'
  | 'terrain.soil'
  | 'terrain.gravel'
  | 'terrain.rock';

export type EltaniaTerrainMaterial = {
  /** `family/key`, the value an authored layer stores in `material`. */
  id: string;
  family: string;
  key: string;
  /** Author-facing name, shown in the Libra dropdown. */
  label: string;
  role: EltaniaTerrainRole;
  note: string;
  /** Upstream scan this was built from, kept for provenance. */
  source: string;
  textures: { albedo: string; normal: string; hrm: string };
  /** Small square albedo thumbnail for editor swatches. */
  preview: string;
  /** World metres covered by one texture repeat at the material's natural scale. */
  tileMetres: number;
  normalScale: number;
  roughness: number;
  /**
   * How sharply this surface wins a height blend, 0 to 1. Gravel and scree sit
   * high because their stones should survive a boundary; smooth earth sits low
   * because a hard-edged mud boundary looks cut out rather than deposited.
   */
  heightContrast: number;
};

export { ELTANIA_TERRAIN_PALETTE_MATERIALS, ELTANIA_TERRAIN_ROLE_LABELS };

/**
 * The semantic control channels an author paints.
 *
 * The old painter asked for a "map slot" and an "RGBA channel", which are the
 * two least meaningful facts about a brush stroke — they describe where the
 * value is stored, not what painting it does. These are what an author is
 * actually deciding, and the storage location is an implementation detail of
 * the name.
 */
export type EltaniaTerrainControlChannel =
  | 'path'
  | 'growth'
  | 'exposure'
  | 'wetness';

export type EltaniaTerrainControlDefinition = {
  channel: EltaniaTerrainControlChannel;
  /** Control map index, then RGBA component index within it. */
  map: number;
  component: 0 | 1 | 2 | 3;
  label: string;
  /** What painting this channel up actually does to the ground. */
  effect: string;
};

export const ELTANIA_TERRAIN_CONTROLS: readonly EltaniaTerrainControlDefinition[] = [
  {
    channel: 'path',
    map: 0,
    component: 0,
    label: 'Path wear',
    effect: 'Suppresses grass, brings up packed earth and loose stones, and stops grass instances spawning.',
  },
  {
    channel: 'growth',
    map: 0,
    component: 1,
    label: 'Growth',
    effect: 'Pushes the ground toward its lushest grass layer and away from bare soil.',
  },
  {
    channel: 'exposure',
    map: 0,
    component: 2,
    label: 'Exposed stone',
    effect: 'Forces rock through regardless of slope, for outcrops and scoured ground.',
  },
  {
    channel: 'wetness',
    map: 0,
    component: 3,
    label: 'Wetness',
    effect: 'Darkens and smooths the surface, favours damp earth and moss, and deepens colour in hollows.',
  },
];

export const ELTANIA_TERRAIN_CONTROL_BY_CHANNEL: Readonly<Record<EltaniaTerrainControlChannel, EltaniaTerrainControlDefinition>> =
  Object.fromEntries(ELTANIA_TERRAIN_CONTROLS.map((entry) => [entry.channel, entry])) as Readonly<
    Record<EltaniaTerrainControlChannel, EltaniaTerrainControlDefinition>
  >;

const MATERIALS_BY_ID = new Map(ELTANIA_TERRAIN_PALETTE_MATERIALS.map((material) => [material.id, material]));

export function terrainMaterial(id: string): EltaniaTerrainMaterial | null {
  return MATERIALS_BY_ID.get(id) ?? null;
}

/** Palette grouped by role, in role order, for a grouped dropdown. */
export function terrainMaterialsByRole(): Array<{ role: EltaniaTerrainRole; label: string; materials: EltaniaTerrainMaterial[] }> {
  const roles: EltaniaTerrainRole[] = ['terrain.grass', 'terrain.path', 'terrain.soil', 'terrain.gravel', 'terrain.rock'];
  return roles.map((role) => ({
    role,
    label: ELTANIA_TERRAIN_ROLE_LABELS[role],
    materials: ELTANIA_TERRAIN_PALETTE_MATERIALS.filter((material) => material.role === role),
  })).filter((group) => group.materials.length > 0);
}

/**
 * A resolved layer: everything the shader needs, and nothing it has to look up.
 *
 * `tileScale` is the reciprocal of the layer's tile width because the shader
 * multiplies world XZ by it. That reciprocal is the number the old manifest
 * stored directly, which is why terrain tiling was unauthorable — `0.14` gives
 * an author no way to reason about how big a repeat is, and no way to notice
 * that the dirt and the grass were tiled at different real-world sizes.
 */
export type ResolvedTerrainLayer = {
  id: string;
  name: string;
  material: EltaniaTerrainMaterial;
  projection: ShadoWorldTerrainProjection;
  weight: number;
  tileScale: number;
  /** Cosine-space steepness window, 0 flat to 1 vertical. */
  slope: [number, number];
  altitude: [number, number];
  /** Reciprocal world size of the macro variation for this layer; 0 disables it. */
  noiseScale: number;
  normalScale: number;
  roughness: number;
  heightContrast: number;
  control: EltaniaTerrainControlDefinition | null;
  /** Signed influence of the control channel: paths add dirt and remove grass. */
  controlSign: number;
  /**
   * World metres the baked height field rises under a fully painted mask, and
   * the metres over which it ramps up at the mask edge.
   *
   * These two stay in author units where every other resolved field is
   * converted, because their consumer is the zone bake rather than the shader.
   * The bake works in world metres and knows its own zone-units-per-metre; a
   * reciprocal or a steepness here would only have to be undone.
   */
  protrusionMetres: number;
  protrusionFalloffMetres: number;
};

/**
 * Slope is authored in degrees and consumed as steepness, which is
 * `1 - |normal.y|`, i.e. `1 - cos(angle)`. Doing this conversion here rather
 * than in the shader is what lets the Libra panel say "rock takes over between
 * 25° and 50°" instead of "slope start 0.18, slope end 0.62" — the same two
 * numbers the character-select manifest carries today, which describe 35° and
 * 68° but say so to nobody.
 */
export function steepnessFromDegrees(degrees: number): number {
  return 1 - Math.cos((Math.max(0, Math.min(90, degrees)) * Math.PI) / 180);
}

export function degreesFromSteepness(steepness: number): number {
  return (Math.acos(1 - Math.max(0, Math.min(1, steepness))) * 180) / Math.PI;
}

/**
 * Layer metadata an author edits in degrees and metres. It lives in the layer's
 * open `metadata` bag rather than widening `ShadoWorldTerrainLayer`, so an
 * older document without it still resolves — the material's own natural tiling
 * and a slope window taken from the stored normalised values are both sensible
 * defaults.
 */
export type TerrainLayerAuthoring = {
  slopeDegrees?: [number, number];
  tileMetres?: number;
  controlChannel?: EltaniaTerrainControlChannel;
  /** Set false for a layer a control channel should suppress rather than add. */
  controlAdds?: boolean;
  macroMetres?: number;
};

function authoring(layer: ShadoWorldTerrainLayer): TerrainLayerAuthoring {
  const value = layer.metadata?.authoring;
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as TerrainLayerAuthoring) : {};
}

export function resolveTerrainLayer(layer: ShadoWorldTerrainLayer): ResolvedTerrainLayer {
  const material = terrainMaterial(layer.material);
  if (!material) {
    throw new Error(
      `Terrain layer "${layer.id}" names material "${layer.material}", which is not in the Eltania terrain palette. ` +
      `Known materials: ${ELTANIA_TERRAIN_PALETTE_MATERIALS.map((entry) => entry.id).join(', ')}`,
    );
  }
  const authored = authoring(layer);
  const tileMetres = authored.tileMetres ?? layer.textureScale ?? material.tileMetres;
  const slope: [number, number] = authored.slopeDegrees
    ? [steepnessFromDegrees(authored.slopeDegrees[0]), steepnessFromDegrees(authored.slopeDegrees[1])]
    : [layer.slope[0], layer.slope[1]];
  const control = authored.controlChannel ? ELTANIA_TERRAIN_CONTROL_BY_CHANNEL[authored.controlChannel] ?? null : null;
  const macroMetres = authored.macroMetres ?? layer.noiseScale;
  return {
    id: layer.id,
    name: layer.name,
    material,
    projection: layer.projection,
    weight: layer.weight,
    tileScale: 1 / Math.max(0.05, tileMetres),
    slope,
    altitude: layer.altitude,
    noiseScale: macroMetres > 0 ? 1 / Math.max(0.05, macroMetres) : 0,
    normalScale: material.normalScale,
    roughness: material.roughness,
    heightContrast: material.heightContrast,
    control,
    controlSign: authored.controlAdds === false ? -1 : 1,
    // No mask, no protrusion: a lift needs somewhere to say where it stops.
    protrusionMetres: control ? Math.max(0, layer.protrusionMetres ?? 0) : 0,
    protrusionFalloffMetres: Math.max(0, layer.protrusionFalloffMetres ?? DEFAULT_PROTRUSION_FALLOFF_METRES),
  };
}

/**
 * The falloff a protruding layer gets when its author has not chosen one.
 *
 * Two metres is a little under one 8-unit Crownward grid cell, which is the
 * coarsest thing the ramp has to be resolvable on, and about the width of the
 * verge beside a real road — narrow enough that the causeway still reads as
 * built, wide enough that nothing about it is a step.
 */
export const DEFAULT_PROTRUSION_FALLOFF_METRES = 2;

/**
 * The most layers the terrain shader binds at once.
 *
 * Each layer costs three texture samples on a planar surface and nine on a
 * fully triplanar one, so this is a real budget rather than a stylistic
 * preference. The authoring doc's 5-7 recommendation sits just under it.
 */
export const ELTANIA_TERRAIN_MAX_LAYERS = 8;

/**
 * A working default palette for a temperate Eltania zone: what an author gets
 * when they enable terrain on a zone that has none, so the first thing they see
 * is ground rather than an empty list.
 */
export function defaultTerrainLayers(): ShadoWorldTerrainLayer[] {
  const layer = (
    id: string,
    name: string,
    material: string,
    projection: ShadoWorldTerrainProjection,
    weight: number,
    slopeDegrees: [number, number],
    extra: TerrainLayerAuthoring = {},
  ): ShadoWorldTerrainLayer => {
    const resolved = terrainMaterial(material);
    return {
      id,
      name,
      enabled: true,
      material,
      projection,
      textureScale: resolved?.tileMetres ?? 6,
      weight,
      slope: [steepnessFromDegrees(slopeDegrees[0]), steepnessFromDegrees(slopeDegrees[1])],
      altitude: [-10000, 10000],
      noiseScale: extra.macroMetres ?? 45,
      metadata: { authoring: { slopeDegrees, tileMetres: resolved?.tileMetres, macroMetres: extra.macroMetres ?? 45, ...extra } },
    };
  };
  return [
    layer('ground-grass-lush', 'Meadow grass', 'eltania-ground-v1/grassLush', 'world-xz', 1, [0, 22], { controlChannel: 'growth' }),
    layer('ground-grass-sparse', 'Thinning grass', 'eltania-ground-v1/grassSparse', 'world-xz', 0.6, [12, 34], { macroMetres: 22 }),
    layer('ground-earth-path', 'Packed earth', 'eltania-ground-v1/earthPacked', 'world-xz', 0.35, [0, 40], { controlChannel: 'path', macroMetres: 18 }),
    layer('ground-scree', 'Scree', 'eltania-ground-v1/scree', 'world-xz', 0.5, [26, 48], { macroMetres: 26 }),
    layer('ground-cliff', 'Cliff stone', 'eltania-ground-v1/cliffRock', 'hybrid', 1, [30, 62], { controlChannel: 'exposure' }),
    layer('ground-rock-wet', 'Wet stone', 'eltania-ground-v1/rockMossy', 'hybrid', 0.5, [24, 70], { controlChannel: 'wetness', macroMetres: 30 }),
  ];
}
