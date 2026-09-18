import { resolveTerrainLayer } from './terrain-palette';
import type { ShadoWorldTerrainMaterialAuthoring } from './types';

/** One decoded control map and the world rectangle it covers. */
export type TerrainControlRaster = {
  width: number;
  height: number;
  /** Row-major RGBA, 8 bits per component. */
  data: Uint8Array;
};

/**
 * Where the ground is not grass, as a raster over the terrain's own rectangle.
 *
 * Grass coverage used to be a purely geometric question — is there an
 * upward-facing surface, and is anything standing on it — which was right when
 * every non-grass floor in a zone WAS a piece of geometry. It stops being right
 * the moment the floor is painted: a dirt track, a stony shoulder and a paved
 * square are all the same terrain triangle as the meadow beside them, so the
 * bake happily grew a full lawn through all three.
 *
 * The rule is the obvious one and it is read off the authored layers rather
 * than hardcoded: a texel is suppressed where any layer whose material is NOT
 * grass has its control channel painted. An author who adds a paved layer on a
 * new channel gets the grass suppression for free, and an author who repaints a
 * track gets the lawn moved with it, because there is one mask and it is the
 * one the shader is already drawing from.
 *
 * Slope is deliberately NOT part of this. The coverage rasteriser already has a
 * `minimumUpNormal`, it applies to every zone rather than only to painted ones,
 * and duplicating it here would mean two numbers that have to agree about what
 * counts as too steep for grass.
 */
export function terrainGrassSuppression(
  terrain: ShadoWorldTerrainMaterialAuthoring | undefined | null,
  maps: readonly TerrainControlRaster[],
): { width: number; height: number; values: Uint8Array } | undefined {
  if (!terrain?.enabled || maps.length === 0) return undefined;
  const width = maps[0]!.width;
  const height = maps[0]!.height;
  /*
   * Every map has to describe the same rectangle at the same resolution, which
   * they do by construction — one paint pass writes them all over the terrain's
   * declared worldMin/worldMax. Refusing rather than resampling keeps that a
   * fact rather than a hope.
   */
  for (const map of maps) {
    if (map.width !== width || map.height !== height) {
      throw new Error(
        `Terrain control maps disagree about resolution (${map.width}x${map.height} vs ${width}x${height}); `
        + 'they describe one rectangle and must be rasterised together.',
      );
    }
  }

  const sources: Array<{ map: number; component: number }> = [];
  for (const layer of terrain.layers) {
    if (!layer.enabled) continue;
    const resolved = resolveTerrainLayer(layer);
    if (!resolved.control) continue;
    if (resolved.material.role === 'terrain.grass') continue;
    /* A layer painted where it should be ABSENT says nothing about grass. */
    if (resolved.controlSign < 0) continue;
    if (!maps[resolved.control.map]) continue;
    sources.push({ map: resolved.control.map, component: resolved.control.component });
  }
  if (sources.length === 0) return undefined;

  const values = new Uint8Array(width * height);
  for (let index = 0; index < values.length; index += 1) {
    let strongest = 0;
    for (const source of sources) {
      const painted = maps[source.map]!.data[index * 4 + source.component]!;
      if (painted > strongest) strongest = painted;
    }
    values[index] = strongest;
  }
  return { width, height, values };
}
