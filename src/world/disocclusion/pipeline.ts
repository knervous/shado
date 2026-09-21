/**
 * Offline orchestration: world + captures -> GPU stages -> classification ->
 * sidecar. Bake-only; the runtime never imports this.
 */
import type { ShadoWorldPrimitive, ShadoWorldSpatialPackage } from '../types';
import { bakeDisocclusionCapture, DisocclusionBakeError, type DisocclusionBakeOptions, type DisocclusionCaptureResult } from './bake';
import { classifyTargets, type DisocclusionTargetClassification } from './classify';
import { clusterRegions, geometryFromWorld } from './geometry';
import { captureFrame } from './layers';
import {
  DISOCCLUSION_GENERATOR_REVISION,
  DISOCCLUSION_SIDECAR_FORMAT,
  sha256Hex,
  worldIdentity,
  wordsToBytes,
  type DisocclusionDomainMeta,
  type DisocclusionSidecarMeta,
} from './sidecar';
import type { DisocclusionCapture, DisocclusionSettings } from './types';

export type DisocclusionDomainResult = {
  meta: DisocclusionDomainMeta;
  capture: DisocclusionCaptureResult;
  classification: DisocclusionTargetClassification;
  row: Uint32Array;
};

export type DisocclusionBakeOutput = {
  meta: DisocclusionSidecarMeta;
  words: Uint32Array;
  payload: Uint8Array;
  domains: DisocclusionDomainResult[];
};

/** Row admitting each region that owns an admitted cluster, and every region owning none. */
export function regionRow(world: ShadoWorldSpatialPackage, admittedClusters: Uint8Array): Uint32Array {
  const v = world.visibility!;
  const regions = v.width * v.height;
  const row = new Uint32Array(Math.ceil(regions / 32));
  const owned = new Uint8Array(regions);
  const owner = clusterRegions(world);
  owner.forEach((region, cluster) => {
    if (region < 0 || region >= regions) return;
    owned[region] = 1;
    if (admittedClusters[cluster]) row[region >>> 5]! |= (1 << (region & 31)) >>> 0;
  });
  for (let r = 0; r < regions; r++) if (!owned[r]) row[r >>> 5]! |= (1 << (r & 31)) >>> 0;
  return row;
}

export async function bakeDisocclusionPvs(
  device: GPUDevice,
  world: ShadoWorldSpatialPackage,
  primitives: readonly ShadoWorldPrimitive[],
  captures: readonly (DisocclusionCapture & { id?: string })[],
  settings: DisocclusionSettings,
  options: DisocclusionBakeOptions & { createdAt?: string } = {}
): Promise<DisocclusionBakeOutput> {
  if (!captures.length) throw new DisocclusionBakeError('no source domains', 'input', {});
  const identity = worldIdentity(world);
  const geometry = geometryFromWorld(world, primitives);
  const targets = world.clusters.firstIndex.length;
  const regions = identity.width * identity.height;
  const wordsPerRow = Math.ceil(regions / 32);
  const domains: DisocclusionDomainResult[] = [];
  for (const [i, capture] of captures.entries()) {
    const frame = captureFrame(capture);
    const result = await bakeDisocclusionCapture(device, frame, settings, geometry, options);
    const classification = classifyTargets(frame, settings, result.layers, result.masks, geometry, targets);
    const row = regionRow(world, classification.expanded);
    let regionsAdmitted = 0;
    for (let r = 0; r < regions; r++) regionsAdmitted += (row[r >>> 5]! >>> (r & 31)) & 1;
    const { id: _id, ...plainCapture } = capture;
    domains.push({
      meta: {
        id: capture.id ?? `domain-${i}`,
        capture: plainCapture,
        extTan: [frame.extTanX, frame.extTanY],
        viewcellHalf: [frame.viewcellHalfX, frame.viewcellHalfY],
        wordOffset: i * wordsPerRow,
        counts: { ...classification.counts, regionsAdmitted, regions },
        timings: result.timings,
      },
      capture: result,
      classification,
      row,
    });
  }
  const words = new Uint32Array(wordsPerRow * domains.length);
  domains.forEach((d, i) => words.set(d.row, i * wordsPerRow));
  const payload = wordsToBytes(words);
  const geometryBytes = new Uint8Array(geometry.positions.byteLength + geometry.indices.byteLength);
  geometryBytes.set(new Uint8Array(geometry.positions.buffer, geometry.positions.byteOffset, geometry.positions.byteLength), 0);
  geometryBytes.set(new Uint8Array(geometry.indices.buffer, geometry.indices.byteOffset, geometry.indices.byteLength), geometry.positions.byteLength);
  const meta: DisocclusionSidecarMeta = {
    format: DISOCCLUSION_SIDECAR_FORMAT,
    version: 1,
    experimental: true,
    generator: { name: 'shado-disocclusion', revision: DISOCCLUSION_GENERATOR_REVISION },
    createdAt: options.createdAt ?? new Date().toISOString(),
    world: identity,
    inputs: { geometrySha256: await sha256Hex(geometryBytes), triangles: geometry.triangleTarget.length, targets },
    settings,
    domains: domains.map(d => d.meta),
    payload: { endianness: 'little', wordsPerRow, wordCount: words.length, sha256: await sha256Hex(payload) },
  };
  return { meta, words, payload, domains };
}
