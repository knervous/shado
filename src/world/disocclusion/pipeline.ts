/**
 * Offline orchestration: world + captures -> GPU stages -> classification ->
 * sidecar. Bake-only; the runtime never imports this.
 */
import type { ShadoWorldPrimitive, ShadoWorldSpatialPackage } from '../types';
import { DisocclusionBaker, DisocclusionBakeError, type DisocclusionBakeOptions, type DisocclusionCaptureResult } from './bake';
import { classifyTargets, type DisocclusionTargetClassification } from './classify';
import { clusterRegions, geometryFromWorld } from './geometry';
import { captureFrame } from './layers';
import {
  clusterRegionDigest,
  DISOCCLUSION_GENERATOR_REVISION,
  DISOCCLUSION_SIDECAR_FORMAT,
  sha256Hex,
  worldIdentity,
  wordsToBytes,
  type DisocclusionDomainMeta,
  type DisocclusionSidecarMeta,
} from './sidecar';
import type { DisocclusionCapture, DisocclusionGeometry, DisocclusionSettings } from './types';
import type { DisocclusionSidecarMeta as Meta } from './sidecar';

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

export type DisocclusionPvsOptions = DisocclusionBakeOptions & {
  createdAt?: string;
  /** Precomputed input (a real zone); defaults to the world's own clusters from `primitives`. */
  geometry?: DisocclusionGeometry;
  zone?: NonNullable<Meta['inputs']['zone']>;
  /** Called after each face with its timings. */
  onFace?: (id: string, timings: Record<string, number>) => void;
  /** Target clusters admitted in every row whatever the raster says (suspect frames, V3). */
  alwaysAdmitClusters?: readonly number[];
};

const bytesOf = (view: ArrayBufferView) => new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

export async function bakeDisocclusionPvs(
  device: GPUDevice,
  world: ShadoWorldSpatialPackage,
  primitives: readonly ShadoWorldPrimitive[],
  captures: readonly (DisocclusionCapture & { id?: string })[],
  settings: DisocclusionSettings,
  options: DisocclusionPvsOptions = {}
): Promise<DisocclusionBakeOutput> {
  if (!captures.length) throw new DisocclusionBakeError('no source domains', 'input', {});
  const identity = worldIdentity(world);
  const geometry = options.geometry ?? geometryFromWorld(world, primitives);
  const targets = world.clusters.firstIndex.length;
  const regions = identity.width * identity.height;
  const wordsPerRow = Math.ceil(regions / 32);
  const domains: DisocclusionDomainResult[] = [];
  // One baker: geometry, pipelines and working buffers shared by every face.
  const baker = await DisocclusionBaker.create(device, settings, geometry, options);
  try {
    for (const [i, capture] of captures.entries()) {
      const frame = captureFrame(capture);
      const result = await baker.capture(frame);
      const classifyStarted = performance.now();
      const classification = classifyTargets(frame, settings, result.layers, result.masks, geometry, targets);
      for (const cluster of options.alwaysAdmitClusters ?? []) {
        if (cluster >= 0 && cluster < targets) classification.admitted[cluster] = 1;
      }
      const row = regionRow(world, classification.admitted);
      result.timings.classifyMs = performance.now() - classifyStarted;
      let regionsAdmitted = 0;
      for (let r = 0; r < regions; r++) regionsAdmitted += (row[r >>> 5]! >>> (r & 31)) & 1;
      const { id: _id, ...plainCapture } = capture;
      const id = capture.id ?? `domain-${i}`;
      options.onFace?.(id, result.timings);
      domains.push({
        meta: {
          id,
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
  } finally {
    baker.dispose();
  }
  const words = new Uint32Array(wordsPerRow * domains.length);
  domains.forEach((d, i) => words.set(d.row, i * wordsPerRow));
  const payload = wordsToBytes(words);
  const geometryBytes = new Uint8Array(geometry.positions.byteLength + geometry.indices.byteLength);
  geometryBytes.set(bytesOf(geometry.positions), 0);
  geometryBytes.set(bytesOf(geometry.indices), geometry.positions.byteLength);
  const triangles = geometry.triangleTarget.length;
  const blockerFlags = new Uint8Array(triangles * 2);
  let blockerTriangles = 0;
  for (let t = 0; t < triangles; t++) {
    const blocks = !geometry.blocker || geometry.blocker[t] ? 1 : 0;
    blockerFlags[t * 2] = blocks;
    blockerFlags[t * 2 + 1] = !geometry.doubleSided || geometry.doubleSided[t] ? 1 : 0;
    blockerTriangles += blocks;
  }
  const meta: DisocclusionSidecarMeta = {
    format: DISOCCLUSION_SIDECAR_FORMAT,
    version: 1,
    experimental: true,
    generator: { name: 'shado-disocclusion', revision: DISOCCLUSION_GENERATOR_REVISION },
    createdAt: options.createdAt ?? new Date().toISOString(),
    world: identity,
    inputs: {
      geometrySha256: await sha256Hex(geometryBytes),
      triangles,
      targets,
      blockerSha256: await sha256Hex(blockerFlags),
      blockerTriangles,
      targetMapSha256: await sha256Hex(bytesOf(geometry.triangleTarget)),
      clusterRegionSha256: await clusterRegionDigest(world),
      settingsSha256: await sha256Hex(new TextEncoder().encode(JSON.stringify(settings))),
      ...(options.zone ? { zone: options.zone } : {}),
      ...(options.alwaysAdmitClusters?.length ? { suspectClusters: [...options.alwaysAdmitClusters] } : {}),
    },
    settings,
    domains: domains.map(d => d.meta),
    payload: { endianness: 'little', wordsPerRow, wordCount: words.length, sha256: await sha256Hex(payload) },
  };
  return { meta, words, payload, domains };
}
