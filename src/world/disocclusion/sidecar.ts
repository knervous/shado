/**
 * The experimental sidecar: JSON metadata plus a little-endian Uint32 payload
 * of per-domain target-region rows. Plain data only: no infinities, no GPU
 * handles. Runtime-safe (no Node or WebGPU imports).
 */
import type { ShadoWorldSpatialPackage } from '../types';
import { computeShadoWorldLayoutHash } from '../validation';
import type { DisocclusionCapture, DisocclusionSettings } from './types';

export const DISOCCLUSION_SIDECAR_FORMAT = 'eltania-disocclusion-pvs';
export const DISOCCLUSION_GENERATOR_REVISION = 'dpvs-1';

export type DisocclusionDomainMeta = {
  id: string;
  capture: DisocclusionCapture;
  /** Derived; recorded so a reader can display it without recomputing. */
  extTan: [number, number];
  viewcellHalf: [number, number];
  /** First payload word of this domain's row. */
  wordOffset: number;
  /** First payload word of this domain's stamp row (version 2). */
  stampWordOffset?: number;
  counts: {
    targets: number;
    raw: number;
    filtered: number;
    expanded: number;
    admitted: number;
    filterOnly: number;
    regionsAdmitted: number;
    regions: number;
    stampsAdmitted?: number;
    stamps?: number;
  };
  timings: Record<string, number>;
};

export type DisocclusionSidecarMeta = {
  format: typeof DISOCCLUSION_SIDECAR_FORMAT;
  /** 2 adds per-stamp rows (placed objects as targets). */
  version: 1 | 2;
  /** Version 2: one bit per placed-object stamp per domain, after the region rows. */
  stamps?: { count: number; wordsPerRow: number };
  experimental: true;
  generator: { name: 'shado-disocclusion'; revision: string };
  createdAt: string;
  world: {
    name: string;
    layoutHash: string;
    regionSize: number;
    originX: number;
    originZ: number;
    width: number;
    height: number;
  };
  inputs: {
    geometrySha256: string;
    triangles: number;
    targets: number;
    /** Blocker eligibility and sidedness per triangle. */
    blockerSha256: string;
    blockerTriangles: number;
    /** Triangle -> target cluster. */
    targetMapSha256: string;
    /** Cluster -> region, recomputable from the world package alone. */
    clusterRegionSha256: string;
    settingsSha256: string;
    /** Real-zone inputs, when the bake read a zone: file digests a reader can re-check. */
    zone?: {
      name: string;
      spatialSha256: string;
      glbSha256: string;
      prototypesSha256: string;
      prototypeFiles: number;
      releaseRevision?: string;
      /**
       * Every placed-object prototype the stamps reference, as content (V3).
       * A reader re-hashes what IT resolves for each source; any difference,
       * or a file it cannot verify, makes the sidecar unsupported. Absent on
       * sidecars baked before V3, which are therefore unsupported too.
       */
      prototypes?: DisocclusionPrototypeManifest;
    };
    /** Clusters whose recovered frame disagreed with the package: admitted in every row, never blockers. */
    suspectClusters?: number[];
  };
  settings: DisocclusionSettings;
  domains: DisocclusionDomainMeta[];
  payload: { endianness: 'little'; wordsPerRow: number; wordCount: number; sha256: string };
};

export type DisocclusionSidecar = { meta: DisocclusionSidecarMeta; words: Uint32Array };

export type DisocclusionPrototypeManifest = {
  /** SHADO_OCCLUDER_ELIGIBILITY_REVISION the bake applied to these files. */
  eligibilityRevision: string;
  /** Sorted by source. sha256 of the DECOMPRESSED content; null = not found at bake time. */
  files: Array<{ source: string; sha256: string | null }>;
};

/**
 * Resolves a prototype source to the content the client actually loads
 * (decompressed), or null when the client cannot find it.
 */
export type DisocclusionPrototypeResolver = (source: string) => Promise<Uint8Array | null>;

/**
 * Checks the bake's placed-object blockers against what this client
 * resolves. Null when every prototype matches; otherwise the reason. `cache`
 * keys content digests by `release|source` so a hot reload of the sidecar in
 * the same asset release does not refetch or rehash anything.
 */
export async function verifyPrototypeManifest(
  world: ShadoWorldSpatialPackage,
  manifest: DisocclusionPrototypeManifest | undefined,
  resolve: DisocclusionPrototypeResolver,
  options: { eligibilityRevision: string; release?: string; cache?: Map<string, string | null> }
): Promise<string | null> {
  if (!manifest) return 'sidecar has no prototype manifest (baked before blocker validation)';
  if (manifest.eligibilityRevision !== options.eligibilityRevision) {
    return `occluder eligibility ${manifest.eligibilityRevision} != reader ${options.eligibilityRevision}`;
  }
  const referenced = [...new Set(world.objects?.prototypes.source ?? [])].sort();
  const listed = manifest.files.map(file => file.source);
  if (referenced.length !== listed.length || referenced.some((source, i) => source !== listed[i])) {
    return 'prototype manifest does not cover the prototypes this world references';
  }
  for (const file of manifest.files) {
    const key = `${options.release ?? ''}|${file.source}`;
    let digest = options.cache?.get(key);
    if (digest === undefined) {
      let content: Uint8Array | null;
      try {
        content = await resolve(file.source);
      } catch {
        return `prototype ${file.source} could not be resolved to verify it`;
      }
      digest = content ? await sha256Hex(content) : null;
      options.cache?.set(key, digest);
    }
    if (file.sha256 === null) {
      // Missing at bake time, so it blocked nothing in the bake. Resolving
      // now means the inputs drifted; refuse rather than argue which
      // direction the drift is safe in.
      if (digest !== null) return `prototype ${file.source} was missing at bake time but resolves now`;
      continue;
    }
    if (digest === null) return `prototype ${file.source} does not resolve (blocker content unverified)`;
    if (digest !== file.sha256) return `prototype ${file.source} content differs from the baked blocker`;
  }
  return null;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes as unknown as ArrayBuffer);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

export function wordsToBytes(words: Uint32Array): Uint8Array {
  const bytes = new Uint8Array(words.length * 4);
  const view = new DataView(bytes.buffer);
  words.forEach((w, i) => view.setUint32(i * 4, w, true));
  return bytes;
}

function bytesToWords(bytes: Uint8Array): Uint32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Uint32Array(bytes.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}

export function worldIdentity(world: ShadoWorldSpatialPackage): DisocclusionSidecarMeta['world'] {
  const v = world.visibility;
  if (!v) throw new Error(`world '${world.name}' has no dense visibility regions`);
  return {
    name: world.name,
    layoutHash: computeShadoWorldLayoutHash(world),
    regionSize: v.size,
    originX: v.originX,
    originZ: v.originZ,
    width: v.width,
    height: v.height,
  };
}

export class DisocclusionSidecarError extends Error {}

function assertFinite(value: unknown, path: string): void {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new DisocclusionSidecarError(`${path} is not finite`);
    return;
  }
  if (value === null) throw new DisocclusionSidecarError(`${path} is null`);
  if (Array.isArray(value)) value.forEach((v, i) => assertFinite(v, `${path}[${i}]`));
  else if (typeof value === 'object') for (const [k, v] of Object.entries(value as object)) assertFinite(v, `${path}.${k}`);
}

/**
 * Parse and fully validate a sidecar: shape, finiteness, payload length and
 * hash, row addressing. Identity against a world is a separate check
 * (`matchesWorld`) so a reader can report WHY it fell back.
 */
export async function decodeDisocclusionSidecar(json: string, payload: Uint8Array): Promise<DisocclusionSidecar> {
  let meta: DisocclusionSidecarMeta;
  try {
    meta = JSON.parse(json) as DisocclusionSidecarMeta;
  } catch (error) {
    throw new DisocclusionSidecarError(`sidecar metadata is not JSON: ${String(error)}`);
  }
  if (meta.format !== DISOCCLUSION_SIDECAR_FORMAT) throw new DisocclusionSidecarError(`unexpected format '${meta.format}'`);
  if (meta.version !== 1 && meta.version !== 2) throw new DisocclusionSidecarError(`unsupported version ${meta.version}`);
  if ((meta.version === 2) !== !!meta.stamps) throw new DisocclusionSidecarError('version 2 carries stamp rows, version 1 does not');
  if (meta.experimental !== true) throw new DisocclusionSidecarError('sidecar must be marked experimental');
  assertFinite(meta, 'meta');
  const { world, payload: p, domains } = meta;
  const regions = world.width * world.height;
  const wordsPerRow = Math.ceil(regions / 32);
  if (p.endianness !== 'little') throw new DisocclusionSidecarError('payload must be little-endian');
  if (p.wordsPerRow !== wordsPerRow) throw new DisocclusionSidecarError(`wordsPerRow ${p.wordsPerRow} != ${wordsPerRow}`);
  const stampWords = meta.stamps ? meta.stamps.wordsPerRow : 0;
  if (meta.stamps && stampWords !== Math.ceil(meta.stamps.count / 32)) throw new DisocclusionSidecarError('stamp wordsPerRow does not match the stamp count');
  if (p.wordCount !== (wordsPerRow + stampWords) * domains.length) throw new DisocclusionSidecarError('payload word count does not match domains');
  if (payload.byteLength !== p.wordCount * 4) throw new DisocclusionSidecarError(`payload is ${payload.byteLength} bytes, expected ${p.wordCount * 4}`);
  const hash = await sha256Hex(payload);
  if (hash !== p.sha256) throw new DisocclusionSidecarError('payload hash mismatch');
  domains.forEach((d, i) => {
    if (d.wordOffset !== i * wordsPerRow) throw new DisocclusionSidecarError(`domain ${d.id} has wordOffset ${d.wordOffset}`);
    if (meta.stamps && d.stampWordOffset !== wordsPerRow * domains.length + i * stampWords) {
      throw new DisocclusionSidecarError(`domain ${d.id} has stampWordOffset ${d.stampWordOffset}`);
    }
    for (let a = 0; a < 3; a++) {
      if (!(d.capture.sourceMax[a]! > d.capture.sourceMin[a]!)) throw new DisocclusionSidecarError(`domain ${d.id} has an empty source box`);
    }
  });
  const words = bytesToWords(payload);
  // Padding bits past the last region must be clear.
  const tail = regions % 32;
  if (tail) {
    const padding = ~((1 << tail) - 1) >>> 0;
    domains.forEach(d => {
      if ((words[d.wordOffset + wordsPerRow - 1]! & padding) !== 0) throw new DisocclusionSidecarError(`domain ${d.id} sets padding bits`);
    });
  }
  const stampTail = meta.stamps ? meta.stamps.count % 32 : 0;
  if (meta.stamps && stampTail) {
    const padding = ~((1 << stampTail) - 1) >>> 0;
    domains.forEach(d => {
      if ((words[d.stampWordOffset! + stampWords - 1]! & padding) !== 0) throw new DisocclusionSidecarError(`domain ${d.id} sets stamp padding bits`);
    });
  }
  return { meta, words };
}

/** Digest of the cluster -> region mapping the rows are addressed through. */
export async function clusterRegionDigest(world: ShadoWorldSpatialPackage): Promise<string> {
  const cellRegion = world.visibility?.cellRegion ?? [];
  const regions = Int32Array.from(world.clusters.cellId, cell => cellRegion[cell] ?? -1);
  return sha256Hex(new Uint8Array(regions.buffer));
}

/**
 * The deep identity check a reader can make: the cluster -> region mapping
 * from the loaded package and, when given, the digests of the exact spatial
 * and GLB bytes the bake read. Null when everything checked matches.
 */
export async function verifySidecarInputs(
  sidecar: DisocclusionSidecar,
  world: ShadoWorldSpatialPackage,
  files: {
    spatial?: Uint8Array;
    glb?: Uint8Array;
    /** Required for a zone bake: placed-object blockers are verified or the sidecar is unsupported. */
    prototypes?: {
      resolve: DisocclusionPrototypeResolver;
      eligibilityRevision: string;
      release?: string;
      cache?: Map<string, string | null>;
    };
  } = {}
): Promise<string | null> {
  const shallow = sidecarMismatch(sidecar, world);
  if (shallow) return shallow;
  const inputs = sidecar.meta.inputs;
  if (inputs.clusterRegionSha256 !== (await clusterRegionDigest(world))) return 'cluster -> region mapping differs';
  if (inputs.zone && files.spatial && inputs.zone.spatialSha256 !== (await sha256Hex(files.spatial))) {
    return 'spatial package bytes differ from the baked input';
  }
  if (inputs.zone && files.glb && inputs.zone.glbSha256 !== (await sha256Hex(files.glb))) {
    return 'world GLB bytes differ from the baked input';
  }
  if (inputs.zone) {
    // Placed objects are most of a town's blockers (Crownward: 4.1 M of
    // 4.3 M triangles); unverified, the row may hide what an edit exposed.
    if (!files.prototypes) return 'placed-object blockers not verified';
    const { resolve, ...options } = files.prototypes;
    const prototypes = await verifyPrototypeManifest(world, inputs.zone.prototypes, resolve, options);
    if (prototypes) return prototypes;
  }
  return null;
}

/** Null when the sidecar was baked for this exact world layout, else the reason. */
export function sidecarMismatch(sidecar: DisocclusionSidecar, world: ShadoWorldSpatialPackage): string | null {
  const expected = worldIdentity(world);
  const actual = sidecar.meta.world;
  for (const key of Object.keys(expected) as (keyof typeof expected)[]) {
    if (expected[key] !== actual[key]) return `world ${key} differs (sidecar ${actual[key]}, world ${expected[key]})`;
  }
  return null;
}
