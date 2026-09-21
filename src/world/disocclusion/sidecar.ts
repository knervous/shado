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
  counts: { targets: number; raw: number; filtered: number; expanded: number; admitted: number; filterOnly: number; regionsAdmitted: number; regions: number };
  timings: Record<string, number>;
};

export type DisocclusionSidecarMeta = {
  format: typeof DISOCCLUSION_SIDECAR_FORMAT;
  version: 1;
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
    };
  };
  settings: DisocclusionSettings;
  domains: DisocclusionDomainMeta[];
  payload: { endianness: 'little'; wordsPerRow: number; wordCount: number; sha256: string };
};

export type DisocclusionSidecar = { meta: DisocclusionSidecarMeta; words: Uint32Array };

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
  if (meta.version !== 1) throw new DisocclusionSidecarError(`unsupported version ${meta.version}`);
  if (meta.experimental !== true) throw new DisocclusionSidecarError('sidecar must be marked experimental');
  assertFinite(meta, 'meta');
  const { world, payload: p, domains } = meta;
  const regions = world.width * world.height;
  const wordsPerRow = Math.ceil(regions / 32);
  if (p.endianness !== 'little') throw new DisocclusionSidecarError('payload must be little-endian');
  if (p.wordsPerRow !== wordsPerRow) throw new DisocclusionSidecarError(`wordsPerRow ${p.wordsPerRow} != ${wordsPerRow}`);
  if (p.wordCount !== wordsPerRow * domains.length) throw new DisocclusionSidecarError('payload word count does not match domains');
  if (payload.byteLength !== p.wordCount * 4) throw new DisocclusionSidecarError(`payload is ${payload.byteLength} bytes, expected ${p.wordCount * 4}`);
  const hash = await sha256Hex(payload);
  if (hash !== p.sha256) throw new DisocclusionSidecarError('payload hash mismatch');
  domains.forEach((d, i) => {
    if (d.wordOffset !== i * wordsPerRow) throw new DisocclusionSidecarError(`domain ${d.id} has wordOffset ${d.wordOffset}`);
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
  files: { spatial?: Uint8Array; glb?: Uint8Array } = {}
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
