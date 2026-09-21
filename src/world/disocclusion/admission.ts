/**
 * Runtime side of the experiment: camera -> supported domains -> region row
 * -> byte-per-cell mask. The mask feeds the EXISTING coordinator through
 * `ShadoWorldVisibilityMasks.portalReachableCells`, which the reducer ANDs
 * into every cluster's admission. No coordinator change, no second manager.
 *
 * Anything uncertain -- no sidecar, identity mismatch, camera outside every
 * domain or too close to its boundary, a view direction the bake did not
 * capture -- yields `mode: 'reference'` with a reason and a null mask, and the
 * caller passes no mask at all: exactly the reference path, on the same frame.
 */
import type { ShadoWorldSpatialPackage } from '../types';
import { captureFrame, dot } from './layers';
import { sidecarMismatch, type DisocclusionSidecar } from './sidecar';
import type { DisocclusionFrame, Vec3 } from './types';

export type DisocclusionCameraPose = {
  position: Vec3;
  /** World-space directions of the four frustum corner rays (any length). */
  cornerRays: Vec3[];
};

export type DisocclusionAdmissionResult = {
  mode: 'reference' | 'baked';
  reason: string;
  /** Domain ids whose rows were unioned. */
  domains: string[];
  /** Region bits admitted (union of rows), or null on the reference path. */
  regionWords: Uint32Array | null;
  /** Byte per render cell, or null on the reference path. */
  cellMask: Uint8Array | null;
  admittedRegions: number;
  admittedCells: number;
};

/** Distance a camera must keep from a domain face before the bake applies. */
export const DOMAIN_BOUNDARY_EPSILON = 1e-3;

export function supportsPose(frame: DisocclusionFrame, min: Vec3, max: Vec3, pose: DisocclusionCameraPose): string | null {
  const e = DOMAIN_BOUNDARY_EPSILON;
  for (let a = 0; a < 3; a++) {
    const p = pose.position[a]!;
    if (!(p >= min[a]! + e && p <= max[a]! - e)) return 'camera outside source box';
  }
  for (const ray of pose.cornerRays) {
    const axial = dot(ray, frame.forward);
    if (!(axial > 0)) return 'view direction behind capture';
    if (Math.abs(dot(ray, frame.right)) / axial > frame.directionTanX) return 'view direction outside capture (lateral)';
    if (Math.abs(dot(ray, frame.up)) / axial > frame.directionTanY) return 'view direction outside capture (vertical)';
  }
  return null;
}

export class DisocclusionAdmission {
  private readonly frames: DisocclusionFrame[];
  private readonly regionCount: number;
  private readonly wordsPerRow: number;
  private readonly cellRegion: readonly number[];
  readonly identityError: string | null;

  constructor(
    private readonly world: ShadoWorldSpatialPackage,
    readonly sidecar: DisocclusionSidecar | null
  ) {
    const v = world.visibility;
    this.regionCount = v ? v.width * v.height : 0;
    this.wordsPerRow = Math.ceil(this.regionCount / 32);
    this.cellRegion = v?.cellRegion ?? [];
    this.identityError = !v ? 'world has no dense regions' : sidecar ? sidecarMismatch(sidecar, world) : 'no sidecar loaded';
    this.frames = sidecar && !this.identityError ? sidecar.meta.domains.map(d => captureFrame(d.capture)) : [];
  }

  evaluate(pose: DisocclusionCameraPose): DisocclusionAdmissionResult {
    const reference = (reason: string): DisocclusionAdmissionResult => ({
      mode: 'reference',
      reason,
      domains: [],
      regionWords: null,
      cellMask: null,
      admittedRegions: this.regionCount,
      admittedCells: this.cellRegion.length,
    });
    if (this.identityError || !this.sidecar) return reference(this.identityError ?? 'no sidecar loaded');
    const words = new Uint32Array(this.wordsPerRow);
    const used: string[] = [];
    let lastReason = 'no domain';
    this.sidecar.meta.domains.forEach((domain, i) => {
      const why = supportsPose(this.frames[i]!, domain.capture.sourceMin, domain.capture.sourceMax, pose);
      if (why) {
        lastReason = `${domain.id}: ${why}`;
        return;
      }
      used.push(domain.id);
      for (let w = 0; w < this.wordsPerRow; w++) words[w]! |= this.sidecar!.words[domain.wordOffset + w]!;
    });
    if (!used.length) return reference(lastReason);
    const cellMask = new Uint8Array(this.cellRegion.length);
    let admittedCells = 0;
    this.cellRegion.forEach((region, cell) => {
      // A cell outside the dense grid is unknown: admit it.
      const admitted = region < 0 || region >= this.regionCount || (words[region >>> 5]! >>> (region & 31)) & 1;
      if (admitted) {
        cellMask[cell] = 1;
        admittedCells++;
      }
    });
    let admittedRegions = 0;
    for (let r = 0; r < this.regionCount; r++) admittedRegions += (words[r >>> 5]! >>> (r & 31)) & 1;
    return { mode: 'baked', reason: 'supported', domains: used, regionWords: words, cellMask, admittedRegions, admittedCells };
  }
}
