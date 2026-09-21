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

export function insideSource(min: Vec3, max: Vec3, position: Vec3): boolean {
  const e = DOMAIN_BOUNDARY_EPSILON;
  for (let a = 0; a < 3; a++) {
    const p = position[a]!;
    if (!(p >= min[a]! + e && p <= max[a]! - e)) return false;
  }
  return true;
}

/**
 * Can a view frustum (convex hull of its corner rays) contain any direction
 * in this face's pyramid? False only when one of the pyramid's own bounding
 * planes has every corner ray strictly outside it: a separating plane, so the
 * cones are disjoint. Anything else counts as touching.
 */
export function frustumTouchesFace(frame: DisocclusionFrame, rays: readonly Vec3[]): boolean {
  const planes: ((r: Vec3) => number)[] = [
    r => dot(r, frame.forward),
    r => dot(r, frame.forward) * frame.directionTanX - dot(r, frame.right),
    r => dot(r, frame.forward) * frame.directionTanX + dot(r, frame.right),
    r => dot(r, frame.forward) * frame.directionTanY - dot(r, frame.up),
    r => dot(r, frame.forward) * frame.directionTanY + dot(r, frame.up),
  ];
  return planes.every(plane => rays.some(r => plane(r) >= 0));
}

export function supportsPose(frame: DisocclusionFrame, min: Vec3, max: Vec3, pose: DisocclusionCameraPose): string | null {
  if (!insideSource(min, max, pose.position)) return 'camera outside source box';
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
    // Faces of one source volume are evaluated together; an unnamed capture is its own volume.
    const byVolume = new Map<string, number[]>();
    (sidecar && !this.identityError ? sidecar.meta.domains : []).forEach((d, i) => {
      const key = d.capture.volume ?? `#${i}`;
      byVolume.set(key, [...(byVolume.get(key) ?? []), i]);
    });
    this.groups = [...byVolume.values()];
  }

  private readonly groups: number[][];

  /**
   * Six faces of one identical box whose pyramids cover every direction: side
   * faces lean >= 1 across and v up, and the +-y faces lean >= 1/v both ways
   * (see sourceVolumeCaptures). Anything else is partial coverage.
   */
  private isSphere(group: number[]): boolean {
    const domains = group.map(i => this.sidecar!.meta.domains[i]!.capture);
    if (new Set(domains.map(d => d.axis)).size !== 6 || group.length !== 6) return false;
    const box = JSON.stringify([domains[0]!.sourceMin, domains[0]!.sourceMax]);
    if (!domains.every(d => JSON.stringify([d.sourceMin, d.sourceMax]) === box)) return false;
    const vertical = (i: number) => domains[group.indexOf(i)]!.axis.endsWith('y');
    const sides = group.filter(i => !vertical(i)).map(i => this.frames[i]!);
    const caps = group.filter(vertical).map(i => this.frames[i]!);
    const v = Math.min(...sides.map(f => f.directionTanY));
    return sides.every(f => f.directionTanX >= 1) && caps.every(f => f.directionTanX * v >= 1 - 1e-9 && f.directionTanY * v >= 1 - 1e-9);
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
    const take = (i: number) => {
      const domain = this.sidecar!.meta.domains[i]!;
      used.push(domain.id);
      for (let w = 0; w < this.wordsPerRow; w++) words[w]! |= this.sidecar!.words[domain.wordOffset + w]!;
    };
    for (const group of this.groups) {
      const first = this.sidecar.meta.domains[group[0]!]!.capture;
      if (!insideSource(first.sourceMin, first.sourceMax, pose.position)) {
        lastReason = `${first.volume ?? this.sidecar.meta.domains[group[0]!]!.id}: camera outside source box`;
        continue;
      }
      if (this.isSphere(group)) {
        // Every direction is some face's: union each face the frustum can touch.
        for (const i of group) if (frustumTouchesFace(this.frames[i]!, pose.cornerRays)) take(i);
        continue;
      }
      // Partial coverage: a face applies only if it holds the WHOLE frustum.
      for (const i of group) {
        const domain = this.sidecar.meta.domains[i]!;
        const why = supportsPose(this.frames[i]!, domain.capture.sourceMin, domain.capture.sourceMax, pose);
        if (why) lastReason = `${domain.id}: ${why}`;
        else take(i);
      }
    }
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
