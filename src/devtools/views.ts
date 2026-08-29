/**
 * Fixed review-camera vocabulary, kept byte-compatible with `@eltania/glb-render`.
 *
 * These angles are a *convention*, not a preference: authored content is built
 * to face a particular way relative to them, and Libra's review renders are read
 * as evidence. Changing an angle silently re-frames every historical comparison,
 * so the table is copied verbatim from the previous renderer rather than
 * re-derived.
 */

/** Babylon ArcRotate (alpha, beta) radians. */
export const NAMED_VIEWS: Readonly<Record<string, readonly [number, number]>> = {
  left: [0, Math.PI / 2],
  front: [-Math.PI / 2, Math.PI / 2],
  back: [Math.PI / 2, Math.PI / 2],
  right: [Math.PI, Math.PI / 2],
  iso: [-Math.PI / 4, Math.PI / 3],
  isoFrontLeft: [-Math.PI / 4, Math.PI / 3],
  isoFrontRight: [(-3 * Math.PI) / 4, Math.PI / 3],
  isoBackRight: [(3 * Math.PI) / 4, Math.PI / 3],
  isoBackLeft: [Math.PI / 4, Math.PI / 3],
  eyeFront: [-Math.PI / 2, Math.PI / 2.25],
  eyeCorner: [-Math.PI / 4, Math.PI / 2.3],
  top: [-Math.PI / 2, 0.001],
};

/** The default four-view set. */
export const MULTIVIEWS = ['left', 'front', 'back', 'right'] as const;

/** `raised` tilts every view except `top` down to this elevation. */
export const RAISED_BETA = Math.PI / 3;

export function viewAngles(view: string, raised = false): readonly [number, number] {
  const angle = NAMED_VIEWS[view];
  if (!angle) throw new Error(`Unknown review view '${view}'; expected one of ${Object.keys(NAMED_VIEWS).join(', ')}`);
  return raised && view !== 'top' ? [angle[0], RAISED_BETA] : angle;
}

/**
 * Fraction of pixels that are not the background — the framing sanity check the
 * previous renderer reported, so a subject that fell out of frame is caught.
 */
export function coverage(pixels: Uint8Array, background: readonly [number, number, number]): number {
  const target = background.map((channel) => Math.round(channel * 255));
  let covered = 0;
  const total = pixels.length / 4;
  for (let i = 0; i < pixels.length; i += 4) {
    if (Math.abs(pixels[i]! - target[0]!) > 6 || Math.abs(pixels[i + 1]! - target[1]!) > 6 || Math.abs(pixels[i + 2]! - target[2]!) > 6) {
      covered++;
    }
  }
  return total ? covered / total : 0;
}
