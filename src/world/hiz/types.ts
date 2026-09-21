/**
 * Runtime Hi-Z contracts (docs/pvs-hiz-prototype.md, H0).
 *
 * Everything here is per view and per render pass. A Hi-Z result answers
 * "is this bound hidden behind the stable opaque occluders *for this camera,
 * this frame*"; it never becomes scene enable state, so shadows, reflections
 * and gameplay keep their own visibility.
 */

/**
 * `normal`: 0 at the near plane, 1 at the far plane, clear = 1, pyramid keeps
 * the MAXIMUM of its children. `reversed`: the opposite, pyramid keeps the
 * MINIMUM. Never silently reinterpret one as the other.
 */
export type ShadoHiZDepthConvention = 'normal' | 'reversed';

export interface ShadoHiZViewInput {
  /** Engine frame the depth was rasterized in. Results never cross frames. */
  readonly frameId: number;
  /** Column-major clip-from-world matrix (Babylon `Matrix.m` layout). */
  readonly viewProjection: ArrayLike<number>;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly convention: ShadoHiZDepthConvention;
  /**
   * WebGPU/Babylon-on-WebGPU projects z into [0, 1]; WebGL into [-1, 1].
   * Depth is `z` or `z * 0.5 + 0.5` accordingly.
   */
  readonly ndcHalfZRange: boolean;
  /** Pixel row 0 is the top of the image (WebGPU) rather than the bottom. */
  readonly topLeftOrigin: boolean;
  /** Bumped whenever the world/candidate layout changes. */
  readonly worldEpoch: number;
  /** Bumped whenever any occluder moves, streams in/out or changes material. */
  readonly opaqueEpoch: number;
}

/** One testable bound. `batch`/`member` address the compacted output. */
export interface ShadoHiZCandidate {
  /** Stable id supplied by the caller (chunk index, instance id...). */
  readonly id: number;
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly batch: number;
  /** Index inside the batch's own instance space (0 for a plain mesh). */
  readonly member: number;
}

/** One indirect draw (a mesh submesh, or an instanced prototype). */
export interface ShadoHiZBatch {
  readonly indexCount: number;
  readonly firstIndex: number;
  /** Candidates addressed to this batch; its visible-id segment size. */
  readonly capacity: number;
  /**
   * > 0: draw all-or-nothing with this many instances (a thin-instanced mesh
   * tested by its whole bound). The count can change per frame through
   * `ShadoWorldHiZ.setWholeInstances`.
   */
  readonly wholeInstances?: number;
}

/** Why a candidate is kept without a depth comparison. */
export enum ShadoHiZAdmitReason {
  None = 0,
  /** A corner is behind the eye or in front of the near plane. */
  NearPlane = 1,
  NonFinite = 2,
  /** Wholly off-screen: the frustum stage owns that decision, not Hi-Z. */
  OffScreen = 3,
}

export interface ShadoHiZProjection {
  readonly admit: ShadoHiZAdmitReason;
  /** Inclusive pixel rectangle, clamped to the viewport, already expanded. */
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
  /** Nearest possible depth of the bound in the view's convention. */
  readonly nearest: number;
}

/**
 * The texel budget per axis at the chosen level. The test visits every
 * intersected texel, so a rectangle spanning N texels costs at most N*N loads.
 */
export const SHADO_HIZ_MAX_TEXELS_PER_AXIS = 4;
/** Rectangle expansion in level-0 pixels for raster/numerical uncertainty. */
export const SHADO_HIZ_PIXEL_EXPANSION = 1;
/**
 * Positive depth slack. A candidate is rejected only when its nearest depth is
 * farther than every covered occluder depth by more than this. Chosen well
 * above float32 interpolation error near z = 1 (~6e-8) so an occluder cannot
 * hide a surface it is coplanar with.
 */
export const SHADO_HIZ_DEPTH_BIAS = 1e-5;
/** Levels the GPU parameter block can describe; 2^15 px covers any viewport. */
export const SHADO_HIZ_MAX_LEVELS = 16;
