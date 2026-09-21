/**
 * Offline disocclusion PVS prototype: shared types.
 *
 * Adapted from Künzel et al., "Potentially Visible Set Generation with the
 * Disocclusion Buffer", SIGGRAPH Asia 2025 (doi:10.1145/3757377.3763981), and
 * the authors' MIT-licensed reference source (DaRUS doi:10.18419/DARUS-5385).
 * See docs/pvs-disocclusion-evidence.md for the stage mapping and deviations.
 *
 * EXPERIMENTAL. Nothing here is a shipped visibility authority.
 */

export type Vec3 = [number, number, number];

/**
 * One directional capture: a camera plane behind a source box, looking along
 * `axis`, with a lateral viewcell and an extended field of view that covers
 * every supported ray. All distances are zone units (3 units per metre).
 */
export type DisocclusionCapture = {
  /** Source box the camera may occupy. */
  sourceMin: Vec3;
  sourceMax: Vec3;
  /** Unit axis the capture looks along; one of the six signed world axes. */
  axis: DisocclusionAxis;
  /**
   * Largest |lateral / axial| direction component a supported view ray may
   * have, per lateral axis. A camera is supported only when its whole view
   * frustum stays inside this pyramid.
   */
  directionTan: number;
  /** Same bound for the vertical (`up`) axis; defaults to `directionTan`. */
  directionTanUp?: number;
  /** Near plane distance from the camera plane. Nearer geometry is always admitted. */
  near: number;
  /** Far plane distance. Farther geometry is unknown, therefore admitted. */
  far: number;
};

export type DisocclusionAxis = '+x' | '-x' | '+y' | '-y' | '+z' | '-z';

export type DisocclusionSettings = {
  /** Square raster resolution per direction. */
  resolution: number;
  /** Tile edge in samples. */
  tileSize: number;
  /** Depth layers; at most 32 so one u32 mask holds a tile's bits. */
  layers: number;
  /** Volumetric filter grid cell edge, zone units; 0 disables the filter. */
  filterCell: number;
};

export const DISOCCLUSION_DEFAULT_SETTINGS: DisocclusionSettings = {
  resolution: 128,
  tileSize: 8,
  layers: 32,
  filterCell: 1.5,
};

/**
 * Derived capture geometry: everything the raster, the propagation and the
 * target classification must agree on.
 */
export type DisocclusionFrame = {
  /** Camera origin: centre of the source box. */
  origin: Vec3;
  /** Forward, right and up unit vectors (right/up are the lateral axes). */
  forward: Vec3;
  right: Vec3;
  up: Vec3;
  /** Viewcell half-extent at the camera plane along right / up. */
  viewcellHalfX: number;
  viewcellHalfY: number;
  /** Raster covers tan in [-extTanX, extTanX] x [-extTanY, extTanY]. */
  extTanX: number;
  extTanY: number;
  near: number;
  far: number;
  /** Axial half depth of the source box about the camera plane. */
  sourceHalfDepth: number;
  /** Lateral half extents of the source box (right, up). */
  sourceHalfRight: number;
  sourceHalfUp: number;
  directionTanX: number;
  directionTanY: number;
};

/** Flat triangle soup handed to the baker. Triangle i is primitive ID i. */
export type DisocclusionGeometry = {
  positions: Float32Array;
  indices: Uint32Array;
  /** Per triangle: owning target (render cluster) index, or -1 for none. */
  triangleTarget: Int32Array;
};

/** Raw buffers after the raster passes, row-major [layer][y][x]. */
export type DisocclusionLayers = {
  settings: DisocclusionSettings;
  /** Ordered depth key per sample and layer; 0xffffffff = empty. */
  depth: Uint32Array;
  /** Winning primitive ID per sample and layer; 0xffffffff = empty. */
  id: Uint32Array;
};

/** Output of the tile stages. Dense, indexed [layer][tileY][tileX]. */
export type DisocclusionMasks = {
  tilesX: number;
  tilesY: number;
  layers: number;
  /** Samples written per tile. */
  count: Uint32Array;
  /** 0 empty, 1 open (partial), 2 closed (full). */
  state: Uint8Array;
  /** Disocclusion bits accumulated per tile from non-degenerate frusta. */
  mask: Uint32Array;
  /** Per tile column [tileY][tileX]: bits from degenerate (column-only) frusta. */
  column: Uint32Array;
  /** 1 when every earlier layer's bit is set, i.e. the cell is potentially visible. */
  visible: Uint8Array;
};

export type DisocclusionStageTimings = Record<string, number>;
