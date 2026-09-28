import { fetchShadoJson } from '../preprocess/runtime';
import type { ShadoWorldSpatialPackage } from './types';
import { validateShadoWorldPackage } from './validation';
import { validateShadoWorldAuthoring } from './authoring';
import type { ShadoWorldAuthoringDocument } from './types';

export async function deserializeShadoWorld(
  url: string,
  options: { fetch?: typeof fetch } = {}
): Promise<ShadoWorldSpatialPackage> {
  const world = await fetchShadoJson<ShadoWorldSpatialPackage>(url, options);
  try {
    validateShadoWorldPackage(world);
  } catch (error) {
    throw new Error(
      `Invalid Shado world spatial artifact at '${url}': ${(error as Error).message}`
    );
  }
  return world;
}

export async function deserializeShadoWorldAuthoring(
  url: string,
  expectedWorld?: string,
  options: { fetch?: typeof fetch } = {}
): Promise<ShadoWorldAuthoringDocument> {
  const document = await fetchShadoJson<ShadoWorldAuthoringDocument>(url, options);
  try {
    return validateShadoWorldAuthoring(document, expectedWorld);
  } catch (error) {
    throw new Error(
      `Invalid Shado world authoring document at '${url}': ${(error as Error).message}`
    );
  }
}

export type ShadoWorldObjectRenderBatch = {
  prototype: number;
  id: string;
  source: string;
  stampIndices: Uint32Array;
  /** Babylon-compatible column-major thin-instance matrices. */
  matrices: Float32Array;
  /** Per-stamp baked irradiance uploaded as Babylon thin-instance colors. */
  colors: Float32Array;
  /**
   * The same stamps split by detail level, nearest first. Always at least one
   * entry; without a {@link ShadoWorldObjectLodSelection} it is the whole set
   * at level 0, which is what every consumer saw before levels existed.
   */
  levels: ShadoWorldObjectRenderLevel[];
};

/** One detail level's share of a prototype's visible stamps. */
export type ShadoWorldObjectRenderLevel = {
  /** 0 is the shipped model; 1 and up are the coarser meshes beside it. */
  level: number;
  stampIndices: Uint32Array;
  matrices: Float32Array;
  colors: Float32Array;
};

/**
 * How to choose a level for each stamp.
 *
 * Selection is by PROJECTED SIZE, not distance, and the difference matters: a
 * zone stamps a cobble tile and a keep from the same layer, and a distance that
 * is far for one is intimate for the other. Screen height also makes the choice
 * resolution- and field-of-view independent for free, and it is the metric the
 * visibility reduction already culls on, so a stamp cannot be culled for being
 * small while being drawn at full detail for being close.
 */
export type ShadoWorldObjectLodSelection = {
  /** Camera position, world space. */
  camera: readonly [number, number, number];
  /**
   * Pixels covered by one world unit of radius at one unit of distance:
   * `(renderHeight / 2) / tan(fov / 2)` for a vertically-fixed field of view.
   */
  pixelsPerRadius: number;
  /**
   * Projected screen height at or below which a stamp drops to the next level.
   * `thresholds[0]` is the 0 -> 1 boundary, `[1]` the 1 -> 2, and so on;
   * descending, and a stamp never drops past what its prototype ships.
   */
  thresholds: readonly number[];
  /** How many COARSER levels this prototype has. 0 means model only. */
  levelsFor: (prototype: number) => number;
  /** Byte per stamp: 1 keeps that stamp at level 0 whatever its size. */
  pinned?: Uint8Array | null;
};

/**
 * Converts visible per-prototype stamp rows into upload-ready matrix buffers.
 * These are final Babylon-space matrices: consumers upload them verbatim and
 * must not apply legacy EQ/Requiem axis swaps or yaw inversion. Asset loading
 * remains a client policy; the package supplies the stable URL.
 */
export function buildShadoWorldObjectRenderBatches(
  world: ShadoWorldSpatialPackage,
  visibleByPrototype?: readonly ArrayLike<number>[],
  lod?: ShadoWorldObjectLodSelection
): ShadoWorldObjectRenderBatch[] {
  const objects = world.objects;
  if (!objects) return [];
  return objects.prototypes.id.map((id, prototype) => {
    const first = objects.prototypes.firstStampRef[prototype];
    const count = objects.prototypes.stampRefCount[prototype];
    const sourceRows =
      visibleByPrototype?.[prototype] ??
      objects.prototypeStampRefs
        .slice(first, first + count)
        .filter(stamp => objects.stamps.enabled[stamp]);
    const stampIndices = Uint32Array.from(sourceRows);
    const matrices = new Float32Array(stampIndices.length * 16);
    const colors = new Float32Array(stampIndices.length * 4);
    const { irradianceR, irradianceG, irradianceB, irradianceA } = objects.stamps;
    for (let index = 0; index < stampIndices.length; index++) {
      const stamp = stampIndices[index];
      writeStampMatrix(objects.stamps, stamp, matrices, index * 16);
      const at = index * 4;
      colors[at] = irradianceR?.[stamp] ?? 1;
      colors[at + 1] = irradianceG?.[stamp] ?? 1;
      colors[at + 2] = irradianceB?.[stamp] ?? 1;
      colors[at + 3] = irradianceA?.[stamp] ?? 1;
    }
    return {
      prototype,
      id,
      source: objects.prototypes.source[prototype],
      stampIndices,
      matrices,
      colors,
      levels: splitByLevel(objects.stamps, stampIndices, matrices, colors, prototype, lod),
    };
  });
}

/**
 * Deal one prototype's visible stamps into its levels.
 *
 * The matrices are re-used rather than recomputed: they were already written
 * above, and a level is a permutation of rows out of that buffer. A prototype
 * whose stamps all land on one level returns a single entry sharing the
 * original arrays, so the common case allocates nothing.
 */
function splitByLevel(
  stamps: NonNullable<ShadoWorldSpatialPackage['objects']>['stamps'],
  stampIndices: Uint32Array,
  matrices: Float32Array,
  colors: Float32Array,
  prototype: number,
  lod?: ShadoWorldObjectLodSelection
): ShadoWorldObjectRenderLevel[] {
  const whole = [{ level: 0, stampIndices, matrices, colors }];
  if (!lod || stampIndices.length === 0) return whole;
  const deepest = Math.min(lod.levelsFor(prototype), lod.thresholds.length);
  if (deepest <= 0) return whole;

  const [cx, cy, cz] = lod.camera;
  const levelOf = new Uint8Array(stampIndices.length);
  let highest = 0;
  stampIndices.forEach((stamp, index) => {
    const dx = stamps.positionX[stamp] - cx;
    const dy = stamps.positionY[stamp] - cy;
    const dz = stamps.positionZ[stamp] - cz;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    // A stamp at or inside the camera is as large as it can be, not infinite.
    const pixels =
      distance > 1e-3
        ? (stamps.radius[stamp] * lod.pixelsPerRadius) / distance
        : Number.POSITIVE_INFINITY;
    let level = 0;
    // A pinned stamp stays at full detail (a blocker the PVS rows rely on).
    if (!lod.pinned?.[stamp]) while (level < deepest && pixels <= lod.thresholds[level]!) level += 1;
    levelOf[index] = level;
    if (level > highest) highest = level;
  });
  if (highest === 0) return whole;

  const out: ShadoWorldObjectRenderLevel[] = [];
  for (let level = 0; level <= highest; level += 1) {
    const rows: number[] = [];
    for (let index = 0; index < levelOf.length; index += 1) {
      if (levelOf[index] === level) rows.push(index);
    }
    if (!rows.length) continue;
    const levelMatrices = new Float32Array(rows.length * 16);
    const levelColors = new Float32Array(rows.length * 4);
    const levelStamps = new Uint32Array(rows.length);
    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index]!;
      // Element copies: two subarray views per stamp were garbage every pass.
      for (let k = 0; k < 16; k += 1) levelMatrices[index * 16 + k] = matrices[row * 16 + k]!;
      for (let k = 0; k < 4; k += 1) levelColors[index * 4 + k] = colors[row * 4 + k]!;
      levelStamps[index] = stampIndices[row]!;
    }
    out.push({ level, stampIndices: levelStamps, matrices: levelMatrices, colors: levelColors });
  }
  return out;
}

const QUATERNION_SCRATCH = new Float32Array(4);

/**
 * Converts an authored stamp's Y-X-Z Euler degrees into a quaternion.
 *
 * Exported because the Euler convention is a package contract: any renderer
 * that places stamps without going through {@link buildShadoWorldObjectRenderBatches}
 * must resolve rotation identically or its objects face the wrong way.
 */
export function shadoWorldStampQuaternion(
  stamps: NonNullable<ShadoWorldSpatialPackage['objects']>['stamps'],
  stamp: number,
  out: Float32Array = new Float32Array(4)
): Float32Array {
  const radians = Math.PI / 180;
  const halfRoll = stamps.rotationZ[stamp] * radians * 0.5;
  const halfPitch = stamps.rotationX[stamp] * radians * 0.5;
  const halfYaw = stamps.rotationY[stamp] * radians * 0.5;
  const sinRoll = Math.sin(halfRoll),
    cosRoll = Math.cos(halfRoll);
  const sinPitch = Math.sin(halfPitch),
    cosPitch = Math.cos(halfPitch);
  const sinYaw = Math.sin(halfYaw),
    cosYaw = Math.cos(halfYaw);
  out[0] = cosYaw * sinPitch * cosRoll + sinYaw * cosPitch * sinRoll;
  out[1] = sinYaw * cosPitch * cosRoll - cosYaw * sinPitch * sinRoll;
  out[2] = cosYaw * cosPitch * sinRoll - sinYaw * sinPitch * cosRoll;
  out[3] = cosYaw * cosPitch * cosRoll + sinYaw * sinPitch * sinRoll;
  return out;
}

function writeStampMatrix(
  stamps: NonNullable<ShadoWorldSpatialPackage['objects']>['stamps'],
  stamp: number,
  target: Float32Array,
  offset: number
): void {
  // Indexed, not destructured: destructuring a typed array runs its iterator,
  // an allocation per stamp per pass.
  const q = shadoWorldStampQuaternion(stamps, stamp, QUATERNION_SCRATCH);
  const x = q[0]!,
    y = q[1]!,
    z = q[2]!,
    w = q[3]!;
  const x2 = x + x,
    y2 = y + y,
    z2 = z + z;
  const xx = x * x2,
    xy = x * y2,
    xz = x * z2;
  const yy = y * y2,
    yz = y * z2,
    zz = z * z2;
  const wx = w * x2,
    wy = w * y2,
    wz = w * z2;
  const sx = stamps.scaleX[stamp],
    sy = stamps.scaleY[stamp],
    sz = stamps.scaleZ[stamp];
  // Element writes, not `target.set([...])`: this runs for every visible
  // stamp on every visibility pass, and the sixteen-number literal it used to
  // build was the largest single source of garbage in a running zone.
  target[offset] = (1 - (yy + zz)) * sx;
  target[offset + 1] = (xy + wz) * sx;
  target[offset + 2] = (xz - wy) * sx;
  target[offset + 3] = 0;
  target[offset + 4] = (xy - wz) * sy;
  target[offset + 5] = (1 - (xx + zz)) * sy;
  target[offset + 6] = (yz + wx) * sy;
  target[offset + 7] = 0;
  target[offset + 8] = (xz + wy) * sz;
  target[offset + 9] = (yz - wx) * sz;
  target[offset + 10] = (1 - (xx + yy)) * sz;
  target[offset + 11] = 0;
  target[offset + 12] = stamps.positionX[stamp];
  target[offset + 13] = stamps.positionY[stamp];
  target[offset + 14] = stamps.positionZ[stamp];
  target[offset + 15] = 1;
}
