import { buildOccluderGrid, highestSurfaceAt, segmentBlocked } from './occlusion';
import type {
  ShadoWorldBounds,
  ShadoWorldPrimitive,
  ShadoWorldSpatialPackage,
  ShadoWorldVisibilityBakeReport,
  ShadoWorldVisibilityMode,
} from './types';

type Point2 = [number, number];

// PVS is allowed to overdraw, but it must never create a visible hole around
// the player. Two grid steps cover a 5x5 local neighborhood; the camera-row
// margin makes adjacent rows overlap before the player crosses their boundary.
//
// These are UNCONDITIONAL admissions, not candidates: a local pair never
// reaches an occlusion test. The camera can walk, and the row it stands on has
// to hold for every point the player reaches before entering the next region,
// so a wall that hides a neighbour from this region's samples proves nothing
// about its far corner.
const LOCAL_FLOOD_RADIUS = 2;
const CAMERA_ROW_MARGIN = 1;

/**
 * Occlusion sampling, and why each number errs the way it does.
 *
 * A from-region PVS has to hold for EVERY point in the region, not the one it
 * was measured from. So a region is only marked hidden when every sample of a
 * viewer inside it fails to see every sample of the thing being looked at, and
 * the samples are deliberately generous in both directions.
 *
 * `EYE_HEIGHTS` are above whatever surface the column has, so a viewpoint is
 * where a player could stand rather than at an arbitrary Y. `TARGET_HEIGHTS`
 * climb far above it because what stands in a region is not necessarily in the
 * occluder set at all: stamped objects stream their own collision, so a region
 * holding nothing but a 40 m tree has no geometry here to sample and must not
 * be judged by the ground alone.
 *
 * `FOOTPRINT_INSET` keeps corner samples off the region boundary, where a
 * coincident wall would make them ambiguous.
 */
const EYE_HEIGHTS = [8] as const;
/** Used only when the caller supplies no cell bounds to sample instead. */
const TARGET_HEIGHTS = [2, 40, 120, 240] as const;
/**
 * Headroom above a region's known geometry for things that are not in it.
 *
 * Stamped objects stream their own collision and are not in the occluder set,
 * so a region whose cells top out at a wall's height may still hold a tree
 * standing over it. Twenty units is a guess, short of the 240 that made the
 * bake a no-op and not a bound anything authored actually guarantees.
 */
const STAMP_HEADROOM = 20;
const FOOTPRINT_INSET = 0.2;
const OCCLUDER_CELL_SIZE = 32;

export type ShadoWorldVisibilityCompileInput = {
  /**
   * Which authority fills the rows. Defaults to `distance-flood`: the shipped
   * behaviour, and the baseline every comparison is measured against.
   *
   * `sampled-occlusion` is EXPERIMENTAL and must be asked for explicitly. It
   * rejects a pair when no sampled viewpoint reaches any sampled target, which
   * is evidence, not proof -- the unsampled space between those points is not
   * covered by it. Do not enable it for a promotion that ships.
   */
  mode?: ShadoWorldVisibilityMode;
  bounds: ShadoWorldBounds;
  regionSize: number;
  maxDistance: number;
  renderCellCenters: readonly Point2[];
  /**
   * Vertical extent of each render cell, if the caller has it.
   *
   * Without this a target region has to be sampled over a guessed column, and
   * the guess has to be tall enough for anything that might stand there -- at
   * which point almost every pair sees almost every other over the rooftops
   * and the bake culls nothing. With it, a region is sampled over what it
   * actually contains.
   */
  renderCellBounds?: readonly ShadoWorldBounds[];
  persistentRenderCells: ArrayLike<number>;
  collisionPrimitives: readonly ShadoWorldPrimitive[];
  /** Optional bake diagnostics; the numbers that say whether it is working. */
  report?: (stats: ShadoWorldVisibilityBakeReport) => void;
};


/**
 * Builds continuous camera/entity regions and a conservative sampled PVS.
 *
 * Regions are deliberately independent of sparse render geometry. Until zones
 * provide authored room/portal topology or height-aware occluders, outdoor PVS
 * is a conservative distance flood. Heightless 2D wall rays cannot prove that
 * an entire vertical region is hidden and caused visible skyline holes.
 */
export function compileShadoWorldVisibility(
  input: ShadoWorldVisibilityCompileInput
): NonNullable<ShadoWorldSpatialPackage['visibility']> {
  const size = input.regionSize;
  const maxDistance = input.maxDistance;
  const originX = Math.floor(input.bounds.min[0] / size) * size;
  const originZ = Math.floor(input.bounds.min[2] / size) * size;
  const width = Math.max(1, Math.ceil((input.bounds.max[0] - originX) / size));
  const height = Math.max(1, Math.ceil((input.bounds.max[2] - originZ) / size));
  const regionCount = width * height;
  const regionForPoint = (x: number, z: number): number => {
    const localX = Math.floor((x - originX) / size);
    const localZ = Math.floor((z - originZ) / size);
    if (localX < 0 || localX >= width || localZ < 0 || localZ >= height) return -1;
    return localZ * width + localX;
  };
  const cellRegion = input.renderCellCenters.map(([x, z]) => regionForPoint(x, z));
  const persistent = new Uint8Array(regionCount);
  cellRegion.forEach((region, cell) => {
    if (region >= 0 && Number(input.persistentRenderCells[cell] ?? 0) !== 0) {
      persistent[region] = 1;
    }
  });
  const persistentRegions = Array.from(persistent, (value, region) => value ? region : -1)
    .filter(region => region >= 0);
  const persistentCells = Array.from(
    input.persistentRenderCells,
    (value, cell) => Number(value) !== 0 ? cell : -1
  ).filter(cell => cell >= 0);

  const wordsPerRow = Math.ceil(regionCount / 32);
  const words = new Uint32Array(regionCount * wordsPerRow);
  let visibleRegionPairs = 0;
  const setVisible = (from: number, to: number) => {
    const index = from * wordsPerRow + (to >>> 5);
    const mask = 1 << (to & 31);
    if (!(words[index] & mask)) {
      words[index] = (words[index] | mask) >>> 0;
      visibleRegionPairs++;
    }
  };
  /*
   * Sample points per region, built once.
   *
   * A region with no surface anywhere under it has nothing to stand on and
   * nothing to sample, and is left to the flood -- admitted. That is the safe
   * reading: an empty region that gets culled is exactly the hole this must
   * never make, and admitting it costs nothing to draw.
   */
  const requestedMode: ShadoWorldVisibilityMode = input.mode ?? 'distance-flood';
  /*
   * A zone with nothing eligible to occlude with cannot be occlusion-tested.
   * Falling back to the flood is the only safe answer: every row would come
   * back fully visible anyway, and labelling that `sampled-occlusion` would
   * advertise a test that never ran (and fail validation, which requires a
   * nonzero occluder count for that mode).
   */
  const grid = requestedMode === 'sampled-occlusion'
    ? buildOccluderGrid(input.collisionPrimitives, input.bounds, OCCLUDER_CELL_SIZE)
    : null;
  const sampled = grid !== null && grid.triangleCount > 0;
  const mode: ShadoWorldVisibilityMode = sampled ? 'sampled-occlusion' : 'distance-flood';
  const eyes: (Float64Array | null)[] = new Array(regionCount).fill(null);
  const targets: (Float64Array | null)[] = new Array(regionCount).fill(null);
  const footprint = (region: number): [number, number][] => {
    const rx = region % width, rz = Math.floor(region / width);
    const x0 = originX + (rx + FOOTPRINT_INSET) * size;
    const x1 = originX + (rx + 1 - FOOTPRINT_INSET) * size;
    const z0 = originZ + (rz + FOOTPRINT_INSET) * size;
    const z1 = originZ + (rz + 1 - FOOTPRINT_INSET) * size;
    const cx = originX + (rx + 0.5) * size, cz = originZ + (rz + 0.5) * size;
    return [[cx, cz], [x0, z0], [x1, z0], [x0, z1], [x1, z1]];
  };
  /*
   * What each region actually contains, vertically, taken from its cells.
   * A region with no cells draws nothing, but it is left samplable anyway:
   * stamps live in regions too and are not represented here.
   */
  const regionLow = new Float64Array(regionCount).fill(Number.POSITIVE_INFINITY);
  const regionHigh = new Float64Array(regionCount).fill(Number.NEGATIVE_INFINITY);
  if (input.renderCellBounds) {
    input.renderCellBounds.forEach((cell, index) => {
      const region = cellRegion[index];
      if (region === undefined || region < 0) return;
      if (cell.min[1] < regionLow[region]!) regionLow[region] = cell.min[1];
      if (cell.max[1] > regionHigh[region]!) regionHigh[region] = cell.max[1];
    });
  }

  for (let region = 0; sampled && region < regionCount; region++) {
    const eyePoints: number[] = [];
    const targetPoints: number[] = [];
    const low = regionLow[region]!;
    const high = regionHigh[region]!;
    const known = Number.isFinite(low) && Number.isFinite(high);
    for (const [x, z] of footprint(region)) {
      const floor = highestSurfaceAt(grid!, x, z, input.bounds);
      if (floor === null) continue;
      for (const eye of EYE_HEIGHTS) eyePoints.push(x, floor + eye, z);
      if (known) {
        // Bottom, middle and top of what is there, plus headroom for a stamp
        // standing in the same region that this compiler cannot see.
        const top = Math.max(high, floor) + STAMP_HEADROOM;
        const bottom = Math.min(low, floor) + 2;
        for (let step = 0; step <= 3; step++) {
          targetPoints.push(x, bottom + ((top - bottom) * step) / 3, z);
        }
      } else {
        for (const height of TARGET_HEIGHTS) targetPoints.push(x, floor + height, z);
      }
    }
    if (eyePoints.length) eyes[region] = Float64Array.from(eyePoints);
    if (targetPoints.length) targets[region] = Float64Array.from(targetPoints);
  }

  let occlusionTested = 0;
  let occluded = 0;
  let forcedLocalPairs = 0;
  const geometry = { originX, originZ, size, width, height, maxDistance };
  for (let from = 0; from < regionCount; from++) {
    for (let to = from; to < regionCount; to++) {
      const local = isLocalPair(from, to, geometry);
      if (!local && !withinRange(from, to, geometry)) continue;
      if (local) {
        forcedLocalPairs++;
      } else if (sampled) {
        /*
         * Symmetric by construction: the pair is tested once and set both
         * ways. Seeing is mutual for a straight segment, and testing each
         * direction separately would let sampling noise produce a row that
         * disagrees with its own transpose -- which shows up as a region that
         * pops in from one approach and not the other.
         */
        const source = eyes[from];
        const target = targets[to];
        const back = eyes[to];
        const forward = targets[from];
        if (source && target && back && forward) {
          occlusionTested++;
          if (
            !anyClearSegment(grid!, source, target) &&
            !anyClearSegment(grid!, back, forward)
          ) {
            occluded++;
            continue;
          }
        }
      }
      setVisible(from, to);
      if (from !== to) setVisible(to, from);
    }
  }
  const rawPairs = visibleRegionPairs;
  const floodedWords = floodCameraRows(
    words,
    wordsPerRow,
    width,
    height,
    CAMERA_ROW_MARGIN
  );
  visibleRegionPairs = countVisibleBits(floodedWords);
  if (input.report) {
    input.report({
      requestedMode,
      mode,
      fallbackReason:
        requestedMode === 'sampled-occlusion' && !sampled
          ? 'no-eligible-occluders'
          : null,
      occluderTriangles: sampled ? grid!.triangleCount : 0,
      forcedLocalPairs,
      occlusionTested,
      occluded,
      pairsBeforeRowFlood: rawPairs,
      pairsAfterRowFlood: visibleRegionPairs,
    });
  }
  return {
    version: 1,
    mode,
    size,
    originX,
    originZ,
    width,
    height,
    maxDistance,
    occluderCount: sampled ? grid!.triangleCount : 0,
    visibleRegionPairs,
    cellRegion,
    persistentRegions,
    persistentCells,
    pvs: { wordsPerRow, words: Array.from(floodedWords) },
  };
}

/**
 * Can any viewpoint reach any target point without crossing geometry?
 *
 * Targets are walked from the TOP down: the highest sample is a roofline or a
 * treetop, which is by far the likeliest thing to be visible over an
 * intervening wall, so the common answer arrives on the first or second test
 * and the expensive all-blocked case is the rare one.
 */
function anyClearSegment(
  grid: ReturnType<typeof buildOccluderGrid>,
  eyes: Float64Array,
  targets: Float64Array
): boolean {
  for (let t = targets.length - 3; t >= 0; t -= 3) {
    for (let e = 0; e < eyes.length; e += 3) {
      if (!segmentBlocked(
        grid,
        eyes[e]!, eyes[e + 1]!, eyes[e + 2]!,
        targets[t]!, targets[t + 1]!, targets[t + 2]!
      )) return true;
    }
  }
  return false;
}

type RegionGeometry = {
  originX: number;
  originZ: number;
  size: number;
  width: number;
  height: number;
  maxDistance: number;
};

/**
 * Is this pair inside the local safety neighborhood?
 *
 * A true here is an admission, not a candidacy: the caller must set the bit
 * without consulting any occluder. Distance and occlusion are separate
 * questions and neither may override this one.
 */
function isLocalPair(from: number, to: number, grid: RegionGeometry): boolean {
  if (from === to) return true;
  const fromX = from % grid.width, fromZ = Math.floor(from / grid.width);
  const toX = to % grid.width, toZ = Math.floor(to / grid.width);
  return (
    Math.max(Math.abs(fromX - toX), Math.abs(fromZ - toZ)) <= LOCAL_FLOOD_RADIUS
  );
}

/** The distance envelope, which is not an occlusion result and never reported as one. */
function withinRange(from: number, to: number, grid: RegionGeometry): boolean {
  const fromX = from % grid.width, fromZ = Math.floor(from / grid.width);
  const toX = to % grid.width, toZ = Math.floor(to / grid.width);
  const center = (x: number, z: number): Point2 => [
    grid.originX + (x + 0.5) * grid.size,
    grid.originZ + (z + 0.5) * grid.size,
  ];
  const source = center(fromX, fromZ);
  const target = center(toX, toZ);
  return Math.hypot(target[0] - source[0], target[1] - source[1]) <= grid.maxDistance;
}

/**
 * Unions each camera row with neighboring camera rows. This is a conservative
 * source-region flood: adjacent player positions share their visibility before
 * the player crosses a region edge, eliminating hard row-transition popping.
 */
function floodCameraRows(
  source: Uint32Array,
  wordsPerRow: number,
  width: number,
  height: number,
  radius: number
): Uint32Array {
  const result = source.slice();
  for (let fromZ = 0; fromZ < height; fromZ++) {
    for (let fromX = 0; fromX < width; fromX++) {
      const targetRow = (fromZ * width + fromX) * wordsPerRow;
      for (let deltaZ = -radius; deltaZ <= radius; deltaZ++) {
        const sourceZ = fromZ + deltaZ;
        if (sourceZ < 0 || sourceZ >= height) continue;
        for (let deltaX = -radius; deltaX <= radius; deltaX++) {
          const sourceX = fromX + deltaX;
          if (sourceX < 0 || sourceX >= width) continue;
          const sourceRow = (sourceZ * width + sourceX) * wordsPerRow;
          for (let word = 0; word < wordsPerRow; word++) {
            result[targetRow + word] =
              (result[targetRow + word] | source[sourceRow + word]) >>> 0;
          }
        }
      }
    }
  }
  return result;
}

function countVisibleBits(words: Uint32Array): number {
  let total = 0;
  for (const value of words) {
    let remaining = value >>> 0;
    while (remaining) {
      remaining &= remaining - 1;
      total++;
    }
  }
  return total;
}
