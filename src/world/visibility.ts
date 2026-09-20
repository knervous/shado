import {
  buildOccluderGrid,
  columnSurfaces,
  estimateGridPayloadBytes,
  highestSurfaceAt,
  segmentBlocked,
} from './occlusion';
import { buildInstancedOccluders, instancedSegmentBlocked } from './occluder-instances';
import type { InstancedOccluders } from './occluder-instances';
import {
  ShadoOccluderBackendError,
  buildOccluderBvh,
  bvhColumnSurfaces,
  bvhHighestSurfaceAt,
  bvhSegmentBlocked,
  estimateBvhBytes,
  hasBoundedLimits,
} from './occluder-bvh';
import type { OccluderBuildLimits, OccluderBuildStop, OccluderBvh } from './occluder-bvh';
import type { OccluderGrid } from './occlusion';
import type {
  ShadoWorldBounds,
  ShadoWorldPrimitive,
  ShadoWorldSpatialPackage,
  ShadoWorldVisibilityBakeReport,
  ShadoWorldVisibilityBudget,
  ShadoWorldVisibilityMode,
} from './types';

/**
 * Which index answers the segment queries.
 *
 * `bvh` is the default because the grid's cost grows with world density: on
 * Crownward's assembled visual scene it tested 640 triangles per cell visited
 * and 5,337 per query. `grid` stays selectable so the two remain
 * differentially testable and a regression in either is visible rather than
 * theoretical.
 */
export type ShadoWorldOccluderIndex = 'bvh' | 'grid' | 'auto';

/** Segment queries across both structures, for the budget and the report. */
function segmentQueryCount(
  grid: Occluders | null,
  placed: InstancedOccluders | null
): number {
  return (grid?.counters.segmentQueries ?? 0) + (placed?.counters.segmentQueries ?? 0);
}

/** One question, two structures: can this segment reach that point? */
type Occluders = {
  readonly kind: 'bvh' | 'grid';
  /** Set when construction refused or was interrupted; the index is empty. */
  readonly aborted?: OccluderBuildStop | null;
  /** Bytes the structure's own buffers hold, as distinct from process RSS. */
  readonly trackedBytes: number;
  readonly triangleCount: number;
  readonly blocked: (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number
  ) => boolean;
  readonly floorAt: (x: number, z: number) => number | null;
  /** Every surface under a column, highest first. */
  readonly floorsAt: (x: number, z: number) => number[];
  readonly counters: {
    segmentQueries: number;
    blockedQueries: number;
    columnQueries: number;
    triangleTests: number;
  };
  readonly nodeVisits: () => number;
  readonly references: number;
};

function buildOccluders(
  kind: 'bvh' | 'grid',
  primitives: readonly ShadoWorldPrimitive[],
  bounds: ShadoWorldBounds,
  cellSize: number,
  limits: OccluderBuildLimits = {}
): Occluders {
  if (kind === 'grid') {
    const grid: OccluderGrid = buildOccluderGrid(primitives, bounds, cellSize, limits);
    return {
      kind,
      aborted: grid.aborted,
      trackedBytes: estimateGridPayloadBytes(grid.triangleCount),
      triangleCount: grid.triangleCount,
      blocked: (ax, ay, az, bx, by, bz) => segmentBlocked(grid, ax, ay, az, bx, by, bz),
      floorAt: (x, z) => highestSurfaceAt(grid, x, z, bounds),
      floorsAt: (x, z) => columnSurfaces(grid, x, z, bounds),
      counters: grid.counters,
      nodeVisits: () => grid.counters.cellVisits,
      references: grid.bucketReferences,
    };
  }
  const bvh: OccluderBvh = buildOccluderBvh(primitives, limits);
  return {
    kind: 'bvh',
    aborted: bvh.aborted,
    trackedBytes: bvh.aborted ? 0 : estimateBvhBytes(bvh.triangleCount),
    triangleCount: bvh.triangleCount,
    blocked: (ax, ay, az, bx, by, bz) => bvhSegmentBlocked(bvh, ax, ay, az, bx, by, bz),
    floorAt: (x, z) => bvhHighestSurfaceAt(bvh, x, z),
    floorsAt: (x, z) => bvhColumnSurfaces(bvh, x, z),
    counters: bvh.counters,
    nodeVisits: () => bvh.counters.nodeVisits,
    references: bvh.nodeCount,
  };
}

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
/**
 * How much clear space a floor needs above it to be somewhere a camera can be.
 *
 * Every surface under a column is a candidate floor, which is what makes the
 * street under an arcade, the room under a roof and each storey of a crypt
 * into camera volumes instead of one rooftop. A surface with a ceiling right
 * on top of it is not a floor, it is the underside of something, and standing
 * a viewer in the gap would sample a place no player reaches.
 */
const MIN_FLOOR_CLEARANCE = 6;
/** Ceiling on floors sampled per column; deep stacks cost queries linearly. */
const MAX_FLOORS_PER_COLUMN = 4;
/**
 * How far a band reaches below its own floor.
 *
 * Enough that a camera standing on the surface is inside the band rather than
 * on its boundary, and small enough that it cannot reach the floor beneath.
 */
const BAND_FOOTING = 0.5;
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
  /** Bounds enforced while the bake runs; exhaustion admits the rest. */
  budget?: ShadoWorldVisibilityBudget;
  /**
   * Occluders kept as prototypes and placements rather than expanded.
   *
   * Indexed once per prototype and queried through their transforms, which is
   * the difference between indexing 145k triangles and 4.09M copies of them.
   * `collisionPrimitives` still carries anything not instanced, and both are
   * consulted.
   */
  instancedOccluders?: {
    prototypes: readonly (readonly ShadoWorldPrimitive[])[];
    instances: readonly { prototype: number; matrix: readonly number[] }[];
  };
  /** Which acceleration structure answers segment queries. Defaults to `bvh`. */
  occluderIndex?: ShadoWorldOccluderIndex;
  /**
   * Split each region into vertical source volumes instead of one column.
   *
   * A column holds a room and the roof above it at once, so its row has to
   * serve both and is therefore a rooftop row. Measured on Crypts: sampling
   * every floor rather than only the roof changed the result by nothing at
   * all, because the roof is still in the same source region. Splitting the
   * column is the only thing that can change it.
   */
  verticalVolumes?: boolean;
  /**
   * How many regions of camera-position slack each source row carries, in
   * place of {@link CAMERA_ROW_MARGIN}. Lowering it is a measurement lever,
   * not a shipping default: the margin is what stops geometry popping as a
   * player crosses a region edge.
   */
  cameraRowMargin?: number;
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
  /**
   * What blocks sight. Historically the zone's collision mesh, hence the
   * name; a caller that has assembled the opaque visual scene passes that
   * instead.
   */
  collisionPrimitives: readonly ShadoWorldPrimitive[];
  /**
   * What a viewer can STAND on, if that is not the same set.
   *
   * Eye samples sit above the highest surface under a region, and a surface
   * you can stand on is not the same thing as a surface that blocks sight: a
   * pane of glass holds nobody up, and the ground holds everybody up while
   * hiding almost nothing. Baking with only placed objects as occluders made
   * the conflation obvious -- 2,529 of Crownward's 2,880 regions had no floor
   * under them, so they were admitted unsampled and the bake culled nothing
   * whatever the objects did. Defaults to the occluder set, which is the
   * historical behaviour.
   */
  groundPrimitives?: readonly ShadoWorldPrimitive[];
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
  /*
   * Rows are allocated once the volumes are known, below: one per volume plus
   * one conservative union per region. Targets stay region-indexed, so the
   * bitset is rectangular rather than square and the kernel -- which already
   * indexes a row by a number the host supplies -- needs no change.
   */
  let words = new Uint32Array(0);
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
  const clock = () => (typeof performance === 'undefined' ? Date.now() : performance.now());
  const bakeStarted = clock();
  /*
   * One deadline for the whole bake, started before anything is built.
   *
   * A budget that begins after the index and the sampling have already run
   * bounds the cheapest stage and leaves the expensive ones unbounded, which
   * is how a `--budget-seconds 0` run finished successfully and reported no
   * failure. Every stage below asks the same question, and every affirmative
   * answer degrades the bake towards admitting rather than towards hiding.
   */
  const budget = input.budget;
  const deadline = budget?.maxSeconds === undefined
    ? Number.POSITIVE_INFINITY
    : bakeStarted + budget.maxSeconds * 1000;
  const maxSegmentQueries = budget?.maxSegmentQueries ?? Number.POSITIVE_INFINITY;
  let stop: ShadoWorldVisibilityBakeReport['limit']['stop'] = 'none';
  let stoppedDuring: ShadoWorldVisibilityBakeReport['limit']['stoppedDuring'] = 'none';
  const exhausted = (segmentQueries: number): typeof stop => {
    if (budget?.signal?.aborted) return 'cancelled';
    if (segmentQueries >= maxSegmentQueries) return 'segment-queries';
    if (clock() > deadline) return 'seconds';
    if (
      budget?.maxResidentBytes !== undefined &&
      budget.residentBytes &&
      budget.residentBytes() > budget.maxResidentBytes
    ) return 'memory';
    return 'none';
  };
  const check = (
    during: ShadoWorldVisibilityBakeReport['limit']['stoppedDuring'],
    segmentQueries: number
  ): boolean => {
    if (stop !== 'none') return true;
    const reason = exhausted(segmentQueries);
    if (reason === 'none') return false;
    stop = reason;
    stoppedDuring = during;
    return true;
  };

  const gridStarted = clock();
  /*
   * Construction is bounded from the inside, not merely surrounded by checks.
   * A synchronous build cannot be interrupted by a test that runs before and
   * after it, and on a five-million-triangle scene the allocation alone can
   * exceed a memory ceiling before control ever comes back. So the limits go
   * in: refuse the allocation up front when it is predictably too large, and
   * poll for cancellation while reading geometry.
   */
  /*
   * Recomputed immediately before each build, never shared between them.
   *
   * One allowance handed to both indexes lets two structures that each fit
   * individually exceed the ceiling together: the first one is resident by the
   * time the second is measured, so the second has to be offered what is
   * actually left rather than what the first was offered.
   */
  const buildLimits = (): OccluderBuildLimits => ({
    ...(budget?.maxResidentBytes !== undefined && budget.residentBytes
      ? { maxBytes: Math.max(0, budget.maxResidentBytes - budget.residentBytes()) }
      : {}),
    ...(budget
      ? {
          stopReason: () => {
            const reason = exhausted(0);
            return reason === 'none' ? null : reason;
          },
        }
      : {}),
  });
  /*
   * Only the hierarchy can be bounded, so a caller asking for both bounded
   * execution and the grid is refused here rather than at the CLI -- the
   * compiler is the public entry point and has to hold the contract itself.
   */
  const requestedIndex = input.occluderIndex ?? 'auto';
  const resolvedIndex: 'bvh' | 'grid' = requestedIndex === 'grid' ? 'grid' : 'bvh';
  if (resolvedIndex === 'grid' && hasBoundedLimits(buildLimits())) {
    throw new ShadoOccluderBackendError(
      "A bounded bake cannot use the 'grid' occluder index: it has no memory " +
        "or cancellation contract. Select 'bvh' or 'auto', or drop the budget."
    );
  }
  const built = requestedMode === 'sampled-occlusion' && !check('index-build', 0)
    ? buildOccluders(
        resolvedIndex,
        input.collisionPrimitives,
        input.bounds,
        OCCLUDER_CELL_SIZE,
        buildLimits()
      )
    : null;
  if (built?.aborted) {
    // The guard carries the exact reason; a deadline noticed inside a build
    // is a deadline and must not be reported as a generic cancellation.
    stop = built.aborted;
    stoppedDuring = 'index-build';
  }
  // An abandoned index holds nothing, so using it would silently mean "no
  // occluders" rather than "stopped"; dropping it makes the fallback explicit.
  const grid = built?.aborted ? null : built;
  /*
   * A second index only when the standing surfaces are a different set. It
   * answers "what is under this column" and never "what blocks this segment",
   * so nothing it contains can hide anything.
   */
  const groundBuilt = grid && input.groundPrimitives && !check('index-build', grid.counters.segmentQueries)
    ? buildOccluders(
        resolvedIndex,
        input.groundPrimitives,
        input.bounds,
        OCCLUDER_CELL_SIZE,
        buildLimits()
      )
    : null;
  if (groundBuilt?.aborted) {
    stop = groundBuilt.aborted;
    stoppedDuring = 'index-build';
  }
  const ground = input.groundPrimitives
    ? (groundBuilt?.aborted ? null : groundBuilt)
    : grid;
  const occluderGridMs = clock() - gridStarted;
  // Without ground there is nothing to stand on, so nothing can be sampled.
  const instanced = requestedMode === 'sampled-occlusion' && input.instancedOccluders
    ? buildInstancedOccluders(
        input.instancedOccluders.prototypes,
        input.instancedOccluders.instances as never,
        buildLimits()
      )
    : null;
  if (instanced?.aborted) {
    stop = instanced.aborted;
    stoppedDuring = 'index-build';
  }
  const placed = instanced && !instanced.aborted && instanced.instanceCount > 0 ? instanced : null;
  const sampled =
    ((grid !== null && grid.triangleCount > 0) || placed !== null) && ground !== null;
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

  const samplingStarted = clock();
  let regionsWithoutFloor = 0;
  let regionsLeftUnsampled = 0;
  /*
   * One source volume per (region, vertical band), or exactly one per region
   * spanning every height when volumes are off -- which is the historical
   * behaviour, expressed in the same loop rather than in a second one.
   */
  /*
   * Vertical volumes are a property of the OCCLUSION bake. A distance flood
   * tests nothing, so splitting its columns would produce rows that differ
   * only in index -- and the flood is the baseline every comparison is read
   * against, which must keep the layout it has always had.
   */
  const useVolumes = input.verticalVolumes === true && sampled;
  const volumeRegion: number[] = [];
  const volumeMinY: number[] = [];
  const volumeMaxY: number[] = [];
  const volumeEyes: (Float64Array | null)[] = [];
  /*
   * A volume's own targets, not its column's.
   *
   * Seeing is mutual, so the pair is tested both ways -- and the reverse
   * direction asks whether the target column can see THIS VOLUME. Handing it
   * the whole column's targets asks whether the street can see the room's
   * roof, which it can, and the room is admitted on the strength of it.
   */
  const volumeTargets: (Float64Array | null)[] = [];
  const volumesOfRegion: number[][] = Array.from({ length: regionCount }, () => []);
  for (let region = 0; sampled && region < regionCount; region++) {
    // Sampling a region walks its footprint against the whole occluder set,
    // which on a dense zone is not cheap; an unsampled region has no eyes and
    // is therefore admitted, so stopping here is safe and merely worse.
    if ((region & 31) === 0 && check('region-sampling', segmentQueryCount(grid, placed))) {
      regionsLeftUnsampled = regionCount - region;
      break;
    }
    const eyePoints: number[] = [];
    const targetPoints: number[] = [];
    const low = regionLow[region]!;
    const high = regionHigh[region]!;
    const known = Number.isFinite(low) && Number.isFinite(high);
    const floorsHere: number[] = [];
    for (const [x, z] of footprint(region)) {
      /*
       * Every floor, not the roof. `floorsAt` comes back highest first, and a
       * surface counts as a floor only when the next surface above it leaves
       * room to stand: the underside of a stair is a surface and not a place.
       */
      const surfaces = ground!.floorsAt(x, z);
      const floors: number[] = [];
      for (let index = 0; index < surfaces.length && floors.length < MAX_FLOORS_PER_COLUMN; index += 1) {
        const height = surfaces[index]!;
        const above = index === 0 ? Number.POSITIVE_INFINITY : surfaces[index - 1]!;
        if (above - height >= MIN_FLOOR_CLEARANCE) floors.push(height);
      }
      const floor = floors.length ? floors[0]! : null;
      for (const level of floors) {
        for (const eye of EYE_HEIGHTS) eyePoints.push(x, level + eye, z);
        // Floors this region offers; the bands that tile them are built once
        // the whole footprint has been walked.
        if (useVolumes) floorsHere.push(level);
      }
      if (floor === null) {
        /*
         * Nowhere to stand here, so no eyes -- but a column with no floor
         * still holds geometry: a wall, a ceiling, the outside of a vault.
         * It has to be TESTABLE as something to look at, or every such
         * region is admitted from everywhere untested, and on an interior
         * zone most regions are exactly this.
         */
        if (known) {
          const top = high + STAMP_HEADROOM;
          for (let step = 0; step <= 3; step++) {
            targetPoints.push(x, low + ((top - low) * step) / 3, z);
          }
        }
        continue;
      }
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
    else regionsWithoutFloor += 1;
    if (targetPoints.length) targets[region] = Float64Array.from(targetPoints);
    if (useVolumes && eyePoints.length) {
      /*
       * The bands TILE the column: each runs from just under its floor to
       * just under the next one, and the top one runs to the sky. Disjoint
       * on purpose -- overlapping bands are unioned into each other by the
       * neighbour flood, which quietly puts the rooftop view back inside the
       * room and undoes the split.
       *
       * Two samples a metre apart on the same floor describe one place to
       * stand, so floors within the clearance are merged first.
       */
      floorsHere.sort((left, right) => left - right);
      const levels: number[] = [];
      for (const level of floorsHere) {
        const last = levels[levels.length - 1];
        if (last !== undefined && level - last < MIN_FLOOR_CLEARANCE) continue;
        levels.push(level);
      }
      const bands = levels.map((level, index) => [
        level - BAND_FOOTING,
        index + 1 < levels.length ? levels[index + 1]! - BAND_FOOTING : Number.POSITIVE_INFINITY,
      ] as [number, number]);
      for (const [minY, maxY] of bands) {
        const volume = volumeRegion.length;
        volumeRegion.push(region);
        volumeMinY.push(minY);
        volumeMaxY.push(maxY);
        const within: number[] = [];
        for (let offset = 0; offset < eyePoints.length; offset += 3) {
          const y = eyePoints[offset + 1]!;
          if (y >= minY && y < maxY) {
            within.push(eyePoints[offset]!, y, eyePoints[offset + 2]!);
          }
        }
        volumeEyes.push(within.length ? Float64Array.from(within) : null);
        const targetsWithin: number[] = [];
        for (let offset = 0; offset < targetPoints.length; offset += 3) {
          const y = targetPoints[offset + 1]!;
          if (y >= minY && y < maxY) {
            targetsWithin.push(targetPoints[offset]!, y, targetPoints[offset + 2]!);
          }
        }
        // A band with no target sample of its own still has its eyes; the
        // reverse test then falls back to the column, which admits more.
        volumeTargets.push(targetsWithin.length ? Float64Array.from(targetsWithin) : null);
        volumesOfRegion[region]!.push(volume);
      }
    }
  }
  if (useVolumes) {
    /*
     * A region the sampler gave no floor -- or never reached, because the
     * budget stopped it -- still needs a row, or the union row a reader
     * lands on would be empty and would hide the whole world. It gets one
     * full-column volume with whatever samples the region has, which is
     * exactly the row an unsplit bake would have written.
     */
    for (let region = 0; region < regionCount; region += 1) {
      if (volumesOfRegion[region]!.length) continue;
      volumesOfRegion[region]!.push(volumeRegion.length);
      volumeRegion.push(region);
      volumeMinY.push(Number.NEGATIVE_INFINITY);
      volumeMaxY.push(Number.POSITIVE_INFINITY);
      volumeEyes.push(eyes[region] ?? null);
      volumeTargets.push(targets[region] ?? null);
    }
  } else {
    // One volume per region, covering every height: the row indexing every
    // existing package uses.
    for (let region = 0; region < regionCount; region += 1) {
      volumeRegion.push(region);
      volumeMinY.push(Number.NEGATIVE_INFINITY);
      volumeMaxY.push(Number.POSITIVE_INFINITY);
      volumeEyes.push(eyes[region] ?? null);
      volumeTargets.push(targets[region] ?? null);
      volumesOfRegion[region]!.push(region);
    }
  }
  const regionSamplingMs = clock() - samplingStarted;

  const volumeCount = volumeRegion.length;
  /*
   * Union rows exist only where volumes do. Without them a row IS a region,
   * every reader indexes it that way, and the package keeps exactly the
   * layout it has always had.
   */
  const rowCount = useVolumes ? volumeCount + regionCount : volumeCount;
  words = new Uint32Array(rowCount * wordsPerRow);
  let occlusionTested = 0;
  let occluded = 0;
  let forcedLocalPairs = 0;
  const pairLoopStarted = clock();
  let pairsAdmittedAfterStop = 0;
  const geometry = { originX, originZ, size, width, height, maxDistance };
  for (let source = 0; source < volumeCount; source++) {
    const from = volumeRegion[source]!;
    for (let to = 0; to < regionCount; to++) {
      const local = isLocalPair(from, to, geometry);
      if (!local && !withinRange(from, to, geometry)) continue;
      if (local) {
        forcedLocalPairs++;
      } else if (sampled) {
        /*
         * The budget is checked before the work, not after it: a budget that
         * reports a breach once the run is over has protected nothing. Once
         * stopped, every remaining pair is ADMITTED untested, so a truncated
         * bake is a worse PVS and never an unsafe one.
         */
        if (check('pair-loop', segmentQueryCount(grid, placed))) {
          pairsAdmittedAfterStop++;
        } else {
          /*
           * Tested from THIS volume's eyes, and against the target column's
           * samples. The reverse direction is tested too, because a straight
           * segment is mutual and testing one way alone lets sampling noise
           * produce a row that disagrees with itself.
           */
          const eyePoints = volumeEyes[source];
          const target = targets[to];
          const back = eyes[to];
          const forward = volumeTargets[source] ?? targets[from];
          /*
           * Both directions when both are samplable, and whichever one is
           * otherwise. Seeing is mutual, so agreement between the two is the
           * better evidence -- but a target region with no floor has no eyes
           * to look back from, and refusing to test it at all admits it from
           * everywhere. One tested direction is weaker evidence than two and
           * far stronger than none.
           */
          const canForward = eyePoints !== null && target !== null;
          const canBack = back !== null && forward !== null;
          if (canForward || canBack) {
            occlusionTested++;
            const forwardBlocked =
              !canForward || !anyClearSegment(grid, placed, eyePoints!, target!);
            const backBlocked =
              !canBack || !anyClearSegment(grid, placed, back!, forward!);
            if (forwardBlocked && backBlocked) {
              occluded++;
              continue;
            }
          }
        }
      }
      setVisible(source, to);
    }
  }
  /*
   * The union row per region: what a camera at a height no volume covers is
   * allowed to see. It admits everything any volume in that column admits,
   * which costs draw calls and cannot hide anything -- the safe answer for
   * debug flight, for a gap between bands, and for a reader that cannot place
   * the camera at all.
   */
  if (useVolumes) {
    for (let region = 0; region < regionCount; region++) {
      const unionRow = (volumeCount + region) * wordsPerRow;
      for (const volume of volumesOfRegion[region]!) {
        const volumeRow = volume * wordsPerRow;
        for (let word = 0; word < wordsPerRow; word++) {
          words[unionRow + word] = (words[unionRow + word]! | words[volumeRow + word]!) >>> 0;
        }
      }
    }
  }
  const pairLoopMs = clock() - pairLoopStarted;
  const rawPairs = visibleRegionPairs;
  const rowFloodStarted = clock();
  const floodedWords = floodVolumeRows(
    words,
    wordsPerRow,
    volumeRegion,
    volumeMinY,
    volumeMaxY,
    volumeCount,
    regionCount,
    volumeEyes.map((points) => points !== null),
    useVolumes,
    width,
    height,
    input.cameraRowMargin ?? CAMERA_ROW_MARGIN
  );
  visibleRegionPairs = countVisibleBits(floodedWords);
  const rowFloodMs = clock() - rowFloodStarted;
  if (input.report) {
    input.report({
      requestedMode,
      mode,
      fallbackReason:
        requestedMode === 'sampled-occlusion' && !sampled
          ? (stop === 'none' ? 'no-eligible-occluders' : 'budget-exhausted')
          : null,
      occluderTriangles: sampled ? (grid?.triangleCount ?? 0) + (placed?.uniqueTriangles ?? 0) : 0,
      forcedLocalPairs,
      occlusionTested,
      occluded,
      pairsBeforeRowFlood: rawPairs,
      pairsAfterRowFlood: visibleRegionPairs,
      stages: {
        occluderGridMs,
        regionSamplingMs,
        pairLoopMs,
        rowFloodMs,
        totalMs: clock() - bakeStarted,
      },
      work: {
        index: grid?.kind ?? 'bvh',
        instancedPrototypes: placed?.prototypes.length ?? 0,
        instancedPlacements: placed?.instanceCount ?? 0,
        instancedUniqueTriangles: placed?.uniqueTriangles ?? 0,
        instancedPlacedTriangles: placed?.placedTriangles ?? 0,
        indexTrackedBytes: (grid?.trackedBytes ?? 0) + (ground && ground !== grid ? ground.trackedBytes : 0),
        segmentQueries: segmentQueryCount(grid, placed),
        blockedQueries: (grid?.counters.blockedQueries ?? 0) + (placed?.counters.blockedQueries ?? 0),
        columnQueries: ground?.counters.columnQueries ?? 0,
        nodeVisits: (grid ? grid.nodeVisits() : 0) + (placed?.counters.nodeVisits ?? 0),
        triangleTests: (grid?.counters.triangleTests ?? 0) + (placed?.counters.triangleTests ?? 0),
        indexEntries: grid?.references ?? 0,
        regionsWithoutFloor,
      },
      limit: { stop, stoppedDuring, pairsAdmittedAfterStop, regionsLeftUnsampled },
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
    occluderCount: sampled ? (grid?.triangleCount ?? 0) + (placed?.uniqueTriangles ?? 0) : 0,
    visibleRegionPairs,
    cellRegion,
    persistentRegions,
    persistentCells,
    ...(useVolumes
      ? {
          volumes: {
            count: volumeCount,
            region: volumeRegion,
            minY: volumeMinY,
            maxY: volumeMaxY,
          },
        }
      : {}),
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
  grid: Occluders | null,
  placed: InstancedOccluders | null,
  eyes: Float64Array,
  targets: Float64Array
): boolean {
  for (let t = targets.length - 3; t >= 0; t -= 3) {
    for (let e = 0; e < eyes.length; e += 3) {
      const blocked =
        (grid !== null && grid.triangleCount > 0 && grid.blocked(
          eyes[e]!, eyes[e + 1]!, eyes[e + 2]!,
          targets[t]!, targets[t + 1]!, targets[t + 2]!
        )) ||
        (placed !== null && instancedSegmentBlocked(
          placed,
          eyes[e]!, eyes[e + 1]!, eyes[e + 2]!,
          targets[t]!, targets[t + 1]!, targets[t + 2]!
        ));
      if (!blocked) return true;
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
 * Unions each row with the rows of neighbouring regions.
 *
 * The same conservative source flood as before, expressed over volumes: a
 * player about to cross a region edge already sees what the next column
 * admits, so nothing pops at the boundary. Rows are unioned from every volume
 * of every neighbouring region, not only from the matching band, because a
 * player crossing an edge may also be changing floor -- and because a row
 * that admits too much is the safe kind of wrong.
 */
function floodVolumeRows(
  source: Uint32Array,
  wordsPerRow: number,
  volumeRegion: readonly number[],
  volumeMinY: readonly number[],
  volumeMaxY: readonly number[],
  volumeCount: number,
  regionCount: number,
  /**
   * Whether each volume is a place a camera can actually be -- whether the
   * sampler found a floor to stand on inside it.
   */
  volumeStandable: readonly boolean[],
  /** Whether the layout carries a union row per region after the volumes. */
  unionRows: boolean,
  width: number,
  height: number,
  radius: number
): Uint32Array {
  const result = source.slice();
  const volumesByRegion: number[][] = Array.from({ length: regionCount }, () => []);
  for (let volume = 0; volume < volumeCount; volume += 1) {
    volumesByRegion[volumeRegion[volume]!]!.push(volume);
  }
  const unionInto = (targetRow: number, sourceRow: number): void => {
    for (let word = 0; word < wordsPerRow; word += 1) {
      result[targetRow + word] = (result[targetRow + word]! | source[sourceRow + word]!) >>> 0;
    }
  };
  for (let volume = 0; volume < volumeCount; volume += 1) {
    const region = volumeRegion[volume]!;
    const regionX = region % width;
    const regionZ = Math.floor(region / width);
    for (let deltaZ = -radius; deltaZ <= radius; deltaZ += 1) {
      const neighbourZ = regionZ + deltaZ;
      if (neighbourZ < 0 || neighbourZ >= height) continue;
      for (let deltaX = -radius; deltaX <= radius; deltaX += 1) {
        const neighbourX = regionX + deltaX;
        if (neighbourX < 0 || neighbourX >= width) continue;
        const neighbour = neighbourZ * width + neighbourX;
        for (const other of volumesByRegion[neighbour]!) {
          /*
           * Only between bands that overlap. The flood exists so nothing pops
           * when a player crosses a region edge, and crossing an edge does
           * not change their height -- so a room has no reason to inherit
           * what the roof above the next column can see. Unioning every band
           * of every neighbour puts the rooftop view straight back into the
           * room and undoes the split entirely.
           */
          if (volumeMinY[other]! >= volumeMaxY[volume]!) continue;
          if (volumeMaxY[other]! <= volumeMinY[volume]!) continue;
          /*
           * And only FROM somewhere a camera can be. A volume with no floor
           * holds no eyes, so its pairs went untested and its row admits
           * everything -- and no player can cross an edge into it, because
           * there is nothing there to stand on. Letting it donate that row
           * makes every room beside solid rock fully visible, which on an
           * interior zone is most of them.
           *
           * Only where volumes exist: an unsplit bake is the layout every
           * shipped package already uses, and narrowing its flood here would
           * change rows nothing in this work asked to change.
           */
          if (unionRows && !volumeStandable[other]) continue;
          unionInto(volume * wordsPerRow, other * wordsPerRow);
        }
      }
    }
  }
  // Union rows follow their column's volumes, after those have been flooded.
  if (!unionRows) return result;
  for (let region = 0; region < regionCount; region += 1) {
    const unionRow = (volumeCount + region) * wordsPerRow;
    for (const volume of volumesByRegion[region]!) {
      for (let word = 0; word < wordsPerRow; word += 1) {
        result[unionRow + word] = (result[unionRow + word]! | result[volume * wordsPerRow + word]!) >>> 0;
      }
    }
  }
  return result;
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
