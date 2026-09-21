import {
  ShadoWorldVisibilityCoordinator,
  buildOccluderBvh,
  buildOccluderGrid,
  bvhColumnSurfaces,
  columnSurfaces,
  compileShadoWorld,
  compileShadoWorldVisibility,
  validateShadoWorldPackage,
} from '../src/world';
import type {
  ShadoWorldPrimitive,
  ShadoWorldSpatialPackage,
  ShadoWorldVisibilityBakeReport,
} from '../src/world';

/**
 * A strip of ground with one tall opaque wall standing across it.
 *
 * The strip is 160 units long and the wall stands at x = 80, so with 16-unit
 * regions there are ten regions in a row and the wall stands on the boundary
 * of the sixth. Regions on opposite sides of it are genuinely hidden from each
 * other at eye level; regions on the same side are not.
 */
function groundStrip(length: number, depth: number, step: number): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let x = 0; x < length; x += step) {
    for (let z = 0; z < depth; z += step) {
      const v = positions.length / 3;
      positions.push(x, 0, z, x + step, 0, z, x + step, 0, z + step, x, 0, z + step);
      indices.push(v, v + 1, v + 2, v, v + 2, v + 3);
    }
  }
  return {
    name: 'ground',
    material: 'stone',
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
  };
}

function wall(x: number, depth: number, top: number): ShadoWorldPrimitive {
  return {
    name: 'wall',
    material: 'stone',
    positions: new Float32Array([x, 0, 0, x, 0, depth, x, top, depth, x, top, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

const LENGTH = 160;
const DEPTH = 16;
const REGION = 16;
const WALL_X = 80;

function walledZone(mode?: 'distance-flood' | 'sampled-occlusion') {
  return compileShadoWorld([groundStrip(LENGTH, DEPTH, 8), wall(WALL_X, DEPTH, 200)], {
    name: 'walled-strip',
    tileSize: REGION,
    visibilityRegionSize: REGION,
    visibilityMaxDistance: 1024,
    visibilityMode: mode,
    maxClusterTriangles: 64,
  });
}

function regionVisible(
  visibility: NonNullable<ReturnType<typeof compileShadoWorld>['visibility']>,
  from: number,
  to: number
): boolean {
  const word = visibility.pvs.words[from * visibility.pvs.wordsPerRow + (to >>> 5)]! >>> 0;
  return (word & (1 << (to & 31))) !== 0;
}

describe('visibility modes', () => {
  it('bakes the distance flood unless occlusion is explicitly asked for', () => {
    for (const world of [walledZone(), walledZone('distance-flood')]) {
      const visibility = world.visibility!;
      expect(visibility.mode).toBe('distance-flood');
      expect(visibility.occluderCount).toBe(0);
      // Everything in range, wall or no wall. That is the baseline being kept.
      expect(regionVisible(visibility, 0, 9)).toBe(true);
      validateShadoWorldPackage(world);
    }
  });

  it('rejects across a wall only in the occlusion mode, and stays valid', () => {
    const world = walledZone('sampled-occlusion');
    const visibility = world.visibility!;
    expect(visibility.mode).toBe('sampled-occlusion');
    expect(visibility.occluderCount).toBeGreaterThan(0);
    validateShadoWorldPackage(world);

    // Far side of the wall: hidden from a viewpoint standing at the near end.
    expect(regionVisible(visibility, 0, 9)).toBe(false);
    expect(regionVisible(visibility, 9, 0)).toBe(false);
    // Same side of the wall: nothing is between them, so nothing may cull them.
    expect(regionVisible(visibility, 0, 4)).toBe(true);
    // And the flood admitted both, so the difference is the occlusion test.
    const flood = walledZone('distance-flood').visibility!;
    expect(regionVisible(flood, 0, 9)).toBe(true);
    expect(visibility.visibleRegionPairs).toBeLessThan(flood.visibleRegionPairs);
  });

  it('never lets the occlusion loop override a local admission', () => {
    const visibility = walledZone('sampled-occlusion').visibility!;
    // Regions 4 and 6 sit on opposite sides of the wall and are two steps
    // apart, which is inside the local safety neighborhood. The camera can
    // walk to the region edge and see over or around, so these bits stand
    // however thoroughly the wall blocks the sampled segments.
    expect(regionVisible(visibility, 4, 6)).toBe(true);
    expect(regionVisible(visibility, 6, 4)).toBe(true);
    expect(regionVisible(visibility, 5, 5)).toBe(true);
    for (let region = 0; region < visibility.width * visibility.height; region += 1) {
      expect(regionVisible(visibility, region, region)).toBe(true);
    }
  });

  it('stands viewers on ground that is not in the occluder set', () => {
    /*
     * The wall occludes; the ground does not. Baking with only the wall as an
     * occluder leaves every region floorless and therefore unsampled, so
     * nothing is culled however solid the wall is. Handing the ground over
     * separately restores the sampling without letting the ground hide
     * anything.
     */
    const strip = groundStrip(LENGTH, DEPTH, 8);
    const blocker = wall(WALL_X, DEPTH, 200);
    const centers: [number, number][] = [];
    for (let x = REGION / 2; x < LENGTH; x += REGION) centers.push([x, DEPTH / 2]);
    const compile = (ground?: typeof strip) =>
      compileShadoWorldVisibility({
        mode: 'sampled-occlusion',
        bounds: { min: [0, 0, 0], max: [LENGTH, 200, DEPTH] },
        regionSize: REGION,
        maxDistance: 1024,
        renderCellCenters: centers,
        persistentRenderCells: new Uint8Array(centers.length),
        collisionPrimitives: [blocker],
        ...(ground ? { groundPrimitives: [ground] } : {}),
      });
    const bit = (v: ReturnType<typeof compile>, from: number, to: number) =>
      ((v.pvs.words[from * v.pvs.wordsPerRow + (to >>> 5)]! >>> 0) & (1 << (to & 31))) !== 0;

    // Wall alone: nothing to stand on, so nothing is judged.
    expect(bit(compile(), 0, 9)).toBe(true);
    // Wall to block, ground to stand on.
    expect(bit(compile(strip), 0, 9)).toBe(false);
  });

  it('falls back to the flood, and says so, when nothing is eligible to occlude', () => {
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    const visibility = compileShadoWorldVisibility({
      mode: 'sampled-occlusion',
      bounds: { min: [0, 0, 0], max: [LENGTH, 8, DEPTH] },
      regionSize: REGION,
      maxDistance: 1024,
      renderCellCenters: [[8, 8], [152, 8]],
      persistentRenderCells: new Uint8Array([0, 0]),
      collisionPrimitives: [],
      report: (report) => reports.push(report),
    });
    expect(visibility.mode).toBe('distance-flood');
    expect(visibility.occluderCount).toBe(0);
    expect(reports[0]).toMatchObject({
      requestedMode: 'sampled-occlusion',
      mode: 'distance-flood',
      fallbackReason: 'no-eligible-occluders',
      occluderTriangles: 0,
      occlusionTested: 0,
      occluded: 0,
    });
    expect(reports[0]!.forcedLocalPairs).toBeGreaterThan(0);
  });

  it('reports occlusion separately from the distance envelope', () => {
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    compileShadoWorld([groundStrip(LENGTH, DEPTH, 8), wall(WALL_X, DEPTH, 200)], {
      name: 'walled-strip-report',
      tileSize: REGION,
      visibilityRegionSize: REGION,
      // Short envelope: most pairs never reach an occlusion test at all, and
      // the ones range removes must not be counted as occlusion wins.
      visibilityMaxDistance: 64,
      visibilityMode: 'sampled-occlusion',
      maxClusterTriangles: 64,
      visibilityReport: (report) => reports.push(report),
    });
    const report = reports[0]!;
    expect(report.mode).toBe('sampled-occlusion');
    expect(report.occluded).toBeLessThanOrEqual(report.occlusionTested);
    expect(report.forcedLocalPairs).toBeGreaterThan(0);
    expect(report.pairsAfterRowFlood).toBeGreaterThanOrEqual(report.pairsBeforeRowFlood);
  });

  it('bypasses baked occlusion in the reference authority, and nothing else', async () => {
    const world = walledZone('sampled-occlusion');
    // Wide enough to reject nothing: the difference between these two runs is
    // the visibility rows, not the frustum.
    const planes = new Float32Array([
      1, 0, 0, 4096, -1, 0, 0, 4096, 0, 1, 0, 4096,
      0, -1, 0, 4096, 0, 0, 1, 4096, 0, 0, -1, 4096,
    ]);
    const camera: [number, number, number] = [8, 8, 8];
    const baked = await ShadoWorldVisibilityCoordinator.create(world);
    const reference = await ShadoWorldVisibilityCoordinator.create(world, {
      visibilityAuthority: 'flood-reference',
    });
    expect(baked.visibilityAuthority).toBe('package');
    expect(reference.visibilityAuthority).toBe('flood-reference');

    const bakedFrame = baked.reduceWorld(planes, camera);
    const referenceFrame = reference.reduceWorld(planes, camera);
    expect(bakedFrame.cameraRegion).toBe(referenceFrame.cameraRegion);

    const bakedClusters = new Set(bakedFrame.visibleClusters);
    const referenceClusters = new Set(referenceFrame.visibleClusters);
    // The reference is a superset: bypassing occlusion can only admit more.
    for (const cluster of bakedClusters) expect(referenceClusters.has(cluster)).toBe(true);
    expect(referenceClusters.size).toBeGreaterThan(bakedClusters.size);
  });
});

describe('bounded bakes', () => {
  /** The walled strip again, driven through the compiler directly. */
  function bake(
    budget: Parameters<typeof compileShadoWorldVisibility>[0]['budget'],
    report: (value: ShadoWorldVisibilityBakeReport) => void,
    occluderIndex: 'bvh' | 'grid' | 'auto' = 'bvh'
  ) {
    const strip = groundStrip(LENGTH, DEPTH, 8);
    const blocker = wall(WALL_X, DEPTH, 200);
    const centers: [number, number][] = [];
    for (let x = REGION / 2; x < LENGTH; x += REGION) centers.push([x, DEPTH / 2]);
    return compileShadoWorldVisibility({
      mode: 'sampled-occlusion',
      budget,
      occluderIndex,
      bounds: { min: [0, 0, 0], max: [LENGTH, 200, DEPTH] },
      regionSize: REGION,
      maxDistance: 1024,
      renderCellCenters: centers,
      persistentRenderCells: new Uint8Array(centers.length),
      collisionPrimitives: [strip, blocker],
      report,
    });
  }

  it('admits everything it did not get to, and says how much that was', () => {
    const full: ShadoWorldVisibilityBakeReport[] = [];
    const stopped: ShadoWorldVisibilityBakeReport[] = [];
    const complete = bake(undefined, (value) => full.push(value));
    // One segment query is enough to start and not enough to finish.
    const truncated = bake({ maxSegmentQueries: 1 }, (value) => stopped.push(value));

    expect(full[0]!.limit).toEqual({
      stop: 'none',
      stoppedDuring: 'none',
      pairsAdmittedAfterStop: 0,
      regionsLeftUnsampled: 0,
    });
    expect(stopped[0]!.limit.stop).toBe('segment-queries');
    expect(stopped[0]!.limit.pairsAdmittedAfterStop).toBeGreaterThan(0);
    expect(stopped[0]!.occluded).toBeLessThan(full[0]!.occluded);
    // Giving up costs selectivity, never correctness: the truncated rows are a
    // superset of the complete ones.
    expect(truncated.visibleRegionPairs).toBeGreaterThan(complete.visibleRegionPairs);
    for (let from = 0; from < truncated.width * truncated.height; from += 1) {
      for (let to = 0; to < truncated.width * truncated.height; to += 1) {
        const bit = (v: ShadoWorldSpatialPackage['visibility']) =>
          ((v!.pvs.words[from * v!.pvs.wordsPerRow + (to >>> 5)]! >>> 0) & (1 << (to & 31))) !== 0;
        if (bit(complete)) expect(bit(truncated)).toBe(true);
      }
    }
  });

  it('spends one deadline across every stage, not just the pair sweep', () => {
    /*
     * A zero-second budget must stop the bake before it builds an index or
     * samples a region, and must say which stage it died in. The rows it
     * returns are a flood, and it says that too rather than labelling them
     * occlusion-tested.
     */
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    const visibility = bake({ maxSeconds: 0 }, (value) => reports.push(value));
    const report = reports[0]!;
    expect(report.limit.stop).toBe('seconds');
    expect(report.limit.stoppedDuring).toBe('index-build');
    expect(report.mode).toBe('distance-flood');
    expect(report.fallbackReason).toBe('budget-exhausted');
    expect(report.occlusionTested).toBe(0);
    expect(report.work.segmentQueries).toBe(0);
    const flood = bake(undefined, () => {});
    expect(visibility.visibleRegionPairs).toBeGreaterThan(flood.visibleRegionPairs);
  });

  it('refuses an index it cannot afford, before allocating it', () => {
    /*
     * A synchronous build cannot be interrupted by checks that run before and
     * after it, so the limits go inside: the allocation is predicted from the
     * triangle count and refused while the memory is still unclaimed. The
     * bake then reports a flood, which is what its rows are.
     */
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    bake(
      { maxResidentBytes: 4096, residentBytes: () => 0 },
      (value) => reports.push(value)
    );
    const report = reports[0]!;
    expect(report.limit.stop).toBe('memory');
    expect(report.limit.stoppedDuring).toBe('index-build');
    expect(report.mode).toBe('distance-flood');
    expect(report.fallbackReason).toBe('budget-exhausted');
    expect(report.occluderTriangles).toBe(0);
  });

  it('does not attempt the instanced build once an earlier stage has stopped', () => {
    /*
     * The flat index is refused for memory. The instanced input below names
     * a prototype that does not exist, so if the builder were reached at all
     * it would throw -- the test passes only if it is skipped, and the stop
     * reason stays the one that actually stopped the bake.
     */
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    const strip = groundStrip(LENGTH, DEPTH, 8);
    const centers: [number, number][] = [];
    for (let x = REGION / 2; x < LENGTH; x += REGION) centers.push([x, DEPTH / 2]);
    compileShadoWorldVisibility({
      mode: 'sampled-occlusion',
      budget: { maxResidentBytes: 4096, residentBytes: () => 0 },
      occluderIndex: 'bvh',
      bounds: { min: [0, 0, 0], max: [LENGTH, 200, DEPTH] },
      regionSize: REGION,
      maxDistance: 1024,
      renderCellCenters: centers,
      persistentRenderCells: new Uint8Array(centers.length),
      collisionPrimitives: [strip, wall(WALL_X, DEPTH, 200)],
      instancedOccluders: {
        prototypes: [[wall(0, DEPTH, 200)]],
        instances: [{ prototype: 99, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }],
      } as never,
      report: (value) => reports.push(value),
    });
    expect(reports[0]!.limit.stop).toBe('memory');
    expect(reports[0]!.limit.stoppedDuring).toBe('index-build');
    expect(reports[0]!.mode).toBe('distance-flood');
  });

  it('abandons a build when cancelled part way through reading geometry', () => {
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    let polls = 0;
    // Aborts once construction has started reading, not before it begins.
    const signal = { get aborted() { return ++polls > 2; } };
    bake({ signal }, (value) => reports.push(value));
    expect(reports[0]!.limit.stop).toBe('cancelled');
    expect(reports[0]!.mode).toBe('distance-flood');
    expect(reports[0]!.occlusionTested).toBe(0);
  });

  it('offers the second index what is left, not what the first was offered', () => {
    /*
     * Two structures that each fit the ceiling can exceed it together, because
     * the first is resident by the time the second is measured. The allowance
     * is therefore recomputed against live usage before each build rather than
     * shared between them.
     */
    const offered: number[] = [];
    let resident = 0;
    const strip = groundStrip(LENGTH, DEPTH, 8);
    compileShadoWorldVisibility({
      mode: 'sampled-occlusion',
      bounds: { min: [0, 0, 0], max: [LENGTH, 200, DEPTH] },
      regionSize: REGION,
      maxDistance: 1024,
      renderCellCenters: [[8, 8]],
      persistentRenderCells: new Uint8Array(1),
      collisionPrimitives: [wall(WALL_X, DEPTH, 200)],
      groundPrimitives: [strip],
      budget: {
        maxResidentBytes: 64 * 1024 * 1024,
        // Each read reports more resident memory than the last, as a real
        // process would once the first index is allocated.
        residentBytes: () => {
          offered.push(resident);
          resident += 32 * 1024 * 1024;
          return resident;
        },
      },
    });
    // Consulted separately for the two builds, with the second seeing more
    // memory already taken than the first did.
    expect(offered.length).toBeGreaterThanOrEqual(2);
    expect(offered[1]!).toBeGreaterThan(offered[0]!);
  });

  it('refuses a bounded bake that asks for the grid backend', () => {
    /*
     * The grid has no memory or cancellation contract, so a caller asking for
     * both bounded execution and that backend is told, rather than silently
     * given unbounded execution or silently switched to another structure.
     * The refusal happens in the compiler, not only in the CLI.
     */
    expect(() =>
      bake({ maxResidentBytes: 4096, residentBytes: () => 0 }, () => {}, 'grid')
    ).toThrow(/bounded bake cannot use the 'grid'/);
    // Unbounded, the same request is a legitimate diagnostic.
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    bake(undefined, (value) => reports.push(value), 'grid');
    expect(reports[0]!.work.index).toBe('grid');
  });

  it('resolves auto to the bounded backend', () => {
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    bake({ maxSeconds: 60 }, (value) => reports.push(value), 'auto');
    expect(reports[0]!.work.index).toBe('bvh');
    expect(reports[0]!.work.indexTrackedBytes).toBeGreaterThan(0);
  });

  it('stops on a memory ceiling the host reports', () => {
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    bake(
      { maxResidentBytes: 1, residentBytes: () => 1024 },
      (value) => reports.push(value)
    );
    expect(reports[0]!.limit.stop).toBe('memory');
    expect(reports[0]!.occluded).toBe(0);
  });

  it('stops when cancelled, without producing a hidden row', () => {
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    const visibility = bake({ signal: { aborted: true } }, (value) => reports.push(value));
    expect(reports[0]!.limit.stop).toBe('cancelled');
    expect(reports[0]!.occlusionTested).toBe(0);
    expect(reports[0]!.occluded).toBe(0);
    // Nothing was tested, so this is the flood's row count by another route.
    const flood = bake(undefined, () => {});
    expect(visibility.visibleRegionPairs).toBeGreaterThan(flood.visibleRegionPairs);
  });

  it('counts the query work it did, and the duplication its grid bought', () => {
    const reports: ShadoWorldVisibilityBakeReport[] = [];
    bake(undefined, (value) => reports.push(value));
    const { work, stages } = reports[0]!;
    expect(work.segmentQueries).toBeGreaterThan(0);
    expect(work.blockedQueries).toBeGreaterThan(0);
    expect(work.blockedQueries).toBeLessThanOrEqual(work.segmentQueries);
    expect(work.columnQueries).toBeGreaterThan(0);
    expect(work.index).toBe('bvh');
    expect(work.nodeVisits).toBeGreaterThanOrEqual(work.segmentQueries);
    expect(work.triangleTests).toBeGreaterThan(0);
    // The hierarchy stores fewer entries than it indexes triangles; the grid
    // stores more. Which is why the field says what the index was.
    expect(work.indexEntries).toBeGreaterThan(0);
    expect(work.indexEntries).toBeLessThan(reports[0]!.occluderTriangles);
    expect(stages.totalMs).toBeGreaterThanOrEqual(stages.pairLoopMs);
    expect(stages.occluderGridMs).toBeGreaterThanOrEqual(0);
  });
});

describe('vertical separation', () => {
  /** A closed room with a roof, and open ground running away from it. */
  function roomAndGround(length: number, depth: number, roomEnd: number) {
    const positions: number[] = [];
    const indices: number[] = [];
    const quad = (corners: number[]) => {
      const v = positions.length / 3;
      positions.push(...corners);
      indices.push(v, v + 1, v + 2, v, v + 2, v + 3);
    };
    for (let x = 0; x < length; x += 8) {
      quad([x, 0, 0, x + 8, 0, 0, x + 8, 0, depth, x, 0, depth]);
      // A roof over the first stretch only.
      if (x < roomEnd) quad([x, 20, 0, x + 8, 20, 0, x + 8, 20, depth, x, 20, depth]);
    }
    // The room's far wall, closing it off from the rest of the strip.
    quad([roomEnd, 0, 0, roomEnd, 0, depth, roomEnd, 20, depth, roomEnd, 20, 0]);
    quad([roomEnd + 1, 0, 0, roomEnd + 1, 20, 0, roomEnd + 1, 20, depth, roomEnd + 1, 0, depth]);
    return {
      name: 'room',
      material: 'stone',
      positions: new Float32Array(positions),
      indices: new Uint32Array(indices),
    };
  }

  it('samples the floor under a roof as well as the roof itself', () => {
    const scene = roomAndGround(160, 16, 48);
    const grid = buildOccluderGrid([scene], { min: [0, 0, 0], max: [160, 20, 16] }, 32);
    const bvh = buildOccluderBvh([scene]);
    for (const surfaces of [
      columnSurfaces(grid, 24, 8, { min: [0, 0, 0], max: [160, 20, 16] }),
      bvhColumnSurfaces(bvh, 24, 8),
    ]) {
      // Roof and floor, both found; the old highest-surface query saw only the roof.
      expect(surfaces).toHaveLength(2);
      expect(surfaces[0]).toBeCloseTo(20, 5);
      expect(surfaces[1]).toBeCloseTo(0, 5);
    }
  });

  it('cannot hide a room from its own rooftop, because the region is both', () => {
    /*
     * This is the shape of the interior problem, and it is not eye placement.
     * A source region is a 2D column spanning every height, so it contains the
     * room AND the roof above it. The row has to hold for a camera anywhere in
     * the region, including the one standing on the roof with a clear view --
     * so the pair stays visible however well the roof hides the room from
     * inside. Recovering interior occlusion needs vertically separated source
     * volumes, not better sampling: adding the interior eye can only ever find
     * MORE ways to see, never fewer.
     */
    const scene = roomAndGround(160, 16, 48);
    const centers: [number, number][] = [];
    for (let x = 8; x < 160; x += 16) centers.push([x, 8]);
    const visibility = compileShadoWorldVisibility({
      mode: 'sampled-occlusion',
      bounds: { min: [0, 0, 0], max: [160, 20, 16] },
      regionSize: 16,
      maxDistance: 1024,
      renderCellCenters: centers,
      persistentRenderCells: new Uint8Array(centers.length),
      collisionPrimitives: [scene],
    });
    const bit = (from: number, to: number) =>
      ((visibility.pvs.words[from * visibility.pvs.wordsPerRow + (to >>> 5)]! >>> 0) &
        (1 << (to & 31))) !== 0;
    // Region 0 is inside the room; region 9 is far down the open strip. From
    // the room's floor the wall and roof block it; from the room's roof they
    // do not, and the roof is in the same region.
    expect(bit(0, 9)).toBe(true);
  });
});
