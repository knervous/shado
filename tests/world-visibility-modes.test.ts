import {
  ShadoWorldVisibilityCoordinator,
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
    report: (value: ShadoWorldVisibilityBakeReport) => void
  ) {
    const strip = groundStrip(LENGTH, DEPTH, 8);
    const blocker = wall(WALL_X, DEPTH, 200);
    const centers: [number, number][] = [];
    for (let x = REGION / 2; x < LENGTH; x += REGION) centers.push([x, DEPTH / 2]);
    return compileShadoWorldVisibility({
      mode: 'sampled-occlusion',
      budget,
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

    expect(full[0]!.limit).toEqual({ stop: 'none', pairsAdmittedAfterStop: 0 });
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
    expect(work.cellVisits).toBeGreaterThanOrEqual(work.segmentQueries);
    expect(work.triangleTests).toBeGreaterThan(0);
    // A triangle spanning several cells is referenced by each of them.
    expect(work.bucketReferences).toBeGreaterThanOrEqual(reports[0]!.occluderTriangles);
    expect(stages.totalMs).toBeGreaterThanOrEqual(stages.pairLoopMs);
    expect(stages.occluderGridMs).toBeGreaterThanOrEqual(0);
  });
});
