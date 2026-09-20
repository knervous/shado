import {
  compileShadoWorld,
  compileShadoWorldVisibility,
  validateShadoWorldPackage,
} from '../src/world';
import type { ShadoWorldPrimitive, ShadoWorldVisibilityBakeReport } from '../src/world';

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
});
