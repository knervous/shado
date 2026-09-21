import { gunzipSync, gzipSync } from 'node:zlib';
import {
  ShadoWorldVisibilityCoordinator,
  compileShadoWorld,
  computeShadoWorldLayoutHash,
  stampShadoWorldIntegrity,
  validateShadoWorldPackage,
} from '../src/world';
import type { ShadoWorldPrimitive, ShadoWorldSpatialPackage } from '../src/world';

const DEPTH = 16;
const REGION = 16;

function surface(name: string, quads: number[][]): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const corners of quads) {
    const base = positions.length / 3;
    positions.push(...corners);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { name, material: 'stone', positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}
const slab = (x0: number, x1: number, y: number) => [x0, y, 0, x1, y, 0, x1, y, DEPTH, x0, y, DEPTH];
const wall = (x: number, y0: number, y1: number) => [x, y0, 0, x, y0, DEPTH, x, y1, DEPTH, x, y1, 0];

/**
 * Every kind of column the wire has to carry: a roofed room, an open street,
 * a floor BELOW zero, and a gap with no floor at all.
 */
function scene(): ShadoWorldPrimitive {
  const quads: number[][] = [];
  for (let x = 0; x < 160; x += 8) {
    if (x >= 96 && x < 112) continue; // a floorless gap
    const floor = x >= 128 ? -12 : 0; // a sunken court, below zero
    quads.push(slab(x, x + 8, floor));
    if (x < 48) quads.push(slab(x, x + 8, 20)); // the room's roof
  }
  quads.push(wall(48, 0, 20));
  return surface('wire-scene', quads);
}

function compile(extra: Record<string, unknown> = {}) {
  const world = compileShadoWorld([scene()], {
    name: 'volume-wire',
    tileSize: REGION,
    visibilityRegionSize: REGION,
    maxClusterTriangles: 2,
    visibilityMode: 'sampled-occlusion',
    visibilityVerticalVolumes: true,
    ...extra,
  });
  stampShadoWorldIntegrity(world);
  return world;
}

/** Exactly what the bake writes and exactly what a reader loads. */
function roundTrip(world: ShadoWorldSpatialPackage): ShadoWorldSpatialPackage {
  return JSON.parse(gunzipSync(gzipSync(JSON.stringify(world))).toString());
}

function rowSelector(world: ShadoWorldSpatialPackage) {
  const coordinator = Object.create(ShadoWorldVisibilityCoordinator.prototype) as ShadoWorldVisibilityCoordinator;
  Object.defineProperty(coordinator, 'world', { value: world });
  return (x: number, y: number, z: number) => coordinator.locateSourceRow(x, y, z);
}

/** Cameras in every kind of column, at every interesting height. */
const CAMERAS: [number, number, number][] = [];
for (const x of [8, 24, 40, 72, 104, 136, 152]) {
  for (const y of [-40, -12, -4, 0, 8, 19.5, 28, 50, 500]) CAMERAS.push([x, y, 8]);
}

describe('v2 volumes survive the wire', () => {
  it('writes version 2 with finite bands, and nothing JSON cannot carry', () => {
    const world = compile();
    const visibility = world.visibility!;
    expect(visibility.version).toBe(2);
    expect(Number.isFinite(visibility.sourceDomain!.minY)).toBe(true);
    expect(Number.isFinite(visibility.sourceDomain!.maxY)).toBe(true);
    for (const value of [...visibility.volumes!.minY, ...visibility.volumes!.maxY]) {
      expect(Number.isFinite(value)).toBe(true);
    }
    // Nothing became null on the way through.
    const wire = roundTrip(world).visibility!;
    expect([...wire.volumes!.minY, ...wire.volumes!.maxY].every((value) => typeof value === 'number')).toBe(true);
  });

  it('validates as the file a reader loads, not only as the object in memory', () => {
    const loaded = roundTrip(compile());
    expect(() => validateShadoWorldPackage(loaded)).not.toThrow();
  });

  it('selects the same row for every camera before and after the round trip', () => {
    const world = compile();
    const before = rowSelector(world);
    const after = rowSelector(roundTrip(world));
    for (const [x, y, z] of CAMERAS) expect(after(x, y, z)).toBe(before(x, y, z));
  });

  it('gives the same candidates through the reducer before and after', async () => {
    const world = compile();
    const loaded = roundTrip(world);
    const planes = new Float32Array([1, 0, 0, 1e5, -1, 0, 0, 1e5, 0, 1, 0, 1e5, 0, -1, 0, 1e5, 0, 0, 1, 1e5, 0, 0, -1, 1e5]);
    const one = await ShadoWorldVisibilityCoordinator.create(world);
    const two = await ShadoWorldVisibilityCoordinator.create(loaded);
    for (const camera of CAMERAS) {
      const a = one.reduceWorld(planes, camera);
      const b = two.reduceWorld(planes, camera);
      expect(Array.from(b.visibleClusters)).toEqual(Array.from(a.visibleClusters));
    }
  });

  it('answers from the union row outside the domain, on a band edge, and across an envelope', () => {
    const world = compile();
    const visibility = world.visibility!;
    const select = rowSelector(world);
    const volumes = visibility.volumes!;
    const union = (region: number) => volumes.count + region;
    // Region 0 is the roofed room: two bands meeting at 19.5.
    expect(select(8, 500, 8)).toBe(union(0));
    expect(select(8, visibility.sourceDomain!.minY - 1, 8)).toBe(union(0));
    expect(select(8, 19.5, 8)).toBe(union(0));
    const room = select(8, 8, 8);
    expect(room).toBeLessThan(volumes.count);
    // An envelope reaching the roof band is two bands, so it is the union.
    const coordinator = Object.create(ShadoWorldVisibilityCoordinator.prototype) as ShadoWorldVisibilityCoordinator;
    Object.defineProperty(coordinator, 'world', { value: world });
    expect(coordinator.locateSourceRow(8, 8, 8, undefined, 20)).toBe(union(0));
    expect(coordinator.locateSourceRow(8, 8, 8, undefined, 1)).toBe(room);
  });

  it('carries a floor below zero and a column with no floor', () => {
    const world = roundTrip(compile());
    const visibility = world.visibility!;
    const volumes = visibility.volumes!;
    const select = rowSelector(world);
    // The sunken court's own band holds a camera standing on it.
    const sunken = select(136, -4, 8);
    expect(sunken).toBeLessThan(volumes.count);
    expect(volumes.minY[sunken]!).toBeLessThan(0);
    // The gap still owns a volume, and it spans the domain.
    const gapRegion = 6;
    const first = volumes.regionOffset[gapRegion]!;
    expect(volumes.regionOffset[gapRegion + 1]! - first).toBeGreaterThanOrEqual(1);
  });

  it('changes the hash when only a bound moves', () => {
    const world = compile();
    const before = computeShadoWorldLayoutHash(world);
    world.visibility!.volumes!.minY[0] = world.visibility!.volumes!.minY[0]! + 5;
    expect(computeShadoWorldLayoutHash(world)).not.toBe(before);
  });

  it('makes a higher camera extent part of the identity', () => {
    const low = compile();
    const high = compile({ visibilityCameraExtent: { minY: -40, maxY: 400 } });
    expect(high.visibility!.sourceDomain).toEqual({ minY: -40, maxY: 400 });
    expect(computeShadoWorldLayoutHash(high)).not.toBe(computeShadoWorldLayoutHash(low));
  });

  it('refuses what JSON would have made of an Infinity', () => {
    const loaded = roundTrip(compile());
    (loaded.visibility!.volumes!.maxY as unknown as (number | null)[])[0] = null;
    expect(() => validateShadoWorldPackage(loaded)).toThrow(/non-finite band/);
  });

  it('refuses bands that overlap, leave a gap, or are out of order', () => {
    const overlap = roundTrip(compile());
    const v = overlap.visibility!.volumes!;
    // Region 0 has two bands; push the second down into the first.
    v.minY[1] = v.minY[1]! - 1;
    expect(() => validateShadoWorldPackage(overlap)).toThrow(/overlaps or leaves a gap/);

    const unsorted = roundTrip(compile());
    const u = unsorted.visibility!.volumes!;
    const last = u.count - 1;
    [u.region[0], u.region[last]] = [u.region[last]!, u.region[0]!];
    expect(() => validateShadoWorldPackage(unsorted)).toThrow();
  });

  it('refuses a version it does not know, and v1 carrying volumes', () => {
    const future = roundTrip(compile());
    (future.visibility as { version: number }).version = 3;
    expect(() => validateShadoWorldPackage(future)).toThrow(/Unsupported Shado world visibility version 3/);

    const mislabelled = roundTrip(compile());
    (mislabelled.visibility as { version: number }).version = 1;
    expect(() => validateShadoWorldPackage(mislabelled)).toThrow(/v1 cannot carry/);
  });

  it('reads a v1 package exactly as before', () => {
    const flat = compileShadoWorld([scene()], {
      name: 'volume-wire-v1',
      tileSize: REGION,
      visibilityRegionSize: REGION,
      maxClusterTriangles: 2,
      visibilityMode: 'sampled-occlusion',
    });
    stampShadoWorldIntegrity(flat);
    const loaded = roundTrip(flat);
    expect(loaded.visibility!.version).toBe(1);
    expect(loaded.visibility!.volumes).toBeUndefined();
    expect(() => validateShadoWorldPackage(loaded)).not.toThrow();
    const select = rowSelector(loaded);
    // Rows are regions, whatever the height.
    expect(select(8, 8, 8)).toBe(0);
    expect(select(8, 500, 8)).toBe(0);
  });
});

describe('a render cell past the geometry bounds', () => {
  it('gets a region instead of -1', () => {
    /*
     * Reported by the disocclusion prototype: geometry ending exactly on a
     * region edge -- a +X box face at x = 64 with 8-unit regions and tiles --
     * starts a render tile at 64 whose centre, 68, is past the bounds. The
     * grid was sized from the bounds alone and that cell got region -1,
     * which validation refused.
     */
    const face = surface('edge', [[64, 0, 0, 64, 0, 8, 64, 8, 8, 64, 8, 0], slab(0, 64, 0)]);
    const world = compileShadoWorld([face], {
      name: 'edge-cell',
      tileSize: 8,
      visibilityRegionSize: 8,
      maxClusterTriangles: 2,
    });
    stampShadoWorldIntegrity(world);
    expect(world.visibility!.cellRegion.every((region) => region >= 0)).toBe(true);
    expect(() => validateShadoWorldPackage(world)).not.toThrow();
  });
});
