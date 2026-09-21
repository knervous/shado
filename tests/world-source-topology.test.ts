import { ShadoWorldVisibilityCoordinator, compileShadoWorldVisibility } from '../src/world';
import type {
  ShadoWorldPrimitive,
  ShadoWorldSourceTopology,
  ShadoWorldSpatialPackage,
} from '../src/world';

const DEPTH = 16;
const REGION = 16;
const ROOF = 20;
const LENGTH = 160;

function surface(quads: number[][]): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const corners of quads) {
    const base = positions.length / 3;
    positions.push(...corners);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { name: 'rooms', material: 'stone', positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}
const slab = (x0: number, x1: number, y: number) => [x0, y, 0, x1, y, 0, x1, y, DEPTH, x0, y, DEPTH];
const wall = (x: number, y0: number, y1: number, z0: number, z1: number) => [x, y0, z0, x, y0, z1, x, y1, z1, x, y1, z0];

/**
 * Two long roofed rooms either side of a wall at x = 64: room A is x 0..64,
 * room B is x 64..160. With `door`, a doorway in the wall; without, sealed.
 */
function twoRooms(door: boolean): ShadoWorldPrimitive {
  const quads: number[][] = [];
  for (let x = 0; x < LENGTH; x += 8) {
    quads.push(slab(x, x + 8, 0));
    quads.push(slab(x, x + 8, ROOF));
  }
  if (door) {
    quads.push(wall(64, 0, ROOF, 0, 6));
    quads.push(wall(64, 0, ROOF, 10, DEPTH));
    quads.push(wall(64, 12, ROOF, 6, 10));
  } else {
    quads.push(wall(64, 0, ROOF, 0, DEPTH));
  }
  return surface(quads);
}

/** The authored contract for that fixture: two rooms, optionally a portal. */
function topology(portal: boolean, roomA: [number, number] = [0, 64]): ShadoWorldSourceTopology {
  return {
    rooms: [
      { id: 'hall-a', min: [roomA[0], -1, 0], max: [roomA[1], ROOF, DEPTH] },
      { id: 'hall-b', min: [64, -1, 0], max: [LENGTH, ROOF, DEPTH] },
    ],
    portals: portal ? [{ a: 'hall-a', b: 'hall-b' }] : [],
  };
}

function bake(scene: ShadoWorldPrimitive, sourceTopology?: ShadoWorldSourceTopology) {
  const centers: [number, number][] = [];
  for (let x = REGION / 2; x < LENGTH; x += REGION) centers.push([x, DEPTH / 2]);
  return compileShadoWorldVisibility({
    mode: 'sampled-occlusion',
    verticalVolumes: true,
    // The default margin, always. Margin 0 is a diagnostic bound only.
    bounds: { min: [0, 0, 0], max: [LENGTH, ROOF, DEPTH] },
    regionSize: REGION,
    maxDistance: 1024,
    renderCellCenters: centers,
    renderCellBounds: centers.map(([x]) => ({
      min: [x - REGION / 2, 0, 0] as [number, number, number],
      max: [x + REGION / 2, 1, DEPTH] as [number, number, number],
    })),
    persistentRenderCells: new Uint8Array(centers.length),
    collisionPrimitives: [scene],
    ...(sourceTopology ? { sourceTopology } : {}),
  });
}

type Visibility = NonNullable<ShadoWorldSpatialPackage['visibility']>;
const bit = (v: Visibility, row: number, to: number) =>
  ((v.pvs.words[row * v.pvs.wordsPerRow + (to >>> 5)]! >>> 0) & (1 << (to & 31))) !== 0;

/** The room-height volume in a region. */
function roomVolume(v: Visibility, region: number): number {
  const volumes = v.volumes!;
  for (let volume = volumes.regionOffset[region]!; volume < volumes.regionOffset[region + 1]!; volume += 1) {
    if (volumes.maxY[volume]! <= ROOF) return volume;
  }
  throw new Error(`region ${region} has no room band`);
}

function selector(v: Visibility) {
  const coordinator = Object.create(ShadoWorldVisibilityCoordinator.prototype) as ShadoWorldVisibilityCoordinator;
  Object.defineProperty(coordinator, 'world', { value: { visibility: v } });
  return coordinator;
}

// Region 3 is room A against the wall; region 8 is deep inside room B --
// beyond the forced-local radius, so only the flood can put it in A's row.
const NEAR_WALL = 3;
const DEEP_IN_B = 8;

describe('the flood withholds only across certified boundaries', () => {
  it('keeps the flood with no topology at all', () => {
    const sealed = bake(twoRooms(false));
    /*
     * Unknown adjacency is connected. Region 4, just across the wall, sees
     * deep into B, and region 3's row inherits it through the margin-1 flood.
     */
    expect(bit(sealed, roomVolume(sealed, NEAR_WALL), DEEP_IN_B)).toBe(true);
  });

  it('withholds a neighbour across an authored wall with no portal', () => {
    const sealed = bake(twoRooms(false), topology(false));
    expect(bit(sealed, roomVolume(sealed, NEAR_WALL), DEEP_IN_B)).toBe(false);
  });

  it('keeps it when the rooms are joined by a portal, baked open', () => {
    const joined = bake(twoRooms(true), topology(true));
    expect(bit(joined, roomVolume(joined, NEAR_WALL), DEEP_IN_B)).toBe(true);
  });

  it('keeps it when a room box only partly covers a column', () => {
    /*
     * Room A authored as x 0..60 -- it clips region 3 (48..64). A volume
     * only partly inside a room is unknown, and unknown stays connected.
     */
    const clipped = bake(twoRooms(false), topology(false, [0, 60]));
    expect(bit(clipped, roomVolume(clipped, NEAR_WALL), DEEP_IN_B)).toBe(true);
  });

  it('keeps floorless space connected; a missed floor sample is not a certificate', () => {
    // Remove the floor under region 4 only. Its volume has no eyes -- and
    // that must not stop it donating to region 3, as it once did.
    const quads: number[][] = [];
    for (let x = 0; x < LENGTH; x += 8) {
      if (x < 64 || x >= 80) quads.push(slab(x, x + 8, 0));
      quads.push(slab(x, x + 8, ROOF));
    }
    const gap = bake(surface(quads));
    const floorless = gap.volumes!;
    expect(floorless.regionOffset[5]! - floorless.regionOffset[4]!).toBeGreaterThanOrEqual(1);
    expect(bit(gap, roomVolume(gap, NEAR_WALL), DEEP_IN_B)).toBe(true);
  });

  it('refuses a portal naming a room that was never authored', () => {
    expect(() =>
      bake(twoRooms(true), { rooms: topology(false).rooms, portals: [{ a: 'hall-a', b: 'hall-z' }] })
    ).toThrow(/unknown room/);
  });
});

describe('cameras the certificate has to survive', () => {
  const sealed = bake(twoRooms(false), topology(false));
  const joined = bake(twoRooms(true), topology(true));

  it('answers a camera offset through the doorway from the room it is in', () => {
    /*
     * A third-person camera swung through the doorway into B is IN B. Rows
     * are chosen by where the camera is, so it answers from B's volume and
     * sees what B sees -- the player standing in A does not have to.
     */
    const row = selector(joined).locateSourceRow(72, 8, 8);
    expect(joined.volumes!.region[row]).toBe(4);
    expect(bit(joined, row, DEEP_IN_B)).toBe(true);
  });

  it('answers a wall-adjacent camera pushed through the wall from the far side', () => {
    const row = selector(sealed).locateSourceRow(65, 8, 8);
    expect(sealed.volumes!.region[row]).toBe(4);
    expect(bit(sealed, row, DEEP_IN_B)).toBe(true);
  });

  it('takes the union row on a stair landing, between two bands', () => {
    const coordinator = selector(sealed);
    const v = sealed.volumes!;
    const edge = v.maxY[roomVolume(sealed, NEAR_WALL)]!;
    expect(coordinator.locateSourceRow(56, edge, 8)).toBe(v.count + NEAR_WALL);
  });

  it('takes the union row in unsupported flight', () => {
    const coordinator = selector(sealed);
    expect(coordinator.locateSourceRow(56, 10_000, 8)).toBe(sealed.volumes!.count + NEAR_WALL);
  });
});
