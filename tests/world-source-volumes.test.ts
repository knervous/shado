import {
  ShadoWorldVisibilityCoordinator,
  compileShadoWorldVisibility,
} from '../src/world';
import type { ShadoWorldPrimitive, ShadoWorldSpatialPackage } from '../src/world';

/** Axis-aligned quads, assembled by hand so the fixture owns its geometry. */
function surface(name: string, quads: number[][]): ShadoWorldPrimitive {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const corners of quads) {
    const base = positions.length / 3;
    positions.push(...corners);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return {
    name,
    material: 'stone',
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
  };
}

const LENGTH = 160;
const DEPTH = 16;
const REGION = 16;
const ROOF_Y = 20;

/**
 * A roofed room at one end of a street, open to the sky at the other.
 *
 * The room's floor and its roof stand in the same 2D column, which is the
 * whole point: one row per column has to serve a viewer inside the room and a
 * viewer standing on its roof, so it serves the roof.
 */
function roomAndStreet(roomEnd: number) {
  const quads: number[][] = [];
  for (let x = 0; x < LENGTH; x += 8) {
    quads.push([x, 0, 0, x + 8, 0, 0, x + 8, 0, DEPTH, x, 0, DEPTH]);
    if (x < roomEnd) {
      quads.push([x, ROOF_Y, 0, x + 8, ROOF_Y, 0, x + 8, ROOF_Y, DEPTH, x, ROOF_Y, DEPTH]);
    }
  }
  // The room's end wall, closing it off from the street.
  quads.push([roomEnd, 0, 0, roomEnd, 0, DEPTH, roomEnd, ROOF_Y, DEPTH, roomEnd, ROOF_Y, 0]);
  quads.push([roomEnd + 1, 0, 0, roomEnd + 1, ROOF_Y, 0, roomEnd + 1, ROOF_Y, DEPTH, roomEnd + 1, 0, DEPTH]);
  return surface('room-and-street', quads);
}


/**
 * Compiles an arbitrary hand-built scene on the same region grid, so each
 * fixture below differs only in its geometry.
 */
function compileScene(
  primitive: ShadoWorldPrimitive,
  options: {
    length?: number;
    top?: number;
    verticalVolumes?: boolean;
    cameraRowMargin?: number;
  } = {}
) {
  const length = options.length ?? LENGTH;
  const top = options.top ?? ROOF_Y;
  const centers: [number, number][] = [];
  for (let x = REGION / 2; x < length; x += REGION) centers.push([x, DEPTH / 2]);
  const cellBounds = centers.map(([x]) => ({
    min: [x - REGION / 2, 0, 0] as [number, number, number],
    max: [x + REGION / 2, 1, DEPTH] as [number, number, number],
  }));
  return compileShadoWorldVisibility({
    mode: 'sampled-occlusion',
    verticalVolumes: options.verticalVolumes ?? true,
    ...(options.cameraRowMargin === undefined
      ? {}
      : { cameraRowMargin: options.cameraRowMargin }),
    bounds: { min: [0, 0, 0], max: [length, top, DEPTH] },
    regionSize: REGION,
    maxDistance: 1024,
    renderCellCenters: centers,
    renderCellBounds: cellBounds,
    persistentRenderCells: new Uint8Array(centers.length),
    collisionPrimitives: [primitive],
  });
}

/** Every volume in a region, lowest band first. */
function volumesOf(
  visibility: NonNullable<ShadoWorldSpatialPackage['visibility']>,
  region: number
): number[] {
  const volumes = visibility.volumes!;
  return [...Array(volumes.count).keys()]
    .filter((volume) => volumes.region[volume] === region)
    .sort((left, right) => volumes.minY[left]! - volumes.minY[right]!);
}

/** A floor slab spanning x0..x1 at height y. */
function slab(x0: number, x1: number, y: number): number[] {
  return [x0, y, 0, x1, y, 0, x1, y, DEPTH, x0, y, DEPTH];
}

/** A cross-wall at x, from y0 to y1, over z0..z1. */
function wall(x: number, y0: number, y1: number, z0: number, z1: number): number[] {
  return [x, y0, z0, x, y0, z1, x, y1, z1, x, y1, z0];
}

function compile(verticalVolumes: boolean) {
  const scene = roomAndStreet(48);
  const centers: [number, number][] = [];
  for (let x = REGION / 2; x < LENGTH; x += REGION) centers.push([x, DEPTH / 2]);
  /*
   * Cell bounds matter here: without them targets are guessed at fixed
   * heights above "the floor", and indoors the floor is the ceiling -- so
   * every target ends up above the roof and the reverse test only ever asks
   * whether the street can see the roof. The real bake passes these.
   */
  const cellBounds = centers.map(([x]) => ({
    min: [x - REGION / 2, 0, 0] as [number, number, number],
    max: [x + REGION / 2, 1, DEPTH] as [number, number, number],
  }));
  return compileShadoWorldVisibility({
    mode: 'sampled-occlusion',
    verticalVolumes,
    bounds: { min: [0, 0, 0], max: [LENGTH, ROOF_Y, DEPTH] },
    regionSize: REGION,
    maxDistance: 1024,
    renderCellCenters: centers,
    renderCellBounds: cellBounds,
    persistentRenderCells: new Uint8Array(centers.length),
    collisionPrimitives: [scene],
  });
}

const bit = (
  visibility: NonNullable<ShadoWorldSpatialPackage['visibility']>,
  row: number,
  to: number
): boolean =>
  ((visibility.pvs.words[row * visibility.pvs.wordsPerRow + (to >>> 5)]! >>> 0) &
    (1 << (to & 31))) !== 0;

describe('vertical source volumes', () => {
  it('leaves a package without them exactly as it was', () => {
    const flat = compile(false);
    expect(flat.volumes).toBeUndefined();
    const regions = flat.width * flat.height;
    expect(flat.pvs.words.length).toBe(regions * flat.pvs.wordsPerRow);
  });

  it('splits a roofed column into a room volume and a roof volume', () => {
    const split = compile(true);
    expect(split.volumes).toBeDefined();
    const volumes = split.volumes!;
    // Region 0 is inside the room, under its roof.
    const inRegionZero = [...Array(volumes.count).keys()].filter(
      (volume) => volumes.region[volume] === 0
    );
    expect(inRegionZero.length).toBe(2);
    const bands = inRegionZero
      .map((volume) => [volumes.minY[volume]!, volumes.maxY[volume]!])
      .sort((left, right) => left[0]! - right[0]!);
    /*
     * The bands tile the column and do not overlap: the room runs from just
     * under its floor to just under the roof, and the roof runs from just
     * under itself to the sky. Overlapping them would let the neighbour flood
     * union one into the other and undo the split.
     */
    expect(bands[0]![0]).toBeCloseTo(-0.5, 5);
    expect(bands[0]![1]).toBeCloseTo(ROOF_Y - 0.5, 5);
    expect(bands[1]![0]).toBeCloseTo(ROOF_Y - 0.5, 5);
    expect(bands[1]![1]).toBe(Number.POSITIVE_INFINITY);
    expect(bands[0]![1]).toBe(bands[1]![0]);
  });

  it('hides the street from inside the room while the roof still sees it', () => {
    /*
     * The measurement this whole thing exists for. With one row per column
     * the room and its roof share a row, so the row is a rooftop row and the
     * far street is admitted from inside. Split, the room's own row can
     * reject it -- without taking that view away from the roof, which really
     * does see it.
     */
    const split = compile(true);
    const volumes = split.volumes!;
    const roomVolume = [...Array(volumes.count).keys()].find(
      (volume) => volumes.region[volume] === 0 && volumes.maxY[volume]! <= ROOF_Y
    )!;
    const roofVolume = [...Array(volumes.count).keys()].find(
      (volume) => volumes.region[volume] === 0 && volumes.maxY[volume]! > ROOF_Y
    )!;
    const farStreet = 9;

    expect(bit(split, roomVolume, farStreet)).toBe(false);
    expect(bit(split, roofVolume, farStreet)).toBe(true);

    // And the flat package admits it from the column, as measured before.
    const flat = compile(false);
    expect(bit(flat, 0, farStreet)).toBe(true);
  });

  it('gives a camera at an unsupported height the union of its column', () => {
    const split = compile(true);
    const volumes = split.volumes!;
    const unionRow = volumes.count + 0;
    const farStreet = 9;
    /*
     * Between bands, or flying above everything: no volume contains the
     * camera, so it gets everything the column admits. That costs draw calls
     * and cannot hide anything, which is the only safe answer.
     */
    expect(bit(split, unionRow, farStreet)).toBe(true);
  });

  it('locates the camera by height, and falls back rather than guessing', async () => {
    const split = compile(true);
    const world = {
      ...({} as ShadoWorldSpatialPackage),
      visibility: split,
      tiles: { x: [], z: [], size: REGION, originX: 0, originZ: 0 },
    } as unknown as ShadoWorldSpatialPackage;
    const coordinator = Object.create(
      ShadoWorldVisibilityCoordinator.prototype
    ) as ShadoWorldVisibilityCoordinator;
    Object.defineProperty(coordinator, 'world', { value: world });

    const volumes = split.volumes!;
    const roomVolume = [...Array(volumes.count).keys()].find(
      (volume) => volumes.region[volume] === 0 && volumes.maxY[volume]! <= ROOF_Y
    )!;
    const roofVolume = [...Array(volumes.count).keys()].find(
      (volume) => volumes.region[volume] === 0 && volumes.maxY[volume]! > ROOF_Y
    )!;

    // Standing in the room, standing on the roof, and far above everything.
    expect(coordinator.locateSourceRow(8, 8, 8, 0)).toBe(roomVolume);
    expect(coordinator.locateSourceRow(8, ROOF_Y + 8, 8, 0)).toBe(roofVolume);
    expect(coordinator.locateSourceRow(8, -500, 8, 0)).toBe(volumes.count + 0);
  });

  it('sees through a doorway, and does not see through the wall around it', () => {
    /*
     * Two roofed rooms sharing a wall. The aperture is baked OPEN: a door is
     * a hole in the geometry, and a bake that sealed it would hide a room a
     * player can walk straight into. The sealed variant is the control -- it
     * is the only thing that proves the open one is not simply admitting
     * everything.
     */
    const build = (door: boolean) => {
      const quads: number[][] = [];
      for (let x = 0; x < 128; x += 8) {
        quads.push(slab(x, x + 8, 0));
        quads.push(slab(x, x + 8, ROOF_Y));
      }
      if (door) {
        // Wall either side of the opening, and the lintel above it.
        quads.push(wall(64, 0, ROOF_Y, 0, 6));
        quads.push(wall(64, 0, ROOF_Y, 10, DEPTH));
        quads.push(wall(64, 12, ROOF_Y, 6, 10));
      } else {
        quads.push(wall(64, 0, ROOF_Y, 0, DEPTH));
      }
      /*
       * Margin 0 on purpose. The camera flood unions the row on the far side
       * of the wall into this one, because in general a player crosses a
       * region edge -- they cannot cross THIS edge, but the flood does not
       * know that, and with it on, the sealed control admits the far room and
       * proves nothing. Room connectivity is what would let the flood know;
       * until then this pair measures the bake's own rejection.
       */
      return compileScene(surface(door ? 'doorway' : 'sealed', quads), {
        length: 128,
        cameraRowMargin: 0,
      });
    };

    const open = build(true);
    const sealed = build(false);
    // Region 3 is up against the wall on one side, region 4 on the other.
    const near = volumesOf(open, 3).find((volume) => open.volumes!.maxY[volume]! <= ROOF_Y)!;
    const sealedNear = volumesOf(sealed, 3).find(
      (volume) => sealed.volumes!.maxY[volume]! <= ROOF_Y
    )!;

    expect(bit(open, near, 6)).toBe(true);
    expect(bit(sealed, sealedNear, 6)).toBe(false);
  });

  it('keeps a doorway crossing continuous across the region edge', () => {
    /*
     * The row a player answers from changes as they cross a region edge, and
     * the camera flood is what stops the far room popping in at the moment it
     * does. Every room-height row along the approach must already admit the
     * far room before the player reaches the door.
     */
    const quads: number[][] = [];
    for (let x = 0; x < 128; x += 8) {
      quads.push(slab(x, x + 8, 0));
      quads.push(slab(x, x + 8, ROOF_Y));
    }
    quads.push(wall(64, 0, ROOF_Y, 0, 6));
    quads.push(wall(64, 0, ROOF_Y, 10, DEPTH));
    quads.push(wall(64, 12, ROOF_Y, 6, 10));
    const open = compileScene(surface('doorway', quads), { length: 128 });

    for (const region of [3, 4]) {
      for (const volume of volumesOf(open, region)) {
        if (open.volumes!.maxY[volume]! > ROOF_Y) continue;
        expect(bit(open, volume, 5)).toBe(true);
      }
    }
  });

  it('separates stacked floors into their own volumes', () => {
    /*
     * A two-storey building: ground floor, upper floor, roof. Three places to
     * stand in one column, and the ground floor must not inherit what the
     * upper floor can see out of the far end.
     */
    const UPPER = 20;
    const TOP = 40;
    const quads: number[][] = [];
    for (let x = 0; x < LENGTH; x += 8) {
      quads.push(slab(x, x + 8, 0));
      if (x < 48) {
        quads.push(slab(x, x + 8, UPPER));
        quads.push(slab(x, x + 8, TOP));
      }
    }
    quads.push(wall(48, 0, UPPER, 0, DEPTH));
    const split = compileScene(surface('two-storey', quads), { top: TOP });

    const bands = volumesOf(split, 0).map((volume) => [
      split.volumes!.minY[volume]!,
      split.volumes!.maxY[volume]!,
    ]);
    expect(bands.length).toBe(3);
    // Tiling, still: each band starts exactly where the one below ends.
    expect(bands[0]![1]).toBe(bands[1]![0]);
    expect(bands[1]![1]).toBe(bands[2]![0]);
    expect(bands[2]![1]).toBe(Number.POSITIVE_INFINITY);

    // The ground floor is walled off from the far street; the roof is not.
    const ground = volumesOf(split, 0)[0]!;
    const roof = volumesOf(split, 0)[2]!;
    expect(bit(split, ground, 9)).toBe(false);
    expect(bit(split, roof, 9)).toBe(true);
  });

  it('keeps a camera under a bridge looking along the street', () => {
    /*
     * A deck over an open street is not a room. Splitting the column gives a
     * band under the deck and a band on it, and the one underneath must still
     * see along the street it stands in -- the split may not turn a bridge
     * into a sealed box.
     */
    const DECK = 20;
    const quads: number[][] = [];
    for (let x = 0; x < LENGTH; x += 8) {
      quads.push(slab(x, x + 8, 0));
      if (x >= 32 && x < 64) quads.push(slab(x, x + 8, DECK));
    }
    const split = compileScene(surface('bridge', quads), { top: DECK + 8 });

    // Region 2 and region 3 are both under the deck.
    const under = volumesOf(split, 2).find(
      (volume) => split.volumes!.maxY[volume]! <= DECK
    )!;
    expect(bit(split, under, 3)).toBe(true);
    expect(bit(split, under, 2)).toBe(true);
  });

  it('answers conservatively for a camera outside every supported volume', () => {
    const split = compile(true);
    const volumes = split.volumes!;
    const world = {
      ...({} as ShadoWorldSpatialPackage),
      visibility: split,
      tiles: { x: [], z: [], size: REGION, originX: 0, originZ: 0 },
    } as unknown as ShadoWorldSpatialPackage;
    const coordinator = Object.create(
      ShadoWorldVisibilityCoordinator.prototype
    ) as ShadoWorldVisibilityCoordinator;
    Object.defineProperty(coordinator, 'world', { value: world });

    /*
     * Below the world, above it, and in a region the fixture never gave a
     * floor: each one must land on a row, and that row must be the column's
     * union -- never nothing, and never a neighbour's room.
     */
    expect(coordinator.locateSourceRow(8, -1e6, 8, 0)).toBe(volumes.count + 0);
    expect(coordinator.locateSourceRow(8, 1e6, 8, 0)).toBeGreaterThanOrEqual(0);

    // Every row in the package admits the region it is for, union rows too.
    for (let row = 0; row < volumes.count + split.width * split.height; row += 1) {
      const region = row < volumes.count ? volumes.region[row]! : row - volumes.count;
      expect(bit(split, row, region)).toBe(true);
    }
  });
});
