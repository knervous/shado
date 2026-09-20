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
});
