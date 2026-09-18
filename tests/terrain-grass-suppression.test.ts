import { terrainGrassSuppression } from '../src/world/terrain-grass';
import { compileCoverage } from '../src/world/grass-coverage';
import type { ShadoWorldPrimitive, ShadoWorldTerrainMaterialAuthoring } from '../src/world/types';

/** A terrain layer, with only the fields the suppression rule reads. */
const layer = (
  id: string,
  material: string,
  controlChannel?: string,
  controlAdds?: boolean,
) => ({
  id,
  name: id,
  enabled: true,
  material: `eltania-ground-v1/${material}`,
  projection: 'world-xz' as const,
  textureScale: 5,
  weight: 1,
  slope: [0, 1] as [number, number],
  altitude: [-1000, 1000] as [number, number],
  noiseScale: 0,
  metadata: {
    authoring: {
      ...(controlChannel ? { controlChannel } : {}),
      ...(controlAdds === false ? { controlAdds: false } : {}),
    },
  },
});

const terrain = (...layers: ReturnType<typeof layer>[]): ShadoWorldTerrainMaterialAuthoring =>
  ({ enabled: true, controlMaps: ['/a.png', '/b.png'], layers }) as never;

/** A 2x2 RGBA raster from four [r,g,b,a] texels. */
const raster = (texels: Array<[number, number, number, number]>) => ({
  width: 2,
  height: 2,
  data: new Uint8Array(texels.flat()),
});

describe('terrainGrassSuppression', () => {
  it('suppresses where a non-grass layer is painted and nowhere else', () => {
    const suppression = terrainGrassSuppression(
      terrain(
        layer('turf', 'grassLush', 'growth'),
        layer('track', 'earthPacked', 'path'),
        layer('plaza', 'stonePaving', 'paving'),
      ),
      [
        //            R path        G growth     B          A
        raster([
          [255, 200, 0, 0], // a fully worn track
          [0, 255, 0, 0], //   lush meadow, growth painted hard
          [90, 120, 0, 0], //  a faint track edge
          [0, 0, 0, 0], //     nothing painted at all
        ]),
        // control 1: R is paving
        raster([
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [255, 0, 0, 0],
        ]),
      ],
    );
    expect(suppression).toBeDefined();
    // The track is suppressed; the meadow is not, however hard growth is painted.
    expect([...suppression!.values]).toEqual([255, 0, 90, 255]);
  });

  it('ignores a layer painted where it should be ABSENT', () => {
    // `controlAdds: false` means "this layer shows where the channel is NOT
    // painted", so the painted texels say nothing about whether grass grows.
    const suppression = terrainGrassSuppression(
      terrain(layer('inverse', 'earthPacked', 'path', false)),
      [raster([[255, 0, 0, 0], [255, 0, 0, 0], [255, 0, 0, 0], [255, 0, 0, 0]]), raster([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])],
    );
    expect(suppression).toBeUndefined();
  });

  it('refuses control maps that disagree about resolution', () => {
    expect(() =>
      terrainGrassSuppression(terrain(layer('track', 'earthPacked', 'path')), [
        raster([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        { width: 4, height: 1, data: new Uint8Array(16) },
      ]),
    ).toThrow(/disagree about resolution/);
  });

  it('is undefined for terrain that is off, unpainted, or purely procedural', () => {
    expect(terrainGrassSuppression(undefined, [])).toBeUndefined();
    expect(terrainGrassSuppression(terrain(layer('track', 'earthPacked', 'path')), [])).toBeUndefined();
    // Every layer procedural: slope and altitude decide, and nothing is painted.
    expect(
      terrainGrassSuppression(terrain(layer('scree', 'scree')), [
        raster([[255, 0, 0, 0], [255, 0, 0, 0], [255, 0, 0, 0], [255, 0, 0, 0]]),
      ]),
    ).toBeUndefined();
  });
});

describe('grass coverage against a painted road', () => {
  /** One flat 40x40 grass-tagged quad at the origin, as a zone floor. */
  const floor = (): ShadoWorldPrimitive =>
    ({
      name: 'floor',
      extraShader: 'grass',
      positions: new Float32Array([0, 0, 0, 40, 0, 0, 40, 0, 40, 0, 0, 40]),
      normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    }) as never;

  const settings = { cellSize: 24, minimumUpNormal: 0.58 };

  it('clears exactly the half of the floor the mask calls road', () => {
    const before = compileCoverage([floor()], settings, []);
    const bare = [...before.values()].reduce(
      (total, cell) => total + cell.heights.filter(Number.isFinite).length,
      0,
    );
    expect(bare).toBeGreaterThan(0);

    // A 2x1 mask over the same 40x40 rectangle: the west half is road.
    const after = compileCoverage([floor()], settings, [], {
      width: 2,
      height: 1,
      values: new Uint8Array([255, 0]),
      worldMin: [0, 0],
      worldMax: [40, 40],
      threshold: 0.35,
    });
    const remaining = [...after.values()].reduce(
      (total, cell) => total + cell.heights.filter(Number.isFinite).length,
      0,
    );
    // Half the floor, give or take the texels straddling the boundary.
    expect(remaining).toBeGreaterThan(bare * 0.4);
    expect(remaining).toBeLessThan(bare * 0.6);
  });
});
