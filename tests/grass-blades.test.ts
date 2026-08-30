import { NullEngine } from '@babylonjs/core';
import {
  ShadoFoliageActor,
  ShadoFoliageContainer,
} from '../src/extensions/ShadoFoliageContainer/ShadoFoliageContainer';
import {
  GRASS_FIELD_TEXELS_PER_CELL,
  packShadoGrassFieldData,
  shadoGrassBlades,
} from '../src/extensions/ShadoFoliageContainer/grass-blades';
import { compileShadoWorldGrassField } from '../src/world/grass-field';
import type { ShadoWorldPrimitive } from '../src/world/types';

/** A grass-tagged surface with real relief, so cell borders have a step to hide. */
function rollingGround(size = 120, divisions = 40): ShadoWorldPrimitive {
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  for (let row = 0; row <= divisions; row++) {
    for (let column = 0; column <= divisions; column++) {
      const x = (column / divisions) * size;
      const z = (row / divisions) * size;
      positions.push(x, 6 * Math.sin(x * 0.07) + 4 * Math.cos(z * 0.05), z);
      normals.push(0, 1, 0);
      uvs.push(column / divisions, row / divisions);
    }
  }
  for (let row = 0; row < divisions; row++) {
    for (let column = 0; column < divisions; column++) {
      const a = row * (divisions + 1) + column;
      indices.push(a, a + 1, a + divisions + 2, a, a + divisions + 2, a + divisions + 1);
    }
  }
  return {
    name: 'rolling-grass-ground',
    material: 'grass1',
    extraShader: 'grass',
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    indices: new Uint32Array(indices),
  } as unknown as ShadoWorldPrimitive;
}

function grassGround(size = 96): ShadoWorldPrimitive {
  return {
    name: 'grass-ground',
    material: 'grass1',
    extraShader: 'grass',
    positions: new Float32Array([0, 2, 0, size, 2, 0, size, 2, size, 0, 2, size]),
    normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  } as unknown as ShadoWorldPrimitive;
}

const CONFIG = {
  cellSize: 24,
  coverageResolution: 32,
  heightResolution: 8,
  bladesPerCell: 512,
  minHeight: 0.32,
  maxHeight: 0.68,
  bladeWidth: 0.09,
};

describe('shadoGrassBlades', () => {
  it('compiles into the foliage container ahead of the displacing plugins', async () => {
    const engine = new NullEngine();
    await ShadoFoliageContainer.initialize(engine, { wasm: false, extra: ShadoFoliageActor });
    const container = new ShadoFoliageContainer(engine);
    const { materialUniforms } = container.configureFoliage({
      sourceHeight: 1,
      plugins: [
        shadoGrassBlades(CONFIG),
        { plugin: 'wind' },
        { plugin: 'tint', variationColor: [0.5, 0.7, 0.3] },
      ],
    });
    const glsl = container.generateGLSLPair();
    const wgsl = container.generateWGSLPair();

    // Blades set the world position absolutely, so they must run before any
    // plugin that displaces it, or the wind is overwritten.
    expect(glsl.vs.indexOf('shadoFoliageWorld = root')).toBeLessThan(
      glsl.vs.indexOf('uShadoFoliageWind.z')
    );
    expect(wgsl.vs.indexOf('shadoFoliageWorld = root')).toBeLessThan(
      wgsl.vs.indexOf('uniforms.uShadoFoliageWind.z')
    );

    // Wind and tint must read the per-blade locals the plugin overwrote, not
    // the per-cell actor header, or a whole cell sways as one object.
    expect(glsl.vs).toContain('shadoFoliagePhase * 6.2831853');
    expect(glsl.vs).toContain('shadoFoliagePhase = randomU');
    expect(glsl.vs).not.toContain('inst.foliageParams.x * 6.2831853');

    for (const name of materialUniforms) {
      expect(wgsl.vs).toMatch(new RegExp(`uniform ${name}: (f32|vec3f|vec4f);`));
    }
    expect(glsl.vs).toContain('uniform highp sampler2D uShadoGrassField;');
    expect(wgsl.vs).toContain('var uShadoGrassField: texture_2d<f32>;');

    container.dispose();
    engine.dispose();
  });

  it('validates its authored parameters', () => {
    expect(() => shadoGrassBlades({ ...CONFIG, cellSize: 0 })).toThrow(/cellSize must be positive/);
    expect(() => shadoGrassBlades({ ...CONFIG, bladesPerCell: 0 })).toThrow(
      /bladesPerCell must be positive/
    );
    expect(() => shadoGrassBlades({ ...CONFIG, minHeight: 2, maxHeight: 1 })).toThrow(
      /minHeight must be positive and not exceed maxHeight/
    );
  });
});

describe('createShadoGrassPatch firstBlade', () => {
  it('bakes the offset so a far ring draws a later slice of the sequence', async () => {
    const { NullEngine, Scene } = await import('@babylonjs/core');
    const engine = new NullEngine();
    const scene = new Scene(engine);
    const { createShadoGrassPatch } = await import(
      '../src/extensions/ShadoFoliageContainer/grass-blades'
    );
    const patch = createShadoGrassPatch(scene, 'far', 4, 1, 12288);
    const bladeData = patch.getVerticesData('aGrassBlade')!;
    // First component of every vertex is the global blade index.
    expect(bladeData[0]).toBe(12288);
    expect(bladeData[bladeData.length - 2]).toBe(12288 + 3);
    engine.dispose();
  });
});

describe('packShadoGrassFieldData', () => {
  const field = compileShadoWorldGrassField([grassGround()], { density: 8 })!;

  it('round-trips coverage bits through 16-bit float halves', () => {
    const cell = 0;
    const data = packShadoGrassFieldData(field, [cell]);
    const wordsPerCell = field.coverage.wordsPerCell;
    for (let word = 0; word < wordsPerCell; word++) {
      const expected = field.coverage.words[cell * wordsPerCell + word]! >>> 0;
      // A float holds 24 bits exactly, so a 32-bit word must arrive as halves.
      const low = data[word * 2]!;
      const high = data[word * 2 + 1]!;
      expect(low).toBeLessThanOrEqual(0xffff);
      expect(high).toBeLessThanOrEqual(0xffff);
      expect(((high << 16) >>> 0) + low).toBe(expected);
    }
  });

  it('packs only the resident cells, in the order given', () => {
    const window = [3, 1];
    const data = packShadoGrassFieldData(field, window);
    expect(data.length).toBe(window.length * GRASS_FIELD_TEXELS_PER_CELL * 4);
    const wordsPerCell = field.coverage.wordsPerCell;
    // Row 0 must be cell 3, not cell 0 — the texture is indexed by residency
    // slot, which is what the actor's field row points at.
    expect(data[0]).toBe(field.coverage.words[3 * wordsPerCell]! & 0xffff);
  });

  it('stays one row per cell however dense the grass is', () => {
    const dense = compileShadoWorldGrassField([grassGround()], { density: 4000 })!;
    expect(packShadoGrassFieldData(dense, [0, 1, 2]).length).toBe(
      packShadoGrassFieldData(field, [0, 1, 2]).length
    );
  });
});


/**
 * The shader's ground lookup, in TypeScript.
 *
 * Kept deliberately literal so it fails if the packing layout and the shader's
 * indexing drift apart.
 */
function sampleGround(
  data: Float32Array,
  resolution: number,
  row: number,
  u: number,
  v: number
): number {
  const extended = resolution + 2;
  const stride = GRASS_FIELD_TEXELS_PER_CELL * 4;
  const heightBase = row * stride + 16 * 4;
  const at = (x: number, z: number) => data[heightBase + z * extended + x]!;
  const sampleU = u * resolution + 0.5;
  const sampleV = v * resolution + 0.5;
  const x0 = Math.floor(sampleU);
  const z0 = Math.floor(sampleV);
  const fx = sampleU - x0;
  const fz = sampleV - z0;
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  return lerp(
    lerp(at(x0, z0), at(x0 + 1, z0), fx),
    lerp(at(x0, z0 + 1), at(x0 + 1, z0 + 1), fx),
    fz
  );
}

describe('grass ground continuity across cell borders', () => {
  const field = compileShadoWorldGrassField([rollingGround()], { density: 8 })!;
  const resolution = field.heightField.resolution;
  const index = new Map<string, number>();
  field.cells.x.forEach((x, cell) => index.set(`${x}:${field.cells.z[cell]}`, cell));

  it('gives the same ground height from either side of a shared border', () => {
    const pairs: [number, number][] = [];
    field.cells.x.forEach((x, left) => {
      const right = index.get(`${x + 1}:${field.cells.z[left]}`);
      if (right !== undefined) pairs.push([left, right]);
    });
    expect(pairs.length).toBeGreaterThan(4);

    const data = packShadoGrassFieldData(field, pairs.flat());
    let worst = 0;
    pairs.forEach(([, ], pair) => {
      for (let step = 0; step <= 8; step++) {
        const v = step / 8;
        // The same world point, reached as the right edge of the left cell and
        // the left edge of the right cell.
        const fromLeft = sampleGround(data, resolution, pair * 2, 1, v);
        const fromRight = sampleGround(data, resolution, pair * 2 + 1, 0, v);
        worst = Math.max(worst, Math.abs(fromLeft - fromRight));
      }
    });
    // Exactly continuous: both sides interpolate between the same two samples.
    expect(worst).toBeLessThan(1e-4);
  });

  it('would step without the neighbour ring, which is why the ring exists', () => {
    // Reconstruct what per-cell clamped sampling produced: each side sees only
    // its own edge sample, and those describe ground 3 m apart.
    const perCell = resolution * resolution;
    let worst = 0;
    field.cells.x.forEach((x, left) => {
      const right = index.get(`${x + 1}:${field.cells.z[left]}`);
      if (right === undefined) return;
      for (let row = 0; row < resolution; row++) {
        const leftEdge =
          field.heightField.minimumY[left]! +
          (field.heightField.samples[left * perCell + row * resolution + resolution - 1]! / 0xffff) *
            field.heightField.heightRange[left]!;
        const rightEdge =
          field.heightField.minimumY[right]! +
          (field.heightField.samples[right * perCell + row * resolution]! / 0xffff) *
            field.heightField.heightRange[right]!;
        worst = Math.max(worst, Math.abs(leftEdge - rightEdge));
      }
    });
    expect(worst).toBeGreaterThan(0.2);
  });

  it('packs absolute world height, so nothing decodes against a per-cell range', () => {
    const data = packShadoGrassFieldData(field, [0]);
    const ground = sampleGround(data, resolution, 0, 0.5, 0.5);
    expect(ground).toBeGreaterThan(field.heightField.minimumY[0]! - 1);
    expect(ground).toBeLessThan(
      field.heightField.minimumY[0]! + field.heightField.heightRange[0]! + 1
    );
  });

  it('refuses a resolution that would overflow the row stride', () => {
    const oversized = {
      ...field,
      heightField: { ...field.heightField, resolution: 16 },
    };
    expect(() => packShadoGrassFieldData(oversized, [0])).toThrow(/row stride/);
  });
});
