import { compileShadoWorld } from '../src/world/compiler';
import {
  compileShadoWorldGrassField,
  shadoWorldGrassBladesPerCell,
  shadoWorldGrassCellCoverage,
} from '../src/world/grass-field';
import type { ShadoWorldPrimitive } from '../src/world/types';

/** A flat grass-tagged quad `size` metres on a side, at the origin. */
function grassGround(size = 96): ShadoWorldPrimitive {
  return {
    name: 'grass-ground',
    material: 'grass1',
    extraShader: 'grass',
    positions: new Float32Array([0, 0, 0, size, 0, 0, size, 0, size, 0, 0, size]),
    normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  } as unknown as ShadoWorldPrimitive;
}

/** An upward-facing slab that should suppress the grass beneath it. */
function platform(size = 24): ShadoWorldPrimitive {
  return {
    name: 'platform',
    material: 'stone',
    positions: new Float32Array([0, 3, 0, size, 3, 0, size, 3, size, 0, 3, size]),
    normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  } as unknown as ShadoWorldPrimitive;
}

function packageSize(value: unknown): number {
  return JSON.stringify(value).length;
}

describe('compileShadoWorldGrassField', () => {
  it('is the same size at any density, which is the whole point', () => {
    const sparse = compileShadoWorldGrassField([grassGround()], { density: 1 })!;
    const lush = compileShadoWorldGrassField([grassGround()], { density: 4000 })!;

    // v1 stores one record per blade, so a 4000x density increase would be a
    // 4000x package. Here density is a single number and nothing else moves.
    expect(packageSize({ ...lush, density: 1 })).toBe(packageSize(sparse));
    expect(lush.cells.x).toEqual(sparse.cells.x);
    expect(lush.coverage.words).toEqual(sparse.coverage.words);
  });

  it('reaches a density v1 cannot express at zone scale', () => {
    // Crownward's floor is one primitive, so v1 clamps it to
    // maxPlacementsPerPrimitive (18,000) however large the lawn is.
    const field = compileShadoWorldGrassField([grassGround(480)], { density: 12 })!;
    const blades = field.cells.x.length * shadoWorldGrassBladesPerCell(field);
    expect(blades).toBeGreaterThan(1_000_000);
    expect(field.cells.x.length).toBeGreaterThan(100);
  });

  it('is deterministic', () => {
    const first = compileShadoWorldGrassField([grassGround()], { density: 8 });
    const second = compileShadoWorldGrassField([grassGround()], { density: 8 });
    expect(second).toEqual(first);
  });

  it('reports per-cell coverage so an edge cell is not given a full quota', () => {
    // 100 m does not tile evenly into 24 m cells, so the far row and column are
    // partial. A lawn sized to an exact multiple would have no edge cell at all.
    const field = compileShadoWorldGrassField([grassGround(100)], { density: 8 })!;
    const coverages = field.cells.x.map((_, cell) =>
      shadoWorldGrassCellCoverage(field, cell)
    );
    for (const coverage of coverages) {
      expect(coverage).toBeGreaterThan(0);
      expect(coverage).toBeLessThanOrEqual(1);
    }
    expect(Math.max(...coverages)).toBeCloseTo(1, 5);
    expect(Math.min(...coverages)).toBeLessThan(0.5);
  });

  it('suppresses grass under a structure, as the blade compiler does', () => {
    const open = compileShadoWorldGrassField([grassGround(96)], { density: 8 })!;
    const built = compileShadoWorldGrassField(
      [grassGround(96)],
      { density: 8 },
      [platform(24)]
    )!;
    const total = (field: typeof open) =>
      field.cells.x.reduce((sum, _, cell) => sum + shadoWorldGrassCellCoverage(field, cell), 0);
    expect(total(built)).toBeLessThan(total(open));
  });

  it('emits nothing when grass was never requested', () => {
    expect(compileShadoWorldGrassField([grassGround()], undefined)).toBeUndefined();
    expect(compileShadoWorldGrassField([grassGround()], false)).toBeUndefined();
  });

  it('validates authored parameters', () => {
    expect(() => compileShadoWorldGrassField([grassGround()], { density: 0 })).toThrow(
      /density must be positive/
    );
    expect(() =>
      compileShadoWorldGrassField([grassGround()], { minHeight: 2, maxHeight: 1 })
    ).toThrow(/minHeight must be positive and not exceed maxHeight/);
    expect(() =>
      compileShadoWorldGrassField([grassGround()], { minimumUpNormal: 1.5 })
    ).toThrow(/minimumUpNormal must be between zero and one/);
  });
});

describe('grass field in the compiled world package', () => {
  const options = { name: 'grass-field-world' };

  it('compiles alongside the blade package rather than replacing it', () => {
    const world = compileShadoWorld([grassGround()], {
      ...options,
      grass: { maxPlacements: 64, maxPlacementsPerPrimitive: 64 },
      grassField: { density: 10 },
    });
    expect(world.grass?.version).toBe(1);
    expect(world.grassField?.version).toBe(2);
    // Both describe the same surface, so they must agree on which cells exist.
    expect(world.grassField?.cells.x.length).toBe(world.grass?.cells.x.length);
  });

  it('stays absent when unrequested, so existing zones are unchanged', () => {
    const world = compileShadoWorld([grassGround()], { ...options, grass: {} });
    expect(world.grassField).toBeUndefined();
  });

  it('covers the field in the package integrity hash', () => {
    const sparse = compileShadoWorld([grassGround()], { ...options, grassField: { density: 2 } });
    const wider = compileShadoWorld([grassGround(120)], {
      ...options,
      grassField: { density: 2 },
    });
    expect(wider.integrity?.layoutHash).not.toBe(sparse.integrity?.layoutHash);
  });
});
