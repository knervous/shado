import { NullEngine, Scene } from '@babylonjs/core';
import {
  SHADO_WORLD_LIGHT_FIELD_PARAMS_2,
  SHADO_WORLD_LIGHT_FIELD_GLSL,
  SHADO_WORLD_LIGHT_FIELD_WGSL,
  ShadoWorldLightBuffer,
} from '../src/materials/ShadoWorldLightBuffer';
import type { ShadoWorldCompiledPointLight, ShadoWorldSpatialPackage } from '../src/world';

function light(index: number, position: [number, number, number]): ShadoWorldCompiledPointLight {
  return {
    id: `light-${index}`,
    name: `Light ${index}`,
    source: 'standalone',
    enabled: true,
    position,
    color: [1, 0.5, 0.25],
    intensity: 13,
    range: 14,
    radius: 0.25,
    castsShadows: false,
    bake: false,
    runtime: true,
    cellId: -1,
    visibilityRegion: -1,
    phaseMask: 0xffffffff,
    tags: [],
    metadata: {},
  };
}

describe('Shado world light field', () => {
  it('avoids reserved shader-language identifiers in shared source', () => {
    expect(SHADO_WORLD_LIGHT_FIELD_WGSL).not.toMatch(/\blet\s+layout\b/);
    expect(SHADO_WORLD_LIGHT_FIELD_GLSL).not.toMatch(/\bvec4\s+layout\b/);
    expect(SHADO_WORLD_LIGHT_FIELD_WGSL).toMatch(/\blet\s+fieldIndex\b/);
  });

  it('keeps more than 16 co-located lights without a global light-list cap', () => {
    const engine = new NullEngine();
    const scene = new Scene(engine);
    const pointLights = Array.from({ length: 40 }, (_, index) => light(index, [16, 4, 16]));
    const world = {
      name: 'dense-light-test',
      bounds: { min: [0, 0, 0], max: [64, 32, 64] },
      tiles: { size: 32 },
      pointLights,
    } as unknown as ShadoWorldSpatialPackage;

    const field = new ShadoWorldLightBuffer(scene, world, {} as never, { cellSize: 16 });

    expect(field.backend).toBe('datatex');
    expect(field.state.count).toBe(40);
    expect(field.activeCount).toBe(40);
    expect(field.diagnostics.maxLightsPerCell).toBe(40);
    expect(field.diagnostics.lightReferences).toBeGreaterThanOrEqual(40);
    expect(field.diagnostics.maxLightsPerCell).toBeGreaterThan(16);
    const arena = (field as unknown as { arena: Float32Array }).arena;
    // The package retains its authored intensity-13/range-14 ABI, while the
    // runtime field maps that into a softer render-space intensity and radius.
    expect(Array.from(arena.slice(0, 4))).toEqual([16, 4, 16, 31.5]);
    expect(arena[4]).toBeCloseTo(1.3);
    expect(arena[5]).toBeCloseTo(0.65);
    expect(arena[6]).toBeCloseTo(0.325);
    expect(arena[7]).toBeCloseTo(0.25);
    let lambertFloor = -1;
    field.bindMaterial({
      setTexture() {},
      setVector4(name, value) {
        if (name === SHADO_WORLD_LIGHT_FIELD_PARAMS_2) lambertFloor = value.w;
      },
    });
    expect(lambertFloor).toBeCloseTo(0.12);
    const first = field.reduce([], {} as never, [0, 0, 0]);
    const cameraMoved = field.reduce([], {} as never, [63, 0, 63]);
    expect(first.activeIndices).toHaveLength(40);
    expect(first.capped).toBe(false);
    expect(cameraMoved).toBe(first);
    field.dispose();
    scene.dispose();
    engine.dispose();
  });

  it('runs legacy flames at night, keeps interiors on, and flickers without repacking', () => {
    const engine = new NullEngine();
    const scene = new Scene(engine);
    const outdoor = light(0, [8, 3, 8]);
    outdoor.metadata = {
      kind: 'torch-flame',
      flickerProfile: 'fire-gentle',
      flickerAmplitude: 0.08,
    };
    const interior = light(1, [9, 3, 8]);
    interior.activation = { mode: 'always', onHour: 18, offHour: 6, transitionMinutes: 25 };
    interior.flicker = { profile: 'steady', amplitude: 0, speed: 0 };
    const world = {
      name: 'runtime-light-behavior-test',
      bounds: { min: [0, 0, 0], max: [32, 16, 32] },
      tiles: { size: 16 },
      pointLights: [outdoor, interior],
    } as unknown as ShadoWorldSpatialPackage;
    const field = new ShadoWorldLightBuffer(scene, world, {} as never, { cellSize: 16 });
    const arena = (field as unknown as { arena: Float32Array }).arena;

    expect(field.tickRuntime(100, 12)).toBe(true);
    expect((field as unknown as { arena: Float32Array }).arena).toBe(arena);
    expect(field.activeCount).toBe(1);
    expect(arena[4]).toBe(0);
    expect(arena[12]).toBeCloseTo(1.3);

    expect(field.tickRuntime(100, 22)).toBe(true);
    expect(field.activeCount).toBe(2);
    const firstNightSample = arena[4]!;
    expect(firstNightSample).toBeGreaterThan(1.1);
    expect(field.tickRuntime(100, 22)).toBe(true);
    expect(arena[4]).not.toBeCloseTo(firstNightSample, 4);

    field.dispose();
    scene.dispose();
    engine.dispose();
  });

  it('keeps field memory linear as authored light count grows', () => {
    const engine = new NullEngine();
    const scene = new Scene(engine);
    const makeDistributedWorld = (count: number) => {
      const side = Math.ceil(Math.sqrt(count));
      const size = side * 32;
      return {
        name: `light-scaling-${count}`,
        bounds: { min: [0, 0, 0], max: [size, 32, size] },
        tiles: { size: 32 },
        pointLights: Array.from({ length: count }, (_, index) =>
          light(index, [(index % side) * 32 + 16, 4, Math.floor(index / side) * 32 + 16])
        ),
      } as unknown as ShadoWorldSpatialPackage;
    };

    const small = new ShadoWorldLightBuffer(scene, makeDistributedWorld(64), {} as never, {
      cellSize: 32,
    });
    const large = new ShadoWorldLightBuffer(scene, makeDistributedWorld(512), {} as never, {
      cellSize: 32,
    });

    expect(small.diagnostics.lightReferences).toBeLessThanOrEqual(64 * 16);
    expect(large.diagnostics.lightReferences).toBeLessThanOrEqual(512 * 16);
    expect(large.diagnostics.arenaBytes / 512).toBeLessThan(256);
    expect(large.diagnostics.arenaBytes / small.diagnostics.arenaBytes).toBeLessThan(12);

    small.dispose();
    large.dispose();
    scene.dispose();
    engine.dispose();
  });

  it('bounds malformed authored radiance while retaining its hue', () => {
    const engine = new NullEngine();
    const scene = new Scene(engine);
    const malformed = light(0, [8, 3, 8]);
    malformed.intensity = 143;
    malformed.range = 24;
    const world = {
      name: 'malformed-light-test',
      bounds: { min: [0, 0, 0], max: [64, 16, 64] },
      tiles: { size: 16 },
      pointLights: [malformed],
    } as unknown as ShadoWorldSpatialPackage;
    const field = new ShadoWorldLightBuffer(scene, world, {} as never, {
      cellSize: 16,
    });
    const arena = (field as unknown as { arena: Float32Array }).arena;

    expect(arena[3]).toBe(54);
    expect(arena[4]).toBeCloseTo(1.75);
    expect(arena[5]).toBeCloseTo(0.875);
    expect(arena[6]).toBeCloseTo(0.4375);

    field.dispose();
    scene.dispose();
    engine.dispose();
  });
});
