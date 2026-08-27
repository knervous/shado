import {
  buildShadoWorldLightingManifest,
  compileShadoWorld,
  createShadoWorldAuthoring,
  resolveShadoWorldAudioEmitters,
  ShadoWorldLightState,
  ShadoWorldVisibilityCoordinator,
  validateShadoWorldLightingManifest,
  type ShadoWorldPrimitive,
} from '../src/world';

function quad(lightmapUvs = true): ShadoWorldPrimitive {
  return {
    name: 'town-square#0',
    material: 'stone',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    lightmapUvs: lightmapUvs ? new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]) : undefined,
  };
}

describe('Shado per-zone lighting build manifest', () => {
  it('binds stable render chunks to UV2 and deterministic bake outputs', () => {
    const primitive = quad();
    const world = compileShadoWorld([primitive], { name: 'lighting-test' });
    const manifest = buildShadoWorldLightingManifest(world, [primitive]);

    expect(manifest).toMatchObject({
      kind: 'shado.world.lighting-build',
      version: 1,
      zone: 'lighting-test',
      status: 'ready-for-bake',
      encoding: 'rgbm',
      dependencies: {
        worldLayoutHash: world.integrity.layoutHash,
        plannerVersion: 'shado-zone-lighting-plan-v1',
      },
    });
    expect(manifest.dayKeyframes).toEqual([
      { phase: 0, name: 'night' },
      { phase: 0.25, name: 'dawn' },
      { phase: 0.5, name: 'noon' },
      { phase: 0.75, name: 'dusk' },
    ]);
    expect(manifest.chunks).toEqual([
      expect.objectContaining({
        id: 'chunk_000',
        renderChunk: 0,
        primitive: 0,
        primitiveName: 'town-square#0',
        material: 'stone',
        vertexCount: 4,
        uv2: { present: true, coordinateCount: 8 },
        outputs: {
          staticMap: 'lightmaps/chunk_000_static.ktx2',
          dayMaps: [
            'lightmaps/chunk_000_day_0.ktx2',
            'lightmaps/chunk_000_day_1.ktx2',
            'lightmaps/chunk_000_day_2.ktx2',
            'lightmaps/chunk_000_day_3.ktx2',
          ],
          nightLightMap: 'lightmaps/chunk_000_night_lights.ktx2',
        },
      }),
    ]);
    expect(manifest.dependencies.geometryHash).toMatch(/^[0-9a-f]{8}$/);
    expect(manifest.dependencies.uv2Hash).toMatch(/^[0-9a-f]{8}$/);
    expect(() => validateShadoWorldLightingManifest(manifest)).not.toThrow();
  });

  it('emits a blocked plan when a source zone still needs a UV2 unwrap', () => {
    const primitive = quad(false);
    const world = compileShadoWorld([primitive], { name: 'missing-uv2' });
    const manifest = buildShadoWorldLightingManifest(world, [primitive]);

    expect(manifest.status).toBe('blocked-missing-uv2');
    expect(manifest.chunks[0].uv2).toEqual({
      present: false,
      coordinateCount: 0,
    });
  });

  it('resolves standalone and object-attached point lights into the bake plan', () => {
    const primitive = quad();
    const authoring = createShadoWorldAuthoring('authored-lights');
    authoring.lighting.pointLights.push({
      id: 'plaza-lamp', name: 'Plaza lamp', enabled: true, offset: [2, 8, 4],
      color: [1, 0.6, 0.25], intensity: 18, range: 30, radius: 0.3,
      castsShadows: true, bake: true, runtime: false, phaseMask: 0xffffffff,
      tags: ['plaza'], metadata: { fixture: 'lamp' },
    });
    authoring.objects.prototypes.push({
      id: 'lantern', source: '/lantern.glb', boundsRadius: 2,
      light: {
        enabled: true, offset: [2, 2, 0], color: [1, 0.4, 0.1], intensity: 9,
        range: 12, radius: 0.1, castsShadows: false, bake: true, runtime: true,
        metadata: { flame: true },
      },
      metadata: {},
    });
    authoring.objects.stamps.push({
      id: 'lantern-1', prototype: 'lantern', enabled: true, position: [10, 6, 10],
      rotationDegrees: [0, 0, 0], scale: [2, 1, 1], phaseMask: 0xffffffff,
      tags: ['fixture'], metadata: {},
    });
    const world = compileShadoWorld([primitive], { name: 'authored-lights', authoring });
    const manifest = buildShadoWorldLightingManifest(world, [primitive]);

    expect(world.pointLights).toHaveLength(2);
    expect(world.pointLights?.map(light => light.runtime)).toEqual([false, true]);
    expect(world.pointLights?.map(light => [light.cellId, light.visibilityRegion])).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(manifest.pointLights).toEqual([
      expect.objectContaining({ id: 'plaza-lamp', source: 'standalone', position: [2, 8, 4] }),
      expect.objectContaining({ id: 'lantern-1:light', source: 'object', ownerStamp: 'lantern-1', position: [14, 8, 10] }),
    ]);
    expect(manifest.dependencies.lightHash).toMatch(/^[0-9a-f]{8}$/);
    expect(() => validateShadoWorldLightingManifest(manifest)).not.toThrow();
  });

  it('inherits object audio, applies stamp overrides, and publishes world-space emitters', () => {
    const authoring = createShadoWorldAuthoring('object-audio');
    authoring.objects.prototypes.push({
      id: 'torch', source: '/torch.glb', boundsRadius: 2,
      audio: {
        enabled: true, source: '/fire.ogg', offset: [1, 0, 0], range: 12,
        volume: 0.55, loop: true, metadata: { family: 'fire' },
      },
      metadata: {},
    });
    authoring.objects.stamps.push(
      {
        id: 'torch-base', prototype: 'torch', enabled: true, position: [10, 2, 4],
        rotationDegrees: [0, 90, 0], scale: [2, 1, 1], phaseMask: 0xffffffff,
        tags: [], metadata: {},
      },
      {
        id: 'torch-quiet', prototype: 'torch', enabled: true, position: [0, 0, 0],
        rotationDegrees: [0, 0, 0], scale: [1, 1, 1], phaseMask: 0xffffffff,
        tags: [], metadata: {},
        audio: {
          enabled: false, source: '/fire-low.ogg', offset: [0, 1, 0], range: 5,
          volume: 0.1, loop: true, metadata: { variant: 'quiet' },
        },
      },
    );

    expect(resolveShadoWorldAudioEmitters(authoring)).toEqual([
      expect.objectContaining({ id: 'torch-base:audio', ownerStamp: 'torch-base', source: '/fire.ogg', position: [10, 2, 2], enabled: true }),
      expect.objectContaining({ id: 'torch-quiet:audio', ownerStamp: 'torch-quiet', source: '/fire-low.ogg', enabled: false }),
    ]);

    const world = compileShadoWorld([quad()], { name: 'object-audio', authoring });
    expect(world.environment?.audioEmitters).toEqual([
      expect.objectContaining({
        id: 'torch-base:audio', source: '/fire.ogg', position: [10, 2, 2],
        metadata: expect.objectContaining({ sourceKind: 'object', ownerStamp: 'torch-base' }),
      }),
    ]);
  });

  it('keeps runtime lights in mutable SoA state and reduces compact GPU rows through PVS', async () => {
    const primitive = quad();
    const authoring = createShadoWorldAuthoring('runtime-lights');
    authoring.performanceBudgets.maxRuntimePointLights = 1;
    authoring.lighting.pointLights.push(
      {
        id: 'near', name: 'Near', enabled: true, offset: [0.25, 1, 0.25],
        color: [1, 0.5, 0.25], intensity: 2, range: 8, radius: 0.1,
        castsShadows: false, bake: false, runtime: true, phaseMask: 1,
        tags: [], metadata: {},
      },
      {
        id: 'far', name: 'Far', enabled: true, offset: [0.75, 1, 0.75],
        color: [0.25, 0.5, 1], intensity: 3, range: 8, radius: 0.1,
        castsShadows: false, bake: false, runtime: true, phaseMask: 1,
        tags: [], metadata: {},
      },
      {
        id: 'other-phase', name: 'Other phase', enabled: true, offset: [0.5, 1, 0.5],
        color: [1, 1, 1], intensity: 1, range: 8, radius: 0,
        castsShadows: false, bake: false, runtime: true, phaseMask: 2,
        tags: [], metadata: {},
      }
    );
    const world = compileShadoWorld([primitive], { name: 'runtime-lights', authoring });
    const lights = new ShadoWorldLightState(world.pointLights ?? []);
    const planes = new Float32Array([
      1, 0, 0, 10, -1, 0, 0, 10, 0, 1, 0, 10,
      0, -1, 0, 10, 0, 0, 1, 10, 0, 0, -1, 10,
    ]);
    const coordinator = await ShadoWorldVisibilityCoordinator.create(world);
    const frame = coordinator.reduceWorld(planes, [0, 1, 0]);
    const reduced = coordinator.reduceLights(lights, planes, frame, {
      camera: [0, 1, 0],
      activePhaseMask: 1,
    });

    expect(lights.count).toBe(3);
    expect(Array.from(lights.packed.slice(0, 7))).toEqual([
      0.25, 1, 0.25, 8, 2, 1, 0.5,
    ]);
    expect(lights.packed[7]).toBeCloseTo(0.1);
    expect(reduced.spatialCandidateCount).toBe(2);
    expect(Array.from(reduced.activeIndices)).toEqual([0]);
    expect(reduced.capped).toBe(true);

    lights.update('near', { position: [4, 2, 3], color: [0.5, 0.25, 0], intensity: 4 });
    expect(Array.from(lights.packed.slice(0, 7))).toEqual([
      4, 2, 3, 8, 2, 1, 0,
    ]);
    expect(lights.packed[7]).toBeCloseTo(0.1);
  });

  it('rejects malformed UV2 streams before handing work to a baker', () => {
    const primitive = quad();
    primitive.lightmapUvs = new Float32Array([0, 0]);
    const world = compileShadoWorld([primitive], { name: 'bad-uv2' });

    expect(() => buildShadoWorldLightingManifest(world, [primitive])).toThrow(
      /UV2 requires 8 coordinates/
    );
  });
});
