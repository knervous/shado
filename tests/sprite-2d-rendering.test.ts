import { beforeAll, describe, expect, it } from '@jest/globals';
import {
  Engine,
  MeshBuilder,
  NullEngine,
  RawTexture,
  Ray,
  Scene,
  Texture,
  Vector3,
} from '@babylonjs/core';

import {
  resolveShadoDynamicEntityRenderMode,
  ShadoDynamicEntityContainer,
} from '../src/render/ShadoDynamicEntityContainer';
import { ShadoDynamicEntityRenderer } from '../src/render/ShadoDynamicEntityRenderer';
import {
  SHADO_ENTITY_HIGHLIGHTED,
  SHADO_ENTITY_SELECTED,
  SHADO_ENTITY_VISIBLE,
} from '../src/render/ShadoEntity2D';
import { ShadoSprite2DRenderer } from '../src/render/ShadoSprite2DRenderer';
import { ShadoSprite2DMotionKernel } from '../src/render/ShadoSprite2DMotionKernel';
import { emitSprite2DMotionWGSL } from '../src/render/ShadoSprite2DGpuMotion';
import { emitSprite2DVisibilityWGSL } from '../src/render/ShadoSprite2DGpuVisibility';
import { ShadoText2DRenderer } from '../src/render/ShadoText2DRenderer';

describe('Shado 2D rendering baseline', () => {
  let engine: NullEngine;

  beforeAll(async () => {
    engine = new NullEngine();
    const initialized = await ShadoDynamicEntityContainer.initialize(engine, {
      backend: 'datatex',
      wasm: false,
    });
    if (!initialized) throw new Error('ShadoDynamicEntityContainer initialization failed');
  });

  it('maps legacy plane options to explicit presentations', () => {
    expect(resolveShadoDynamicEntityRenderMode({ geometry: 'plane' }).presentation).toBe(
      'billboard-y'
    );
    expect(
      resolveShadoDynamicEntityRenderMode({ geometry: 'plane', billboard: false }).presentation
    ).toBe('ground');
    expect(resolveShadoDynamicEntityRenderMode({ presentation: 'billboard-screen' })).toMatchObject(
      {
        geometry: 'plane',
        billboard: true,
        presentation: 'billboard-screen',
      }
    );
    expect(resolveShadoDynamicEntityRenderMode({ presentation: 'slab' }).geometry).toBe(
      'spriteSlab'
    );
    expect(resolveShadoDynamicEntityRenderMode().alphaMode).toBe('premultiplied');
  });

  it('uses declared height and a bottom pivot in cylindrical billboard shaders', () => {
    const container = new ShadoDynamicEntityContainer(engine);
    container.configureRenderMode({ presentation: 'billboard-y', alphaMode: 'cutout' });
    const glsl = container.generateGLSLPair();
    const wgsl = container.generateWGSLPair();

    expect(glsl.vs).toContain('pivotOffset.y * render.y');
    expect(glsl.vs).toContain('vec3(0.0, 1.0, 0.0)');
    expect(glsl.vs).not.toContain('position.y * positionSize.w');
    expect(wgsl.vs).toContain('pivotOffset.y * render.y');
    expect(wgsl.vs).toContain('vec3f(0.0, 1.0, 0.0)');
    expect(wgsl.vs).not.toContain('vertexInputs.position.y * positionSize.w');
  });

  it('keeps screen-facing camera-up separate from world-up billboards', () => {
    const container = new ShadoDynamicEntityContainer(engine);
    container.configureRenderMode({ presentation: 'billboard-screen' });
    const pair = container.generateGLSLPair();
    expect(pair.vs).toContain('vec3(view[0][1], view[1][1], view[2][1])');
    expect(pair.vs).not.toContain(
      'cameraRight = normalize(vec3(cameraRight.x, 0.0, cameraRight.z))'
    );
  });

  it('emits explicit cutout and premultiplied fragment variants', () => {
    const cutout = new ShadoDynamicEntityContainer(engine);
    cutout.configureRenderMode({ presentation: 'billboard-y', alphaMode: 'cutout' });
    expect(cutout.generateGLSLPair().fs).toContain('outColor.a < uShadoAlphaCutoff');
    expect(cutout.generateGLSLPair().fs).not.toContain('outColor.rgb *= outColor.a');

    const premultiplied = new ShadoDynamicEntityContainer(engine);
    premultiplied.configureRenderMode({
      presentation: 'billboard-screen',
      alphaMode: 'premultiplied',
    });
    expect(premultiplied.generateGLSLPair().fs).toContain('outColor.rgb *= outColor.a');
    expect(premultiplied.generateWGSLPair().fs).toContain(
      'vec4f(outColor.rgb * outColor.a, outColor.a)'
    );
  });

  it('owns visibility, selection, and highlight mutations at container level', () => {
    const container = new ShadoDynamicEntityContainer(engine);
    container.add({ id: 'hero', x: 0, y: 0, width: 1, height: 2 });
    container.syncDrawList();
    expect(container.drawCount).toBe(1);

    expect(container.setSelected('hero', true)).toBe(true);
    expect(container.setHighlighted('hero', true)).toBe(true);
    const entity = container.getEntity(0)!;
    expect((entity.renderState[1] | 0) & SHADO_ENTITY_SELECTED).toBeTruthy();
    expect((entity.renderState[1] | 0) & SHADO_ENTITY_HIGHLIGHTED).toBeTruthy();

    expect(container.setVisible('hero', false)).toBe(true);
    expect((entity.renderState[1] | 0) & SHADO_ENTITY_VISIBLE).toBeFalsy();
    expect(container.drawCount).toBe(0);
    expect(container.setVisible('hero', true)).toBe(true);
    expect(container.drawCount).toBe(1);
  });

  it('sorts premultiplied draw buckets by camera depth, far to near', () => {
    const container = new ShadoDynamicEntityContainer(engine);
    container.add({ id: 'near', x: 0, y: 2, width: 1, height: 1 });
    container.add({ id: 'far', x: 0, y: 8, width: 1, height: 1 });
    container.syncDrawList();
    container.sortDrawListByCamera([0, 1, 0], [0, 0, 1]);

    expect((container as any).meshBuckets.get(0)).toEqual([1, 0]);
  });

  it('uses compact records, tiled culling, zoom LOD, and exact 2D picking', async () => {
    const scene = new Scene(engine);
    const texture = RawTexture.CreateRGBATexture(
      new Uint8Array([255, 255, 255, 255]),
      1,
      1,
      scene,
      false,
      false,
      Texture.NEAREST_SAMPLINGMODE
    );
    const atlas = {
      texture: texture as any,
      entries: { default: { layer: 0, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } } },
      get() {
        return this.entries.default;
      },
      dispose() {
        texture.dispose();
      },
    };
    const renderer = new ShadoSprite2DRenderer(scene, atlas, {
      tileSize: 4,
      minPixelSize: 2,
    });
    renderer.upsertMany([
      { id: 'lower', position: [0, 0], size: [4, 4], layer: 1, selected: true },
      { id: 'upper', position: [0, 0], size: [2, 2], layer: 2, order: 1 },
      { id: 'culled', position: [100, 100], size: [2, 2] },
      { id: 'subpixel', position: [3, 3], size: [0.001, 0.001], minPixelSize: 2 },
    ]);
    renderer.setView({ center: [0, 0], halfExtent: [10, 10], viewportPixels: [200, 200] });
    (renderer as any).rebuildVisibleDrawList();

    expect(renderer.getStats()).toMatchObject({
      total: 4,
      visible: 2,
      recordBytes: 48,
    });
    const pick = renderer.pickScreen(100, 100, 200, 200);
    expect(pick?.id).toBe('upper');
    expect(pick?.world).toEqual([0, 0]);
    expect(pick?.uv).toEqual([0.5, 0.5]);
    await expect(renderer.readCpuPositions({ tier: 'visible' })).resolves.toMatchObject({
      source: 'cpu',
      inBand: true,
      stale: false,
      entries: expect.arrayContaining([
        expect.objectContaining({ id: 'lower', position: [0, 0] }),
        expect.objectContaining({ id: 'upper', position: [0, 0] }),
      ]),
    });
    const selected = await renderer.readCpuPositions({ tier: 'selected' });
    expect(selected.entries.map(entry => entry.id)).toEqual(['lower']);
    expect(
      (await renderer.readCpuPositions({ tier: 'range', start: 1, count: 2 })).entries
    ).toHaveLength(2);
    expect(
      (await renderer.readCpuPositions({ tier: 'ids', ids: ['upper', 'missing'] })).entries
    ).toHaveLength(1);

    const rebuilds = renderer.getStats().drawListRebuilds;
    renderer.setView({ center: [0.25, 0.25], halfExtent: [10, 10], viewportPixels: [200, 200] });
    (renderer as any).rebuildVisibleDrawList();
    expect(renderer.getStats().drawListRebuilds).toBe(rebuilds);

    expect(
      renderer.setPositions([
        { id: 'lower', position: [1, 1] },
        { id: 'upper', position: [1, 1] },
        { id: 'missing', position: [1, 1] },
      ])
    ).toBe(2);
    expect(renderer.setPosition('culled', [2, 2])).toBe(true);
    (renderer as any).rebuildVisibleDrawList();
    expect(renderer.getStats().visible).toBe(3);

    (renderer as any).gpuMotionActive = true;
    (renderer as any).gpuCullingActive = true;
    expect(renderer.setVisible('upper', false)).toBe(true);
    expect(renderer.isGpuMotionEnabled).toBe(false);
    expect(renderer.getStats().gpuCullingActive).toBe(false);
    expect(renderer.setVisible('upper', true)).toBe(true);

    (renderer as any).gpuMotionActive = true;
    (renderer as any).gpuCullingActive = true;
    expect(renderer.setPosition('upper', [1.25, 1.25])).toBe(true);
    expect(renderer.isGpuMotionEnabled).toBe(false);
    expect(renderer.getStats().gpuCullingActive).toBe(false);

    const lodRecord = new Float32Array(12);
    (renderer as any).packSpriteInto(lodRecord, 0, {
      id: 'lod',
      position: [0, 0],
      size: [1, 1],
      minPixelSize: 2.5,
    });
    const packedState = Math.round(lodRecord[11]);
    expect((Math.floor(packedState / 8) - 1) / 16).toBe(2.5);

    renderer.setMinPixelSize(0);
    renderer.setView({
      center: [0, 0],
      halfExtent: [1_000_000_000_000, 1_000_000_000_000],
      viewportPixels: [200, 200],
    });
    (renderer as any).rebuildVisibleDrawList();
    expect(renderer.getStats().visible).toBe(3);

    renderer.setMinPixelSize(1_000);
    (renderer as any).rebuildVisibleDrawList();
    expect(renderer.getStats().visible).toBe(0);
    renderer.clear();
    (renderer as any).rebuildVisibleDrawList();
    expect(renderer.getStats()).toMatchObject({ total: 0, visible: 0 });

    renderer.dispose();
    atlas.dispose();
    scene.dispose();
  });

  it('lays out, wraps, culls, updates, and picks arbitrary MSDF text in 2D', () => {
    const scene = new Scene(engine);
    const texture = RawTexture.CreateRGBATexture(
      new Uint8Array([255, 255, 255, 255]),
      1,
      1,
      scene,
      false,
      false,
      Texture.NEAREST_SAMPLINGMODE
    );
    const glyph = (id: number, x: number) => ({
      id,
      x,
      y: 0,
      width: 8,
      height: 12,
      xoffset: 0,
      yoffset: 0,
      xadvance: 9,
    });
    const font = {
      textures: [texture],
      _font: {
        info: { size: 16 },
        common: { scaleW: 32, scaleH: 16, lineHeight: 18 },
        distanceField: { distanceRange: 4 },
        chars: [
          glyph(65, 0),
          glyph(66, 8),
          glyph(63, 16),
          { id: 32, x: 24, y: 0, width: 0, height: 0, xoffset: 0, yoffset: 0, xadvance: 4 },
        ],
        kernings: [{ first: 65, second: 66, amount: -1 }],
      },
    };
    const renderer = new ShadoText2DRenderer(scene, font, { tileSize: 4 });
    renderer.upsertMany([
      {
        id: 'title',
        text: 'AB AB',
        position: [0, 0],
        fontSize: 2,
        maxWidth: 3,
        align: 'center',
        layer: 3,
      },
      { id: 'far', text: 'AB', position: [100, 100], fontSize: 2 },
      { id: 'fallback', text: 'AΩ', position: [4, 0], fontSize: 2 },
    ]);
    renderer.setView({ center: [0, 0], halfExtent: [8, 8], viewportPixels: [320, 320] });
    (renderer as any).rebuildVisibleDrawList();

    expect(renderer.getStats()).toMatchObject({
      textBlocks: 3,
      totalGlyphs: 8,
      visibleGlyphs: 6,
      recordBytes: 64,
      unsupportedCharacters: ['Ω'],
    });
    expect(renderer.pickScreen(160, 160, 320, 320)?.id).toBe('title');
    expect(renderer.setText('title', 'A')).toBe(true);
    (renderer as any).rebuildVisibleDrawList();
    expect(renderer.getStats().totalGlyphs).toBe(5);

    const rebuilds = renderer.getStats().drawListRebuilds;
    renderer.setView({ center: [0.2, 0.2], halfExtent: [8, 8], viewportPixels: [320, 320] });
    (renderer as any).rebuildVisibleDrawList();
    expect(renderer.getStats().drawListRebuilds).toBe(rebuilds);
    expect(renderer.material.options.attributes).toEqual(
      expect.arrayContaining(['iTransform', 'iUvRect', 'iColor', 'iState'])
    );
    expect(renderer.material.options.uniforms).not.toContain('worldViewProjection');
    expect(renderer.material.options.uniforms).not.toContain('view');

    const spaced = (renderer as any).layout({
      id: 'spacing',
      text: 'A A',
      position: [0, 0],
      fontSize: 16,
    });
    expect(spaced.glyphs).toHaveLength(2);
    expect(spaced.glyphs[1].center[0] - spaced.glyphs[0].center[0]).toBe(13);

    renderer.dispose();
    texture.dispose();
    scene.dispose();
  });

  it('integrates deterministic seeded motion in the WASM SIMD kernel', async () => {
    const first = await ShadoSprite2DMotionKernel.create();
    const second = await ShadoSprite2DMotionKernel.create();
    const positions = new Float32Array([0, 0, 0.75, -0.5, -0.25, 0.4]);
    const config = {
      seed: 4242,
      speed: 1.5,
      cadenceMs: 10,
      bounds: [-1, -1, 1, 1] as const,
    };
    first.setPopulation(positions, config, 0);
    second.setPopulation(positions, config, 0);

    const firstStep = Array.from(first.step(4, 0.1));
    const matchingStep = Array.from(second.step(4, 0.1));
    expect(firstStep).toEqual(matchingStep);
    expect(firstStep).not.toEqual(Array.from(positions));

    const afterVectorChanges = Array.from(first.step(100, 0.1));
    const matchingVectorChanges = Array.from(second.step(100, 0.1));
    expect(afterVectorChanges).toEqual(matchingVectorChanges);
    expect(first.size).toBe(3);
  });

  it('emits a bounds-safe GPU motion kernel with stable per-instance state', () => {
    const wgsl = emitSprite2DMotionWGSL();
    expect(wgsl).toContain('var<storage, read_write> motionState: array<vec4f>');
    expect(wgsl).toContain('if (index >= count) { return; }');
    expect(wgsl).toContain('let absoluteIndex = motionParams[10] + index');
    expect(wgsl).toContain('state = vec4f(state.xy, velocity)');
    expect(wgsl).toContain('motionState[index] = state');
  });

  it('compacts GPU-visible sprites and writes an indirect instance count', () => {
    const wgsl = emitSprite2DVisibilityWGSL();
    expect(wgsl).toContain('var<storage, read_write> visibleIndices: array<u32>');
    expect(wgsl).toContain('var<storage, read_write> drawArgs: array<atomic<u32>>');
    expect(wgsl).toContain('let outputIndex = atomicAdd(&drawArgs[1], 1u)');
    expect(wgsl).toContain('visibleIndices[outputIndex] = index');
    expect(wgsl).toContain('let passesLod =');
    expect(wgsl).toContain('rotationCos * unrotatedHalfExtent.x');
    expect(wgsl).toContain('let lodCode = packedState >> 3u');
  });

  it('picks the rendered ground quad instead of the legacy footprint box', () => {
    const scene = new Scene(engine);
    const texture = RawTexture.CreateRGBATexture(
      new Uint8Array([255, 255, 255, 255]),
      1,
      1,
      scene,
      false,
      false,
      Texture.NEAREST_SAMPLINGMODE
    );
    const atlas = {
      texture: texture as any,
      entries: { default: { layer: 0, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } } },
      get() {
        return this.entries.default;
      },
      dispose() {
        texture.dispose();
      },
    };
    const container = new ShadoDynamicEntityContainer(engine, atlas);
    container.add({ id: 'ground', x: 0, y: 0, width: 2, depth: 0.1, height: 2 });
    const renderer = new ShadoDynamicEntityRenderer(scene, container, atlas, {
      presentation: 'ground',
      pivot: [0.5, 0.5],
    });
    const ray = new Ray(new Vector3(0, 5, 0.75), new Vector3(0, -1, 0), 10);
    expect(renderer.pickWithRay(ray)?.id).toBe('ground');
    renderer.dispose();
    atlas.dispose();
    scene.dispose();
  });

  it('forwards alpha options through mesh variant renderer creation', () => {
    const scene = new Scene(engine);
    const texture = RawTexture.CreateRGBATexture(
      new Uint8Array([255, 255, 255, 255]),
      1,
      1,
      scene,
      false,
      false,
      Texture.NEAREST_SAMPLINGMODE
    );
    const atlas = {
      texture: texture as any,
      entries: { default: { layer: 0, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } } },
      get() {
        return this.entries.default;
      },
      dispose() {
        texture.dispose();
      },
    };
    const container = new ShadoDynamicEntityContainer(engine, atlas);
    const mesh = MeshBuilder.CreateBox('variant', {}, scene);
    const [renderer] = ShadoDynamicEntityRenderer.createMeshVariantRenderers(
      scene,
      container,
      atlas,
      { variants: [{ meshIndex: 0, mesh }], alphaMode: 'cutout', alphaCutoff: 0.2 }
    );
    expect(renderer.material.alphaMode).toBe(Engine.ALPHA_DISABLE);
    expect(renderer.material.forceDepthWrite).toBe(true);
    renderer.dispose();
    mesh.dispose();
    atlas.dispose();
    scene.dispose();
  });

  it('keeps pixel LOD rejection in the no-compaction GPU render path', () => {
    const scene = new Scene(engine);
    const texture = RawTexture.CreateRGBATexture(
      new Uint8Array([255, 255, 255, 255]),
      1,
      1,
      scene,
      false,
      false,
      Texture.NEAREST_SAMPLINGMODE
    );
    const atlas = {
      texture: texture as any,
      entries: { default: { layer: 0, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } } },
      get() {
        return this.entries.default;
      },
      dispose() {
        texture.dispose();
      },
    };
    const renderer = new ShadoSprite2DRenderer(scene, atlas, { minPixelSize: 1 });
    expect(renderer.material.options.uniforms).toEqual(
      expect.arrayContaining(['uViewportHeight', 'uMinimumPixelSize'])
    );
    renderer.dispose();
    atlas.dispose();
    scene.dispose();
  });

  it('preserves one deterministic sequence across copy-isolated SIMD shards', async () => {
    const config = {
      seed: 91,
      speed: 2,
      cadenceMs: 20,
      bounds: [-4, -4, 4, 4] as const,
    };
    const source = new Float32Array([0, 0, 1, 1, 2, 2, 3, 3, -1, -1]);
    const whole = await ShadoSprite2DMotionKernel.create();
    const left = await ShadoSprite2DMotionKernel.create();
    const right = await ShadoSprite2DMotionKernel.create();
    whole.setPopulation(source, config, 0);
    left.setPopulation(source.slice(0, 4), config, 0, 0);
    right.setPopulation(source.slice(4), config, 0, 2);

    const expected = Array.from(whole.step(100, 0.05));
    const sharded = [...Array.from(left.step(100, 0.05)), ...Array.from(right.step(100, 0.05))];
    expect(sharded).toEqual(expected);
  });
});
