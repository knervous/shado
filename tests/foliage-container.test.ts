import { NullEngine } from '@babylonjs/core';
import {
  ShadoFoliageActor,
  ShadoFoliageContainer,
  seedFoliageParams,
} from '../src/extensions/ShadoFoliageContainer/ShadoFoliageContainer';
import {
  resolveShadoFoliagePlugins,
  shadoFoliageWind,
  type ShadoFoliagePluginSpec,
} from '../src/extensions/ShadoFoliageContainer/plugins';

const FULL_SET: ShadoFoliagePluginSpec[] = [
  { plugin: 'wind', amplitude: 0.24, frequency: 1.1, direction: [1, 0.4] },
  { plugin: 'playerBend', radius: 2.5, strength: 0.4 },
  { plugin: 'proximityFade', fadeStart: 58, fadeEnd: 76 },
  { plugin: 'tint', variationColor: [0.62, 0.74, 0.36], rootDarkening: 0.4 },
];

async function makeContainer(plugins: ShadoFoliagePluginSpec[] = FULL_SET) {
  const engine = new NullEngine();
  await ShadoFoliageContainer.initialize(engine, { wasm: false, extra: ShadoFoliageActor });
  const container = new ShadoFoliageContainer(engine);
  const material = container.configureFoliage({ plugins, sourceHeight: 12 });
  return { engine, container, material };
}

describe('ShadoFoliageContainer', () => {
  it('compiles every plugin into one GLSL displacement block', async () => {
    const { engine, container } = await makeContainer();
    const { vs, fs } = container.generateGLSLPair();

    // The block is emitted once and each plugin contributes into it, rather
    // than each plugin writing its own clip position.
    expect(vs.match(/vec3 shadoFoliageWorld = p;/g)).toHaveLength(1);
    expect(vs.match(/gl_Position = worldViewProjection \* vec4\(shadoFoliageWorld/g)).toHaveLength(1);

    expect(vs).toContain('uniform vec4 uShadoFoliageWind;');
    expect(vs).toContain('uniform vec4 uShadoFoliageBend;');
    expect(vs).toContain('uniform vec4 uShadoFoliageFade;');
    expect(vs).toContain('uniform vec4 uShadoFoliageTint;');
    expect(vs).toContain('inst.foliageParams.x');
    expect(vs).toContain('varying float vShadoFoliageFade;');
    expect(fs).toContain('varying float vShadoFoliageFade;');
    expect(fs).toContain('discard;');

    // Plugin order is the order given, so a caller can reason about composition.
    expect(vs.indexOf('uShadoFoliageWind.w')).toBeLessThan(vs.indexOf('uShadoFoliageBend.y'));
    expect(vs.indexOf('uShadoFoliageBend.y')).toBeLessThan(vs.indexOf('uShadoFoliageFade.x'));

    container.dispose();
    engine.dispose();
  });

  it('emits the WGSL equivalent without swizzle assignment or untyped locals', async () => {
    const { engine, container } = await makeContainer();
    const { vs, fs } = container.generateWGSLPair();

    expect(vs).toContain('var shadoFoliageWorld = worldPosition;');
    expect(vs).toContain('vertexOutputs.position = uniforms.worldViewProjection');
    expect(vs).toContain('uniforms.uShadoFoliageWind');
    expect(vs).toContain('varying vShadoFoliageFade: f32;');
    expect(fs).toContain('discard;');

    // WGSL forbids assigning through a swizzle. A plugin that did so would
    // compile in GLSL and fail only on WebGPU.
    expect(vs).not.toMatch(/shadoFoliageWorld\.[xyzrgb]+\s*=/);
    expect(vs).not.toMatch(/shadoColor\.[xyzrgb]+\s*=/);

    container.dispose();
    engine.dispose();
  });

  it('declares every WGSL uniform in the shader source, not just to the material', async () => {
    const { engine, container, material } = await makeContainer();
    const { vs } = container.generateWGSLPair();

    // WGSL has no loose uniforms. Babylon builds its leftover uniform struct
    // from `uniform name: type;` declarations in the source, so a uniform named
    // only in the material options compiles to "struct member not found" on the
    // device and never fails in a NullEngine test.
    for (const name of material.materialUniforms) {
      expect(vs).toMatch(new RegExp(`uniform ${name}: (f32|vec3f|vec4f);`));
    }

    container.dispose();
    engine.dispose();
  });

  it('declares every plugin uniform to the material', async () => {
    const { engine, container, material } = await makeContainer();
    expect(material.materialUniforms).toEqual([
      'uShadoFoliageTime',
      'uShadoFoliageFocus',
      'uShadoFoliageCamera',
      'uShadoFoliageInverseHeight',
      'uShadoFoliageWind',
      'uShadoFoliageGust',
      'uShadoFoliageBend',
      'uShadoFoliageFade',
      'uShadoFoliageTint',
      'uShadoFoliageTintWeights',
    ]);
    container.dispose();
    engine.dispose();
  });

  it('renders inert when no plugins are configured', async () => {
    const { engine, container } = await makeContainer([]);
    const { vs } = container.generateGLSLPair();
    expect(vs).toContain('float shadoFoliageFade = 1.0;');
    expect(vs).not.toContain('uShadoFoliageWind');
    container.dispose();
    engine.dispose();
  });

  it('refuses a second behavior set on one container', async () => {
    const { engine, container } = await makeContainer();
    expect(() => container.configureFoliage({ plugins: [] })).toThrow(
      /already has compiled foliage plugins/
    );
    container.dispose();
    engine.dispose();
  });

  it('accepts already-resolved plugins alongside declarative specs', async () => {
    const engine = new NullEngine();
    await ShadoFoliageContainer.initialize(engine, { wasm: false, extra: ShadoFoliageActor });
    const container = new ShadoFoliageContainer(engine);
    container.configureFoliage({
      sourceHeight: 1,
      plugins: [
        shadoFoliageWind({ amplitude: 0.05 }),
        { plugin: 'proximityFade', fadeStart: 10, fadeEnd: 20 },
      ],
    });
    expect(container.plugins.map(plugin => plugin.name)).toEqual(['wind', 'proximityFade']);
    container.dispose();
    engine.dispose();
  });
});

describe('foliage plugin configuration', () => {
  it('rejects a plugin configured twice, which would silently lose a config', () => {
    expect(() =>
      resolveShadoFoliagePlugins([
        { plugin: 'wind', amplitude: 0.1 },
        { plugin: 'wind', amplitude: 0.4 },
      ])
    ).toThrow(/configured more than once/);
  });

  it('rejects an unknown plugin instead of silently rendering nothing', () => {
    expect(() =>
      resolveShadoFoliagePlugins([{ plugin: 'sway' } as unknown as ShadoFoliagePluginSpec])
    ).toThrow(/Unknown foliage plugin 'sway'/);
  });

  it('validates plugin parameters at configuration time', () => {
    expect(() => resolveShadoFoliagePlugins([{ plugin: 'proximityFade', fadeStart: 80, fadeEnd: 40 }]))
      .toThrow(/fadeEnd must exceed/);
    expect(() => resolveShadoFoliagePlugins([{ plugin: 'wind', direction: [0, 0] }]))
      .toThrow(/must not be zero-length/);
    expect(() => resolveShadoFoliagePlugins([{ plugin: 'playerBend', radius: 0, strength: 1 }]))
      .toThrow(/radius must be positive/);
    expect(() =>
      resolveShadoFoliagePlugins([{ plugin: 'tint', variationColor: [1.4, 0, 0] }])
    ).toThrow(/variationColor.r must be between zero and one/);
  });
});

describe('seedFoliageParams', () => {
  it('is a stable function of position, so streaming cannot reroll a plant', () => {
    const first = seedFoliageParams(128.25, -64.5);
    const second = seedFoliageParams(128.25, -64.5);
    expect(Array.from(second)).toEqual(Array.from(first));
  });

  it('decorrelates neighbouring plants', () => {
    const a = seedFoliageParams(10, 10);
    const b = seedFoliageParams(10.5, 10);
    expect(a[0]).not.toBeCloseTo(b[0]!, 2);
  });

  it('stays inside the unit range every plugin assumes', () => {
    for (let step = 0; step < 512; step++) {
      const params = seedFoliageParams(step * 3.7, step * -1.9);
      for (const value of params) {
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThan(1);
      }
    }
  });
});
