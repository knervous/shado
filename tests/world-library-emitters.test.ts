import {
  compileShadoWorld,
  createShadoWorldAuthoring,
  validateShadoWorldAuthoring,
  type ShadoWorldLibraryEmitter,
  type ShadoWorldPrimitive,
} from '../src/world';

function quad(): ShadoWorldPrimitive {
  return {
    name: 'plaza#0',
    material: 'stone',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

const BRAZIER: ShadoWorldLibraryEmitter = {
  id: 'gate-brazier',
  kind: 'library',
  effect: 'fy_brazier',
  position: [120, 4, -36],
  scale: 1.5,
  yaw: 0.5,
  power: 0.8,
  tint: [1, 0.9, 0.7, 1],
  interval: 6,
  range: 600,
  hours: [18, 6],
};

/**
 * A zone placing an effect from the converted library (`client/src/fx/zone-particles.ts`
 * plays it), beside the procedural motes emitters.
 */
describe('Shado library effect emitters', () => {
  it('carries a library emitter through compilation', () => {
    const authoring = createShadoWorldAuthoring('library-test');
    authoring.environment.particleEmitters = [BRAZIER];
    validateShadoWorldAuthoring(authoring, 'library-test');
    const world = compileShadoWorld([quad()], { name: 'library-test', authoring });
    expect(world.environment?.particleEmitters).toEqual([BRAZIER]);
  });

  it('accepts the bare minimum: an effect, a place and a range', () => {
    const authoring = createShadoWorldAuthoring('library-test');
    authoring.environment.particleEmitters = [{ id: 'm', kind: 'library', effect: 'stardust', position: [0, 0, 0], range: 100 }];
    expect(() => validateShadoWorldAuthoring(authoring, 'library-test')).not.toThrow();
  });

  it.each([
    ['an effect name with a path in it', { effect: '../library.json' }],
    ['a zero scale', { scale: 0 }],
    ['a power above full', { power: 1.5 }],
    ['an interval too quick to read', { interval: 0.01 }],
    ['a three-channel tint', { tint: [1, 1, 1] as never }],
    ['no range', { range: undefined as never }],
    ['an hour outside the clock', { hours: [3, 30] as [number, number] }],
    ['an unknown kind', { kind: 'mesh' as never }],
  ])('refuses %s', (_label, patch) => {
    const authoring = createShadoWorldAuthoring('library-test');
    authoring.environment.particleEmitters = [{ ...BRAZIER, ...patch }];
    expect(() => validateShadoWorldAuthoring(authoring, 'library-test')).toThrow();
  });
});
