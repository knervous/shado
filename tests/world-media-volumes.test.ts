import {
  compileShadoWorld,
  createShadoWorldAuthoring,
  validateShadoWorldAuthoring,
  type ShadoWorldMediaVolume,
  type ShadoWorldPrimitive,
} from '../src/world';

function quad(): ShadoWorldPrimitive {
  return {
    name: 'terrace#0',
    material: 'stone',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

const BANK: ShadoWorldMediaVolume = {
  id: 'church-fog',
  shape: 'box',
  position: [-1387, 33, 605],
  size: [211, 34, 91],
  density: 0.0042,
  albedo: 0.5,
  anisotropy: 0.28,
  color: [0.085, 0.06, 0.185],
  heightFalloff: 0.03,
  feather: 48,
  priority: 2,
};

/**
 * Authored participating media. The runtime medium is one global set of
 * parameters per zone; a volume is how a site gets its own without moving the
 * zone's. See `client/src/fx/volumetric-fog.ts` for what consumes these.
 */
describe('Shado authored media volumes', () => {
  it('carries a volume through compilation into the runtime package', () => {
    const authoring = createShadoWorldAuthoring('media-test');
    authoring.environment.mediaVolumes = [BANK];
    validateShadoWorldAuthoring(authoring, 'media-test');
    const world = compileShadoWorld([quad()], { name: 'media-test', authoring });
    expect(world.environment?.mediaVolumes).toEqual([BANK]);
  });

  it('leaves a world that authors none exactly as it was', () => {
    const authoring = createShadoWorldAuthoring('media-test');
    const world = compileShadoWorld([quad()], { name: 'media-test', authoring });
    expect(world.environment?.mediaVolumes).toEqual([]);
  });

  it.each([
    ['a density that is not a number', { density: Number.NaN }],
    ['a negative density', { density: -1 }],
    ['a zero extent', { size: [0, 10, 10] as [number, number, number] }],
    ['an albedo outside 0..1', { albedo: 1.4 }],
    ['a physically impossible phase term', { anisotropy: 1 }],
    ['an hour outside the clock', { hours: [3, 30] as [number, number] }],
    ['an unknown shape', { shape: 'cylinder' as never }],
  ])('refuses %s', (_label, patch) => {
    const authoring = createShadoWorldAuthoring('media-test');
    // Every one of these reaches the GPU as a NaN or a divide by zero, which
    // does not fail loudly: it poisons the froxel grid and the world renders
    // black with no error anywhere. Refusing at authoring time is the only
    // place the mistake has a name.
    authoring.environment.mediaVolumes = [{ ...BANK, ...patch }];
    expect(() => validateShadoWorldAuthoring(authoring, 'media-test')).toThrow();
  });

  it('refuses two volumes that share an id', () => {
    const authoring = createShadoWorldAuthoring('media-test');
    authoring.environment.mediaVolumes = [BANK, { ...BANK, density: 0.02 }];
    expect(() => validateShadoWorldAuthoring(authoring, 'media-test')).toThrow(/unique IDs/);
  });
});
