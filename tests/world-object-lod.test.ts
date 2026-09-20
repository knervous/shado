import {
  buildShadoWorldObjectRenderBatches,
  type ShadoWorldObjectLodSelection,
} from '../src/world';

/**
 * A package with one prototype and four stamps strung out along +Z, so their
 * projected sizes fall in a known order.
 */
function packageWith(radius: number, distances: number[]) {
  const count = distances.length;
  return {
    objects: {
      prototypes: {
        id: ['prototype'],
        source: ['/objects/prototype/final.glb.gz'],
        boundsRadius: [radius],
        firstStampRef: [0],
        stampRefCount: [count],
        metadata: [{}],
      },
      prototypeStampRefs: distances.map((_, index) => index),
      stamps: {
        id: distances.map((_, index) => `stamp-${index}`),
        prototype: distances.map(() => 0),
        enabled: distances.map(() => 1),
        positionX: distances.map(() => 0),
        positionY: distances.map(() => 0),
        positionZ: [...distances],
        rotationX: distances.map(() => 0),
        rotationY: distances.map(() => 0),
        rotationZ: distances.map(() => 0),
        scaleX: distances.map(() => 1),
        scaleY: distances.map(() => 1),
        scaleZ: distances.map(() => 1),
        radius: distances.map(() => radius),
        cellId: distances.map(() => 0),
        phaseMask: distances.map(() => 0xffffffff),
        tags: distances.map(() => []),
        metadata: distances.map(() => ({})),
      },
    },
  } as never;
}

const selection = (
  overrides: Partial<ShadoWorldObjectLodSelection> = {},
): ShadoWorldObjectLodSelection => ({
  camera: [0, 0, 0],
  // One world unit of radius at one unit of distance covers 1000 pixels, so a
  // radius-1 stamp at distance d is 1000/d pixels high. Keeps the arithmetic
  // in the test readable rather than encoding a field of view.
  pixelsPerRadius: 1000,
  thresholds: [100, 25],
  levelsFor: () => 2,
  ...overrides,
});

describe('object render batch levels', () => {
  it('is the whole set at level 0 when nothing asks for levels', () => {
    const [batch] = buildShadoWorldObjectRenderBatches(packageWith(1, [5, 50, 500]));
    expect(batch.levels).toHaveLength(1);
    expect(batch.levels[0].level).toBe(0);
    // The single level shares the batch's own arrays rather than copying them.
    expect(batch.levels[0].matrices).toBe(batch.matrices);
    expect(batch.levels[0].stampIndices).toBe(batch.stampIndices);
  });

  it('deals each stamp into a level by projected size', () => {
    // 1000/5 = 200px -> level 0; 1000/50 = 20px -> level 2 (past both);
    // 1000/20 = 50px -> level 1; 1000/500 = 2px -> level 2.
    const [batch] = buildShadoWorldObjectRenderBatches(
      packageWith(1, [5, 20, 50, 500]),
      undefined,
      selection(),
    );
    const byLevel = new Map(batch.levels.map((entry) => [entry.level, entry]));
    expect([...byLevel.keys()].sort()).toEqual([0, 1, 2]);
    expect([...byLevel.get(0)!.stampIndices]).toEqual([0]);
    expect([...byLevel.get(1)!.stampIndices]).toEqual([1]);
    expect([...byLevel.get(2)!.stampIndices].sort()).toEqual([2, 3]);
    // Every stamp lands somewhere, exactly once.
    const total = batch.levels.reduce((sum, entry) => sum + entry.stampIndices.length, 0);
    expect(total).toBe(batch.stampIndices.length);
  });

  it('carries each stamp its own matrix into the level it landed in', () => {
    const [batch] = buildShadoWorldObjectRenderBatches(
      packageWith(1, [5, 500]),
      undefined,
      selection(),
    );
    const near = batch.levels.find((entry) => entry.level === 0)!;
    const far = batch.levels.find((entry) => entry.level === 2)!;
    // Translation is the last column of the column-major matrix; z is index 14.
    expect(near.matrices[14]).toBeCloseTo(5);
    expect(far.matrices[14]).toBeCloseTo(500);
  });

  it('never drops a stamp past the levels its prototype ships', () => {
    const [batch] = buildShadoWorldObjectRenderBatches(
      packageWith(1, [5, 50, 500]),
      undefined,
      selection({ levelsFor: () => 1 }),
    );
    expect(batch.levels.map((entry) => entry.level)).toEqual([0, 1]);
  });

  it('stays one level when every stamp is near, so the common case allocates nothing', () => {
    const [batch] = buildShadoWorldObjectRenderBatches(
      packageWith(1, [1, 2, 3]),
      undefined,
      selection(),
    );
    expect(batch.levels).toHaveLength(1);
    expect(batch.levels[0].matrices).toBe(batch.matrices);
  });

  it('treats a stamp at the camera as the largest it can be, not as a division by zero', () => {
    const [batch] = buildShadoWorldObjectRenderBatches(
      packageWith(1, [0]),
      undefined,
      selection(),
    );
    expect(batch.levels).toHaveLength(1);
    expect(batch.levels[0].level).toBe(0);
  });
});
