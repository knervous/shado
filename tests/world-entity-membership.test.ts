import {
  ShadoVisibilityBits,
  ShadoWorldVisibilityCoordinator,
  compileShadoWorld,
} from '../src/world';

/** One quad per x, so each lands in its own render cell and region. */
function quad(x: number) {
  return {
    name: `quad-${x}`,
    material: 'stone',
    positions: new Float32Array([x, 0, 0, x + 1, 0, 0, x + 1, 1, 0, x, 1, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

/** Wide enough to reject nothing; membership is what is under test. */
const PLANES = new Float32Array([
  1, 0, 0, 4096, -1, 0, 0, 4096, 0, 1, 0, 4096,
  0, -1, 0, 4096, 0, 0, 1, 4096, 0, 0, -1, 4096,
]);

describe('entity membership spans an entity, not its centre', () => {
  /** Four 16-unit regions in a row, with a cell in each. */
  const world = compileShadoWorld([quad(0), quad(16), quad(32), quad(48)], {
    name: 'membership',
    tileSize: 16,
    visibilityRegionSize: 16,
    maxClusterTriangles: 2,
  });

  async function reduce(
    entities: { x: number; z?: number; radius: number }[],
    loadedCells: number[]
  ) {
    const coordinator = await ShadoWorldVisibilityCoordinator.create(world);
    const frame = coordinator.reduceWorld(PLANES, [0, 0, 0], {
      loadedCells: Uint8Array.from(loadedCells),
      phaseCells: Uint8Array.from(loadedCells.map(() => 1)),
      portalReachableCells: Uint8Array.from(loadedCells.map(() => 1)),
    });
    const result = coordinator.reduceEntities(
      {
        count: entities.length,
        positionX: Float32Array.from(entities.map((entry) => entry.x)),
        positionY: new Float32Array(entities.length),
        positionZ: Float32Array.from(entities.map((entry) => entry.z ?? 0.5)),
        radius: Float32Array.from(entities.map((entry) => entry.radius)),
      },
      PLANES,
      frame,
      { camera: [0, 0, 0], outsideWorldVisible: false }
    );
    return {
      visible: Array.from(result.visibleIndices),
      overflow: coordinator.overflowEntities,
    };
  }

  it('keeps an actor whose centre is in a rejected region but whose body is not', () => {
    /*
     * Cell 1 is not loaded, so region 1 fails its topology bits. The actor
     * stands at x = 20 -- inside region 1 -- with a radius that reaches back
     * into region 0, which is admitted. A player standing in region 0 can see
     * it, so it has to survive.
     */
    return reduce([{ x: 20, radius: 6 }], [1, 0, 1, 1]).then(({ visible }) => {
      expect(visible).toEqual([0]);
    });
  });

  it('rejects an actor that lies wholly inside rejected regions', async () => {
    const { visible } = await reduce([{ x: 20, radius: 2 }], [1, 0, 1, 1]);
    expect(visible).toEqual([]);
  });

  it('admits an actor spanning several regions when any one of them admits', async () => {
    // Straddles regions 1 and 2; region 1 is rejected, region 2 is not.
    const { visible } = await reduce([{ x: 31, radius: 3 }], [1, 0, 1, 1]);
    expect(visible).toEqual([0]);
  });

  it('never combines topology bits from different regions', async () => {
    /*
     * Region 1 is unloaded and region 2 is loaded, but the run below also
     * clears region 2's phase bit, so neither region passes everything on its
     * own. An entity touching both must not be admitted by their union.
     */
    const coordinator = await ShadoWorldVisibilityCoordinator.create(world);
    const frame = coordinator.reduceWorld(PLANES, [0, 0, 0], {
      loadedCells: Uint8Array.from([1, 0, 1, 1]),
      phaseCells: Uint8Array.from([1, 1, 0, 1]),
      portalReachableCells: Uint8Array.from([1, 1, 1, 1]),
    });
    const result = coordinator.reduceEntities(
      {
        count: 1,
        positionX: Float32Array.from([31]),
        positionY: new Float32Array(1),
        positionZ: Float32Array.from([0.5]),
        radius: Float32Array.from([3]),
      },
      PLANES,
      frame,
      { camera: [0, 0, 0], outsideWorldVisible: false }
    );
    expect(Array.from(result.visibleIndices)).toEqual([]);
  });

  it('admits an entity far larger than the world it stands in', async () => {
    /*
     * Its bound covers every region, so some region admits it and it survives.
     * The enumeration cap itself is exercised in the membership unit tests,
     * which can build a grid large enough to exceed it; this world has four
     * regions and cannot.
     */
    const { visible } = await reduce([{ x: 20, radius: 100000 }], [1, 0, 1, 1]);
    expect(visible).toEqual([0]);
  });

  it('reports no overflow for ordinary entities', async () => {
    const { overflow } = await reduce([{ x: 20, radius: 6 }], [1, 0, 1, 1]);
    expect(overflow).toBe(0);
  });

  it('still rejects by the frustum and range it always did', async () => {
    const coordinator = await ShadoWorldVisibilityCoordinator.create(world);
    const frame = coordinator.reduceWorld(PLANES, [0, 0, 0]);
    const result = coordinator.reduceEntities(
      {
        count: 1,
        positionX: Float32Array.from([20]),
        positionY: new Float32Array(1),
        positionZ: Float32Array.from([0.5]),
        radius: Float32Array.from([1]),
      },
      PLANES,
      frame,
      { camera: [0, 0, 0], maxDistance: 5, outsideWorldVisible: false }
    );
    expect(Array.from(result.visibleIndices)).toEqual([]);
    expect(result.flags[0]! & ShadoVisibilityBits.Distance).toBe(0);
  });
});
