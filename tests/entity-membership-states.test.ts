import { Worker } from 'node:worker_threads';
import {
  ShadoEntityVisibilityWorker,
  classifyRegionMembership,
  compileShadoWorld,
  regionsForBounds,
  type ShadoVisibilityWorkerPort,
} from '../src/world';
import type { RegionGrid, ShadoWorldSpatialPackage } from '../src/world';

/** The real worker source in a Node thread; no stand-in anywhere. */
function nodeWorkerAdapter(source: string): ShadoVisibilityWorkerPort {
  const shim = `
    const { parentPort } = require('node:worker_threads');
    const self = {
      postMessage: (message, transfer) => parentPort.postMessage(message, transfer),
      onmessage: null,
    };
    parentPort.on('message', (data) => { void self.onmessage({ data }); });
  `;
  const worker = new Worker(`${shim}\n${source}`, { eval: true });
  const messageListeners: ((event: MessageEvent) => void)[] = [];
  const errorListeners: ((event: ErrorEvent) => void)[] = [];
  worker.on('message', (data) => {
    for (const listener of messageListeners) listener({ data } as MessageEvent);
  });
  worker.on('error', (error) => {
    for (const listener of errorListeners) {
      listener({ message: error.message, error } as unknown as ErrorEvent);
    }
  });
  return {
    postMessage: (message, transfer) => worker.postMessage(message, transfer as never),
    addEventListener: (type, listener) => {
      if (type === 'message') messageListeners.push(listener as (event: MessageEvent) => void);
      else errorListeners.push(listener as (event: ErrorEvent) => void);
    },
    terminate: () => void worker.terminate(),
  } as ShadoVisibilityWorkerPort;
}

const WIDE_PLANES = new Float32Array([
  1, 0, 0, 1e7, -1, 0, 0, 1e7, 0, 1, 0, 1e7,
  0, -1, 0, 1e7, 0, 0, 1, 1e7, 0, 0, -1, 1e7,
]);

const REGION = 16;

/** A quad in one region, so every region owns a cell. */
function quad(x: number, z: number) {
  return {
    name: `quad-${x}-${z}`,
    material: 'stone',
    positions: new Float32Array([
      x, 0, z, x + 1, 0, z, x + 1, 0, z + 1, x, 0, z + 1,
    ]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

type Entity = { x: number; z: number; radius: number };

/**
 * The oracle: the three membership states and the same policy bits, written
 * out in plain scalar code.
 *
 * It calls the shared classifier, which is the point — the worker's copy of
 * that classifier is spliced from this one's source, so a disagreement here
 * is a disagreement about POLICY, which is what this file is testing.
 */
function scalarVisible(
  world: ShadoWorldSpatialPackage,
  entities: Entity[],
  regionFlags: Uint8Array,
  outsideWorldVisible: boolean
): number[] {
  const visibility = world.visibility!;
  const grid: RegionGrid = {
    originX: visibility.originX,
    originZ: visibility.originZ,
    size: visibility.size,
    width: visibility.width,
    height: visibility.height,
  };
  const required = 0x71;
  const scratch = new Uint32Array(64);
  const visible: number[] = [];
  entities.forEach((entity, index) => {
    const membership = regionsForBounds(
      grid,
      entity.x - entity.radius,
      entity.z - entity.radius,
      entity.x + entity.radius,
      entity.z + entity.radius,
      scratch
    );
    if (membership.state === 'unknown') {
      visible.push(index);
      return;
    }
    if (membership.state === 'whollyOutside') {
      if (outsideWorldVisible) visible.push(index);
      return;
    }
    for (const region of membership.regions) {
      if (((regionFlags[region] ?? 0) & required) === required) {
        visible.push(index);
        return;
      }
    }
  });
  return visible;
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

async function waitForResult(worker: ShadoEntityVisibilityWorker, timeoutMs = 8000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = worker.acquireLatest();
    if (result) return result;
    await settle();
  }
  throw new Error('the real worker never published a result');
}

describe('unknown is not outside, and neither is dropped', () => {
  /** A 16 x 16 grid of regions, so a bound can be well inside it. */
  const primitives: ReturnType<typeof quad>[] = [];
  for (let z = 0; z < 16; z += 1) {
    for (let x = 0; x < 16; x += 1) primitives.push(quad(x * REGION, z * REGION));
  }
  const world = compileShadoWorld(primitives, {
    name: 'membership-states',
    tileSize: REGION,
    visibilityRegionSize: REGION,
    maxClusterTriangles: 2,
  });

  /** Astra's reproduction shape: 80 regions in a row, one very wide actor. */
  const strip: ReturnType<typeof quad>[] = [];
  for (let x = 0; x < 80; x += 1) strip.push(quad(x * REGION, 0));
  const stripWorld = compileShadoWorld(strip, {
    name: 'membership-strip',
    tileSize: REGION,
    visibilityRegionSize: REGION,
    maxClusterTriangles: 2,
  });

  let worker: ShadoEntityVisibilityWorker | null = null;
  afterEach(() => {
    worker?.dispose();
    worker = null;
  });

  async function run(
    target: ShadoWorldSpatialPackage,
    entities: Entity[],
    options: {
      outsideWorldVisible?: boolean;
      allVisible?: boolean;
      publishFlags?: boolean;
      hiddenRegions?: number[];
    } = {}
  ) {
    const visibility = target.visibility!;
    const regionCount = visibility.width * visibility.height;
    worker = await ShadoEntityVisibilityWorker.create(target, {
      capacity: Math.max(8, entities.length),
      publishFlags: options.publishFlags ?? false,
      workerFactory: nodeWorkerAdapter,
    });
    worker.projection.load({
      count: entities.length,
      positionX: entities.map((entry) => entry.x),
      positionY: entities.map(() => 0),
      positionZ: entities.map((entry) => entry.z),
      radius: entities.map((entry) => entry.radius),
    });
    const regionFlags = new Uint8Array(regionCount);
    regionFlags.fill(options.allVisible === false ? 0x10 : 0x71);
    for (const region of options.hiddenRegions ?? []) regionFlags[region] = 0x10;
    const outsideWorldVisible = options.outsideWorldVisible ?? false;
    worker.request(WIDE_PLANES, regionFlags, { camera: [0, 0, 0], outsideWorldVisible });
    const result = await waitForResult(worker);
    return {
      worker: Array.from(result.visibleIndices).sort((a, b) => a - b),
      oracle: scalarVisible(target, entities, regionFlags, outsideWorldVisible),
    };
  }

  it('keeps an actor too wide to enumerate, with outsideWorldVisible false', async () => {
    /*
     * The audit's reproduction. 80 regions of size 16, every region visible,
     * one actor at x = 640 with radius 600, and the outside policy the client
     * uses whenever it has a world coordinator. The actor touches far more
     * than 64 regions, so its membership is unknown -- and unknown is not
     * outside, so the policy must not reach it.
     */
    const { worker: got, oracle } = await run(stripWorld, [{ x: 640, z: 0.5, radius: 600 }], {
      outsideWorldVisible: false,
    });
    expect(got).toEqual([0]);
    expect(oracle).toEqual([0]);
  }, 20000);

  it('keeps it even when every region is hidden', async () => {
    /*
     * Unknown membership is not an admission by the regions; it is the
     * absence of a proof. The frustum, range, phase and enabled tests still
     * decide, and none of them is skipped here -- but topology may not
     * reject what it never classified.
     */
    const { worker: got, oracle } = await run(stripWorld, [{ x: 640, z: 0.5, radius: 600 }], {
      outsideWorldVisible: false,
      allVisible: false,
    });
    expect(got).toEqual([0]);
    expect(oracle).toEqual([0]);
  }, 20000);

  it('drops a wholly outside actor under one policy and keeps it under the other', async () => {
    const far: Entity[] = [{ x: 500_000, z: 500_000, radius: 1 }];
    const hidden = await run(stripWorld, far, { outsideWorldVisible: false });
    expect(hidden.worker).toEqual([]);
    expect(hidden.oracle).toEqual([]);
    worker?.dispose();
    worker = null;
    const shown = await run(stripWorld, far, { outsideWorldVisible: true });
    expect(shown.worker).toEqual([0]);
    expect(shown.oracle).toEqual([0]);
  }, 20000);

  it('keeps an actor with invalid bounds whatever the outside policy says', async () => {
    const { worker: got, oracle } = await run(
      stripWorld,
      [{ x: Number.NaN, z: 0.5, radius: 1 }],
      { outsideWorldVisible: false }
    );
    expect(got).toEqual([0]);
    expect(oracle).toEqual([0]);
  }, 20000);

  it('keeps an actor only partly over the grid, rather than clipping it', async () => {
    /*
     * Half in, half out. Clipping and enumerating the half that is inside
     * would assert that no camera is in the half that is not.
     */
    const { worker: got, oracle } = await run(world, [{ x: 8, z: 8, radius: 64 }], {
      outsideWorldVisible: false,
      allVisible: false,
    });
    expect(got).toEqual([0]);
    expect(oracle).toEqual([0]);
  }, 20000);

  it('enumerates at the cap and goes unknown past it', async () => {
    /*
     * A bound covering 8 x 8 regions is enumerated and answers to its
     * regions; one covering 9 x 9 cannot be, and is unknown. Both are placed
     * well inside the grid so the only thing distinguishing them is the cap.
     */
    const centre = 8 * REGION;
    const inside: Entity[] = [{ x: centre, z: centre, radius: 4 * REGION - 0.5 }];
    const over: Entity[] = [{ x: centre, z: centre, radius: 4 * REGION + 0.5 }];
    const hiddenEverywhere = { outsideWorldVisible: false, allVisible: false } as const;

    const capped = await run(world, inside, hiddenEverywhere);
    expect(capped.worker).toEqual([]);
    expect(capped.oracle).toEqual([]);
    worker?.dispose();
    worker = null;
    const past = await run(world, over, hiddenEverywhere);
    expect(past.worker).toEqual([0]);
    expect(past.oracle).toEqual([0]);
  }, 20000);

  it('agrees with the oracle in full-flags mode as well as compact', async () => {
    /*
     * Compact mode walks admitted buckets and never visits the rest; full
     * flags walks every entity. The unknown bucket has to be taken in both,
     * and the two modes must not disagree about one entity.
     */
    const entities: Entity[] = [
      { x: 8, z: 8, radius: 1 },
      { x: 640, z: 8, radius: 600 },
      { x: 500_000, z: 500_000, radius: 1 },
    ];
    const compact = await run(stripWorld, entities, { outsideWorldVisible: false });
    worker?.dispose();
    worker = null;
    const full = await run(stripWorld, entities, {
      outsideWorldVisible: false,
      publishFlags: true,
    });
    expect(compact.worker).toEqual(compact.oracle);
    expect(full.worker).toEqual(full.oracle);
    expect(full.worker).toEqual(compact.worker);
  }, 20000);

  it('follows a radius-only change from enumerated into unknown and back', async () => {
    /*
     * Membership is cached per slot and repaired when the slot is dirty. A
     * radius change moves an actor between buckets without moving it at all,
     * which is the case a position-keyed cache misses.
     */
    const visibility = world.visibility!;
    const regionCount = visibility.width * visibility.height;
    worker = await ShadoEntityVisibilityWorker.create(world, {
      capacity: 4,
      publishFlags: false,
      workerFactory: nodeWorkerAdapter,
    });
    worker.projection.load({
      count: 1,
      positionX: [8 * REGION],
      positionY: [0],
      positionZ: [8 * REGION],
      radius: [1],
    });
    const regionFlags = new Uint8Array(regionCount).fill(0x10);
    const ask = async () => {
      worker!.request(WIDE_PLANES, regionFlags, {
        camera: [0, 0, 0],
        outsideWorldVisible: false,
      });
      return Array.from((await waitForResult(worker!)).visibleIndices);
    };

    expect(await ask()).toEqual([]);
    worker.projection.setEntity(0, 8 * REGION, 0, 8 * REGION, 4 * REGION + 0.5);
    expect(await ask()).toEqual([0]);
    worker.projection.setEntity(0, 8 * REGION, 0, 8 * REGION, 1);
    expect(await ask()).toEqual([]);
  }, 20000);

  it('treats a region with no cell behind it as unknown, not as outside', async () => {
    /*
     * A sparse world has regions the lookup cannot resolve to a cell. An
     * actor standing in one has no row to be judged by -- which is unknown.
     * Sending it to the outside bucket instead made it answerable to
     * outsideWorldVisible, and under the client's `false` it vanished while
     * standing on ground the player can walk to.
     */
    const gap = compileShadoWorld([quad(0, 0), quad(3 * REGION, 0)], {
      name: 'membership-sparse',
      tileSize: REGION,
      visibilityRegionSize: REGION,
      maxClusterTriangles: 2,
    });
    const { worker: got, oracle } = await run(
      gap,
      // Mid-region, in the empty column between the two quads.
      [{ x: 1.5 * REGION, z: 0.5 * REGION, radius: 1 }],
      { outsideWorldVisible: false, allVisible: false }
    );
    expect(got).toEqual(oracle);
  }, 20000);

  it('refuses radiusScale rather than binning by one radius and testing another', async () => {
    worker = await ShadoEntityVisibilityWorker.create(world, {
      capacity: 4,
      publishFlags: false,
      workerFactory: nodeWorkerAdapter,
    });
    worker.projection.load({
      count: 1,
      positionX: [8],
      positionY: [0],
      positionZ: [8],
      radius: [1],
    });
    const regionFlags = new Uint8Array(world.visibility!.width * world.visibility!.height);
    expect(() =>
      worker!.request(WIDE_PLANES, regionFlags, {
        camera: [0, 0, 0],
        outsideWorldVisible: false,
        radiusScale: 2,
      })
    ).toThrow(/radiusScale/);
  }, 20000);
});

describe('the enumeration cap, exactly', () => {
  const grid: RegionGrid = {
    originX: 0,
    originZ: 0,
    size: 1,
    width: 100,
    height: 100,
  };
  const scratch = new Uint32Array(64);

  /*
   * A circle can only ever produce a square box, so 64 versus 65 is asked of
   * the classifier directly with a rectangle. 13 x 5 is 65; 12 x 5 is 60.
   */
  it('enumerates 64 regions and refuses 65', () => {
    const at = (w: number, h: number) =>
      classifyRegionMembership(0, 0, 1, 0, 0, 100, 100, 10.5, 10.5, 10.5 + w - 1, 10.5 + h - 1, 64, scratch);
    expect(at(8, 8)).toBe(64);
    expect(at(13, 5)).toBe(-1);
    expect(at(12, 5)).toBe(60);
  });

  it('refuses a span the buffer the caller supplied cannot hold', () => {
    const small = new Uint32Array(4);
    expect(classifyRegionMembership(0, 0, 1, 0, 0, 100, 100, 10.5, 10.5, 13.5, 13.5, 64, small)).toBe(-1);
  });

  it('reports the three states apart from each other', () => {
    expect(regionsForBounds(grid, 10, 10, 12, 12, scratch).state).toBe('enumerated');
    expect(regionsForBounds(grid, -50, -50, -10, -10, scratch).state).toBe('whollyOutside');
    expect(regionsForBounds(grid, -5, 10, 5, 12, scratch).state).toBe('unknown');
    expect(regionsForBounds(grid, Number.NaN, 10, 5, 12, scratch).state).toBe('unknown');
  });
});
