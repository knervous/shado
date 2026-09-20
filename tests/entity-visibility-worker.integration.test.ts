import { Worker } from 'node:worker_threads';
import {
  ShadoEntityVisibilityWorker,
  compileShadoWorld,
  regionsForBounds,
  type ShadoVisibilityWorkerPort,
} from '../src/world';
import type { RegionGrid, ShadoWorldSpatialPackage } from '../src/world';

/**
 * A real worker, running the real source, over the real WASM.
 *
 * The fake-worker tests prove the controller's protocol; they cannot prove
 * that the shipped worker source and the shipped reducer agree with it,
 * because they replace both. This adapter gives the browser worker surface the
 * controller expects -- `postMessage`, `addEventListener('message')`,
 * `addEventListener('error')`, `terminate` -- backed by a Node worker thread
 * executing the exact string the browser would.
 */
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

/** One quad per region, so each region has a cell and a distinct PVS bit. */
function quad(x: number) {
  return {
    name: `quad-${x}`,
    material: 'stone',
    positions: new Float32Array([x, 0, 0, x + 1, 0, 0, x + 1, 1, 0, x, 1, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

const WIDE_PLANES = new Float32Array([
  1, 0, 0, 4096, -1, 0, 0, 4096, 0, 1, 0, 4096,
  0, -1, 0, 4096, 0, 0, 1, 4096, 0, 0, -1, 4096,
]);

/**
 * The oracle: full-bounds membership and the same policy bits, in plain
 * scalar code with no shared memory, no worker and no WASM.
 *
 * Deliberately a second implementation. Comparing the worker against itself
 * proves only that it is consistent.
 */
function scalarVisible(
  world: ShadoWorldSpatialPackage,
  entities: { x: number; z: number; radius: number; enabled?: boolean; phase?: number }[],
  regionFlags: Uint8Array,
  activePhaseMask = 0xffffffff
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
    if (entity.enabled === false) return;
    if (((entity.phase ?? 0xffffffff) & activePhaseMask) === 0) return;
    const membership = regionsForBounds(
      grid,
      entity.x - entity.radius,
      entity.z - entity.radius,
      entity.x + entity.radius,
      entity.z + entity.radius,
      scratch
    );
    if (membership.overflow) {
      visible.push(index);
      return;
    }
    if (!membership.regions.length) return;
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

async function waitForResult(worker: ShadoEntityVisibilityWorker, timeoutMs = 4000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = worker.acquireLatest();
    if (result) return result;
    await settle();
  }
  throw new Error('the real worker never published a result');
}

describe('the shipped worker and the shipped reducer agree with the oracle', () => {
  const world = compileShadoWorld([quad(0), quad(16), quad(32), quad(48)], {
    name: 'worker-integration',
    tileSize: 16,
    visibilityRegionSize: 16,
    maxClusterTriangles: 2,
  });

  let worker: ShadoEntityVisibilityWorker | null = null;
  afterEach(() => {
    worker?.dispose();
    worker = null;
  });

  async function run(
    entities: { x: number; z: number; radius: number; enabled?: boolean; phase?: number }[],
    loadedCells: number[],
    activePhaseMask = 0xffffffff
  ) {
    worker = await ShadoEntityVisibilityWorker.create(world, {
      capacity: Math.max(8, entities.length),
      publishFlags: false,
      workerFactory: nodeWorkerAdapter,
    });
    worker.projection.load({
      count: entities.length,
      positionX: entities.map((entry) => entry.x),
      positionY: entities.map(() => 0),
      positionZ: entities.map((entry) => entry.z),
      radius: entities.map((entry) => entry.radius),
    });
    entities.forEach((entry, index) => {
      if (entry.enabled === false || entry.phase !== undefined) {
        worker!.projection.setEntityPolicy(index, entry.enabled !== false, entry.phase ?? 0xffffffff);
      }
    });
    const regionFlags = new Uint8Array(world.visibility!.width * world.visibility!.height);
    loadedCells.forEach((value, region) => {
      regionFlags[region] = value ? 0x71 : 0x10;
    });
    worker.request(WIDE_PLANES, regionFlags, {
      camera: [0, 0, 0],
      outsideWorldVisible: false,
      activePhaseMask,
    });
    const result = await waitForResult(worker);
    return {
      worker: Array.from(result.visibleIndices),
      generations: Array.from(result.visibleGenerations),
      oracle: scalarVisible(world, entities, regionFlags, activePhaseMask),
    };
  }

  it('agrees on two adjacent regions with different PVS bits', async () => {
    const { worker: got, oracle } = await run(
      [
        { x: 8, z: 0.5, radius: 1 },
        { x: 24, z: 0.5, radius: 1 },
      ],
      [1, 0, 1, 1]
    );
    expect(got).toEqual(oracle);
    expect(got).toEqual([0]);
  }, 20000);

  it('agrees on an actor spanning a rejected and an admitted region', async () => {
    const { worker: got, oracle } = await run([{ x: 20, z: 0.5, radius: 6 }], [1, 0, 1, 1]);
    expect(got).toEqual(oracle);
    expect(got).toEqual([0]);
  }, 20000);

  it('agrees when a phase mask excludes an entity', async () => {
    const { worker: got, oracle } = await run(
      [
        { x: 8, z: 0.5, radius: 1, phase: 0b10 },
        { x: 8, z: 0.5, radius: 1, phase: 0b01 },
      ],
      [1, 1, 1, 1],
      0b01
    );
    expect(got).toEqual(oracle);
    expect(got).toEqual([1]);
  }, 20000);

  it('agrees when an entity is disabled', async () => {
    const { worker: got, oracle } = await run(
      [
        { x: 8, z: 0.5, radius: 1, enabled: false },
        { x: 8, z: 0.5, radius: 1 },
      ],
      [1, 1, 1, 1]
    );
    expect(got).toEqual(oracle);
    expect(got).toEqual([1]);
  }, 20000);

  it('agrees that an entity outside every region is not admitted', async () => {
    const { worker: got, oracle } = await run([{ x: 5000, z: 5000, radius: 1 }], [1, 1, 1, 1]);
    expect(got).toEqual(oracle);
    expect(got).toEqual([]);
  }, 20000);

  it('publishes the slot generation the shipped worker computed against', async () => {
    const { generations, worker: got } = await run([{ x: 8, z: 0.5, radius: 1 }], [1, 1, 1, 1]);
    expect(got).toEqual([0]);
    expect(generations).toEqual([1]);
  }, 20000);

  it('applies a delta sent after the first reduction', async () => {
    worker = await ShadoEntityVisibilityWorker.create(world, {
      capacity: 8,
      publishFlags: false,
      workerFactory: nodeWorkerAdapter,
    });
    worker.projection.load({
      count: 1,
      positionX: [8],
      positionY: [0],
      positionZ: [0.5],
      radius: [1],
    });
    const regionFlags = new Uint8Array(world.visibility!.width * world.visibility!.height);
    regionFlags.fill(0x71);
    worker.request(WIDE_PLANES, regionFlags, { camera: [0, 0, 0], outsideWorldVisible: false });
    expect(Array.from((await waitForResult(worker)).visibleIndices)).toEqual([0]);

    // Move it out of the world entirely and reuse its slot; the worker only
    // learns either fact from the delta batch.
    worker.projection.setEntity(0, 5000, 0, 5000, 1);
    worker.projection.reuseSlot(0);
    worker.request(WIDE_PLANES, regionFlags, { camera: [0, 0, 0], outsideWorldVisible: false });
    const second = await waitForResult(worker);
    expect(Array.from(second.visibleIndices)).toEqual([]);
  }, 20000);
});
