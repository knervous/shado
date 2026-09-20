import {
  ShadoEntityVisibilityWorker,
  ShadoVisibilityWorkerControl,
  compileShadoWorld,
  type ShadoEntityVisibilityWorkerLayout,
} from '../src/world';

function world() {
  return compileShadoWorld(
    [
      {
        name: 'quad',
        material: 'stone',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
      },
    ],
    { name: 'ownership-proof', tileSize: 16 }
  );
}

type Delta = {
  slots: Uint32Array;
  generations: Uint32Array;
  transforms: Float32Array;
  policy: Uint32Array;
  enabled: Uint8Array;
  count: number;
};

/**
 * A worker that owns its state the way the real one does.
 *
 * It never reads the shared projection: everything it knows arrives in the
 * delta batches its requests carry, which is the property under test. Visible
 * means "x >= 0", so a move across zero is a reveal or a hide.
 */
class OwningFakeWorker {
  public readonly seenDeltas: Delta[] = [];
  public terminated = false;

  private readonly listeners: Array<(event: MessageEvent) => void> = [];
  private init?: { buffer: SharedArrayBuffer; layout: ShadoEntityVisibilityWorkerLayout };
  private pending: { generation: number }[] = [];
  private x = new Float32Array(0);
  private generation = new Uint32Array(0);
  private enabled = new Uint8Array(0);
  private count = 0;

  public postMessage(message: unknown): void {
    const value = message as {
      type: string;
      generation?: number;
      delta?: Delta;
      buffer?: SharedArrayBuffer;
      layout?: ShadoEntityVisibilityWorkerLayout;
    };
    if (value.type === 'init') {
      this.init = { buffer: value.buffer!, layout: value.layout! };
      this.x = new Float32Array(value.layout!.capacity);
      this.generation = new Uint32Array(value.layout!.capacity);
      this.enabled = new Uint8Array(value.layout!.capacity);
      queueMicrotask(() => this.emit({ type: 'ready' }));
      return;
    }
    if (value.type !== 'reduce') return;
    const delta = value.delta!;
    this.seenDeltas.push(delta);
    this.count = delta.count;
    for (let index = 0; index < delta.slots.length; index += 1) {
      const slot = delta.slots[index]!;
      this.x[slot] = delta.transforms[index * 4]!;
      this.enabled[slot] = delta.enabled[index]!;
      this.generation[slot] = delta.generations[index]!;
    }
    this.pending.push({ generation: value.generation! });
  }

  public addEventListener(type: string, listener: (event: MessageEvent) => void): void {
    if (type === 'message') this.listeners.push(listener);
  }

  public terminate(): void {
    this.terminated = true;
  }

  public completeNext(): void {
    const request = this.pending.shift();
    if (!request || !this.init) throw new Error('nothing pending');
    const { buffer, layout } = this.init;
    const control = new Int32Array(buffer, layout.controlOffset, 16);
    const output = 1 - Atomics.load(control, ShadoVisibilityWorkerControl.PublishedOutputBuffer);
    const indices = new Uint32Array(buffer, layout.visibleIndicesOffsets[output], layout.capacity);
    const generations = new Uint32Array(
      buffer,
      layout.resultGenerationOffsets[output],
      layout.capacity
    );
    let visible = 0;
    for (let slot = 0; slot < this.count; slot += 1) {
      if (this.x[slot]! < 0 || !this.enabled[slot]) continue;
      generations[visible] = this.generation[slot]!;
      indices[visible++] = slot;
    }
    Atomics.store(
      control,
      output === 0
        ? ShadoVisibilityWorkerControl.ResultCount0
        : ShadoVisibilityWorkerControl.ResultCount1,
      visible
    );
    Atomics.store(
      control,
      output === 0
        ? ShadoVisibilityWorkerControl.ResultEntityCount0
        : ShadoVisibilityWorkerControl.ResultEntityCount1,
      0
    );
    Atomics.store(control, ShadoVisibilityWorkerControl.PublishedOutputBuffer, output);
    Atomics.store(control, ShadoVisibilityWorkerControl.CompletedGeneration, request.generation);
    this.emit({ type: 'complete', generation: request.generation });
  }

  /** Overwrites both output buffers, as a worker reusing them would. */
  public scribbleOutputs(): void {
    const { buffer, layout } = this.init!;
    for (const offset of layout.visibleIndicesOffsets) {
      new Uint32Array(buffer, offset, layout.capacity).fill(0xdead);
    }
  }

  private emit(data: unknown): void {
    for (const listener of this.listeners) listener({ data } as MessageEvent);
  }
}

const PLANES = new Float32Array(24);
const CAMERA = { camera: [0, 0, 0] as [number, number, number] };

async function makeWorker() {
  const port = new OwningFakeWorker();
  const worker = await ShadoEntityVisibilityWorker.create(world(), {
    capacity: 8,
    publishFlags: false,
    workerFactory: () => port,
  });
  worker.projection.load({
    count: 3,
    positionX: [1, 2, 3],
    positionY: [0, 0, 0],
    positionZ: [0, 0, 0],
    radius: [1, 1, 1],
  });
  return { port, worker };
}

describe('worker input ownership', () => {
  it('sends a full snapshot first and only changes afterwards', async () => {
    const { port, worker } = await makeWorker();
    worker.request(PLANES, [0xff], CAMERA);
    expect(Array.from(port.seenDeltas[0]!.slots)).toEqual([0, 1, 2]);

    port.completeNext();
    // A camera move with nothing else changed carries no slot work at all.
    worker.request(PLANES, [0xff], CAMERA);
    expect(Array.from(port.seenDeltas[1]!.slots)).toEqual([]);
  });

  it('never loses a move that happened while the worker was busy', async () => {
    const { port, worker } = await makeWorker();
    worker.request(PLANES, [0xff], CAMERA);
    port.completeNext();

    // Entity 1 crosses to the hidden side while a request is in flight, and
    // then the camera moves twice more before the worker frees up.
    worker.request(PLANES, [0xff], CAMERA);
    worker.projection.setEntity(1, -5, 0, 0, 1);
    worker.request(PLANES, [0xff], CAMERA);
    worker.request(PLANES, [0xff], CAMERA);
    port.completeNext();

    // The superseded request's batch was handed on, not dropped with it.
    const dispatched = port.seenDeltas[port.seenDeltas.length - 1]!;
    expect(Array.from(dispatched.slots)).toEqual([1]);
    expect(dispatched.transforms[0]).toBe(-5);

    port.completeNext();
    const result = worker.acquireLatest();
    expect(Array.from(result!.visibleIndices)).toEqual([0, 2]);
  });

  it('coalesces repeated changes to one slot into a single entry', async () => {
    const { port, worker } = await makeWorker();
    worker.request(PLANES, [0xff], CAMERA);
    port.completeNext();
    worker.projection.setEntity(1, 10, 0, 0, 1);
    worker.projection.setEntity(1, 11, 0, 0, 1);
    worker.projection.setEntity(1, 12, 0, 0, 1);
    worker.request(PLANES, [0xff], CAMERA);
    const delta = port.seenDeltas[1]!;
    expect(Array.from(delta.slots)).toEqual([1]);
    expect(delta.transforms[0]).toBe(12);
  });
});

describe('worker result ownership', () => {
  it('reports the slot generation each visible index was computed for', async () => {
    const { port, worker } = await makeWorker();
    worker.request(PLANES, [0xff], CAMERA);
    port.completeNext();
    const first = worker.acquireLatest()!;
    expect(Array.from(first.visibleGenerations)).toEqual([1, 1, 1]);

    /*
     * Slot 1 is handed to a new occupant. A result computed for the previous
     * one carries the old generation, so an adapter can tell that it is not
     * about the entity standing there now.
     */
    const next = worker.projection.reuseSlot(1);
    expect(next).toBe(2);
    worker.request(PLANES, [0xff], CAMERA);
    port.completeNext();
    const second = worker.acquireLatest()!;
    expect(Array.from(second.visibleGenerations)).toEqual([1, 2, 1]);
  });

  it('refuses a result computed against a topology that has since changed', async () => {
    const { port, worker } = await makeWorker();
    worker.setEpochs({ world: 1, topology: 7, policy: 1 });
    worker.request(PLANES, [0xff], CAMERA);
    // The zone's topology is replaced while the worker is mid-flight.
    worker.setEpochs({ topology: 8 });
    port.completeNext();
    expect(worker.acquireLatest()).toBeNull();
    expect(worker.stats.staleEpochResults).toBe(1);
  });

  it('marks a result older than the permitted age instead of hiding with it', async () => {
    const { port, worker } = await makeWorker();
    worker.maxResultAgeMs = -1; // Everything is too old.
    worker.request(PLANES, [0xff], CAMERA);
    port.completeNext();
    const result = worker.acquireLatest()!;
    expect(result.stale).toBe(true);
    expect(worker.stats.staleAgeResults).toBe(1);
    // Still delivered: the caller decides to fall back, rather than being
    // handed nothing and guessing why.
    expect(Array.from(result.visibleIndices)).toEqual([0, 1, 2]);
  });

  it('hands out a copy, not a view the worker will overwrite', async () => {
    const { port, worker } = await makeWorker();
    worker.request(PLANES, [0xff], CAMERA);
    port.completeNext();
    const result = worker.acquireLatest()!;
    const before = Array.from(result.visibleIndices);
    port.scribbleOutputs();
    expect(Array.from(result.visibleIndices)).toEqual(before);
  });
});
