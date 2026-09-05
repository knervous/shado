import type { ShadoSprite2DMotionConfig } from '@knervous/shado/render';
import type {
  Sprite2DMotionWorkerRequest,
  Sprite2DMotionWorkerResponse,
} from './Sprite2DMotionWorkerProtocol';

export type Sprite2DMotionShardResult = {
  start: number;
  positions: Float32Array;
};

type WorkerSlot = {
  worker: Worker;
  pending: Map<number, {
    resolve(response: Sprite2DMotionWorkerResponse): void;
    reject(error: Error): void;
  }>;
};

export class Sprite2DMotionWorkerPool {
  private readonly slots: WorkerSlot[] = [];
  private activeSlots: WorkerSlot[] = [];
  private requestId = 0;
  private ready: Promise<void> = Promise.resolve();

  public constructor(public readonly maxWorkers: number) {}

  public get activeWorkers(): number {
    return this.activeSlots.length;
  }

  public setPopulation(
    positions: Float32Array,
    config: ShadoSprite2DMotionConfig,
    nowMs = performance.now()
  ): Promise<void> {
    const count = positions.length / 2;
    const desiredWorkers = Math.min(
      this.maxWorkers,
      Math.max(1, Math.ceil(count / 16_384))
    );
    while (this.slots.length < desiredWorkers) this.slots.push(this.createSlot());
    this.activeSlots = this.slots.slice(0, desiredWorkers);
    const shardSize = Math.ceil(count / desiredWorkers);
    const slots = [...this.activeSlots];
    const previousReady = this.ready;
    this.ready = previousReady.then(() => Promise.all(slots.map((slot, shardIndex) => {
      const start = shardIndex * shardSize;
      const end = Math.min(count, start + shardSize);
      const shard = positions.slice(start * 2, end * 2);
      return this.request(slot, {
        type: 'init',
        requestId: this.nextRequestId(),
        start,
        positions: shard.buffer,
        config,
        nowMs,
      }, [shard.buffer]).then(() => undefined);
    }))).then(() => undefined);
    return this.ready;
  }

  public async configure(
    config: ShadoSprite2DMotionConfig,
    nowMs = performance.now()
  ): Promise<void> {
    await this.ready;
    await Promise.all(this.activeSlots.map(slot => this.request(slot, {
      type: 'configure',
      requestId: this.nextRequestId(),
      config,
      nowMs,
    })));
  }

  public async step(nowMs: number, dtSeconds: number): Promise<Sprite2DMotionShardResult[]> {
    await this.ready;
    const responses = await Promise.all(this.activeSlots.map(slot => this.request(slot, {
      type: 'step',
      requestId: this.nextRequestId(),
      nowMs,
      dtSeconds,
    })));
    return responses.map(response => {
      if (response.type !== 'result') throw new Error('Motion worker returned no result');
      return { start: response.start, positions: new Float32Array(response.positions) };
    });
  }

  public dispose(): void {
    for (const slot of this.slots) {
      for (const pending of slot.pending.values()) {
        pending.reject(new Error('Motion worker pool disposed'));
      }
      slot.pending.clear();
      slot.worker.terminate();
    }
    this.slots.length = 0;
    this.activeSlots = [];
  }

  private createSlot(): WorkerSlot {
    const worker = new Worker(new URL('./Sprite2DMotionWorker.ts', import.meta.url), {
      type: 'module',
      name: `shado-2d-motion-${this.slots.length}`,
    });
    const slot: WorkerSlot = { worker, pending: new Map() };
    worker.onmessage = (event: MessageEvent<Sprite2DMotionWorkerResponse>) => {
      const response = event.data;
      const pending = slot.pending.get(response.requestId);
      if (!pending) return;
      slot.pending.delete(response.requestId);
      if (response.type === 'error') pending.reject(new Error(response.message));
      else pending.resolve(response);
    };
    worker.onerror = event => {
      const error = new Error(event.message || 'Motion worker failed');
      for (const pending of slot.pending.values()) pending.reject(error);
      slot.pending.clear();
    };
    return slot;
  }

  private request(
    slot: WorkerSlot,
    message: Sprite2DMotionWorkerRequest,
    transfer: Transferable[] = []
  ): Promise<Sprite2DMotionWorkerResponse> {
    return new Promise((resolve, reject) => {
      slot.pending.set(message.requestId, { resolve, reject });
      slot.worker.postMessage(message, transfer);
    });
  }

  private nextRequestId(): number {
    return ++this.requestId;
  }
}
