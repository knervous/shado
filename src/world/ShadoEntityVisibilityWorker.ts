import { SHADO_WORLD_REDUCER_WASM_BASE64 } from './world-reducer-wasm.generated';
import { REGION_MEMBERSHIP_SOURCE } from './region-membership';
import type { ShadoWorldSpatialPackage, WorldVec3 } from './types';
import type {
  ShadoEntityVisibilitySoA,
  ShadoEntityVisibilityOptions,
} from './ShadoWorldVisibilityCoordinator';

const CONTROL_LENGTH = 16;

export const ShadoVisibilityWorkerControl = {
  RequestedGeneration: 0,
  CompletedGeneration: 1,
  PublishedOutputBuffer: 2,
  EntityCount: 3,
  ResultCount0: 4,
  ResultCount1: 5,
  SpatialRevision: 6,
  WorkerDurationMicros: 7,
  WorkerState: 8,
  ResultEntityCount0: 9,
  ResultEntityCount1: 10,
  CandidateCount: 11,
  HierarchyRebuildMicros: 12,
  CopiedInputBytes: 13,
  PublishedFlagBytes: 14,
} as const;

export type ShadoEntityVisibilityWorkerLayout = {
  byteLength: number;
  capacity: number;
  controlOffset: number;
  positionXOffset: number;
  positionYOffset: number;
  positionZOffset: number;
  radiusOffset: number;
  enabledOffset: number;
  phaseMaskOffset: number;
  /** Per-slot generation, bumped whenever a slot is reused by a new occupant. */
  slotGenerationOffset: number;
  /** Generation of each visible slot, published beside the compact indices. */
  resultGenerationOffsets: readonly [number, number];
  visibleIndicesOffsets: readonly [number, number];
  flagsOffsets: readonly [number, number];
  flagsCapacity: number;
};

export type ShadoEntityVisibilityWorkerResult = {
  generation: number;
  visibleIndices: Uint32Array;
  /**
   * The slot generation each visible index was computed for.
   *
   * A slot reused by a new occupant carries a new generation, so a result
   * computed for the previous occupant can be recognised and dropped instead
   * of revealing or hiding whoever holds the slot now.
   */
  visibleGenerations: Uint32Array;
  flags: Uint8Array;
  /** Milliseconds between the request being dispatched and this being read. */
  ageMs: number;
  /** The epochs this result was computed against, for the caller to match. */
  epochs: ShadoEntityVisibilityEpochs;
  /** True when the result is older than the caller's permitted age. */
  stale: boolean;
  workerDurationMs: number;
  candidateCount: number;
  hierarchyRebuildMs: number;
  copiedInputBytes: number;
  publishedFlagBytes: number;
};

/**
 * What a result was computed against.
 *
 * A result is only applicable to the state it was asked about: a different
 * package, a changed topology or a changed policy makes an old hidden
 * decision unsafe immediately, rather than merely out of date.
 */
export type ShadoEntityVisibilityEpochs = {
  world: number;
  topology: number;
  policy: number;
};

export type ShadoEntityVisibilityWorkerStats = {
  requestedGeneration: number;
  completedGeneration: number;
  workerDurationMs: number;
  inFlight: boolean;
  hasPendingRequest: boolean;
  candidateCount: number;
  hierarchyRebuildMs: number;
  copiedInputBytes: number;
  publishedFlagBytes: number;
  scheduledSkips: number;
  error: string | null;
  /** Results discarded because their epochs no longer match the caller's. */
  staleEpochResults: number;
  /** Results delivered past the permitted age; the caller falls back. */
  staleAgeResults: number;
  /** Age of the last delivered result, in milliseconds. */
  lastResultAgeMs: number;
  /** Slots whose deltas are waiting for the next dispatch. */
  pendingDeltaSlots: number;
};

type WorkerRequest = {
  type: 'reduce';
  generation: number;
  planes: Float32Array;
  cellFlags: Uint8Array;
  camera: WorldVec3;
  maxDistance: number;
  outsideWorldVisible: boolean;
  activePhaseMask: number;
  /** Slot changes this request carries; applied before it reduces. */
  delta: ShadoEntityVisibilityDelta;
  /** What this request is asking about; a result only applies to these. */
  epochs: ShadoEntityVisibilityEpochs;
  /** When it was dispatched, for the age the caller is allowed to accept. */
  dispatchedAtMs: number;
};

type WorkerMessage =
  { type: 'ready' } | { type: 'complete'; generation: number } | { type: 'error'; message: string };

export type ShadoVisibilityWorkerPort = {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerMessage>) => void): void;
  addEventListener(type: 'error', listener: (event: ErrorEvent) => void): void;
  terminate(): void;
};

export type ShadoEntityVisibilityWorkerOptions = {
  capacity: number;
  /** Publish entity-indexed reason flags. Compact-only mode avoids the full-size copy. */
  publishFlags?: boolean;
  workerFactory?: (source: string) => ShadoVisibilityWorkerPort;
};

export type ShadoEntityVisibilitySchedule = {
  cameraEpoch: number;
  cellEpoch: number;
  policyEpoch?: number;
  minimumIntervalMs?: number;
  nowMs?: number;
  force?: boolean;
};

export type ShadoEntityVisibilityWorkerWorld = {
  tiles: Pick<ShadoWorldSpatialPackage['tiles'], 'x' | 'z' | 'size' | 'originX' | 'originZ'>;
  visibility?: Pick<
    NonNullable<ShadoWorldSpatialPackage['visibility']>,
    'size' | 'originX' | 'originZ' | 'width' | 'height'
  >;
};

/**
 * Fixed-capacity, SharedArrayBuffer-backed visibility projection.
 *
 * Update only entities that moved or changed bounds. Publishing a visibility
 * request never walks this projection, so request cost does not grow with the
 * total entity count.
 */
/**
 * One immutable batch of slot changes, handed to the worker with a request.
 *
 * Packed rather than per-slot messages, and drained rather than re-sent: a
 * camera move must not copy the whole population, and a population change
 * must not be lost because a camera move replaced the request carrying it.
 */
export type ShadoEntityVisibilityDelta = {
  /** Slots this batch describes. */
  slots: Uint32Array;
  /** Generation of each slot at the moment it was drained. */
  generations: Uint32Array;
  /** x, y, z, radius per slot, in slot order. */
  transforms: Float32Array;
  /**
   * The phase mask per slot, untouched.
   *
   * Deliberately not packed with `enabled`: bit 0 of a phase mask is a real
   * phase, and borrowing it to carry a boolean made an entity in phase 2
   * answer to a request for phase 1.
   */
  policy: Uint32Array;
  /** Whether each slot is enabled, in its own byte. */
  enabled: Uint8Array;
  /** Entity count at drain time; the worker resizes to it. */
  count: number;
};

export class ShadoEntityVisibilityProjection {
  public readonly positionX: Float32Array;
  public readonly positionY: Float32Array;
  public readonly positionZ: Float32Array;
  public readonly radius: Float32Array;
  public readonly enabled: Uint8Array;
  public readonly phaseMask: Uint32Array;
  /** Per-slot generation; a reused slot is a different entity. */
  public readonly slotGeneration: Uint32Array;
  /** Slots changed since the last drain, in insertion order, deduplicated. */
  private readonly dirty = new Set<number>();
  /** Set once the whole population must be resent, e.g. after a bulk load. */
  private dirtyAll = true;

  public constructor(
    public readonly buffer: SharedArrayBuffer,
    public readonly layout: ShadoEntityVisibilityWorkerLayout
  ) {
    this.positionX = new Float32Array(buffer, layout.positionXOffset, layout.capacity);
    this.positionY = new Float32Array(buffer, layout.positionYOffset, layout.capacity);
    this.positionZ = new Float32Array(buffer, layout.positionZOffset, layout.capacity);
    this.radius = new Float32Array(buffer, layout.radiusOffset, layout.capacity);
    this.enabled = new Uint8Array(buffer, layout.enabledOffset, layout.capacity);
    this.phaseMask = new Uint32Array(buffer, layout.phaseMaskOffset, layout.capacity);
    this.slotGeneration = new Uint32Array(buffer, layout.slotGenerationOffset, layout.capacity);
  }

  public get capacity(): number {
    return this.layout.capacity;
  }

  public get count(): number {
    return Atomics.load(this.control, ShadoVisibilityWorkerControl.EntityCount);
  }

  public set count(value: number) {
    const count = Math.max(0, value | 0);
    if (count > this.capacity) {
      throw new RangeError(
        `Visibility projection count ${count} exceeds reserved capacity ${this.capacity}`
      );
    }
    Atomics.store(this.control, ShadoVisibilityWorkerControl.EntityCount, count);
    Atomics.add(this.control, ShadoVisibilityWorkerControl.SpatialRevision, 1);
  }

  public setEntity(index: number, x: number, y: number, z: number, radius: number): void {
    this.assertIndex(index);
    this.positionX[index] = x;
    this.positionY[index] = y;
    this.positionZ[index] = z;
    this.radius[index] = Math.max(0, radius);
    this.dirty.add(index);
    this.markSpatialChange();
  }

  public setEntityPolicy(index: number, enabled: boolean, phaseMask = 0xffffffff): void {
    this.assertIndex(index);
    this.enabled[index] = enabled ? 1 : 0;
    this.phaseMask[index] = phaseMask >>> 0;
    this.dirty.add(index);
    // Policy decides admission, so an old hidden decision made under the
    // previous policy is not merely stale -- it is wrong now.
    this.markSpatialChange();
  }

  /**
   * Hands a slot to a new occupant, so results for the previous one can be
   * recognised and dropped rather than applied to whoever holds it now.
   */
  public reuseSlot(index: number): number {
    this.assertIndex(index);
    const next = (this.slotGeneration[index]! + 1) >>> 0;
    this.slotGeneration[index] = next === 0 ? 1 : next;
    this.dirty.add(index);
    this.markSpatialChange();
    return this.slotGeneration[index]!;
  }

  /** Slots whose changes have not been handed to a request yet. */
  public get pendingDeltaSlots(): number {
    return this.dirtyAll ? this.count : this.dirty.size;
  }

  /**
   * Takes every pending change as one immutable batch and clears the record.
   *
   * Drained, not copied: a slot that changes twice between dispatches is one
   * entry, and a dispatch that carries the batch is the only thing that clears
   * it -- so a request replaced by a later camera move hands its batch on
   * rather than dropping it.
   */
  public drainDelta(): ShadoEntityVisibilityDelta {
    const count = this.count;
    const slots = this.dirtyAll
      ? Uint32Array.from({ length: count }, (_, index) => index)
      : Uint32Array.from([...this.dirty].filter((slot) => slot < count));
    const generations = new Uint32Array(slots.length);
    const transforms = new Float32Array(slots.length * 4);
    const policy = new Uint32Array(slots.length);
    const enabled = new Uint8Array(slots.length);
    for (let index = 0; index < slots.length; index += 1) {
      const slot = slots[index]!;
      generations[index] = this.slotGeneration[slot]!;
      transforms[index * 4] = this.positionX[slot]!;
      transforms[index * 4 + 1] = this.positionY[slot]!;
      transforms[index * 4 + 2] = this.positionZ[slot]!;
      transforms[index * 4 + 3] = this.radius[slot]!;
      policy[index] = this.phaseMask[slot]! >>> 0;
      enabled[index] = this.enabled[slot]!;
    }
    this.dirty.clear();
    this.dirtyAll = false;
    return { slots, generations, transforms, policy, enabled, count };
  }

  /** One-time/bulk synchronization. Prefer setEntity for normal moving updates. */
  public load(entities: ShadoEntityVisibilitySoA, defaultRadius = 0): void {
    const count = Math.max(0, entities.count | 0);
    if (count > this.capacity) {
      throw new RangeError(
        `Visibility projection input ${count} exceeds reserved capacity ${this.capacity}`
      );
    }
    copyNumbers(this.positionX, entities.positionX, count);
    copyNumbers(this.positionY, entities.positionY, count);
    copyNumbers(this.positionZ, entities.positionZ, count);
    if (entities.radius) copyNumbers(this.radius, entities.radius, count);
    else this.radius.fill(Math.max(0, defaultRadius), 0, count);
    this.enabled.fill(1, 0, count);
    this.phaseMask.fill(0xffffffff, 0, count);
    for (let slot = 0; slot < count; slot += 1) {
      if (!this.slotGeneration[slot]) this.slotGeneration[slot] = 1;
    }
    // A bulk load replaces everything, so the next batch is a full snapshot.
    this.dirtyAll = true;
    this.count = count;
  }

  public markSpatialChange(): number {
    return Atomics.add(this.control, ShadoVisibilityWorkerControl.SpatialRevision, 1) + 1;
  }

  private get control(): Int32Array {
    return new Int32Array(this.buffer, this.layout.controlOffset, CONTROL_LENGTH);
  }

  private assertIndex(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.capacity) {
      throw new RangeError(`Visibility projection index ${index} is out of range`);
    }
  }
}

/**
 * Main-thread controller for the amortized entity-visibility worker.
 *
 * It never waits for visibility. acquireLatest() returns null until a complete
 * generation is available, and callers continue rendering the previous result.
 */
export class ShadoEntityVisibilityWorker {
  public readonly projection: ShadoEntityVisibilityProjection;

  private readonly control: Int32Array;
  private readonly visibleIndices: readonly [Uint32Array, Uint32Array];
  private readonly resultGenerations: readonly [Uint32Array, Uint32Array];
  private readonly flags: readonly [Uint8Array, Uint8Array];
  private inFlight = false;
  private pendingRequest: WorkerRequest | null = null;
  private consumedGeneration = 0;
  private disposed = false;
  private error: string | null = null;
  private lastScheduledSignature = '';
  private lastScheduledAt = Number.NEGATIVE_INFINITY;
  private scheduledSkips = 0;
  private epochs: ShadoEntityVisibilityEpochs = { world: 0, topology: 0, policy: 0 };
  private inFlightRequest: WorkerRequest | null = null;
  private completedRequest: WorkerRequest | null = null;
  private staleEpochResults = 0;
  private staleAgeResults = 0;
  private lastResultAgeMs = 0;
  /**
   * Two updates at the 30 Hz cadence. A result older than this describes a
   * world the caller has already moved past, so its hidden decisions are not
   * used -- the caller falls back to conservative candidates until a matching
   * one arrives. It is a validity limit, not permission to hide a newly
   * visible actor for two updates.
   */
  public maxResultAgeMs = (1000 / 30) * 2;

  private constructor(
    private readonly worker: ShadoVisibilityWorkerPort,
    buffer: SharedArrayBuffer,
    public readonly layout: ShadoEntityVisibilityWorkerLayout,
    private readonly cellCount: number
  ) {
    this.control = new Int32Array(buffer, layout.controlOffset, CONTROL_LENGTH);
    this.projection = new ShadoEntityVisibilityProjection(buffer, layout);
    this.visibleIndices = [
      new Uint32Array(buffer, layout.visibleIndicesOffsets[0], layout.capacity),
      new Uint32Array(buffer, layout.visibleIndicesOffsets[1], layout.capacity),
    ];
    this.resultGenerations = [
      new Uint32Array(buffer, layout.resultGenerationOffsets[0], layout.capacity),
      new Uint32Array(buffer, layout.resultGenerationOffsets[1], layout.capacity),
    ];
    this.flags = [
      new Uint8Array(buffer, layout.flagsOffsets[0], layout.flagsCapacity),
      new Uint8Array(buffer, layout.flagsOffsets[1], layout.flagsCapacity),
    ];
    worker.addEventListener('message', event => this.handleWorkerMessage(event.data));
    worker.addEventListener('error', event => {
      this.fail(event.error instanceof Error ? event.error.message : event.message);
    });
  }

  public static get supported(): boolean {
    return (
      typeof SharedArrayBuffer !== 'undefined' &&
      typeof Atomics !== 'undefined' &&
      typeof Worker !== 'undefined'
    );
  }

  public static async create(
    world: ShadoEntityVisibilityWorkerWorld,
    options: ShadoEntityVisibilityWorkerOptions
  ): Promise<ShadoEntityVisibilityWorker> {
    if (typeof SharedArrayBuffer === 'undefined') {
      throw new Error(
        'SharedArrayBuffer is unavailable; visibility offload requires cross-origin isolation'
      );
    }
    const layout = createShadoEntityVisibilityWorkerLayout(
      options.capacity,
      options.publishFlags !== false
    );
    const buffer = new SharedArrayBuffer(layout.byteLength);
    const workerFactory = options.workerFactory ?? createBrowserWorker;
    const worker = workerFactory(SHADO_ENTITY_VISIBILITY_WORKER_SOURCE);
    const visibility = world.visibility;
    const tiles = visibility
      ? {
          x: [] as number[],
          z: [] as number[],
          size: visibility.size,
          originX: visibility.originX,
          originZ: visibility.originZ,
          denseWidth: visibility.width,
          denseHeight: visibility.height,
        }
      : { ...world.tiles, denseWidth: 0, denseHeight: 0 };
    const cellCount = visibility ? visibility.width * visibility.height : world.tiles.x.length;
    const controller = new ShadoEntityVisibilityWorker(worker, buffer, layout, cellCount);
    const ready = new Promise<void>((resolve, reject) => {
      const onMessage = (event: MessageEvent<WorkerMessage>) => {
        if (event.data.type === 'ready') resolve();
        if (event.data.type === 'error') reject(new Error(event.data.message));
      };
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', event => reject(event.error ?? new Error(event.message)));
    });
    const wasmBytes = decodeBase64(SHADO_WORLD_REDUCER_WASM_BASE64);
    const wasmBuffer = wasmBytes.buffer.slice(
      wasmBytes.byteOffset,
      wasmBytes.byteOffset + wasmBytes.byteLength
    ) as ArrayBuffer;
    worker.postMessage(
      {
        type: 'init',
        buffer,
        layout,
        publishFlags: layout.flagsCapacity === layout.capacity,
        wasmBytes: wasmBuffer,
        tiles,
      },
      [wasmBuffer]
    );
    try {
      await ready;
      return controller;
    } catch (error) {
      controller.dispose();
      throw error;
    }
  }

  /**
   * Publishes a small camera/cell snapshot. If work is already running, the
   * previous pending snapshot is replaced rather than queued.
   */
  public request(
    planes: ArrayLike<number>,
    cellFlags: ArrayLike<number>,
    options: ShadoEntityVisibilityOptions & {
      activePhaseMask?: number;
      /**
       * Legacy multiplier applied to radii at request time.
       *
       * Deprecated and defaulted to 1. Bucket membership is decided from the
       * STORED radius, so a scale applied here would make membership and the
       * final test disagree about the entity's size. Any value other than 1
       * is now REFUSED; the option remains so an older caller gets an error
       * rather than silently unsafe rows.
       */
      radiusScale?: number;
    }
  ): number {
    if (this.disposed) throw new Error('Visibility worker has been disposed');
    if (this.error) throw new Error(`Visibility worker failed: ${this.error}`);
    if (planes.length < 24) {
      throw new Error('Entity visibility requires six vec4 frustum planes');
    }
    /*
     * Refused rather than honoured, and refused BEFORE the delta is drained
     * so a rejected request costs no slot changes.
     *
     * Buckets are built from the stored radius; only the final test would see
     * a scaled one. Any scale above 1 therefore makes the final radius larger
     * than the radius the entity was binned by, so an actor can be admitted
     * by a test whose membership never put it in the admitting region -- and
     * a scale below 1 hides it. The adapter writes its effective radius and
     * has no use for this.
     */
    if (options.radiusScale !== undefined && options.radiusScale !== 1) {
      throw new Error(
        'Entity visibility no longer supports radiusScale; write the effective radius instead'
      );
    }
    const generation =
      Atomics.add(this.control, ShadoVisibilityWorkerControl.RequestedGeneration, 1) + 1;
    const cellSnapshot = new Uint8Array(this.cellCount);
    cellSnapshot.set(Uint8Array.from(cellFlags as ArrayLike<number>).subarray(0, this.cellCount));
    const request: WorkerRequest = {
      type: 'reduce',
      generation,
      planes: Float32Array.from(planes as ArrayLike<number>).subarray(0, 24),
      cellFlags: cellSnapshot,
      camera: [...options.camera] as WorldVec3,
      maxDistance: Math.max(0, options.maxDistance ?? 0),
      outsideWorldVisible: options.outsideWorldVisible !== false,
      activePhaseMask: (options.activePhaseMask ?? 0xffffffff) >>> 0,
      delta: this.projection.drainDelta(),
      epochs: { ...this.epochs },
      dispatchedAtMs: 0,
    };
    if (this.inFlight) {
      /*
       * A pending request is replaced by the newer camera, but its slot
       * changes are not: a spawn, a move or a despawn that arrived while the
       * worker was busy has to reach it, and dropping the batch with the
       * request it happened to ride on would lose the entity, not just delay
       * it.
       */
      const superseded = this.pendingRequest;
      if (superseded) request.delta = mergeDeltas(superseded.delta, request.delta);
      this.pendingRequest = request;
    } else {
      this.dispatch(request);
    }
    return generation;
  }

  /**
   * Issues visibility work only when an input epoch changes and the configured
   * interval has elapsed. Callers keep rendering the last complete generation.
   */
  public requestScheduled(
    planes: ArrayLike<number>,
    cellFlags: ArrayLike<number>,
    options: ShadoEntityVisibilityOptions & {
      activePhaseMask?: number;
      radiusScale?: number;
    },
    schedule: ShadoEntityVisibilitySchedule
  ): number | null {
    const spatialEpoch = Atomics.load(this.control, ShadoVisibilityWorkerControl.SpatialRevision);
    const signature = [
      schedule.cameraEpoch,
      schedule.cellEpoch,
      schedule.policyEpoch ?? 0,
      spatialEpoch,
    ].join(':');
    const now = schedule.nowMs ?? performance.now();
    const minimumIntervalMs = Math.max(0, schedule.minimumIntervalMs ?? 0);
    if (
      !schedule.force &&
      (signature === this.lastScheduledSignature || now - this.lastScheduledAt < minimumIntervalMs)
    ) {
      this.scheduledSkips++;
      return null;
    }
    this.lastScheduledSignature = signature;
    this.lastScheduledAt = now;
    return this.request(planes, cellFlags, options);
  }

  /**
   * Acquires the latest complete shared output without waiting.
   *
   * Returned views are valid for immediate consumption. Do not retain them
   * across multiple later generations because the worker reuses both buffers.
   */
  public acquireLatest(): ShadoEntityVisibilityWorkerResult | null {
    const generation = Atomics.load(this.control, ShadoVisibilityWorkerControl.CompletedGeneration);
    if (generation === this.consumedGeneration) return null;
    const output = Atomics.load(
      this.control,
      ShadoVisibilityWorkerControl.PublishedOutputBuffer
    ) as 0 | 1;
    const count = Atomics.load(
      this.control,
      output === 0
        ? ShadoVisibilityWorkerControl.ResultCount0
        : ShadoVisibilityWorkerControl.ResultCount1
    );
    const entityCount = Atomics.load(
      this.control,
      output === 0
        ? ShadoVisibilityWorkerControl.ResultEntityCount0
        : ShadoVisibilityWorkerControl.ResultEntityCount1
    );
    this.consumedGeneration = generation;
    const request = this.completedRequest;
    this.completedRequest = null;
    /*
     * A result belongs to the state it was asked about. If the package,
     * topology or policy has changed since it was dispatched, its hidden
     * decisions are wrong now rather than late, and it is dropped outright
     * so the caller falls back to conservative candidates.
     */
    if (request && !sameEpochs(request.epochs, this.epochs)) {
      this.staleEpochResults += 1;
      return null;
    }
    const ageMs = request ? now() - request.dispatchedAtMs : 0;
    this.lastResultAgeMs = ageMs;
    const stale = ageMs > this.maxResultAgeMs;
    if (stale) this.staleAgeResults += 1;
    /*
     * Copied, not viewed. The worker reuses both output buffers, so a caller
     * holding a view would find it rewritten underneath: the copy is what
     * makes the result the caller's own.
     */
    return {
      generation,
      visibleIndices: this.visibleIndices[output].slice(0, count),
      visibleGenerations: this.resultGenerations[output].slice(0, count),
      ageMs,
      epochs: request ? { ...request.epochs } : { ...this.epochs },
      stale,
      flags: this.flags[output].slice(0, entityCount),
      workerDurationMs:
        Atomics.load(this.control, ShadoVisibilityWorkerControl.WorkerDurationMicros) / 1000,
      candidateCount: Atomics.load(this.control, ShadoVisibilityWorkerControl.CandidateCount),
      hierarchyRebuildMs:
        Atomics.load(this.control, ShadoVisibilityWorkerControl.HierarchyRebuildMicros) / 1000,
      copiedInputBytes: Atomics.load(this.control, ShadoVisibilityWorkerControl.CopiedInputBytes),
      publishedFlagBytes: Atomics.load(
        this.control,
        ShadoVisibilityWorkerControl.PublishedFlagBytes
      ),
    };
  }

  public get stats(): ShadoEntityVisibilityWorkerStats {
    return {
      requestedGeneration: Atomics.load(
        this.control,
        ShadoVisibilityWorkerControl.RequestedGeneration
      ),
      completedGeneration: Atomics.load(
        this.control,
        ShadoVisibilityWorkerControl.CompletedGeneration
      ),
      workerDurationMs:
        Atomics.load(this.control, ShadoVisibilityWorkerControl.WorkerDurationMicros) / 1000,
      inFlight: this.inFlight,
      hasPendingRequest: this.pendingRequest !== null,
      candidateCount: Atomics.load(this.control, ShadoVisibilityWorkerControl.CandidateCount),
      hierarchyRebuildMs:
        Atomics.load(this.control, ShadoVisibilityWorkerControl.HierarchyRebuildMicros) / 1000,
      copiedInputBytes: Atomics.load(this.control, ShadoVisibilityWorkerControl.CopiedInputBytes),
      publishedFlagBytes: Atomics.load(
        this.control,
        ShadoVisibilityWorkerControl.PublishedFlagBytes
      ),
      scheduledSkips: this.scheduledSkips,
      error: this.error,
      staleEpochResults: this.staleEpochResults,
      staleAgeResults: this.staleAgeResults,
      lastResultAgeMs: this.lastResultAgeMs,
      pendingDeltaSlots: this.projection.pendingDeltaSlots,
    };
  }

  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.pendingRequest = null;
    this.worker.terminate();
  }

  private dispatch(request: WorkerRequest): void {
    this.inFlight = true;
    request.dispatchedAtMs = now();
    this.inFlightRequest = request;
    /*
     * Everything the worker reads is transferred, so neither side holds a
     * view the other may write. The projection stays authoritative here and
     * the worker keeps a private copy it owns outright.
     */
    this.worker.postMessage(request, [
      request.planes.buffer as ArrayBuffer,
      request.cellFlags.buffer as ArrayBuffer,
      request.delta.slots.buffer as ArrayBuffer,
      request.delta.generations.buffer as ArrayBuffer,
      request.delta.transforms.buffer as ArrayBuffer,
      request.delta.policy.buffer as ArrayBuffer,
      request.delta.enabled.buffer as ArrayBuffer,
    ]);
  }

  /**
   * Declares what later results will be compared against.
   *
   * A change here makes every outstanding result inapplicable at once: a
   * different package, topology or policy means an old hidden decision is
   * wrong now rather than merely old.
   */
  public setEpochs(epochs: Partial<ShadoEntityVisibilityEpochs>): void {
    this.epochs = { ...this.epochs, ...epochs };
  }

  private handleWorkerMessage(message: WorkerMessage): void {
    if (message.type === 'error') {
      this.fail(message.message);
      return;
    }
    if (message.type !== 'complete') return;
    this.inFlight = false;
    this.completedRequest = this.inFlightRequest;
    this.inFlightRequest = null;
    const pending = this.pendingRequest;
    this.pendingRequest = null;
    if (pending && !this.disposed) this.dispatch(pending);
  }

  private fail(message: string): void {
    this.error = message;
    this.inFlight = false;
    this.pendingRequest = null;
  }
}

/** Monotonic milliseconds, wherever this runs. */
function now(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function sameEpochs(
  left: ShadoEntityVisibilityEpochs,
  right: ShadoEntityVisibilityEpochs
): boolean {
  return (
    left.world === right.world &&
    left.topology === right.topology &&
    left.policy === right.policy
  );
}

/**
 * Folds an older batch into a newer one, newest value per slot winning.
 *
 * A slot described by both is one entry: the later state is the true one, and
 * sending both would make the worker apply a change it has already been told
 * about.
 */
function mergeDeltas(
  older: ShadoEntityVisibilityDelta,
  newer: ShadoEntityVisibilityDelta
): ShadoEntityVisibilityDelta {
  const bySlot = new Map<number, number>();
  for (let index = 0; index < older.slots.length; index += 1) bySlot.set(older.slots[index]!, index);
  const newerIndex = new Map<number, number>();
  for (let index = 0; index < newer.slots.length; index += 1) newerIndex.set(newer.slots[index]!, index);
  const slots = [...new Set([...bySlot.keys(), ...newerIndex.keys()])].sort((a, b) => a - b);
  const generations = new Uint32Array(slots.length);
  const transforms = new Float32Array(slots.length * 4);
  const policy = new Uint32Array(slots.length);
  const enabled = new Uint8Array(slots.length);
  for (let index = 0; index < slots.length; index += 1) {
    const slot = slots[index]!;
    const from = newerIndex.has(slot) ? newer : older;
    const at = (newerIndex.has(slot) ? newerIndex.get(slot) : bySlot.get(slot))!;
    generations[index] = from.generations[at]!;
    transforms.set(from.transforms.subarray(at * 4, at * 4 + 4), index * 4);
    policy[index] = from.policy[at]!;
    enabled[index] = from.enabled[at]!;
  }
  return {
    slots: Uint32Array.from(slots),
    generations,
    transforms,
    policy,
    enabled,
    count: newer.count,
  };
}

export function createShadoEntityVisibilityWorkerLayout(
  requestedCapacity: number,
  publishFlags = true
): ShadoEntityVisibilityWorkerLayout {
  const capacity = Math.max(1, requestedCapacity | 0);
  let offset = CONTROL_LENGTH * Int32Array.BYTES_PER_ELEMENT;
  const take = (bytes: number, alignment: number): number => {
    offset = Math.ceil(offset / alignment) * alignment;
    const result = offset;
    offset += bytes;
    return result;
  };
  const floats = capacity * Float32Array.BYTES_PER_ELEMENT;
  const indices = capacity * Uint32Array.BYTES_PER_ELEMENT;
  const positionXOffset = take(floats, 4);
  const positionYOffset = take(floats, 4);
  const positionZOffset = take(floats, 4);
  const radiusOffset = take(floats, 4);
  const enabledOffset = take(capacity, 1);
  const phaseMaskOffset = take(indices, 4);
  const slotGenerationOffset = take(indices, 4);
  const resultGenerationOffsets = [take(indices, 4), take(indices, 4)] as const;
  const visibleIndicesOffsets = [take(indices, 4), take(indices, 4)] as const;
  const flagsCapacity = publishFlags ? capacity : 1;
  const flagsOffsets = [take(flagsCapacity, 1), take(flagsCapacity, 1)] as const;
  return {
    byteLength: offset,
    capacity,
    controlOffset: 0,
    positionXOffset,
    positionYOffset,
    positionZOffset,
    radiusOffset,
    enabledOffset,
    phaseMaskOffset,
    slotGenerationOffset,
    resultGenerationOffsets,
    visibleIndicesOffsets,
    flagsOffsets,
    flagsCapacity,
  };
}

function copyNumbers(target: Float32Array, source: ArrayLike<number>, count: number): void {
  for (let i = 0; i < count; i++) target[i] = Number(source[i] ?? 0);
}

function createBrowserWorker(source: string): ShadoVisibilityWorkerPort {
  if (typeof Worker === 'undefined') {
    throw new Error('Web Workers are unavailable in this environment');
  }
  const objectUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  const worker = new Worker(objectUrl, { name: 'shado-entity-visibility' });
  URL.revokeObjectURL(objectUrl);
  return worker;
}

function decodeBase64(value: string): Uint8Array {
  if (typeof Buffer !== 'undefined') return Uint8Array.from(Buffer.from(value, 'base64'));
  const binary = atob(value);
  return Uint8Array.from(binary, character => character.charCodeAt(0));
}

/*
 * The worker's source, with the shared region classifier spliced in.
 *
 * The worker is a standalone script -- it cannot import -- so the one
 * implementation of membership arithmetic arrives as its own text. Splicing
 * rather than restating it is the point: the synchronous reducer and the
 * worker are then the same function by construction, which is the only way
 * they cannot answer differently for the same entity.
 */
const SHADO_ENTITY_VISIBILITY_WORKER_SOURCE = String.raw`
let state;
` + REGION_MEMBERSHIP_SOURCE + String.raw`

/**
 * Regions stored inline per entity before spilling to a map.
 *
 * Eight covers a bound up to three regions across in each axis, which is
 * every ordinary actor and most props; the rare wider thing spills rather
 * than forcing every entity to reserve room for it. At 65k entities the
 * inline arrays are 4 MB, where reserving the 64-region cap for everyone
 * would be 33 MB.
 */
const INLINE_MEMBERSHIPS = 8;

self.onmessage = async event => {
  try {
    const message = event.data;
    if (message.type === 'init') {
      state = await createState(message);
      self.postMessage({ type: 'ready' });
      return;
    }
    if (message.type === 'reduce') reduce(message);
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    });
  }
};

async function createState(message) {
  const { instance } = await WebAssembly.instantiate(message.wasmBytes, {});
  const wasm = instance.exports;
  const { layout, buffer, tiles } = message;
  const control = new Int32Array(buffer, layout.controlOffset, 16);
  /*
   * PRIVATE arrays, not views into the shared projection.
   *
   * The main thread owns the projection and writes it whenever an entity
   * moves; reading it here while it does that is a read of a value in the
   * middle of being written. These are the worker's own, fed by the delta
   * batch each request carries, so what a reduction sees is exactly the state
   * its caller described and nothing later.
   */
  const positions = [
    new Float32Array(layout.capacity),
    new Float32Array(layout.capacity),
    new Float32Array(layout.capacity),
  ];
  const radius = new Float32Array(layout.capacity);
  const enabled = new Uint8Array(layout.capacity);
  const phaseMask = new Uint32Array(layout.capacity);
  const slotGeneration = new Uint32Array(layout.capacity);
  let entityCount = 0;
  const sharedIndices = layout.visibleIndicesOffsets.map(
    offset => new Uint32Array(buffer, offset, layout.capacity)
  );
  const sharedGenerations = layout.resultGenerationOffsets.map(
    offset => new Uint32Array(buffer, offset, layout.capacity)
  );
  const sharedFlags = layout.flagsOffsets.map(
    offset => new Uint8Array(buffer, offset, layout.flagsCapacity)
  );
  const dense = tiles.denseWidth > 0 && tiles.denseHeight > 0;
  const minX = dense ? 0 : tiles.x.length ? Math.min(...tiles.x) : 0;
  const maxX = dense ? tiles.denseWidth - 1 : tiles.x.length ? Math.max(...tiles.x) : 0;
  const minZ = dense ? 0 : tiles.z.length ? Math.min(...tiles.z) : 0;
  const maxZ = dense ? tiles.denseHeight - 1 : tiles.z.length ? Math.max(...tiles.z) : 0;
  const gridWidth = Math.max(0, maxX - minX + 1);
  const gridHeight = Math.max(0, maxZ - minZ + 1);
  const cellCount = dense ? tiles.denseWidth * tiles.denseHeight : tiles.x.length;
  let tileLookup;
  if (!dense) {
    tileLookup = new Int32Array(gridWidth * gridHeight).fill(-1);
    tiles.x.forEach((x, cell) => {
      tileLookup[(tiles.z[cell] - minZ) * gridWidth + x - minX] = cell;
    });
  }
  const alloc = values => {
    const pointer = wasm.alloc(values.byteLength) >>> 0;
    if (values instanceof Int32Array)
      new Int32Array(wasm.memory.buffer, pointer, values.length).set(values);
    return pointer;
  };
  return {
    wasm,
    layout,
    control,
    positions,
    radius,
    enabled,
    phaseMask,
    slotGeneration,
    entityCount,
    sharedIndices,
    sharedGenerations,
    sharedFlags,
    publishFlags: message.publishFlags,
    tiles,
    dense,
    cellCount,
    gridWidth,
    gridHeight,
    gridMinX: minX,
    gridMinZ: minZ,
    tileLookup,
    tileLookupPtr: dense ? 0 : alloc(tileLookup),
    // 92: the entity descriptor grew when per-candidate admission was added.
    descriptorPtr: wasm.alloc(92) >>> 0,
    planesPtr: wasm.alloc(24 * 4) >>> 0,
    cellFlagsPtr: wasm.alloc(Math.max(1, cellCount)) >>> 0,
    capacity: 0,
    xPtr: 0,
    yPtr: 0,
    zPtr: 0,
    radiusPtr: 0,
    outputPtr: 0,
    flagsPtr: 0,
    admissionPtr: 0,
    hierarchyRevision: -1,
    hierarchyCount: -1,
    /*
     * Buckets as dense id arrays, plus each entity's record of where it sits
     * in each of them. Removal is swap-with-last, and the entity that moved
     * has its record repaired, so nothing walks the population to maintain
     * the index.
     */
    /*
     * Two buckets past the cells, not one. the cellCount bucket holds things PROVED
     * wholly outside the grid, which 'outsideWorldVisible' governs;
     * the one past it holds things whose membership could not be enumerated
     * at all, which nothing governs -- they are always candidates. Sharing
     * one bucket is what hid a 600-unit actor from a client that had a world
     * coordinator and therefore passed outsideWorldVisible: false.
     */
    bucketMembers: new Array(cellCount + 2).fill(null),
    bucketCounts: new Uint32Array(cellCount + 2),
    entityRegionCount: new Uint8Array(layout.capacity),
    entityRegions: new Int32Array(layout.capacity * INLINE_MEMBERSHIPS),
    entitySlotIn: new Int32Array(layout.capacity * INLINE_MEMBERSHIPS),
    /** Entities touching more regions than fit inline; rare and bounded. */
    entitySpill: new Map(),
    membershipChanges: 0,
    bucketsTouched: 0,
    candidateIds: new Uint32Array(layout.capacity),
    /* Scratch for the shared classifier and for reading a stored membership. */
    membershipScratch: new Uint32Array(64),
    admissionRegions: new Int32Array(64),
    admissionSlots: new Int32Array(64),
    /*
     * Query-generation stamps, so an entity listed in several admitted
     * buckets is taken once. A Set per query would allocate per frame and a
     * duplicate would draw the entity twice.
     */
    seenStamp: new Uint32Array(layout.capacity),
    queryGeneration: 0,
  };
}

function ensureCapacity(count) {
  if (count <= state.capacity) return;
  let capacity = Math.max(4, state.capacity);
  while (capacity < count) capacity *= 2;
  state.capacity = capacity;
  state.xPtr = state.wasm.alloc(capacity * 4) >>> 0;
  state.yPtr = state.wasm.alloc(capacity * 4) >>> 0;
  state.zPtr = state.wasm.alloc(capacity * 4) >>> 0;
  state.radiusPtr = state.wasm.alloc(capacity * 4) >>> 0;
  state.outputPtr = state.wasm.alloc(capacity * 4) >>> 0;
  state.flagsPtr = state.wasm.alloc(capacity) >>> 0;
  state.admissionPtr = state.wasm.alloc(capacity) >>> 0;
}

/**
 * Topology admission per candidate, decided over the entity's whole bound.
 *
 * The same rule the synchronous path applies: a candidate is admitted when
 * ANY region its XZ bound touches passes every required bit on its own, and
 * bits are never combined across regions. Without this the kernel would fall
 * back to the single region the entity's centre sits in, which is not
 * conservative for anything wider than a region.
 */
function computeAdmission(candidateCount, message) {
  const admission = new Uint8Array(state.wasm.memory.buffer, state.admissionPtr, candidateCount);
  const required = 0x71;
  if (!(state.tiles.size > 0) || !state.cellCount) {
    admission.fill(required);
    return;
  }
  const outsideBucket = state.cellCount;
  const unknownBucket = outsideBucket + 1;
  const regions = state.admissionRegions;
  const slots = state.admissionSlots;
  for (let local = 0; local < candidateCount; local++) {
    const entity = state.candidateIds[local];
    /*
     * The membership the buckets were built from, not a second computation
     * of it. Recomputing here is how the two could classify one entity
     * differently within a single request -- enqueued as unknown, then
     * admitted as if it were inside the grid, or the reverse.
     */
    const count = readMembership(entity, regions, slots);
    let granted = 0;
    let unknown = false;
    let outside = false;
    for (let index = 0; index < count; index++) {
      const bucket = regions[index];
      if (bucket === unknownBucket) { unknown = true; break; }
      if (bucket === outsideBucket) { outside = true; continue; }
      const flags = bucket < state.cellCount ? message.cellFlags[bucket] : 0;
      if ((flags & required) === required) {
        granted = flags & 0x73;
        break;
      }
    }
    if (unknown || count === 0) {
      // Nothing was proved about this one, so nothing may be rejected on it.
      admission[local] = required;
      continue;
    }
    if (!granted && outside) {
      admission[local] = message.outsideWorldVisible ? required : 0;
      continue;
    }
    admission[local] = granted;
  }
}

function locateCell(x, z) {
  if (!state.cellCount || !(state.tiles.size > 0)) return -1;
  const tileX = Math.floor((x - state.tiles.originX) / state.tiles.size);
  const tileZ = Math.floor((z - state.tiles.originZ) / state.tiles.size);
  const localX = tileX - state.gridMinX;
  const localZ = tileZ - state.gridMinZ;
  if (
    localX < 0 || localX >= state.gridWidth ||
    localZ < 0 || localZ >= state.gridHeight
  ) return -1;
  const denseCell = localZ * state.gridWidth + localX;
  return state.dense ? denseCell : state.tileLookup[denseCell];
}

/**
 * Visits every region an entity's bound touches, or the outside bucket.
 *
 * Binning by the entity's CENTRE is what let a building whose origin sits in
 * a rejected region vanish while its body reached into an admitted one: it
 * was never even a candidate, so no later test could save it.
 */
function forEachMembership(entity, outsideBucket, visit) {
  const unknownBucket = outsideBucket + 1;
  if (!(state.tiles.size > 0) || !state.cellCount) {
    visit(unknownBucket);
    return;
  }
  const x = state.positions[0][entity];
  const z = state.positions[2][entity];
  const radius = state.radius[entity];
  const written = classifyRegionMembership(
    state.tiles.originX, state.tiles.originZ, state.tiles.size,
    state.gridMinX, state.gridMinZ, state.gridWidth, state.gridHeight,
    x - radius, z - radius, x + radius, z + radius,
    64, state.membershipScratch
  );
  if (written === -2) {
    visit(outsideBucket);
    return;
  }
  if (written < 0) {
    visit(unknownBucket);
    return;
  }
  for (let index = 0; index < written; index++) {
    const dense = state.membershipScratch[index];
    const cell = state.tileLookupPtr === 0 ? dense : -1;
    /*
     * A dense index with no cell behind it is not an outside entity -- it is
     * one whose region this worker cannot resolve, which is unknown.
     */
    if (cell >= 0 && cell < state.cellCount) visit(cell);
    else visit(unknownBucket);
  }
}

/** Reads an entity's stored membership into 'regions' and 'slots'. */
function readMembership(entity, regions, slots) {
  const count = state.entityRegionCount[entity];
  if (count <= INLINE_MEMBERSHIPS) {
    const base = entity * INLINE_MEMBERSHIPS;
    for (let index = 0; index < count; index++) {
      regions[index] = state.entityRegions[base + index];
      slots[index] = state.entitySlotIn[base + index];
    }
    return count;
  }
  const spill = state.entitySpill.get(entity);
  for (let index = 0; index < count; index++) {
    regions[index] = spill.regions[index];
    slots[index] = spill.slots[index];
  }
  return count;
}

function writeMembership(entity, regions, slots, count) {
  state.entityRegionCount[entity] = count;
  if (count <= INLINE_MEMBERSHIPS) {
    state.entitySpill.delete(entity);
    const base = entity * INLINE_MEMBERSHIPS;
    for (let index = 0; index < count; index++) {
      state.entityRegions[base + index] = regions[index];
      state.entitySlotIn[base + index] = slots[index];
    }
    return;
  }
  state.entitySpill.set(entity, {
    regions: regions.slice(0, count),
    slots: slots.slice(0, count),
  });
}

/** Repairs one entity's record of where it sits in one bucket. */
function repairSlot(entity, bucket, slot) {
  const count = state.entityRegionCount[entity];
  if (count <= INLINE_MEMBERSHIPS) {
    const base = entity * INLINE_MEMBERSHIPS;
    for (let index = 0; index < count; index++) {
      if (state.entityRegions[base + index] === bucket) {
        state.entitySlotIn[base + index] = slot;
        return;
      }
    }
    return;
  }
  const spill = state.entitySpill.get(entity);
  for (let index = 0; index < count; index++) {
    if (spill.regions[index] === bucket) {
      spill.slots[index] = slot;
      return;
    }
  }
}

function bucketAdd(bucket, entity) {
  let members = state.bucketMembers[bucket];
  const count = state.bucketCounts[bucket];
  if (!members) {
    members = new Uint32Array(8);
    state.bucketMembers[bucket] = members;
  } else if (count >= members.length) {
    const grown = new Uint32Array(members.length * 2);
    grown.set(members);
    state.bucketMembers[bucket] = grown;
    members = grown;
  }
  members[count] = entity;
  state.bucketCounts[bucket] = count + 1;
  state.bucketsTouched++;
  return count;
}

function bucketRemove(bucket, slot) {
  const members = state.bucketMembers[bucket];
  const last = state.bucketCounts[bucket] - 1;
  if (!members || last < 0) return;
  if (slot !== last) {
    // Swap-with-last, then tell the entity that moved where it now sits.
    const moved = members[last];
    members[slot] = moved;
    repairSlot(moved, bucket, slot);
  }
  state.bucketCounts[bucket] = last;
  state.bucketsTouched++;
}

const MEMBERSHIP_SCRATCH_A = { regions: new Int32Array(64), slots: new Int32Array(64) };
const MEMBERSHIP_SCRATCH_B = { regions: new Int32Array(64), slots: new Int32Array(64) };

/**
 * Brings one entity's bucket membership up to date.
 *
 * Returns without touching a bucket when the region list has not changed,
 * which is the common case for ordinary motion: an actor takes many steps
 * inside one region for every step that leaves it.
 */
function updateMembership(entity, outsideBucket) {
  const wanted = MEMBERSHIP_SCRATCH_A.regions;
  let wantedCount = 0;
  forEachMembership(entity, outsideBucket, bucket => {
    if (wantedCount < 64) wanted[wantedCount++] = bucket;
  });
  const current = MEMBERSHIP_SCRATCH_B.regions;
  const currentSlots = MEMBERSHIP_SCRATCH_B.slots;
  const currentCount = readMembership(entity, current, currentSlots);

  let identical = currentCount === wantedCount;
  for (let index = 0; identical && index < wantedCount; index++) {
    if (current[index] !== wanted[index]) identical = false;
  }
  if (identical) return false;

  state.membershipChanges++;
  for (let index = 0; index < currentCount; index++) {
    bucketRemove(current[index], currentSlots[index]);
  }
  const slots = MEMBERSHIP_SCRATCH_A.slots;
  for (let index = 0; index < wantedCount; index++) {
    slots[index] = bucketAdd(wanted[index], entity);
  }
  writeMembership(entity, wanted, slots, wantedCount);
  return true;
}

/** Drops an entity out of every bucket, for a slot that left the population. */
function clearMembership(entity) {
  const regions = MEMBERSHIP_SCRATCH_B.regions;
  const slots = MEMBERSHIP_SCRATCH_B.slots;
  const count = readMembership(entity, regions, slots);
  for (let index = 0; index < count; index++) bucketRemove(regions[index], slots[index]);
  state.entityRegionCount[entity] = 0;
  state.entitySpill.delete(entity);
}

function prepareCandidateIds(count, message) {
  // Full reason flags require visiting every entity. Compact-only consumers can
  // skip whole cells before copying positions into private WASM memory.
  if (state.publishFlags || !state.cellCount) {
    for (let entity = 0; entity < count; entity++) state.candidateIds[entity] = entity;
    return count;
  }
  let candidateCount = 0;
  const requiredCellBits = 0x71;
  /*
   * One entity can be listed in several admitted buckets, so each is taken
   * once. The stamp is bumped per query instead of clearing an array, and
   * wraps back to a cleared array rather than colliding.
   */
  state.queryGeneration = (state.queryGeneration + 1) >>> 0;
  if (state.queryGeneration === 0) {
    state.seenStamp.fill(0);
    state.queryGeneration = 1;
  }
  const stamp = state.queryGeneration;
  const take = (bucket) => {
    const members = state.bucketMembers[bucket];
    const total = state.bucketCounts[bucket];
    for (let index = 0; index < total; index++) {
      const entity = members[index];
      if (state.seenStamp[entity] === stamp) continue;
      state.seenStamp[entity] = stamp;
      state.candidateIds[candidateCount++] = entity;
    }
  };
  for (let cell = 0; cell < state.cellCount; cell++) {
    if ((message.cellFlags[cell] & requiredCellBits) !== requiredCellBits) continue;
    take(cell);
  }
  /*
   * The unknown bucket is taken unconditionally. It holds everything whose
   * membership could not be enumerated, and skipping it is a rejection made
   * on the strength of not knowing.
   */
  take(state.cellCount + 1);
  if (message.outsideWorldVisible) take(state.cellCount);
  return candidateCount;
}

/**
 * Applies one request's slot changes to the worker's private arrays.
 *
 * Done before anything is reduced, so the reduction describes the state the
 * request carried rather than a mixture of that and whatever arrived since.
 */
function applyDelta(delta) {
  if (!delta) return;
  const outsideBucket = state.cellCount;
  const previousCount = state.entityCount;
  state.entityCount = delta.count;
  /*
   * A slot that left the population is removed from its buckets. Leaving it
   * in would keep a departed entity as a candidate, and worse, a later
   * occupant of the slot would inherit its membership.
   */
  for (let slot = delta.count; slot < previousCount; slot++) clearMembership(slot);
  for (let index = 0; index < delta.slots.length; index++) {
    const slot = delta.slots[index];
    if (slot >= state.layout.capacity) continue;
    state.positions[0][slot] = delta.transforms[index * 4];
    state.positions[1][slot] = delta.transforms[index * 4 + 1];
    state.positions[2][slot] = delta.transforms[index * 4 + 2];
    state.radius[slot] = delta.transforms[index * 4 + 3];
    state.enabled[slot] = delta.enabled[index];
    state.phaseMask[slot] = delta.policy[index] >>> 0;
    state.slotGeneration[slot] = delta.generations[index];
    /*
     * Transform and membership are separate facts. Only the slots this batch
     * describes are considered, and of those only the ones whose region list
     * actually changed touch a bucket -- an actor takes many steps inside one
     * region for every step that leaves it.
     */
    if (state.cellCount) updateMembership(slot, outsideBucket);
  }
  // Slots appended since the last batch that this batch did not describe.
  for (let slot = previousCount; slot < delta.count; slot++) {
    if (state.cellCount && !state.entityRegionCount[slot]) {
      updateMembership(slot, outsideBucket);
    }
  }
}

function reduce(message) {
  const started = performance.now();
  const membershipStarted = performance.now();
  state.membershipChanges = 0;
  state.bucketsTouched = 0;
  applyDelta(message.delta);
  const count = state.entityCount;
  // Kept under the old name in the control block: it is still "time spent
  // maintaining the spatial index", it is simply no longer a rebuild.
  const hierarchyRebuildMs = performance.now() - membershipStarted;
  const candidateCount = prepareCandidateIds(count, message);
  ensureCapacity(candidateCount);
  const memory = state.wasm.memory.buffer;
  const wasmX = new Float32Array(memory, state.xPtr, candidateCount);
  const wasmY = new Float32Array(memory, state.yPtr, candidateCount);
  const wasmZ = new Float32Array(memory, state.zPtr, candidateCount);
  const wasmRadius = new Float32Array(memory, state.radiusPtr, candidateCount);
  for (let local = 0; local < candidateCount; local++) {
    const entity = state.candidateIds[local];
    wasmX[local] = state.positions[0][entity];
    wasmY[local] = state.positions[1][entity];
    wasmZ[local] = state.positions[2][entity];
    wasmRadius[local] = state.radius[entity];
  }
  new Float32Array(memory, state.planesPtr, 24).set(message.planes);
  new Uint8Array(memory, state.cellFlagsPtr, state.cellCount).set(message.cellFlags);
  computeAdmission(candidateCount, message);
  const descriptor = new DataView(memory, state.descriptorPtr, 92);
  [
    candidateCount, state.xPtr, state.yPtr, state.zPtr, state.radiusPtr, state.planesPtr,
    state.cellFlagsPtr, state.tileLookupPtr,
  ].forEach((value, index) => descriptor.setUint32(index * 4, value >>> 0, true));
  descriptor.setInt32(32, state.gridWidth, true);
  descriptor.setInt32(36, state.gridHeight, true);
  descriptor.setInt32(40, state.gridMinX, true);
  descriptor.setInt32(44, state.gridMinZ, true);
  descriptor.setFloat32(48, state.tiles.originX, true);
  descriptor.setFloat32(52, state.tiles.originZ, true);
  descriptor.setFloat32(56, state.tiles.size, true);
  message.camera.forEach((value, axis) => descriptor.setFloat32(60 + axis * 4, value, true));
  descriptor.setFloat32(72, message.maxDistance, true);
  descriptor.setInt32(76, message.outsideWorldVisible ? 1 : 0, true);
  descriptor.setUint32(80, state.outputPtr, true);
  descriptor.setUint32(84, state.flagsPtr, true);
  descriptor.setUint32(88, state.admissionPtr, true);
  const wasmVisibleCount = state.wasm.reduceEntityVisibility(state.descriptorPtr);
  if (wasmVisibleCount < 0 || wasmVisibleCount > candidateCount) {
    throw new Error('WASM visibility reducer returned invalid count ' + wasmVisibleCount);
  }
  const output = 1 - Atomics.load(state.control, 2);
  const wasmIndices = new Uint32Array(memory, state.outputPtr, wasmVisibleCount);
  const wasmFlags = new Uint8Array(memory, state.flagsPtr, candidateCount);
  if (state.publishFlags) {
    state.sharedFlags[output].set(wasmFlags, 0);
  }
  let visibleCount = 0;
  for (let i = 0; i < wasmVisibleCount; i++) {
    const local = wasmIndices[i];
    const entity = state.candidateIds[local];
    if (
      !state.enabled[entity] ||
      !(state.phaseMask[entity] & message.activePhaseMask)
    ) {
      if (state.publishFlags) state.sharedFlags[output][entity] &= 0x7f;
      continue;
    }
    state.sharedGenerations[output][visibleCount] = state.slotGeneration[entity];
    state.sharedIndices[output][visibleCount++] = entity;
  }
  Atomics.store(state.control, output === 0 ? 4 : 5, visibleCount);
  Atomics.store(state.control, output === 0 ? 9 : 10, state.publishFlags ? count : 0);
  Atomics.store(state.control, 11, candidateCount);
  Atomics.store(state.control, 12, Math.max(0, Math.round(hierarchyRebuildMs * 1000)));
  Atomics.store(state.control, 13, candidateCount * 16);
  Atomics.store(state.control, 14, state.publishFlags ? count : 0);
  Atomics.store(state.control, 7, Math.max(0, Math.round((performance.now() - started) * 1000)));
  Atomics.store(state.control, 2, output);
  Atomics.store(state.control, 1, message.generation);
  self.postMessage({ type: 'complete', generation: message.generation });
}
`;
