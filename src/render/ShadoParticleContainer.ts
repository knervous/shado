import { Shado } from '../core/Shado';
import { field, gpuStruct } from '../decorators';
import type { InitializeConfig } from '../types';
import {
  SHADO_PARTICLE_EMITTER_FLOATS,
  SHADO_PARTICLE_FLOATS,
  SHADO_PARTICLE_STRIDE_BYTES,
  ShadoParticleEmitterField as F,
} from './ShadoParticleLayout';
import {
  assertShadoParticleReducerExports,
  ShadoParticleReducer,
  shadoParticleReducerWasmBytes,
  type ShadoParticleSlotRange,
} from './ShadoParticleReducer';
import { encodeShadoParticleEmitter, type ShadoParticleEmitterSpec, type ShadoVec3 } from './ShadoParticleEmitters';

/**
 * One particle, as the GPU reads it. Field order and widths are the layout in
 * `ShadoParticleLayout.ts`, and the container refuses to run if Shado lays it out any
 * other way.
 */
@gpuStruct({ name: 'ShadoParticle', useWasm: false })
export class ShadoParticle extends Shado {
  @field('vec4')
  birth!: Float32Array;

  @field('vec4')
  velocity!: Float32Array;

  @field('vec4')
  accel!: Float32Array;

  @field('vec4')
  look!: Float32Array;

  @field('vec4')
  collision!: Float32Array;

  @field('vec4')
  extra!: Float32Array;
}

export interface ShadoParticleContainerOptions {
  /** Particle slots. The ring overwrites the oldest particle when it is full. */
  capacity: number;
  /** Emitter slots across every effect playing at once. */
  emitterCapacity?: number;
  /** Anchors (moving origins particles follow). */
  anchorCapacity?: number;
  pendingCapacity?: number;
  trailCapacity?: number;
  seed?: number;
}

/** A handle to one emitter and the children created under it. */
export interface ShadoParticleEmitterHandle {
  readonly index: number;
  readonly children: readonly ShadoParticleEmitterHandle[];
}

/**
 * GPU particles for any number of effects, in one instanced draw.
 *
 * The container holds particle records in the Shado arena, whose memory is the particle
 * reducer's own, so the reducer writes new particles straight into the bytes the GPU
 * upload reads and only those slots are marked dirty. Nothing is written for a particle
 * after it is born; see `ShadoParticleRenderer` for how it is drawn.
 *
 * Capacity is fixed at construction. Growing would repack the arena and move the records
 * under the reducer, which is not worth supporting for a pool sized by a budget.
 */
@gpuStruct({ name: 'ShadoParticleContainer', useWasm: true })
export class ShadoParticleContainer extends Shado {
  @field('f32')
  particleCapacity!: number;

  @field('f32')
  anchorCapacity!: number;

  @field('f32')
  padding0!: number;

  @field('f32')
  padding1!: number;

  @field({ arrayOf: 'vec4' })
  anchors!: Float32Array;

  public readonly capacity: number;
  public readonly emitterCapacity: number;
  private readonly anchorSlots: number;
  private readonly options: ShadoParticleContainerOptions;
  private reducer?: ShadoParticleReducer;
  private reducerSignature = '';
  private readonly freeEmitters: number[] = [];
  private readonly childrenOf = new Map<number, number[]>();
  private readonly freeAnchors: number[] = [];
  private lastRanges: ShadoParticleSlotRange[] = [];
  private highWater = 0;

  public static override async initialize(engine: any, config: InitializeConfig = {}): Promise<boolean> {
    const { additionalFields = [], wasm, backend, ...rest } = config;
    delete (this as any).__cachedSchema;
    return super.initialize(engine, {
      backend:
        backend ??
        ((engine as any).isWebGPU || engine?.getClassName?.() === 'WebGPUEngine' ? 'storage' : 'datatex'),
      ...rest,
      wasm: wasm ?? { mode: 'precompiled', module: shadoParticleReducerWasmBytes() },
      additionalFields: [...additionalFields, { name: 'particles', type: { arrayOf: { structOf: ShadoParticle } } }],
    });
  }

  public constructor(engine: any, options: ShadoParticleContainerOptions) {
    super(engine);
    this.options = options;
    this.capacity = Math.max(1, Math.floor(options.capacity));
    this.emitterCapacity = Math.max(1, Math.floor(options.emitterCapacity ?? 256));
    this.anchorSlots = Math.max(1, Math.floor(options.anchorCapacity ?? 128));
    this.particleCapacity = this.capacity;
    this.anchorCapacity = this.anchorSlots;
    this.padding0 = 0;
    this.padding1 = 0;

    const stride = this.getStructArrayStrideBytes('particles');
    if (stride !== SHADO_PARTICLE_STRIDE_BYTES) {
      throw new Error(`ShadoParticle stride is ${stride} bytes; the reducer expects ${SHADO_PARTICLE_STRIDE_BYTES}`);
    }
    // Allocate everything once, before the reducer is pointed at it.
    this.setVarArray('anchors', new Float32Array(this.anchorSlots * 4));
    this.setStructArrayCount('particles', this.capacity);
    for (let i = this.emitterCapacity - 1; i >= 0; i--) this.freeEmitters.push(i);
    for (let i = this.anchorSlots - 1; i >= 0; i--) this.freeAnchors.push(i);
    this.ensureReducer();
  }

  // ------------------------------------------------------------------ anchors

  /** A slot for a moving origin. Returns -1 when every anchor is taken. */
  public acquireAnchor(position: ShadoVec3 = [0, 0, 0], yaw = 0): number {
    const slot = this.freeAnchors.pop();
    if (slot === undefined) return -1;
    this.setAnchor(slot, position, yaw);
    return slot;
  }

  /**
   * Moves an anchor. Particles following it are placed in its frame: their positions are
   * turned by `yaw` (radians about +Y, Babylon's `rotation.y` sense) and then offset.
   */
  public setAnchor(slot: number, position: ShadoVec3, yaw = 0): void {
    if (slot < 0 || slot >= this.anchorSlots) return;
    // `writeVarArrayRange` takes an element index; each anchor is one vec4 element.
    this.writeVarArrayRange('anchors', slot, [position[0], position[1], position[2], yaw]);
  }

  public releaseAnchor(slot: number): void {
    if (slot < 0 || slot >= this.anchorSlots) return;
    this.writeVarArrayRange('anchors', slot, [0, 0, 0, 0]);
    this.freeAnchors.push(slot);
  }

  // ------------------------------------------------------------------ emitters

  /**
   * Starts an emitter (and, recursively, the children in `children`) at `startTime` on
   * the container's clock. Returns null when there are not enough emitter slots for the
   * whole tree, in which case nothing is started.
   */
  public startEmitter(
    spec: ShadoParticleEmitterSpec,
    startTime: number,
    children: readonly { spec: ShadoParticleEmitterSpec; children?: readonly any[] }[] = []
  ): ShadoParticleEmitterHandle | null {
    const needed = 1 + countTree(children);
    if (this.freeEmitters.length < needed) return null;
    const reducer = this.ensureReducer();
    const table = reducer.emitterView();
    return this.startTree(table, spec, children, startTime, -1);
  }

  /** Stops emitting. Live particles, and any queued on-death or trail work, still finish. */
  public stopEmitter(handle: ShadoParticleEmitterHandle): void {
    const table = this.ensureReducer().emitterView();
    const base = handle.index * SHADO_PARTICLE_EMITTER_FLOATS;
    if (table[base + F.state] === 1) table[base + F.state] = 2;
  }

  /**
   * Frees an emitter tree's slots. Call once its particles have had time to die: a freed
   * child can no longer be triggered by particles still alive.
   */
  public releaseEmitter(handle: ShadoParticleEmitterHandle): void {
    const reducer = this.ensureReducer();
    const table = reducer.emitterView();
    const release = (h: ShadoParticleEmitterHandle) => {
      for (const child of h.children) release(child);
      table.fill(0, h.index * SHADO_PARTICLE_EMITTER_FLOATS, (h.index + 1) * SHADO_PARTICLE_EMITTER_FLOATS);
      reducer.exports.clearQueuesFor(h.index);
      this.childrenOf.delete(h.index);
      this.freeEmitters.push(h.index);
    };
    release(handle);
  }

  public get freeEmitterSlots(): number {
    return this.freeEmitters.length;
  }

  // ------------------------------------------------------------------ simulation

  /**
   * Advances emission to `now` (seconds) and marks exactly the written slots dirty.
   * Returns the slot ranges written.
   */
  public step(now: number): readonly ShadoParticleSlotRange[] {
    const reducer = this.ensureReducer();
    const ranges = reducer.step(now);
    if (ranges.length) this.markSlotsDirty(ranges);
    for (const range of ranges) this.highWater = Math.max(this.highWater, range.first + range.count);
    this.lastRanges = ranges;
    return ranges;
  }

  /** Slots that have ever held a particle: the instance count worth drawing. */
  public get drawCount(): number {
    return this.highWater;
  }

  public get lastWrittenRanges(): readonly ShadoParticleSlotRange[] {
    return this.lastRanges;
  }

  public get spawnedTotal(): number {
    return this.ensureReducer().spawnedTotal;
  }

  public get droppedTotal(): number {
    return this.ensureReducer().droppedTotal;
  }

  /** Every particle record, for tests and diagnostics. Rebuilt per call. */
  public particleView(): Float32Array {
    return this.ensureReducer().particleView();
  }

  /** Kills every particle and forgets queued sub-emitter work. */
  public killAll(): void {
    const reducer = this.ensureReducer();
    reducer.exports.killAll();
    this.markSlotsDirty([{ first: 0, count: this.capacity }]);
  }

  // ------------------------------------------------------------------ internals

  private startTree(
    table: Float32Array,
    spec: ShadoParticleEmitterSpec,
    children: readonly { spec: ShadoParticleEmitterSpec; children?: readonly any[] }[],
    startTime: number,
    parent: number
  ): ShadoParticleEmitterHandle {
    const index = this.freeEmitters.pop()!;
    // Children first, so their indices exist when this slot links to them.
    const childHandles = children.map(child => this.startTree(table, child.spec, child.children ?? [], startTime, index));
    const childIndices = childHandles.map(child => child.index);
    this.childrenOf.set(index, childIndices);
    encodeShadoParticleEmitter(table, index, spec, startTime, {
      parent,
      firstChild: childIndices[0] ?? -1,
      nextSibling: -1,
    });
    childIndices.forEach((child, i) => {
      table[child * SHADO_PARTICLE_EMITTER_FLOATS + F.nextSibling] = childIndices[i + 1] ?? -1;
    });
    return { index, children: childHandles };
  }

  private ensureReducer(): ShadoParticleReducer {
    const module = (this as any).wasmModule;
    this.getWasmArenaBasePtr();
    const particlePtr = this.getStructArrayPtr('particles');
    if (!module?.exports || !particlePtr) {
      throw new Error('ShadoParticleContainer needs its wasm arena; await ShadoParticleContainer.initialize() first');
    }
    const signature = `${particlePtr}:${this.capacity}`;
    if (this.reducer && signature === this.reducerSignature) return this.reducer;
    if (this.reducer) {
      // The arena moved under a live reducer. Fixed capacity should make this impossible.
      throw new Error('ShadoParticleContainer arena was repacked; particle capacity is fixed at construction');
    }
    this.reducer = new ShadoParticleReducer(assertShadoParticleReducerExports(module.exports));
    this.reducer.initArena({
      particlePtr,
      particleCapacity: this.capacity,
      emitterCapacity: this.emitterCapacity,
      pendingCapacity: this.options.pendingCapacity,
      trailCapacity: this.options.trailCapacity,
      seed: this.options.seed,
    });
    this.reducerSignature = signature;
    // Allocating the tables may have grown memory; the arena base is re-read by Shado.
    this.getWasmArenaBasePtr();
    return this.reducer;
  }

  private markSlotsDirty(ranges: readonly ShadoParticleSlotRange[]): void {
    const seg = (this as any)._structSeg?.particles;
    const arena = (this as any)._arena ?? (this as any).arena;
    if (!seg || !arena?.markDirtyFloats) {
      this.markArenaDirty();
      return;
    }
    for (const range of ranges) {
      arena.markDirtyFloats((seg.offF | 0) + range.first * SHADO_PARTICLE_FLOATS, range.count * SHADO_PARTICLE_FLOATS);
    }
  }
}

function countTree(children: readonly { children?: readonly any[] }[]): number {
  return children.reduce((sum, child) => sum + 1 + countTree(child.children ?? []), 0);
}
