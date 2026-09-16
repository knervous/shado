import {
  SHADO_PARTICLE_EMITTER_FLOATS,
  SHADO_PARTICLE_FLOATS,
  SHADO_PARTICLE_PENDING_FLOATS,
  SHADO_PARTICLE_TRAIL_FLOATS,
} from './ShadoParticleLayout';
import { SHADO_PARTICLE_REDUCER_WASM_BASE64 } from './wasm/shado-particle-reducer-b64';

/** The raw exports of `assembly/particle-reducer.ts`. */
export interface ShadoParticleReducerExports {
  readonly memory: WebAssembly.Memory;
  alloc(byteLength: number): number;
  resetAllocator(byteOffset?: number): void;
  initArena(
    particlePtr: number,
    particleCapacity: number,
    emitterPtr: number,
    emitterCapacity: number,
    pendingPtr: number,
    pendingCapacity: number,
    trailPtr: number,
    trailCapacity: number,
    seed: number
  ): void;
  step(now: number): number;
  getHead(): number;
  getChangedFirst(): number;
  getChangedCount(): number;
  getSpawnedTotal(): number;
  getDroppedTotal(): number;
  getPendingCount(): number;
  getTrailCount(): number;
  clearQueuesFor(emitter: number): void;
  killAll(): void;
}

export interface ShadoParticleArena {
  readonly particlePtr: number;
  readonly particleCapacity: number;
  readonly emitterPtr: number;
  readonly emitterCapacity: number;
  readonly pendingPtr: number;
  readonly pendingCapacity: number;
  readonly trailPtr: number;
  readonly trailCapacity: number;
}

/** A contiguous run of particle slots written by one step. */
export interface ShadoParticleSlotRange {
  readonly first: number;
  readonly count: number;
}

const REQUIRED_EXPORTS = [
  'memory',
  'alloc',
  'initArena',
  'step',
  'getChangedFirst',
  'getChangedCount',
  'getHead',
  'clearQueuesFor',
  'killAll',
] as const;

let moduleBytes: Uint8Array | null = null;

/** The embedded reducer module, decoded once. */
export function shadoParticleReducerWasmBytes(): Uint8Array {
  if (!moduleBytes) {
    const binary = atob(SHADO_PARTICLE_REDUCER_WASM_BASE64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    moduleBytes = bytes;
  }
  return moduleBytes;
}

export function assertShadoParticleReducerExports(exports: WebAssembly.Exports): ShadoParticleReducerExports {
  for (const name of REQUIRED_EXPORTS) {
    if (!(name in exports)) throw new Error(`Particle reducer is missing export '${name}'`);
  }
  return exports as unknown as ShadoParticleReducerExports;
}

/**
 * Typed access to a particle reducer's memory.
 *
 * Views are rebuilt on every call rather than cached: `alloc` may grow the memory, and a
 * grown `WebAssembly.Memory` detaches every typed array made over its old buffer.
 */
export class ShadoParticleReducer {
  private arena: ShadoParticleArena | null = null;

  public constructor(public readonly exports: ShadoParticleReducerExports) {}

  public get memory(): WebAssembly.Memory {
    return this.exports.memory;
  }

  public get arenaLayout(): ShadoParticleArena | null {
    return this.arena;
  }

  /**
   * Points the reducer at its regions. Particle records may already be allocated (the
   * Shado arena owns them); the emitter table and sub-emitter queues are allocated here
   * when not given.
   */
  public initArena(
    options: {
      particlePtr: number;
      particleCapacity: number;
      emitterCapacity: number;
      pendingCapacity?: number;
      trailCapacity?: number;
      seed?: number;
    }
  ): ShadoParticleArena {
    const pendingCapacity = options.pendingCapacity ?? Math.max(64, options.particleCapacity >> 2);
    const trailCapacity = options.trailCapacity ?? Math.max(32, options.particleCapacity >> 4);
    const emitterPtr = this.exports.alloc(options.emitterCapacity * SHADO_PARTICLE_EMITTER_FLOATS * 4);
    const pendingPtr = this.exports.alloc(pendingCapacity * SHADO_PARTICLE_PENDING_FLOATS * 4);
    const trailPtr = this.exports.alloc(trailCapacity * SHADO_PARTICLE_TRAIL_FLOATS * 4);
    const arena: ShadoParticleArena = {
      particlePtr: options.particlePtr,
      particleCapacity: options.particleCapacity,
      emitterPtr,
      emitterCapacity: options.emitterCapacity,
      pendingPtr,
      pendingCapacity,
      trailPtr,
      trailCapacity,
    };
    this.exports.initArena(
      arena.particlePtr,
      arena.particleCapacity,
      arena.emitterPtr,
      arena.emitterCapacity,
      arena.pendingPtr,
      arena.pendingCapacity,
      arena.trailPtr,
      arena.trailCapacity,
      (options.seed ?? 0x2545f491) >>> 0
    );
    this.arena = arena;
    this.emitterView().fill(0);
    return arena;
  }

  /** Every emitter slot, `SHADO_PARTICLE_EMITTER_FLOATS` floats each. */
  public emitterView(): Float32Array {
    const arena = this.requireArena();
    return new Float32Array(this.memory.buffer, arena.emitterPtr, arena.emitterCapacity * SHADO_PARTICLE_EMITTER_FLOATS);
  }

  /** Every particle record, `SHADO_PARTICLE_FLOATS` floats each. */
  public particleView(): Float32Array {
    const arena = this.requireArena();
    return new Float32Array(this.memory.buffer, arena.particlePtr, arena.particleCapacity * SHADO_PARTICLE_FLOATS);
  }

  /** Advances emission to `now` and returns the slot ranges written (at most two). */
  public step(now: number): ShadoParticleSlotRange[] {
    const arena = this.requireArena();
    const count = this.exports.step(now);
    if (count <= 0) return [];
    const first = this.exports.getChangedFirst();
    if (count >= arena.particleCapacity) return [{ first: 0, count: arena.particleCapacity }];
    const tail = Math.min(count, arena.particleCapacity - first);
    return tail === count ? [{ first, count }] : [{ first, count: tail }, { first: 0, count: count - tail }];
  }

  public get spawnedTotal(): number {
    return this.exports.getSpawnedTotal();
  }

  public get droppedTotal(): number {
    return this.exports.getDroppedTotal();
  }

  private requireArena(): ShadoParticleArena {
    if (!this.arena) throw new Error('ShadoParticleReducer.initArena() has not been called');
    return this.arena;
  }
}

/** A standalone reducer with its own memory, for tests and headless tools. */
export async function createShadoParticleReducer(): Promise<ShadoParticleReducer> {
  const { instance } = await WebAssembly.instantiate(shadoParticleReducerWasmBytes(), {
    env: {
      abort: () => {
        throw new Error('particle reducer aborted');
      },
    },
  });
  return new ShadoParticleReducer(assertShadoParticleReducerExports(instance.exports));
}
