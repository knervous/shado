import { SPRITE_2D_MOTION_KERNEL_SIMD_BASE64 } from './wasm/sprite-2d-motion-kernel-simd-b64';

export interface ShadoSprite2DMotionConfig {
  seed: number;
  speed: number;
  cadenceMs: number;
  bounds: readonly [minX: number, minY: number, maxX: number, maxY: number];
}

type MotionKernelExports = {
  memory: WebAssembly.Memory;
  alloc(byteLength: number): number;
  resetAllocator(byteOffset?: number): void;
  init(
    count: number,
    globalStart: number,
    positionPtr: number,
    velocityPtr: number,
    epochPtr: number,
    nextChangePtr: number
  ): void;
  configure(
    seed: number,
    speed: number,
    cadenceMs: number,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void;
  reset(nowMs: number): void;
  step(nowMs: number, dtSeconds: number): number;
};

/** SIMD-only WASM motion reducer for locked 2D sprite populations. */
export class ShadoSprite2DMotionKernel {
  private positionPtr = 0;
  private count = 0;
  private positionView = new Float32Array(0);

  private constructor(private readonly wasm: MotionKernelExports) {}

  public static async create(): Promise<ShadoSprite2DMotionKernel> {
    const binary = atob(SPRITE_2D_MOTION_KERNEL_SIMD_BASE64);
    const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
    if (!WebAssembly.validate(bytes)) {
      throw new Error('ShadoSprite2DMotionKernel requires WebAssembly SIMD support');
    }
    const instantiated = await WebAssembly.instantiate(bytes, {
      env: { abort: () => { throw new Error('Shado sprite 2D motion kernel aborted'); } },
    });
    const instance = instantiated instanceof WebAssembly.Instance
      ? instantiated
      : instantiated.instance;
    return new ShadoSprite2DMotionKernel(instance.exports as unknown as MotionKernelExports);
  }

  public setPopulation(
    positions: Float32Array,
    config: ShadoSprite2DMotionConfig,
    nowMs = performance.now(),
    globalStart = 0
  ): void {
    if (positions.length % 2 !== 0) throw new Error('2D motion positions must contain XY pairs');
    this.count = positions.length / 2;
    this.wasm.resetAllocator(1024);
    this.positionPtr = this.wasm.alloc(positions.byteLength);
    const velocityPtr = this.wasm.alloc(positions.byteLength);
    const epochPtr = this.wasm.alloc(this.count * 4);
    const nextChangePtr = this.wasm.alloc(this.count * 4);
    this.positionView = new Float32Array(
      this.wasm.memory.buffer,
      this.positionPtr,
      positions.length
    );
    this.positionView.set(positions);
    this.wasm.init(
      this.count,
      globalStart,
      this.positionPtr,
      velocityPtr,
      epochPtr,
      nextChangePtr
    );
    this.configure(config);
    this.wasm.reset(nowMs);
  }

  public configure(config: ShadoSprite2DMotionConfig, nowMs = performance.now()): void {
    this.wasm.configure(
      config.seed | 0,
      Math.max(0, config.speed),
      Math.max(1, config.cadenceMs),
      config.bounds[0],
      config.bounds[1],
      config.bounds[2],
      config.bounds[3]
    );
    if (this.count) this.wasm.reset(nowMs);
  }

  public step(nowMs: number, dtSeconds: number): Float32Array {
    this.wasm.step(nowMs, dtSeconds);
    // A later population allocation may grow memory and detach the old view.
    if (this.positionView.buffer !== this.wasm.memory.buffer) {
      this.positionView = new Float32Array(
        this.wasm.memory.buffer,
        this.positionPtr,
        this.count * 2
      );
    }
    return this.positionView;
  }

  public get size(): number {
    return this.count;
  }
}
