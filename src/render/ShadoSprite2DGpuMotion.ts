import {
  BABYLON,
  type ComputeShader,
  type StorageBuffer,
  type WebGPUEngine,
} from '../babylon';
import type { ShadoSprite2DMotionConfig } from './ShadoSprite2DMotionKernel';

const WORKGROUP_SIZE = 64;
const STATE_WORDS = 4;
const TIMING_WORDS = 2;
const PARAM_WORDS = 12;

/** WebGPU-owned motion state for a stable, densely indexed 2D sprite batch. */
export class ShadoSprite2DGpuMotion {
  private stateBuffer: StorageBuffer;
  private timingBuffer: StorageBuffer;
  private readonly paramsBuffer: StorageBuffer;
  private readonly params = new Uint32Array(PARAM_WORDS);
  private readonly paramsFloat = new Float32Array(this.params.buffer);
  private readonly shader: ComputeShader;
  private config: ShadoSprite2DMotionConfig = {
    seed: 1,
    speed: 0,
    cadenceMs: 1_000,
    bounds: [-1, -1, 1, 1],
  };
  private count = 0;
  private globalStart = 0;
  private resetPending = true;
  private dispatches = 0;
  private error = '';

  public constructor(private readonly engine: WebGPUEngine) {
    if (!engine.isWebGPU) throw new Error('ShadoSprite2DGpuMotion requires WebGPU.');
    this.stateBuffer = this.createBuffer(
      STATE_WORDS * 4,
      'Shado 2D GPU motion state',
      BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE
    );
    this.timingBuffer = this.createBuffer(TIMING_WORDS * 4, 'Shado 2D GPU motion timing');
    this.paramsBuffer = this.createBuffer(this.params.byteLength, 'Shado 2D GPU motion params');
    this.shader = new BABYLON.ComputeShader(
      'Shado 2D GPU motion',
      engine,
      { computeSource: emitSprite2DMotionWGSL() },
      {
        bindingsMapping: {
          motionState: { group: 0, binding: 0 },
          motionTiming: { group: 0, binding: 1 },
          motionParams: { group: 0, binding: 2 },
        },
      }
    );
    this.shader.onError = (_effect, errors) => { this.error = errors; };
    this.bindComputeBuffers();
  }

  public get state(): StorageBuffer {
    return this.stateBuffer;
  }

  public get size(): number {
    return this.count;
  }

  public get dispatchCount(): number {
    return this.dispatches;
  }

  public get lastError(): string {
    return this.error;
  }

  public setPopulation(
    positions: Float32Array,
    config: ShadoSprite2DMotionConfig,
    globalStart = 0
  ): void {
    if (positions.length % 2 !== 0) throw new Error('2D motion positions must contain XY pairs');
    this.count = positions.length / 2;
    this.globalStart = globalStart >>> 0;
    const state = new Float32Array(Math.max(1, this.count) * STATE_WORDS);
    for (let index = 0; index < this.count; index++) {
      state[index * STATE_WORDS] = positions[index * 2];
      state[index * STATE_WORDS + 1] = positions[index * 2 + 1];
    }
    this.stateBuffer.dispose();
    this.timingBuffer.dispose();
    this.stateBuffer = this.createBuffer(
      state.byteLength,
      'Shado 2D GPU motion state',
      BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE
    );
    this.timingBuffer = this.createBuffer(
      Math.max(1, this.count) * TIMING_WORDS * 4,
      'Shado 2D GPU motion timing'
    );
    this.stateBuffer.update(state);
    this.timingBuffer.update(new Uint32Array(Math.max(1, this.count) * TIMING_WORDS));
    this.bindComputeBuffers();
    this.configure(config);
  }

  public configure(config: ShadoSprite2DMotionConfig): void {
    this.config = {
      seed: config.seed | 0,
      speed: Math.max(0, config.speed),
      cadenceMs: Math.max(1, config.cadenceMs),
      bounds: [config.bounds[0], config.bounds[1], config.bounds[2], config.bounds[3]],
    };
    this.resetPending = true;
  }

  public dispatch(nowMs: number, dtSeconds: number): boolean {
    if (!this.count) return false;
    this.uploadParams(nowMs, dtSeconds);
    const dispatched = this.shader.dispatch(Math.ceil(this.count / WORKGROUP_SIZE), 1, 1);
    if (dispatched) {
      this.resetPending = false;
      this.dispatches++;
    }
    return dispatched;
  }

  public async warmUp(nowMs = performance.now()): Promise<void> {
    if (!this.count) return;
    this.uploadParams(nowMs, 0);
    await this.shader.dispatchWhenReady(Math.ceil(this.count / WORKGROUP_SIZE), 1, 1);
    this.resetPending = false;
    this.dispatches++;
  }

  /** Explicit asynchronous readback; never used by the frame loop. */
  public async readStateRange(start: number, count: number): Promise<Float32Array> {
    const safeStart = Math.max(0, Math.min(this.count, Math.trunc(start)));
    const safeCount = Math.max(0, Math.min(this.count - safeStart, Math.trunc(count)));
    if (!safeCount) return new Float32Array(0);
    const target = new Float32Array(safeCount * STATE_WORDS);
    const result = await this.stateBuffer.read(
      safeStart * STATE_WORDS * Float32Array.BYTES_PER_ELEMENT,
      target.byteLength,
      target
    );
    if (result instanceof Float32Array) return result;
    return new Float32Array(result.buffer, result.byteOffset, result.byteLength / 4).slice();
  }

  public dispose(): void {
    this.stateBuffer.dispose();
    this.timingBuffer.dispose();
    this.paramsBuffer.dispose();
  }

  private createBuffer(
    byteLength: number,
    label: string,
    creationFlags = BABYLON.Constants.BUFFER_CREATIONFLAG_WRITE
  ): StorageBuffer {
    return new BABYLON.StorageBuffer(
      this.engine,
      Math.max(4, byteLength),
      creationFlags,
      label
    );
  }

  private bindComputeBuffers(): void {
    this.shader.setStorageBuffer('motionState', this.stateBuffer);
    this.shader.setStorageBuffer('motionTiming', this.timingBuffer);
    this.shader.setStorageBuffer('motionParams', this.paramsBuffer);
  }

  private uploadParams(nowMs: number, dtSeconds: number): void {
    this.params[0] = this.count >>> 0;
    this.params[1] = this.config.seed >>> 0;
    this.paramsFloat[2] = nowMs;
    this.paramsFloat[3] = Math.max(0, Math.min(0.25, dtSeconds));
    this.paramsFloat[4] = this.config.bounds[0];
    this.paramsFloat[5] = this.config.bounds[1];
    this.paramsFloat[6] = this.config.bounds[2];
    this.paramsFloat[7] = this.config.bounds[3];
    this.paramsFloat[8] = this.config.speed;
    this.paramsFloat[9] = this.config.cadenceMs;
    this.params[10] = this.globalStart;
    this.params[11] = this.resetPending ? 1 : 0;
    this.paramsBuffer.update(this.params);
  }
}

export function emitSprite2DMotionWGSL(): string {
  return `
@group(0) @binding(0) var<storage, read_write> motionState: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> motionTiming: array<vec2u>;
@group(0) @binding(2) var<storage, read> motionParams: array<u32>;

fn hash32(value: u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn unitFloat(value: u32) -> f32 {
  return f32(value & 0x00ffffffu) / 16777216.0;
}

fn direction(index: u32, epoch: u32, seed: u32, speed: f32) -> vec2f {
  let h = hash32(index ^ hash32(epoch + seed * 0x9e3779b9u));
  let angle = unitFloat(h) * 6.28318530718;
  let magnitude = speed * (0.35 + unitFloat(hash32(h + 0x85ebca6bu)) * 0.65);
  return vec2f(cos(angle), sin(angle)) * magnitude;
}

fn wrap(value: f32, minimum: f32, maximum: f32) -> f32 {
  let span = maximum - minimum;
  if (span <= 0.000001) { return minimum; }
  return minimum + ((value - minimum) - floor((value - minimum) / span) * span);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let count = motionParams[0];
  if (index >= count) { return; }

  let seed = motionParams[1];
  let nowMs = bitcast<f32>(motionParams[2]);
  let dt = bitcast<f32>(motionParams[3]);
  let bounds = vec4f(
    bitcast<f32>(motionParams[4]), bitcast<f32>(motionParams[5]),
    bitcast<f32>(motionParams[6]), bitcast<f32>(motionParams[7])
  );
  let speed = bitcast<f32>(motionParams[8]);
  let cadenceMs = bitcast<f32>(motionParams[9]);
  let absoluteIndex = motionParams[10] + index;
  let reset = motionParams[11] != 0u;
  var state = motionState[index];
  var timing = motionTiming[index];
  var epoch = timing.x;
  var nextChange = bitcast<f32>(timing.y);

  if (reset) {
    epoch = 0u;
    let velocity = direction(absoluteIndex, epoch, seed, speed);
    state = vec4f(state.xy, velocity);
    nextChange = nowMs + cadenceMs * (0.5 + unitFloat(hash32(absoluteIndex + seed)));
  } else if (nowMs >= nextChange) {
    epoch = epoch + 1u;
    let velocity = direction(absoluteIndex, epoch, seed, speed);
    state = vec4f(state.xy, velocity);
    nextChange = nowMs + cadenceMs * (0.5 + unitFloat(hash32(absoluteIndex + epoch + seed)));
  }

  state.x = wrap(state.x + state.z * dt, bounds.x, bounds.z);
  state.y = wrap(state.y + state.w * dt, bounds.y, bounds.w);
  motionState[index] = state;
  motionTiming[index] = vec2u(epoch, bitcast<u32>(nextChange));
}`;
}
