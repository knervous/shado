import { BABYLON, type ComputeShader, type StorageBuffer, type WebGPUEngine } from '../babylon';

const WORKGROUP_SIZE = 64;
const PARAM_WORDS = 8;

/**
 * GPU visibility reduction for the stable sprite/motion index space.
 *
 * The compact index buffer is consumed directly by the vertex shader. The
 * draw arguments buffer is also suitable for Babylon's WebGPU indirect draw
 * context; neither path requires a visibility readback.
 */
export class ShadoSprite2DGpuVisibility {
  private recordsBuffer: StorageBuffer;
  private visibleIndicesBuffer: StorageBuffer;
  private drawArgsBuffer: StorageBuffer;
  private readonly paramsBuffer: StorageBuffer;
  private readonly params = new Uint32Array(PARAM_WORDS);
  private readonly paramsFloat = new Float32Array(this.params.buffer);
  private readonly shader: ComputeShader;
  private count = 0;
  private dispatches = 0;
  private error = '';

  public constructor(
    private readonly engine: WebGPUEngine,
    private motionState: StorageBuffer
  ) {
    this.recordsBuffer = this.createBuffer(48, 'Shado 2D GPU sprite records');
    this.visibleIndicesBuffer = this.createBuffer(4, 'Shado 2D GPU visible indices');
    this.drawArgsBuffer = this.createBuffer(
      20,
      'Shado 2D GPU indirect arguments',
      BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE |
        BABYLON.Constants.BUFFER_CREATIONFLAG_INDIRECT
    );
    this.paramsBuffer = this.createBuffer(this.params.byteLength, 'Shado 2D GPU visibility params');
    this.shader = new BABYLON.ComputeShader(
      'Shado 2D GPU visibility',
      engine,
      { computeSource: emitSprite2DVisibilityWGSL() },
      {
        bindingsMapping: {
          spriteRecords: { group: 0, binding: 0 },
          motionState: { group: 0, binding: 1 },
          visibleIndices: { group: 0, binding: 2 },
          drawArgs: { group: 0, binding: 3 },
          visibilityParams: { group: 0, binding: 4 },
        },
      }
    );
    this.shader.onError = (_effect, errors) => {
      this.error = errors;
    };
    this.bindBuffers();
  }

  public get records(): StorageBuffer {
    return this.recordsBuffer;
  }
  public get visibleIndices(): StorageBuffer {
    return this.visibleIndicesBuffer;
  }
  public get drawArgs(): StorageBuffer {
    return this.drawArgsBuffer;
  }
  public get drawArgsResource(): unknown {
    return this.drawArgsBuffer.getBuffer().underlyingResource;
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

  public setMotionState(state: StorageBuffer): void {
    this.motionState = state;
    this.bindBuffers();
  }

  public setPopulation(records: Float32Array, count: number): void {
    this.count = Math.max(0, Math.trunc(count));
    this.recordsBuffer.dispose();
    this.visibleIndicesBuffer.dispose();
    this.drawArgsBuffer.dispose();
    this.recordsBuffer = this.createBuffer(
      Math.max(48, records.byteLength),
      'Shado 2D GPU sprite records'
    );
    this.visibleIndicesBuffer = this.createBuffer(
      Math.max(4, this.count * Uint32Array.BYTES_PER_ELEMENT),
      'Shado 2D GPU visible indices'
    );
    this.drawArgsBuffer = this.createBuffer(
      20,
      'Shado 2D GPU indirect arguments',
      BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE |
        BABYLON.Constants.BUFFER_CREATIONFLAG_INDIRECT
    );
    if (records.byteLength) this.recordsBuffer.update(records);
    this.bindBuffers();
    this.resetDrawArguments();
  }

  public dispatch(
    center: readonly [number, number],
    halfExtent: readonly [number, number],
    viewportHeight: number,
    minimumPixelSize: number
  ): boolean {
    if (!this.count) return false;
    // Tiny ordered queue writes reset the append counter and indirect instance
    // count. This is CPU -> GPU only; visibility never comes back to the CPU.
    this.resetDrawArguments();
    this.params[0] = this.count >>> 0;
    this.paramsFloat[1] = center[0];
    this.paramsFloat[2] = center[1];
    this.paramsFloat[3] = Math.max(0.0001, halfExtent[0]);
    this.paramsFloat[4] = Math.max(0.0001, halfExtent[1]);
    this.paramsFloat[5] = Math.max(1, viewportHeight);
    this.paramsFloat[6] = Math.max(0, minimumPixelSize);
    this.params[7] = 0;
    this.paramsBuffer.update(this.params);
    const dispatched = this.shader.dispatch(Math.ceil(this.count / WORKGROUP_SIZE), 1, 1);
    if (dispatched) this.dispatches++;
    return dispatched;
  }

  public dispose(): void {
    this.recordsBuffer.dispose();
    this.visibleIndicesBuffer.dispose();
    this.drawArgsBuffer.dispose();
    this.paramsBuffer.dispose();
  }

  private resetDrawArguments(): void {
    this.drawArgsBuffer.update(new Uint32Array([6, 0, 0, 0, 0]));
  }

  private createBuffer(
    byteLength: number,
    label: string,
    creationFlags = BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE
  ): StorageBuffer {
    return new BABYLON.StorageBuffer(this.engine, Math.max(4, byteLength), creationFlags, label);
  }

  private bindBuffers(): void {
    this.shader.setStorageBuffer('spriteRecords', this.recordsBuffer);
    this.shader.setStorageBuffer('motionState', this.motionState);
    this.shader.setStorageBuffer('visibleIndices', this.visibleIndicesBuffer);
    this.shader.setStorageBuffer('drawArgs', this.drawArgsBuffer);
    this.shader.setStorageBuffer('visibilityParams', this.paramsBuffer);
  }
}

export function emitSprite2DVisibilityWGSL(): string {
  return `
@group(0) @binding(0) var<storage, read> spriteRecords: array<vec4f>;
@group(0) @binding(1) var<storage, read> motionState: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgs: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> visibilityParams: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let count = visibilityParams[0];
  if (index >= count) { return; }

  let center = vec2f(bitcast<f32>(visibilityParams[1]), bitcast<f32>(visibilityParams[2]));
  let halfExtent = vec2f(bitcast<f32>(visibilityParams[3]), bitcast<f32>(visibilityParams[4]));
  let viewportHeight = bitcast<f32>(visibilityParams[5]);
  let minimumPixelSize = bitcast<f32>(visibilityParams[6]);
  let transform = spriteRecords[index * 3u];
  let spriteState = spriteRecords[index * 3u + 2u];
  let position = motionState[index].xy;
  let unrotatedHalfExtent = transform.zw * 0.5;
  let rotationCos = abs(cos(spriteState.x));
  let rotationSin = abs(sin(spriteState.x));
  let spriteHalfExtent = vec2f(
    rotationCos * unrotatedHalfExtent.x + rotationSin * unrotatedHalfExtent.y,
    rotationSin * unrotatedHalfExtent.x + rotationCos * unrotatedHalfExtent.y
  );
  let relative = abs(position - center);
  let intersects = all(relative <= halfExtent + spriteHalfExtent);
  let pixelsPerUnit = viewportHeight / max(0.0002, halfExtent.y * 2.0);
  let packedState = u32(spriteState.w + 0.5);
  let lodCode = packedState >> 3u;
  let spriteMinimumPixelSize = select(
    minimumPixelSize,
    (f32(lodCode) - 1.0) / 16.0,
    lodCode > 0u
  );
  let passesLod = max(transform.z, transform.w) * pixelsPerUnit >= spriteMinimumPixelSize;
  if (!intersects || !passesLod) { return; }

  let outputIndex = atomicAdd(&drawArgs[1], 1u);
  visibleIndices[outputIndex] = index;
}`;
}
