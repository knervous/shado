import {
  BABYLON,
  type BaseTexture,
  type ComputeShader,
  type StorageBuffer,
  type WebGPUEngine,
} from '../../babylon';
import { shadoHiZLayout, type ShadoHiZLayout } from './reference';
import {
  SHADO_HIZ_DEPTH_BIAS,
  SHADO_HIZ_MAX_LEVELS,
  type ShadoHiZBatch,
  type ShadoHiZCandidate,
  type ShadoHiZViewInput,
} from './types';
import {
  emitShadoHiZCullWGSL,
  emitShadoHiZFinalizeWGSL,
  emitShadoHiZReduceWGSL,
  emitShadoHiZResetWGSL,
  emitShadoHiZSeedWGSL,
  SHADO_HIZ_BATCH_WORDS,
  SHADO_HIZ_CANDIDATE_WORDS,
  SHADO_HIZ_DRAW_ARGS_WORDS,
  SHADO_HIZ_LEVEL_PARAM_WORDS,
  SHADO_HIZ_VIEW_WORDS,
  SHADO_HIZ_WORKGROUP_1D,
  SHADO_HIZ_WORKGROUP_2D,
} from './wgsl';

export interface ShadoWorldHiZRunResult {
  /** Every pass dispatched, so the draw arguments describe THIS frame. */
  readonly complete: boolean;
  /** The cull ran in admit-all mode (invalid view, mismatch, disabled). */
  readonly admitAll: boolean;
  readonly reason: string;
  readonly dispatches: number;
}

/**
 * Current-frame Hi-Z on WebGPU: depth texture -> conservative max pyramid ->
 * per-candidate test -> per-batch compaction + indexed-indirect arguments.
 *
 * This is a filter, not a visibility manager. It owns GPU resources and the
 * pass sequence; the caller owns which candidates exist, the depth pass and how
 * the arguments are consumed (see render/BabylonHiZAdapter). Nothing is read
 * back on the render-critical path.
 */
export class ShadoWorldHiZ {
  private layout?: ShadoHiZLayout;
  private pyramid?: StorageBuffer;
  private levelParams: StorageBuffer[] = [];
  private reducers: ComputeShader[] = [];
  private readonly seed: ComputeShader;
  private readonly reset: ComputeShader;
  private readonly cull: ComputeShader;
  private readonly finalize: ComputeShader;
  private readonly viewBuffer: StorageBuffer;
  private readonly viewWords = new Uint32Array(SHADO_HIZ_VIEW_WORDS);
  private readonly viewFloats = new Float32Array(this.viewWords.buffer);
  private candidateBuffer: StorageBuffer;
  private batchBuffer: StorageBuffer;
  private drawArgsBuffer: StorageBuffer;
  private visibleBuffer: StorageBuffer;
  private overflowBuffer: StorageBuffer;
  private flagsBuffer: StorageBuffer;
  private candidateWords = new Float32Array(SHADO_HIZ_CANDIDATE_WORDS);
  private batchWords = new Uint32Array(SHADO_HIZ_BATCH_WORDS);
  private candidateCount = 0;
  private batchCount = 0;
  private segments: number[] = [];
  private errors: string[] = [];
  public bias = SHADO_HIZ_DEPTH_BIAS;

  public constructor(private readonly engine: WebGPUEngine) {
    this.seed = this.shader('Shado Hi-Z seed', emitShadoHiZSeedWGSL(), {
      hizDepth: { group: 0, binding: 0 },
      hizPyramid: { group: 0, binding: 1 },
      hizLevelParams: { group: 0, binding: 2 },
    });
    this.reset = this.shader('Shado Hi-Z reset', emitShadoHiZResetWGSL(), {
      hizBatches: { group: 0, binding: 0 },
      hizDrawArgs: { group: 0, binding: 1 },
      hizOverflow: { group: 0, binding: 2 },
    });
    this.cull = this.shader('Shado Hi-Z cull', emitShadoHiZCullWGSL(), {
      hizPyramid: { group: 0, binding: 0 },
      hizView: { group: 0, binding: 1 },
      hizCandidates: { group: 0, binding: 2 },
      hizBatches: { group: 0, binding: 3 },
      hizDrawArgs: { group: 0, binding: 4 },
      hizVisible: { group: 0, binding: 5 },
      hizFlags: { group: 0, binding: 6 },
    });
    this.finalize = this.shader('Shado Hi-Z finalize', emitShadoHiZFinalizeWGSL(), {
      hizBatches: { group: 0, binding: 0 },
      hizDrawArgs: { group: 0, binding: 1 },
      hizOverflow: { group: 0, binding: 2 },
    });
    this.viewBuffer = this.buffer(this.viewWords.byteLength, 'Shado Hi-Z view');
    this.candidateBuffer = this.buffer(32, 'Shado Hi-Z candidates');
    this.batchBuffer = this.buffer(16, 'Shado Hi-Z batches');
    this.drawArgsBuffer = this.argsBuffer(1);
    this.visibleBuffer = this.buffer(4, 'Shado Hi-Z visible members');
    this.overflowBuffer = this.buffer(4, 'Shado Hi-Z overflow');
    this.flagsBuffer = this.buffer(4, 'Shado Hi-Z flags');
  }

  /** Indexed-indirect blocks, 5 words per batch, written every run. */
  public get drawArgs(): StorageBuffer {
    return this.drawArgsBuffer;
  }
  /** Compacted batch-local member indices; batch b starts at segmentOffset(b). */
  public get visibleMembers(): StorageBuffer {
    return this.visibleBuffer;
  }
  /** 1 when a batch overflowed and must draw every member uncompacted. */
  public get overflow(): StorageBuffer {
    return this.overflowBuffer;
  }
  /** One flag per candidate (see SHADO_HIZ_FLAG_*); debug readback only. */
  public get flags(): StorageBuffer {
    return this.flagsBuffer;
  }
  public get candidates(): number {
    return this.candidateCount;
  }
  public get batches(): number {
    return this.batchCount;
  }
  public get pyramidLayout(): ShadoHiZLayout | undefined {
    return this.layout;
  }
  public get lastErrors(): readonly string[] {
    return this.errors;
  }
  /**
   * GPU time of the last frame's Hi-Z compute passes, in ms, from Babylon's
   * per-dispatch timestamp queries; null when the engine was not measuring
   * (engine.enableGPUTimingMeasurements must be on before this was built, and
   * the device needs 'timestamp-query').
   */
  public gpuComputeMs(): number | null {
    const shaders = [this.seed, ...this.reducers, this.reset, this.cull, this.finalize];
    let total = 0;
    for (const shader of shaders) {
      const counter = (shader as any).gpuTimeInFrame?.counter;
      if (!counter) return null;
      total += counter.current ?? 0;
    }
    // Babylon's timestamp counters are nanoseconds.
    return total / 1e6;
  }

  public segmentOffset(batch: number): number {
    return this.segments[batch] ?? 0;
  }

  /**
   * Publishes the candidate/batch tables. Call again only when the set
   * changes; `updateBounds` handles a moved candidate without a rebuild.
   */
  public setCandidates(candidates: readonly ShadoHiZCandidate[], batches: readonly ShadoHiZBatch[]): void {
    const used = new Array<number>(batches.length).fill(0);
    for (const c of candidates) {
      const batch = batches[c.batch];
      if (!batch) throw new Error(`Hi-Z candidate ${c.id} addresses missing batch ${c.batch}`);
      if (!(c.member >= 0 && c.member < batch.capacity)) {
        throw new Error(`Hi-Z candidate ${c.id} member ${c.member} outside batch capacity ${batch.capacity}`);
      }
      used[c.batch]!++;
    }
    const batchWords = new Uint32Array(Math.max(1, batches.length) * SHADO_HIZ_BATCH_WORDS);
    this.segments = [];
    let segment = 0;
    batches.forEach((batch, b) => {
      this.segments.push(segment);
      batchWords[b * SHADO_HIZ_BATCH_WORDS + 0] = batch.indexCount >>> 0;
      batchWords[b * SHADO_HIZ_BATCH_WORDS + 1] = batch.firstIndex >>> 0;
      batchWords[b * SHADO_HIZ_BATCH_WORDS + 2] = batch.capacity >>> 0;
      batchWords[b * SHADO_HIZ_BATCH_WORDS + 3] = segment;
      batchWords[b * SHADO_HIZ_BATCH_WORDS + 4] = (batch.wholeInstances ?? 0) >>> 0;
      segment += batch.capacity;
    });
    this.batchWords = batchWords;
    const words = new Float32Array(Math.max(1, candidates.length) * SHADO_HIZ_CANDIDATE_WORDS);
    const bits = new Uint32Array(words.buffer);
    candidates.forEach((c, i) => {
      const o = i * SHADO_HIZ_CANDIDATE_WORDS;
      words[o + 0] = c.min[0];
      words[o + 1] = c.min[1];
      words[o + 2] = c.min[2];
      bits[o + 3] = c.batch >>> 0;
      words[o + 4] = c.max[0];
      words[o + 5] = c.max[1];
      words[o + 6] = c.max[2];
      bits[o + 7] = c.member >>> 0;
    });
    this.candidateWords = words;
    this.candidateCount = candidates.length;
    this.batchCount = batches.length;

    for (const b of [this.candidateBuffer, this.batchBuffer, this.drawArgsBuffer, this.visibleBuffer, this.overflowBuffer, this.flagsBuffer]) {
      b.dispose();
    }
    this.candidateBuffer = this.buffer(words.byteLength, 'Shado Hi-Z candidates');
    this.candidateBuffer.update(words);
    this.batchBuffer = this.buffer(batchWords.byteLength, 'Shado Hi-Z batches');
    this.batchBuffer.update(batchWords);
    this.drawArgsBuffer = this.argsBuffer(Math.max(1, batches.length));
    this.visibleBuffer = this.buffer(Math.max(1, segment) * 4, 'Shado Hi-Z visible members');
    // arrayLength(&hizOverflow) is the batch count in reset/finalize.
    this.overflowBuffer = this.buffer(Math.max(1, batches.length) * 4, 'Shado Hi-Z overflow');
    this.flagsBuffer = this.buffer(Math.max(1, candidates.length) * 4, 'Shado Hi-Z flags');
    this.bindTables();
  }

  /**
   * Changes an all-or-nothing batch's instance count (a thin-instance buffer
   * was rewritten). A queue write, so it lands before this frame's passes.
   */
  public setWholeInstances(batch: number, count: number): void {
    if (batch < 0 || batch >= this.batchCount) return;
    const at = batch * SHADO_HIZ_BATCH_WORDS + 4;
    if (this.batchWords[at] === count >>> 0) return;
    this.batchWords[at] = count >>> 0;
    // Babylon's update() reads from the START of `data` whatever the
    // destination offset, so hand it exactly the slice to write.
    this.batchBuffer.update(this.batchWords.subarray(at, at + 1), at * 4, 4);
  }

  /** Rewrites one candidate's bound in place (dirty update, no rebuild). */
  public updateBounds(index: number, min: readonly number[], max: readonly number[]): void {
    if (index < 0 || index >= this.candidateCount) return;
    const o = index * SHADO_HIZ_CANDIDATE_WORDS;
    const w = this.candidateWords;
    w[o + 0] = min[0]!;
    w[o + 1] = min[1]!;
    w[o + 2] = min[2]!;
    w[o + 4] = max[0]!;
    w[o + 5] = max[1]!;
    w[o + 6] = max[2]!;
    this.candidateBuffer.update(w.subarray(o, o + SHADO_HIZ_CANDIDATE_WORDS), o * 4, SHADO_HIZ_CANDIDATE_WORDS * 4);
  }

  /** (Re)allocates the pyramid for a viewport. Cheap when unchanged. */
  public resize(width: number, height: number): void {
    const current = this.layout?.levels[0];
    if (current && current.width === width && current.height === height) return;
    this.disposePyramid();
    // A minimised or not-yet-laid-out canvas: no pyramid, and run() admits
    // everything until a real size arrives.
    if (!(Number.isInteger(width) && Number.isInteger(height) && width >= 1 && height >= 1)) return;
    const layout = shadoHiZLayout(width, height);
    this.layout = layout;
    this.pyramid = this.buffer(layout.words * 4, 'Shado Hi-Z pyramid');
    // One params buffer (and reducer) per level: StorageBuffer.update is a
    // queue write that lands before the frame's commands, so a shared
    // per-dispatch parameter block would give every level the last value.
    layout.levels.forEach((dst, l) => {
      const src = layout.levels[Math.max(0, l - 1)]!;
      const params = new Uint32Array(SHADO_HIZ_LEVEL_PARAM_WORDS);
      params.set([src.offset, src.width, src.height, dst.offset, dst.width, dst.height]);
      const buffer = this.buffer(params.byteLength, `Shado Hi-Z level ${l} params`);
      this.levelParams.push(buffer);
      if (l === 0) return;
      const reducer = this.shader(`Shado Hi-Z reduce ${l}`, emitShadoHiZReduceWGSL(), {
        hizPyramid: { group: 0, binding: 0 },
        hizLevelParams: { group: 0, binding: 1 },
      });
      reducer.setStorageBuffer('hizPyramid', this.pyramid!);
      reducer.setStorageBuffer('hizLevelParams', buffer);
      this.reducers.push(reducer);
    });
    this.seed.setStorageBuffer('hizPyramid', this.pyramid);
    this.seed.setStorageBuffer('hizLevelParams', this.levelParams[0]!);
    this.cull.setStorageBuffer('hizPyramid', this.pyramid);
    // Every new buffer starts zeroed; a zero parameter block makes the seed
    // write nothing and leaves a pyramid of depth 0 -- a wall at the near
    // plane that hides everything. Write them now, whatever convention the
    // previous pyramid used.
    this.levelParamsWritten = false;
    this.writeLevelParams(this.convention);
  }

  /**
   * Runs the whole sequence into Babylon's frame encoder, after the depth
   * pass and before the draws that consume the arguments. With `view` null,
   * or a viewport that does not match the depth, every candidate is admitted
   * (the arguments stay valid, nothing is rejected).
   */
  public run(depth: BaseTexture | null, view: ShadoHiZViewInput | null, admitReason = ''): ShadoWorldHiZRunResult {
    let reason = admitReason;
    const size = depth?.getSize();
    if (!reason && !view) reason = 'no view';
    if (!reason && (!depth || !size)) reason = 'no depth';
    if (!reason && view && size && (size.width !== view.viewportWidth || size.height !== view.viewportHeight)) {
      reason = `depth ${size.width}x${size.height} != viewport ${view.viewportWidth}x${view.viewportHeight}`;
    }
    const base = this.layout?.levels[0];
    if (!reason && view && (!base || base.width !== view.viewportWidth || base.height !== view.viewportHeight)) {
      reason = base
        ? `pyramid ${base.width}x${base.height} != viewport ${view.viewportWidth}x${view.viewportHeight}`
        : 'no pyramid';
    }
    const admitAll = reason !== '';
    let dispatches = 0;
    let complete = true;
    const go = (shader: ComputeShader, x: number, y = 1): void => {
      if (x <= 0 || y <= 0) return;
      if (shader.dispatch(x, y, 1)) dispatches++;
      else complete = false;
    };

    if (!admitAll && view && depth && this.layout) {
      this.writeLevelParams(view.convention);
      this.seed.setTexture('hizDepth', depth, false);
      go(this.seed, Math.ceil(base!.width / SHADO_HIZ_WORKGROUP_2D), Math.ceil(base!.height / SHADO_HIZ_WORKGROUP_2D));
      this.reducers.forEach((reducer, i) => {
        const dst = this.layout!.levels[i + 1]!;
        go(reducer, Math.ceil(dst.width / SHADO_HIZ_WORKGROUP_2D), Math.ceil(dst.height / SHADO_HIZ_WORKGROUP_2D));
      });
    }
    this.writeView(view, admitAll || !this.layout);
    go(this.reset, Math.ceil(this.batchCount / SHADO_HIZ_WORKGROUP_1D));
    go(this.cull, Math.ceil(this.candidateCount / SHADO_HIZ_WORKGROUP_1D));
    go(this.finalize, Math.ceil(this.batchCount / SHADO_HIZ_WORKGROUP_1D));
    return { complete, admitAll, reason, dispatches };
  }

  /** Debug only: resolves a copy of the per-candidate flags a frame later. */
  public async readFlags(): Promise<Uint32Array> {
    // The candidate set may be rebuilt while this is in flight; answer for
    // the buffer that was read, never index past it.
    const count = this.candidateCount;
    const view = await this.flagsBuffer.read();
    return new Uint32Array(view.buffer, view.byteOffset, Math.min(count, Math.floor(view.byteLength / 4)));
  }

  /** Debug only. */
  public async readDrawArgs(): Promise<Uint32Array> {
    const words = this.batchCount * SHADO_HIZ_DRAW_ARGS_WORDS;
    const view = await this.drawArgsBuffer.read();
    return new Uint32Array(view.buffer, view.byteOffset, Math.min(words, Math.floor(view.byteLength / 4)));
  }

  /** Debug only: level 0 of the pyramid (the seeded depth), row-major. */
  public async readPyramid(): Promise<Float32Array | undefined> {
    if (!this.pyramid || !this.layout) return undefined;
    const view = await this.pyramid.read();
    return new Float32Array(view.buffer.slice(view.byteOffset, view.byteOffset + this.layout.words * 4));
  }

  public dispose(): void {
    this.disposePyramid();
    for (const b of [this.viewBuffer, this.candidateBuffer, this.batchBuffer, this.drawArgsBuffer, this.visibleBuffer, this.overflowBuffer, this.flagsBuffer]) {
      b.dispose();
    }
  }

  private disposePyramid(): void {
    this.pyramid?.dispose();
    this.pyramid = undefined;
    for (const b of this.levelParams) b.dispose();
    this.levelParams = [];
    this.reducers = [];
    this.layout = undefined;
  }

  private convention: 'normal' | 'reversed' = 'normal';
  private levelParamsWritten = false;
  private writeLevelParams(convention: 'normal' | 'reversed'): void {
    if (!this.layout) return;
    if (this.levelParamsWritten && convention === this.convention) return;
    this.convention = convention;
    this.levelParamsWritten = true;
    const clear = new Float32Array([convention === 'normal' ? 1 : 0]);
    const clearBits = new Uint32Array(clear.buffer)[0]!;
    this.layout.levels.forEach((dst, l) => {
      const src = this.layout!.levels[Math.max(0, l - 1)]!;
      const params = new Uint32Array(SHADO_HIZ_LEVEL_PARAM_WORDS);
      params.set([src.offset, src.width, src.height, dst.offset, dst.width, dst.height, clearBits, convention === 'reversed' ? 1 : 0]);
      this.levelParams[l]!.update(params);
    });
  }

  private writeView(view: ShadoHiZViewInput | null, admitAll: boolean): void {
    const w = this.viewWords;
    const f = this.viewFloats;
    w.fill(0);
    if (view) {
      for (let i = 0; i < 16; i++) f[i] = view.viewProjection[i]!;
      w[16] = view.viewportWidth >>> 0;
      w[17] = view.viewportHeight >>> 0;
      w[18] = view.convention === 'reversed' ? 1 : 0;
      w[19] = view.ndcHalfZRange ? 1 : 0;
      w[20] = view.topLeftOrigin ? 1 : 0;
    }
    w[21] = admitAll ? 1 : 0;
    f[22] = this.bias;
    const levels = this.layout?.levels ?? [];
    w[23] = levels.length;
    w[24] = this.candidateCount;
    levels.slice(0, SHADO_HIZ_MAX_LEVELS).forEach((level, i) => {
      w[28 + i * 4] = level.width;
      w[29 + i * 4] = level.height;
      w[30 + i * 4] = level.offset;
    });
    this.viewBuffer.update(w);
  }

  private bindTables(): void {
    this.reset.setStorageBuffer('hizBatches', this.batchBuffer);
    this.reset.setStorageBuffer('hizDrawArgs', this.drawArgsBuffer);
    this.reset.setStorageBuffer('hizOverflow', this.overflowBuffer);
    this.cull.setStorageBuffer('hizView', this.viewBuffer);
    this.cull.setStorageBuffer('hizCandidates', this.candidateBuffer);
    this.cull.setStorageBuffer('hizBatches', this.batchBuffer);
    this.cull.setStorageBuffer('hizDrawArgs', this.drawArgsBuffer);
    this.cull.setStorageBuffer('hizVisible', this.visibleBuffer);
    this.cull.setStorageBuffer('hizFlags', this.flagsBuffer);
    // The cull needs a pyramid binding even in admit-all mode.
    if (!this.pyramid) this.resize(1, 1);
    this.finalize.setStorageBuffer('hizBatches', this.batchBuffer);
    this.finalize.setStorageBuffer('hizDrawArgs', this.drawArgsBuffer);
    this.finalize.setStorageBuffer('hizOverflow', this.overflowBuffer);
  }

  private shader(name: string, source: string, bindingsMapping: Record<string, { group: number; binding: number }>): ComputeShader {
    const shader = new BABYLON.ComputeShader(name, this.engine, { computeSource: source }, { bindingsMapping });
    shader.onError = (_effect, errors) => {
      this.errors.push(`${name}: ${errors}`);
    };
    return shader;
  }

  private buffer(byteLength: number, label: string): StorageBuffer {
    return new BABYLON.StorageBuffer(this.engine, Math.max(4, byteLength), BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE, label);
  }

  private argsBuffer(batches: number): StorageBuffer {
    return new BABYLON.StorageBuffer(
      this.engine,
      batches * SHADO_HIZ_DRAW_ARGS_WORDS * 4,
      BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE | BABYLON.Constants.BUFFER_CREATIONFLAG_INDIRECT,
      'Shado Hi-Z draw arguments'
    );
  }
}
