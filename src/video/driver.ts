/**
 * Drives one capture: pull a frame, encode it, repeat, finalize.
 *
 * Two clocks, chosen by `pacing`:
 *
 * - `offline` runs as fast as the renderer manages and stamps exact timestamps.
 *   A heavy scene can take seconds per frame and the output is still exactly
 *   `fps` frames per second of playback. That is the difference between
 *   rendering a video and screen-recording one.
 * - `realtime` holds each frame until its wall-clock moment, for streaming a
 *   running app to a viewer.
 *
 * Frames are requested ahead of where encoding has reached and encoded strictly
 * in order. That is worth real throughput rather than being tidiness: a
 * headless GPU readback is ~99% of capture time and is latency, not bandwidth,
 * so N in flight cost about what one costs — measured 8x at depth 8 on Dawn.
 */

import type {
  CapturedFrame,
  FrameSource,
  ShadoVideoEncoder,
  VideoCodec,
  VideoSink,
  VideoSpec,
} from './types';

export type Pacing = 'offline' | 'realtime';

export interface RenderVideoOptions {
  source: FrameSource;
  encoder: ShadoVideoEncoder;
  sink: VideoSink;
  /**
   * Duration of the finished video; frame count is `round(seconds * fps)`.
   * Omit it for an open-ended realtime stream, which then needs `signal`.
   */
  seconds?: number;
  fps?: number;
  /** Defaults to `offline`. */
  pacing?: Pacing;
  codec?: VideoCodec;
  bitrate?: number;
  crf?: number;
  preset?: VideoSpec['preset'];
  /** Encode on the GPU's fixed-function video block instead of in software. */
  hardware?: boolean;
  hardwareEncoder?: string;
  /**
   * Overrides how many frames may be in flight. Defaults to the source's own
   * `depth` offline, and to at most {@link REALTIME_LOOKAHEAD} in realtime,
   * where every frame of lookahead is a frame of latency.
   */
  lookahead?: number;
  onProgress?: (frame: number, total: number, elapsedMs: number) => void;
  /**
   * Called when a frame misses its realtime deadline, with how late it was.
   *
   * Nothing is dropped or duplicated in response — the renderer measures 62fps
   * at 1080p, so persistent lateness means something upstream is wrong and
   * silently papering over it would hide that. Lower the fps or the resolution.
   */
  onLate?: (frame: number, lateMs: number) => void;
  signal?: AbortSignal;
  /** Dispose the frame source when the run ends. Off by default; the caller owns it. */
  disposeSource?: boolean;
}

export interface VideoResult {
  frames: number;
  seconds: number;
  fps: number;
  width: number;
  height: number;
  elapsedMs: number;
  /** Frames that missed their realtime deadline. Always 0 offline. */
  lateFrames: number;
  /**
   * Where the time went, in milliseconds.
   *
   * Worth having because the answer is rarely what you assume: on the headless
   * path `source` dominates so heavily that encoding is unmeasurable, and it
   * took splitting these apart to discover that the cost was a fixed readback
   * stall rather than anything to do with resolution or the encoder.
   *
   * `sourceMs` is time blocked waiting for frames, so with pipelining it is
   * overlap-adjusted rather than the raw cost of rendering one.
   */
  timing: {
    sourceMs: number;
    encodeMs: number;
    /** Realtime pacing only: time deliberately spent waiting for the clock. */
    pacingMs: number;
  };
}

const DEFAULT_FPS = 30;

/** Realtime lookahead cap: latency is `lookahead / fps`, so 2 is ~66ms at 30fps. */
export const REALTIME_LOOKAHEAD = 2;

/** Catches a source whose frames disagree with what it promised the encoder. */
function assertMatchesSpec(frame: CapturedFrame, spec: VideoSpec, index: number): void {
  if (frame.width !== spec.width || frame.height !== spec.height) {
    throw new Error(
      `Frame ${index} is ${frame.width}x${frame.height}, but the encoder was opened for ${spec.width}x${spec.height}`,
    );
  }
  if (frame.kind !== spec.sourceKind) {
    throw new Error(`Frame ${index} is a ${frame.kind} frame, but the source declared ${spec.sourceKind}`);
  }
  if (frame.kind !== 'buffer') return;
  if (frame.format !== spec.sourceFormat || frame.flipped !== spec.sourceFlipped) {
    throw new Error(
      `Frame ${index} is ${frame.format}${frame.flipped ? ' flipped' : ''}, but the source declared `
      + `${spec.sourceFormat}${spec.sourceFlipped ? ' flipped' : ''}`,
    );
  }
  const expected = frame.format === 'yuv420p'
    ? (frame.width * frame.height * 3) / 2
    : frame.width * frame.height * 4;
  if (frame.data.byteLength !== expected) {
    throw new Error(`Frame ${index} carries ${frame.data.byteLength} bytes; ${expected} expected`);
  }
}

const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

export async function renderVideo(options: RenderVideoOptions): Promise<VideoResult> {
  const { source, encoder, sink } = options;
  const fps = options.fps ?? DEFAULT_FPS;
  const pacing = options.pacing ?? 'offline';
  if (!(fps > 0)) throw new Error(`fps must be positive, got ${fps}`);
  if (options.seconds !== undefined && !(options.seconds > 0)) {
    throw new Error(`seconds must be positive, got ${options.seconds}`);
  }
  if (options.seconds === undefined && !options.signal) {
    throw new Error('An open-ended capture needs a signal to stop it; pass seconds or signal');
  }
  // Infinity is deliberate: it makes an open-ended stream fall out of the same
  // loop as a fixed-length one instead of needing a second code path.
  const total = options.seconds === undefined ? Infinity : Math.max(1, Math.round(options.seconds * fps));

  const spec: VideoSpec = {
    width: source.width,
    height: source.height,
    fps,
    sourceKind: source.kind,
    ...(source.kind === 'buffer' ? { sourceFormat: source.format, sourceFlipped: source.flipped } : {}),
    ...(pacing === 'realtime' ? { latencyCritical: true } : {}),
    ...(options.codec ? { codec: options.codec } : {}),
    ...(options.bitrate ? { bitrate: options.bitrate } : {}),
    ...(options.crf !== undefined ? { crf: options.crf } : {}),
    ...(options.preset ? { preset: options.preset } : {}),
    ...(options.hardware ? { hardware: true } : {}),
    ...(options.hardwareEncoder ? { hardwareEncoder: options.hardwareEncoder } : {}),
  };

  const sourceDepth = source.depth ?? 1;
  const requested = options.lookahead ?? (pacing === 'realtime' ? Math.min(sourceDepth, REALTIME_LOOKAHEAD) : sourceDepth);
  const depth = Math.max(1, Math.min(requested, total));

  // Held outside the try: on failure these are still in flight and their
  // rejections have to be absorbed somewhere.
  const pending: Array<Promise<CapturedFrame>> = [];
  let queued = 0;
  const requestNext = (): void => {
    if (queued >= total) return;
    // Times are derived from the index rather than accumulated, so rounding
    // cannot drift the last frame of a long capture off the grid.
    const timeSeconds = queued / fps;
    const timestampUs = Math.round((queued * 1_000_000) / fps);
    queued++;
    pending.push(source.frame(timeSeconds, timestampUs));
  };

  const frameMs = 1000 / fps;
  const started = Date.now();
  let lateFrames = 0;
  let encoded = 0;
  let sourceMs = 0;
  let encodeMs = 0;
  let pacingMs = 0;

  await encoder.open(spec, sink);
  try {
    for (let i = 0; i < depth; i++) requestNext();

    for (let index = 0; index < total; index++) {
      if (options.signal?.aborted) break;
      const waitStarted = Date.now();
      const frame = await pending.shift()!;
      sourceMs += Date.now() - waitStarted;
      assertMatchesSpec(frame, spec, index);

      if (pacing === 'realtime') {
        // Paced here rather than at request time so the pipeline still gets to
        // overlap its readbacks; the cost is `depth` frames of latency, which
        // is why realtime caps the depth rather than disabling it.
        const due = started + index * frameMs;
        const early = due - Date.now();
        if (early > 0) { await delay(early); pacingMs += early; }
        else if (early < -frameMs) {
          lateFrames++;
          options.onLate?.(index, -early);
        }
      }

      const encodeStarted = Date.now();
      await encoder.encode(frame);
      encodeMs += Date.now() - encodeStarted;
      // A VideoFrame holds a GPU allocation open until it is closed, and the
      // browser stops handing out new ones long before it warns you.
      if (frame.kind === 'image') frame.release?.();
      encoded++;
      // Refilled only after the frame is encoded, so a slow sink still bounds
      // how far ahead the renderer may run.
      requestNext();
      options.onProgress?.(encoded, total, Date.now() - started);
      if (options.signal?.aborted) break;
    }
    await encoder.close();
  } catch (error) {
    // The encoder owns the sink, so aborting it is what tears the sink down;
    // doing both here would close a stream twice.
    await encoder.abort(error instanceof Error ? error : new Error(String(error)));
    throw error;
  } finally {
    // In-flight frames outlive a failure or an early stop. Unhandled rejections
    // from them would crash the process and bury whatever actually went wrong.
    for (const frame of pending) frame.catch(() => undefined);
    if (options.disposeSource) await source.dispose?.();
  }

  return {
    frames: encoded,
    seconds: encoded / fps,
    fps,
    width: spec.width,
    height: spec.height,
    elapsedMs: Date.now() - started,
    lateFrames,
    timing: { sourceMs, encodeMs, pacingMs },
  };
}
