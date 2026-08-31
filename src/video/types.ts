/**
 * The three seams of shado video capture: where frames come from, how they are
 * compressed, and where the bytes go.
 *
 * They are deliberately ignorant of each other. A frame source does not know
 * whether it is feeding a file or a socket; an encoder does not know whether it
 * is fed by a headless Dawn render or a live browser canvas; a sink is a byte
 * pipe that knows nothing about video at all. Adding a transport is a new
 * {@link VideoSink} and nothing else.
 *
 * Nothing in this file imports node or the DOM, so both runtimes can hold the
 * same contract.
 */

/**
 * Pixel layout of a read-back frame.
 *
 * `yuv420p` is planar Y/U/V at 1.5 bytes per pixel, already in the layout an
 * H.264 encoder wants — converting on the GPU instead of letting the encoder
 * do it on the CPU is the single largest saving in the pipeline.
 */
export type FrameFormat = 'rgba8' | 'bgra8' | 'yuv420p';

interface FrameBase {
  width: number;
  height: number;
  /** Presentation time from the start of the stream, in microseconds. */
  timestampUs: number;
}

/**
 * A frame that has been read back to CPU memory.
 *
 * `data` is borrowed, not owned: a source is free to hand back the same buffer
 * every frame, and the encoder consumes it before asking for the next one.
 * Anything that keeps a frame past `encode()` must copy it.
 */
export interface BufferFrame extends FrameBase {
  kind: 'buffer';
  data: Uint8Array;
  format: FrameFormat;
  /** Rows run bottom-up, the GL readback convention. */
  flipped: boolean;
}

/**
 * A frame that never left the GPU.
 *
 * This is the whole reason the browser path is cheap: a canvas can be handed
 * straight to WebCodecs, so there is no readback, no format conversion and no
 * copy. `source` is anything `new VideoFrame(source)` accepts — a canvas, an
 * `OffscreenCanvas`, an `ImageBitmap` — or an already-built `VideoFrame`. Typed
 * as `unknown` because this file must not depend on the DOM.
 */
export interface ImageFrame extends FrameBase {
  kind: 'image';
  source: unknown;
  /** Called once the encoder is done with it; closes a `VideoFrame`. */
  release?: () => void;
}

export type CapturedFrame = BufferFrame | ImageFrame;

interface FrameSourceBase {
  readonly width: number;
  readonly height: number;
  /**
   * How many `frame()` calls the driver may leave outstanding at once.
   *
   * Defaults to 1. A source that can overlap its work reports more, and the
   * driver keeps that many in flight while still encoding strictly in order.
   * A source declaring more than 1 must apply its state and queue its work
   * synchronously before returning, or frames will race each other.
   */
  readonly depth?: number;
  dispose?(): Promise<void> | void;
}

export interface BufferFrameSource extends FrameSourceBase {
  readonly kind: 'buffer';
  /** Declared up front so the encoder can be configured before frame zero. */
  readonly format: FrameFormat;
  readonly flipped: boolean;
  frame(timeSeconds: number, timestampUs: number): Promise<BufferFrame>;
}

export interface ImageFrameSource extends FrameSourceBase {
  readonly kind: 'image';
  frame(timeSeconds: number, timestampUs: number): Promise<ImageFrame>;
}

/**
 * A producer of frames at requested points in time.
 *
 * Time is an argument rather than a wall clock even for live capture: the
 * driver decides the schedule, so the same source serves an offline render that
 * runs as fast as it can and a realtime stream paced to the clock.
 */
export type FrameSource = BufferFrameSource | ImageFrameSource;

export type VideoCodec = 'h264' | 'vp9';

export interface VideoSpec {
  width: number;
  height: number;
  fps: number;
  codec?: VideoCodec;
  /** Target bitrate in bits per second. Ignored when `crf` is set. */
  bitrate?: number;
  /**
   * Constant-quality target, 0 (lossless) to 51. Preferred over `bitrate` for
   * files; a live transport usually wants a bitrate it can plan around.
   */
  crf?: number;
  /** Trades encoder CPU for compression. Ignored when `hardware` is set. */
  preset?: 'fast' | 'balanced' | 'quality';
  /**
   * Encode on the GPU's dedicated video block rather than in software.
   *
   * Not WebGPU — that has no encode API. This is the fixed-function encoder
   * every modern GPU carries (VideoToolbox, NVENC, VAAPI), which ffmpeg can
   * drive. It trades compression efficiency for speed and does not support
   * CRF, so a bitrate is used instead.
   */
  hardware?: boolean;
  /** Explicit encoder name, e.g. `h264_nvenc`. Overrides the platform default. */
  hardwareEncoder?: string;
  /** Whether frames will arrive as CPU buffers or as GPU-resident images. */
  sourceKind: 'buffer' | 'image';
  /** Channel order of incoming frames. Present when `sourceKind` is 'buffer'. */
  sourceFormat?: FrameFormat;
  /** Whether those frames arrive bottom-up. Present when `sourceKind` is 'buffer'. */
  sourceFlipped?: boolean;
  /**
   * Hint that latency matters more than compression — a live transport rather
   * than a file. Encoders use it to shorten their keyframe interval and stop
   * buffering frames internally.
   */
  latencyCritical?: boolean;
}

/**
 * Somewhere for encoded bytes to go.
 *
 * `write` returns a promise so a slow transport applies real backpressure — the
 * driver awaits it, and a long capture to a congested socket stalls rather than
 * growing without bound in memory.
 */
export interface VideoSink {
  write(bytes: Uint8Array): Promise<void>;
  /** Finalizes the stream. Only called when the run succeeded. */
  close(): Promise<void>;
  /** Abandons the stream. A file sink deletes; an HTTP sink cancels. */
  abort?(error: Error): Promise<void> | void;
}

/**
 * Compresses frames into a container and writes it to a sink as it goes.
 *
 * The encoder holds the sink rather than returning chunks, because the byte
 * order of a container is the encoder's business and threading it back through
 * the driver would only give the driver a chance to get it wrong.
 */
export interface ShadoVideoEncoder {
  open(spec: VideoSpec, sink: VideoSink): Promise<void>;
  encode(frame: CapturedFrame): Promise<void>;
  /** Flushes, finalizes the container, and closes the sink. */
  close(): Promise<void>;
  abort(error: Error): Promise<void>;
}
