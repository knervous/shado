/**
 * A {@link FrameSource} over a headless preview session.
 *
 * Typed structurally rather than against `PreviewSession`, so `@knervous/shado/video`
 * does not import `devtools` and drag Dawn, sharp and node:zlib behind it. Any
 * object that can render a frame into a buffer satisfies this — the browser
 * client's own engine wrapper included.
 */

import type { BufferFrame, BufferFrameSource } from './types';

/** A frame the session hands back, before it is stamped with a timestamp. */
export interface SessionRawFrame {
  data: Uint8Array;
  width: number;
  height: number;
  flipped: boolean;
  format?: 'rgba8' | 'bgra8' | 'yuv420p';
}

/** The pipelined capture loop a session exposes. */
export interface SessionCaptureStream {
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly format?: 'rgba8' | 'bgra8' | 'yuv420p';
  readonly flipped?: boolean;
  capture(): Promise<SessionRawFrame>;
  dispose(): void;
}

/** The slice of a preview session a capture loop actually needs. */
export interface CaptureCapableSession {
  readonly scene: unknown;
  /**
   * Pixel layout `captureRaw` returns, when the session has no capture stream
   * to declare it. Defaults to `rgba8` bottom-up, which is what Babylon's
   * `readPixels` produces — a session whose readback differs (Lite reads its
   * swapchain, which is bgra8 top-down) must say so, or the driver rejects its
   * first frame.
   */
  readonly captureFormat?: 'rgba8' | 'bgra8' | 'yuv420p';
  readonly captureFlipped?: boolean;
  captureRaw(options?: { width?: number; height?: number; camera?: unknown }): Promise<SessionRawFrame>;
  createCaptureStream?(options?: {
    width?: number;
    height?: number;
    camera?: unknown;
    depth?: number;
    format?: 'rgba8' | 'bgra8' | 'yuv420p';
  }): Promise<SessionCaptureStream>;
}

export interface SessionFrameSourceOptions {
  width: number;
  height: number;
  /** Rendered from this camera; defaults to whatever the session last framed. */
  camera?: unknown;
  /**
   * Called before each frame renders, with that frame's time in seconds.
   * This is where a capture animates: move the camera, seek an animation
   * group, advance a simulation.
   *
   * It must be synchronous when the source is pipelined. Returning a promise
   * would let the next frame's state be applied before this one has rendered,
   * and the capture would silently come out with frames on the wrong poses.
   */
  onFrame?: (timeSeconds: number) => void;
  /**
   * GPU readbacks in flight. Left unset, the session picks; 1 disables
   * pipelining, which is the safe choice if anything in the scene is driven
   * asynchronously between frames.
   */
  depth?: number;
  /**
   * Convert to yuv420p on the GPU rather than reading back RGBA. Much faster
   * end to end, but needs width % 8 == 0 and height % 2 == 0.
   */
  format?: 'rgba8' | 'bgra8' | 'yuv420p';
}

/**
 * Builds a frame source over a session, pipelining if the session supports it.
 *
 * `createCaptureStream` is optional on the session so an older or simpler
 * implementation still works — it just runs one readback at a time, which on
 * Dawn is roughly eight times slower.
 */
export async function createSessionFrameSource(
  session: CaptureCapableSession,
  options: SessionFrameSourceOptions,
): Promise<BufferFrameSource> {
  const { width, height } = options;
  const camera = options.camera !== undefined ? { camera: options.camera } : {};
  const stream = session.createCaptureStream
    ? await session.createCaptureStream({
        width,
        height,
        ...camera,
        ...(options.depth ? { depth: options.depth } : {}),
        ...(options.format ? { format: options.format } : {}),
      })
    : null;

  const advance = (timeSeconds: number): void => {
    const result = options.onFrame?.(timeSeconds) as unknown;
    // Enforced rather than documented: an async onFrame under pipelining
    // produces frames rendered from the wrong state, and nothing about the
    // output would tell you that is what happened.
    if (stream && stream.depth > 1 && result && typeof (result as Promise<void>).then === 'function') {
      throw new Error('onFrame must be synchronous when the capture is pipelined; pass depth: 1 to opt out');
    }
  };

  const stamp = (raw: SessionRawFrame, timestampUs: number): BufferFrame => ({
    kind: 'buffer',
    data: raw.data,
    width: raw.width,
    height: raw.height,
    // Babylon's readback is RGBA8 with bottom-up rows; the encoder flips it,
    // which costs nothing there and a full-resolution pass here.
    format: raw.format ?? 'rgba8',
    flipped: raw.flipped,
    timestampUs,
  });

  const format = stream?.format ?? session.captureFormat ?? 'rgba8';
  // Always declared by whoever produces the frames — inferring it from the
  // format is wrong the moment a second backend reads a swapchain instead of
  // calling readPixels.
  const flipped = stream
    ? stream.flipped ?? format !== 'yuv420p'
    : session.captureFlipped ?? true;
  return {
    kind: 'buffer',
    width,
    height,
    format,
    flipped,
    depth: stream?.depth ?? 1,
    async frame(timeSeconds, timestampUs): Promise<BufferFrame> {
      advance(timeSeconds);
      // No await between advancing the state and queueing the render, so
      // several of these may be outstanding at once without racing.
      const raw = stream ? stream.capture() : session.captureRaw({ width, height, ...camera });
      return stamp(await raw, timestampUs);
    },
    dispose() {
      stream?.dispose();
    },
  };
}

export interface OrbitOptions {
  /** Duration the orbit is spread across — normally the capture's length. */
  seconds: number;
  /** Full turns completed over that duration. */
  revolutions?: number;
  /** Starting angle in radians; defaults to the camera's current alpha. */
  from?: number;
  /** Elevation to hold, in radians. Left alone when unset. */
  beta?: number;
}

/**
 * An `onFrame` that spins an ArcRotate camera — the turntable shot.
 *
 * Untyped in the camera because this module deliberately does not import
 * Babylon; it needs `alpha` and nothing else.
 */
export function orbitCamera(camera: { alpha: number; beta: number }, options: OrbitOptions): (t: number) => void {
  const revolutions = options.revolutions ?? 1;
  const from = options.from ?? camera.alpha;
  return (timeSeconds: number) => {
    camera.alpha = from + 2 * Math.PI * revolutions * (timeSeconds / options.seconds);
    if (options.beta !== undefined) camera.beta = options.beta;
  };
}
