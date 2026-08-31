/**
 * A {@link ImageFrameSource} over a live canvas — the browser capture path.
 *
 * The frame never leaves the GPU: the canvas goes straight into a `VideoFrame`
 * and from there into the hardware encoder. That is what makes in-browser
 * capture nearly free compared with the headless path, where reading pixels
 * back is ~99% of the cost.
 *
 * Typed against a minimal structural canvas so this file does not depend on the
 * DOM and stays importable from anywhere.
 */

import type { ImageFrame, ImageFrameSource } from './types';

/** A canvas or OffscreenCanvas — anything `new VideoFrame(source)` accepts. */
export interface CanvasLike {
  readonly width: number;
  readonly height: number;
}

export interface CanvasFrameSourceOptions {
  /**
   * Called before each frame is captured, with that frame's time in seconds.
   *
   * For a live app this is usually where you render — the driver decides *when*
   * a frame is wanted, and this produces it. Leave it out if something else
   * already drives the render loop and the canvas is simply read as it stands.
   */
  onFrame?: (timeSeconds: number) => void;
  /** Overrides the canvas dimensions; defaults to the canvas's own. */
  width?: number;
  height?: number;
}

export function createCanvasFrameSource(
  canvas: CanvasLike,
  options: CanvasFrameSourceOptions = {},
): ImageFrameSource {
  const width = options.width ?? canvas.width;
  const height = options.height ?? canvas.height;
  const VideoFrameCtor = (globalThis as unknown as {
    VideoFrame?: new (source: unknown, init: unknown) => { close(): void };
  }).VideoFrame;
  if (!VideoFrameCtor) throw new Error('This runtime has no WebCodecs VideoFrame');

  return {
    kind: 'image',
    width,
    height,
    // No pipelining: there is nothing to overlap. The capture is a reference to
    // the canvas, not a readback, so a second one in flight would buy nothing
    // and would let two frames of the same canvas race.
    depth: 1,
    async frame(timeSeconds, timestampUs): Promise<ImageFrame> {
      options.onFrame?.(timeSeconds);
      const videoFrame = new VideoFrameCtor(canvas, { timestamp: timestampUs });
      return {
        kind: 'image',
        source: videoFrame,
        width,
        height,
        timestampUs,
        // The driver calls this once the encoder is done. A VideoFrame holds a
        // GPU allocation open until closed and the browser throttles hard
        // before it complains.
        release: () => videoFrame.close(),
      };
    },
  };
}
