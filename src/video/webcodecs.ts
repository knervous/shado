/**
 * The browser encoder: WebCodecs plus an MP4 muxer, and no ffmpeg anywhere.
 *
 * This is the path where the whole readback problem disappears. A canvas can be
 * handed straight to `VideoFrame`, so the pixels never come back to the CPU;
 * the hardware encoder takes them from there and what reaches this code is
 * already-compressed chunks. On the headless node path the same frames cost
 * ~100ms each to read back (see the driver's notes) — here they cost nothing.
 *
 * Output is fragmented MP4, byte-for-byte the same shape the ffmpeg encoder
 * produces, so every {@link VideoSink} and every receiver works unchanged.
 *
 * Requires `VideoEncoder` (Chromium, Safari 17+, Firefox 130+) and the
 * `mp4-muxer` package. This is the browser path only: Node has no built-in
 * WebCodecs, and the native implementations that provide it are themselves
 * built on ffmpeg, so `createFfmpegEncoder` remains the Node encoder.
 */

import type { CapturedFrame, ShadoVideoEncoder, VideoSink, VideoSpec } from './types';

/** Enough of the WebCodecs surface to encode; avoids depending on DOM lib types. */
interface EncoderLike {
  configure(config: unknown): void;
  encode(frame: unknown, options?: { keyFrame?: boolean }): void;
  flush(): Promise<void>;
  close(): void;
  readonly encodeQueueSize: number;
}

export interface WebCodecsEncoderOptions {
  /**
   * Codec string for `VideoEncoder.configure`. Defaults to H.264 baseline 4.2,
   * which is the widest-decoding profile and what MSE will accept everywhere.
   */
  codecString?: string;
  /** Prefer the hardware encoder, the software one, or let the browser choose. */
  acceleration?: 'prefer-hardware' | 'prefer-software' | 'no-preference';
  /** Overrides the muxer import, for bundlers that need it resolved statically. */
  loadMuxer?: () => Promise<typeof import('mp4-muxer')>;
}

const DEFAULT_CODEC_STRING = 'avc1.42E02A';

export function isWebCodecsSupported(): boolean {
  return typeof (globalThis as { VideoEncoder?: unknown }).VideoEncoder === 'function';
}

export function createWebCodecsEncoder(options: WebCodecsEncoderOptions = {}): ShadoVideoEncoder {
  let encoder: EncoderLike | null = null;
  let muxer: {
    addVideoChunkRaw(
      data: Uint8Array, type: string, timestamp: number, duration: number, meta?: unknown,
    ): void;
    finalize(): void;
  } | null = null;
  let frameDurationUs = 0;
  let sink: VideoSink | null = null;
  // Muxer output is synchronous while sink writes are not, so writes are
  // chained: a container's bytes are only valid in order.
  let pumping: Promise<void> = Promise.resolve();
  let failure: Error | null = null;
  let frames = 0;
  let keyEvery = 60;

  return {
    async open(spec: VideoSpec, target: VideoSink) {
      if (encoder) throw new Error('This WebCodecs encoder is already open');
      if (!isWebCodecsSupported()) throw new Error('This runtime has no WebCodecs VideoEncoder');
      sink = target;
      // Joining a live stream costs up to one keyframe interval, so a latency
      // critical stream pays the bitrate for a shorter one.
      keyEvery = Math.max(1, Math.round(spec.fps * (spec.latencyCritical ? 1 : 2)));
      frameDurationUs = Math.round(1_000_000 / spec.fps);

      const { Muxer, StreamTarget } = await (options.loadMuxer?.() ?? import('mp4-muxer'));
      // A sink is an append-only byte stream, so it can only honour writes that
      // arrive in order. Fragmented MP4 does emit sequentially, but the muxer
      // is explicit that ignoring `position` can silently corrupt output — so
      // this checks rather than assumes.
      let written = 0;
      muxer = new Muxer({
        target: new StreamTarget({
          onData: (data: Uint8Array, position: number) => {
            if (position !== written) {
              failure ??= new Error(
                `The muxer seeked to byte ${position} after ${written}; a streaming sink cannot rewrite. `
                + 'This should not happen with fastStart: fragmented.',
              );
              return;
            }
            written += data.byteLength;
            // Copy: the muxer reuses its scratch buffer, and a sink that
            // defers the write would otherwise send whatever came next.
            const bytes = new Uint8Array(data);
            pumping = pumping.then(() => sink!.write(bytes)).catch((error: Error) => { failure ??= error; });
          },
        }),
        video: {
          codec: spec.codec === 'vp9' ? 'vp9' : 'avc',
          width: spec.width,
          height: spec.height,
          frameRate: spec.fps,
        },
        // The same choice the ffmpeg encoder makes: playable as it arrives and
        // valid as a file once it stops.
        fastStart: 'fragmented',
      });

      const VideoEncoderCtor = (globalThis as unknown as { VideoEncoder: new (init: unknown) => EncoderLike }).VideoEncoder;
      encoder = new VideoEncoderCtor({
        // Deliberately the raw path. `addVideoChunk` reads `duration` off the
        // chunk, and a native WebCodecs implementation leaves it unset unless
        // the frame carried one — which surfaces as a muxer type error rather
        // than anything about durations. Supplying it from the frame rate is
        // both correct and independent of the implementation.
        output: (chunk: any, meta: unknown) => {
          const data = new Uint8Array(chunk.byteLength);
          chunk.copyTo(data);
          muxer!.addVideoChunkRaw(data, chunk.type, chunk.timestamp, chunk.duration ?? frameDurationUs, meta);
        },
        error: (error: Error) => { failure ??= error; },
      });
      encoder.configure({
        codec: options.codecString ?? (spec.codec === 'vp9' ? 'vp09.00.10.08' : DEFAULT_CODEC_STRING),
        width: spec.width,
        height: spec.height,
        framerate: spec.fps,
        // WebCodecs has no constant-quality mode — bitrate is the only knob.
        // A flat default is wrong at every resolution but one: it produced the
        // same ~1MB for a 3s clip at 720p and at 1440p, quietly starving the
        // larger one. Scaling with pixel rate is the closest thing to CRF.
        bitrate: spec.bitrate ?? Math.round(spec.width * spec.height * spec.fps * 0.07),
        latencyMode: spec.latencyCritical ? 'realtime' : 'quality',
        // MP4 carries length-prefixed NAL units (AVCC). Left unset, an encoder
        // may emit Annex-B start codes instead, and the muxer then reads those
        // start codes as lengths — the stream muxes and appends without
        // complaint, then fails to decode with "Invalid NAL unit size".
        ...(spec.codec === 'vp9' ? {} : { avc: { format: 'avc' } }),
        ...(options.acceleration ? { hardwareAcceleration: options.acceleration } : {}),
      });
    },

    async encode(frame: CapturedFrame) {
      if (!encoder) throw new Error('encode() before open()');
      if (failure) throw failure;
      const VideoFrameCtor = (globalThis as unknown as {
        VideoFrame: new (source: unknown, init: unknown) => { close(): void };
      }).VideoFrame;

      let videoFrame: { close(): void };
      let owned = true;
      if (frame.kind === 'image') {
        // An already-built VideoFrame carries its own timestamp; anything else
        // is a canvas or bitmap that needs one attached. Ownership matters:
        // closing a frame the caller still holds breaks their next encode.
        const source = frame.source as { timestamp?: number; close?: () => void };
        if (typeof source?.timestamp === 'number') {
          videoFrame = source as { close(): void };
          owned = false;
        } else {
          videoFrame = new VideoFrameCtor(frame.source, { timestamp: frame.timestampUs });
        }
      } else {
        // WebCodecs has no bottom-up pixel layout, so a flipped buffer would
        // silently encode upside down. Refuse it instead of shipping that.
        if (frame.flipped) {
          throw new Error('WebCodecs cannot take bottom-up rows; flip the frame first, or use an image source');
        }
        // I420 is exactly what the GPU converter emits, and exactly what the
        // encoder wants — so a yuv420p frame reaches the codec with no
        // conversion anywhere in the pipeline.
        const format = frame.format === 'yuv420p' ? 'I420' : frame.format === 'bgra8' ? 'BGRA' : 'RGBA';
        videoFrame = new VideoFrameCtor(frame.data, {
          format,
          codedWidth: frame.width,
          codedHeight: frame.height,
          timestamp: frame.timestampUs,
          duration: frameDurationUs,
          // Planar YUV carries no colour space of its own, and without this the
          // encoder guesses — it tagged the stream `gbr`, which is a matrix no
          // I420 stream should ever claim. The GPU converter uses BT.709
          // limited, so say so.
          ...(format === 'I420'
            ? { colorSpace: { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false } }
            : {}),
        });
      }

      try {
        encoder.encode(videoFrame, { keyFrame: frames % keyEvery === 0 });
        frames++;
      } finally {
        if (owned) videoFrame.close();
      }

      // The encoder queue is where a too-fast producer piles up; waiting on it
      // is what makes backpressure reach the driver.
      while (encoder.encodeQueueSize > 2) await new Promise((resolve) => setTimeout(resolve, 1));
    },

    async close() {
      if (!encoder) throw new Error('close() before open()');
      await encoder.flush();
      encoder.close();
      muxer!.finalize();
      await pumping;
      if (failure) { await sink!.abort?.(failure); throw failure; }
      await sink!.close();
    },

    async abort(error: Error) {
      try {
        encoder?.close();
      } catch {
        // Already closed or errored; the sink teardown below is what matters.
      }
      await pumping.catch(() => undefined);
      await sink?.abort?.(error);
    },
  };
}
