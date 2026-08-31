/**
 * The receiving half: fragmented MP4 into a `<video>` element via Media Source
 * Extensions.
 *
 * This is not a sink — it is what the far end of one does. A sink moves bytes;
 * this turns them back into playback. It exists here because the format
 * contract is shared: the encoders emit fMP4 precisely so that a `SourceBuffer`
 * will accept the stream as it arrives, and pairing them in one package is what
 * keeps that contract from drifting.
 *
 * `appendBuffer` is asynchronous and rejects a second call while one is in
 * flight, so appends are queued. Live streams also grow without bound, so old
 * buffered media is evicted rather than left to hit the browser's quota.
 */

/** The slice of SourceBuffer used here; avoids depending on DOM lib types. */
interface SourceBufferLike {
  updating: boolean;
  appendBuffer(data: Uint8Array): void;
  remove(start: number, end: number): void;
  addEventListener(type: string, handler: () => void): void;
  buffered: { length: number; start(index: number): number; end(index: number): number };
}

interface MediaSourceLike {
  readyState: string;
  addSourceBuffer(mimeType: string): SourceBufferLike;
  endOfStream(): void;
  addEventListener(type: string, handler: () => void): void;
}

export interface MseAppenderOptions {
  /**
   * Full MIME type including the codec. Must match what the encoder produced;
   * the default is the H.264 baseline profile both encoders default to.
   */
  mimeType?: string;
  /**
   * Seconds of played-out media to keep. Older media is evicted as the stream
   * runs, so an open-ended broadcast does not eventually hit the buffer quota.
   * Set to 0 to keep everything, which suits a short recording.
   */
  keepSeconds?: number;
  /**
   * Jump the element forward when it falls behind the live edge by more than
   * this many seconds. Set to 0 to never seek.
   */
  maxDriftSeconds?: number;
}

export interface MseAppender {
  /** Queues bytes for playback. Safe to call as fast as they arrive. */
  append(bytes: Uint8Array): void;
  /** Signals that no more bytes are coming. */
  end(): void;
  /** How far the buffered media runs ahead of the playhead, in seconds. */
  drift(): number;
}

const DEFAULT_MIME = 'video/mp4; codecs="avc1.42E02A"';

export function isMseSupported(mimeType = DEFAULT_MIME): boolean {
  const MediaSourceCtor = (globalThis as { MediaSource?: { isTypeSupported(t: string): boolean } }).MediaSource;
  return Boolean(MediaSourceCtor?.isTypeSupported(mimeType));
}

/**
 * Attaches a `MediaSource` to a video element and returns an appender for it.
 *
 * `element` is anything with `src` and `currentTime` — a real `HTMLVideoElement`
 * in practice.
 */
export async function createMseAppender(
  element: { src: string; currentTime: number; play?: () => Promise<void> },
  options: MseAppenderOptions = {},
): Promise<MseAppender> {
  const mimeType = options.mimeType ?? DEFAULT_MIME;
  const keepSeconds = options.keepSeconds ?? 30;
  const maxDrift = options.maxDriftSeconds ?? 4;
  const MediaSourceCtor = (globalThis as unknown as { MediaSource?: new () => MediaSourceLike }).MediaSource;
  if (!MediaSourceCtor) throw new Error('This runtime has no MediaSource');
  if (!isMseSupported(mimeType)) throw new Error(`MediaSource cannot play ${mimeType}`);

  const mediaSource = new MediaSourceCtor();
  const URLCtor = (globalThis as unknown as { URL: { createObjectURL(o: unknown): string } }).URL;
  element.src = URLCtor.createObjectURL(mediaSource);
  await new Promise<void>((resolve) => mediaSource.addEventListener('sourceopen', () => resolve()));

  const sourceBuffer = mediaSource.addSourceBuffer(mimeType);
  const queue: Uint8Array[] = [];
  let ended = false;
  let started = false;

  const bufferedEnd = (): number =>
    sourceBuffer.buffered.length ? sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1) : 0;

  const pump = (): void => {
    if (sourceBuffer.updating) return;
    // Eviction has to happen between appends, never during one.
    if (keepSeconds > 0 && sourceBuffer.buffered.length) {
      const dropBefore = element.currentTime - keepSeconds;
      if (dropBefore > sourceBuffer.buffered.start(0)) {
        sourceBuffer.remove(0, dropBefore);
        return;
      }
    }
    const next = queue.shift();
    if (next) {
      sourceBuffer.appendBuffer(next);
      return;
    }
    if (ended && mediaSource.readyState === 'open') mediaSource.endOfStream();
  };

  sourceBuffer.addEventListener('updateend', () => {
    // A live stream that buffers faster than it plays drifts behind the edge
    // forever; seeking is the only way back and is imperceptible at this size.
    if (maxDrift > 0 && bufferedEnd() - element.currentTime > maxDrift) {
      element.currentTime = bufferedEnd() - maxDrift / 2;
    }
    pump();
  });

  return {
    append(bytes) {
      queue.push(bytes);
      pump();
      if (!started) {
        started = true;
        void element.play?.().catch(() => undefined);
      }
    },
    end() {
      ended = true;
      pump();
    },
    drift() {
      return bufferedEnd() - element.currentTime;
    },
  };
}
