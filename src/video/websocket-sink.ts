/**
 * Streams the encoded video over a WebSocket.
 *
 * The natural pairing with {@link createMseAppender} on the far end: fragmented
 * MP4 arrives as binary messages and goes straight into a `SourceBuffer`, which
 * is the shortest path from a running renderer to a `<video>` element.
 *
 * Works in both runtimes — Node has had a global `WebSocket` client since 22.
 */

import type { VideoSink } from './types';

/** The slice of WebSocket used here; avoids depending on DOM lib types. */
export interface WebSocketLike {
  readonly readyState: number;
  bufferedAmount: number;
  binaryType: string;
  send(data: Uint8Array): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: string, handler: (event: unknown) => void): void;
}

export interface WebSocketSinkOptions {
  /**
   * How many bytes may sit unsent before `write` starts waiting.
   *
   * A WebSocket accepts everything you hand it and buffers the rest in memory,
   * so without a ceiling a congested link turns into unbounded growth rather
   * than backpressure. There is no drain event, hence the poll.
   */
  highWaterMark?: number;
  /** How often to re-check `bufferedAmount` while waiting. */
  pollMs?: number;
  /** Close the socket when the stream ends. On by default. */
  closeSocket?: boolean;
}

const OPEN = 1;
const DEFAULT_HIGH_WATER_MARK = 4 * 1024 * 1024;

/** Resolves once the socket is open, or rejects if it never gets there. */
async function waitForOpen(socket: WebSocketLike): Promise<void> {
  if (socket.readyState === OPEN) return;
  await new Promise<void>((resolve, reject) => {
    socket.addEventListener('open', () => resolve());
    socket.addEventListener('error', () => reject(new Error('WebSocket failed before it opened')));
    socket.addEventListener('close', () => reject(new Error('WebSocket closed before it opened')));
  });
}

export function createWebSocketSink(socket: WebSocketLike, options: WebSocketSinkOptions = {}): VideoSink {
  const highWaterMark = options.highWaterMark ?? DEFAULT_HIGH_WATER_MARK;
  const pollMs = options.pollMs ?? 4;
  socket.binaryType = 'arraybuffer';
  let opened: Promise<void> | null = null;

  return {
    async write(bytes) {
      opened ??= waitForOpen(socket);
      await opened;
      while (socket.bufferedAmount > highWaterMark) {
        if (socket.readyState !== OPEN) throw new Error('WebSocket closed mid-stream');
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
      if (socket.readyState !== OPEN) throw new Error('WebSocket closed mid-stream');
      socket.send(bytes);
    },
    async close() {
      // Let the queued fragments actually leave before hanging up, or the
      // receiver's last fragment never arrives and playback stops short.
      while (socket.readyState === OPEN && socket.bufferedAmount > 0) {
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
      if (options.closeSocket !== false) socket.close(1000, 'stream complete');
    },
    abort(error) {
      // 1011 rather than 1000: the receiver should know the stream is truncated
      // and not treat a partial file as the whole thing.
      if (options.closeSocket !== false) socket.close(1011, error.message.slice(0, 120));
    },
  };
}
