/**
 * Streams the encoded video to an HTTP endpoint as it is produced.
 *
 * Nothing here knows what a video is. Because the encoder emits fragmented MP4,
 * the receiving end can play the body while it is still uploading or just write
 * it to disk — the sink's only job is to move bytes and to push back when the
 * network is slower than the encoder.
 *
 * Uses a streaming request body, which needs `duplex: 'half'`. Node's fetch
 * supports it; among browsers only Chromium does, and only over HTTP/2 or
 * HTTP/3. A browser without it should buffer and POST once instead.
 */

import type { VideoSink } from './types';

export interface HttpSinkOptions {
  url: string;
  method?: 'POST' | 'PUT';
  headers?: Record<string, string>;
  /** Defaults to video/mp4, matching the encoder's fragmented-MP4 output. */
  contentType?: string;
  /** Rejects `close()` when the endpoint answers with a non-2xx status. */
  expectOk?: boolean;
}

export function createHttpSink(options: HttpSinkOptions): VideoSink {
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
  // Resolved by `pull` when the consumer has room for more; this is what turns
  // a congested upload into a stalled encoder instead of unbounded memory.
  let drained: (() => void) | null = null;
  let response: Promise<Response> | null = null;
  const aborter = new AbortController();

  const start = (): Promise<Response> => {
    if (response) return response;
    const body = new ReadableStream<Uint8Array>(
      {
        start(streamController) {
          controller = streamController;
        },
        pull() {
          const waiter = drained;
          drained = null;
          waiter?.();
        },
        cancel() {
          const waiter = drained;
          drained = null;
          waiter?.();
        },
      },
      { highWaterMark: 1 },
    );
    response = fetch(options.url, {
      method: options.method ?? 'POST',
      headers: {
        'content-type': options.contentType ?? 'video/mp4',
        ...options.headers,
      },
      body,
      signal: aborter.signal,
      // Required for a streaming request body; the type is not in every lib.dom.
      duplex: 'half',
    } as RequestInit);
    return response;
  };

  return {
    async write(bytes) {
      start();
      controller!.enqueue(bytes);
      if ((controller!.desiredSize ?? 1) <= 0) {
        await new Promise<void>((resolve) => { drained = resolve; });
      }
    },
    async close() {
      // A capture that produced nothing should still make the request, so the
      // endpoint sees an empty upload rather than silence.
      const pending = start();
      controller!.close();
      const result = await pending;
      if ((options.expectOk ?? true) && !result.ok) {
        throw new Error(`${options.url} answered ${result.status} ${result.statusText}`);
      }
      // Drain the body so the connection can be reused rather than left open.
      await result.arrayBuffer().catch(() => undefined);
    },
    async abort(error) {
      try {
        controller?.error(error);
      } catch {
        // Already errored or closed; the fetch abort below is what matters.
      }
      aborter.abort();
      await response?.catch(() => undefined);
    },
  };
}
