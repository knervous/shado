/**
 * Streams the encoded video over an existing WebRTC data channel.
 *
 * Deliberately not a media track. A media track would mean SDP renegotiation
 * and the browser's own encoder and pacing, and this project's peer connections
 * negotiate exactly once through a read-once mailbox — there is no channel to
 * renegotiate over. Sending fragmented MP4 down a data channel reuses the
 * connection that already exists and adds no signalling at all.
 *
 * The channel must be **pre-negotiated** (`negotiated: true` with a fixed `id`,
 * or created before the offer). Creating one after the connection is up fires
 * `negotiationneeded`, which is exactly what this avoids.
 *
 * Reliable and ordered is required: a container cannot survive a lost or
 * reordered fragment. That means a stalled receiver applies backpressure rather
 * than dropping, which is the right trade for a video stream that must decode.
 */

import type { VideoSink } from './types';

/** The slice of RTCDataChannel used here; avoids depending on DOM lib types. */
export interface DataChannelLike {
  readonly readyState: string;
  readonly ordered?: boolean;
  bufferedAmount: number;
  bufferedAmountLowThreshold: number;
  binaryType: string;
  send(data: Uint8Array): void;
  close(): void;
  addEventListener(type: string, handler: (event: unknown) => void): void;
  removeEventListener(type: string, handler: (event: unknown) => void): void;
}

export interface DataChannelSinkOptions {
  /**
   * Bytes per message. 16 KiB is the size every WebRTC implementation agrees
   * on; larger messages work between browsers but fail against some native
   * stacks, and the failure looks like a silently dead channel.
   */
  chunkSize?: number;
  /** Bytes allowed in the send queue before `write` waits. */
  highWaterMark?: number;
  /** Close the channel when the stream ends. Off by default — it is shared. */
  closeChannel?: boolean;
}

const DEFAULT_CHUNK_SIZE = 16 * 1024;
const DEFAULT_HIGH_WATER_MARK = 1024 * 1024;

export function createDataChannelSink(
  channel: DataChannelLike,
  options: DataChannelSinkOptions = {},
): VideoSink {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const highWaterMark = options.highWaterMark ?? DEFAULT_HIGH_WATER_MARK;
  if (channel.ordered === false) {
    throw new Error('A video data channel must be ordered; a reordered fragment cannot be decoded');
  }
  channel.binaryType = 'arraybuffer';
  channel.bufferedAmountLowThreshold = Math.floor(highWaterMark / 2);

  const drain = async (): Promise<void> => {
    if (channel.bufferedAmount <= highWaterMark) return;
    // Event-driven rather than polled: the channel tells us when it has room,
    // which is the one place WebRTC gives real backpressure.
    await new Promise<void>((resolve) => {
      const onLow = (): void => { channel.removeEventListener('bufferedamountlow', onLow); resolve(); };
      channel.addEventListener('bufferedamountlow', onLow);
    });
  };

  return {
    async write(bytes) {
      for (let offset = 0; offset < bytes.byteLength; offset += chunkSize) {
        if (channel.readyState !== 'open') throw new Error(`Data channel is ${channel.readyState}`);
        await drain();
        channel.send(bytes.subarray(offset, Math.min(offset + chunkSize, bytes.byteLength)));
      }
    },
    async close() {
      while (channel.readyState === 'open' && channel.bufferedAmount > 0) {
        await new Promise((resolve) => setTimeout(resolve, 4));
      }
      if (options.closeChannel) channel.close();
    },
    abort() {
      if (options.closeChannel) channel.close();
    },
  };
}
