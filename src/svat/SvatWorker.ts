/**
 * `.svat` decode off the main thread.
 *
 * `decodeSvat` is fine for tools and tests but, on a cold zone entry, a dozen
 * character bodies put ~100 MB of checksum + unshuffle + delta + scatter on the
 * main thread. `decodeSvatInWorker` does the same decode in a small pool of
 * dedicated workers and hands the atlas back as a transferred ArrayBuffer, so
 * the main thread pays only for parsing the (tiny) directory and one memcpy of
 * the compressed container.
 *
 * The worker runs the exact kernel the main-thread decoder runs: the hot
 * functions in `SvatKernel` are spliced into the worker script with
 * `Function.prototype.toString()`, not restated.
 *
 * Fallbacks, so adopting this can never lose a character that `decodeSvat`
 * would have loaded:
 *  - no `Worker` (Node, some embedders), or a worker that fails to start (a CSP
 *    that forbids `blob:` workers): decode on the main thread from then on;
 *  - a chunk codec the worker's `DecompressionStream` lacks (Zstd on today's
 *    browsers): that container is decoded on the main thread, with
 *    `zstdFallback` if one was given.
 * Genuine corruption (checksum or size mismatch) rejects, exactly as
 * `decodeSvat` does.
 */

import type { DQClipInfo, PackedDQVAT } from '../extensions/VATBuilder/VATBuilder';
import { decodeSvat, type SvatDecompress } from './SvatCodec';
import {
  decodeSvatDirectory,
  SvatCodec,
  SvatFilter,
  svatComponentBytes,
  type SvatDirectory,
} from './SvatFormat';
import { svatDecodeChunkInto, svatFnv1a32 } from './SvatKernel';
import { createSvatDecompressor, supportsDecompressionFormat } from './SvatRuntime';

export type SvatWorkerDecodeOptions = {
  /** Verify each chunk's FNV-1a checksum. Defaults to true, as in `decodeSvat`. */
  verifyChecksums?: boolean;
  /** Zstd decoder for the main-thread fallback. See `createSvatDecompressor`. */
  zstdFallback?: SvatDecompress;
};

/** The subset of `Worker` the pool uses, so a test or an embedder can supply one. */
export type SvatDecodeWorkerLike = {
  postMessage(message: unknown, transfer: Transferable[]): void;
  terminate(): void;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: ((event: unknown) => void) | null;
};

export type SvatDecodeWorkerConfig = {
  /** Workers kept at most. Defaults to min(2, hardwareConcurrency - 1), at least 1. */
  maxWorkers?: number;
  /** Idle workers are terminated after this long. Defaults to 10 s. */
  idleMillis?: number;
  /**
   * Builds a worker from the script text. Defaults to a `blob:` URL dedicated
   * worker. Pass `null` to restore the default.
   */
  workerFactory?: ((source: string) => SvatDecodeWorkerLike) | null;
};

type WorkerChunk = {
  compressedOffset: number;
  compressedBytes: number;
  decodedBytes: number;
  checksum: number;
  filter: number;
  frameStart: number;
  frameCount: number;
};

type WorkerRequest = {
  id: number;
  buffer: ArrayBuffer;
  byteOffset: number;
  byteLength: number;
  payloadOffset: number;
  verify: boolean;
  layout: SvatDirectory['layout'];
  chunks: WorkerChunk[];
};

type WorkerReply =
  | { id: number; buffer: ArrayBuffer }
  | { id: number; error: string; unsupported: boolean };

/** Rejection that means "this worker cannot do it", not "the data is bad". */
class SvatWorkerUnavailable extends Error {}

/**
 * The worker script. Exported so an embedder whose CSP forbids `blob:` workers
 * can serve it from a static URL and pass a `workerFactory`.
 */
export function svatDecodeWorkerSource(): string {
  return `'use strict';
const svatFnv1a32 = ${svatFnv1a32.toString()};
const svatDecodeChunkInto = ${svatDecodeChunkInto.toString()};
const DELTA_XOR_SHUFFLE = ${SvatFilter.DeltaXorShuffle};

function unsupported(message) {
  const error = new Error(message);
  error.svatUnsupported = true;
  return error;
}
function canDecompress(format) {
  if (typeof DecompressionStream === 'undefined') return false;
  try { new DecompressionStream(format); return true; } catch (error) { return false; }
}
async function streamDecompress(bytes, format) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(format));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
function inflate(bytes) {
  if (bytes[0] === 0x28 && bytes[1] === 0xb5 && bytes[2] === 0x2f && bytes[3] === 0xfd) {
    if (!canDecompress('zstd')) throw unsupported('This worker cannot decode Zstd .svat chunks');
    return streamDecompress(bytes, 'zstd');
  }
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (!canDecompress('gzip')) throw unsupported('This worker cannot decode gzip .svat chunks');
    return streamDecompress(bytes, 'gzip');
  }
  return bytes;
}
async function decode(request) {
  const source = new Uint8Array(request.buffer, request.byteOffset, request.byteLength);
  const layout = request.layout;
  const elementBytes = layout.componentType === 'float16' ? 2 : 4;
  const componentCount = layout.widthTexels * layout.heightTexels * 4;
  const buffer = new ArrayBuffer(componentCount * elementBytes);
  const words = elementBytes === 2 ? new Uint16Array(buffer) : new Uint32Array(buffer);
  const chunks = request.chunks;
  // Probe codecs synchronously first so an unsupported one fails before any work.
  const pending = [];
  for (let i = 0; i < chunks.length; i++) {
    const start = request.payloadOffset + chunks[i].compressedOffset;
    pending.push(inflate(source.subarray(start, start + chunks[i].compressedBytes)));
  }
  const decoded = await Promise.all(pending);
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const bytes = decoded[i];
    if (bytes.byteLength !== chunk.decodedBytes) {
      throw new Error('.svat chunk decoded to ' + bytes.byteLength + ' bytes, expected ' + chunk.decodedBytes);
    }
    if (request.verify) {
      const actual = svatFnv1a32(bytes);
      if (actual !== chunk.checksum) {
        throw new Error('.svat chunk checksum mismatch: expected ' + chunk.checksum + ', got ' + actual);
      }
    }
    const expected = layout.bones * layout.strideTexels * 4 * chunk.frameCount;
    if (bytes.byteLength !== expected * elementBytes) {
      throw new Error('.svat chunk holds ' + ((bytes.byteLength / elementBytes) | 0) + ' components, expected ' + expected);
    }
    const filtered = chunk.filter === DELTA_XOR_SHUFFLE;
    svatDecodeChunkInto(words, bytes, elementBytes, filtered, filtered, layout, chunk.frameStart, chunk.frameCount);
  }
  return buffer;
}
self.onmessage = async function (event) {
  const request = event.data;
  try {
    const buffer = await decode(request);
    self.postMessage({ id: request.id, buffer: buffer }, [buffer]);
  } catch (error) {
    self.postMessage({
      id: request.id,
      error: String((error && error.message) || error),
      unsupported: Boolean(error && error.svatUnsupported),
    });
  }
};
`;
}

type Pending = {
  resolve: (buffer: ArrayBuffer) => void;
  reject: (error: Error) => void;
};

type Slot = {
  worker: SvatDecodeWorkerLike;
  pending: Map<number, Pending>;
  idleTimer: ReturnType<typeof setTimeout> | null;
};

const config: {
  maxWorkers: number;
  idleMillis: number;
  workerFactory: ((source: string) => SvatDecodeWorkerLike) | null;
} = {
  maxWorkers: defaultMaxWorkers(),
  idleMillis: 10_000,
  workerFactory: null,
};
let slots: Slot[] = [];
let workersUnavailable = false;
let nextRequestId = 1;
let cachedSource: string | null = null;

function defaultMaxWorkers(): number {
  const cores =
    typeof navigator !== 'undefined' && navigator && Number.isFinite(navigator.hardwareConcurrency)
      ? navigator.hardwareConcurrency
      : 4;
  return Math.max(1, Math.min(2, cores - 1));
}

function defaultWorkerFactory(source: string): SvatDecodeWorkerLike {
  if (typeof Worker === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined') {
    throw new SvatWorkerUnavailable('Dedicated workers are unavailable in this environment');
  }
  const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  try {
    return new Worker(url, { name: 'shado-svat-decode' }) as unknown as SvatDecodeWorkerLike;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Tune or replace the decode worker pool. Existing workers keep running. */
export function configureSvatDecodeWorkers(options: SvatDecodeWorkerConfig): void {
  if (options.maxWorkers !== undefined) config.maxWorkers = Math.max(1, Math.floor(options.maxWorkers));
  if (options.idleMillis !== undefined) config.idleMillis = Math.max(0, options.idleMillis);
  if (options.workerFactory !== undefined) {
    config.workerFactory = options.workerFactory;
    workersUnavailable = false;
  }
}

/** Terminate every decode worker. In-flight decodes fall back to the main thread. */
export function disposeSvatDecodeWorkers(): void {
  const current = slots;
  slots = [];
  for (const slot of current) retire(slot, new SvatWorkerUnavailable('.svat decode worker disposed'));
}

function retire(slot: Slot, reason: Error): void {
  if (slot.idleTimer) clearTimeout(slot.idleTimer);
  slot.idleTimer = null;
  slots = slots.filter(candidate => candidate !== slot);
  try {
    slot.worker.terminate();
  } catch {
    // Already gone.
  }
  const pending = [...slot.pending.values()];
  slot.pending.clear();
  for (const entry of pending) entry.reject(reason);
}

function spawn(): Slot {
  cachedSource ??= svatDecodeWorkerSource();
  const factory = config.workerFactory ?? defaultWorkerFactory;
  let worker: SvatDecodeWorkerLike;
  try {
    worker = factory(cachedSource);
  } catch (error) {
    // No Worker, or one the page may not create (CSP): main thread from now on.
    workersUnavailable = true;
    throw new SvatWorkerUnavailable(String((error as Error)?.message ?? error));
  }
  const slot: Slot = { worker, pending: new Map(), idleTimer: null };
  worker.onmessage = event => {
    const reply = event.data as WorkerReply;
    const entry = slot.pending.get(reply.id);
    if (!entry) return;
    slot.pending.delete(reply.id);
    if ('buffer' in reply) entry.resolve(reply.buffer);
    else if (reply.unsupported) entry.reject(new SvatWorkerUnavailable(reply.error));
    else entry.reject(new Error(reply.error));
    scheduleIdle(slot);
  };
  worker.onerror = () => {
    // A worker that dies (or never loads) is an environment problem: stop
    // trying and let every waiting decode run on the main thread.
    workersUnavailable = true;
    retire(slot, new SvatWorkerUnavailable('.svat decode worker failed'));
  };
  slots.push(slot);
  return slot;
}

function scheduleIdle(slot: Slot): void {
  if (slot.pending.size > 0 || slot.idleTimer) return;
  slot.idleTimer = setTimeout(() => {
    slot.idleTimer = null;
    if (slot.pending.size === 0) retire(slot, new SvatWorkerUnavailable('idle'));
  }, config.idleMillis);
  // Never keep a Node process alive for an idle pool.
  (slot.idleTimer as { unref?: () => void }).unref?.();
}

function acquire(): Slot {
  let best: Slot | null = null;
  for (const slot of slots) if (!best || slot.pending.size < best.pending.size) best = slot;
  if (best && (best.pending.size === 0 || slots.length >= config.maxWorkers)) return best;
  return spawn();
}

function runInWorker(request: Omit<WorkerRequest, 'id'>, transfer: Transferable[]): Promise<ArrayBuffer> {
  const slot = acquire();
  if (slot.idleTimer) {
    clearTimeout(slot.idleTimer);
    slot.idleTimer = null;
  }
  const id = nextRequestId++;
  return new Promise<ArrayBuffer>((resolve, reject) => {
    slot.pending.set(id, { resolve, reject });
    try {
      slot.worker.postMessage({ ...request, id }, transfer);
    } catch (error) {
      slot.pending.delete(id);
      reject(new SvatWorkerUnavailable(String((error as Error)?.message ?? error)));
      scheduleIdle(slot);
    }
  });
}

function packedFromDirectory(directory: SvatDirectory, buffer: ArrayBuffer): PackedDQVAT {
  const { layout } = directory;
  const clips: DQClipInfo[] = directory.clips.map(clip => ({
    name: clip.name,
    from: clip.from,
    to: clip.to,
    frames: clip.frames,
    fps: clip.fps,
  }));
  return {
    componentType: layout.componentType,
    widthTexels: layout.widthTexels,
    heightTexels: layout.heightTexels,
    framesTotal: layout.framesTotal,
    bones: layout.bones,
    dqWidthBones: layout.widthBones,
    dqTilesX: layout.tilesX,
    dqFramesX: layout.framesX,
    dqStrideTexels: layout.strideTexels,
    dqHasScale: layout.hasScale,
    clips,
    pixels: layout.componentType === 'float16' ? new Uint16Array(buffer) : new Float32Array(buffer),
  };
}

function decodeOnMainThread(
  source: Uint8Array,
  options: SvatWorkerDecodeOptions
): Promise<PackedDQVAT> {
  return decodeSvat(source, {
    decompress: createSvatDecompressor({ zstdFallback: options.zstdFallback }),
    verifyChecksums: options.verifyChecksums,
  });
}

/**
 * Decode a `.svat` container in a worker. Same result as
 * `decodeSvat(source, { decompress: createSvatDecompressor(...) })`, bit for
 * bit, without the main-thread cost.
 */
export async function decodeSvatInWorker(
  source: ArrayBuffer | Uint8Array,
  options: SvatWorkerDecodeOptions = {}
): Promise<PackedDQVAT> {
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
  // Parse on this thread: it is tiny, and it keeps header errors synchronous
  // with the call and identical to decodeSvat's.
  const directory = decodeSvatDirectory(bytes);
  const chunks: WorkerChunk[] = directory.chunks.map(chunk => {
    const clip = directory.clips[chunk.clipIndex];
    if (!clip) throw new Error(`.svat chunk references missing clip ${chunk.clipIndex}`);
    return {
      compressedOffset: chunk.compressedOffset,
      compressedBytes: chunk.compressedBytes,
      decodedBytes: chunk.decodedBytes,
      checksum: chunk.checksum,
      filter: chunk.filter,
      frameStart: clip.firstFrame + chunk.firstFrame,
      frameCount: chunk.frameCount,
    };
  });
  // Validate the component type up front so the worker and fallback agree.
  svatComponentBytes(directory.layout.componentType);

  if (workersUnavailable) return decodeOnMainThread(bytes, options);
  // A Zstd container on a platform without DecompressionStream('zstd') can
  // only be decoded here, through the caller's zstdFallback; skip the trip.
  if (
    directory.chunks.some(chunk => chunk.codec === SvatCodec.Zstd) &&
    !supportsDecompressionFormat('zstd')
  ) {
    return decodeOnMainThread(bytes, options);
  }

  // The worker gets its own copy (one memcpy of the compressed container), so
  // the caller's bytes stay usable and a fallback can still decode them.
  const buffer = bytes.slice().buffer;
  try {
    const pixels = await runInWorker(
      {
        buffer,
        byteOffset: 0,
        byteLength: bytes.byteLength,
        payloadOffset: directory.payloadOffset,
        verify: options.verifyChecksums ?? true,
        layout: directory.layout,
        chunks,
      },
      [buffer]
    );
    return packedFromDirectory(directory, pixels);
  } catch (error) {
    if (!(error instanceof SvatWorkerUnavailable)) throw error;
    return decodeOnMainThread(bytes, options);
  }
}
