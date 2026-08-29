/**
 * Basis (.basis) transcoding in Node.
 *
 * Character atlases ship as `.basis` texture arrays — an equipment variant is a
 * layer of one. Babylon transcodes them in a DOM Worker created from a blob
 * URL, pulling the transcoder off a CDN; neither exists here. But
 * `SetBasisTranscoderWorker` lets a host supply its own worker, and Babylon
 * ships the transcoder locally at `@babylonjs/core/assets/Basis`, so the CDN
 * dependency disappears the same way `installNodeDracoDecoder` removes Draco's.
 *
 * The adapter below backs Babylon's worker contract with a real
 * `node:worker_threads` worker: it only has to expose `postMessage` and
 * `addEventListener`, and provide `importScripts` and a global `postMessage`
 * inside the worker so Babylon's own worker body runs unmodified.
 */

import { babylonImport } from './babylon';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

export interface BasisImage {
  width: number;
  height: number;
  data: Uint8Array;
}

let installed: Promise<void> | null = null;

/**
 * Points Babylon at the locally shipped transcoder and installs a worker it can
 * drive. Idempotent; safe to call before every transcode.
 */
export function installHeadlessBasisTranscoder(): Promise<void> {
  installed ??= (async () => {
    const { BasisToolsOptions, SetBasisTranscoderWorker } = await babylonImport('@babylonjs/core/Misc/basis.js');
    const { workerFunction, initializeWebWorker } = await babylonImport('@babylonjs/core/Misc/basisWorker.js');
    const { Worker } = await import('node:worker_threads');

    // The transcoder itself is a self-contained basis_universal build, not part
    // of Babylon's module graph, so shado's own copy is fine even when the host
    // has claimed the runtime through `useBabylonRuntime`.
    const require = createRequire(import.meta.url);
    const assets = dirname(require.resolve('@babylonjs/core/assets/Basis/basis_transcoder.js'));
    const transcoderJs = join(assets, 'basis_transcoder.js');
    BasisToolsOptions.JSModuleURL = pathToFileURL(transcoderJs).href;
    BasisToolsOptions.WasmModuleURL = pathToFileURL(join(assets, 'basis_transcoder.wasm')).href;

    // Babylon's worker body expects a browser worker scope: a global
    // postMessage, importScripts, and an assignable onmessage.
    const bootstrap = `
      const { parentPort } = require('node:worker_threads');
      const fs = require('node:fs');
      globalThis.postMessage = (message, transfer) => parentPort.postMessage(message);
      globalThis.importScripts = (url) => {
        const path = url.startsWith('file:') ? new URL(url).pathname : url;
        const source = fs.readFileSync(path, 'utf8');
        // The transcoder is an emscripten bundle: under CommonJS it assigns to
        // module.exports rather than leaving a BASIS global, which is what the
        // worker body expects. Run it, then bind whichever form it produced.
        // Nothing is appended to the source — a trailing return would need a
        // newline, and escaping one through this nested template is a trap.
        // BASIS is a FUNCTION, so do not test it with Object.keys — that is
        // zero for functions and silently discards the module. Compare against
        // the sentinel instead.
        const sentinel = {};
        const shim = { exports: sentinel };
        new Function('module', 'exports', 'require', source)(shim, shim.exports, require);
        const produced = shim.exports !== sentinel ? shim.exports : globalThis.BASIS;
        if (produced) { globalThis.BASIS = produced; }
      };
      (${String(workerFunction)})();
      parentPort.on('message', (data) => { globalThis.onmessage?.({ data }); });
    `;
    const worker = new Worker(bootstrap, { eval: true });
    worker.unref();

    // Babylon speaks the DOM worker API; node:worker_threads speaks EventEmitter.
    const handlers = new Map<(event: unknown) => void, (data: unknown) => void>();
    const adapter = {
      postMessage: (message: unknown) => worker.postMessage(message),
      addEventListener: (type: string, handler: (event: unknown) => void) => {
        if (type !== 'message') return;
        const bridge = (data: unknown) => handler({ data });
        handlers.set(handler, bridge);
        worker.on('message', bridge);
      },
      removeEventListener: (type: string, handler: (event: unknown) => void) => {
        const bridge = handlers.get(handler);
        if (bridge) { worker.off('message', bridge); handlers.delete(handler); }
      },
      terminate: () => { void worker.terminate(); },
    };

    // Supplying a worker makes CreateWorkerAsync hand it straight back without
    // ever calling initializeWebWorker — so the transcoder script would never
    // be imported and the worker body would throw "BASIS is not defined".
    // Initialise it here instead.
    const wasmBinary = new Uint8Array(await readFile(join(assets, 'basis_transcoder.wasm')));
    await initializeWebWorker(adapter as never, wasmBinary.buffer as never, BasisToolsOptions.JSModuleURL);
    SetBasisTranscoderWorker(adapter as never);
  })();
  return installed;
}

/**
 * Transcodes a `.basis` array texture to uncompressed layers.
 *
 * Uses the CPU path deliberately — asking for a GPU-compressed format ties the
 * result to whatever the adapter happens to support, and a preview does not
 * need it.
 */
export async function transcodeBasisLayers(bytes: Uint8Array): Promise<{
  width: number;
  height: number;
  layers: BasisImage[];
}> {
  await installHeadlessBasisTranscoder();
  const { TranscodeAsync } = await babylonImport('@babylonjs/core/Misc/basis.js');
  const result: any = await TranscodeAsync(bytes as never, {
    supportedCompressionFormats: {},
    loadMipmapLevels: false,
    loadSingleImage: undefined,
  } as never);
  const images = result?.fileInfo?.images ?? [];
  if (!images.length) throw new Error('Basis transcode returned no images; was this built as a 2D array?');
  const layers: BasisImage[] = images.map((image: any) => {
    const level = image.levels[0];
    return { width: level.width, height: level.height, data: new Uint8Array(level.transcodedPixels) };
  });
  return { width: layers[0]!.width, height: layers[0]!.height, layers };
}

/** Convenience: transcode a `.basis` file from disk. */
export async function loadBasisFile(path: string): Promise<{ width: number; height: number; layers: BasisImage[] }> {
  return transcodeBasisLayers(new Uint8Array(await readFile(path)));
}
