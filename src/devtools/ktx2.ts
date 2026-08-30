/**
 * KTX2 transcoding in Node, with no CDN and no Worker.
 *
 * Promoted objects reference their textures out of a shared, supercompressed
 * KTX2 store instead of embedding them. Babylon ships the KTX2 *loader* but not
 * the decoder: `KhronosTextureContainer2` points `jsDecoderModule` at
 * cdn.babylonjs.com and leaves every wasm URL null, and each transcoder falls
 * back to its own CDN constant. In a headless render none of that is reachable,
 * so a KTX2 texture simply never resolves and the preview stalls waiting for a
 * scene that will never be ready.
 *
 * Everything needed is already on disk in `@babylonjs/ktx2decoder`. This wires
 * it up the same way `basis.ts` does for `.basis`:
 *
 *   - the decoder module is handed over as the `KTX2DECODER` global, which
 *     `KhronosTextureContainer2` prefers over fetching a script;
 *   - workers are disabled, because the worker path builds a blob URL and calls
 *     `importScripts`, neither of which exists here;
 *   - each `LiteTranscoder_*` and the MSC transcoder take a pre-supplied
 *     `WasmBinary` rather than fetching `WasmModuleURL`;
 *   - `MSCTranscoder` additionally needs its emscripten glue as a global, since
 *     its fallback path appends a `<script>` element;
 *   - `ZSTDDecoder` is the one with no binary hook — it always fetches — so it
 *     gets a `data:` URL, which Node's fetch does support. UASTC payloads are
 *     Zstandard-supercompressed by default, so this is not optional.
 */
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { babylonImport } from './babylon';

let installed: Promise<void> | null = null;

/** Idempotent; safe to call before every load. */
export function installHeadlessKtx2Transcoder(): Promise<void> {
  installed ??= install();
  return installed;
}

async function install(): Promise<void> {
  const require = createRequire(import.meta.url);
  const root = dirname(require.resolve('@babylonjs/ktx2decoder/package.json'));
  const wasmDir = join(root, 'wasm');
  const binary = async (name: string): Promise<Uint8Array> =>
    new Uint8Array(await readFile(join(wasmDir, name)));

  const decoder = (await import('@babylonjs/ktx2decoder')) as Record<string, any>;
  (globalThis as Record<string, any>).KTX2DECODER = decoder;

  const { KhronosTextureContainer2 } = await babylonImport(
    '@babylonjs/core/Misc/khronosTextureContainer2.js'
  );
  // The worker path builds a blob URL and calls importScripts. Decode inline.
  KhronosTextureContainer2.DefaultNumWorkers = 0;

  const lite: Array<[string, string]> = [
    ['LiteTranscoder_UASTC_ASTC', 'uastc_astc.wasm'],
    ['LiteTranscoder_UASTC_BC7', 'uastc_bc7.wasm'],
    ['LiteTranscoder_UASTC_R8_UNORM', 'uastc_r8_unorm.wasm'],
    ['LiteTranscoder_UASTC_RG8_UNORM', 'uastc_rg8_unorm.wasm'],
    ['LiteTranscoder_UASTC_RGBA_SRGB', 'uastc_rgba8_srgb_v2.wasm'],
    ['LiteTranscoder_UASTC_RGBA_UNORM', 'uastc_rgba8_unorm_v2.wasm'],
  ];
  for (const [exportName, file] of lite) {
    const transcoder = decoder[exportName];
    if (transcoder) transcoder.WasmBinary = await binary(file);
  }

  if (decoder.MSCTranscoder) {
    decoder.MSCTranscoder.WasmBinary = await binary('msc_basis_transcoder.wasm');
    decoder.MSCTranscoder.UseFromPixelsFallback = false;
    // The glue is a classic script declaring `var MSC_TRANSCODER = ...`.
    // Evaluating it and publishing the global short-circuits the <script> path.
    if (typeof (globalThis as Record<string, any>).MSC_TRANSCODER === 'undefined') {
      const glue = await readFile(join(wasmDir, 'msc_basis_transcoder.js'), 'utf8');
      // The glue is emscripten output with a Node branch that reads
      // `__dirname`. `new Function` bodies run in global scope, where the CJS
      // module wrapper's variables do not exist, so they are passed in
      // explicitly — otherwise the transcode dies with
      // "ReferenceError: __dirname is not defined" and the texture silently
      // falls back.
      const factory = new Function(
        '__dirname',
        '__filename',
        `${glue}\nreturn MSC_TRANSCODER;`
      )(wasmDir, join(wasmDir, 'msc_basis_transcoder.js'));
      (globalThis as Record<string, any>).MSC_TRANSCODER = factory;
      decoder.MSCTranscoder.JSModule = factory;
    }
  }

  if (decoder.ZSTDDecoder) {
    const zstd = await binary('zstddec.wasm');
    decoder.ZSTDDecoder.WasmModuleURL =
      `data:application/wasm;base64,${Buffer.from(zstd).toString('base64')}`;
  }
}
