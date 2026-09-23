/**
 * Cold zone-entry decode hot paths, old vs new, on the real Eltania assets.
 *
 *   npx tsx bench/decode-hot-paths-bench.ts [--assets <client/public/eqrequiem>] [--runs 5]
 *
 * "old" is the frozen pre-optimisation implementation in tests/oracle; "new" is
 * src/. SVAT chunk decompression is done once up front and replayed from memory
 * so the timings are the decoder's own CPU (checksum, unshuffle, delta,
 * scatter), which is what the browser main thread paid for.
 */
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { decodeSvat as decodeSvatNew, decodeSvatDirectory } from '../src/svat';
import { decodeSvat as decodeSvatOld } from '../tests/oracle/svat/SvatCodec';
import { nodeSvatDecompressor } from '../src/svat/SvatNode';
import { decodeShadoWorldCollision as collisionNew } from '../src/world/collision';
import { decodeShadoWorldCollision as collisionOld } from '../tests/oracle/world/collision';
import { validateShadoWorldPackage as validateNew } from '../src/world/validation';
import { validateShadoWorldPackage as validateOld } from '../tests/oracle/world/validation';

const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1]! : fallback;
};
const assets = flag('--assets', join(import.meta.dirname, '../../client/public/eqrequiem'));
const runs = Number(flag('--runs', '5'));
const zone = flag('--zone', 'talioscrownward');

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1]!;
}

async function time(label: string, fn: () => unknown | Promise<unknown>): Promise<number> {
  await fn(); // warm
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    await fn();
    samples.push(performance.now() - start);
  }
  const value = median(samples);
  console.log(`  ${label.padEnd(34)} ${value.toFixed(1).padStart(8)} ms`);
  return value;
}

async function svat(): Promise<void> {
  const dir = join(assets, 'vat');
  const files = readdirSync(dir).filter(file => file.endsWith('.svat'));
  const inflate = nodeSvatDecompressor();
  const prepared: Array<{ bytes: Uint8Array; replay: Map<number, Uint8Array> }> = [];
  let decodedBytes = 0;
  for (const file of files) {
    const bytes = new Uint8Array(readFileSync(join(dir, file)));
    const directory = decodeSvatDirectory(bytes);
    const replay = new Map<number, Uint8Array>();
    for (const chunk of directory.chunks) {
      const start = directory.payloadOffset + chunk.compressedOffset;
      const out = await inflate(bytes.subarray(start, start + chunk.compressedBytes), chunk.decodedBytes);
      replay.set(bytes.byteOffset + start, out);
    }
    decodedBytes += directory.decodedByteLength;
    prepared.push({ bytes, replay });
  }
  console.log(`SVAT: ${files.length} containers, ${(decodedBytes / 1e6).toFixed(1)} MB decoded atlas`);
  const run = (decode: typeof decodeSvatNew) => async () => {
    for (const { bytes, replay } of prepared) {
      await decode(bytes, { decompress: compressed => replay.get(compressed.byteOffset)! });
    }
  };
  const before = await time('decodeSvat old (all)', run(decodeSvatOld));
  const after = await time('decodeSvat new (all)', run(decodeSvatNew));
  console.log(`  speedup ${(before / after).toFixed(1)}x`);
}

async function world(): Promise<void> {
  const dir = join(assets, 'worlds');
  const spatialPath = join(dir, `${zone}.spatial.json.gz`);
  const collisionPath = join(dir, `${zone}.collision.bin.gz`);
  if (!existsSync(spatialPath) || !existsSync(collisionPath)) {
    console.log(`world: ${zone} assets missing, skipped`);
    return;
  }
  const worldJson = JSON.parse(gunzipSync(readFileSync(spatialPath)).toString('utf8'));
  const collisionBytes = new Uint8Array(gunzipSync(readFileSync(collisionPath)));
  console.log(`World ${zone}: collision ${(collisionBytes.byteLength / 1e6).toFixed(1)} MB`);
  const c0 = await time('decodeShadoWorldCollision old', () => collisionOld(collisionBytes, worldJson.collision));
  const c1 = await time('decodeShadoWorldCollision new', () => collisionNew(collisionBytes, worldJson.collision));
  console.log(`  speedup ${(c0 / c1).toFixed(1)}x`);
  const v0 = await time('validateShadoWorldPackage old', () => validateOld(worldJson));
  const v1 = await time('validateShadoWorldPackage new', () => validateNew(worldJson));
  console.log(`  speedup ${(v0 / v1).toFixed(1)}x`);
}

await svat();
await world();
