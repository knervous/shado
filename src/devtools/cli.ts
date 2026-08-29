/**
 * shado preview — headless renders for quick iteration and pipeline proof.
 *
 *   tsx src/devtools/cli.ts --input raw.glb --input baked.glb.gz --out ./previews
 *   tsx src/devtools/cli.ts --input world.glb.gz --field lighting.json
 *
 * Every input is rendered from the same camera, so a raw asset, a bake's
 * output and the finalized runtime artifact line up pixel for pixel. Two
 * inputs also produce an amplified difference image.
 */

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { createHeadlessPreview, differenceImage, type PreviewImage } from './preview';
import { MULTIVIEWS } from './views';
import { encodePng } from './png';

const argv = process.argv.slice(2);
const flag = (name: string, fallback?: string): string | undefined => {
  const index = argv.indexOf(`--${name}`);
  return index >= 0 && argv[index + 1] ? argv[index + 1] : fallback;
};
const repeated = (name: string): string[] =>
  argv.reduce<string[]>((all, value, index) => (value === `--${name}` && argv[index + 1] ? [...all, argv[index + 1]!] : all), []);

async function loadGlb(path: string): Promise<Uint8Array> {
  const raw = await readFile(path);
  return path.endsWith('.gz') ? new Uint8Array(gunzipSync(raw)) : new Uint8Array(raw);
}

async function main(): Promise<void> {
  const inputs = repeated('input');
  if (!inputs.length) {
    process.stderr.write(
      'usage: --input <glb|glb.gz> [--input ...] [--field <lighting.json>] [--out <dir>]\n'
      + '       [--views front,iso,...] [--raised] [--materials] [--width N] [--height N]\n'
      + `       views: ${MULTIVIEWS.join(', ')}, iso, isoFrontLeft, isoFrontRight, isoBackLeft, isoBackRight, eyeFront, eyeCorner, top\n`,
    );
    process.exitCode = 2;
    return;
  }
  const outDir = flag('out', './shado-previews')!;
  const width = Number(flag('width', '1024'));
  const height = Number(flag('height', '768'));
  const camera = {
    alpha: Number(flag('alpha', String(Math.PI / 4))),
    beta: Number(flag('beta', String(Math.PI / 3))),
    zoom: Number(flag('zoom', '2.4')),
  };
  const materials = argv.includes('--materials');
  const raised = argv.includes('--raised');
  const views = flag('views')?.split(',').map((name) => name.trim()).filter(Boolean);
  const fieldPath = flag('field');
  const vertexColors = fieldPath
    ? (JSON.parse(await readFile(fieldPath, 'utf8')) as { meshes: Record<string, number[]> }).meshes
    : undefined;

  await mkdir(outDir, { recursive: true });
  // sharp handles PNG/JPEG/WebP, which covers every image format the authored
  // GLBs use. Loaded lazily so the tool still runs without it when --materials
  // is not requested.
  const decodeImage = materials
    ? async (bytes: Uint8Array, _mimeType: string) => {
        const sharp = (await import('sharp')).default;
        const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
        return { width: info.width, height: info.height, data: new Uint8Array(data) };
      }
    : undefined;
  const preview = await createHeadlessPreview({ width, height, ...(decodeImage ? { decodeImage } : {}) });
  const rendered: Array<{ name: string; image: PreviewImage }> = [];
  try {
    for (const input of inputs) {
      const started = Date.now();
      const glb = await loadGlb(input);
      const name = basename(input).replace(/\.(glb|gz)$/g, '');
      const shared = { camera, materials, ...(vertexColors ? { vertexColors } : {}) };
      if (views) {
        // Named review views: one file each, with the framing coverage the
        // previous renderer reported so a subject out of frame is caught.
        const shots = await preview.renderViews(glb, { ...shared, views, raised });
        for (const shot of shots) {
          const file = join(outDir, `${name}.${shot.view}.png`);
          await writeFile(file, encodePng(shot.image.pixels, shot.image.width, shot.image.height));
          process.stdout.write(`  ${file}  (coverage ${(shot.coverage * 100).toFixed(1)}%)\n`);
        }
        rendered.push({ name, image: shots[0]!.image });
      } else {
        const image = await preview.renderGlb(glb, shared);
        const file = join(outDir, `${name}.png`);
        await writeFile(file, encodePng(image.pixels, image.width, image.height));
        rendered.push({ name, image });
        process.stdout.write(`  ${file}  (${Date.now() - started}ms)\n`);
      }
    }
    if (rendered.length >= 2) {
      const diff = differenceImage(rendered[0]!.image, rendered[1]!.image);
      const file = join(outDir, `${rendered[0]!.name}--vs--${rendered[1]!.name}.png`);
      await writeFile(file, encodePng(diff.pixels, diff.width, diff.height));
      process.stdout.write(`  ${file}  (8x amplified difference)\n`);
    }
  } finally {
    await preview.dispose();
  }
}

await main();
