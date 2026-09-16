/**
 * Headless render of the Shado particle container on the real WebGPU (Dawn) path.
 *
 *   npx tsx scripts/particle-preview.ts [outDir]
 *   FX_DEBUG_VIEW=grid|color npx tsx scripts/particle-preview.ts   # see ShadoParticleRenderer debugView
 *
 * Starts a burst, a rate emitter under gravity with an on-death child, and an anchored
 * emitter that moves, then writes frames at a few moments plus stats. This is the check
 * that the WGSL compiles and binds, that records reach the GPU, and that the closed-form
 * motion, ramps and premultiplied blending look right.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import sharp from 'sharp';

import { createPreviewSession } from '../src/devtools';
import {
  ShadoParticleAtlas,
  ShadoParticleContainer,
  ShadoParticleRamps,
  ShadoParticleRenderer,
} from '../src/render';

const outDir = path.resolve(process.argv[2] ?? 'build/particle-preview');
const SIZE = 256;

const session = await createPreviewSession({ width: SIZE, height: SIZE });
const scene = await session.newScene({ clearColor: [0.02, 0.02, 0.03], defaultLights: false });
const engine = session.engine;
console.log('engine', engine.getClassName?.(), 'webgpu', engine.isWebGPU);

const ok = await ShadoParticleContainer.initialize(engine);
if (!ok) throw new Error('container failed to initialize');
const container = new ShadoParticleContainer(engine, { capacity: 4096, emitterCapacity: 32, anchorCapacity: 8 });
const atlas = new ShadoParticleAtlas(scene, 64, 4);
const ramps = new ShadoParticleRamps();
const renderer = new ShadoParticleRenderer(scene, container, { atlas, ramps, renderingGroupId: 0, debugView: (process.env.FX_DEBUG_VIEW as 'color' | 'grid' | undefined) ?? 'none' });
renderer.material.onError = (_effect, errors) => console.error('[particle material]', errors);

const fire = ramps.acquire({
  color: [
    { t: 0, value: [1, 0.9, 0.4, 1] },
    { t: 0.5, value: [1, 0.35, 0.05, 1] },
    { t: 1, value: [0.3, 0.05, 0, 0] },
  ],
  size: [
    { t: 0, value: 0.3 },
    { t: 1, value: 1.2 },
  ],
});
const spark = ramps.acquire({
  color: [
    { t: 0, value: [0.6, 0.8, 1, 1] },
    { t: 1, value: [0.1, 0.2, 1, 0] },
  ],
  size: [{ t: 0, value: 0.25 }],
});

const disc = atlas.get('default');
// A fountain under gravity whose particles burst into sparks when they die.
container.startEmitter(
  {
    emission: { mode: 'rate', rate: 60 },
    shape: { kind: 'conic', dir: [0, 1, 0], radius: 0.35 },
    power: [4, 6],
    life: [0.9, 1.2],
    gravity: [0, -6, 0],
    rampRow: fire.row,
    sizeScale: fire.sizeScale,
    layer: disc.layer,
    origin: [-1.5, -2, 0],
  },
  0,
  [{ spec: { emission: { mode: 'burst', count: 4 }, attach: 'onDeath', shape: { kind: 'omni' }, power: [1, 2], life: [0.4, 0.6], rampRow: spark.row, sizeScale: spark.sizeScale, layer: disc.layer } }]
);
// A burst of alpha-blended blue.
container.startEmitter(
  {
    emission: { mode: 'burst', count: 120 },
    delay: 0.2,
    shape: { kind: 'sphere', radius: 0.2 },
    power: [1.5, 2.5],
    life: [1.5, 2],
    drag: 1.5,
    rampRow: spark.row,
    sizeScale: spark.sizeScale * 1.5,
    layer: disc.layer,
    additive: false,
    origin: [1.5, 1, 0],
  },
  0
);
// Following an anchor that moves across the frame.
const anchor = container.acquireAnchor([0, 0, 0]);
container.startEmitter(
  {
    emission: { mode: 'rate', rate: 90 },
    shape: { kind: 'omni' },
    power: [0.2, 0.5],
    life: [0.6, 0.8],
    rampRow: fire.row,
    sizeScale: fire.sizeScale * 0.6,
    layer: disc.layer,
    anchor,
  },
  0
);

const { ArcRotateCamera } = await import('@babylonjs/core/Cameras/arcRotateCamera.js');
const { Vector3 } = await import('@babylonjs/core/Maths/math.vector.js');
const camera = new ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 2, 9, new Vector3(0, 0, 0), scene);

await mkdir(outDir, { recursive: true });
const FRAME = 1 / 60;
let time = 0;
const moments = [0.25, 0.75, 1.5];
for (const moment of moments) {
  while (time + FRAME <= moment) {
    time += FRAME;
    container.setAnchor(anchor, [Math.sin(time * 2) * 2.5, -0.5, 0]);
    container.step(time);
    renderer.time = time;
    await session.captureRaw({ camera, width: SIZE, height: SIZE });
  }
  const image = await session.capture({ camera, width: SIZE, height: SIZE });
  const file = path.join(outDir, `frame-${moment.toFixed(2)}.png`);
  await writeFile(file, await sharp(Buffer.from(image.pixels), { raw: { width: SIZE, height: SIZE, channels: 4 } }).png().toBuffer());
  let lit = 0;
  for (let i = 0; i < image.pixels.length; i += 4) if (image.pixels[i] + image.pixels[i + 1] + image.pixels[i + 2] > 60) lit++;
  console.log(file, 'lit pixels', lit, 'drawCount', container.drawCount, 'spawned', container.spawnedTotal);
}
renderer.dispose();
await session.dispose();
process.exit(0);
