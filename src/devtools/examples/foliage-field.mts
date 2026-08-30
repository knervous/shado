/**
 * Renders a field of foliage instances through ShadoFoliageContainer.
 *
 *   npx tsx src/devtools/examples/foliage-field.mts <plant.glb[.gz]> [outDir]
 *
 * The point is compilation evidence: the container's generated WGSL only exists
 * once plugins are attached, and only a real device will reject it. This runs
 * the whole path on Dawn — atlas build, arena upload, plugin uniforms, and the
 * composed displacement block — and shoots two frames a second apart so the
 * wind plugin has visibly moved something.
 */
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

import { createPreviewSession } from '../session';
import {
  ShadoFoliageActor,
  ShadoFoliageContainer,
  seedFoliageParams,
} from '../../extensions/ShadoFoliageContainer';
import { ShadoLightingMode } from '../../extensions/ShadoActor';

const [input, outDir = './shado-previews/foliage'] = process.argv.slice(2);
if (!input) throw new Error('usage: foliage-field.mts <plant.glb[.gz]> [outDir]');

const raw = await readFile(input);
const glb = new Uint8Array(input.endsWith('.gz') ? gunzipSync(raw) : raw);

const decodeImage = async (bytes: Uint8Array) => {
  const sharp = (await import('sharp')).default;
  const { data, info } = await sharp(bytes)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data) };
};

// Shado's atlas builder returns browser `ImageData`. Node has no such global,
// and this is the only thing standing between the instancing path and a
// headless device, so supply the two fields the builder actually reads.
if (typeof (globalThis as any).ImageData === 'undefined') {
  (globalThis as any).ImageData = class {
    readonly data: Uint8ClampedArray;
    readonly width: number;
    readonly height: number;
    readonly colorSpace = 'srgb';
    constructor(data: Uint8ClampedArray, width: number, height: number) {
      this.data = data;
      this.width = width;
      this.height = height;
    }
  };
}

const GRID = 12;
const SPACING = 3.5;

const session = await createPreviewSession({ width: 768, height: 512, decodeImage });
try {
  await session.newScene({ clearColor: [0.06, 0.08, 0.09], materials: true });
  const asset = await session.loadGlb(glb, { id: 'plant' });
  const meshes = asset.meshes.filter((mesh: any) => mesh.getTotalVertices?.() > 0);
  if (!meshes.length) throw new Error('This GLB carries no renderable meshes');
  // The source meshes are the instancing template only; Shado draws the merged
  // copy it builds, so leaving these enabled would draw one plant twice.
  for (const mesh of meshes) mesh.setEnabled(false);

  const engine = session.engine as any;
  const ready = await ShadoFoliageContainer.initialize(engine, {
    backend: engine.isWebGPU ? 'storage' : 'datatex',
    extra: ShadoFoliageActor,
    wasm: false,
  });
  if (!ready) throw new Error('Unable to initialize the Shado foliage arena');

  const container = new ShadoFoliageContainer(engine);
  const count = GRID * GRID;
  container.reserveInstances(count);
  const actors = container.addInstances(count, undefined, {
    playRandomAnimation: false,
    rebuildNameplates: false,
  });

  for (let index = 0; index < count; index++) {
    const gridX = (index % GRID) - (GRID - 1) / 2;
    const gridZ = Math.floor(index / GRID) - (GRID - 1) / 2;
    const x = gridX * SPACING;
    const z = gridZ * SPACING;
    const params = seedFoliageParams(x, z);
    const yaw = params[3]! * Math.PI * 2;
    const actor = actors[index]!;
    actor.translation = new Float32Array([x, 0, z, 0.85 + params[1]! * 0.4]);
    actor.rotation = new Float32Array([0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)]);
    actor.color = new Float32Array([1, 1, 1, 1]);
    actor.lightingMode = ShadoLightingMode.Lambert;
    actor.foliageParams = params;
  }

  let clock = 0;
  await container.attachFoliage(session.scene as any, meshes, {
    sourceHeight: Math.max(
      ...meshes.map((mesh: any) => mesh.getBoundingInfo().boundingBox.maximum.y)
    ),
    timeSource: () => clock,
    focus: () => [0, 0, 0],
    merge: true,
    plugins: [
      { plugin: 'wind', amplitude: 0.35, frequency: 0.9, gustAmplitude: 0.5 },
      { plugin: 'proximityFade', fadeStart: 22, fadeEnd: 30 },
      { plugin: 'tint', variationColor: [0.55, 0.72, 0.3], variationStrength: 0.5 },
    ],
  });
  container.commit();
  // The template meshes are disabled, so the scene has no bounds to frame
  // against: place the camera on the field's own extent instead.
  await session.frameCamera({ view: 'front', raised: true });
  const camera = session.scene.activeCamera! as any;
  const extent = (GRID - 1) * SPACING;
  camera.setTarget?.(new (await import('@babylonjs/core')).Vector3(0, 1.5, 0));
  camera.alpha = -Math.PI / 2 - 0.6;
  camera.beta = 1.15;
  camera.radius = extent * 1.15;
  camera.minZ = 0.1;
  camera.maxZ = 500;
  container.frustumCull(camera as any, 1, 0);
  container.commit();
  session.scene.render();
  console.log(`  ${container.getVisibleCount?.() ?? '?'} of ${count} instances visible`);
  await session.captureToFile(join(outDir, 'foliage.t0.png'));

  // A second sample proves the wind uniform reaches the shader: identical
  // frames here would mean the plugin bind ran but changed nothing.
  clock = 1.6;
  session.scene.render();
  await session.captureToFile(join(outDir, 'foliage.t1.png'));
  console.log(`  wrote 2 frames to ${outDir}`);
} finally {
  await session.dispose();
}
