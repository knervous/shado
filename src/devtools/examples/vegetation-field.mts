/**
 * The transient vegetation stack in one frame: procedural grass from the v2
 * field plus the zone's real understory stamps (shrubs, flowers, ferns), every
 * plant drawn through ShadoFoliageContainer with ring-radius culling and
 * shared wind. Trees are deliberately absent: they are explicitly placed
 * landmarks and render through the stamped-object layer.
 *
 *   npx tsx src/devtools/examples/vegetation-field.mts <world.spatial.json.gz> [outDir]
 *
 * Foliage LOD is deliberately absent: a distance ring is the whole scheme, the
 * same as the client layer. Trees are few enough per ring that instanced
 * rigid draws need nothing cleverer.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

import { createPreviewSession } from '../session';
import {
  ShadoFoliageActor,
  ShadoFoliageContainer,
  createShadoGrassPatch,
  packShadoGrassField,
  seedFoliageParams,
  shadoGrassBlades,
  type ShadoFoliagePluginSpec,
} from '../../extensions/ShadoFoliageContainer';
import { ShadoLightingMode } from '../../extensions/ShadoActor';
import { shadoWorldGrassFieldFromPackage } from '../../world/grass-field';
import { isShadoWorldTransientFoliageMetadata } from '../../world/foliage';
import { shadoWorldStampQuaternion } from '../../world/runtime';

const [input, outDir = './shado-previews/vegetation'] = process.argv.slice(2);
if (!input) throw new Error('usage: vegetation-field.mts <world.spatial.json.gz> [outDir]');
const centre = (process.env.VEG_AT ?? '-499,-118').split(',').map(Number);
const RING = Number(process.env.VEG_RING ?? 70);
const BLADES = Number(process.env.VEG_BLADES ?? 12288);

if (typeof (globalThis as any).ImageData === 'undefined') {
  (globalThis as any).ImageData = class {
    constructor(
      readonly data: Uint8ClampedArray,
      readonly width: number,
      readonly height: number
    ) {}
    readonly colorSpace = 'srgb';
  };
}

const raw = await readFile(input);
const world = JSON.parse(
  Buffer.from(input.endsWith('.gz') ? gunzipSync(raw) : raw).toString('utf8')
);
const objectRoot = path.join(path.dirname(path.resolve(input)), '..');

const field = shadoWorldGrassFieldFromPackage(world, { bladeWidth: 0.09 })!;
const decodeImage = async (bytes: Uint8Array) => {
  const sharp = (await import('sharp')).default;
  const { data, info } = await sharp(bytes)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data) };
};

/** The same per-canopy behavior the client layer applies. */
function foliagePlugins(boundsRadius: number): ShadoFoliagePluginSpec[] {
  const canopy = Math.max(1, boundsRadius);
  return [
    {
      plugin: 'wind',
      amplitude: 0.03 * canopy,
      frequency: 1.9 / Math.sqrt(canopy),
      gustAmplitude: 0.05 * canopy,
      gustFrequency: 0.18,
      gustWavelength: 64,
      direction: [1, 0.35],
      stiffnessInfluence: 0.55,
    },
  ];
}

const session = await createPreviewSession({ width: 1100, height: 640, decodeImage });
try {
  // Materials stay off: promoted objects now reference shared KTX2 textures
  // (KHR_texture_basisu), which Babylon routes to the Khronos transcoder — a
  // worker/wasm pipeline with nothing to load in Node, so a textured load
  // stalls forever. Understory renders neutral gray here; texture fidelity is
  // validated in the real client, which vendors the decoder.
  await session.newScene({ clearColor: [0.5, 0.62, 0.72], materials: false });
  const engine = session.engine as any;
  let clock = 0;

  // ── Load every understory asset before any Shado material exists. GLB
  // loads through the session stall once a ShaderMaterial is live in the
  // scene, so all imports happen first. ────────────────────────────────────
  const objects = world.objects;
  type Understory = { prototype: number; id: string; rows: number[]; meshes: any[] };
  const understory: Understory[] = [];
  for (let prototype = 0; prototype < objects.prototypes.id.length; prototype++) {
    const id = objects.prototypes.id[prototype];
    if (!isShadoWorldTransientFoliageMetadata(id, objects.prototypes.metadata[prototype] ?? {})) continue;
    const rows: number[] = [];
    objects.stamps.prototype.forEach((p: number, stamp: number) => {
      if (p !== prototype || !objects.stamps.enabled[stamp]) return;
      const dx = objects.stamps.positionX[stamp] - centre[0]!;
      const dz = objects.stamps.positionZ[stamp] - centre[1]!;
      if (Math.hypot(dx, dz) <= RING) rows.push(stamp);
    });
    if (!rows.length) continue;
    const file = path.join(objectRoot, decodeURIComponent(objects.prototypes.source[prototype]).replace(/^\/eqrequiem\//, ''));
    const glb = new Uint8Array(gunzipSync(await readFile(file)));
    const asset = await session.loadGlb(glb, { id });
    const meshes = asset.meshes.filter((mesh: any) => mesh.getTotalVertices?.() > 0);
    for (const mesh of meshes) mesh.setEnabled(false);
    understory.push({ prototype, id, rows, meshes });
  }

  // Initialize Shado only after every GLB is resident. The renderer adapter
  // hooks the engine in a way the session's glTF load path then trips over.
  const ready = await ShadoFoliageContainer.initialize(engine, {
    backend: engine.isWebGPU ? 'storage' : 'datatex',
    extra: ShadoFoliageActor,
    wasm: false,
  });
  if (!ready) throw new Error('Unable to initialize the Shado foliage arena');

  // ── Grass: resident cells inside the ring ─────────────────────────────────
  const resident: number[] = [];
  field.cells.x.forEach((cellX, cell) => {
    const cx = (cellX + 0.5) * field.cellSize;
    const cz = (field.cells.z[cell]! + 0.5) * field.cellSize;
    if (Math.hypot(cx - centre[0]!, cz - centre[1]!) <= RING) resident.push(cell);
  });
  const patch = createShadoGrassPatch(session.scene, 'grass', BLADES, 3);
  const bladeRows = 64;
  const bladePixels = new Uint8Array(bladeRows * 4);
  for (let row = 0; row < bladeRows; row++) {
    const luminance = Math.round(255 * (0.55 + (row / (bladeRows - 1)) * 0.45));
    bladePixels.set([luminance, luminance, luminance, 255], row * 4);
  }
  const { RawTexture, StandardMaterial, Texture, Engine, Vector3 } = await import('@babylonjs/core');
  const bladeTexture = RawTexture.CreateRGBATexture(
    bladePixels, 1, bladeRows, session.scene, false, false,
    Texture.BILINEAR_SAMPLINGMODE, Engine.TEXTURETYPE_UNSIGNED_BYTE
  );
  const bladeMaterial = new StandardMaterial('blade', session.scene);
  bladeMaterial.diffuseTexture = bladeTexture;
  patch.material = bladeMaterial;

  const grass = new ShadoFoliageContainer(engine);
  grass.reserveInstances(resident.length);
  const grassActors = grass.addInstances(resident.length, undefined, {
    playRandomAnimation: false, rebuildNameplates: false,
  });
  resident.forEach((cell, row) => {
    const actor = grassActors[row]!;
    actor.translation = new Float32Array([
      field.cells.x[cell]! * field.cellSize, 0, field.cells.z[cell]! * field.cellSize, 1,
    ]);
    actor.rotation = new Float32Array([0, 0, 0, 1]);
    actor.color = new Float32Array([0.44, 0.58, 0.24, 1]);
    actor.lightingMode = ShadoLightingMode.Lambert;
    actor.foliageParams = new Float32Array([row, 0, 0, 0]);
  });
  await grass.attachFoliage(session.scene, [patch], {
    sourceHeight: 1,
    focus: () => [centre[0]!, 0, centre[1]!],
    timeSource: () => clock,
    merge: false,
    materialTextures: {
      uShadoGrassField: packShadoGrassField(session.scene, field, resident),
    },
    plugins: [
      shadoGrassBlades({
        cellSize: field.cellSize,
        coverageResolution: field.coverage.resolution,
        heightResolution: field.heightField.resolution,
        bladesPerCell: BLADES,
        minHeight: field.minHeight,
        maxHeight: field.maxHeight,
        bladeWidth: field.bladeWidth,
      }),
      { plugin: 'wind', amplitude: 0.16, frequency: 1.1, gustAmplitude: 0.22 },
      { plugin: 'tint', variationColor: [0.66, 0.82, 0.38], variationStrength: 0.9, rootDarkening: 0.35 },
    ],
  });
  grass.commit();

  // ── Understory: one container per preloaded prototype ─────────────────────
  const containers: { container: ShadoFoliageContainer; radius: number }[] = [{ container: grass, radius: field.cellSize }];
  let treeCount = 0;
  let groundY = 0;
  for (const { prototype, id, rows, meshes } of understory) {
    const container = new ShadoFoliageContainer(engine);
    container.reserveInstances(rows.length);
    const actors = container.addInstances(rows.length, undefined, {
      playRandomAnimation: false, rebuildNameplates: false,
    });
    const rotation = new Float32Array(4);
    rows.forEach((stamp, index) => {
      const actor = actors[index]!;
      const x = objects.stamps.positionX[stamp];
      const y = objects.stamps.positionY[stamp];
      const z = objects.stamps.positionZ[stamp];
      actor.translation = new Float32Array([x, y, z, objects.stamps.scaleX[stamp]]);
      shadoWorldStampQuaternion(objects.stamps, stamp, rotation);
      actor.rotation = new Float32Array(rotation);
      actor.color = new Float32Array([1, 1, 1, 1]);
      actor.lightingMode = ShadoLightingMode.Lambert;
      actor.foliageParams = seedFoliageParams(x, z);
      groundY = y;
    });
    await container.attachFoliage(session.scene, meshes, {
      focus: () => [centre[0]!, groundY, centre[1]!],
      timeSource: () => clock,
      merge: true,
      plugins: foliagePlugins(objects.prototypes.boundsRadius[prototype]),
    });
    container.commit();
    containers.push({ container, radius: objects.prototypes.boundsRadius[prototype] });
    treeCount += rows.length;
    console.log(`  ${id}: ${rows.length} stamps`);
  }
  console.log(`  ring ${RING} m: ${resident.length} grass cells + ${treeCount} foliage stamps`);

  await session.frameCamera({ view: 'front', raised: true });
  const camera = session.scene.activeCamera! as any;
  camera.setTarget(new Vector3(centre[0]!, groundY + 3, centre[1]!));
  camera.minZ = 0.1;
  camera.maxZ = 1200;

  const shots: [string, number, number, number][] = [
    ['grove', 1.1, 1.32, RING * 0.55],
    ['overview', 1.1, 0.85, RING * 1.6],
  ];
  for (const [name, alpha, beta, radius] of shots) {
    camera.alpha = -Math.PI / 2 - alpha;
    camera.beta = beta;
    camera.radius = radius;
    for (const { container, radius: cullRadius } of containers) {
      container.frustumCull(camera, cullRadius, 0);
      container.commit();
    }
    session.scene.render();
    await session.captureToFile(join(outDir, `veg.${name}.png`));
  }
  // Prove shared wind reaches the trees, not only the grass.
  clock = 2.2;
  session.scene.render();
  await session.captureToFile(join(outDir, 'veg.grove.t1.png'));
  console.log(`  wrote 3 frames to ${outDir}`);
} finally {
  await session.dispose();
}
