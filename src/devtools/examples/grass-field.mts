/**
 * Renders procedural grass from a compiled v2 grass field.
 *
 *   npx tsx src/devtools/examples/grass-field.mts <world.spatial.json.gz> [outDir]
 *
 * Nothing here places a blade. The field says only where grass may grow; every
 * blade's position, size and lean is a hash of its cell origin and index,
 * evaluated in the vertex shader. Density is `--blades`, and changing it
 * changes no data — which is the property the whole v2 package exists for.
 */
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

import { createPreviewSession } from '../session';
import {
  ShadoFoliageActor,
  ShadoFoliageContainer,
  GRASS_FIELD_TEXELS_PER_CELL,
  createShadoGrassPatch,
  packShadoGrassField,
  shadoGrassBlades,
} from '../../extensions/ShadoFoliageContainer';
import { ShadoLightingMode } from '../../extensions/ShadoActor';
import { shadoWorldGrassFieldFromPackage } from '../../world/grass-field';

const [input, outDir = './shado-previews/grass'] = process.argv.slice(2);
if (!input) throw new Error('usage: grass-field.mts <world.spatial.json.gz> [outDir]');
const bladesPerCell = Number(process.env.GRASS_BLADES ?? 6912);
const BLADE_SEGMENTS = Number(process.env.GRASS_SEGMENTS ?? 3);

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

// The same adapter the client uses, so zones baked before the field package
// existed render from the coverage they already carry.
const field = shadoWorldGrassFieldFromPackage(world, { bladeWidth: 0.09 });
if (!field?.cells.x.length) {
  throw new Error('This world carries no grass coverage to build a field from');
}
console.log(
  `  field: ${field.cells.x.length} cells at ${field.cellSize} m, ${bladesPerCell} blades/cell`
);

// A square block of cells around a chosen world position — GRASS_AT="x,z" — or
// around the best-covered cell, the way a streaming residency pass would hold a
// window around the player. Taking cells by array order instead would give a
// kilometres-wide strip one cell deep.
const BLOCK_RADIUS = 3;
const cellByKey = new Map<string, number>();
field.cells.x.forEach((x, cell) => cellByKey.set(`${x}:${field.cells.z[cell]}`, cell));

const wordsPerCell = field.coverage.wordsPerCell;
const coverageOf = (cell: number) => {
  let covered = 0;
  for (let word = 0; word < wordsPerCell; word++) {
    let bits = field.coverage.words[cell * wordsPerCell + word]!;
    while (bits) {
      bits &= bits - 1;
      covered++;
    }
  }
  return covered;
};
let anchor = 0;
const requested = process.env.GRASS_AT?.split(',').map(Number);
if (requested?.length === 2 && requested.every(Number.isFinite)) {
  const targetX = Math.floor(requested[0]! / field.cellSize);
  const targetZ = Math.floor(requested[1]! / field.cellSize);
  anchor = cellByKey.get(`${targetX}:${targetZ}`) ?? 0;
} else {
  let best = -1;
  field.cells.x.forEach((_, cell) => {
    const covered = coverageOf(cell);
    if (covered > best) {
      best = covered;
      anchor = cell;
    }
  });
}

const anchorX = field.cells.x[anchor]!;
const anchorZ = field.cells.z[anchor]!;
const resident: number[] = [];
for (let dz = -BLOCK_RADIUS; dz <= BLOCK_RADIUS; dz++) {
  for (let dx = -BLOCK_RADIUS; dx <= BLOCK_RADIUS; dx++) {
    const cell = cellByKey.get(`${anchorX + dx}:${anchorZ + dz}`);
    if (cell !== undefined) resident.push(cell);
  }
}

const session = await createPreviewSession({
  width: Number(process.env.GRASS_WIDTH ?? 900),
  height: Number(process.env.GRASS_HEIGHT ?? 560),
});
try {
  await session.newScene({ clearColor: [0.42, 0.56, 0.68], materials: true });
  const engine = session.engine as any;

  const ready = await ShadoFoliageContainer.initialize(engine, {
    backend: engine.isWebGPU ? 'storage' : 'datatex',
    extra: ShadoFoliageActor,
    wasm: false,
  });
  if (!ready) throw new Error('Unable to initialize the Shado foliage arena');

  const patch = createShadoGrassPatch(session.scene, 'grass-patch', bladesPerCell, BLADE_SEGMENTS);
  // The atlas is built from the source meshes' materials. Without one the page
  // is empty and Dawn rejects the upload, so give the blade a real gradient:
  // dark at the root, bright at the tip.
  const { RawTexture, StandardMaterial, Texture, Engine } = await import('@babylonjs/core');
  // A luminance ramp only. Hue comes from the instance colour and the tint
  // plugin, so the texture must not also carry green or the two multiply into
  // a washed-out khaki.
  const bladeRows = 64;
  const bladePixels = new Uint8Array(bladeRows * 4);
  for (let row = 0; row < bladeRows; row++) {
    const t = row / (bladeRows - 1);
    const luminance = Math.round(255 * (0.55 + t * 0.45));
    bladePixels[row * 4] = luminance;
    bladePixels[row * 4 + 1] = luminance;
    bladePixels[row * 4 + 2] = luminance;
    bladePixels[row * 4 + 3] = 255;
  }
  const bladeTexture = RawTexture.CreateRGBATexture(
    bladePixels,
    1,
    bladeRows,
    session.scene,
    false,
    false,
    Texture.BILINEAR_SAMPLINGMODE,
    Engine.TEXTURETYPE_UNSIGNED_BYTE
  );
  const bladeMaterial = new StandardMaterial('grass-blade', session.scene);
  bladeMaterial.diffuseTexture = bladeTexture;
  patch.material = bladeMaterial;
  const fieldTexture = packShadoGrassField(session.scene, field, resident);

  const container = new ShadoFoliageContainer(engine);
  container.reserveInstances(resident.length);
  const actors = container.addInstances(resident.length, undefined, {
    playRandomAnimation: false,
    rebuildNameplates: false,
  });

  let minimumX = Infinity;
  let minimumZ = Infinity;
  let maximumX = -Infinity;
  let maximumZ = -Infinity;
  let groundY = 0;
  resident.forEach((cell, row) => {
    const originX = field.cells.x[cell]! * field.cellSize;
    const originZ = field.cells.z[cell]! * field.cellSize;
    const actor = actors[row]!;
    actor.translation = new Float32Array([originX, 0, originZ, 1]);
    actor.rotation = new Float32Array([0, 0, 0, 1]);
    // The base of the two-tone range; the tint plugin varies each blade toward
    // the lighter colour by its own hash.
    actor.color = new Float32Array([0.44, 0.58, 0.24, 1]);
    actor.lightingMode = ShadoLightingMode.Lambert;
    // Only the cell's row in the field texture: ground heights are packed as
    // absolute world Y, so nothing has to be decoded against a per-cell range.
    actor.foliageParams = new Float32Array([row, 0, 0, 0]);
    minimumX = Math.min(minimumX, originX);
    minimumZ = Math.min(minimumZ, originZ);
    maximumX = Math.max(maximumX, originX + field.cellSize);
    maximumZ = Math.max(maximumZ, originZ + field.cellSize);
    groundY = field.heightField.minimumY[cell]!;
  });

  await container.attachFoliage(session.scene, [patch], {
    sourceHeight: 1,
    focus: () => [(minimumX + maximumX) / 2, groundY, (minimumZ + maximumZ) / 2],
    merge: false,
    materialTextures: { uShadoGrassField: fieldTexture },
    plugins: [
      shadoGrassBlades({
        cellSize: field.cellSize,
        coverageResolution: field.coverage.resolution,
        heightResolution: field.heightField.resolution,
        bladesPerCell,
        minHeight: field.minHeight,
        maxHeight: field.maxHeight,
        bladeWidth: field.bladeWidth,
      }),
      { plugin: 'wind', amplitude: 0.16, frequency: 1.1, gustAmplitude: 0.22 },
      {
        plugin: 'tint',
        variationColor: [0.66, 0.82, 0.38],
        variationStrength: 0.9,
        rootDarkening: 0.35,
      },
    ],
  });
  container.commit();

  const { Vector3 } = await import('@babylonjs/core');
  console.log(`  block spans ${(maximumX - minimumX).toFixed(0)} x ${(maximumZ - minimumZ).toFixed(0)} m`);
  await session.frameCamera({ view: 'front', raised: true });
  const camera = session.scene.activeCamera! as any;
  const centreX = (minimumX + maximumX) / 2;
  const centreZ = (minimumZ + maximumZ) / 2;
  const span = Math.max(maximumX - minimumX, maximumZ - minimumZ);
  camera.setTarget(new Vector3(centreX, groundY + 0.4, centreZ));
  camera.minZ = 0.05;
  camera.maxZ = 800;

  if (process.env.GRASS_BENCH) {
    // Time steady-state frames at the ground camera, where the most blades are
    // on screen, after a warm-up that lets pipelines compile.
    camera.alpha = -Math.PI / 2 - 1.05;
    camera.beta = 1.4;
    camera.radius = span * 0.12;
    container.frustumCull(camera, field.cellSize, 0);
    container.commit();
    // scene.render() only submits; the GPU finishes later. A capture reads the
    // framebuffer back and therefore waits for everything queued before it, so
    // timing two different frame counts and taking the slope isolates the
    // per-frame GPU cost from the fixed readback cost.
    const timeFrames = async (frames: number) => {
      for (let warm = 0; warm < 10; warm++) session.scene.render();
      await session.capture();
      const started = performance.now();
      for (let frame = 0; frame < frames; frame++) session.scene.render();
      await session.capture();
      return performance.now() - started;
    };
    const few = 5;
    const many = 65;
    const shortRun = await timeFrames(few);
    const longRun = await timeFrames(many);
    const perFrame = (longRun - shortRun) / (many - few);
    const visible = container.getVisibleCount?.() ?? 0;
    const perBlade = (BLADE_SEGMENTS + 1) * 2;
    console.log(
      `  BENCH blades/cell=${bladesPerCell} seg=${BLADE_SEGMENTS} visible=${visible} ` +
        `verts=${((visible * bladesPerCell * perBlade) / 1000).toFixed(0)}k ` +
        `frame=${perFrame.toFixed(2)}ms`
    );
  }

  const shots: [string, number, number, number][] = [
    ['overview', 1.05, 0.78, span * 0.75],
    ['ground', 1.05, 1.40, span * 0.12],
  ];
  for (const [name, alpha, beta, radius] of shots) {
    camera.alpha = -Math.PI / 2 - alpha;
    camera.beta = beta;
    camera.radius = radius;
    container.frustumCull(camera, field.cellSize, 0);
    container.commit();
    session.scene.render();
    console.log(`  ${name}: ${container.getVisibleCount?.() ?? '?'} cells drawn`);
    await session.captureToFile(join(outDir, `grass.${name}.png`));
  }
  console.log(
    `  ${resident.length} cells x ${bladesPerCell} blades = ` +
      `${(resident.length * bladesPerCell).toLocaleString()} blades from ` +
      `${((resident.length * GRASS_FIELD_TEXELS_PER_CELL * 4 * 4) / 1024).toFixed(1)} KiB of field data`
  );
} finally {
  await session.dispose();
}
