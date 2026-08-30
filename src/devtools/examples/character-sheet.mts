/**
 * A character sheet, rendered in Node with no browser anywhere.
 *
 *   npx tsx src/devtools/examples/character-sheet.mts <rig.glb> [outDir]
 *
 * The session hands over the real `WebGPUEngine` and `Scene` running on Dawn,
 * so this is ordinary Babylon code: a turnaround is four camera angles, and an
 * action still is `AnimationGroup.goToFrame`. Nothing here is a preview-specific
 * API — the only things the session keeps are device setup, the browser shims,
 * offscreen capture and a stall watchdog.
 */
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

import { createPreviewSession } from '../session';

const [input, outDir = './shado-previews/character'] = process.argv.slice(2);
if (!input) throw new Error('usage: character-sheet.mts <rig.glb> [outDir]');

const raw = await readFile(input);
const glb = new Uint8Array(input.endsWith('.gz') ? gunzipSync(raw) : raw);

// Babylon's texture pipeline needs something that can turn bytes into pixels.
// `sharp` satisfies it; without a decoder the scene still renders, untextured.
const decodeImage = async (bytes: Uint8Array) => {
  const sharp = (await import('sharp')).default;
  const { data, info } = await sharp(bytes).ensureAlpha().raw()
    .toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data) };
};

/**
 * A wardrobe rig carries every outfit and every hairstyle at once, and the
 * runtime shows one of each. Drawn as loaded they are coincident surfaces that
 * z-fight into a white blob, so dressing the actor is step one of looking at
 * it. The rig spells the choice in its mesh names —
 * `<actor>_<outfit>_<part>`, `<actor>_cheveux_<style>`, `<actor>_visage` — and
 * the session hands over the whole scene graph, so this is four lines rather
 * than a render flag.
 */
const wear = (container: any, actor: string, outfit: string, hair: string) => {
  const keep = (name: string) =>
    name.startsWith(`${actor}_${outfit}_`)
    || name === `${actor}_cheveux_${hair}`
    || name.startsWith(`${actor}_visage`);
  let worn = 0;
  for (const mesh of container.meshes) {
    if (!mesh.getTotalVertices?.()) continue;
    mesh.setEnabled(keep(mesh.name));
    if (keep(mesh.name)) worn += 1;
  }
  return worn;
};

/** Rest turnaround, then two clips caught mid-motion. */
const TURNAROUND = ['front', 'isoFrontLeft', 'left', 'back'] as const;
const ACTION: ReadonlyArray<{ clip: string; at: number; view: string }> = [
  { clip: 'c05', at: 0.45, view: 'isoFrontLeft' },  // a sword swing, mid-arc
  { clip: 's28', at: 0.55, view: 'front' },         // a bow, at its lowest
];

const session = await createPreviewSession({ width: 512, height: 700, decodeImage });
try {
  await session.newScene({ clearColor: [0.04, 0.04, 0.05], materials: true });
  const container = await session.loadGlb(glb, { id: 'rig' });
  const groups = container.animationGroups ?? [];
  console.log(
    `${container.meshes.length} meshes, ${container.skeletons.length} skeletons, ${groups.length} clips`,
  );
  const worn = wear(
    container,
    process.env.ACTOR ?? 'FY_HOM',
    process.env.OUTFIT ?? 'armor01',
    process.env.HAIR ?? 'short01',
  );
  console.log(`  ${worn} meshes worn, the rest hidden`);

  // The glTF loader arrives with the clips already playing, and `stop()` leaves
  // every target at whatever value it was last evaluated at — so a rest pose
  // has to be asked for, not assumed.
  const rest = () => {
    for (const group of groups) { group.reset(); group.stop(); }
    for (const skeleton of container.skeletons ?? []) skeleton.returnToRest();
  };

  rest();
  for (const view of TURNAROUND) {
    // Camera first: the scene cannot render without one, and posing needs a
    // render to push the skeleton onto the frame.
    await session.frameCamera({ view, zoom: 2.15 });
    session.scene.render();
    await session.captureToFile(join(outDir, `rest.${view}.png`));
  }

  for (const { clip, at, view } of ACTION) {
    const group = groups.find((candidate: any) => candidate.name === clip);
    if (!group) { console.warn(`  no clip '${clip}'`); continue; }
    rest();
    const frame = group.from + (group.to - group.from) * at;
    await session.frameCamera({ view, zoom: 2.15 });
    group.start(false, 1, frame, frame, false);
    group.goToFrame(frame);
    group.pause();
    session.scene.render();
    await session.captureToFile(join(outDir, `${clip}.${view}.png`));
  }
  console.log(`  wrote ${TURNAROUND.length + ACTION.length} frames to ${outDir}`);
} finally {
  await session.dispose();
}
