/**
 * Animated GLB: pose a rig at chosen keyframes and shoot each from a chosen angle.
 *
 *   npx tsx src/devtools/examples/animated-poses.mts <rig.glb> [outDir]
 *
 * The point of the session API is that this file is ordinary Babylon code. The
 * session hands over the real engine and scene; stepping an animation is
 * `AnimationGroup.goToFrame`, and the camera is an ordinary ArcRotateCamera.
 */
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

import { createPreviewSession } from '../session';

const [input, outDir = './shado-previews/poses'] = process.argv.slice(2);
if (!input) throw new Error('usage: animated-poses.mts <rig.glb> [outDir]');

const raw = await readFile(input);
const glb = new Uint8Array(input.endsWith('.gz') ? gunzipSync(raw) : raw);

// sharp decodes the rig's embedded textures; without a decoder the scene can
// still render, but untextured.
const decodeImage = async (bytes: Uint8Array) => {
  const sharp = (await import('sharp')).default;
  const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data) };
};

const session = await createPreviewSession({ width: 512, height: 640, decodeImage });
try {
  await session.newScene({ clearColor: [0.05, 0.05, 0.07], materials: true });
  const container = await session.loadGlb(glb, { id: 'rig' });

  const groups = container.animationGroups ?? [];
  console.log(`  ${container.meshes.length} meshes, ${container.skeletons.length} skeletons, ${groups.length} animations`);
  if (!groups.length) throw new Error('This GLB carries no animation groups');

  // Animations arrive playing; stop them all so frames are set explicitly.
  for (const group of groups) group.stop();

  const clip = groups.find((g: any) => g.name === process.env.CLIP) ?? groups[0];
  const from = clip.from as number;
  const to = clip.to as number;
  console.log(`  posing '${clip.name}' over frames ${from}..${to}`);

  // A pose is a keyframe plus an angle: sample the clip at even intervals and
  // orbit as we go, so one sheet shows both the motion and the silhouette.
  const poses = 4;
  for (let index = 0; index < poses; index++) {
    const frame = from + ((to - from) * index) / Math.max(1, poses - 1);
    // Camera first: the scene cannot render without one, and posing needs a
    // render to push the skeleton onto the frame.
    await session.frameCamera({ view: index % 2 === 0 ? 'front' : 'isoFrontLeft', zoom: 2.1 });

    clip.start(false, 1, frame, frame, false);
    clip.goToFrame(frame);
    clip.pause();
    // Let the skeleton settle onto the posed frame before capturing.
    session.scene.render();
    const file = join(outDir, `${clip.name}.f${Math.round(frame)}.png`);
    await session.captureToFile(file);
    console.log(`  ${file}`);
  }
} finally {
  await session.dispose();
}
