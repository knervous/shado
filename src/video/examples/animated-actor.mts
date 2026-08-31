/**
 * A scene script that captures a skinned animation — the case the turntable
 * cannot express.
 *
 *   npx tsx src/video/cli.ts --script src/video/examples/animated-actor.mts \
 *     --out fox.mp4 --materials --width 720 --height 720
 *
 * Ships with the Khronos Fox: a skinned, textured, openly-licensed model with
 * three clips (Survey, Walk, Run), so the demo runs on a clean checkout with
 * nothing to source. See assets/ATTRIBUTION.md.
 *
 * Three things here are worth copying. First, the clip is *seeked* every frame
 * rather than played: an animation left to `play()` runs on wall clock, so an
 * offline capture would come out at whatever speed the renderer happened to
 * manage. Seeking makes a slow scene and a fast one produce identical video.
 * Second, `setup` returns its own `seconds`, computed from the clip's authored
 * length — the script knows how long it is, so the caller does not have to.
 * Third, the session's default lighting and framing are calibrated for a
 * building; a character needs both changed, and the values below say why.
 */

import { seekAnimationGroup, type SceneContext, type VideoScene } from '../scene-script';

/** Override per run; this is how a batch of previews gets made. */
const ASSET = process.env.SHADO_ASSET ?? '../../../assets/Fox.glb';
const CLIP = process.env.SHADO_CLIP ?? 'Run';
/** Cycles of the clip to capture, so the loop point is visible. */
const CYCLES = Number(process.env.SHADO_CYCLES ?? 2);

interface State extends Record<string, unknown> {
  clip: any;
  camera: any;
}

const scene: VideoScene<State> = {
  async setup(context: SceneContext<State>) {
    const actor = await context.loadGlb(ASSET);
    const groups = actor.animationGroups as any[];
    if (!groups?.length) throw new Error(`${ASSET} carries no animation groups`);
    const clip = groups.find((group) => group.name === CLIP) ?? groups[0];

    // Every group starts out playing when a container is added to the scene,
    // and a second group animating the same skeleton fights the one being
    // seeked. Stop them all; seekAnimationGroup restarts the one it needs.
    for (const group of groups) group.stop();

    // The session's default key-and-fill is calibrated for framing a building
    // against a dark background; a textured character under it reads as a
    // silhouette. A back rim and a lifted fill are what make detail legible.
    const { DirectionalLight } = await import('@babylonjs/core/Lights/directionalLight.js');
    const { Vector3 } = await import('@babylonjs/core/Maths/math.js');
    const rim = new DirectionalLight('rim', new Vector3(0.6, -0.3, 1), context.scene);
    rim.intensity = 2.2;
    const bounce = new DirectionalLight('bounce', new Vector3(0.2, 1, -0.4), context.scene);
    bounce.intensity = 0.9;
    for (const light of context.scene.lights) {
      if (light.name === 'key') light.intensity = 3.2;
      if (light.name === 'fill') light.intensity = 0.8;
    }

    // Framed off the bind pose, and a rig's limbs swing well outside it. The
    // default framing that suits a building leaves a character at 3.7% of frame.
    const camera = await context.session.frameCamera({ beta: Math.PI / 2.2, zoom: 1.85 });

    const fps = clip.targetedAnimations?.[0]?.animation?.framePerSecond ?? 30;
    const clipSeconds = (clip.to - clip.from) / fps;
    return {
      camera,
      seconds: Math.max(1, clipSeconds * CYCLES),
      state: { clip, camera },
    };
  },

  frame(timeSeconds, context) {
    seekAnimationGroup(context.state.clip, timeSeconds, { loop: true });
    // Drift the camera round as it plays, so the capture proves the rig deforms
    // rather than just that something on screen is moving.
    context.state.camera.alpha = -Math.PI / 2.6 + (timeSeconds / context.seconds) * Math.PI * 0.9;
  },
};

export default scene;
