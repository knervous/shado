/**
 * Custom capture entrypoints: a script is handed the scene and decides what
 * happens in it.
 *
 * The CLI's built-in turntable covers "spin this GLB". Everything past that —
 * stepping an animation, posing a rig, driving the shado runtime, moving a
 * camera along a path, compositing several assets and animating between them —
 * wants the engine itself rather than another flag. A scene script gets the
 * real `Scene`, the real `Engine` and the real asset containers, and the
 * pipeline captures whatever it does.
 *
 * The shape is deliberately three hooks rather than one callback:
 *
 *     export default {
 *       async setup({ loadGlb, session }) {
 *         const actor = await loadGlb('hero.glb');
 *         const clip = actor.animationGroups[0];
 *         return { camera: await session.frameCamera({ zoom: 1.6 }), clip };
 *       },
 *       frame(t, { state }) { seekAnimationGroup(state.clip, t); },
 *     };
 *
 * `frame` **must be synchronous**. The capture pipeline renders several frames
 * ahead, so an async hook would let the next frame's state be applied before
 * this one has rendered — the video would come out with frames on the wrong
 * poses and nothing about it would say so.
 */

import type { BufferFrame, BufferFrameSource } from './types';
import { createSessionFrameSource, type CaptureCapableSession, type SessionFrameSourceOptions } from './session-source';

/**
 * Loader controls a scene script may use, mirroring the session's own.
 *
 * Kept structural rather than imported from `devtools`, so `video` does not
 * drag Dawn behind it — see the note on {@link ScriptCapableSession}.
 */
export interface ScriptLoadOptions {
  id?: string;
  /** Which Babylon loader to use. Defaults to `.glb`. */
  pluginExtension?: string;
  /** Base URL for formats with external references. */
  rootUrl?: string;
  /** Rewrites URIs the asset references; composed with the session's resolvers. */
  resolveUri?: (url: string) => string | Promise<string>;
  /** Merged over the loader defaults — Draco, meshopt, extension toggles. */
  pluginOptions?: Record<string, any>;
}

/** The session surface a scene script may use. Structural, so `devtools` stays out of the import graph. */
export interface ScriptCapableSession extends CaptureCapableSession {
  readonly engine: unknown;
  loadGlb(source: Uint8Array | string, options?: ScriptLoadOptions): Promise<any>;
  frameCamera(framing?: {
    view?: string;
    raised?: boolean;
    alpha?: number;
    beta?: number;
    zoom?: number;
    target?: [number, number, number];
  }): Promise<any>;
}

/** What a script is handed, in both `setup` and `frame`. */
export interface SceneContext<State = Record<string, unknown>> {
  readonly session: ScriptCapableSession;
  readonly scene: any;
  readonly engine: unknown;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  /** Capture length in seconds — scripts normally scale their motion by this. */
  readonly seconds: number;
  /** Total frames that will be captured. */
  readonly frames: number;
  /**
   * Loads a model into the scene and returns Babylon's AssetContainer, so
   * `animationGroups`, `skeletons` and `meshes` are all reachable. Paths
   * resolve relative to the script.
   *
   * Takes the full loader options, so a script can bring its own URI resolver
   * for an external texture store, load a `.gltf` with siblings, or configure
   * Draco — without the pipeline having to know about any of it.
   */
  loadGlb(path: string, options?: ScriptLoadOptions): Promise<any>;
  /** Whatever `setup` put in its returned `state`. Empty until then. */
  state: State;
}

export interface SceneSetup<State = Record<string, unknown>> {
  /** Camera to render from. Defaults to a framing of whatever the scene holds. */
  camera?: unknown;
  /** Overrides the capture length the caller asked for. */
  seconds?: number;
  fps?: number;
  /** Anything `frame` needs later — clips, meshes, a path to walk. */
  state?: State;
}

export interface VideoScene<State = Record<string, unknown>> {
  /** Builds the scene. Runs once, may await anything. */
  setup(context: SceneContext<State>): Promise<SceneSetup<State> | void> | SceneSetup<State> | void;
  /**
   * Advances the scene to `timeSeconds`. Runs before every captured frame and
   * MUST be synchronous — see the note at the top of this file.
   */
  frame?(timeSeconds: number, context: SceneContext<State>): void;
  /** Runs after the last frame, before the session is torn down. */
  teardown?(context: SceneContext<State>): Promise<void> | void;
}

/**
 * Accepts either shape a module might export: the scene object itself, or a
 * bare function, which is read as `setup`.
 */
export type SceneScriptModule<State = Record<string, unknown>> =
  | VideoScene<State>
  | ((context: SceneContext<State>) => Promise<SceneSetup<State> | void> | SceneSetup<State> | void);

export function toVideoScene<State>(loaded: SceneScriptModule<State>): VideoScene<State> {
  return typeof loaded === 'function' ? { setup: loaded } : loaded;
}

export interface ScriptFrameSourceOptions<State = Record<string, unknown>> {
  script: SceneScriptModule<State>;
  session: ScriptCapableSession;
  width: number;
  height: number;
  fps: number;
  seconds: number;
  /** Reads an asset the script asked for by path. Node supplies `readFile`. */
  readAsset: (path: string) => Promise<Uint8Array>;
  depth?: SessionFrameSourceOptions['depth'];
  /** Pixel layout to capture; see {@link SessionFrameSourceOptions.format}. */
  format?: SessionFrameSourceOptions['format'];
}

export interface ScriptFrameSource {
  source: BufferFrameSource;
  /** After `setup`, the script may have changed these. */
  seconds: number;
  fps: number;
  teardown(): Promise<void>;
}

/**
 * Runs a script's `setup`, then builds the frame source that will drive its
 * `frame` hook.
 *
 * Returns the possibly-adjusted timing because a script that knows its own clip
 * is 2.4 seconds long should be able to say so rather than making the caller
 * guess on the command line.
 */
export async function createScriptFrameSource<State extends Record<string, unknown>>(
  options: ScriptFrameSourceOptions<State>,
): Promise<ScriptFrameSource> {
  const scene = toVideoScene(options.script);
  const context: SceneContext<State> = {
    session: options.session,
    scene: options.session.scene,
    engine: options.session.engine,
    width: options.width,
    height: options.height,
    fps: options.fps,
    seconds: options.seconds,
    frames: Math.max(1, Math.round(options.seconds * options.fps)),
    async loadGlb(path, loadOptions) {
      return options.session.loadGlb(await options.readAsset(path), { id: path, ...loadOptions });
    },
    state: {} as State,
  };

  const setup = (await scene.setup(context)) ?? {};
  if (setup.state) context.state = setup.state;
  const seconds = setup.seconds ?? options.seconds;
  const fps = setup.fps ?? options.fps;
  // Rewritten so a `frame` hook reading `context.seconds` sees the length the
  // capture will actually run for, not the one that was requested.
  Object.assign(context, { seconds, fps, frames: Math.max(1, Math.round(seconds * fps)) });

  const camera = setup.camera ?? (await options.session.frameCamera());
  const source = await createSessionFrameSource(options.session, {
    width: options.width,
    height: options.height,
    camera,
    ...(options.depth ? { depth: options.depth } : {}),
    ...(options.format ? { format: options.format } : {}),
    ...(scene.frame ? { onFrame: (timeSeconds: number) => scene.frame!(timeSeconds, context) } : {}),
  });

  return {
    source: source as BufferFrameSource & { frame(t: number, u: number): Promise<BufferFrame> },
    seconds,
    fps,
    teardown: async () => { await scene.teardown?.(context); },
  };
}

/**
 * Seeks an animation group to a point in time — the deterministic way to
 * capture an animation.
 *
 * Letting a group `play()` and rendering as fast as possible ties the animation
 * to wall clock, which is exactly wrong for an offline capture: the video comes
 * out running at whatever speed the renderer happened to manage. Seeking every
 * frame instead means a slow scene and a fast one produce the same video.
 *
 * Babylon only applies a frame to its targets once the group has been started,
 * so a group that has never played is started and immediately paused.
 */
export function seekAnimationGroup(
  group: any,
  timeSeconds: number,
  options: { loop?: boolean; speed?: number } = {},
): void {
  if (!group) return;
  if (!group.isStarted) {
    group.start(false, 1);
    group.pause();
  }
  const from = group.from ?? 0;
  const to = group.to ?? 0;
  const span = to - from;
  // A group carries its own authored frame rate; using the capture's fps here
  // would play every clip at the wrong speed.
  const fps = group.targetedAnimations?.[0]?.animation?.framePerSecond ?? 30;
  let frame = from + timeSeconds * fps * (options.speed ?? 1);
  if (span > 0) {
    frame = options.loop === false
      ? Math.min(frame, to)
      : from + (((frame - from) % span) + span) % span;
  }
  group.goToFrame(frame);
}

/** {@link seekAnimationGroup} across every group in a container or scene. */
export function seekAnimationGroups(
  groups: readonly any[],
  timeSeconds: number,
  options: { loop?: boolean; speed?: number } = {},
): void {
  for (const group of groups) seekAnimationGroup(group, timeSeconds, options);
}
