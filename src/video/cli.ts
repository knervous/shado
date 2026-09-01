#!/usr/bin/env node

/**
 * shado video — renders a GLB or a composed scene to a video file or an endpoint.
 *
 *   tsx src/video/cli.ts --input church.glb --out spin.mp4 --seconds 6
 *   tsx src/video/cli.ts --scene plaza.json --materials --out plaza.mp4
 *   tsx src/video/cli.ts --input church.glb --sink http://localhost:9000/upload
 *
 * The camera orbits the framed subject once over the capture by default, which
 * is the turntable an asset review wants. Anything more elaborate is a few
 * lines against `createSessionFrameSource` directly — the CLI is the proof, not
 * the surface.
 */

import { readFile } from 'node:fs/promises';
import { dirname, resolve as resolvePath } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { createPreviewSession } from '../devtools/session';
import { supportsGpuYuv } from '../devtools/yuv';
import { createSessionFrameSource, orbitCamera } from './session-source';
import { createScriptFrameSource, type SceneScriptModule } from './scene-script';
import { createHttpSink } from './http-sink';
import { createWebSocketSink, type WebSocketLike } from './websocket-sink';
import { renderVideo, type Pacing } from './driver';
import { createFfmpegEncoder, createFileSink, resolveFfmpegPath, writeGif } from './node';
import type { BufferFrameSource, VideoCodec, VideoSink } from './types';

const argv = process.argv.slice(2);
const flag = (name: string, fallback?: string): string | undefined => {
  const index = argv.indexOf(`--${name}`);
  return index >= 0 && argv[index + 1] ? argv[index + 1] : fallback;
};
const repeated = (name: string): string[] =>
  argv.reduce<string[]>((all, value, index) => (value === `--${name}` && argv[index + 1] ? [...all, argv[index + 1]!] : all), []);

interface Placement {
  glb: string;
  position?: [number, number, number];
  rotationDegrees?: [number, number, number];
  scale?: [number, number, number];
}

async function loadGlb(path: string): Promise<Uint8Array> {
  const raw = await readFile(path);
  return path.endsWith('.gz') ? new Uint8Array(gunzipSync(raw)) : new Uint8Array(raw);
}

const USAGE = `usage: --input <glb|glb.gz> [--input ...] | --scene <scene.json> | --script <script.mts>
       [--out <file.mp4>] [--sink <url>] [--ws <url>] [--seconds N] [--fps N]
       [--pacing offline|realtime]
       [--width N] [--height N] [--revolutions N] [--zoom N] [--beta rad]
       [--codec h264|vp9] [--crf N] [--bitrate bps] [--preset fast|balanced|quality]
       [--depth N] [--dither] [--stats] [--gif [file.gif]] [--gif-fps N] [--gif-width N]
       [--ffmpeg <path>] [--system-ffmpeg] [--hw] [--hw-encoder <name>] [--rgba]
       [--materials] [--verbose]

  Frames convert to yuv420p on the GPU by default, which needs width % 8 == 0
  and height % 2 == 0. --rgba forces the portable RGBA readback instead.

  ffmpeg resolves in order: --ffmpeg, $SHADO_FFMPEG, the bundled
  @ffmpeg-installer/ffmpeg, then PATH if --system-ffmpeg or
  $SHADO_FFMPEG_FROM_PATH=1. PATH is never searched unless asked for.

  --scene   JSON of { "placements": [{ "glb": "path.glb", "position": [x,y,z],
            "rotationDegrees": [p,y,r], "scale": [x,y,z] }] }, paths relative
            to the JSON file.
  --script  A module exporting a scene: { setup(ctx), frame?(t, ctx),
            teardown?(ctx) }, or a bare function read as setup. It is handed
            the real Scene and Engine, so animations, rigs and the shado
            runtime can all be driven. Its frame() hook must be synchronous. Asset
            paths resolve relative to the script. See
            src/video/examples/animated-actor.mts.
`;

async function main(): Promise<void> {
  const inputs = repeated('input');
  const scenePath = flag('scene');
  const scriptPath = flag('script');
  if (!inputs.length && !scenePath && !scriptPath) {
    process.stderr.write(USAGE);
    process.exitCode = 2;
    return;
  }

  const out = flag('out');
  const sinkUrl = flag('sink');
  const socketUrl = flag('ws');
  if (!out && !sinkUrl && !socketUrl) {
    process.stderr.write('Nothing to write to: pass --out <file>, --sink <url> or --ws <url>.\n');
    process.exitCode = 2;
    return;
  }

  let seconds = Number(flag('seconds', '6'));
  let fps = Number(flag('fps', '30'));
  const width = Number(flag('width', '1280'));
  const height = Number(flag('height', '720'));
  const revolutions = Number(flag('revolutions', '1'));
  const zoom = Number(flag('zoom', '2.4'));
  const beta = Number(flag('beta', String(Math.PI / 3)));
  const codec = (flag('codec', 'h264') as VideoCodec);
  const preset = flag('preset', 'balanced') as 'fast' | 'balanced' | 'quality';
  const crf = flag('crf');
  const bitrate = flag('bitrate');
  const materials = argv.includes('--materials');
  const verbose = argv.includes('--verbose');
  const pacing = (flag('pacing', 'offline') as Pacing);
  // Opt-in, not default. Dithering measurably removes banding from the render,
  // and h264 at normal quality then removes the dither — leaving output very
  // slightly worse than not dithering at all. Worth it only near-lossless.
  const dithering = argv.includes('--dither');
  // GPU conversion is the default, but it is a default rather than a request:
  // silently taking the slower path beats failing a command that used to work.
  // The API still throws when a caller asks for yuv420p explicitly.
  const gpuYuv = !argv.includes('--rgba') && supportsGpuYuv(width, height);
  if (!argv.includes('--rgba') && !gpuYuv) {
    process.stdout.write(
      `  note: ${width}x${height} cannot use GPU yuv420p (needs width % 8 == 0, height % 2 == 0); reading back RGBA\n`,
    );
  }
  const captureFormat = gpuYuv ? { format: 'yuv420p' as const } : {};

  const placements: Placement[] = scenePath
    ? (JSON.parse(await readFile(scenePath, 'utf8')) as { placements: Placement[] }).placements.map((entry) => ({
        ...entry,
        glb: resolvePath(dirname(scenePath), entry.glb),
      }))
    : inputs.map((glb) => ({ glb }));
  const depth = flag('depth') ? { depth: Number(flag('depth')) } : {};

  // sharp handles every image format the authored GLBs use, and is loaded only
  // when a textured render is actually asked for.
  const decodeImage = materials
    ? async (bytes: Uint8Array, _mimeType: string) => {
        const sharp = (await import('sharp')).default;
        const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
        return { width: info.width, height: info.height, data: new Uint8Array(data) };
      }
    : undefined;

  // Checked before anything expensive happens. Discovering the encoder is
  // missing after a GPU device, a scene and a textured model have loaded is a
  // slow way to learn a one-line fix.
  // CI images ship their own ffmpeg; --system-ffmpeg (or SHADO_FFMPEG_FROM_PATH=1)
  // opts into it rather than installing a second 35MB copy per job.
  const ffmpeg = {
    ...(flag('ffmpeg') ? { ffmpegPath: flag('ffmpeg')! } : {}),
    ...(argv.includes('--system-ffmpeg') ? { allowSystemPath: true } : {}),
  };
  await resolveFfmpegPath(ffmpeg.ffmpegPath, ffmpeg);

  const session = await createPreviewSession({ width, height, ...(decodeImage ? { decodeImage } : {}) });
  let teardown: () => Promise<void> = async () => {};
  try {
    let source: BufferFrameSource;

    if (scriptPath) {
      await session.newScene({ materials, dithering });
      const resolved = resolvePath(process.cwd(), scriptPath);
      const loaded = (await import(resolved)) as { default?: SceneScriptModule; scene?: SceneScriptModule };
      const script = loaded.default ?? loaded.scene;
      if (!script) throw new Error(`${scriptPath} exports neither a default scene nor a named 'scene'`);
      const built = await createScriptFrameSource({
        script,
        session,
        width,
        height,
        fps,
        seconds,
        // Relative to the script, so a scene script and its assets move together.
        readAsset: (path) => loadGlb(resolvePath(dirname(resolved), path)),
        ...depth,
        ...captureFormat,
      });
      // A script that knows its own clip length gets to say so.
      ({ seconds, fps } = built);
      source = built.source;
      teardown = built.teardown;
      process.stdout.write(`  ${scriptPath}  ${seconds}s @${fps}fps\n`);
    } else {
      // The built-in turntable: compose the placements and orbit them once.
      const { Vector3 } = await import('@babylonjs/core/Maths/math.js');
      await session.newScene({ materials, dithering });
      for (const placement of placements) {
        const container = await session.loadGlb(await loadGlb(placement.glb), { id: placement.glb });
        for (const root of container.rootNodes as any[]) {
          if (placement.position) root.position = new Vector3(...placement.position);
          if (placement.scale) root.scaling = new Vector3(...placement.scale);
          if (placement.rotationDegrees) {
            const [pitch, yaw, roll] = placement.rotationDegrees;
            root.rotation = new Vector3((pitch * Math.PI) / 180, (yaw * Math.PI) / 180, (roll * Math.PI) / 180);
          }
          root.computeWorldMatrix?.(true);
        }
      }
      session.scene.meshes.forEach((mesh: any) => mesh.computeWorldMatrix(true));
      const camera = await session.frameCamera({ beta, zoom });
      source = await createSessionFrameSource(session, {
        width,
        height,
        camera,
        onFrame: orbitCamera(camera, { seconds, revolutions }),
        ...depth,
        ...captureFormat,
      });
    }

    let sink: VideoSink;
    if (socketUrl) {
      // Node has had a global WebSocket client since 22, so this needs no
      // dependency; the sink only wants send/bufferedAmount/close.
      sink = createWebSocketSink(new (globalThis as any).WebSocket(socketUrl) as WebSocketLike);
    } else if (sinkUrl) {
      sink = createHttpSink({ url: sinkUrl, contentType: codec === 'vp9' ? 'video/webm' : 'video/mp4' });
    } else {
      sink = createFileSink(out!);
    }
    const encoder = createFfmpegEncoder({
      ...ffmpeg,
      ...(verbose ? { onLog: (line: string) => process.stderr.write(`ffmpeg: ${line}\n`) } : {}),
    });

    let lastReport = 0;
    const result = await renderVideo({
      source,
      encoder,
      sink,
      seconds,
      fps,
      pacing,
      codec,
      preset,
      ...(argv.includes('--hw') ? { hardware: true } : {}),
      ...(flag('hw-encoder') ? { hardwareEncoder: flag('hw-encoder')! } : {}),
      ...(crf !== undefined ? { crf: Number(crf) } : {}),
      ...(bitrate !== undefined ? { bitrate: Number(bitrate) } : {}),
      onProgress: (frame, total, elapsedMs) => {
        if (frame !== total && elapsedMs - lastReport < 1000) return;
        lastReport = elapsedMs;
        process.stdout.write(`  frame ${frame}/${total}  (${(frame / (elapsedMs / 1000)).toFixed(1)} fps)\n`);
      },
    });
    process.stdout.write(
      `  ${socketUrl ?? sinkUrl ?? out}  ${result.width}x${result.height} ${result.frames} frames `
      + `@${result.fps}fps in ${(result.elapsedMs / 1000).toFixed(1)}s`
      + `${result.lateFrames ? `, ${result.lateFrames} late` : ''}\n`,
    );
    if (argv.includes('--stats')) {
      const per = (ms: number): string => `${(ms / result.frames).toFixed(1)}ms/frame`;
      process.stdout.write(
        `  source ${per(result.timing.sourceMs)}  encode ${per(result.timing.encodeMs)}`
        + `${result.timing.pacingMs ? `  pacing ${per(result.timing.pacingMs)}` : ''}\n`,
      );
    }
    // A post-step on the finished file, which is why it needs --out. GIF has
    // no streaming form worth having.
    if (argv.includes('--gif')) {
      if (!out) throw new Error('--gif converts the written file, so it needs --out');
      const gifPath = flag('gif')?.endsWith('.gif') ? flag('gif')! : out.replace(/\.\w+$/, '.gif');
      await writeGif(out, gifPath, {
        ...ffmpeg,
        ...(flag('gif-fps') ? { fps: Number(flag('gif-fps')) } : {}),
        ...(flag('gif-width') ? { width: Number(flag('gif-width')) } : {}),
      });
      const { stat } = await import('node:fs/promises');
      process.stdout.write(`  ${gifPath}  ${Math.round((await stat(gifPath)).size / 1024)}KB\n`);
    }
  } finally {
    await teardown();
    await session.dispose();
  }
}

// Not top-level await: this entry is bundled for CJS as well, which cannot
// support it. Handling the rejection here also gives a real exit code instead
// of an unhandled-rejection trace.
main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
