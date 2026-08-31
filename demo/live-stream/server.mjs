/**
 * Server-side live rendering: the backend runs Babylon headless on Dawn,
 * renders continuously, encodes to fragmented MP4, and streams it to browsers.
 *
 * The browser does no rendering at all — it receives video. That is the whole
 * point: this is the same offline capture pipeline with `pacing: 'realtime'`,
 * pointed at an HTTP response instead of a file.
 *
 *   npm run demo:live      # then open http://localhost:8787
 *
 * One render is shared by every viewer. It starts when the first viewer
 * connects and stops when the last leaves, so an idle demo costs no GPU. Late
 * joiners are replayed the init segment (`ftyp` + `moov`), without which the
 * fragments they receive are undecodable.
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createPreviewSession } from '../../dist/devtools/index.js';
import { createSessionFrameSource, renderVideo } from '../../dist/video/index.js';
import { createFfmpegEncoder } from '../../dist/video/node.js';

const PORT = Number(process.env.PORT ?? 8787);
const HERE = fileURLToPath(new URL('./', import.meta.url));
const ASSET = fileURLToPath(new URL('../../assets/Fox.glb', import.meta.url));
const WIDTH = 960;
const HEIGHT = 540;
const FPS = 30;

const viewers = new Set();
let initSegment = [];
let initComplete = false;
let broadcast = null;

/** Walks top-level MP4 boxes to find where the init segment ends. */
function containsMoov(bytes) {
  let offset = 0;
  while (offset + 8 <= bytes.length) {
    const size = new DataView(bytes.buffer, bytes.byteOffset + offset).getUint32(0);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    if (type === 'moov') return true;
    if (size < 8) return false;
    offset += size;
  }
  return false;
}

/** A VideoSink that fans out to every connected response. */
const fanOutSink = {
  async write(bytes) {
    if (!initComplete) {
      initSegment.push(Buffer.from(bytes));
      if (containsMoov(bytes)) initComplete = true;
    }
    const chunk = Buffer.from(bytes);
    let blocked = null;
    for (const response of viewers) {
      // A viewer that cannot keep up must not stall the render for everyone
      // else, so backpressure is honoured for at most one slow client.
      if (!response.write(chunk)) blocked ??= response;
    }
    if (blocked) await new Promise((resolve) => blocked.once('drain', resolve));
  },
  async close() { for (const response of viewers) response.end(); },
  async abort() { for (const response of viewers) response.destroy(); },
};

/** Renders the scene forever, at wall-clock rate, into the fan-out sink. */
async function startBroadcast() {
  const controller = new AbortController();
  const session = await createPreviewSession({
    width: WIDTH,
    height: HEIGHT,
    decodeImage: async (bytes) => {
      const sharp = (await import('sharp')).default;
      const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      return { width: info.width, height: info.height, data: new Uint8Array(data) };
    },
  });

  await session.newScene({ materials: true });
  const container = await session.loadGlb(new Uint8Array(await readFile(ASSET)), { id: 'Fox.glb' });
  const groups = container.animationGroups;
  for (const group of groups) group.stop();
  const clip = groups.find((g) => g.name === 'Run') ?? groups[0];
  const clipFps = clip.targetedAnimations?.[0]?.animation?.framePerSecond ?? 30;

  const { DirectionalLight } = await import('@babylonjs/core/Lights/directionalLight.js');
  const { Vector3 } = await import('@babylonjs/core/Maths/math.js');
  new DirectionalLight('rim', new Vector3(0.6, -0.3, 1), session.scene).intensity = 2.2;
  for (const light of session.scene.lights) {
    if (light.name === 'key') light.intensity = 3.2;
    if (light.name === 'fill') light.intensity = 0.8;
  }
  const camera = await session.frameCamera({ beta: Math.PI / 2.2, zoom: 1.9 });

  const source = await createSessionFrameSource(session, {
    width: WIDTH,
    height: HEIGHT,
    camera,
    format: 'yuv420p',
    // Seeked, never played: the clip must advance with the stream clock rather
    // than with however fast the renderer happens to run.
    onFrame: (t) => {
      const span = clip.to - clip.from;
      clip.goToFrame(clip.from + ((t * clipFps) % span));
      camera.alpha = -Math.PI / 2.6 + t * 0.6;
    },
  });
  clip.start(false, 1);
  clip.pause();

  const run = renderVideo({
    source,
    encoder: createFfmpegEncoder(),
    sink: fanOutSink,
    fps: FPS,
    pacing: 'realtime',
    signal: controller.signal,
    preset: 'fast',
    // Realtime defaults to a lookahead of 2 to keep latency down, but this
    // binding dispatches GPU readbacks on a shared ~100ms tick, so 2 in flight
    // is ~50ms per frame and the render falls steadily behind wall clock.
    // Eight costs 8/30 = 266ms of latency and comfortably keeps up.
    lookahead: 8,
    onLate: (frame, lateMs) => {
      if (frame % 60 === 0) console.log(`  frame ${frame} was ${lateMs.toFixed(0)}ms late`);
    },
  }).catch((error) => {
    if (!controller.signal.aborted) console.error('broadcast failed:', error.message);
  }).finally(async () => {
    await session.dispose();
    initSegment = [];
    initComplete = false;
    broadcast = null;
    console.log('broadcast stopped');
  });

  console.log(`broadcasting ${WIDTH}x${HEIGHT} @${FPS}fps`);
  return { controller, run };
}

const TYPES = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript' };

const server = createServer(async (request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);

  if (url.pathname === '/live.mp4') {
    response.writeHead(200, {
      'content-type': 'video/mp4',
      'cache-control': 'no-store',
      // No content-length: the stream has no end until the viewer leaves.
      'transfer-encoding': 'chunked',
    });
    for (const chunk of initSegment) response.write(chunk);
    viewers.add(response);
    console.log(`viewer joined (${viewers.size} total)`);
    broadcast ??= await startBroadcast();
    const leave = () => {
      if (!viewers.delete(response)) return;
      console.log(`viewer left (${viewers.size} total)`);
      if (viewers.size === 0 && broadcast) broadcast.controller.abort();
    };
    request.on('close', leave);
    response.on('close', leave);
    return;
  }

  const file = join(HERE, url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\//, ''));
  try {
    const body = await readFile(file);
    response.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    response.end(body);
  } catch {
    response.writeHead(404).end('not found');
  }
});

server.listen(PORT, () => console.log(`live render demo on http://localhost:${PORT}`));
