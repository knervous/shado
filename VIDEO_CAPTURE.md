# Headless video capture

Render Babylon scenes to video with no browser process, or stream a running
scene live. Babylon's real WebGPU engine runs on Dawn in Node, so what is
captured is the engine the game uses rather than an approximation.

```bash
npm install @knervous/shado @ffmpeg-installer/ffmpeg
npx shado-video --input model.glb --out spin.mp4 --seconds 6 --materials
```

## The three seams

A capture is three parts that know nothing about each other:

| seam | responsibility | implementations |
| --- | --- | --- |
| `FrameSource` | produces frames at requested times | `createSessionFrameSource` (headless), `createCanvasFrameSource` (browser), `createScriptFrameSource` (scene scripts) |
| `ShadoVideoEncoder` | compresses into a container | `createFfmpegEncoder` (Node), `createWebCodecsEncoder` (browser) |
| `VideoSink` | moves the bytes | `createFileSink`, `createHttpSink`, `createWebSocketSink`, `createDataChannelSink` |

`renderVideo({ source, encoder, sink, ... })` drives them.

The container is **fragmented MP4**, and that one choice is what makes the
seams work: the same byte stream is a valid file *and* is playable
incrementally, so swapping a destination from disk to a socket changes one
argument and nothing else. Every sink here produces byte-identical output.

Note that the three names people reach for are not peers. **WebCodecs is an
encoder**, **MSE is a receiver** (`createMseAppender`), and a WebSocket or data
channel is a **transport**. Only the last is a `VideoSink`.

## Two clocks

- `pacing: 'offline'` (default) renders as fast as the machine manages and
  stamps exact timestamps. A scene that takes seconds per frame still produces
  exactly `fps` frames per second of playback. This is the difference between
  rendering a video and screen-recording one.
- `pacing: 'realtime'` holds each frame until its wall-clock moment, for
  streaming a running app. `seconds` becomes optional; pass a `signal` to stop.

Nothing is dropped or duplicated when a realtime frame is late — `onLate` and
`result.lateFrames` report it. Silently duplicating frames would hide a real
problem, and the renderer is fast enough that persistent lateness means
something upstream is wrong.

## CLI

```
usage: --input <glb|glb.gz> [--input ...] | --scene <scene.json> | --script <script.mts>
       [--out <file.mp4>] [--sink <url>] [--ws <url>] [--seconds N] [--fps N]
       [--pacing offline|realtime]
       [--width N] [--height N] [--revolutions N] [--zoom N] [--beta rad]
       [--codec h264|vp9] [--crf N] [--bitrate bps] [--preset fast|balanced|quality]
       [--depth N] [--dither] [--stats] [--gif [file.gif]] [--gif-fps N] [--gif-width N]
       [--ffmpeg <path>] [--hw] [--hw-encoder <name>] [--rgba]
       [--materials] [--verbose]
```

- `--input` repeats, composing several models into one scene; `--scene` takes a
  JSON file of placements with positions, rotations and scales.
- `--materials` loads real glTF materials and textures through Babylon's own
  pipeline (needs `sharp`). Without it, geometry renders under a neutral
  material, which is enough to prove a mesh or a bake.
- `--stats` reports where the time went — `source`, `encode`, and `pacing`.
- `--gif` converts the finished file; see [GIF](#gif) below.

## Programmatic use

```ts
import { createSessionFrameSource, orbitCamera, renderVideo } from '@knervous/shado/video';
import { createFfmpegEncoder, createFileSink } from '@knervous/shado/video/node';
import { createPreviewSession } from '@knervous/shado/devtools';

const session = await createPreviewSession({ width: 1280, height: 720 });
await session.newScene({ materials: true });
await session.loadGlb(bytes);
const camera = await session.frameCamera({ zoom: 2.4 });

const result = await renderVideo({
  source: await createSessionFrameSource(session, {
    width: 1280, height: 720, camera,
    onFrame: orbitCamera(camera, { seconds: 6 }),
  }),
  encoder: createFfmpegEncoder(),
  sink: createFileSink('spin.mp4'),
  seconds: 6, fps: 30,
});
console.log(result.timing);   // { sourceMs, encodeMs, pacingMs }
```

## Scene scripts

The built-in turntable covers "spin this model". Anything past that — stepping
an animation, posing a rig, driving the shado runtime, moving a camera along a
path — wants the engine itself rather than another flag. A scene script is
handed the real `Scene` and `Engine`:

```ts
import { seekAnimationGroup, type VideoScene } from '@knervous/shado/video';

const scene: VideoScene = {
  async setup(context) {
    const actor = await context.loadGlb('hero.glb');
    const clip = actor.animationGroups[0];
    for (const group of actor.animationGroups) group.stop();
    return {
      camera: await context.session.frameCamera({ zoom: 1.85 }),
      seconds: (clip.to - clip.from) / 30,
      state: { clip },
    };
  },
  frame(t, context) {
    seekAnimationGroup(context.state.clip, t, { loop: true });
  },
};
export default scene;
```

Run it with `--script`. Two rules the API enforces or the example demonstrates:

- **Seek clips, never `play()`.** A played animation advances on wall clock, so
  an offline capture comes out at whatever speed the renderer managed.
  `seekAnimationGroup` positions the clip from the frame's time using the
  clip's own authored frame rate.
- **A container starts every animation group playing** when added to a scene.
  With several clips on one skeleton they fight the one being seeked; stop them
  all first.
- **`frame()` must be synchronous.** The pipeline renders ahead, so an async
  hook would let the next frame's state be applied before this one has
  rendered. This throws rather than producing frames on the wrong pose.

`src/video/examples/animated-actor.mts` is the worked reference. It runs on the
bundled openly-licensed Fox (see [`assets/ATTRIBUTION.md`](./assets/ATTRIBUTION.md))
and is parameterised by `SHADO_ASSET`, `SHADO_CLIP` and `SHADO_CYCLES`.

It also encodes something worth knowing: **a character needs different lighting
and framing than a building.** The session defaults are tuned for a building on
a dark background; under them a textured character renders as a near-silhouette
at 3.7% of frame.

## Live streaming

`demo/live-stream/` is a runnable example — `npm run demo:live`, then open
<http://localhost:8787>. Babylon renders headless on the server, encodes in
real time, and streams fragmented MP4 into a plain `<video src="/live.mp4">`.
The page contains no script at all.

In the browser, `createCanvasFrameSource` plus `createWebCodecsEncoder` needs no
ffmpeg: a canvas goes straight into a `VideoFrame`, so the pixels never leave
the GPU. `createMseAppender` is the receiving half for a WebSocket transport.

## Performance

Two findings dominate, both counter-intuitive and both measured rather than
assumed.

**GPU readback, not rendering, is the cost.** Rendering a framed model is
~0.6ms; reading it back was ~100ms — and flat across resolutions, because the
Dawn binding dispatches map callbacks on a shared tick rather than being
bandwidth-bound. The tick is *shared*, so readbacks pipeline almost perfectly:

| in flight | ms/frame at 960x540 |
| --- | --- |
| 1 | 100.0 |
| 2 | 51.2 |
| 8 | 12.8 |
| 16 | 8.3 |

`createCaptureStream` does this automatically, sizing the ring from a 64MB
budget. `--depth` overrides it.

**~71% of "encoding" was colour conversion.** Feeding ffmpeg RGBA made it
convert to YUV420 on the CPU; converting on the GPU instead removes that and
cuts readback to 1.5 bytes per pixel:

| 2560x1440, 60 frames | ms/frame | MB/frame |
| --- | --- | --- |
| rgba in, x264 converts | 23.5 | 14.7 |
| yuv420p in, no conversion | 6.7 | 5.5 |
| plumbing alone (`-f null`) | 3.4 | 14.7 |

This is on by default and needs `width % 8 == 0` and `height % 2 == 0`; the CLI
falls back to RGBA with a note at other sizes, and the API throws if you asked
for it explicitly. End to end, a 1440p capture went 2.8s to 1.4s.

**Hardware encoders are not the answer.** `--hw` drives the GPU's
fixed-function encoder, but it loses below 1080p, gains ~17% at 1440p on RGBA
input, and *loses to software x264 once the input is already YUV* (11.4ms vs
6.7ms). WebGPU itself has no encode API and none is proposed.

## Formats and quality

**Do not dither for video.** Dithering before 8-bit quantization genuinely
removes banding from the render — a smooth gradient goes from 216 to 3030 value
changes along its rows. But h264 at normal quality strips exactly that
low-amplitude high-frequency signal back out (3030 down to 322), and the
dithered clip ends with *fewer* distinct levels than the undithered one, for
slightly more bitrate. `--dither` exists for PNG stills and near-lossless
encodes. Measure image-quality tricks *through* the encoder, not on the render.

### GIF

`--gif` converts the finished file, and is deliberately a post-step rather than
a codec: a good GIF needs its 256-colour palette computed from the whole clip
before any frame is written, so it has no incremental form. Making it a codec
would break the property that lets everything else stream.

Sizes for a 2.3s clip: 360px/12fps 580KB, 480px/15fps 1146KB, 640px/20fps
2541KB. Pick 360–480 for anything going in a pull request. Note the inversion
from the rule above: GIF *does* need dithering, because 256 colours posterise
without it.

## Dependencies

`@ffmpeg-installer/ffmpeg` is an **optional peer**, not a dependency: it ships
~35MB of prebuilt binaries, and video is one of twenty subpaths. Install it
alongside if that is what you came for. The tools check for it before doing any
work and name the install command if it is missing. `--ffmpeg <path>` points at
your own build; nothing ever searches `PATH`.

`mp4-muxer` is likewise an optional peer, needed only for the browser
WebCodecs encoder.

Node has no built-in WebCodecs and none is on its roadmap — there is not a
single mention of it in the Node issue tracker. The native implementations that
provide it are themselves built on ffmpeg, so they relocate the dependency
rather than removing it.
