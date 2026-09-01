/**
 * Node encoding and transports for shado video (`@knervous/shado/video/node`).
 *
 * Node has no WebCodecs — `VideoEncoder` is undefined as of Node 24 — so the
 * encoder here shells out to ffmpeg. That binary is a package dependency, never
 * something found on PATH: `@ffmpeg-installer/ffmpeg` ships prebuilt binaries
 * inside its per-platform npm tarballs, so an install brings the encoder with
 * it and a machine without ffmpeg is not a machine that fails.
 *
 * The container is fragmented MP4. That one choice is what lets a single
 * encoder serve every transport: the byte stream is playable incrementally as
 * it arrives AND is a valid file once it stops, so a sink never has to know it
 * is carrying video.
 */

import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { constants, createWriteStream, type WriteStream } from 'node:fs';
import { access, chmod, mkdir, unlink } from 'node:fs/promises';
import { delimiter, dirname, join } from 'node:path';
import { once } from 'node:events';

import type { CapturedFrame, ShadoVideoEncoder, VideoSink, VideoSpec } from './types';

/** Environment overrides, so CI can configure this without touching code. */
export const FFMPEG_PATH_ENV = 'SHADO_FFMPEG';
export const FFMPEG_FROM_PATH_ENV = 'SHADO_FFMPEG_FROM_PATH';

export interface FfmpegResolveOptions {
  /**
   * Search `PATH` if the bundled package is absent.
   *
   * Off by default: a library that silently uses whatever binary the host
   * happens to have is a library whose output depends on the machine. It is
   * opt-in rather than forbidden because CI images ship their own ffmpeg and
   * installing a second 35MB copy per job is waste, not safety.
   *
   * Also enabled by `SHADO_FFMPEG_FROM_PATH=1`.
   */
  allowSystemPath?: boolean;
}

/** Finds an executable on `PATH` without shelling out, so it works on Windows too. */
async function findOnSystemPath(name: string): Promise<string | undefined> {
  const directories = (process.env.PATH ?? '').split(delimiter).filter(Boolean);
  // PATHEXT is how Windows decides what counts as executable; POSIX has no
  // equivalent and an empty suffix is the only candidate.
  const suffixes = process.platform === 'win32'
    ? (process.env.PATHEXT ?? '.EXE;.CMD;.BAT').split(';').filter(Boolean)
    : [''];
  for (const directory of directories) {
    for (const suffix of suffixes) {
      const candidate = join(directory, name + suffix);
      try {
        await access(candidate, constants.X_OK);
        return candidate;
      } catch {
        // Not here, or not executable; keep looking.
      }
    }
  }
  return undefined;
}

/**
 * Locates ffmpeg, in order: an explicit path, `SHADO_FFMPEG`, the bundled
 * package, then — only when allowed — `PATH`.
 *
 * The default stays package-only by design, so a normal install cannot end up
 * depending on the host's software without someone saying so.
 */
export async function resolveFfmpegPath(
  override?: string,
  options: FfmpegResolveOptions = {},
): Promise<string> {
  override ??= process.env[FFMPEG_PATH_ENV] || undefined;
  if (override) {
    // Validated rather than trusted: an unchecked override defeats the whole
    // point of resolving early, and surfaces as a spawn failure after a GPU
    // device and a scene have already been built.
    try {
      await access(override, constants.X_OK);
    } catch {
      try {
        await chmod(override, 0o755);
      } catch {
        throw new Error(`No executable ffmpeg at ${override}`);
      }
    }
    return override;
  }
  let path: string | undefined;
  try {
    const loaded: any = await import('@ffmpeg-installer/ffmpeg');
    path = (loaded.default ?? loaded)?.path;
  } catch (error) {
    const allowSystem = options.allowSystemPath || process.env[FFMPEG_FROM_PATH_ENV] === '1';
    const found = allowSystem ? await findOnSystemPath('ffmpeg') : undefined;
    if (found) return found;
    throw new Error(
      'shado video needs an ffmpeg. Install the bundled one (`npm i @ffmpeg-installer/ffmpeg`), '
      + `set ${FFMPEG_PATH_ENV}=/path/to/ffmpeg, or allow the system one with `
      + `${FFMPEG_FROM_PATH_ENV}=1. Import failed: ${(error as Error).message}`,
    );
  }
  if (!path) throw new Error('@ffmpeg-installer/ffmpeg resolved without a binary path');
  // The package sets its own execute bit in a postinstall script, which npm
  // skips under --ignore-scripts. Without this the spawn fails with EACCES and
  // nothing explains why.
  try {
    await access(path, constants.X_OK);
  } catch {
    await chmod(path, 0o755);
  }
  return path;
}

const PRESETS: Record<NonNullable<VideoSpec['preset']>, string> = {
  fast: 'veryfast',
  balanced: 'medium',
  quality: 'slow',
};

/**
 * The platform's fixed-function video encoder.
 *
 * Linux is deliberately absent: VAAPI needs an explicit device and a hardware
 * upload filter, and NVENC needs an NVIDIA card, so guessing produces a
 * confusing ffmpeg error rather than a working encode. Name one explicitly
 * with `hardwareEncoder` there.
 */
const HARDWARE_ENCODERS: Record<string, string | undefined> = {
  darwin: 'h264_videotoolbox',
  win32: 'h264_mf',
};

function hardwareEncoderFor(spec: VideoSpec): string {
  const named = spec.hardwareEncoder ?? HARDWARE_ENCODERS[process.platform];
  if (!named) {
    throw new Error(
      `No default hardware encoder for ${process.platform}; name one with hardwareEncoder `
      + '(h264_nvenc, h264_vaapi, ...) or leave hardware off.',
    );
  }
  return named;
}

/** Builds the argument list for one encode. Exported for tests and for debugging a bad run. */
export function ffmpegArgs(spec: VideoSpec): string[] {
  const codec = spec.codec ?? 'h264';
  const args = [
    '-hide_banner',
    '-loglevel', 'error',
    '-f', 'rawvideo',
    '-pix_fmt', spec.sourceFormat === 'yuv420p' ? 'yuv420p' : spec.sourceFormat === 'bgra8' ? 'bgra' : 'rgba',
    '-s', `${spec.width}x${spec.height}`,
    '-r', String(spec.fps),
    '-i', 'pipe:0',
    '-an',
  ];
  // The GPU readback is bottom-up. Flipping it here rides along with the colour
  // conversion ffmpeg is already doing, instead of costing a full-resolution
  // CPU pass per frame on our side.
  if (spec.sourceFlipped) args.push('-vf', 'vflip');

  const hardware = spec.hardware === true && codec !== 'vp9';
  if (hardware) {
    args.push('-c:v', hardwareEncoderFor(spec));
  } else if (codec === 'vp9') {
    args.push('-c:v', 'libvpx-vp9', '-deadline', 'realtime', '-cpu-used', '4');
  } else {
    args.push('-c:v', 'libx264', '-preset', PRESETS[spec.preset ?? 'balanced']);
  }
  // yuv420p is the only chroma layout every player and browser will decode.
  args.push('-pix_fmt', 'yuv420p');
  // Frames converted on the GPU carry BT.709 limited range. Tagging it means a
  // player uses the matrix we actually encoded with rather than guessing from
  // the resolution.
  if (spec.sourceFormat === 'yuv420p') {
    args.push('-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv');
  }
  if (spec.bitrate) args.push('-b:v', String(Math.round(spec.bitrate)));
  // Hardware encoders have no CRF: rate control lives in the fixed-function
  // block and only understands a bitrate. Silently dropping a requested crf
  // would be worse than picking a defensible default from it.
  else if (hardware) args.push('-b:v', String(Math.round(spec.width * spec.height * spec.fps * 0.07)));
  else args.push('-crf', String(spec.crf ?? (codec === 'vp9' ? 32 : 20)));
  // A keyframe bounds how long a late subscriber waits for a fragment it can
  // decode. Two seconds for a file; one for a live stream, where joining
  // quickly is worth the bitrate.
  args.push('-g', String(Math.max(1, Math.round(spec.fps * (spec.latencyCritical ? 1 : 2)))));
  // Without this x264 buffers frames internally to look ahead, which is free
  // compression for a file and pure added latency for a stream.
  if (spec.latencyCritical && codec !== 'vp9' && !hardware) args.push('-tune', 'zerolatency');

  if (codec === 'vp9') {
    args.push('-f', 'webm');
  } else {
    args.push(
      '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
      '-f', 'mp4',
    );
  }
  args.push('pipe:1');
  return args;
}

export interface FfmpegEncoderOptions extends FfmpegResolveOptions {
  /** Overrides the bundled binary. Also settable with `SHADO_FFMPEG`. */
  ffmpegPath?: string;
  /** Receives ffmpeg's stderr lines; useful when a run produces nothing. */
  onLog?: (line: string) => void;
}

/**
 * ffmpeg as a {@link ShadoVideoEncoder}: raw frames down stdin, fragmented MP4
 * up stdout and straight into the sink.
 */
export function createFfmpegEncoder(options: FfmpegEncoderOptions = {}): ShadoVideoEncoder {
  let child: ChildProcessWithoutNullStreams | null = null;
  let sink: VideoSink | null = null;
  let exited: Promise<number> | null = null;
  // Sink writes are serialized through this chain: ffmpeg's stdout arrives in
  // whatever chunks it likes, and a container's bytes are only valid in order.
  let pumping: Promise<void> = Promise.resolve();
  let failure: Error | null = null;
  const stderrTail: string[] = [];

  const fail = (error: Error): void => {
    failure ??= error;
  };

  return {
    async open(spec, target) {
      if (child) throw new Error('This ffmpeg encoder is already open');
      if (spec.sourceKind !== 'buffer') {
        throw new Error(
          'The ffmpeg encoder needs CPU-side frames. A GPU-resident source belongs with the '
          + 'WebCodecs encoder, which is the point of that source not reading back at all.',
        );
      }
      sink = target;
      const binary = await resolveFfmpegPath(options.ffmpegPath, options);
      const args = ffmpegArgs(spec);
      child = spawn(binary, args, { stdio: ['pipe', 'pipe', 'pipe'] });

      child.on('error', (error) => fail(new Error(`ffmpeg failed to start: ${error.message}`)));
      // EPIPE on stdin is how a crashed ffmpeg reports itself to the writer.
      // The real reason is in stderr, so swallow it and let close() explain.
      child.stdin.on('error', (error) => fail(new Error(`ffmpeg stdin: ${(error as Error).message}`)));

      child.stdout.on('data', (chunk: Buffer) => {
        const bytes = new Uint8Array(chunk);
        // Pause while the sink is busy, so a slow transport backs pressure all
        // the way up rather than buffering the stream in memory.
        child!.stdout.pause();
        pumping = pumping
          .then(() => sink!.write(bytes))
          .then(() => { child?.stdout.resume(); })
          .catch((error: Error) => { fail(error); child?.kill('SIGKILL'); });
      });

      child.stderr.on('data', (chunk: Buffer) => {
        for (const line of chunk.toString().split('\n')) {
          if (!line.trim()) continue;
          options.onLog?.(line);
          stderrTail.push(line);
          if (stderrTail.length > 40) stderrTail.shift();
        }
      });

      exited = new Promise<number>((resolve) => {
        child!.on('close', (code) => resolve(code ?? 0));
      });
    },

    async encode(frame: CapturedFrame) {
      if (!child) throw new Error('encode() before open()');
      if (frame.kind !== 'buffer') throw new Error('The ffmpeg encoder cannot take a GPU-resident frame');
      if (failure) throw failure;
      const stdin = child.stdin;
      if (!stdin.write(frame.data)) {
        // Backpressure from ffmpeg itself. Racing the exit keeps a dead
        // process from hanging the capture on a drain that will never fire.
        await Promise.race([once(stdin, 'drain'), exited!]);
        if (failure) throw failure;
      }
    },

    async close() {
      if (!child) throw new Error('close() before open()');
      child.stdin.end();
      const code = await exited!;
      // stdout data handlers can still be settling after the process exits;
      // the container's last fragment is in there.
      await pumping;
      if (failure) { await sink!.abort?.(failure); throw failure; }
      if (code !== 0) {
        const error = new Error(`ffmpeg exited with ${code}${stderrTail.length ? `:\n${stderrTail.join('\n')}` : ''}`);
        await sink!.abort?.(error);
        throw error;
      }
      await sink!.close();
    },

    async abort(error) {
      child?.kill('SIGKILL');
      if (exited) await exited.catch(() => 0);
      await pumping.catch(() => {});
      await sink?.abort?.(error);
    },
  };
}

export interface GifOptions extends FfmpegResolveOptions {
  /** Frame rate of the GIF. 12-15 reads as smooth and keeps the file sane. */
  fps?: number;
  /** Output width; height follows the aspect ratio. Defaults to 480. */
  width?: number;
  /**
   * Dithering when mapping to the 256-colour palette. Unlike h264 — where
   * dithering is worse than useless because the encoder strips it — a GIF
   * really does need it, or every gradient posterises into bands.
   */
  dither?: 'sierra2_4a' | 'bayer' | 'none';
  /** Bayer pattern size, 1-5. Larger is a finer pattern and a bigger file. */
  bayerScale?: number;
  /** Loop forever by default. */
  loop?: number;
  ffmpegPath?: string;
}

/**
 * Converts a finished video file to an animated GIF.
 *
 * Deliberately a post-step rather than a codec. A good GIF needs its 256-colour
 * palette computed from the whole clip before any frame can be written, so it
 * cannot be produced incrementally — which is exactly the property that lets
 * every other output here stream to a socket as well as a file. Offering
 * `codec: 'gif'` would be a lie as soon as you pointed it at a transport.
 *
 * Runs both passes in one invocation: `split` feeds the frames to `palettegen`
 * and to `paletteuse` at once. A single-pass GIF on ffmpeg's default web
 * palette looks far worse for no saving.
 */
export async function writeGif(input: string, output: string, options: GifOptions = {}): Promise<void> {
  const binary = await resolveFfmpegPath(options.ffmpegPath, options);
  const fps = options.fps ?? 15;
  const width = options.width ?? 480;
  const dither = options.dither ?? 'sierra2_4a';
  const use = dither === 'none'
    ? 'paletteuse=dither=none'
    : dither === 'bayer'
      ? `paletteuse=dither=bayer:bayer_scale=${options.bayerScale ?? 3}`
      : 'paletteuse=dither=sierra2_4a';
  const args = [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-i', input,
    '-filter_complex',
    `[0:v] fps=${fps},scale=${width}:-1:flags=lanczos,split [a][b];[a] palettegen=stats_mode=full [p];[b][p] ${use}`,
    '-loop', String(options.loop ?? 0),
    output,
  ];
  await mkdir(dirname(output), { recursive: true });
  const stderr: string[] = [];
  const child = spawn(binary, args, { stdio: ['ignore', 'ignore', 'pipe'] });
  child.stderr.on('data', (chunk: Buffer) => stderr.push(chunk.toString()));
  const code = await new Promise<number>((resolve) => child.on('close', (value) => resolve(value ?? 0)));
  if (code !== 0) throw new Error(`ffmpeg gif pass exited with ${code}${stderr.length ? `:\n${stderr.join('')}` : ''}`);
}

/** Writes the stream to a file. */
export function createFileSink(path: string): VideoSink {
  let stream: WriteStream | null = null;
  const open = async (): Promise<WriteStream> => {
    if (stream) return stream;
    await mkdir(dirname(path), { recursive: true });
    stream = createWriteStream(path);
    return stream;
  };
  return {
    async write(bytes) {
      const target = await open();
      if (!target.write(bytes)) await once(target, 'drain');
    },
    async close() {
      if (!stream) return;
      stream.end();
      await once(stream, 'close');
    },
    async abort() {
      if (!stream) return;
      stream.destroy();
      // A half-written video is worse than none: it looks like output.
      await unlink(path).catch(() => {});
    },
  };
}
