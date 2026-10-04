import { describe, expect, it } from '@jest/globals';
import type { DQClipInfo, PackedDQVAT } from '../src/extensions/VATBuilder/VATBuilder';
import {
  buildDQLibrary,
  compareDQAtlases,
  createPackedDQAtlas,
  decodeSvat,
  encodeSvat,
  assembleDQBody,
  planBodyFromLibrary,
  rebuildBodyFromLibrary,
  SvatCodec,
  svatComponentIndex,
  svatLayoutOf,
  type DQLibraryBody,
} from '../src/svat';
import { nodeSvatDecompressor } from '../src/svat/SvatNode';
import { gzipSync } from 'node:zlib';

/** A deterministic unit quaternion + translation per (clip, frame, bone name). */
function pose(clip: string, frame: number, bone: string): number[] {
  let h = 2166136261;
  for (const ch of `${clip}|${frame}|${bone}`) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
  const r = (k: number) => (((h >>> (k * 5)) & 1023) / 1023) * 2 - 1;
  const q = [r(0), r(1), r(2), 0.5 + Math.abs(r(3))];
  const n = Math.hypot(...q);
  return [...q.map((x) => x / n), r(4) * 0.3, r(5) * 0.3, r(1) * 0.3, r(2) * 0.3];
}

/**
 * A float32 body atlas (unpacked, one tile) whose clips sample `pose`, so two
 * bodies sharing bone names and clip names bake identical palettes.
 */
function bodyAtlas(options: {
  bones: string[];
  clips: Array<{ name: string; frames: number; source?: string }>;
  stride?: number;
  scale?: (clip: string, frame: number, bone: string) => number;
  negate?: Set<string>;
}): { atlas: PackedDQVAT; bones: string[] } {
  const stride = options.stride ?? 2;
  const clips: DQClipInfo[] = [];
  let frames = 0;
  for (const clip of options.clips) {
    clips.push({ name: clip.name, from: frames, to: frames + clip.frames - 1, frames: clip.frames, fps: 30 });
    frames += clip.frames;
  }
  const widthTexels = options.bones.length * stride;
  const atlas: PackedDQVAT = {
    componentType: 'float32', widthTexels, heightTexels: frames, framesTotal: frames,
    bones: options.bones.length, dqWidthBones: options.bones.length, dqTilesX: 1, dqFramesX: 1,
    dqStrideTexels: stride, dqHasScale: stride >= 3, clips, pixels: new Float32Array(widthTexels * frames * 4),
  };
  const layout = svatLayoutOf(atlas);
  options.clips.forEach((clip, i) => {
    for (let f = 0; f < clip.frames; f++) {
      options.bones.forEach((bone, b) => {
        const value = pose(clip.source ?? clip.name, f, bone);
        const sign = options.negate?.has(clip.name) ? -1 : 1;
        for (let c = 0; c < 8; c++) {
          atlas.pixels[svatComponentIndex(layout, clips[i].from + f, b, c >> 2, c & 3)] = sign * value[c];
        }
        if (stride >= 3) {
          atlas.pixels[svatComponentIndex(layout, clips[i].from + f, b, 2, 0)] = options.scale?.(clip.name, f, bone) ?? 1;
        }
      });
    }
  });
  return { atlas, bones: options.bones };
}

const CORE = ['root', 'pelvis', 'spine', 'head', 'arm_l', 'arm_r'];

describe('buildDQLibrary', () => {
  it('stores a clip shared by every body once, with no overrides', () => {
    const a = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 10 }, { name: 'walk', frames: 8 }] });
    const b = bodyAtlas({ bones: CORE, clips: [{ name: 'walk', frames: 8 }, { name: 'idle', frames: 10 }] });
    const library = buildDQLibrary([{ key: 'a', ...a }, { key: 'b', ...b }]);
    expect(library.atlas.clips.map((clip) => clip.name)).toEqual(['idle', 'walk']);
    expect(library.atlas.framesTotal).toBe(18);
    expect(library.overrides).toEqual({ a: null, b: null });
    expect(library.manifests.b.clips.walk).toMatchObject({ source: 'library', from: 10, to: 17 });
    expect(library.clipSources).toEqual({ idle: 'a', walk: 'a' });
  });

  it('overrides a clip that differs on a skinned bone, not on an unskinned one', () => {
    const a = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 6 }, { name: 'walk', frames: 6 }] });
    // b's walk is a different motion; only arm_r (column 5) is wrong in b's idle.
    const b = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 6 }, { name: 'walk', frames: 6, source: 'strut' }] });
    const layout = svatLayoutOf(b.atlas);
    for (let f = 0; f < 6; f++) b.atlas.pixels[svatComponentIndex(layout, f, 5, 0, 0)] += 0.5;
    const unskinnedArm = buildDQLibrary([{ key: 'a', ...a }, { key: 'b', ...b, skinnedColumns: [0, 1, 2, 3, 4] }]);
    expect(unskinnedArm.manifests.b.overrideClips).toEqual(['walk']);
    const skinnedArm = buildDQLibrary([{ key: 'a', ...a }, { key: 'b', ...b }]);
    expect(skinnedArm.manifests.b.overrideClips).toEqual(['idle', 'walk']);
    expect(skinnedArm.overrides.b?.framesTotal).toBe(12);
  });

  it('treats a sign-flipped dual quaternion as the same pose', () => {
    const a = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 6 }] });
    const b = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 6 }], negate: new Set(['idle']) });
    expect(buildDQLibrary([{ key: 'a', ...a }, { key: 'b', ...b }]).manifests.b.overrideClips).toEqual([]);
  });

  it('appends extra bones to the union and maps each body onto it', () => {
    const a = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 4 }] });
    const b = bodyAtlas({ bones: [...CORE.slice(0, 3), 'tail', ...CORE.slice(3)], clips: [{ name: 'idle', frames: 4 }] });
    const library = buildDQLibrary([{ key: 'a', ...a }, { key: 'b', ...b }]);
    expect(library.bones).toEqual([...CORE, 'tail']);
    expect(library.manifests.b.boneToUnion).toEqual([0, 1, 2, 6, 3, 4, 5]);
    // The library's idle came from a, which has no tail: b must keep its own.
    expect(library.manifests.b.overrideClips).toEqual(['idle']);
    const rebuilt = rebuildBodyFromLibrary(library, 'b', b.atlas.clips);
    expect(compareDQAtlases(rebuilt.atlas, b.atlas).worst).toBe(0);
  });

  it('narrows a stride-3 atlas whose skinned scale is 1, and refuses real scale', () => {
    const a = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 4 }] });
    const unit = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 4 }], stride: 3 });
    const library = buildDQLibrary([{ key: 'a', ...a }, { key: 'u', ...unit }]);
    expect(library.atlas.dqStrideTexels).toBe(2);
    expect(library.manifests.u.overrideClips).toEqual([]);
    const scaled = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 4 }], stride: 3, scale: (_c, _f, bone) => (bone === 'head' ? 1.2 : 1) });
    expect(() => buildDQLibrary([{ key: 'a', ...a }, { key: 's', ...scaled }])).toThrow(/scaled 1.2/);
    // ...unless that bone is skinned by nothing.
    expect(() => buildDQLibrary([{ key: 'a', ...a }, { key: 's', ...scaled, skinnedColumns: [0, 1, 2, 4, 5] }])).not.toThrow();
  });

  it('frame-packs the library and clamps frames-per-row to the texture width', () => {
    const a = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 40 }, { name: 'walk', frames: 25 }] });
    const library = buildDQLibrary([{ key: 'a', ...a }], { framesX: 8 });
    expect(library.atlas.dqFramesX).toBe(8);
    expect(library.atlas.widthTexels).toBe(8 * CORE.length * 2);
    expect(library.atlas.heightTexels).toBe(Math.ceil(65 / 8));
    const narrow = buildDQLibrary([{ key: 'a', ...a }], { framesX: 64, maxTextureDimension: 48 });
    expect(narrow.atlas.dqFramesX).toBe(4); // 48 / (6 bones * 2 texels)
    expect(() => buildDQLibrary([{ key: 'a', ...a }], { framesX: 1, maxTextureDimension: 32 })).toThrow(/rows/);
  });
});

describe('rebuildBodyFromLibrary', () => {
  const human = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 9 }, { name: 'cast', frames: 7 }, { name: 'walk', frames: 5 }] });
  const other = bodyAtlas({ bones: CORE, clips: [{ name: 'walk', frames: 5, source: 'stride' }, { name: 'idle', frames: 9 }] });
  const library = buildDQLibrary([{ key: 'human', ...human }, { key: 'other', ...other }], { framesX: 4 });

  it('reproduces each body exactly, in its own frame numbering, frame-packed', () => {
    for (const [key, body] of [['human', human], ['other', other]] as const) {
      const { atlas, filled } = rebuildBodyFromLibrary(library, key, body.atlas.clips);
      expect(filled).toEqual([]);
      expect(atlas.dqFramesX).toBe(4);
      expect(atlas.clips).toEqual(body.atlas.clips);
      expect(compareDQAtlases(atlas, body.atlas)).toMatchObject({ worst: 0, mismatchedLengths: [] });
    }
  });

  it('fills in library clips a body never baked, after its own frames', () => {
    const { atlas, filled } = rebuildBodyFromLibrary(library, 'other', other.atlas.clips, { fill: true });
    expect(filled).toEqual(['cast']);
    const cast = atlas.clips.find((clip) => clip.name === 'cast')!;
    expect(cast.from).toBe(14); // other's own 5 + 9 frames come first, unmoved
    expect(compareDQAtlases(atlas, human.atlas, { clips: ['cast'] }).worst).toBe(0);
    expect(compareDQAtlases(atlas, other.atlas).worst).toBe(0);
  });

  it('survives a .svat round trip, packing included', async () => {
    const half = createPackedDQAtlas({ bones: 3, frames: 10, clips: [{ name: 'x', from: 0, to: 9, frames: 10, fps: 30 }], componentType: 'float16', framesX: 4 });
    const layout = svatLayoutOf(half);
    // Real frames only: the padding slots after the last frame are not stored.
    for (let f = 0; f < 10; f++) for (let b = 0; b < 3; b++) for (let c = 0; c < 8; c++) {
      const i = svatComponentIndex(layout, f, b, c >> 2, c & 3);
      (half.pixels as Uint16Array)[i] = (i * 2654435761 + 7) & 0x3bff;
    }
    const bytes = await encodeSvat(half, { codec: SvatCodec.Gzip, compress: (b) => gzipSync(b), continuity: false });
    const back = await decodeSvat(bytes, { decompress: nodeSvatDecompressor() });
    expect(back.dqFramesX).toBe(4);
    for (let f = 0; f < 10; f++) for (let b = 0; b < 3; b++) for (let c = 0; c < 8; c++) {
      const i = svatComponentIndex(layout, f, b, c >> 2, c & 3);
      expect(back.pixels[i]).toBe(half.pixels[i]);
    }
  });
});

describe('assembleDQBody / planBodyFromLibrary', () => {
  const base = bodyAtlas({ bones: CORE, clips: [{ name: 'idle', frames: 5 }, { name: 'cast', frames: 4 }] });
  const tailed = bodyAtlas({ bones: [...CORE, 'tail'], clips: [{ name: 'idle', frames: 5 }, { name: 'swish', frames: 3 }] });
  const library = buildDQLibrary([{ key: 'base', ...base }, { key: 'tailed', ...tailed, skinnedColumns: [0, 1, 2, 3, 4, 5] }]);

  it('leaves bones a clip source never carried as the identity, not zero', () => {
    // 'cast' came from base, which has no tail: the library's tail slot is identity.
    const atlas = rebuildBodyFromLibrary(library, 'tailed', tailed.atlas.clips, { fill: true }).atlas;
    const layout = svatLayoutOf(atlas);
    const cast = atlas.clips.find((clip) => clip.name === 'cast')!;
    const tail = 6;
    const qr = [0, 1, 2, 3].map((c) => atlas.pixels[svatComponentIndex(layout, cast.from, tail, 0, c)]);
    expect(qr).toEqual([0, 0, 0, 1]);
  });

  it('assembles a body from a plan alone', () => {
    const plan = [
      { name: 'cast', source: 'library' as const, from: library.atlas.clips.find((c) => c.name === 'cast')!.from, frames: 4, fps: 30 },
      { name: 'idle', source: 'library' as const, from: 0, frames: 5, fps: 30 },
    ];
    const atlas = assembleDQBody(library.atlas, null, library.manifests.base.boneToUnion, plan, { framesX: 2 });
    expect(atlas.clips.map((c) => [c.name, c.from, c.to])).toEqual([['cast', 0, 3], ['idle', 4, 8]]);
    expect(compareDQAtlases(atlas, base.atlas).worst).toBe(0);
  });

  it('skips filling a clip whose source lacks a bone the body needs', () => {
    const bodyBones = { base: CORE, tailed: [...CORE, 'tail'] };
    const needsTail = planBodyFromLibrary(library, 'tailed', tailed.atlas.clips, { fill: true, requiredBones: ['tail'], bodyBones });
    expect(needsTail.filled).toEqual([]);
    expect(needsTail.skipped).toEqual(['cast']);
    const free = planBodyFromLibrary(library, 'base', base.atlas.clips, { fill: true, requiredBones: CORE, bodyBones });
    expect(free.filled).toEqual(['swish']);
  });

  it('refuses a clip table with gaps', () => {
    const gappy = [{ name: 'idle', from: 0, to: 4, frames: 5, fps: 30 }, { name: 'cast', from: 7, to: 10, frames: 4, fps: 30 }];
    expect(() => planBodyFromLibrary(library, 'base', gappy)).toThrow(/expected 5/);
  });
});

// Keep the body type in the public surface honest.
const _typed: DQLibraryBody = { key: 'k', atlas: bodyAtlas({ bones: CORE, clips: [] }).atlas, bones: CORE };
void _typed;
