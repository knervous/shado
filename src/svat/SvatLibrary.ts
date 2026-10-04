/**
 * Shared dual-quaternion animation libraries.
 *
 * A DQ atlas stores each bone's *skinning* transform per frame — the animated
 * pose already composed with that body's bind pose. Bodies that share one
 * skeleton (same bone names, rest pose and bind) therefore bake the same clip
 * to the same palette, and a family of such bodies is mostly the same atlas
 * repeated. This module bakes that family into:
 *
 * - **one library atlas**: every clip once, in the family's *union* bone order
 *   (the reference body's bones, then each body's extra bones), frame-packed so
 *   the frame count stops being bounded by the texture's row limit;
 * - **per-body overrides**: only the clips a body plays differently, compared on
 *   the bones that body's mesh actually skins;
 * - **per-body manifests**: for each clip, library or override and its frame
 *   range, plus the body-bone -> union-column map.
 *
 * {@link rebuildBodyFromLibrary} reverses it: it reassembles a body's own atlas
 * from library + override (frame-packed, same frame numbering as the source),
 * optionally {@link DQLibraryRebuildOptions.fill | filling in} every library
 * clip the body never baked. That is both the round-trip proof and a way to
 * ship library content to a runtime that samples one atlas per body.
 *
 * Pure: no I/O and no Babylon. Pair with `encodeSvat` / `decodeSvat` to read and
 * write containers.
 */
import type { DQClipInfo, PackedDQVAT } from '../extensions/VATBuilder/VATBuilder';
import { halfToFloat } from './SvatFilters';
import { svatComponentIndex, type SvatLayout } from './SvatFormat';

/** One body of a family, as decoded from its atlas. */
export type DQLibraryBody = {
  /** Stable key, e.g. the model code. */
  key: string;
  atlas: PackedDQVAT;
  /** Atlas column -> bone name. Names are how bones are matched across bodies. */
  bones: readonly string[];
  /**
   * Columns the body's mesh skins (any vertex weight > 0). Only these must
   * match for a clip to come from the library, and only these are checked for
   * a non-unit scale before a stride-3 atlas is narrowed. Defaults to every column.
   */
  skinnedColumns?: Iterable<number>;
};

export type DQLibraryOptions = {
  /** Key of the body whose bone order and clips the library starts from. Defaults to the first. */
  reference?: string;
  /** Frames packed side by side per atlas row. Clamped to the texture width. Default 16. */
  framesX?: number;
  /** Largest texture dimension the consumer allows. Default 8192 (WebGPU's guaranteed minimum). */
  maxTextureDimension?: number;
  /**
   * Largest per-component difference, after the dual-quaternion sign is
   * resolved, that still counts as the same pose. Default 0.01 — well above
   * half-float noise, far below any visible change.
   */
  tolerance?: number;
};

/** Where one clip of one body lives. */
export type DQLibraryClipRef = {
  source: 'library' | 'override';
  /** First and last frame in that atlas. */
  from: number;
  to: number;
  fps: number;
};

export type DQLibraryManifest = {
  key: string;
  /** Body column -> library (union) column. */
  boneToUnion: number[];
  /** Every clip the body bakes, by name. */
  clips: Record<string, DQLibraryClipRef>;
  /** Names of the clips that come from the body's override atlas. */
  overrideClips: string[];
};

export type DQLibrary = {
  /** Union bone order: library column -> bone name. */
  bones: string[];
  /** Every clip once, in union space, frame-packed. */
  atlas: PackedDQVAT;
  /** Clip name -> body key the library copy was taken from. */
  clipSources: Record<string, string>;
  /** Per body: its override atlas (union space), or null when it needs none. */
  overrides: Record<string, PackedDQVAT | null>;
  manifests: Record<string, DQLibraryManifest>;
};

export type DQLibraryRebuildOptions = {
  /** Append every library clip the body does not bake, after its own frames. */
  fill?: boolean;
  /** Frames per row of the rebuilt atlas. Default: the library's. */
  framesX?: number;
  maxTextureDimension?: number;
};

export type DQLibraryRebuild = {
  atlas: PackedDQVAT;
  /** Clips appended from the library (only with `fill`). */
  filled: string[];
};

const DEFAULT_FRAMES_X = 16;
const DEFAULT_MAX_TEXTURE = 8192;
const DEFAULT_TOLERANCE = 0.01;
/** A library atlas carries rotation + translation only. */
const OUT_STRIDE = 2;

/** The layout `svatComponentIndex` addresses, from a packed atlas. */
export function svatLayoutOf(atlas: PackedDQVAT): SvatLayout {
  return {
    bones: atlas.bones,
    framesTotal: atlas.framesTotal,
    widthBones: atlas.dqWidthBones,
    tilesX: atlas.dqTilesX,
    framesX: atlas.dqFramesX ?? 1,
    strideTexels: atlas.dqStrideTexels,
    widthTexels: atlas.widthTexels,
    heightTexels: atlas.heightTexels,
    hasScale: atlas.dqHasScale,
    componentType: atlas.componentType,
  };
}

/**
 * An empty frame-packed atlas: `bones` columns in one tile, `framesX` frames per
 * row, as many rows as `frames` needs. `framesX` is clamped so the row fits
 * `maxTextureDimension`; throws if the rows still do not.
 */
export function createPackedDQAtlas(options: {
  bones: number;
  frames: number;
  clips: DQClipInfo[];
  componentType: PackedDQVAT['componentType'];
  framesX?: number;
  maxTextureDimension?: number;
}): PackedDQVAT {
  const max = options.maxTextureDimension ?? DEFAULT_MAX_TEXTURE;
  const perFrame = options.bones * OUT_STRIDE;
  if (perFrame > max) throw new Error(`${options.bones} bones do not fit one ${max}-texel row`);
  const framesX = Math.max(1, Math.min(options.framesX ?? DEFAULT_FRAMES_X, Math.floor(max / perFrame)));
  const widthTexels = framesX * perFrame;
  const heightTexels = Math.max(1, Math.ceil(options.frames / framesX));
  if (heightTexels > max) {
    throw new Error(`${options.frames} frames need ${heightTexels} rows at ${framesX} per row (limit ${max})`);
  }
  const components = widthTexels * heightTexels * 4;
  const pixels = options.componentType === 'float16' ? new Uint16Array(components) : new Float32Array(components);
  // Every slot starts as the identity dual quaternion (qr = 0,0,0,1; qd = 0), so
  // a bone a clip's source never carried skins in place instead of collapsing a
  // zero quaternion to the origin.
  const one = options.componentType === 'float16' ? 0x3c00 : 1;
  for (let texel = 0; texel < components / 4; texel += OUT_STRIDE) pixels[texel * 4 + 3] = one;
  return {
    componentType: options.componentType,
    widthTexels,
    heightTexels,
    framesTotal: options.frames,
    bones: options.bones,
    dqWidthBones: options.bones,
    dqTilesX: 1,
    dqFramesX: framesX,
    dqStrideTexels: OUT_STRIDE,
    dqHasScale: false,
    clips: options.clips,
    pixels,
  };
}

/** Copy one bone's dual quaternion (8 stored components) between atlases. */
function copyDQ(
  from: PackedDQVAT, fromLayout: SvatLayout, fromFrame: number, fromBone: number,
  to: PackedDQVAT, toLayout: SvatLayout, toFrame: number, toBone: number,
): void {
  for (let slot = 0; slot < 2; slot++) {
    const src = svatComponentIndex(fromLayout, fromFrame, fromBone, slot, 0);
    const dst = svatComponentIndex(toLayout, toFrame, toBone, slot, 0);
    for (let c = 0; c < 4; c++) to.pixels[dst + c] = from.pixels[src + c];
  }
}

function readComponent(atlas: PackedDQVAT, index: number): number {
  return atlas.pixels instanceof Uint16Array ? halfToFloat(atlas.pixels[index]) : atlas.pixels[index];
}

/** Largest difference between two bones' dual quaternions, up to sign. */
export function dqDifference(
  a: PackedDQVAT, aLayout: SvatLayout, aFrame: number, aBone: number,
  b: PackedDQVAT, bLayout: SvatLayout, bFrame: number, bBone: number,
): number {
  let same = 0;
  let negated = 0;
  for (let slot = 0; slot < 2; slot++) {
    const ia = svatComponentIndex(aLayout, aFrame, aBone, slot, 0);
    const ib = svatComponentIndex(bLayout, bFrame, bBone, slot, 0);
    for (let c = 0; c < 4; c++) {
      const x = readComponent(a, ia + c);
      const y = readComponent(b, ib + c);
      same = Math.max(same, Math.abs(x - y));
      negated = Math.max(negated, Math.abs(x + y));
    }
  }
  return Math.min(same, negated);
}

function columnsOf(body: DQLibraryBody): number[] {
  const all = [...(body.skinnedColumns ?? body.bones.keys())];
  return all.filter((column) => column >= 0 && column < body.bones.length);
}

/** Refuse to drop a scale texel that carries anything but 1. */
function assertNarrowable(body: DQLibraryBody, layout: SvatLayout, columns: readonly number[]): void {
  if (!body.atlas.dqHasScale || body.atlas.dqStrideTexels < 3) return;
  for (let frame = 0; frame < body.atlas.framesTotal; frame++) {
    for (const bone of columns) {
      const scale = readComponent(body.atlas, svatComponentIndex(layout, frame, bone, 2, 0));
      if (Math.abs(scale - 1) > 1e-3) {
        throw new Error(`${body.key}: ${body.bones[bone]} is scaled ${scale} at frame ${frame}; a library atlas carries no scale`);
      }
    }
  }
}

/** Bake one shared library from a family of bodies on the same skeleton. */
export function buildDQLibrary(bodies: readonly DQLibraryBody[], options: DQLibraryOptions = {}): DQLibrary {
  if (!bodies.length) throw new Error('buildDQLibrary needs at least one body');
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const reference = bodies.find((body) => body.key === (options.reference ?? bodies[0].key));
  if (!reference) throw new Error(`no body ${options.reference}`);
  const componentType = reference.atlas.componentType;

  const prepared = bodies.map((body) => {
    if (body.atlas.componentType !== componentType) throw new Error(`${body.key}: mixed component types`);
    if (body.bones.length !== body.atlas.bones) {
      throw new Error(`${body.key}: ${body.bones.length} bone names for ${body.atlas.bones} atlas columns`);
    }
    const layout = svatLayoutOf(body.atlas);
    const columns = columnsOf(body);
    assertNarrowable(body, layout, columns);
    return { body, layout, columns, clips: new Map(body.atlas.clips.map((clip) => [clip.name, clip])) };
  });
  const ordered = [prepared.find((p) => p.body === reference)!, ...prepared.filter((p) => p.body !== reference)];

  // Union bone order: the reference body's, then every other body's extras.
  const union = [...reference.bones];
  for (const { body } of ordered) for (const name of body.bones) if (!union.includes(name)) union.push(name);
  const unionColumn = new Map(union.map((name, column) => [name, column]));
  const toUnion = new Map(prepared.map((p) => [p, p.body.bones.map((name) => unionColumn.get(name)!)]));

  // Library clip set: every clip once, taken from the first body (reference
  // first) that bakes it.
  const names: string[] = [];
  for (const p of ordered) for (const name of p.clips.keys()) if (!names.includes(name)) names.push(name);
  const owner = new Map(names.map((name) => [name, ordered.find((p) => p.clips.has(name))!]));

  // Which body clips differ from the library copy on that body's skinned bones.
  const own = new Map(prepared.map((p) => [p, [] as string[]]));
  for (const p of prepared) {
    for (const [name, clip] of p.clips) {
      const source = owner.get(name)!;
      if (source === p) continue;
      const libClip = source.clips.get(name)!;
      let same = libClip.to - libClip.from === clip.to - clip.from;
      const sourceColumn = new Map(source.body.bones.map((bone, column) => [bone, column]));
      for (const column of p.columns) {
        if (!same) break;
        const sc = sourceColumn.get(p.body.bones[column]);
        if (sc === undefined) { same = false; break; }
        for (let f = 0; same && f <= clip.to - clip.from; f++) {
          same = dqDifference(source.body.atlas, source.layout, libClip.from + f, sc, p.body.atlas, p.layout, clip.from + f, column) <= tolerance;
        }
      }
      if (!same) own.get(p)!.push(name);
    }
  }

  // Lay out an atlas of the given clips in union space.
  const assemble = (entries: Array<{ name: string; from: typeof prepared[number]; clip: DQClipInfo }>): PackedDQVAT => {
    const clips: DQClipInfo[] = [];
    let frames = 0;
    for (const { name, clip } of entries) {
      const count = clip.to - clip.from + 1;
      clips.push({ name, from: frames, to: frames + count - 1, frames: count, fps: clip.fps });
      frames += count;
    }
    const atlas = createPackedDQAtlas({
      bones: union.length, frames, clips, componentType,
      framesX: options.framesX, maxTextureDimension: options.maxTextureDimension,
    });
    const layout = svatLayoutOf(atlas);
    entries.forEach(({ from, clip }, i) => {
      const map = toUnion.get(from)!;
      for (let f = 0; f <= clip.to - clip.from; f++) {
        for (let column = 0; column < from.body.bones.length; column++) {
          copyDQ(from.body.atlas, from.layout, clip.from + f, column, atlas, layout, clips[i].from + f, map[column]);
        }
      }
    });
    return atlas;
  };

  const atlas = assemble(names.map((name) => ({ name, from: owner.get(name)!, clip: owner.get(name)!.clips.get(name)! })));
  const libraryRange = new Map(atlas.clips.map((clip) => [clip.name, clip]));
  const overrides: Record<string, PackedDQVAT | null> = {};
  const manifests: Record<string, DQLibraryManifest> = {};
  for (const p of prepared) {
    const ownNames = own.get(p)!;
    const override = ownNames.length ? assemble(ownNames.map((name) => ({ name, from: p, clip: p.clips.get(name)! }))) : null;
    const ownRange = new Map((override?.clips ?? []).map((clip) => [clip.name, clip]));
    overrides[p.body.key] = override;
    manifests[p.body.key] = {
      key: p.body.key,
      boneToUnion: toUnion.get(p)!,
      overrideClips: ownNames,
      clips: Object.fromEntries([...p.clips.keys()].map((name) => {
        const clip = ownRange.get(name) ?? libraryRange.get(name)!;
        return [name, { source: ownRange.has(name) ? 'override' : 'library', from: clip.from, to: clip.to, fps: clip.fps }];
      })),
    };
  }
  return {
    bones: union,
    atlas,
    clipSources: Object.fromEntries(names.map((name) => [name, owner.get(name)!.body.key])),
    overrides,
    manifests,
  };
}

/** One clip of an assembled body: where its frames come from. */
export type DQBodyPlanClip = {
  name: string;
  source: 'library' | 'override';
  /** First frame of the clip in the source atlas. */
  from: number;
  frames: number;
  fps: number;
};

/**
 * Assemble a body's own atlas (its bone order) from a library atlas and its
 * override atlas, following `plan` — clips laid out contiguously in plan order.
 * This is the runtime half of the library: it needs only the two atlases, the
 * body's bone map and its plan, not the whole {@link DQLibrary}.
 */
export function assembleDQBody(
  libraryAtlas: PackedDQVAT,
  overrideAtlas: PackedDQVAT | null,
  boneToUnion: readonly number[],
  plan: readonly DQBodyPlanClip[],
  options: { framesX?: number; maxTextureDimension?: number } = {},
): PackedDQVAT {
  const clips: DQClipInfo[] = [];
  let frames = 0;
  for (const clip of plan) {
    clips.push({ name: clip.name, from: frames, to: frames + clip.frames - 1, frames: clip.frames, fps: clip.fps });
    frames += clip.frames;
  }
  const atlas = createPackedDQAtlas({
    bones: boneToUnion.length, frames, clips, componentType: libraryAtlas.componentType,
    framesX: options.framesX ?? libraryAtlas.dqFramesX, maxTextureDimension: options.maxTextureDimension,
  });
  const layout = svatLayoutOf(atlas);
  const libraryLayout = svatLayoutOf(libraryAtlas);
  const overrideLayout = overrideAtlas ? svatLayoutOf(overrideAtlas) : null;
  plan.forEach((clip, i) => {
    const source = clip.source === 'override' ? overrideAtlas : libraryAtlas;
    const sourceLayout = clip.source === 'override' ? overrideLayout : libraryLayout;
    if (!source || !sourceLayout) throw new Error(`${clip.name} needs an override atlas`);
    if (clip.from + clip.frames > source.framesTotal) throw new Error(`${clip.name} runs past its source atlas`);
    for (let f = 0; f < clip.frames; f++) {
      for (let column = 0; column < boneToUnion.length; column++) {
        copyDQ(source, sourceLayout, clip.from + f, boneToUnion[column], atlas, layout, clips[i].from + f, column);
      }
    }
  });
  return atlas;
}

/**
 * A body's assembly plan from a library: its own clips in `bodyClips` order
 * (which must be contiguous from frame 0, as a baked atlas's are) and — with
 * `fill` — every library clip it never baked, appended. Restricting the fill to
 * clips whose source body carries every bone in `requiredBones` keeps a filled
 * clip from driving a skinned bone the source never animated.
 */
export function planBodyFromLibrary(
  library: DQLibrary,
  key: string,
  bodyClips: readonly DQClipInfo[],
  options: { fill?: boolean; requiredBones?: Iterable<string>; bodyBones?: Record<string, readonly string[]> } = {},
): { plan: DQBodyPlanClip[]; filled: string[]; skipped: string[] } {
  const manifest = library.manifests[key];
  if (!manifest) throw new Error(`library has no body ${key}`);
  const plan: DQBodyPlanClip[] = [];
  let expected = 0;
  for (const clip of [...bodyClips].sort((a, b) => a.from - b.from)) {
    if (clip.from !== expected) throw new Error(`${key}: clip ${clip.name} starts at ${clip.from}, expected ${expected}`);
    const ref = manifest.clips[clip.name];
    if (!ref) throw new Error(`${key}: ${clip.name} is not in its manifest`);
    plan.push({ name: clip.name, source: ref.source, from: ref.from, frames: clip.to - clip.from + 1, fps: clip.fps });
    expected = clip.to + 1;
  }
  const filled: string[] = [];
  const skipped: string[] = [];
  if (options.fill) {
    const required = [...(options.requiredBones ?? [])];
    for (const clip of library.atlas.clips) {
      if (manifest.clips[clip.name]) continue;
      const sourceBones = options.bodyBones?.[library.clipSources[clip.name]];
      if (required.length && sourceBones && !required.every((bone) => sourceBones.includes(bone))) {
        skipped.push(clip.name);
        continue;
      }
      plan.push({ name: clip.name, source: 'library', from: clip.from, frames: clip.frames, fps: clip.fps });
      filled.push(clip.name);
    }
  }
  return { plan, filled, skipped };
}

/**
 * Reassemble one body's own atlas (its bone order) from a library: its clip
 * table in its original order and numbering, then — with `fill` — every library
 * clip it never baked, appended. Frame-packed.
 *
 * `bodyClips` is the body's original clip table (the frame numbering held items
 * and indexes refer to); every clip in it must be in the body's manifest.
 */
export function rebuildBodyFromLibrary(
  library: DQLibrary,
  key: string,
  bodyClips: readonly DQClipInfo[],
  options: DQLibraryRebuildOptions = {},
): DQLibraryRebuild {
  const { plan, filled } = planBodyFromLibrary(library, key, bodyClips, { fill: options.fill });
  const atlas = assembleDQBody(library.atlas, library.overrides[key], library.manifests[key].boneToUnion, plan, {
    framesX: options.framesX, maxTextureDimension: options.maxTextureDimension,
  });
  return { atlas, filled };
}

/**
 * Compare two atlases of the same body clip by clip (matched by name) on the
 * given columns. Returns the largest sign-resolved difference and how many
 * bone-frames were checked; clips of different length count as a mismatch.
 */
export function compareDQAtlases(
  a: PackedDQVAT,
  b: PackedDQVAT,
  options: { columns?: Iterable<number>; clips?: Iterable<string> } = {},
): { worst: number; checked: number; mismatchedLengths: string[] } {
  const al = svatLayoutOf(a);
  const bl = svatLayoutOf(b);
  const bClips = new Map(b.clips.map((clip) => [clip.name, clip]));
  const wanted = options.clips ? new Set(options.clips) : null;
  const columns = [...(options.columns ?? Array.from({ length: Math.min(a.bones, b.bones) }, (_, i) => i))];
  let worst = 0;
  let checked = 0;
  const mismatchedLengths: string[] = [];
  for (const clip of a.clips) {
    if (wanted && !wanted.has(clip.name)) continue;
    const other = bClips.get(clip.name);
    if (!other) continue;
    if (other.to - other.from !== clip.to - clip.from) { mismatchedLengths.push(clip.name); continue; }
    for (let f = 0; f <= clip.to - clip.from; f++) {
      for (const column of columns) {
        worst = Math.max(worst, dqDifference(a, al, clip.from + f, column, b, bl, other.from + f, column));
        checked++;
      }
    }
  }
  return { worst, checked, mismatchedLengths };
}
