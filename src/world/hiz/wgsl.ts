/**
 * WGSL for the runtime Hi-Z path. Mirrors `./reference.ts`; the Dawn test in
 * tests/world-hiz-gpu.test.ts holds the two in step.
 *
 * The pyramid lives in ONE storage buffer of f32 words (levels back to back,
 * offsets from `shadoHiZLayout`). Each reduction is its own dispatch, reading
 * level L and writing level L+1 of that buffer; dispatches are separate compute
 * passes, so every level is complete before the next reads it. A buffer keeps
 * integer addressing exact and needs no float-filterable/storage-texture
 * format support, sampler or per-level binding.
 */
import {
  SHADO_HIZ_DEPTH_BIAS,
  SHADO_HIZ_MAX_LEVELS,
  SHADO_HIZ_MAX_TEXELS_PER_AXIS,
  SHADO_HIZ_PIXEL_EXPANSION,
} from './types';

export const SHADO_HIZ_WORKGROUP_2D = 8;
export const SHADO_HIZ_WORKGROUP_1D = 64;

/** Words per candidate record: (min.xyz, batch) (max.xyz, member). */
export const SHADO_HIZ_CANDIDATE_WORDS = 8;
/** Words per batch record: indexCount, firstIndex, capacity, segmentOffset. */
export const SHADO_HIZ_BATCH_WORDS = 4;
/** Words per indexed-indirect argument block. */
export const SHADO_HIZ_DRAW_ARGS_WORDS = 5;

/**
 * Per-level parameter block for seed/reduce:
 * [srcOffset, srcWidth, srcHeight, dstOffset, dstWidth, dstHeight, clearBits, convention]
 */
export const SHADO_HIZ_LEVEL_PARAM_WORDS = 8;

/**
 * View block for the cull, in f32/u32 words:
 *   0..15  viewProjection (column-major)
 *   16     viewportWidth (u32)   17 viewportHeight (u32)
 *   18     convention (0 normal, 1 reversed)
 *   19     ndcHalfZRange (u32)   20 topLeftOrigin (u32)
 *   21     admitAll (u32): any epoch/viewport mismatch admits everything
 *   22     bias (f32)            23 levelCount (u32)
 *   24     candidateCount (u32)  25..27 pad
 *   28..   levels: (width, height, offset, pad) x SHADO_HIZ_MAX_LEVELS
 */
export const SHADO_HIZ_VIEW_WORDS = 28 + 4 * SHADO_HIZ_MAX_LEVELS;

/** Result flag per candidate, readable for debugging only. */
export const SHADO_HIZ_FLAG_REJECTED = 0;
export const SHADO_HIZ_FLAG_VISIBLE_DEPTH = 1;
/** Flags >= this are `ShadoHiZAdmitReason + SHADO_HIZ_FLAG_ADMIT_BASE`. */
export const SHADO_HIZ_FLAG_ADMIT_BASE = 16;
export const SHADO_HIZ_FLAG_ADMIT_ALL = 15;

/** level-0 depth texture -> pyramid level 0. Unknown samples become far. */
export function emitShadoHiZSeedWGSL(): string {
  return /* wgsl */ `
@group(0) @binding(0) var hizDepth: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> hizPyramid: array<f32>;
@group(0) @binding(2) var<storage, read> hizLevelParams: array<u32>;

@compute @workgroup_size(${SHADO_HIZ_WORKGROUP_2D}, ${SHADO_HIZ_WORKGROUP_2D})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dstOffset = hizLevelParams[3];
  let width = hizLevelParams[4];
  let height = hizLevelParams[5];
  if (id.x >= width || id.y >= height) { return; }
  let clearDepth = bitcast<f32>(hizLevelParams[6]);
  var value = textureLoad(hizDepth, vec2i(id.xy), 0).r;
  // NaN compares false both ways; infinities fail the range test.
  if (!(value >= 0.0 && value <= 1.0)) { value = clearDepth; }
  hizPyramid[dstOffset + id.y * width + id.x] = value;
}`;
}

/** level L -> level L+1, keeping the FARTHEST of up to four children. */
export function emitShadoHiZReduceWGSL(): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> hizPyramid: array<f32>;
@group(0) @binding(1) var<storage, read> hizLevelParams: array<u32>;

fn hizFarther(a: f32, b: f32, reversed: bool) -> f32 {
  return select(max(a, b), min(a, b), reversed);
}

@compute @workgroup_size(${SHADO_HIZ_WORKGROUP_2D}, ${SHADO_HIZ_WORKGROUP_2D})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let srcOffset = hizLevelParams[0];
  let srcWidth = hizLevelParams[1];
  let srcHeight = hizLevelParams[2];
  let dstOffset = hizLevelParams[3];
  let dstWidth = hizLevelParams[4];
  let dstHeight = hizLevelParams[5];
  let reversed = hizLevelParams[7] != 0u;
  if (id.x >= dstWidth || id.y >= dstHeight) { return; }
  let sx0 = id.x * 2u;
  let sy0 = id.y * 2u;
  let sx1 = min(sx0 + 1u, srcWidth - 1u);
  let sy1 = min(sy0 + 1u, srcHeight - 1u);
  var value = hizPyramid[srcOffset + sy0 * srcWidth + sx0];
  value = hizFarther(value, hizPyramid[srcOffset + sy0 * srcWidth + sx1], reversed);
  value = hizFarther(value, hizPyramid[srcOffset + sy1 * srcWidth + sx0], reversed);
  value = hizFarther(value, hizPyramid[srcOffset + sy1 * srcWidth + sx1], reversed);
  hizPyramid[dstOffset + id.y * dstWidth + id.x] = value;
}`;
}

/**
 * Writes every batch's indexed-indirect block with instanceCount 0 and
 * firstInstance 0, and clears its overflow flag. Runs before the cull each
 * frame, so an empty batch still has valid arguments.
 */
export function emitShadoHiZResetWGSL(): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> hizBatches: array<u32>;
@group(0) @binding(1) var<storage, read_write> hizDrawArgs: array<u32>;
@group(0) @binding(2) var<storage, read_write> hizOverflow: array<u32>;

@compute @workgroup_size(${SHADO_HIZ_WORKGROUP_1D})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let batchCount = arrayLength(&hizOverflow);
  let b = id.x;
  if (b >= batchCount) { return; }
  let a = b * ${SHADO_HIZ_DRAW_ARGS_WORDS}u;
  hizDrawArgs[a + 0u] = hizBatches[b * ${SHADO_HIZ_BATCH_WORDS}u + 0u];
  hizDrawArgs[a + 1u] = 0u;
  hizDrawArgs[a + 2u] = hizBatches[b * ${SHADO_HIZ_BATCH_WORDS}u + 1u];
  hizDrawArgs[a + 3u] = 0u;
  hizDrawArgs[a + 4u] = 0u;
  hizOverflow[b] = 0u;
}`;
}

/**
 * One invocation per candidate: conservative projection, level choice, every
 * covered texel, then compaction of visible members into the batch's own
 * fixed segment. The shipping decision never leaves the GPU.
 */
export function emitShadoHiZCullWGSL(): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> hizPyramid: array<f32>;
@group(0) @binding(1) var<storage, read> hizView: array<u32>;
@group(0) @binding(2) var<storage, read> hizCandidates: array<vec4f>;
@group(0) @binding(3) var<storage, read> hizBatches: array<u32>;
@group(0) @binding(4) var<storage, read_write> hizDrawArgs: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> hizVisible: array<u32>;
@group(0) @binding(6) var<storage, read_write> hizFlags: array<u32>;

const MAX_TEXELS: u32 = ${SHADO_HIZ_MAX_TEXELS_PER_AXIS}u;
const EXPAND: i32 = ${SHADO_HIZ_PIXEL_EXPANSION};
const ADMIT_BASE: u32 = ${SHADO_HIZ_FLAG_ADMIT_BASE}u;
const ADMIT_ALL: u32 = ${SHADO_HIZ_FLAG_ADMIT_ALL}u;

fn viewF(i: u32) -> f32 { return bitcast<f32>(hizView[i]); }

fn isFinite4(v: vec4f) -> bool {
  // Finite values satisfy |x| <= f32 max; NaN and inf do not.
  let a = abs(v);
  return all(a <= vec4f(3.4e38));
}

// Returns the flag for this candidate and writes its footprint into rect/nearest.
fn project(minB: vec3f, maxB: vec3f, rect: ptr<function, vec4i>, nearest: ptr<function, f32>) -> u32 {
  let m = mat4x4f(
    viewF(0u), viewF(1u), viewF(2u), viewF(3u),
    viewF(4u), viewF(5u), viewF(6u), viewF(7u),
    viewF(8u), viewF(9u), viewF(10u), viewF(11u),
    viewF(12u), viewF(13u), viewF(14u), viewF(15u));
  let reversed = hizView[18] != 0u;
  let halfZ = hizView[19] != 0u;
  let topLeft = hizView[20] != 0u;
  var ndcMin = vec2f(3.4e38);
  var ndcMax = vec2f(-3.4e38);
  var near = select(3.4e38, -3.4e38, reversed);
  for (var corner = 0u; corner < 8u; corner++) {
    let p = vec3f(
      select(minB.x, maxB.x, (corner & 1u) != 0u),
      select(minB.y, maxB.y, (corner & 2u) != 0u),
      select(minB.z, maxB.z, (corner & 4u) != 0u));
    let clip = m * vec4f(p, 1.0);
    if (!isFinite4(clip)) { return ADMIT_BASE + 2u; }
    if (!(clip.w > 1e-6)) { return ADMIT_BASE + 1u; }
    let ndcZ = clip.z / clip.w;
    let depth = select(ndcZ * 0.5 + 0.5, ndcZ, halfZ);
    if (select((depth < 0.0), (depth > 1.0), reversed)) { return ADMIT_BASE + 1u; }
    let ndc = clip.xy / clip.w;
    ndcMin = min(ndcMin, ndc);
    ndcMax = max(ndcMax, ndc);
    near = select(min(near, depth), max(near, depth), reversed);
  }
  if (ndcMax.x < -1.0 || ndcMin.x > 1.0 || ndcMax.y < -1.0 || ndcMin.y > 1.0) {
    return ADMIT_BASE + 3u;
  }
  let w = f32(hizView[16]);
  let h = f32(hizView[17]);
  let px0 = (ndcMin.x * 0.5 + 0.5) * w;
  let px1 = (ndcMax.x * 0.5 + 0.5) * w;
  let rowA = select(ndcMin.y * 0.5 + 0.5, 0.5 - ndcMax.y * 0.5, topLeft) * h;
  let rowB = select(ndcMax.y * 0.5 + 0.5, 0.5 - ndcMin.y * 0.5, topLeft) * h;
  let hiX = i32(hizView[16]) - 1;
  let hiY = i32(hizView[17]) - 1;
  *rect = vec4i(
    clamp(i32(floor(px0)) - EXPAND, 0, hiX),
    clamp(i32(floor(rowA)) - EXPAND, 0, hiY),
    clamp(i32(floor(px1)) + EXPAND, 0, hiX),
    clamp(i32(floor(rowB)) + EXPAND, 0, hiY));
  *nearest = near;
  return ${SHADO_HIZ_FLAG_VISIBLE_DEPTH}u;
}

fn depthTest(rect: vec4i, nearest: f32) -> u32 {
  let reversed = hizView[18] != 0u;
  let bias = viewF(22u);
  let levelCount = hizView[23];
  let r = vec4u(rect);
  var level = 0u;
  loop {
    if (level + 1u >= levelCount) { break; }
    let spanX = (r.z >> level) - (r.x >> level) + 1u;
    let spanY = (r.w >> level) - (r.y >> level) + 1u;
    if (spanX <= MAX_TEXELS && spanY <= MAX_TEXELS) { break; }
    level++;
  }
  let lw = hizView[28u + level * 4u];
  let lh = hizView[29u + level * 4u];
  let lo = hizView[30u + level * 4u];
  let tx0 = r.x >> level;
  let tx1 = min(r.z >> level, lw - 1u);
  let ty0 = r.y >> level;
  let ty1 = min(r.w >> level, lh - 1u);
  var occluder = select(-3.4e38, 3.4e38, reversed);
  for (var y = ty0; y <= ty1; y++) {
    for (var x = tx0; x <= tx1; x++) {
      let d = hizPyramid[lo + y * lw + x];
      occluder = select(max(occluder, d), min(occluder, d), reversed);
    }
  }
  let hidden = select((nearest > occluder + bias), (nearest < occluder - bias), reversed);
  return select(${SHADO_HIZ_FLAG_VISIBLE_DEPTH}u, 0u, hidden);
}

@compute @workgroup_size(${SHADO_HIZ_WORKGROUP_1D})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x;
  if (index >= hizView[24]) { return; }
  let a = hizCandidates[index * 2u];
  let b = hizCandidates[index * 2u + 1u];
  let batch = bitcast<u32>(a.w);
  let member = bitcast<u32>(b.w);

  var flag = ADMIT_ALL;
  if (hizView[21] == 0u) {
    var rect = vec4i(0);
    var nearest = 0.0;
    flag = project(a.xyz, b.xyz, &rect, &nearest);
    if (flag == ${SHADO_HIZ_FLAG_VISIBLE_DEPTH}u) {
      flag = depthTest(rect, nearest);
    }
  }
  hizFlags[index] = flag;
  if (flag == ${SHADO_HIZ_FLAG_REJECTED}u) { return; }

  let capacity = hizBatches[batch * ${SHADO_HIZ_BATCH_WORDS}u + 2u];
  let segment = hizBatches[batch * ${SHADO_HIZ_BATCH_WORDS}u + 3u];
  let slot = atomicAdd(&hizDrawArgs[batch * ${SHADO_HIZ_DRAW_ARGS_WORDS}u + 1u], 1u);
  // Past capacity the finalize pass switches this batch to an uncompacted
  // draw of every member; nothing is dropped.
  if (slot < capacity) {
    hizVisible[segment + slot] = member;
  }
}`;
}

/**
 * Clamps each batch's instance count to its capacity and records overflow.
 * An overflowed batch draws all `capacity` members through the identity
 * mapping (the vertex path reads hizOverflow), so overflow can only admit.
 */
export function emitShadoHiZFinalizeWGSL(): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> hizBatches: array<u32>;
@group(0) @binding(1) var<storage, read_write> hizDrawArgs: array<u32>;
@group(0) @binding(2) var<storage, read_write> hizOverflow: array<u32>;

@compute @workgroup_size(${SHADO_HIZ_WORKGROUP_1D})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let batchCount = arrayLength(&hizOverflow);
  let b = id.x;
  if (b >= batchCount) { return; }
  let capacity = hizBatches[b * ${SHADO_HIZ_BATCH_WORDS}u + 2u];
  let at = b * ${SHADO_HIZ_DRAW_ARGS_WORDS}u + 1u;
  if (hizDrawArgs[at] > capacity) {
    hizDrawArgs[at] = capacity;
    hizOverflow[b] = 1u;
  }
}`;
}

export const SHADO_HIZ_DEFAULT_BIAS = SHADO_HIZ_DEPTH_BIAS;
