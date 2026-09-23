/**
 * The `.svat` decode hot path: chunk checksum and the fused
 * unshuffle → XOR-delta → scatter that rebuilds atlas order.
 *
 * SELF-CONTAINED BY CONTRACT. Every function here references only its own
 * parameters, locals and built-in globals — no imports, no module constants,
 * no calls to each other. `SvatWorker` splices them into its worker script with
 * `Function.prototype.toString()`, so the main-thread decoder and the worker are
 * the same code by construction. A minifier renames identifiers consistently
 * inside a function, which keeps that safe; an outer reference would not
 * survive, and the worker would throw a ReferenceError on its first chunk.
 * Plain loops only: no spread, destructuring or optional chaining that a
 * downlevelling build could turn into an injected helper call.
 */

/**
 * FNV-1a 32-bit over bytes, identical to the historical byte-at-a-time loop.
 *
 * FNV is a serial dependency chain (xor → multiply per byte), so it cannot be
 * widened to words without changing the value; the cost here is the chain
 * itself (~1.25 ns/byte, ~9 ms per 7 MB body). The state stays an int32 —
 * Math.imul already returns one and XOR with a byte leaves the bit pattern
 * ToUint32 would give — and the loop is unrolled by four: called per ~100 KB
 * chunk the plain loop measured 3.5x slower than the unrolled one (it is only
 * even under a single long OSR'd call). The rest of the saving is running it
 * off the main thread (`decodeSvatInWorker`).
 */
export function svatFnv1a32(bytes: Uint8Array): number {
  let hash = 0x811c9dc5 | 0;
  const length = bytes.length;
  const blockEnd = length & ~3;
  let i = 0;
  for (; i < blockEnd; i += 4) {
    hash = Math.imul(hash ^ bytes[i], 0x01000193);
    hash = Math.imul(hash ^ bytes[i + 1], 0x01000193);
    hash = Math.imul(hash ^ bytes[i + 2], 0x01000193);
    hash = Math.imul(hash ^ bytes[i + 3], 0x01000193);
  }
  for (; i < length; i++) hash = Math.imul(hash ^ bytes[i], 0x01000193);
  return hash >>> 0;
}

/**
 * Decode one filtered chunk straight into the atlas.
 *
 * `bytes` is the decompressed chunk exactly as stored: storage order
 * (bone, slot, component, frame), optionally byte-shuffled into
 * `elementBytes` planes and XOR-delta coded along each frame stream. `target`
 * is an integer view of the atlas (`Uint16Array` for float16, a `Uint32Array`
 * over the Float32Array's buffer for float32) so values move as bit patterns
 * and nothing is ever canonicalised through a JS number.
 *
 * Equivalent to `byteUnshuffle` → `deltaDecode` → `scatterChunk`, fused into
 * one pass with no intermediate buffers. Atlas addressing is separable —
 * `svatComponentIndex(frame, bone, slot, c)` = frameOffset(frame) +
 * texelOffset(bone, slot) + c — so the per-frame part is tabulated once per
 * chunk and the inner loop is adds and loads. It walks one texel at a time,
 * frame by frame, reading the four component streams in step and writing each
 * frame's four components as one contiguous texel.
 */
export function svatDecodeChunkInto(
  target: Uint16Array | Uint32Array,
  bytes: Uint8Array,
  elementBytes: number,
  shuffled: boolean,
  delta: boolean,
  layout: {
    bones: number;
    strideTexels: number;
    widthBones: number;
    tilesX: number;
    framesX: number;
    widthTexels: number;
  },
  frameStart: number,
  frameCount: number
): void {
  const bones = layout.bones;
  const stride = layout.strideTexels;
  const widthBones = layout.widthBones;
  const widthTexels = layout.widthTexels;
  const tilesX = layout.tilesX;
  const framesX = layout.framesX > 0 ? layout.framesX : 1;
  const frames = frameCount;

  const frameOffset = new Int32Array(frames);
  for (let f = 0; f < frames; f++) {
    const frame = frameStart + f;
    const column = frame % framesX;
    const row = (frame / framesX) | 0;
    frameOffset[f] = (row * tilesX * widthTexels + column * widthBones * stride) * 4;
  }

  const count = (bytes.length / elementBytes) | 0;
  // Element e's byte k sits at e*elementStride + k*planeStride.
  const elementStride = shuffled ? 1 : elementBytes;
  const planeStride = shuffled ? count : 1;
  const p1 = planeStride;
  const p2 = planeStride * 2;
  const p3 = planeStride * 3;
  // With delta off the running predictor is masked to zero.
  const keep = delta ? -1 : 0;

  let stream = 0; // storage index of (texel, component 0, frame 0)
  for (let bone = 0; bone < bones; bone++) {
    const tile = (bone / widthBones) | 0;
    const xBone = bone % widthBones;
    for (let slot = 0; slot < stride; slot++) {
      const texel = (tile * widthTexels + xBone * stride + slot) * 4;
      let s0 = stream * elementStride;
      let s1 = (stream + frames) * elementStride;
      let s2 = (stream + frames * 2) * elementStride;
      let s3 = (stream + frames * 3) * elementStride;
      let q0 = 0;
      let q1 = 0;
      let q2 = 0;
      let q3 = 0;
      if (elementBytes === 2) {
        for (let f = 0; f < frames; f++) {
          const o = frameOffset[f] + texel;
          const w0 = (bytes[s0] | (bytes[s0 + p1] << 8)) ^ q0;
          const w1 = (bytes[s1] | (bytes[s1 + p1] << 8)) ^ q1;
          const w2 = (bytes[s2] | (bytes[s2 + p1] << 8)) ^ q2;
          const w3 = (bytes[s3] | (bytes[s3 + p1] << 8)) ^ q3;
          target[o] = w0;
          target[o + 1] = w1;
          target[o + 2] = w2;
          target[o + 3] = w3;
          q0 = w0 & keep;
          q1 = w1 & keep;
          q2 = w2 & keep;
          q3 = w3 & keep;
          s0 += elementStride;
          s1 += elementStride;
          s2 += elementStride;
          s3 += elementStride;
        }
      } else {
        for (let f = 0; f < frames; f++) {
          const o = frameOffset[f] + texel;
          const w0 =
            (bytes[s0] | (bytes[s0 + p1] << 8) | (bytes[s0 + p2] << 16) | (bytes[s0 + p3] << 24)) ^ q0;
          const w1 =
            (bytes[s1] | (bytes[s1 + p1] << 8) | (bytes[s1 + p2] << 16) | (bytes[s1 + p3] << 24)) ^ q1;
          const w2 =
            (bytes[s2] | (bytes[s2 + p1] << 8) | (bytes[s2 + p2] << 16) | (bytes[s2 + p3] << 24)) ^ q2;
          const w3 =
            (bytes[s3] | (bytes[s3 + p1] << 8) | (bytes[s3 + p2] << 16) | (bytes[s3 + p3] << 24)) ^ q3;
          // Negative int32 stores wrap to the same u32 bit pattern.
          target[o] = w0;
          target[o + 1] = w1;
          target[o + 2] = w2;
          target[o + 3] = w3;
          q0 = w0 & keep;
          q1 = w1 & keep;
          q2 = w2 & keep;
          q3 = w3 & keep;
          s0 += elementStride;
          s1 += elementStride;
          s2 += elementStride;
          s3 += elementStride;
        }
      }
      stream += frames * 4;
    }
  }
}
