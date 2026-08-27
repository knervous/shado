import { describe, expect, it } from '@jest/globals';
import { NullEngine } from '@babylonjs/core';
import { Shado } from '../src/core/Shado';
import { field, gpuStruct } from '../src/decorators';

/**
 * `getVarArrayCount` divided by `floatStride | 1`, which reads as a default but
 * is a bitwise OR: every even stride came back one too large, so a vec4 array
 * reported 4/5ths of its real element count.
 *
 * The count is not cosmetic. `writeVarArrayRange` compares the write index
 * against it, concludes the array is too short, and calls `resizeVarArray` -
 * which shrinks `lenF` and zero-fills everything above it. Writing past 80% of
 * a vec4 array therefore wiped its tail, and the write itself did not land.
 */
@gpuStruct({ name: 'VarStrideProbe' })
class VarStrideProbe extends Shado {
  @field({ arrayOf: 'vec4' }) quads!: Float32Array;
  @field({ arrayOf: 'vec2' }) pairs!: Float32Array;
  @field({ arrayOf: 'f32' }) scalars!: Float32Array;
}

describe('variable array element counts', () => {
  let engine: NullEngine;

  beforeEach(() => {
    engine = new NullEngine();
  });

  afterEach(() => {
    engine.dispose();
  });

  it('reports whole element counts for every stride, odd and even', async () => {
    await VarStrideProbe.initialize(engine, { backend: 'datatex', wasm: false });
    const probe = new VarStrideProbe(engine);

    for (const [field, count] of [
      ['quads', 1000],
      ['pairs', 1000],
      ['scalars', 1000],
    ] as const) {
      probe.resizeVarArray(field, count);
      const reported = probe.getVarArrayCount(field);
      expect(Number.isInteger(reported)).toBe(true);
      expect(reported).toBe(count);
    }
  });

  it('keeps writes near the end of a vec4 array instead of wiping the tail', async () => {
    await VarStrideProbe.initialize(engine, { backend: 'datatex', wasm: false });
    const probe = new VarStrideProbe(engine);

    const count = 500;
    probe.resizeVarArray('quads', count);
    // Fill every element with a value derived from its index.
    for (let index = 0; index < count; index++) {
      probe.writeVarArrayRange('quads', index, [index + 1, 1, 1, 1]);
    }

    expect(probe.getVarArrayCount('quads')).toBe(count);

    // Read the whole plane back. The old bug shrank the array once the write
    // index passed 80% of it, so everything above ~400 came back zeroed.
    const arena = (probe as unknown as { _arena: { f32: Float32Array } })._arena;
    const base = probe.getVarArrayPtr('quads');
    const zeroed: number[] = [];
    for (let index = 0; index < count; index++) {
      if (arena.f32[base + index * 4] !== index + 1) zeroed.push(index);
    }
    expect(zeroed).toEqual([]);
  });
});
