import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import { NullEngine } from '@babylonjs/core';

import { ShadoParticleContainer } from '../src/render/ShadoParticleContainer';
import { SHADO_PARTICLE_EMITTER_FLOATS, SHADO_PARTICLE_FLOATS, ShadoParticleField } from '../src/render/ShadoParticleLayout';
import type { ShadoParticleEmitterSpec } from '../src/render/ShadoParticleEmitters';

const spec = (overrides: Partial<ShadoParticleEmitterSpec> = {}): ShadoParticleEmitterSpec => ({
  emission: { mode: 'rate', rate: 20 },
  shape: { kind: 'omni' },
  power: [1, 1],
  life: [1, 1],
  rampRow: 0,
  layer: 0,
  ...overrides,
});

describe('ShadoParticleContainer', () => {
  let engine: NullEngine;

  beforeAll(async () => {
    engine = new NullEngine();
    expect(await ShadoParticleContainer.initialize(engine, { backend: 'datatex' })).toBe(true);
  });

  afterAll(() => engine.dispose());

  it('writes particles into its own arena and marks exactly those slots dirty', () => {
    const container = new ShadoParticleContainer(engine, { capacity: 64, emitterCapacity: 8, anchorCapacity: 4 });
    const arena = (container as any)._arena;
    arena.consumeDirtyRanges();
    container.startEmitter(spec(), 0);
    const ranges = container.step(0.5);
    expect(ranges).toEqual([{ first: 0, count: 10 }]);
    expect(container.drawCount).toBe(10);

    const seg = (container as any)._structSeg.particles;
    const dirty = arena.consumeDirtyRanges();
    const start = (seg.offF | 0) * 4;
    // Shado's tracker coalesces to pages: the written bytes are covered, and ten particles
    // cost one page of upload rather than the whole 64-slot arena.
    const written = { start, end: start + 10 * SHADO_PARTICLE_FLOATS * 4 };
    expect(dirty).toHaveLength(1);
    expect(dirty[0].start).toBeLessThanOrEqual(written.start);
    expect(dirty[0].end).toBeGreaterThanOrEqual(written.end);
    expect(dirty[0].end - dirty[0].start).toBeLessThanOrEqual(4096);
    // The bytes the upload reads are the bytes the reducer wrote.
    const upload: Float32Array = arena.take();
    expect(upload[(seg.offF | 0) + ShadoParticleField.velocity + 3]).toBe(1);
    expect(upload[(seg.offF | 0) + ShadoParticleField.birth + 3]).toBeCloseTo(0.5);
  });

  it('never repacks while running, so the reducer always writes the uploaded bytes', () => {
    const container = new ShadoParticleContainer(engine, { capacity: 256, emitterCapacity: 8, anchorCapacity: 4 });
    const pointer = container.getStructArrayPtr('particles');
    container.startEmitter(spec({ emission: { mode: 'rate', rate: 1000 } }), 0);
    const anchor = container.acquireAnchor([1, 2, 3]);
    for (let frame = 1; frame <= 600; frame++) {
      container.setAnchor(anchor, [frame, 0, 0]);
      container.step(frame / 60);
    }
    expect(container.getStructArrayPtr('particles')).toBe(pointer);
    expect(container.drawCount).toBe(256);
  });

  it('writes each anchor slot to the element the shader reads, and no further', () => {
    const container = new ShadoParticleContainer(engine, { capacity: 16, emitterCapacity: 2, anchorCapacity: 4 });
    const slots = [container.acquireAnchor([1, 2, 3], 0.5), container.acquireAnchor([4, 5, 6], 1), container.acquireAnchor([7, 8, 9], 1.5)];
    expect(slots).toEqual([0, 1, 2]);
    expect(container.getVarArrayCount('anchors')).toBe(4);
    const c = container as any;
    const anchors = new Float32Array(c._arena.f32.buffer, c._arena.f32.byteOffset + c._varSeg.anchors.offF * 4, 16);
    // ShadoParticleContainer_anchors_get(i) reads element i: floats 4i..4i+3.
    expect(Array.from(anchors)).toEqual([1, 2, 3, 0.5, 4, 5, 6, 1, 7, 8, 9, 1.5, 0, 0, 0, 0]);
    container.releaseAnchor(1);
    expect(Array.from(anchors.subarray(4, 8))).toEqual([0, 0, 0, 0]);
    expect(Array.from(anchors.subarray(8, 12))).toEqual([7, 8, 9, 1.5]);
  });

  it('accounts for emitter slots across whole trees and gives them back', () => {
    const container = new ShadoParticleContainer(engine, { capacity: 32, emitterCapacity: 3, anchorCapacity: 1 });
    const child = { spec: spec({ emission: { mode: 'burst', count: 2 }, attach: 'onDeath' as const }) };
    const handle = container.startEmitter(spec(), 0, [child, child]);
    expect(handle?.children).toHaveLength(2);
    expect(container.freeEmitterSlots).toBe(0);
    expect(container.startEmitter(spec(), 0)).toBeNull();

    const table = (container as any).reducer.emitterView() as Float32Array;
    const root = handle!.index * SHADO_PARTICLE_EMITTER_FLOATS;
    expect(table[root + 42]).toBe(handle!.children[0].index);
    expect(table[handle!.children[0].index * SHADO_PARTICLE_EMITTER_FLOATS + 43]).toBe(handle!.children[1].index);

    container.stopEmitter(handle!);
    expect(table[root]).toBe(2);
    container.releaseEmitter(handle!);
    expect(container.freeEmitterSlots).toBe(3);
    expect(container.acquireAnchor()).toBe(0);
    expect(container.acquireAnchor()).toBe(-1);
  });
});
