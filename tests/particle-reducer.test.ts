import { describe, expect, it } from '@jest/globals';
import { readFileSync } from 'node:fs';
import path from 'node:path';

import {
  SHADO_PARTICLE_EMITTER_FLOATS,
  SHADO_PARTICLE_FLOATS,
  ShadoParticleEmitterField,
  ShadoParticleField,
} from '../src/render/ShadoParticleLayout';
import { createShadoParticleReducer } from '../src/render/ShadoParticleReducer';
import {
  encodeShadoParticleEmitter,
  sampleShadoRamp,
  ShadoParticleRamps,
  type ShadoParticleEmitterSpec,
} from '../src/render/ShadoParticleEmitters';

const P = ShadoParticleField;

const baseSpec = (overrides: Partial<ShadoParticleEmitterSpec> = {}): ShadoParticleEmitterSpec => ({
  emission: { mode: 'rate', rate: 10 },
  shape: { kind: 'directional', dir: [0, 1, 0] },
  power: [2, 2],
  life: [1, 1],
  rampRow: 4,
  layer: 3,
  ...overrides,
});

async function arena(capacity = 32, emitters = 8) {
  const reducer = await createShadoParticleReducer();
  const particlePtr = reducer.exports.alloc(capacity * SHADO_PARTICLE_FLOATS * 4);
  reducer.initArena({ particlePtr, particleCapacity: capacity, emitterCapacity: emitters, pendingCapacity: 16, trailCapacity: 8, seed: 7 });
  return reducer;
}

const record = (view: Float32Array, slot: number, field: number, lane = 0) => view[slot * SHADO_PARTICLE_FLOATS + field + lane];

describe('particle layout', () => {
  it('matches the assembly reducer constant for constant', () => {
    const source = readFileSync(path.join(process.cwd(), 'assembly/particle-reducer.ts'), 'utf8');
    const constant = (name: string) => {
      const match = new RegExp(`const ${name}: i32 = (\\d+);`).exec(source);
      if (!match) throw new Error(`no ${name} in the reducer`);
      return Number(match[1]);
    };
    expect(constant('PARTICLE_FLOATS')).toBe(SHADO_PARTICLE_FLOATS);
    expect(constant('EMITTER_FLOATS')).toBe(SHADO_PARTICLE_EMITTER_FLOATS);
    const particle = { P_BIRTH: 'birth', P_VELOCITY: 'velocity', P_ACCEL: 'accel', P_LOOK: 'look', P_COLLISION: 'collision', P_EXTRA: 'extra' } as const;
    for (const [name, key] of Object.entries(particle)) expect(constant(name)).toBe(ShadoParticleField[key]);
    for (const [key, value] of Object.entries(ShadoParticleEmitterField)) {
      const name = `E_${key.replace(/[A-Z]/g, c => `_${c}`).toUpperCase()}`;
      expect([name, constant(name)]).toEqual([name, value]);
    }
  });
});

describe('particle reducer', () => {
  it('emits at its rate, writing records in place, and reports the written range', async () => {
    const reducer = await arena();
    encodeShadoParticleEmitter(reducer.emitterView(), 0, baseSpec(), 0);
    expect(reducer.step(0.05)).toEqual([]);
    const ranges = reducer.step(0.25);
    expect(ranges).toEqual([{ first: 0, count: 2 }]);
    const view = reducer.particleView();
    expect(record(view, 0, P.birth, 3)).toBeCloseTo(0.25);
    expect(record(view, 0, P.velocity, 1)).toBeCloseTo(2);
    expect(record(view, 0, P.velocity, 3)).toBeCloseTo(1);
    expect(record(view, 0, P.look, 0)).toBe(4);
    expect(record(view, 0, P.look, 1)).toBe(3);
    expect(record(view, 0, P.extra, 2)).toBe(1);
    expect(reducer.step(1.25)).toEqual([{ first: 2, count: 10 }]);
    expect(reducer.spawnedTotal).toBe(12);
  });

  it('bursts once after its delay, and stops a rate emitter after its window', async () => {
    const reducer = await arena();
    const table = reducer.emitterView();
    encodeShadoParticleEmitter(table, 0, baseSpec({ emission: { mode: 'burst', count: 5 }, delay: 0.5 }), 0);
    encodeShadoParticleEmitter(table, 1, baseSpec({ emission: { mode: 'rate', rate: 4, stopAfter: 0.5 } }), 0);
    expect(reducer.step(0.25).reduce((n, r) => n + r.count, 0)).toBe(1);
    expect(reducer.step(0.75).reduce((n, r) => n + r.count, 0)).toBe(5 + 1);
    expect(table[1 * SHADO_PARTICLE_EMITTER_FLOATS + ShadoParticleEmitterField.state]).toBe(2);
    expect(reducer.step(5)).toEqual([]);
  });

  it('owes primed particles at the start, so a single short emission is never lost to rounding', async () => {
    const reducer = await arena();
    const table = reducer.emitterView();
    // One emission of one particle every 0.02s, stopping half a particle after the first.
    encodeShadoParticleEmitter(table, 0, baseSpec({ emission: { mode: 'rate', rate: 50, stopAfter: 0.01, prime: 1 } }), 0);
    let spawned = 0;
    for (let t = 1 / 60; t < 0.5; t += 1 / 60) spawned += reducer.step(t).reduce((n, r) => n + r.count, 0);
    expect(spawned).toBe(1);
  });

  it('wraps the ring, overwriting the oldest particles, in at most two ranges', async () => {
    const reducer = await arena(8);
    encodeShadoParticleEmitter(reducer.emitterView(), 0, baseSpec({ emission: { mode: 'rate', rate: 6 } }), 0);
    expect(reducer.step(1)).toEqual([{ first: 0, count: 6 }]);
    expect(reducer.step(2)).toEqual([{ first: 6, count: 2 }, { first: 0, count: 4 }]);
    expect(reducer.step(10)).toEqual([{ first: 0, count: 8 }]);
  });

  it('spawns on-death children at the parent\'s closed-form death position', async () => {
    const reducer = await arena(64);
    const table = reducer.emitterView();
    // Parent: one particle up at 2 m/s for 1 s under gravity 2 -> y = 2 - 1 = 1 at death.
    encodeShadoParticleEmitter(table, 1, baseSpec({ emission: { mode: 'burst', count: 3 }, attach: 'onDeath', life: [0.5, 0.5] }), 0, {
      parent: 0,
      firstChild: -1,
      nextSibling: -1,
    });
    encodeShadoParticleEmitter(table, 0, baseSpec({ emission: { mode: 'burst', count: 1 }, gravity: [0, -2, 0] }), 0, {
      parent: -1,
      firstChild: 1,
      nextSibling: -1,
    });
    reducer.step(0);
    expect(reducer.exports.getPendingCount()).toBe(1);
    expect(reducer.step(0.5)).toEqual([]);
    const ranges = reducer.step(1.01);
    expect(ranges).toEqual([{ first: 1, count: 3 }]);
    const view = reducer.particleView();
    expect(record(view, 1, P.birth, 1)).toBeCloseTo(1, 4);
    expect(reducer.exports.getPendingCount()).toBe(0);
  });

  it('spawns on-birth children in the same step, where the parent is born', async () => {
    const reducer = await arena(64);
    const table = reducer.emitterView();
    encodeShadoParticleEmitter(table, 1, baseSpec({ emission: { mode: 'burst', count: 4 }, attach: 'onBirth', power: [0, 0] }), 0, {
      parent: 0,
      firstChild: -1,
      nextSibling: -1,
    });
    encodeShadoParticleEmitter(table, 0, baseSpec({ emission: { mode: 'burst', count: 2 }, origin: [0, 5, 0], power: [0, 0] }), 0, {
      parent: -1,
      firstChild: 1,
      nextSibling: -1,
    });
    expect(reducer.step(0)).toEqual([{ first: 0, count: 2 + 2 * 4 }]);
    expect(reducer.exports.getPendingCount()).toBe(0);
    const view = reducer.particleView();
    for (let slot = 0; slot < 10; slot++) expect(view[slot * 24 + 1]).toBeCloseTo(5);
  });

  it('ends a destroyed particle on its ground plane, so what it spawns on death starts there', async () => {
    const reducer = await arena(64);
    const table = reducer.emitterView();
    encodeShadoParticleEmitter(table, 1, baseSpec({ emission: { mode: 'burst', count: 1 }, power: [0, 0], life: [0.5, 0.5] }), 0, {
      parent: 0, firstChild: -1, nextSibling: -1,
    });
    // Born 2 m up at rest under g = -4: lands after 1 s, well inside its 5 s life.
    encodeShadoParticleEmitter(table, 0, baseSpec({
      emission: { mode: 'burst', count: 1 }, power: [0, 0], life: [5, 5], origin: [0, 2, 0], gravity: [0, -4, 0],
      collision: { mode: 'destroy', ground: 0 },
    }), 0, { parent: -1, firstChild: 1, nextSibling: -1 });
    reducer.step(0);
    const view = reducer.particleView();
    expect(record(view, 0, P.velocity, 3)).toBeCloseTo(1, 5);
    expect(record(view, 0, P.collision, 3)).toBe(-1);
    reducer.step(1.01);
    expect(record(view, 1, P.birth, 1)).toBeCloseTo(0, 4);
  });

  it('records when a bouncing particle lands, and emits on-bounce children there with the bounced velocity', async () => {
    const reducer = await arena(64);
    const table = reducer.emitterView();
    encodeShadoParticleEmitter(table, 1, baseSpec({ emission: { mode: 'burst', count: 2 }, attach: 'onBounce', power: [0, 0], inheritSpeed: 1 }), 0, {
      parent: 0, firstChild: -1, nextSibling: -1,
    });
    // Thrown up at 2 m/s from 1.5 m under g = -4: -2t^2 + 2t + 1.5 = 0 lands at t = 1.5 s.
    encodeShadoParticleEmitter(table, 0, baseSpec({
      emission: { mode: 'burst', count: 1 }, power: [2, 2], life: [5, 5], origin: [0, 1.5, 0], gravity: [0, -4, 0],
      collision: { mode: 'bounce', ground: 0, restitution: 0.5 },
    }), 0, { parent: -1, firstChild: 1, nextSibling: -1 });
    reducer.step(0);
    const landing = 1.5;
    const view = reducer.particleView();
    expect(record(view, 0, P.collision, 0)).toBe(0);
    expect(record(view, 0, P.collision, 1)).toBeCloseTo(0.5);
    expect(record(view, 0, P.collision, 2)).toBe(1);
    expect(record(view, 0, P.collision, 3)).toBeCloseTo(landing, 4);
    expect(reducer.exports.getPendingCount()).toBe(1);
    reducer.step(landing + 0.01);
    // Landing speed is 2 - 4t (downward); bounced up at half of it.
    expect(record(view, 1, P.birth, 1)).toBeCloseTo(0, 4);
    expect(record(view, 1, P.velocity, 1)).toBeCloseTo(-(2 - 4 * landing) * 0.5, 3);
  });

  it('never collides a particle born on or under its plane, or one that never comes down', async () => {
    const reducer = await arena(64);
    const table = reducer.emitterView();
    encodeShadoParticleEmitter(table, 0, baseSpec({ emission: { mode: 'burst', count: 1 }, power: [3, 3], life: [2, 2], collision: { mode: 'bounce', ground: 0 } }), 0);
    encodeShadoParticleEmitter(table, 1, baseSpec({ emission: { mode: 'burst', count: 1 }, power: [3, 3], life: [2, 2], origin: [0, 1, 0], collision: { mode: 'bounce', ground: 0 } }), 0);
    reducer.step(0);
    const view = reducer.particleView();
    expect(record(view, 0, P.collision, 3)).toBe(-1);
    expect(record(view, 1, P.collision, 3)).toBe(-1);
  });

  it('emits trail children along the parent\'s path while it lives', async () => {
    const reducer = await arena(64);
    const table = reducer.emitterView();
    encodeShadoParticleEmitter(table, 1, baseSpec({ emission: { mode: 'rate', rate: 10 }, attach: 'whileAlive', power: [0, 0] }), 0, {
      parent: 0,
      firstChild: -1,
      nextSibling: -1,
    });
    encodeShadoParticleEmitter(table, 0, baseSpec({ emission: { mode: 'burst', count: 1 }, shape: { kind: 'directional', dir: [1, 0, 0] }, power: [4, 4] }), 0, {
      parent: -1,
      firstChild: 1,
      nextSibling: -1,
    });
    reducer.step(0);
    expect(reducer.exports.getTrailCount()).toBe(1);
    reducer.step(0.5);
    const view = reducer.particleView();
    // Five trail particles, spawned where the parent is at t = 0.5: x = 2.
    expect(record(view, 1, P.birth, 0)).toBeCloseTo(2, 4);
    expect(reducer.step(1.5).reduce((n, r) => n + r.count, 0)).toBe(5);
    expect(reducer.exports.getTrailCount()).toBe(0);
  });

  it('never grows memory in steady state', async () => {
    const reducer = await arena(256);
    encodeShadoParticleEmitter(reducer.emitterView(), 0, baseSpec({ emission: { mode: 'rate', rate: 500 } }), 0);
    const bytes = reducer.memory.buffer.byteLength;
    for (let frame = 1; frame < 2_000; frame++) reducer.step(frame / 60);
    expect(reducer.memory.buffer.byteLength).toBe(bytes);
  });
});

describe('particle ramps', () => {
  it('bakes colour and normalised size rows, shares identical curves and reuses freed rows', () => {
    const ramps = new ShadoParticleRamps(1);
    const curve = { color: [{ t: 0, value: [1, 0, 0, 1] as const }, { t: 1, value: [0, 0, 1, 0] as const }], size: [{ t: 0, value: 1 }, { t: 1, value: 4 }] };
    const first = ramps.acquire(curve);
    expect(first).toEqual({ row: 0, sizeScale: 4 });
    expect(ramps.acquire(curve)).toEqual(first);
    const other = ramps.acquire({ color: [{ t: 0, value: [1, 1, 1, 1] }], size: [{ t: 0, value: 2 }] });
    expect(other).toEqual({ row: 3, sizeScale: 2 });
    expect(ramps.height).toBeGreaterThanOrEqual(6);
    const at = (r: number, x: number, c: number) => ramps.data[(r * ramps.width + x) * 4 + c];
    const last = ramps.width - 1;
    expect([at(0, 0, 0), at(0, 0, 2), at(0, last, 2), at(0, last, 3)]).toEqual([255, 0, 255, 0]);
    // Size row: 1/4 at birth, 4/4 at death; with no upper curve, green repeats it.
    expect(at(1, 0, 0)).toBe(64);
    expect(at(1, last, 0)).toBe(255);
    expect(at(1, 0, 1)).toBe(64);
    // With no second colour, the third row repeats the first.
    expect([at(2, 0, 0), at(2, last, 2)]).toEqual([255, 255]);
    ramps.release(first.row);
    ramps.release(first.row);
    ramps.release(other.row);
    expect(ramps.acquire({ color: [{ t: 0, value: [0, 1, 0, 1] }], size: [{ t: 0, value: 5 }] }).row).toBe(3);
    // A random pair: second colour row and upper size channel, scaled by the larger curve.
    const pair = ramps.acquire({
      color: [{ t: 0, value: [0, 0, 0, 1] }],
      color2: [{ t: 0, value: [1, 1, 1, 1] }],
      size: [{ t: 0, value: 1 }],
      sizeMax: [{ t: 0, value: 3 }],
    });
    expect(pair.sizeScale).toBe(3);
    expect([at(pair.row, 0, 0), at(pair.row + 2, 0, 0), at(pair.row + 1, 0, 0), at(pair.row + 1, 0, 1)]).toEqual([0, 255, 85, 255]);
    expect(sampleShadoRamp([{ t: 0, value: 0 }, { t: 1, value: 10 }], 0.25, (a, b, u) => a + (b - a) * u)).toBe(2.5);
  });
});
