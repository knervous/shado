import {
  SHADO_PARTICLE_EMITTER_FLOATS,
  SHADO_PARTICLE_RAMP_ROWS,
  SHADO_PARTICLE_RAMP_WIDTH,
  ShadoParticleAttach,
  ShadoParticleCollision,
  ShadoParticleEmitterField as F,
  ShadoParticleShape,
} from './ShadoParticleLayout';
import type { ShadoRgba } from './ShadoEntity2D';

export type ShadoVec3 = readonly [number, number, number];

export type ShadoParticleShapeSpec =
  | { kind: 'omni' }
  | { kind: 'directional'; dir: ShadoVec3 }
  | { kind: 'conic'; dir: ShadoVec3; radius: number }
  | { kind: 'radial'; dir: ShadoVec3 }
  | { kind: 'sphere'; radius: number }
  | { kind: 'rectangle'; dir: ShadoVec3; width: number; height: number };

/** One emitter, as the host describes it. Times in seconds, distances in world units. */
export interface ShadoParticleEmitterSpec {
  /**
   * `rate` emits continuously; `prime` particles are owed at the start (an emitter that fires
   * its first emission immediately), and `stopAfter` counts from the end of `delay`.
   */
  emission:
    | { mode: 'rate'; rate: number; stopAfter?: number | null; prime?: number }
    | { mode: 'burst'; count: number };
  delay?: number;
  shape: ShadoParticleShapeSpec;
  power: readonly [number, number];
  life: readonly [number, number];
  gravity?: ShadoVec3;
  drag?: number;
  /** Row returned by `ShadoParticleRamps.acquire`. */
  rampRow: number;
  /** Atlas layer the particle samples (each image fills its layer). */
  layer: number;
  rotation?: readonly [number, number];
  spin?: number;
  sizeScale?: number;
  /** Additive (the default for effects) or alpha-blended. */
  additive?: boolean;
  /** `camera` quads face the viewer (billboards); `ground` quads lie flat in world XZ. */
  orientation?: 'camera' | 'ground';
  /** Anchor slot particles follow, or -1 / undefined for world space. */
  anchor?: number;
  origin?: ShadoVec3;
  /** How a child attaches to its parent's particles. Ignored without `parent`. */
  attach?: 'whileAlive' | 'onDeath' | 'onBirth' | 'onBounce';
  /**
   * A horizontal ground plane the particles collide with, at `ground` in the frame they are
   * born in (anchor-relative for anchored emitters, world for world-space ones).
   */
  collision?: { mode: 'bounce' | 'destroy'; ground: number; restitution?: number };
  maxSpawnPerStep?: number;
  inheritSpeed?: number;
}

const SHAPES: Record<ShadoParticleShapeSpec['kind'], number> = {
  omni: ShadoParticleShape.omni,
  directional: ShadoParticleShape.directional,
  conic: ShadoParticleShape.conic,
  radial: ShadoParticleShape.radial,
  sphere: ShadoParticleShape.sphere,
  rectangle: ShadoParticleShape.rectangle,
};

/** Writes `spec` into emitter slot `index` of `table`, starting at `startTime`. */
export function encodeShadoParticleEmitter(
  table: Float32Array,
  index: number,
  spec: ShadoParticleEmitterSpec,
  startTime: number,
  link: { parent: number; firstChild: number; nextSibling: number } = { parent: -1, firstChild: -1, nextSibling: -1 }
): void {
  const base = index * SHADO_PARTICLE_EMITTER_FLOATS;
  table.fill(0, base, base + SHADO_PARTICLE_EMITTER_FLOATS);
  const set = (field: number, value: number) => {
    table[base + field] = Number.isFinite(value) ? value : 0;
  };
  const child = link.parent >= 0;
  set(F.state, 1);
  if (spec.emission.mode === 'rate') {
    set(F.mode, 0);
    set(F.rate, Math.max(0, spec.emission.rate));
    set(F.stopAfter, spec.emission.stopAfter == null ? -1 : Math.max(0, spec.emission.stopAfter));
    set(F.accumulator, Math.max(0, spec.emission.prime ?? 0));
  } else {
    set(F.mode, 1);
    set(F.burstCount, Math.max(0, Math.round(spec.emission.count)));
    set(F.stopAfter, -1);
  }
  set(F.delay, Math.max(0, spec.delay ?? 0));
  set(F.startTime, startTime);
  set(F.lastTime, startTime);
  const shape = spec.shape;
  set(F.shape, SHAPES[shape.kind]);
  if ('dir' in shape) {
    set(F.dirX, shape.dir[0]);
    set(F.dirY, shape.dir[1]);
    set(F.dirZ, shape.dir[2]);
  }
  if (shape.kind === 'conic' || shape.kind === 'sphere') set(F.radius, shape.radius);
  if (shape.kind === 'rectangle') {
    set(F.radius, shape.width);
    set(F.height, shape.height);
  }
  set(F.powerMin, spec.power[0]);
  set(F.powerMax, spec.power[1]);
  set(F.lifeMin, Math.max(0.001, spec.life[0]));
  set(F.lifeMax, Math.max(0.001, spec.life[1]));
  const gravity = spec.gravity ?? [0, 0, 0];
  set(F.gravityX, gravity[0]);
  set(F.gravityY, gravity[1]);
  set(F.gravityZ, gravity[2]);
  set(F.drag, Math.max(0, spec.drag ?? 0));
  set(F.rampRow, spec.rampRow);
  set(F.layer, spec.layer);
  if (spec.collision) {
    set(F.collisionMode, ShadoParticleCollision[spec.collision.mode]);
    set(F.ground, spec.collision.ground);
    set(F.restitution, Math.max(0, Math.min(1, spec.collision.restitution ?? 0.5)));
  }
  const rotation = spec.rotation ?? [0, 0];
  set(F.rotationMin, rotation[0]);
  set(F.rotationMax, rotation[1]);
  set(F.spin, spec.spin ?? 0);
  set(F.sizeScale, spec.sizeScale ?? 1);
  // A small bitfield in one float: bit 0 additive, bit 1 ground-facing.
  set(F.additive, (spec.additive === false ? 0 : SHADO_PARTICLE_FLAG_ADDITIVE) | (spec.orientation === 'ground' ? SHADO_PARTICLE_FLAG_GROUND : 0));
  set(F.anchor, spec.anchor ?? -1);
  const origin = spec.origin ?? [0, 0, 0];
  set(F.originX, origin[0]);
  set(F.originY, origin[1]);
  set(F.originZ, origin[2]);
  set(F.parent, link.parent);
  set(
    F.attach,
    !child ? ShadoParticleAttach.origin : ShadoParticleAttach[spec.attach ?? 'onDeath']
  );
  set(F.maxSpawnPerStep, Math.max(1, Math.round(spec.maxSpawnPerStep ?? 256)));
  set(F.firstChild, link.firstChild);
  set(F.nextSibling, link.nextSibling);
  set(F.inheritSpeed, spec.inheritSpeed ?? 0);
}

export type ShadoRampKey<T> = { readonly t: number; readonly value: T };

/** Particle flag bits, stored in the emitter's `additive` field and each particle's `extra.z`. */
export const SHADO_PARTICLE_FLAG_ADDITIVE = 1;
export const SHADO_PARTICLE_FLAG_GROUND = 2;

export interface ShadoParticleCurve {
  /** Colour over normalised life. One key is a constant. */
  color: readonly ShadoRampKey<ShadoRgba>[];
  /**
   * A second colour curve. Each particle takes its own mix of `color` and `color2` by its
   * seed, for effects authored with a random colour between two. Defaults to `color`.
   */
  color2?: readonly ShadoRampKey<ShadoRgba>[];
  /** Size (full width, world units) over normalised life. One key is a constant. */
  size: readonly ShadoRampKey<number>[];
  /** Upper size curve: each particle picks between `size` and this by its seed. Defaults to `size`. */
  sizeMax?: readonly ShadoRampKey<number>[];
}

/** Linear interpolation of `keys` at `t`, clamped at both ends. */
export function sampleShadoRamp<T>(keys: readonly ShadoRampKey<T>[], t: number, lerp: (a: T, b: T, u: number) => T): T {
  if (keys.length === 0) throw new Error('A ramp needs at least one key');
  if (t <= keys[0].t) return keys[0].value;
  for (let i = 1; i < keys.length; i++) {
    const hi = keys[i];
    if (t <= hi.t) {
      const lo = keys[i - 1];
      return lerp(lo.value, hi.value, hi.t === lo.t ? 1 : (t - lo.t) / (hi.t - lo.t));
    }
  }
  return keys[keys.length - 1].value;
}

/** A baked curve: its first row, and the size its normalised size row is a fraction of. */
export interface ShadoParticleRampRow {
  readonly row: number;
  readonly sizeScale: number;
}

/**
 * Colour and size curves baked into rows of an 8-bit RGBA texture.
 *
 * Each curve takes three rows: colour, size, second colour. Size holds the lower curve in red
 * and the upper in green, both as a fraction of the
 * curve's largest size (returned as `sizeScale`, which the emitter multiplies back in).
 * Eight bits rather than float because every backend can filter and bind RGBA8 without
 * a feature check, and a 1/255 step is invisible on a particle's size or colour.
 *
 * The vertex shader reads both rows by the particle's age over its life, so no curve ever
 * costs a CPU write after it is baked. Identical curves share rows; rows are counted and
 * reused.
 */
export class ShadoParticleRamps {
  public readonly width = SHADO_PARTICLE_RAMP_WIDTH;
  private rows = 0;
  private capacityRows: number;
  private buffer: Uint8Array;
  private readonly byKey = new Map<string, number>();
  private readonly sizeScales = new Map<number, number>();
  private readonly refs = new Map<number, number>();
  private readonly free: number[] = [];
  /** Bumped whenever the data changes; the renderer re-uploads on a change. */
  public version = 0;

  public constructor(initialCurves = 16) {
    this.capacityRows = Math.max(SHADO_PARTICLE_RAMP_ROWS, initialCurves * SHADO_PARTICLE_RAMP_ROWS);
    this.buffer = new Uint8Array(this.capacityRows * this.width * 4);
  }

  public get data(): Uint8Array {
    return this.buffer;
  }

  /** Rows the texture must have; a multiple of `SHADO_PARTICLE_RAMP_ROWS`. */
  public get height(): number {
    return this.capacityRows;
  }

  public get usedRows(): number {
    return this.rows;
  }

  /** The rows for `curve`, baking it if new. Pair with `release(row)`. */
  public acquire(curve: ShadoParticleCurve): ShadoParticleRampRow {
    const key = JSON.stringify(curve);
    let row = this.byKey.get(key);
    if (row === undefined) {
      row = this.free.pop() ?? this.allocateRows();
      this.sizeScales.set(row, this.bake(row, curve));
      this.byKey.set(key, row);
    }
    this.refs.set(row, (this.refs.get(row) ?? 0) + 1);
    return { row, sizeScale: this.sizeScales.get(row) ?? 1 };
  }

  public release(row: number): void {
    const count = (this.refs.get(row) ?? 0) - 1;
    if (count > 0) {
      this.refs.set(row, count);
      return;
    }
    this.refs.delete(row);
    this.sizeScales.delete(row);
    for (const [key, value] of this.byKey) {
      if (value === row) {
        this.byKey.delete(key);
        break;
      }
    }
    this.free.push(row);
  }

  private allocateRows(): number {
    if (this.rows + SHADO_PARTICLE_RAMP_ROWS > this.capacityRows) {
      const next = new Uint8Array(this.capacityRows * 2 * this.width * 4);
      next.set(this.buffer);
      this.buffer = next;
      this.capacityRows *= 2;
    }
    const row = this.rows;
    this.rows += SHADO_PARTICLE_RAMP_ROWS;
    return row;
  }

  private bake(row: number, curve: ShadoParticleCurve): number {
    const lerpRgba = (a: ShadoRgba, b: ShadoRgba, u: number): ShadoRgba => [
      a[0] + (b[0] - a[0]) * u,
      a[1] + (b[1] - a[1]) * u,
      a[2] + (b[2] - a[2]) * u,
      a[3] + (b[3] - a[3]) * u,
    ];
    const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
    const byte = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));
    const color2 = curve.color2 ?? curve.color;
    const sizeMax = curve.sizeMax ?? curve.size;
    const sizeScale = Math.max(1e-6, ...curve.size.map(k => Math.abs(k.value)), ...sizeMax.map(k => Math.abs(k.value)));
    const writeColor = (at: number, color: ShadoRgba) => {
      this.buffer[at] = byte(color[0]);
      this.buffer[at + 1] = byte(color[1]);
      this.buffer[at + 2] = byte(color[2]);
      this.buffer[at + 3] = byte(color[3]);
    };
    for (let x = 0; x < this.width; x++) {
      const t = x / (this.width - 1);
      writeColor((row * this.width + x) * 4, sampleShadoRamp(curve.color, t, lerpRgba));
      writeColor(((row + 2) * this.width + x) * 4, sampleShadoRamp(color2, t, lerpRgba));
      const sizeAt = ((row + 1) * this.width + x) * 4;
      this.buffer[sizeAt] = byte(Math.abs(sampleShadoRamp(curve.size, t, lerp)) / sizeScale);
      this.buffer[sizeAt + 1] = byte(Math.abs(sampleShadoRamp(sizeMax, t, lerp)) / sizeScale);
      this.buffer[sizeAt + 2] = 0;
      this.buffer[sizeAt + 3] = 255;
    }
    this.version++;
    return sizeScale;
  }
}
