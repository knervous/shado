/**
 * Byte layout shared by the particle reducer (`assembly/particle-reducer.ts`), the
 * container that owns its memory, and the shaders that read it.
 *
 * AssemblyScript cannot import TypeScript, so the reducer repeats these numbers; the
 * `particle-layout` test parses the assembly source and fails if the two drift.
 *
 * A particle is written once, when it is born, and never again: everything the shader
 * needs to place and colour it at any later moment is in its record, evaluated in closed
 * form against the frame time. That is what makes the per-frame cost of a live particle
 * zero on both the WebGPU storage path and the WebGL data-texture path.
 */

/** Floats per particle record: six vec4s. */
export const SHADO_PARTICLE_FLOATS = 24;
export const SHADO_PARTICLE_STRIDE_BYTES = SHADO_PARTICLE_FLOATS * 4;

/** Float offsets inside one particle record. */
export const ShadoParticleField = {
  /** xyz birth position (relative to the anchor, if any), w birth time in seconds. */
  birth: 0,
  /** xyz initial velocity, w life in seconds (0 = never born / dead slot). */
  velocity: 4,
  /** xyz constant acceleration (gravity and directional forces), w linear drag. */
  accel: 8,
  /** x ramp row, y atlas layer, z initial rotation (radians), w spin (radians/second). */
  look: 12,
  /**
   * Ground collision: x plane height (in the same frame as the birth position), y
   * restitution, z 1 when the particle bounces, w seconds from birth to its first landing
   * (-1 for none). A destroyed particle has no entry: its life already ends on the plane.
   */
  collision: 16,
  /** x anchor index (-1 for world space), y size scale, z additive (1) or alpha (0), w seed. */
  extra: 20,
} as const;

/** Floats per emitter slot in the reducer's emitter table. */
export const SHADO_PARTICLE_EMITTER_FLOATS = 48;

/** Float offsets inside one emitter slot. */
export const ShadoParticleEmitterField = {
  /** 0 free, 1 emitting, 2 stopped (children of its live particles still run). */
  state: 0,
  /** 0 rate, 1 burst. */
  mode: 1,
  rate: 2,
  burstCount: 3,
  delay: 4,
  /** Seconds after the first emission to stop emitting; negative for never. */
  stopAfter: 5,
  startTime: 6,
  lastTime: 7,
  accumulator: 8,
  burstDone: 9,
  /** See `ShadoParticleShape`. */
  shape: 10,
  dirX: 11,
  dirY: 12,
  dirZ: 13,
  /** Sphere radius, cone spread, or rectangle width. */
  radius: 14,
  /** Rectangle height. */
  height: 15,
  powerMin: 16,
  powerMax: 17,
  lifeMin: 18,
  lifeMax: 19,
  gravityX: 20,
  gravityY: 21,
  gravityZ: 22,
  drag: 23,
  rampRow: 24,
  layer: 25,
  /** See `ShadoParticleCollision`. */
  collisionMode: 26,
  /** Ground plane height, in the frame the emitter's particles are born in. */
  ground: 27,
  /** Share of velocity kept by a bounce (NeL's bounce factor). */
  restitution: 28,
  reserved: 29,
  rotationMin: 30,
  rotationMax: 31,
  spin: 32,
  sizeScale: 33,
  additive: 34,
  anchor: 35,
  originX: 36,
  originY: 37,
  originZ: 38,
  /** Parent emitter index, or -1 for an emitter that spawns from its origin. */
  parent: 39,
  /** See `ShadoParticleAttach`. */
  attach: 40,
  /** Most particles one step may spawn from this emitter; bounds a hitch. */
  maxSpawnPerStep: 41,
  /** First child emitter index, or -1. */
  firstChild: 42,
  /** Next sibling emitter index under the same parent, or -1. */
  nextSibling: 43,
  /** Share of the parent particle's velocity a child inherits. */
  inheritSpeed: 44,
} as const;

export const ShadoParticleShape = {
  omni: 0,
  directional: 1,
  conic: 2,
  radial: 3,
  sphere: 4,
  rectangle: 5,
} as const;

export const ShadoParticleAttach = {
  origin: 0,
  whileAlive: 1,
  onDeath: 2,
  /** Once, where and as the parent particle is born (NeL's `once` emission). */
  onBirth: 3,
  /** Once, where the parent particle first lands on its ground plane, with its bounced velocity. */
  onBounce: 4,
} as const;

export const ShadoParticleCollision = {
  none: 0,
  /** Reflected about the plane and scaled by restitution, like NeL's bounce zone. */
  bounce: 1,
  /** Dies on the plane. */
  destroy: 2,
} as const;

/** Floats per pending on-death spawn: due time, xyz position, xyz velocity, child emitter. */
export const SHADO_PARTICLE_PENDING_FLOATS = 8;
/**
 * Floats per live trail (a particle whose child emits while it lives): start, end, xyz
 * birth position, xyz velocity, xyz acceleration, drag, child emitter, accumulator,
 * last time, reserved.
 */
export const SHADO_PARTICLE_TRAIL_FLOATS = 16;

/** Samples per curve row in the ramp texture. */
/** Ramp rows per curve: colour, size (lower in red, upper in green), second colour. */
export const SHADO_PARTICLE_RAMP_ROWS = 3;
export const SHADO_PARTICLE_RAMP_WIDTH = 64;
