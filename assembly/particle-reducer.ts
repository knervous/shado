// Shado particle reducer.
//
// Owns emission. Particles themselves are never touched after birth: a record holds its
// birth position, velocity, acceleration, drag and life, and the vertex shader evaluates
// where it is and how it looks in closed form. So the only per-frame work in this module
// is deciding how many particles each emitter spawns and writing those records.
//
// Memory: the host allocates every region through `alloc` (this module's memory is the
// Shado arena, so particle records live in the same bytes the GPU upload reads) and
// passes the pointers to `initArena`. Layout numbers mirror
// `src/render/ShadoParticleLayout.ts`; the layout test fails if they drift.
//
// Slots are a ring: a new particle takes the slot after the last one written, and a full
// ring overwrites its oldest particle. That makes allocation O(1) with no free list, and
// makes every step's writes one contiguous range (two when it wraps), which is exactly
// the shape a partial GPU upload wants.

// ---- particle record (floats) ------------------------------------------------------
const PARTICLE_FLOATS: i32 = 24;
const P_BIRTH: i32 = 0;
const P_VELOCITY: i32 = 4;
const P_ACCEL: i32 = 8;
const P_LOOK: i32 = 12;
const P_COLLISION: i32 = 16;
const P_EXTRA: i32 = 20;

// ---- emitter slot (floats) ---------------------------------------------------------
const EMITTER_FLOATS: i32 = 48;
const E_STATE: i32 = 0;
const E_MODE: i32 = 1;
const E_RATE: i32 = 2;
const E_BURST_COUNT: i32 = 3;
const E_DELAY: i32 = 4;
const E_STOP_AFTER: i32 = 5;
const E_START_TIME: i32 = 6;
const E_LAST_TIME: i32 = 7;
const E_ACCUMULATOR: i32 = 8;
const E_BURST_DONE: i32 = 9;
const E_SHAPE: i32 = 10;
const E_DIR_X: i32 = 11;
const E_DIR_Y: i32 = 12;
const E_DIR_Z: i32 = 13;
const E_RADIUS: i32 = 14;
const E_HEIGHT: i32 = 15;
const E_POWER_MIN: i32 = 16;
const E_POWER_MAX: i32 = 17;
const E_LIFE_MIN: i32 = 18;
const E_LIFE_MAX: i32 = 19;
const E_GRAVITY_X: i32 = 20;
const E_GRAVITY_Y: i32 = 21;
const E_GRAVITY_Z: i32 = 22;
const E_DRAG: i32 = 23;
const E_RAMP_ROW: i32 = 24;
const E_LAYER: i32 = 25;
const E_COLLISION_MODE: i32 = 26;
const E_GROUND: i32 = 27;
const E_RESTITUTION: i32 = 28;
const E_RESERVED: i32 = 29;
const E_ROTATION_MIN: i32 = 30;
const E_ROTATION_MAX: i32 = 31;
const E_SPIN: i32 = 32;
const E_SIZE_SCALE: i32 = 33;
const E_ADDITIVE: i32 = 34;
const E_ANCHOR: i32 = 35;
const E_ORIGIN_X: i32 = 36;
const E_ORIGIN_Y: i32 = 37;
const E_ORIGIN_Z: i32 = 38;
const E_PARENT: i32 = 39;
const E_ATTACH: i32 = 40;
const E_MAX_SPAWN_PER_STEP: i32 = 41;
const E_FIRST_CHILD: i32 = 42;
const E_NEXT_SIBLING: i32 = 43;
const E_INHERIT_SPEED: i32 = 44;

const SHAPE_OMNI: i32 = 0;
const SHAPE_DIRECTIONAL: i32 = 1;
const SHAPE_CONIC: i32 = 2;
const SHAPE_RADIAL: i32 = 3;
const SHAPE_SPHERE: i32 = 4;
const SHAPE_RECTANGLE: i32 = 5;

const ATTACH_ORIGIN: i32 = 0;
const ATTACH_WHILE_ALIVE: i32 = 1;
const ATTACH_ON_DEATH: i32 = 2;
const ATTACH_ON_BIRTH: i32 = 3;
const ATTACH_ON_BOUNCE: i32 = 4;

const COLLISION_NONE: i32 = 0;
const COLLISION_BOUNCE: i32 = 1;
const COLLISION_DESTROY: i32 = 2;

const PENDING_FLOATS: i32 = 8;
const TRAIL_FLOATS: i32 = 16;

// ---- state -------------------------------------------------------------------------
let heapPtr: i32 = 1024;

let particlePtr: i32 = 0;
let particleCapacity: i32 = 0;
let head: i32 = 0;

let emitterPtr: i32 = 0;
let emitterCapacity: i32 = 0;

let pendingPtr: i32 = 0;
let pendingCapacity: i32 = 0;
let pendingCount: i32 = 0;

let trailPtr: i32 = 0;
let trailCapacity: i32 = 0;
let trailCount: i32 = 0;

let changedFirst: i32 = 0;
let changedCount: i32 = 0;
let spawnedTotal: f64 = 0;
let droppedTotal: f64 = 0;

let rngState: u32 = 0x9e3779b9;

// ---- memory ------------------------------------------------------------------------
export function alloc(byteLength: i32): i32 {
  const alignedLength = (byteLength + 15) & ~15;
  const ptr = heapPtr;
  const next = ptr + alignedLength;
  const currentBytes = memory.size() << 16;
  if (next > currentBytes) memory.grow((next - currentBytes + 0xffff) >> 16);
  heapPtr = next;
  return ptr;
}

export function resetAllocator(byteOffset: i32 = 1024): void {
  heapPtr = (byteOffset + 15) & ~15;
}

export function initArena(
  particlePtrArg: i32,
  particleCapacityArg: i32,
  emitterPtrArg: i32,
  emitterCapacityArg: i32,
  pendingPtrArg: i32,
  pendingCapacityArg: i32,
  trailPtrArg: i32,
  trailCapacityArg: i32,
  seed: u32
): void {
  particlePtr = particlePtrArg;
  particleCapacity = particleCapacityArg > 0 ? particleCapacityArg : 0;
  emitterPtr = emitterPtrArg;
  emitterCapacity = emitterCapacityArg > 0 ? emitterCapacityArg : 0;
  pendingPtr = pendingPtrArg;
  pendingCapacity = pendingCapacityArg > 0 ? pendingCapacityArg : 0;
  trailPtr = trailPtrArg;
  trailCapacity = trailCapacityArg > 0 ? trailCapacityArg : 0;
  head = 0;
  pendingCount = 0;
  trailCount = 0;
  changedFirst = 0;
  changedCount = 0;
  spawnedTotal = 0;
  droppedTotal = 0;
  rngState = seed != 0 ? seed : 0x9e3779b9;
}

export function getHead(): i32 {
  return head;
}
export function getChangedFirst(): i32 {
  return changedFirst;
}
/** Slots written by the last step, counted from `getChangedFirst`, wrapping the ring. */
export function getChangedCount(): i32 {
  return changedCount;
}
export function getSpawnedTotal(): f64 {
  return spawnedTotal;
}
/** Sub-emitter work that did not fit its queue and was dropped. */
export function getDroppedTotal(): f64 {
  return droppedTotal;
}
export function getPendingCount(): i32 {
  return pendingCount;
}
export function getTrailCount(): i32 {
  return trailCount;
}

// ---- helpers -----------------------------------------------------------------------
@inline function ef(emitter: i32, field: i32): f32 {
  return load<f32>(emitterPtr + (emitter * EMITTER_FLOATS + field) * 4);
}
@inline function setEf(emitter: i32, field: i32, value: f32): void {
  store<f32>(emitterPtr + (emitter * EMITTER_FLOATS + field) * 4, value);
}
@inline function pf(slot: i32, field: i32, value: f32): void {
  store<f32>(particlePtr + (slot * PARTICLE_FLOATS + field) * 4, value);
}

@inline function random(): f32 {
  let x = rngState;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rngState = x;
  return <f32>(x >> 8) / <f32>16777216.0;
}

@inline function between(min: f32, max: f32): f32 {
  return min + (max - min) * random();
}

/** Position after `t` seconds of constant acceleration `g` and linear drag `k`. */
@inline function closedPosition(p: f32, v: f32, g: f32, k: f32, t: f32): f32 {
  if (k < <f32>0.0001) return p + v * t + <f32>0.5 * g * t * t;
  const terminal = g / k;
  return p + terminal * t + (v - terminal) * (<f32>1.0 - Mathf.exp(-k * t)) / k;
}

@inline function closedVelocity(v: f32, g: f32, k: f32, t: f32): f32 {
  if (k < <f32>0.0001) return v + g * t;
  const terminal = g / k;
  return (v - terminal) * Mathf.exp(-k * t) + terminal;
}

/**
 * Seconds until a particle moving in closed form falls to height `ground`, or -1 if it
 * never does within `limit`. Exact without drag (a quadratic); with drag, Newton steps on
 * the closed form from the drag-free guess. A particle born at or under the ground never
 * collides: it was placed there on purpose.
 */
function impactTime(y0: f32, vy: f32, gy: f32, k: f32, ground: f32, limit: f32): f32 {
  const h = y0 - ground;
  if (h <= <f32>0.0) return -1;
  let t: f32 = -1;
  if (Mathf.abs(gy) < <f32>0.00001) {
    if (vy < <f32>0.0) t = -h / vy;
  } else {
    // 0.5 gy t^2 + vy t + h = 0; the later root is the fall back through the plane.
    const disc = vy * vy - <f32>2.0 * gy * h;
    if (disc >= <f32>0.0) {
      const root = Mathf.sqrt(disc);
      const a = (-vy - root) / gy;
      const b = (-vy + root) / gy;
      t = a > <f32>0.0 && b > <f32>0.0 ? Mathf.min(a, b) : Mathf.max(a, b);
    }
  }
  if (k >= <f32>0.0001) {
    // Drag slows everything down, so the drag-free time is early; walk forward to the plane.
    if (t <= <f32>0.0) t = limit;
    for (let i = 0; i < 6; i++) {
      const f = closedPosition(y0, vy, gy, k, t) - ground;
      const df = closedVelocity(vy, gy, k, t);
      if (Mathf.abs(df) < <f32>0.00001) break;
      t = t - f / df;
      if (t <= <f32>0.0) return -1;
    }
    if (Mathf.abs(closedPosition(y0, vy, gy, k, t) - ground) > <f32>0.01) return -1;
  }
  return t > <f32>0.0 && t < limit ? t : -1;
}

// Scratch for the direction and offset a shape produces.
let sx: f32 = 0;
let sy: f32 = 0;
let sz: f32 = 0;
let ox: f32 = 0;
let oy: f32 = 0;
let oz: f32 = 0;

function randomUnit(): void {
  // Uniform on the sphere: z uniform in [-1, 1], azimuth uniform.
  const z = <f32>2.0 * random() - <f32>1.0;
  const a = <f32>6.2831853 * random();
  const r = Mathf.sqrt(Mathf.max(<f32>0.0, <f32>1.0 - z * z));
  sx = r * Mathf.cos(a);
  sy = z;
  sz = r * Mathf.sin(a);
}

function sampleShape(emitter: i32): void {
  const shape = <i32>ef(emitter, E_SHAPE);
  const dx = ef(emitter, E_DIR_X);
  const dy = ef(emitter, E_DIR_Y);
  const dz = ef(emitter, E_DIR_Z);
  const radius = ef(emitter, E_RADIUS);
  ox = 0;
  oy = 0;
  oz = 0;
  if (shape == SHAPE_OMNI) {
    randomUnit();
  } else if (shape == SHAPE_SPHERE) {
    randomUnit();
    ox = sx * radius;
    oy = sy * radius;
    oz = sz * radius;
  } else if (shape == SHAPE_CONIC) {
    let x = dx + (random() * <f32>2.0 - <f32>1.0) * radius;
    let y = dy + (random() * <f32>2.0 - <f32>1.0) * radius;
    let z = dz + (random() * <f32>2.0 - <f32>1.0) * radius;
    const len = Mathf.sqrt(x * x + y * y + z * z);
    if (len > <f32>0.00001) {
      x /= len;
      y /= len;
      z /= len;
    }
    sx = x;
    sy = y;
    sz = z;
  } else if (shape == SHAPE_RADIAL) {
    // Outward, in the plane perpendicular to the axis.
    randomUnit();
    const len = Mathf.sqrt(dx * dx + dy * dy + dz * dz);
    if (len > <f32>0.00001) {
      const ax = dx / len;
      const ay = dy / len;
      const az = dz / len;
      const along = sx * ax + sy * ay + sz * az;
      let px = sx - ax * along;
      let py = sy - ay * along;
      let pz = sz - az * along;
      const plen = Mathf.sqrt(px * px + py * py + pz * pz);
      if (plen > <f32>0.00001) {
        px /= plen;
        py /= plen;
        pz /= plen;
      }
      sx = px;
      sy = py;
      sz = pz;
    }
  } else if (shape == SHAPE_RECTANGLE) {
    ox = (random() - <f32>0.5) * radius;
    oz = (random() - <f32>0.5) * ef(emitter, E_HEIGHT);
    sx = dx;
    sy = dy;
    sz = dz;
  } else {
    sx = dx;
    sy = dy;
    sz = dz;
  }
}

function markWritten(slot: i32): void {
  if (changedCount == 0) changedFirst = slot;
  if (changedCount < particleCapacity) changedCount++;
}

/**
 * Writes one particle for `emitter` at position (x, y, z) plus the shape's offset,
 * inheriting `inherit` times (vx, vy, vz), and schedules its children.
 */
function spawn(emitter: i32, now: f32, x: f32, y: f32, z: f32, vx: f32, vy: f32, vz: f32): void {
  if (particleCapacity == 0) return;
  sampleShape(emitter);
  const power = between(ef(emitter, E_POWER_MIN), ef(emitter, E_POWER_MAX));
  let life = between(ef(emitter, E_LIFE_MIN), ef(emitter, E_LIFE_MAX));
  const px = x + ox;
  const py = y + oy;
  const pz = z + oz;
  const velX = vx + sx * power;
  const velY = vy + sy * power;
  const velZ = vz + sz * power;
  const gx = ef(emitter, E_GRAVITY_X);
  const gy = ef(emitter, E_GRAVITY_Y);
  const gz = ef(emitter, E_GRAVITY_Z);
  const drag = ef(emitter, E_DRAG);

  // Collision with a ground plane, decided once at birth: a destroyed particle simply dies
  // on the plane (so its on-death children start there); a bouncing one records when it
  // lands, and the shader continues it in closed form from that moment.
  const collision = <i32>ef(emitter, E_COLLISION_MODE);
  const ground = ef(emitter, E_GROUND);
  let impact: f32 = -1;
  if (collision != COLLISION_NONE) {
    impact = impactTime(py, velY, gy, drag, ground, life);
    if (collision == COLLISION_DESTROY && impact > <f32>0.0) {
      life = impact;
      impact = -1;
    }
  }

  const slot = head;
  head = head + 1 >= particleCapacity ? 0 : head + 1;
  pf(slot, P_BIRTH, px);
  pf(slot, P_BIRTH + 1, py);
  pf(slot, P_BIRTH + 2, pz);
  pf(slot, P_BIRTH + 3, now);
  pf(slot, P_VELOCITY, velX);
  pf(slot, P_VELOCITY + 1, velY);
  pf(slot, P_VELOCITY + 2, velZ);
  pf(slot, P_VELOCITY + 3, life);
  pf(slot, P_ACCEL, gx);
  pf(slot, P_ACCEL + 1, gy);
  pf(slot, P_ACCEL + 2, gz);
  pf(slot, P_ACCEL + 3, drag);
  pf(slot, P_LOOK, ef(emitter, E_RAMP_ROW));
  pf(slot, P_LOOK + 1, ef(emitter, E_LAYER));
  pf(slot, P_LOOK + 2, between(ef(emitter, E_ROTATION_MIN), ef(emitter, E_ROTATION_MAX)));
  pf(slot, P_LOOK + 3, ef(emitter, E_SPIN));
  pf(slot, P_COLLISION, ground);
  pf(slot, P_COLLISION + 1, ef(emitter, E_RESTITUTION));
  pf(slot, P_COLLISION + 2, collision == COLLISION_BOUNCE ? <f32>1.0 : <f32>0.0);
  pf(slot, P_COLLISION + 3, impact);
  pf(slot, P_EXTRA, ef(emitter, E_ANCHOR));
  pf(slot, P_EXTRA + 1, ef(emitter, E_SIZE_SCALE));
  pf(slot, P_EXTRA + 2, ef(emitter, E_ADDITIVE));
  pf(slot, P_EXTRA + 3, random());
  markWritten(slot);
  spawnedTotal += 1;

  // Children: what this particle does when it is born, dies, or while it lives.
  let child = <i32>ef(emitter, E_FIRST_CHILD);
  let guard = 0;
  while (child >= 0 && child < emitterCapacity && guard < emitterCapacity) {
    guard++;
    const attach = <i32>ef(child, E_ATTACH);
    if (ef(child, E_STATE) > <f32>0.0) {
      if (attach == ATTACH_ON_BOUNCE) {
        // NeL emits on every bounce; here, on the first, where and as it lands.
        if (impact > <f32>0.0 && pendingCount < pendingCapacity) {
          const e = ef(emitter, E_RESTITUTION);
          const at = pendingPtr + pendingCount * PENDING_FLOATS * 4;
          store<f32>(at, now + impact);
          store<f32>(at + 4, closedPosition(px, velX, gx, drag, impact));
          store<f32>(at + 8, ground);
          store<f32>(at + 12, closedPosition(pz, velZ, gz, drag, impact));
          store<f32>(at + 16, closedVelocity(velX, gx, drag, impact) * e);
          store<f32>(at + 20, -closedVelocity(velY, gy, drag, impact) * e);
          store<f32>(at + 24, closedVelocity(velZ, gz, drag, impact) * e);
          store<f32>(at + 28, <f32>child);
          pendingCount++;
        } else if (impact > <f32>0.0) {
          droppedTotal += 1;
        }
      } else if (attach == ATTACH_ON_BIRTH) {
        // Queued rather than spawned here, so a deep chain of births cannot recurse; the
        // pending pass later in this same step emits it.
        if (pendingCount < pendingCapacity) {
          const at = pendingPtr + pendingCount * PENDING_FLOATS * 4;
          store<f32>(at, now);
          store<f32>(at + 4, px);
          store<f32>(at + 8, py);
          store<f32>(at + 12, pz);
          store<f32>(at + 16, velX);
          store<f32>(at + 20, velY);
          store<f32>(at + 24, velZ);
          store<f32>(at + 28, <f32>child);
          pendingCount++;
        } else {
          droppedTotal += 1;
        }
      } else if (attach == ATTACH_ON_DEATH) {
        if (pendingCount < pendingCapacity) {
          const at = pendingPtr + pendingCount * PENDING_FLOATS * 4;
          store<f32>(at, now + life);
          store<f32>(at + 4, closedPosition(px, velX, gx, drag, life));
          store<f32>(at + 8, closedPosition(py, velY, gy, drag, life));
          store<f32>(at + 12, closedPosition(pz, velZ, gz, drag, life));
          store<f32>(at + 16, closedVelocity(velX, gx, drag, life));
          store<f32>(at + 20, closedVelocity(velY, gy, drag, life));
          store<f32>(at + 24, closedVelocity(velZ, gz, drag, life));
          store<f32>(at + 28, <f32>child);
          pendingCount++;
        } else {
          droppedTotal += 1;
        }
      } else if (attach == ATTACH_WHILE_ALIVE) {
        if (trailCount < trailCapacity) {
          const at = trailPtr + trailCount * TRAIL_FLOATS * 4;
          store<f32>(at, now);
          store<f32>(at + 4, now + life);
          store<f32>(at + 8, px);
          store<f32>(at + 12, py);
          store<f32>(at + 16, pz);
          store<f32>(at + 20, velX);
          store<f32>(at + 24, velY);
          store<f32>(at + 28, velZ);
          store<f32>(at + 32, gx);
          store<f32>(at + 36, gy);
          store<f32>(at + 40, gz);
          store<f32>(at + 44, drag);
          store<f32>(at + 48, <f32>child);
          store<f32>(at + 52, 0);
          store<f32>(at + 56, now);
          store<f32>(at + 60, 0);
          trailCount++;
        } else {
          droppedTotal += 1;
        }
      }
    }
    child = <i32>ef(child, E_NEXT_SIBLING);
  }
}

/** Spawns for one origin emitter this step. */
function stepRoot(emitter: i32, now: f32): void {
  const started = ef(emitter, E_START_TIME) + ef(emitter, E_DELAY);
  const last = ef(emitter, E_LAST_TIME);
  setEf(emitter, E_LAST_TIME, now);
  if (ef(emitter, E_STATE) != <f32>1.0 || now < started) return;
  const stopAfter = ef(emitter, E_STOP_AFTER);
  const maxSpawn = <i32>Mathf.max(<f32>1.0, ef(emitter, E_MAX_SPAWN_PER_STEP));
  const x = ef(emitter, E_ORIGIN_X);
  const y = ef(emitter, E_ORIGIN_Y);
  const z = ef(emitter, E_ORIGIN_Z);

  if (<i32>ef(emitter, E_MODE) == 1) {
    if (ef(emitter, E_BURST_DONE) > <f32>0.0) return;
    setEf(emitter, E_BURST_DONE, 1);
    const count = <i32>Mathf.min(ef(emitter, E_BURST_COUNT), <f32>maxSpawn);
    for (let i = 0; i < count; i++) spawn(emitter, now, x, y, z, 0, 0, 0);
    return;
  }

  // Emission runs from `started` until `started + stopAfter`.
  const windowEnd = stopAfter >= <f32>0.0 ? started + stopAfter : now;
  const from = Mathf.max(last, started);
  const to = Mathf.min(now, windowEnd);
  if (to > from) {
    let accumulator = ef(emitter, E_ACCUMULATOR) + (to - from) * ef(emitter, E_RATE);
    let count = <i32>Mathf.floor(accumulator);
    accumulator -= <f32>count;
    if (count > maxSpawn) count = maxSpawn;
    setEf(emitter, E_ACCUMULATOR, accumulator);
    for (let i = 0; i < count; i++) spawn(emitter, now, x, y, z, 0, 0, 0);
  }
  if (stopAfter >= <f32>0.0 && now >= windowEnd) setEf(emitter, E_STATE, 2);
}

function stepPending(now: f32): void {
  let i = 0;
  while (i < pendingCount) {
    const at = pendingPtr + i * PENDING_FLOATS * 4;
    if (load<f32>(at) > now) {
      i++;
      continue;
    }
    const child = <i32>load<f32>(at + 28);
    if (child >= 0 && child < emitterCapacity && ef(child, E_STATE) > <f32>0.0) {
      const count = <i32>Mathf.min(
        ef(child, E_MODE) == <f32>1.0 ? ef(child, E_BURST_COUNT) : <f32>1.0,
        Mathf.max(<f32>1.0, ef(child, E_MAX_SPAWN_PER_STEP))
      );
      const x = load<f32>(at + 4);
      const y = load<f32>(at + 8);
      const z = load<f32>(at + 12);
      const vx = load<f32>(at + 16);
      const vy = load<f32>(at + 20);
      const vz = load<f32>(at + 24);
      const inherit = ef(child, E_INHERIT_SPEED);
      for (let n = 0; n < count; n++) spawn(child, now, x, y, z, vx * inherit, vy * inherit, vz * inherit);
    }
    // Swap-remove.
    pendingCount--;
    if (i != pendingCount) {
      memory.copy(at, pendingPtr + pendingCount * PENDING_FLOATS * 4, PENDING_FLOATS * 4);
    }
  }
}

function stepTrails(now: f32): void {
  let i = 0;
  while (i < trailCount) {
    const at = trailPtr + i * TRAIL_FLOATS * 4;
    const start = load<f32>(at);
    const end = load<f32>(at + 4);
    const child = <i32>load<f32>(at + 48);
    const last = load<f32>(at + 56);
    const to = Mathf.min(now, end);
    if (child >= 0 && child < emitterCapacity && ef(child, E_STATE) > <f32>0.0 && to > last) {
      let accumulator = load<f32>(at + 52) + (to - last) * ef(child, E_RATE);
      let count = <i32>Mathf.floor(accumulator);
      accumulator -= <f32>count;
      const maxSpawn = <i32>Mathf.max(<f32>1.0, ef(child, E_MAX_SPAWN_PER_STEP));
      if (count > maxSpawn) count = maxSpawn;
      store<f32>(at + 52, accumulator);
      if (count > 0) {
        // Where the parent is now, from its own closed form.
        const t = to - start;
        const drag = load<f32>(at + 44);
        const x = closedPosition(load<f32>(at + 8), load<f32>(at + 20), load<f32>(at + 32), drag, t);
        const y = closedPosition(load<f32>(at + 12), load<f32>(at + 24), load<f32>(at + 36), drag, t);
        const z = closedPosition(load<f32>(at + 16), load<f32>(at + 28), load<f32>(at + 40), drag, t);
        const inherit = ef(child, E_INHERIT_SPEED);
        const vx = closedVelocity(load<f32>(at + 20), load<f32>(at + 32), drag, t) * inherit;
        const vy = closedVelocity(load<f32>(at + 24), load<f32>(at + 36), drag, t) * inherit;
        const vz = closedVelocity(load<f32>(at + 28), load<f32>(at + 40), drag, t) * inherit;
        for (let n = 0; n < count; n++) spawn(child, now, x, y, z, vx, vy, vz);
      }
    }
    store<f32>(at + 56, to);
    if (now >= end) {
      trailCount--;
      if (i != trailCount) memory.copy(at, trailPtr + trailCount * TRAIL_FLOATS * 4, TRAIL_FLOATS * 4);
    } else {
      i++;
    }
  }
}

/**
 * Advances every emitter to `now` (seconds on the container's clock) and writes the
 * particles they spawn. Returns how many slots were written; read the range with
 * `getChangedFirst` / `getChangedCount`.
 */
export function step(now: f32): i32 {
  changedFirst = head;
  changedCount = 0;
  for (let e = 0; e < emitterCapacity; e++) {
    const state = ef(e, E_STATE);
    if (state <= <f32>0.0) continue;
    if (<i32>ef(e, E_ATTACH) != ATTACH_ORIGIN) continue;
    stepRoot(e, now);
  }
  if (pendingCount > 0) stepPending(now);
  if (trailCount > 0) stepTrails(now);
  return changedCount;
}

/** Forgets queued sub-emitter work for `emitter` and its descendants' queue entries. */
export function clearQueuesFor(emitter: i32): void {
  let i = 0;
  while (i < pendingCount) {
    const at = pendingPtr + i * PENDING_FLOATS * 4;
    if (<i32>load<f32>(at + 28) == emitter) {
      pendingCount--;
      if (i != pendingCount) memory.copy(at, pendingPtr + pendingCount * PENDING_FLOATS * 4, PENDING_FLOATS * 4);
    } else {
      i++;
    }
  }
  i = 0;
  while (i < trailCount) {
    const at = trailPtr + i * TRAIL_FLOATS * 4;
    if (<i32>load<f32>(at + 48) == emitter) {
      trailCount--;
      if (i != trailCount) memory.copy(at, trailPtr + trailCount * TRAIL_FLOATS * 4, TRAIL_FLOATS * 4);
    } else {
      i++;
    }
  }
}

/** Marks every slot dead (life 0), e.g. when a scene is reset. */
export function killAll(): void {
  for (let slot = 0; slot < particleCapacity; slot++) pf(slot, P_VELOCITY + 3, 0);
  changedFirst = 0;
  changedCount = particleCapacity;
  pendingCount = 0;
  trailCount = 0;
  head = 0;
}
