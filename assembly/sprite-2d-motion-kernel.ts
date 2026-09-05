let heapPtr: i32 = 1024;
let count: i32 = 0;
let globalStart: i32 = 0;
let positionPtr: i32 = 0;
let velocityPtr: i32 = 0;
let epochPtr: i32 = 0;
let nextChangePtr: i32 = 0;
let motionSeed: i32 = 1337;
let motionSpeed: f32 = 0.8;
let cadenceMs: f32 = 2000;
let minX: f32 = -10;
let minY: f32 = -7;
let maxX: f32 = 10;
let maxY: f32 = 7;

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

export function init(
  countArg: i32,
  globalStartArg: i32,
  positionPtrArg: i32,
  velocityPtrArg: i32,
  epochPtrArg: i32,
  nextChangePtrArg: i32
): void {
  count = countArg > 0 ? countArg : 0;
  globalStart = globalStartArg;
  positionPtr = positionPtrArg;
  velocityPtr = velocityPtrArg;
  epochPtr = epochPtrArg;
  nextChangePtr = nextChangePtrArg;
}

export function configure(
  seed: i32,
  speed: f32,
  cadenceMsArg: f32,
  minXArg: f32,
  minYArg: f32,
  maxXArg: f32,
  maxYArg: f32
): void {
  motionSeed = seed;
  motionSpeed = speed > 0 ? speed : 0;
  cadenceMs = cadenceMsArg > 1 ? cadenceMsArg : 1;
  minX = minXArg;
  minY = minYArg;
  maxX = maxXArg;
  maxY = maxYArg;
}

export function reset(nowMs: f64): void {
  for (let index = 0; index < count; index++) {
    store<i32>(epochPtr + index * 4, 0);
    writeVelocity(index, 0);
    store<f32>(
      nextChangePtr + index * 4,
      <f32>nowMs + cadenceMs * (0.5 + seededUnit(index, 0, 9))
    );
  }
}

export function step(nowMs: f64, dtSecondsArg: f32): i32 {
  if (count <= 0) return 0;
  const now = <f32>nowMs;
  const dtSeconds: f32 = dtSecondsArg < 0 ? 0 : dtSecondsArg > 0.2 ? 0.2 : dtSecondsArg;

  // Vector epochs and their deadlines remain in WASM so a seed produces the
  // same sequence independently of JS object allocation and frame cadence.
  for (let index = 0; index < count; index++) {
    let epoch = load<i32>(epochPtr + index * 4);
    let nextChange = load<f32>(nextChangePtr + index * 4);
    let changed = false;
    while (now >= nextChange) {
      epoch++;
      nextChange += cadenceMs * (0.5 + seededUnit(index, epoch, 9));
      changed = true;
    }
    if (changed) {
      store<i32>(epochPtr + index * 4, epoch);
      store<f32>(nextChangePtr + index * 4, nextChange);
      writeVelocity(index, epoch);
    }
  }

  const low = f32x4(minX, minY, minX, minY);
  const high = f32x4(maxX, maxY, maxX, maxY);
  const delta = f32x4.splat(dtSeconds);
  let index = 0;
  for (; index + 1 < count; index += 2) {
    const ptr = positionPtr + index * 8;
    let position = f32x4.add(
      v128.load(<usize>ptr),
      f32x4.mul(v128.load(<usize>(velocityPtr + index * 8)), delta)
    );
    position = v128.bitselect(high, position, f32x4.lt(position, low));
    position = v128.bitselect(low, position, f32x4.gt(position, high));
    v128.store(<usize>ptr, position);
  }
  if (index < count) integrateScalar(index, dtSeconds);
  return count;
}

function integrateScalar(index: i32, dtSeconds: f32): void {
  const offset = index * 8;
  let x = load<f32>(positionPtr + offset) + load<f32>(velocityPtr + offset) * dtSeconds;
  let y = load<f32>(positionPtr + offset + 4) + load<f32>(velocityPtr + offset + 4) * dtSeconds;
  if (x < minX) x = maxX;
  else if (x > maxX) x = minX;
  if (y < minY) y = maxY;
  else if (y > maxY) y = minY;
  store<f32>(positionPtr + offset, x);
  store<f32>(positionPtr + offset + 4, y);
}

function writeVelocity(index: i32, epoch: i32): void {
  let x = seededUnit(index, epoch, 5) * 2 - 1;
  let y = seededUnit(index, epoch, 6) * 2 - 1;
  const length = <f32>Math.sqrt(x * x + y * y);
  if (length < 0.0001) { x = 1; y = 0; }
  else { x /= length; y /= length; }
  const magnitude = motionSpeed * (0.35 + seededUnit(index, epoch, 7) * 0.65);
  const offset = index * 8;
  store<f32>(velocityPtr + offset, x * magnitude);
  store<f32>(velocityPtr + offset + 4, y * magnitude);
}

function seededUnit(index: i32, epoch: i32, salt: i32): f32 {
  let value = <u32>motionSeed ^ (<u32>(globalStart + index + 1) * 0x9e3779b1) ^
    (<u32>(epoch + 1) * 0x85ebca6b) ^ (<u32>(salt + 1) * 0xc2b2ae35);
  value = (value ^ (value >> 16)) * 0x7feb352d;
  value = (value ^ (value >> 15)) * 0x846ca68b;
  value ^= value >> 16;
  return <f32>(<f64>value / 4294967296.0);
}
