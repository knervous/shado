import type {
  ShadoWorldAuthoringDocument,
  ShadoWorldCompiledPointLight,
  ShadoWorldObjectStamp,
  ShadoWorldPointLightEmitter,
  WorldVec3,
} from './types';

/** Resolves standalone and inherited object emitters into canonical Babylon world space. */
export function resolveShadoWorldPointLights(
  document: Pick<ShadoWorldAuthoringDocument, 'lighting' | 'objects'>
): ShadoWorldCompiledPointLight[] {
  const standalone = document.lighting.pointLights.map(light => ({
    id: light.id,
    name: light.name,
    source: 'standalone' as const,
    enabled: light.enabled,
    position: [...light.offset] as WorldVec3,
    color: [...light.color] as WorldVec3,
    intensity: light.intensity,
    range: light.range,
    radius: light.radius,
    castsShadows: light.castsShadows,
    bake: light.bake,
    runtime: light.runtime,
    activation: light.activation ? { ...light.activation } : undefined,
    flicker: light.flicker ? { ...light.flicker } : undefined,
    cellId: -1,
    visibilityRegion: -1,
    phaseMask: light.phaseMask,
    tags: [...light.tags],
    metadata: { ...light.metadata },
  }));
  const prototypes = new Map(document.objects.prototypes.map(prototype => [prototype.id, prototype]));
  const attached = document.objects.stamps.flatMap(stamp => {
    const prototype = prototypes.get(stamp.prototype);
    const emitter = stamp.light ?? prototype?.light;
    if (!emitter) return [];
    return [resolveObjectEmitter(stamp, emitter)];
  });
  return [...standalone, ...attached];
}

function resolveObjectEmitter(
  stamp: ShadoWorldObjectStamp,
  emitter: ShadoWorldPointLightEmitter
): ShadoWorldCompiledPointLight {
  const offset = transformStampOffset(stamp, emitter.offset);
  return {
    id: `${stamp.id}:light`,
    name: `${stamp.id} light`,
    source: 'object',
    ownerStamp: stamp.id,
    enabled: stamp.enabled && emitter.enabled,
    position: [
      stamp.position[0] + offset[0],
      stamp.position[1] + offset[1],
      stamp.position[2] + offset[2],
    ],
    color: [...emitter.color],
    intensity: emitter.intensity,
    range: emitter.range,
    radius: emitter.radius,
    castsShadows: emitter.castsShadows,
    bake: emitter.bake,
    runtime: emitter.runtime,
    activation: emitter.activation ? { ...emitter.activation } : undefined,
    flicker: emitter.flicker ? { ...emitter.flicker } : undefined,
    cellId: -1,
    visibilityRegion: -1,
    phaseMask: stamp.phaseMask,
    tags: [...stamp.tags],
    metadata: { ...emitter.metadata },
  };
}

/** Applies Babylon's documented Y-X-Z yaw/pitch/roll order to a scaled local offset. */
export function transformStampOffset(stamp: ShadoWorldObjectStamp, offset: WorldVec3): WorldVec3 {
  const radians = stamp.rotationDegrees.map(value => value * Math.PI / 180) as WorldVec3;
  const qx = axisQuaternion(1, 0, 0, radians[0]);
  const qy = axisQuaternion(0, 1, 0, radians[1]);
  const qz = axisQuaternion(0, 0, 1, radians[2]);
  const quaternion = multiplyQuaternion(multiplyQuaternion(qy, qx), qz);
  const scaled: WorldVec3 = [
    offset[0] * stamp.scale[0],
    offset[1] * stamp.scale[1],
    offset[2] * stamp.scale[2],
  ];
  return rotateVector(quaternion, scaled);
}

type Quaternion = [number, number, number, number];

function axisQuaternion(x: number, y: number, z: number, angle: number): Quaternion {
  const half = angle * 0.5;
  const sine = Math.sin(half);
  return [x * sine, y * sine, z * sine, Math.cos(half)];
}

function multiplyQuaternion(a: Quaternion, b: Quaternion): Quaternion {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function rotateVector([x, y, z, w]: Quaternion, [vx, vy, vz]: WorldVec3): WorldVec3 {
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

/**
 * Mutable structure-of-arrays state for every runtime light in a world.
 *
 * Each eight-float packed row is the GPU ABI: position/range followed by
 * linear color times intensity/radius. A separate reducer-produced index
 * buffer selects which of these rows a draw evaluates.
 */
export class ShadoWorldLightState {
  public readonly ids: string[];
  public readonly enabled: Uint8Array;
  public readonly phaseMask: Uint32Array;
  public readonly positionX: Float32Array;
  public readonly positionY: Float32Array;
  public readonly positionZ: Float32Array;
  public readonly colorR: Float32Array;
  public readonly colorG: Float32Array;
  public readonly colorB: Float32Array;
  public readonly intensity: Float32Array;
  public readonly range: Float32Array;
  public readonly radius: Float32Array;
  public readonly packed: Float32Array;
  private readonly rows = new Map<string, number>();

  public constructor(lights: readonly ShadoWorldCompiledPointLight[]) {
    const runtimeLights = lights.filter(light => light.runtime);
    const count = runtimeLights.length;
    this.ids = runtimeLights.map(light => light.id);
    this.enabled = new Uint8Array(count);
    this.phaseMask = new Uint32Array(count);
    this.positionX = new Float32Array(count);
    this.positionY = new Float32Array(count);
    this.positionZ = new Float32Array(count);
    this.colorR = new Float32Array(count);
    this.colorG = new Float32Array(count);
    this.colorB = new Float32Array(count);
    this.intensity = new Float32Array(count);
    this.range = new Float32Array(count);
    this.radius = new Float32Array(count);
    // Babylon/WebGPU reject zero-byte storage buffers; retain one inert row.
    this.packed = new Float32Array(Math.max(1, count) * 8);
    runtimeLights.forEach((light, row) => {
      this.rows.set(light.id, row);
      this.enabled[row] = light.enabled ? 1 : 0;
      this.phaseMask[row] = light.phaseMask >>> 0;
      this.positionX[row] = light.position[0];
      this.positionY[row] = light.position[1];
      this.positionZ[row] = light.position[2];
      this.colorR[row] = light.color[0];
      this.colorG[row] = light.color[1];
      this.colorB[row] = light.color[2];
      this.intensity[row] = light.intensity;
      this.range[row] = light.range;
      this.radius[row] = light.radius;
      this.writePackedRow(row);
    });
  }

  public get count(): number {
    return this.ids.length;
  }

  public indexOf(id: string): number {
    return this.rows.get(id) ?? -1;
  }

  /** Updates a light in-place without rebuilding world/package objects. */
  public update(
    light: string | number,
    patch: Partial<Pick<ShadoWorldCompiledPointLight,
      'enabled' | 'position' | 'color' | 'intensity' | 'range' | 'radius' | 'phaseMask'>>
  ): number {
    const row = typeof light === 'number' ? light : this.indexOf(light);
    if (!Number.isInteger(row) || row < 0 || row >= this.count) {
      throw new Error(`Unknown Shado world runtime light '${String(light)}'`);
    }
    if (patch.enabled !== undefined) this.enabled[row] = patch.enabled ? 1 : 0;
    if (patch.position) {
      this.positionX[row] = finite(patch.position[0], 'position.x');
      this.positionY[row] = finite(patch.position[1], 'position.y');
      this.positionZ[row] = finite(patch.position[2], 'position.z');
    }
    if (patch.color) {
      this.colorR[row] = nonNegative(patch.color[0], 'color.r');
      this.colorG[row] = nonNegative(patch.color[1], 'color.g');
      this.colorB[row] = nonNegative(patch.color[2], 'color.b');
    }
    if (patch.intensity !== undefined) {
      this.intensity[row] = nonNegative(patch.intensity, 'intensity');
    }
    if (patch.range !== undefined) {
      const value = finite(patch.range, 'range');
      if (value <= 0) throw new Error('Shado world light range must be positive');
      this.range[row] = value;
    }
    if (patch.radius !== undefined) {
      this.radius[row] = nonNegative(patch.radius, 'radius');
    }
    if (patch.phaseMask !== undefined) this.phaseMask[row] = patch.phaseMask >>> 0;
    this.writePackedRow(row);
    return row;
  }

  private writePackedRow(row: number): void {
    const offset = row * 8;
    const strength = this.intensity[row];
    this.packed.set([
      this.positionX[row], this.positionY[row], this.positionZ[row], this.range[row],
      this.colorR[row] * strength, this.colorG[row] * strength,
      this.colorB[row] * strength, this.radius[row],
    ], offset);
  }
}

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new Error(`Shado world light ${label} must be finite`);
  return value;
}

function nonNegative(value: number, label: string): number {
  const result = finite(value, label);
  if (result < 0) throw new Error(`Shado world light ${label} must be non-negative`);
  return result;
}
