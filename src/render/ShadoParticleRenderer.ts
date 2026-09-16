import { BABYLON, type AbstractEngine, type Mesh, type Nullable, type Observer, type RawTexture, type Scene, type ShaderMaterial } from '../babylon';
import type { ShadoConcreteCtor } from '../types';
import type { ShadoParticleAtlas } from './ShadoParticleAtlas';
import type { ShadoParticleContainer } from './ShadoParticleContainer';
import type { ShadoParticleRamps } from './ShadoParticleEmitters';

export interface ShadoParticleRendererOptions {
  atlas: ShadoParticleAtlas;
  ramps: ShadoParticleRamps;
  /** Rendering group; effects usually draw after the world. */
  renderingGroupId?: number;
  /**
   * Diagnostics. `color` draws each particle's ramp colour as a solid quad, skipping the
   * atlas, which separates "particles are not reaching the screen" from "the texture is
   * empty". `grid` (WGSL) lays instances out by index, ignoring positions, and colours each
   * quad from its record: red when it has a life, green when it is alive now, blue when
   * the time uniform is set. That separates "the draw is broken" from "the data is not
   * being read".
   */
  debugView?: 'none' | 'color' | 'grid';
}

const NAME = 'shadoParticle';

/**
 * Draws every particle in a `ShadoParticleContainer` with one instanced quad.
 *
 * The vertex shader does all the per-frame work. For each slot it reads the record
 * written at birth and evaluates, against `uShadoParticleTime`:
 *
 *   position   birth + velocity*t + gravity*t^2/2, or its linear-drag closed form,
 *              plus the anchor's position when the particle follows one
 *   size       the curve's size row at age/life, times the record's size scale
 *   colour     the curve's colour row at age/life
 *   rotation   initial + spin*t, as a camera-facing quad
 *
 * Dead slots (never written, or older than their life) are moved outside clip space.
 * Blending is premultiplied: an alpha particle writes (rgb*a, a); an additive one writes
 * (rgb, 0), which the same blend state adds. So both kinds share this one draw. Additive
 * ignores alpha entirely, like a ONE/ONE blend: authored additive curves often fade by
 * colour while leaving alpha at 0.
 */
export class ShadoParticleRenderer {
  public readonly mesh: Mesh;
  public readonly material: ShaderMaterial;
  /** Seconds on the container's clock. Set before each frame renders. */
  public time = 0;
  private rampTexture: RawTexture;
  private rampVersion = -1;
  private rampHeight = 0;
  private readonly engine: AbstractEngine;
  private readonly observer: Nullable<Observer<Scene>>;
  private readonly useWGSL: boolean;

  public constructor(
    private readonly scene: Scene,
    public readonly container: ShadoParticleContainer,
    private readonly options: ShadoParticleRendererOptions
  ) {
    this.engine = scene.getEngine();
    const ctor = container.constructor as unknown as ShadoConcreteCtor;
    this.useWGSL = (this.engine as any).isWebGPU && (ctor as any).backingPreference === 'storage';
    const shaderIo = ctor.shaderIO(this.engine);
    const names = registerShadoParticleShaders(this.useWGSL, options.debugView ?? 'none');

    this.mesh = BABYLON.MeshBuilder.CreatePlane(`${NAME}Quads`, { size: 1 }, scene);
    this.mesh.alwaysSelectAsActiveMesh = true;
    this.mesh.isPickable = false;
    this.mesh.renderingGroupId = options.renderingGroupId ?? 1;
    this.mesh.forcedInstanceCount = 0;

    this.material = new BABYLON.ShaderMaterial(`${NAME}Material`, scene, names, {
      attributes: ['position', 'uv'],
      uniforms: [
        'worldViewProjection',
        'view',
        'uShadoParticleTime',
        'uShadoParticleRampWidth',
        ...(this.useWGSL ? [] : shaderIo.uniforms),
      ],
      samplers: ['uShadoParticleAtlas', 'uShadoParticleRamp', ...shaderIo.samplers],
      uniformBuffers: ['Scene'],
      needAlphaBlending: true,
      shaderLanguage: this.useWGSL ? BABYLON.ShaderLanguage.WGSL : BABYLON.ShaderLanguage.GLSL,
    });
    this.material.backFaceCulling = false;
    this.material.disableDepthWrite = true;
    this.material.alphaMode = BABYLON.Engine.ALPHA_PREMULTIPLIED_PORTERDUFF;
    this.mesh.material = this.material;

    this.rampTexture = this.createRampTexture();
    this.syncMaterial();
    this.observer = scene.onBeforeRenderObservable.add(() => this.syncMaterial());
  }

  private createRampTexture(): RawTexture {
    const ramps = this.options.ramps;
    this.rampHeight = ramps.height;
    this.rampVersion = ramps.version;
    return new BABYLON.RawTexture(
      ramps.data,
      ramps.width,
      ramps.height,
      BABYLON.Constants.TEXTUREFORMAT_RGBA,
      this.scene,
      false,
      false,
      BABYLON.Texture.NEAREST_SAMPLINGMODE
    );
  }

  private syncMaterial(): void {
    const ramps = this.options.ramps;
    if (ramps.height !== this.rampHeight) {
      this.rampTexture.dispose();
      this.rampTexture = this.createRampTexture();
    } else if (ramps.version !== this.rampVersion) {
      this.rampTexture.update(ramps.data);
      this.rampVersion = ramps.version;
    }
    this.options.atlas.flush();
    // Keyed by the scene's render id, not the engine's frame id: headless and offscreen
    // engines do not always advance `frameId`, and a stuck id means one upload ever.
    this.container.syncGpu(this.scene.getRenderId());
    this.container.bindMaterial(this.material);
    this.material.setTexture('uShadoParticleAtlas', this.options.atlas.texture);
    this.material.setTexture('uShadoParticleRamp', this.rampTexture);
    this.material.setFloat('uShadoParticleTime', this.time);
    this.material.setFloat('uShadoParticleRampWidth', ramps.width);
    this.mesh.forcedInstanceCount = this.container.drawCount;
    this.mesh.isVisible = this.container.drawCount > 0;
  }

  public dispose(): void {
    this.scene.onBeforeRenderObservable.remove(this.observer);
    this.mesh.dispose();
    this.material.dispose();
    this.rampTexture.dispose();
  }
}

/** Registers the particle shader pair for a backend and returns its store names. */
export function registerShadoParticleShaders(
  wgsl: boolean,
  debugView: 'none' | 'color' | 'grid' = 'none'
): { vertex: string; fragment: string } {
  const base = `${wgsl ? `${NAME}Wgsl` : `${NAME}Glsl`}${debugView === 'none' ? '' : `_${debugView}`}`;
  const store = wgsl ? (BABYLON.ShaderStore as any).ShadersStoreWGSL : (BABYLON.Effect as any).ShadersStore;
  const pair = wgsl ? shadoParticleWGSL(debugView) : shadoParticleGLSL(debugView);
  store[`${base}VertexShader`] = pair.vs;
  store[`${base}FragmentShader`] = pair.fs;
  return { vertex: base, fragment: base };
}

export function shadoParticleGLSL(debugView: 'none' | 'color' | 'grid' = 'none'): { vs: string; fs: string } {
  const solid = debugView !== 'none';
  const grid = debugView === 'grid';
  const vs = /* glsl */ `
precision highp float;
precision highp int;
attribute vec3 position;
attribute vec2 uv;
uniform mat4 worldViewProjection;
uniform mat4 view;
uniform float uShadoParticleTime;
uniform float uShadoParticleRampWidth;
uniform highp sampler2D uShadoParticleRamp;
#include<ShadoParticle>
#include<ShadoParticleOffsets>
#include<ShadoParticleContainerStorage>
varying vec2 vUV;
varying vec4 vColor;
varying float vLayer;
varying float vAdditive;

vec3 shadoParticlePosition(vec3 p, vec3 v, vec3 g, float k, float t) {
  if (k < 0.0001) return p + v * t + 0.5 * g * t * t;
  vec3 terminal = g / k;
  return p + terminal * t + (v - terminal) * (1.0 - exp(-k * t)) / k;
}

vec3 shadoParticleVelocity(vec3 v, vec3 g, float k, float t) {
  if (k < 0.0001) return v + g * t;
  vec3 terminal = g / k;
  return (v - terminal) * exp(-k * t) + terminal;
}

// After the first landing at \`impact\` the particle hops on its ground plane in closed form:
// each bounce reflects the velocity and scales all of it by e (NeL's bounce zone), so hop
// durations and horizontal speeds form geometric series and the hop in progress at any age
// is found with one logarithm. Drag is ignored after the first landing.
vec3 shadoParticleBounce(vec3 p, vec3 v, vec3 g, float k, float ground, float e, float impact, float age) {
  vec3 landed = shadoParticlePosition(p, v, g, k, impact);
  landed.y = ground;
  vec3 hit = shadoParticleVelocity(v, g, k, impact);
  vec3 bounced = vec3(hit.x, -hit.y, hit.z) * e;
  float s = age - impact;
  float fall = -g.y;
  if (fall <= 0.0001 || bounced.y <= 0.0001) {
    vec3 q = landed + bounced * s + 0.5 * g * s * s;
    q.y = max(q.y, ground);
    return q;
  }
  float hop = 2.0 * bounced.y / fall;
  vec2 slide = bounced.xz;
  if (e >= 0.999) {
    float tau = s - floor(s / hop) * hop;
    vec2 xz = landed.xz + slide * s;
    return vec3(xz.x, ground + bounced.y * tau - 0.5 * fall * tau * tau, xz.y);
  }
  float settle = hop / (1.0 - e);
  float e2 = e * e;
  if (e <= 0.001 || s >= settle) {
    vec2 rest = landed.xz + slide * hop / (1.0 - e2);
    return vec3(rest.x, ground, rest.y);
  }
  float n = floor(log(1.0 - s / settle) / log(e));
  float en = pow(e, n);
  float tau = s - settle * (1.0 - en);
  vec2 xz = landed.xz + slide * hop * (1.0 - en * en) / (1.0 - e2) + slide * en * tau;
  return vec3(xz.x, max(ground, ground + bounced.y * en * tau - 0.5 * fall * tau * tau), xz.y);
}

vec4 shadoParticleRamp(int row, float u) {
  float width = uShadoParticleRampWidth;
  float fx = clamp(u, 0.0, 1.0) * (width - 1.0);
  int x0 = int(floor(fx));
  int x1 = min(x0 + 1, int(width) - 1);
  return mix(
    texelFetch(uShadoParticleRamp, ivec2(x0, row), 0),
    texelFetch(uShadoParticleRamp, ivec2(x1, row), 0),
    fract(fx)
  );
}

void main(void) {
  ${grid ? `
  float gx = float(gl_InstanceID % 16) - 8.0;
  float gy = float(gl_InstanceID / 16) - 4.0;
  vUV = uv; vColor = vec4(1.0, 0.0, 0.0, 1.0); vLayer = 0.0; vAdditive = 0.0;
  gl_Position = worldViewProjection * vec4(gx * 0.5 + position.x * 0.4, gy * 0.5 + position.y * 0.4, 0.0, 1.0);
  return;` : ''}
  ShadoParticleHeader particle = ShadoParticleContainer_particles_get(gl_InstanceID);
  float life = particle.velocity.w;
  float age = uShadoParticleTime - particle.birth.w;
  vUV = vec2(0.0);
  vColor = vec4(0.0);
  vLayer = 0.0;
  vAdditive = 0.0;
  if (life <= 0.0 || age < 0.0 || age > life) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  float u = age / life;
  int row = int(particle.look.x + 0.5);
  // Each particle's seed mixes between the curve's two colours and its two sizes.
  float seed = particle.extra.w;
  vec4 color = ${solid ? 'vec4(1.0, 0.0, 0.0, 1.0)' : 'mix(shadoParticleRamp(row, u), shadoParticleRamp(row + 2, u), seed)'};
  vec4 sizes = shadoParticleRamp(row + 1, u);
  float size = ${solid ? '0.5' : 'mix(sizes.r, sizes.g, seed) * particle.extra.y'};
  float flags = particle.extra.z;
  float ground = step(1.5, flags);

  vec3 world = shadoParticlePosition(particle.birth.xyz, particle.velocity.xyz, particle.accel.xyz, particle.accel.w, age);
  if (particle.collision.z > 0.5 && particle.collision.w > 0.0 && age > particle.collision.w) {
    world = shadoParticleBounce(particle.birth.xyz, particle.velocity.xyz, particle.accel.xyz, particle.accel.w,
      particle.collision.x, particle.collision.y, particle.collision.w, age);
  }
  if (particle.extra.x > -0.5) {
    vec4 anchor = ShadoParticleContainer_anchors_get(int(particle.extra.x + 0.5));
    float ac = cos(anchor.w);
    float as_ = sin(anchor.w);
    world = vec3(world.x * ac + world.z * as_, world.y, -world.x * as_ + world.z * ac) + anchor.xyz;
  }
  float angle = particle.look.z + particle.look.w * age;
  float c = cos(angle);
  float s = sin(angle);
  vec2 corner = vec2(position.x * c - position.y * s, position.x * s + position.y * c) * size;
  vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
  vec3 up = vec3(view[0][1], view[1][1], view[2][1]);
  world += mix(right * corner.x + up * corner.y, vec3(corner.x, 0.0, corner.y), ground);

  // Every atlas image fills its own layer, so a quad samples the whole cell.
  vUV = vec2(uv.x, 1.0 - uv.y);
  vColor = color;
  vLayer = particle.look.y;
  vAdditive = mod(flags, 2.0);
  gl_Position = worldViewProjection * vec4(world, 1.0);
}
`;
  const fs = /* glsl */ `
precision highp float;
precision highp int;
varying vec2 vUV;
varying vec4 vColor;
varying float vLayer;
varying float vAdditive;
uniform highp sampler2DArray uShadoParticleAtlas;

void main(void) {
  vec4 texel = ${solid ? 'vec4(1.0)' : 'texture(uShadoParticleAtlas, vec3(vUV, floor(vLayer + 0.5)))'};
  vec4 color = texel * vColor;
  if (color.a <= 0.002 && dot(color.rgb, color.rgb) <= 0.00001) discard;
  float additive = step(0.5, vAdditive);
  gl_FragColor = vec4(color.rgb * mix(color.a, 1.0, additive), color.a * (1.0 - additive));
}
`;
  return { vs, fs };
}

export function shadoParticleWGSL(debugView: 'none' | 'color' | 'grid' = 'none'): { vs: string; fs: string } {
  const solid = debugView !== 'none';
  const grid = debugView === 'grid';
  const vs = /* wgsl */ `
attribute position: vec3f;
attribute uv: vec2f;
uniform worldViewProjection: mat4x4f;
uniform view: mat4x4f;
uniform uShadoParticleTime: f32;
uniform uShadoParticleRampWidth: f32;
var uShadoParticleRampSampler: sampler;
var uShadoParticleRamp: texture_2d<f32>;
#include<ShadoParticle>
#include<ShadoParticleContainerStorage>
varying vUV: vec2f;
varying vColor: vec4f;
varying vLayer: f32;
varying vAdditive: f32;

fn shadoParticlePosition(p: vec3f, v: vec3f, g: vec3f, k: f32, t: f32) -> vec3f {
  if (k < 0.0001) {
    return p + v * t + 0.5 * g * t * t;
  }
  let terminal = g / k;
  return p + terminal * t + (v - terminal) * (1.0 - exp(-k * t)) / k;
}

fn shadoParticleVelocity(v: vec3f, g: vec3f, k: f32, t: f32) -> vec3f {
  if (k < 0.0001) {
    return v + g * t;
  }
  let terminal = g / k;
  return (v - terminal) * exp(-k * t) + terminal;
}

// See the GLSL twin: closed-form hops on the ground plane after the first landing.
fn shadoParticleBounce(p: vec3f, v: vec3f, g: vec3f, k: f32, ground: f32, e: f32, impact: f32, age: f32) -> vec3f {
  var landed = shadoParticlePosition(p, v, g, k, impact);
  landed.y = ground;
  let hit = shadoParticleVelocity(v, g, k, impact);
  let bounced = vec3f(hit.x, -hit.y, hit.z) * e;
  let s = age - impact;
  let fall = -g.y;
  if (fall <= 0.0001 || bounced.y <= 0.0001) {
    var q = landed + bounced * s + 0.5 * g * s * s;
    q.y = max(q.y, ground);
    return q;
  }
  let hop = 2.0 * bounced.y / fall;
  let slide = bounced.xz;
  if (e >= 0.999) {
    let tau = s - floor(s / hop) * hop;
    let xz = landed.xz + slide * s;
    return vec3f(xz.x, ground + bounced.y * tau - 0.5 * fall * tau * tau, xz.y);
  }
  let settle = hop / (1.0 - e);
  let e2 = e * e;
  if (e <= 0.001 || s >= settle) {
    let rest = landed.xz + slide * hop / (1.0 - e2);
    return vec3f(rest.x, ground, rest.y);
  }
  let n = floor(log(1.0 - s / settle) / log(e));
  let en = pow(e, n);
  let tau = s - settle * (1.0 - en);
  let xz = landed.xz + slide * hop * (1.0 - en * en) / (1.0 - e2) + slide * en * tau;
  return vec3f(xz.x, max(ground, ground + bounced.y * en * tau - 0.5 * fall * tau * tau), xz.y);
}

fn shadoParticleRamp(row: i32, u: f32) -> vec4f {
  let width = uniforms.uShadoParticleRampWidth;
  let fx = clamp(u, 0.0, 1.0) * (width - 1.0);
  let x0 = i32(floor(fx));
  let x1 = min(x0 + 1, i32(width) - 1);
  return mix(
    textureLoad(uShadoParticleRamp, vec2i(x0, row), 0),
    textureLoad(uShadoParticleRamp, vec2i(x1, row), 0),
    fract(fx)
  );
}

@vertex
fn main(input: VertexInputs) -> FragmentInputs {
  ${grid ? `
  let gi = i32(vertexInputs.instanceIndex);
  let gx = f32(gi % 16) - 8.0;
  let gy = f32(gi / 16) - 4.0;
  vertexOutputs.vUV = vertexInputs.uv;
  let probe = ShadoParticleContainer_particles_get(gi);
  let probeAge = uniforms.uShadoParticleTime - probe.birth.w;
  vertexOutputs.vColor = vec4f(
    select(0.0, 1.0, probe.velocity.w > 0.0),
    select(0.0, 1.0, probeAge >= 0.0 && probeAge <= probe.velocity.w),
    select(0.0, 1.0, uniforms.uShadoParticleTime > 0.0),
    1.0
  );
  vertexOutputs.vLayer = 0.0;
  vertexOutputs.vAdditive = 0.0;
  vertexOutputs.position = uniforms.worldViewProjection * vec4f(gx * 0.5 + vertexInputs.position.x * 0.4, gy * 0.5 + vertexInputs.position.y * 0.4, 0.0, 1.0);
  if (true) { return vertexOutputs; }` : ''}
  let particle = ShadoParticleContainer_particles_get(i32(vertexInputs.instanceIndex));
  let life = particle.velocity.w;
  let age = uniforms.uShadoParticleTime - particle.birth.w;
  vertexOutputs.vUV = vec2f(0.0);
  vertexOutputs.vColor = vec4f(0.0);
  vertexOutputs.vLayer = 0.0;
  vertexOutputs.vAdditive = 0.0;
  if (life <= 0.0 || age < 0.0 || age > life) {
    vertexOutputs.position = vec4f(2.0, 2.0, 2.0, 1.0);
    return vertexOutputs;
  }
  let u = age / life;
  let row = i32(particle.look.x + 0.5);
  // Each particle's seed mixes between the curve's two colours and its two sizes.
  let seed = particle.extra.w;
  let color = ${solid ? 'vec4f(1.0, 0.0, 0.0, 1.0)' : 'mix(shadoParticleRamp(row, u), shadoParticleRamp(row + 2, u), seed)'};
  let sizes = shadoParticleRamp(row + 1, u);
  let size = ${solid ? '0.5' : 'mix(sizes.r, sizes.g, seed) * particle.extra.y'};
  let flags = particle.extra.z;
  let ground = step(1.5, flags);

  var world = shadoParticlePosition(particle.birth.xyz, particle.velocity.xyz, particle.accel.xyz, particle.accel.w, age);
  if (particle.collision.z > 0.5 && particle.collision.w > 0.0 && age > particle.collision.w) {
    world = shadoParticleBounce(particle.birth.xyz, particle.velocity.xyz, particle.accel.xyz, particle.accel.w,
      particle.collision.x, particle.collision.y, particle.collision.w, age);
  }
  if (particle.extra.x > -0.5) {
    let anchor = ShadoParticleContainer_anchors_get(i32(particle.extra.x + 0.5));
    let ac = cos(anchor.w);
    let asn = sin(anchor.w);
    world = vec3f(world.x * ac + world.z * asn, world.y, -world.x * asn + world.z * ac) + anchor.xyz;
  }
  let angle = particle.look.z + particle.look.w * age;
  let c = cos(angle);
  let s = sin(angle);
  let corner = vec2f(
    vertexInputs.position.x * c - vertexInputs.position.y * s,
    vertexInputs.position.x * s + vertexInputs.position.y * c
  ) * size;
  let right = vec3f(uniforms.view[0].x, uniforms.view[1].x, uniforms.view[2].x);
  let up = vec3f(uniforms.view[0].y, uniforms.view[1].y, uniforms.view[2].y);
  world = world + mix(right * corner.x + up * corner.y, vec3f(corner.x, 0.0, corner.y), ground);

  // Every atlas image fills its own layer, so a quad samples the whole cell.
  vertexOutputs.vUV = vec2f(vertexInputs.uv.x, 1.0 - vertexInputs.uv.y);
  vertexOutputs.vColor = color;
  vertexOutputs.vLayer = particle.look.y;
  vertexOutputs.vAdditive = flags - 2.0 * floor(flags / 2.0);
  vertexOutputs.position = uniforms.worldViewProjection * vec4f(world, 1.0);
}
`;
  const fs = /* wgsl */ `
varying vUV: vec2f;
varying vColor: vec4f;
varying vLayer: f32;
varying vAdditive: f32;
var uShadoParticleAtlasSampler: sampler;
var uShadoParticleAtlas: texture_2d_array<f32>;

@fragment
fn main(input: FragmentInputs) -> FragmentOutputs {
  let sampled = textureSample(
    uShadoParticleAtlas,
    uShadoParticleAtlasSampler,
    fragmentInputs.vUV,
    i32(floor(fragmentInputs.vLayer + 0.5))
  );
  let texel = ${solid ? 'vec4f(1.0)' : 'sampled'};
  let color = texel * fragmentInputs.vColor;
  if (color.a <= 0.002 && dot(color.rgb, color.rgb) <= 0.00001) {
    discard;
  }
  let additive = step(0.5, fragmentInputs.vAdditive);
  fragmentOutputs.color = vec4f(color.rgb * mix(color.a, 1.0, additive), color.a * (1.0 - additive));
}
`;
  return { vs, fs };
}
