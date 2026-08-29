import * as BABYLON from '@babylonjs/core';

/**
 * The demo's actor material: matrix VAT skinning, an outfit atlas layer per
 * submesh, a per-instance appearance slice with tint, and a compact per-module
 * draw list.
 *
 * Still the *thin* version of a real game's actor shader - no cloak/helm
 * atlases, no picking or selection glow, no world light field - but it does now
 * carry the **material response**, because without it the wardrobe demo could
 * only show meshes. A dye was a flat multiply over the whole garment, so it
 * repainted the face and hands too and there was no way to show a faction that
 * differs only in colour, which is most of them.
 *
 * The response is one mask sampled per texel alongside the albedo: `r` is skin,
 * `g` is the heraldic charge, `b` is what a dye may touch. Three channels turn
 * one atlas layer into an outfit, a complexion and a banner.
 *
 * The appearance var-array is therefore `submeshCount + 2` wide, not
 * `submeshCount`: the two extra rows are per-entity rather than per-submesh -
 * complexion first, heraldry second - which is the same stride the game uses,
 * for the same reason. A complexion belongs to a person, not to their sleeve.
 *
 * `uShadoVisibleIndices` is the seam that matters. The arena binds its global
 * visible list there; a module's own compact list replaces it per draw, which
 * is why every module needs its own material - on WebGPU that binding lives in
 * the material's draw context and a shared material would be last-bind-wins.
 */

const VERTEX_GLSL = `
precision highp float;

attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
attribute vec2 submeshData;

flat varying int vSlice;
varying vec2 vUV;
varying vec3 vTint;
varying vec3 vWorldNormal;
varying float vNeutralize;
varying vec3 vSkin;
varying vec4 vDevice;

uniform mat4 worldViewProjection;
uniform int uSubmeshCount;
uniform int uShadoVisibleCount;
uniform highp sampler2D uShadoVisibleIndices;
uniform int uShadoVisibleIndexTexWidth;

#define THIN_INSTANCES
#define INSTANCES
#define BAKED_VERTEX_ANIMATION_TEXTURE
#include<bonesDeclaration>
#undef INSTANCES
#include<bakedVertexAnimationDeclaration>
#define INSTANCES
#include<HumWardrobeActor>
#include<HumWardrobeActorOffsets>
#include<HumWardrobeContainerStorage>

int visibleActorIndex(int drawIndex) {
  int texel = drawIndex / 4;
  vec4 packed = texelFetch(
    uShadoVisibleIndices,
    ivec2(texel % uShadoVisibleIndexTexWidth, texel / uShadoVisibleIndexTexWidth),
    0
  );
  int lane = drawIndex - texel * 4;
  return int((lane == 0 ? packed.x : lane == 1 ? packed.y : lane == 2 ? packed.z : packed.w) + 0.5);
}

vec3 rotateByQuat(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
  int drawIndex = gl_InstanceID;
  if (drawIndex >= uShadoVisibleCount) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  int sourceIndex = visibleActorIndex(drawIndex);
  HumWardrobeActorHeader actor = HumWardrobeContainer_instances_get(sourceIndex);

  vec4 anim = actor.animationBuffer;
  float totalFrames = anim.y - anim.x + 1.0;
  float vatTime = bakedVertexAnimationTime * anim.w / totalFrames;
  float correction = vatTime < 1.0 ? 0.0 : 1.0;
  float frameCount = totalFrames - correction;
  float frame = floor(mod(fract(vatTime) * frameCount + anim.z, frameCount)) + anim.x + correction;

  mat4 influence = readMatrixFromRawSamplerVAT(bakedVertexAnimationTexture, matricesIndices[0], frame) * matricesWeights[0];
  #if NUM_BONE_INFLUENCERS > 1
    influence += readMatrixFromRawSamplerVAT(bakedVertexAnimationTexture, matricesIndices[1], frame) * matricesWeights[1];
  #endif
  #if NUM_BONE_INFLUENCERS > 2
    influence += readMatrixFromRawSamplerVAT(bakedVertexAnimationTexture, matricesIndices[2], frame) * matricesWeights[2];
  #endif
  #if NUM_BONE_INFLUENCERS > 3
    influence += readMatrixFromRawSamplerVAT(bakedVertexAnimationTexture, matricesIndices[3], frame) * matricesWeights[3];
  #endif

  vUV = uv;
  int submeshIndex = int(submeshData.y + 0.5);
  // Stride is submeshCount + 2. The two rows past the submeshes are the
  // entity's own: complexion, then heraldry.
  int stride = uSubmeshCount + 2;
  int base = sourceIndex * stride;
  vec4 appearance = HumWardrobeContainer_appearance_get(submeshIndex + base);
  vec4 complexion = HumWardrobeContainer_appearance_get(uSubmeshCount + base);
  vec4 heraldry = HumWardrobeContainer_appearance_get(uSubmeshCount + 1 + base);
  vSlice = int(appearance.x);
  vTint = appearance.yzw;
  vNeutralize = complexion.x;
  vSkin = complexion.yzw;
  vDevice = heraldry;

  if (vSlice < 0) {
    // The variant this actor is not wearing. Retained so an unmigrated path
    // still works; inside a module draw it is simply never reached.
    gl_Position = vec4(0.0);
    return;
  }
  vec3 skinned = (influence * vec4(position, 1.0)).xyz * actor.translation.w;
  vec3 world = rotateByQuat(actor.rotation, skinned) + actor.translation.xyz;
  vec3 skinnedNormal = normalize((influence * vec4(normal, 0.0)).xyz);
  vWorldNormal = normalize(rotateByQuat(actor.rotation, skinnedNormal));
  gl_Position = worldViewProjection * vec4(world, 1.0);
}
`;

const FRAGMENT_GLSL = `
precision highp float;

flat varying int vSlice;
varying vec2 vUV;
varying vec3 vTint;
varying vec3 vWorldNormal;
varying float vNeutralize;
varying vec3 vSkin;
varying vec4 vDevice;

uniform highp sampler2DArray uAtlasArray;
uniform highp sampler2DArray uResponseArray;
uniform vec3 uLightDirection;
uniform float uHasResponse;

void main() {
  float slice = clamp(float(vSlice), 0.0, float(textureSize(uAtlasArray, 0).z - 1));
  vec4 base = texture(uAtlasArray, vec3(vUV, slice));

  // No mask means wholly dyeable and no skin, which is the safe reading: it
  // recolours and nothing is protected.
  vec3 response = mix(
    vec3(0.0, 0.0, 1.0),
    texture(uResponseArray, vec3(vUV, slice)).rgb,
    uHasResponse
  );
  float skinMask = response.r;
  float deviceMask = response.g;
  float tintMask = response.b;

  // A dye can only darken and shift, so a palette that wants its own hue has to
  // start from a neutral sheet rather than the donor's. Collapsing to luminance
  // is that neutral, and it happens only where the mask says dyeable.
  float outfitLuminance = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
  vec3 outfitBase = mix(base.rgb, vec3(outfitLuminance), clamp(vNeutralize * tintMask, 0.0, 1.0));
  vec3 surfaceTint = mix(vec3(1.0), vTint, tintMask);
  vec3 skinTone = mix(vec3(1.0), vSkin, skinMask);

  // The charge keeps some of the cloth's own shading so a banner reads as
  // painted onto fabric rather than pasted over it.
  float device = smoothstep(0.05, 0.95, deviceMask) * vDevice.x;
  float chargeShade = mix(0.95, clamp(outfitLuminance * 1.6 + 0.25, 0.0, 1.6), 0.35);
  vec3 charge = vDevice.yzw * chargeShade;

  vec3 diffuse = mix(outfitBase * surfaceTint, charge, clamp(device, 0.0, 1.0)) * skinTone;
  vec3 n = normalize(vWorldNormal);
  float lambert = max(dot(n, uLightDirection), 0.0) * 0.75 + 0.35;
  gl_FragColor = vec4(diffuse * lambert, 1.0);
}
`;

const VERTEX_WGSL = `
attribute position: vec3f;
attribute normal: vec3f;
attribute uv: vec2f;
attribute submeshData: vec2f;

flat varying vSlice: i32;
varying vUV: vec2f;
varying vTint: vec3f;
varying vWorldNormal: vec3f;
varying vNeutralize: f32;
varying vSkin: vec3f;
varying vDevice: vec4f;

uniform worldViewProjection: mat4x4f;
uniform uSubmeshCount: i32;
uniform uShadoVisibleCount: i32;
var<storage, read> uShadoVisibleIndices: array<u32>;

#include<HumWardrobeActor>
#include<HumWardrobeContainerStorage>

var bakedVertexAnimationTexture: texture_2d<f32>;
uniform bakedVertexAnimationTextureSizeInverted: vec2f;
uniform bakedVertexAnimationTime: f32;
attribute matricesIndices: vec4f;
attribute matricesWeights: vec4f;

fn readMatrixVAT(index: f32, frame: f32) -> mat4x4f {
  let offset = i32(index) * 4;
  let frameUV = i32(frame) * 4;
  let m0 = textureLoad(bakedVertexAnimationTexture, vec2i(offset + 0, frameUV / 4), 0);
  let m1 = textureLoad(bakedVertexAnimationTexture, vec2i(offset + 1, frameUV / 4), 0);
  let m2 = textureLoad(bakedVertexAnimationTexture, vec2i(offset + 2, frameUV / 4), 0);
  let m3 = textureLoad(bakedVertexAnimationTexture, vec2i(offset + 3, frameUV / 4), 0);
  return mat4x4f(m0, m1, m2, m3);
}

fn rotateByQuat(q: vec4f, v: vec3f) -> vec3f {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@vertex
fn main(input: VertexInputs) -> FragmentInputs {
  let drawIndex = i32(vertexInputs.instanceIndex);
  if (drawIndex >= uniforms.uShadoVisibleCount) {
    vertexOutputs.position = vec4f(2.0, 2.0, 2.0, 1.0);
    return vertexOutputs;
  }
  let sourceIndex = i32(uShadoVisibleIndices[drawIndex]);
  let actor = HumWardrobeContainer_instances_get(sourceIndex);

  let anim = actor.animationBuffer;
  let totalFrames = anim.y - anim.x + 1.0;
  let vatTime = uniforms.bakedVertexAnimationTime * anim.w / totalFrames;
  let correction = select(0.0, 1.0, vatTime >= 1.0);
  let frameCount = totalFrames - correction;
  let frame = floor((vatTime - floor(vatTime)) * frameCount + anim.z) % frameCount + anim.x + correction;

  var influence = readMatrixVAT(vertexInputs.matricesIndices[0], frame) * vertexInputs.matricesWeights[0];
  influence += readMatrixVAT(vertexInputs.matricesIndices[1], frame) * vertexInputs.matricesWeights[1];
  influence += readMatrixVAT(vertexInputs.matricesIndices[2], frame) * vertexInputs.matricesWeights[2];
  influence += readMatrixVAT(vertexInputs.matricesIndices[3], frame) * vertexInputs.matricesWeights[3];

  vertexOutputs.vUV = vertexInputs.uv;
  let submeshIndex = i32(vertexInputs.submeshData.y + 0.5);
  let stride = uniforms.uSubmeshCount + 2;
  let arrayBase = sourceIndex * stride;
  let appearance = HumWardrobeContainer_appearance_get(submeshIndex + arrayBase);
  let complexion = HumWardrobeContainer_appearance_get(uniforms.uSubmeshCount + arrayBase);
  let heraldry = HumWardrobeContainer_appearance_get(uniforms.uSubmeshCount + 1 + arrayBase);
  vertexOutputs.vSlice = i32(appearance.x);
  vertexOutputs.vTint = appearance.yzw;
  vertexOutputs.vNeutralize = complexion.x;
  vertexOutputs.vSkin = complexion.yzw;
  vertexOutputs.vDevice = heraldry;

  if (vertexOutputs.vSlice < 0) {
    vertexOutputs.position = vec4f(0.0);
    return vertexOutputs;
  }
  let skinned = (influence * vec4f(vertexInputs.position, 1.0)).xyz * actor.translation.w;
  let world = rotateByQuat(actor.rotation, skinned) + actor.translation.xyz;
  let skinnedNormal = normalize((influence * vec4f(vertexInputs.normal, 0.0)).xyz);
  vertexOutputs.vWorldNormal = normalize(rotateByQuat(actor.rotation, skinnedNormal));
  vertexOutputs.position = uniforms.worldViewProjection * vec4f(world, 1.0);
}
`;

const FRAGMENT_WGSL = `
flat varying vSlice: i32;
varying vUV: vec2f;
varying vTint: vec3f;
varying vWorldNormal: vec3f;
varying vNeutralize: f32;
varying vSkin: vec3f;
varying vDevice: vec4f;

var uAtlasArraySampler: sampler;
var uAtlasArray: texture_2d_array<f32>;
var uResponseArraySampler: sampler;
var uResponseArray: texture_2d_array<f32>;
uniform uLightDirection: vec3f;
uniform uHasResponse: f32;

@fragment
fn main(input: FragmentInputs) -> FragmentOutputs {
  let layers = i32(textureNumLayers(uAtlasArray));
  let slice = clamp(fragmentInputs.vSlice, 0, layers - 1);
  let base = textureSample(uAtlasArray, uAtlasArraySampler, fragmentInputs.vUV, slice);

  let sampled = textureSample(uResponseArray, uResponseArraySampler, fragmentInputs.vUV, slice).rgb;
  let response = mix(vec3f(0.0, 0.0, 1.0), sampled, uniforms.uHasResponse);
  let skinMask = response.r;
  let deviceMask = response.g;
  let tintMask = response.b;

  let outfitLuminance = dot(base.rgb, vec3f(0.2126, 0.7152, 0.0722));
  let outfitBase = mix(base.rgb, vec3f(outfitLuminance), clamp(fragmentInputs.vNeutralize * tintMask, 0.0, 1.0));
  let surfaceTint = mix(vec3f(1.0), fragmentInputs.vTint, vec3f(tintMask));
  let skinTone = mix(vec3f(1.0), fragmentInputs.vSkin, vec3f(skinMask));

  let device = smoothstep(0.05, 0.95, deviceMask) * fragmentInputs.vDevice.x;
  let chargeShade = mix(0.95, clamp(outfitLuminance * 1.6 + 0.25, 0.0, 1.6), 0.35);
  let charge = fragmentInputs.vDevice.yzw * chargeShade;

  let diffuse = mix(outfitBase * surfaceTint, charge, vec3f(clamp(device, 0.0, 1.0))) * skinTone;
  let n = normalize(fragmentInputs.vWorldNormal);
  let lambert = max(dot(n, uniforms.uLightDirection), 0.0) * 0.75 + 0.35;
  fragmentOutputs.color = vec4f(diffuse * lambert, 1.0);
}
`;

BABYLON.Effect.ShadersStore['humWardrobeVertexShader'] = VERTEX_GLSL;
BABYLON.Effect.ShadersStore['humWardrobeFragmentShader'] = FRAGMENT_GLSL;
BABYLON.ShaderStore.ShadersStoreWGSL['humWardrobeVertexShader'] = VERTEX_WGSL;
BABYLON.ShaderStore.ShadersStoreWGSL['humWardrobeFragmentShader'] = FRAGMENT_WGSL;

export type HumWardrobeMaterialDeps = {
  container: any;
  atlas: BABYLON.BaseTexture;
  /** The material-response mask array; null falls back to "wholly dyeable". */
  response: BABYLON.BaseTexture | null;
  vatTexture: BABYLON.BaseTexture;
  submeshCount: number;
  /** Seconds, advanced by the playground. */
  time: () => number;
  lightDirection: BABYLON.Vector3;
  /**
   * Uploads and binds this module's compact actor list, returning how many
   * instances the draw should submit. This is the whole point of the pattern:
   * the arena's global visible list is bound first, then replaced here by the
   * subset that actually wears this module.
   */
  bindSelection: (target: any) => number;
};

/**
 * One material per module. They share a compiled effect when the defines
 * match, so the extra materials cost bind-group state, not shader compiles.
 */
export function createHumWardrobeMaterial(
  scene: BABYLON.Scene,
  name: string,
  deps: HumWardrobeMaterialDeps
): BABYLON.ShaderMaterial {
  const engine = scene.getEngine();
  const useStorage = (engine as any).isWebGPU;
  const shaderIO = deps.container.constructor.shaderIO(engine);

  const material = new BABYLON.ShaderMaterial(
    name,
    scene,
    { vertex: 'humWardrobe', fragment: 'humWardrobe' },
    {
      attributes: ['position', 'normal', 'uv', 'submeshData'],
      uniforms: [
        'worldViewProjection',
        'uSubmeshCount',
        'uShadoVisibleCount',
        'uLightDirection',
        'uHasResponse',
        'bakedVertexAnimationTextureSizeInverted',
        'bakedVertexAnimationTime',
        ...(useStorage ? [] : ['uShadoVisibleIndexTexWidth', ...shaderIO.uniforms]),
      ],
      samplers: [
        'uAtlasArray',
        'uResponseArray',
        'bakedVertexAnimationTexture',
        ...(useStorage ? [] : ['uShadoVisibleIndices', ...shaderIO.samplers]),
      ],
      defines: ['INSTANCES', 'THIN_INSTANCES', 'BAKED_VERTEX_ANIMATION_TEXTURE'],
      shaderLanguage: useStorage ? BABYLON.ShaderLanguage.WGSL : BABYLON.ShaderLanguage.GLSL,
    },
    true
  );

  // WebGPU builds the bind group before the first draw, so seed the arena here
  // rather than waiting for onBind.
  deps.container.bindMaterial(material);

  material.onBind = (mesh) => {
    const effect = material.getEffect();
    if (!effect) return;
    deps.container.syncGpu?.(0);
    // Binds the arena and, with it, the pool's global visible list...
    deps.container.bind(effect);
    // ...which this module's own list then replaces for this draw.
    const drawn = deps.bindSelection(effect);
    (mesh as BABYLON.Mesh).forcedInstanceCount = drawn;
    effect.setInt('uShadoVisibleCount', drawn);
    effect.setInt('uSubmeshCount', deps.submeshCount);
    effect.setTexture('uAtlasArray', deps.atlas);
    // The sampler must be bound even when there is no mask: an unbound array
    // sampler is a validation error on WebGPU, so the albedo stands in and
    // `uHasResponse` discards what it read.
    effect.setTexture('uResponseArray', deps.response ?? deps.atlas);
    effect.setFloat('uHasResponse', deps.response ? 1 : 0);
    effect.setTexture('bakedVertexAnimationTexture', deps.vatTexture);
    effect.setVector2(
      'bakedVertexAnimationTextureSizeInverted',
      new BABYLON.Vector2(
        1 / deps.vatTexture.getSize().width,
        1 / deps.vatTexture.getSize().height
      )
    );
    effect.setFloat('bakedVertexAnimationTime', deps.time());
    effect.setVector3('uLightDirection', deps.lightDirection);
  };

  material.onError = (effect, errors) => {
    if (errors) console.error('[HumWardrobe] shader error', errors);
  };
  return material;
}
