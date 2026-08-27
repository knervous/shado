import * as BABYLON from '@babylonjs/core';

/**
 * The demo's actor material: matrix VAT skinning, an outfit atlas layer per
 * submesh, a per-instance appearance slice with tint, and a compact per-module
 * draw list.
 *
 * This is deliberately the *thin* version of a real game's actor shader - no
 * material response, no cloak/helm atlases, no picking or selection glow, no
 * world light field. What survives is exactly what the module-draw pattern
 * needs, so the file reads as the contract rather than as one game's renderer.
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
  vec4 appearance = HumWardrobeContainer_appearance_get(submeshIndex + sourceIndex * uSubmeshCount);
  vSlice = int(appearance.x);
  vTint = appearance.yzw;

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

uniform highp sampler2DArray uAtlasArray;
uniform vec3 uLightDirection;

void main() {
  float slice = clamp(float(vSlice), 0.0, float(textureSize(uAtlasArray, 0).z - 1));
  vec4 base = texture(uAtlasArray, vec3(vUV, slice));
  vec3 n = normalize(vWorldNormal);
  float lambert = max(dot(n, uLightDirection), 0.0) * 0.75 + 0.35;
  gl_FragColor = vec4(base.rgb * vTint * lambert, 1.0);
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
  let appearance = HumWardrobeContainer_appearance_get(
    submeshIndex + sourceIndex * uniforms.uSubmeshCount
  );
  vertexOutputs.vSlice = i32(appearance.x);
  vertexOutputs.vTint = appearance.yzw;

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

var uAtlasArraySampler: sampler;
var uAtlasArray: texture_2d_array<f32>;
uniform uLightDirection: vec3f;

@fragment
fn main(input: FragmentInputs) -> FragmentOutputs {
  let layers = i32(textureNumLayers(uAtlasArray));
  let slice = clamp(fragmentInputs.vSlice, 0, layers - 1);
  let base = textureSample(uAtlasArray, uAtlasArraySampler, fragmentInputs.vUV, slice);
  let n = normalize(fragmentInputs.vWorldNormal);
  let lambert = max(dot(n, uniforms.uLightDirection), 0.0) * 0.75 + 0.35;
  fragmentOutputs.color = vec4f(base.rgb * fragmentInputs.vTint * lambert, 1.0);
}
`;

BABYLON.Effect.ShadersStore['humWardrobeVertexShader'] = VERTEX_GLSL;
BABYLON.Effect.ShadersStore['humWardrobeFragmentShader'] = FRAGMENT_GLSL;
BABYLON.ShaderStore.ShadersStoreWGSL['humWardrobeVertexShader'] = VERTEX_WGSL;
BABYLON.ShaderStore.ShadersStoreWGSL['humWardrobeFragmentShader'] = FRAGMENT_WGSL;

export type HumWardrobeMaterialDeps = {
  container: any;
  atlas: BABYLON.BaseTexture;
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
        'bakedVertexAnimationTextureSizeInverted',
        'bakedVertexAnimationTime',
        ...(useStorage ? [] : ['uShadoVisibleIndexTexWidth', ...shaderIO.uniforms]),
      ],
      samplers: [
        'uAtlasArray',
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
