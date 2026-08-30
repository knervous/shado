import { BABYLON } from '../../babylon';
import type { Mesh, Scene, Texture } from '../../babylon';
import type { ShadoWorldGrassFieldPackage } from '../../world/types';
import type { ShadoFoliagePlugin } from './plugins';

/** Texels of coverage data per cell: 64 floats of 16 packed bits each. */
const COVERAGE_TEXELS = 16;
const COVERAGE_BITS_PER_FLOAT = 16;
/**
 * Height samples per cell, as an `(resolution + 2)²` grid.
 *
 * The ring is the neighbouring cells' edge samples. Without it a blade near a
 * cell border interpolates against that cell's own clamped edge value while the
 * blade just across the border interpolates against a different one, and the
 * ground visibly steps. The ring makes the sample grid uniform across borders,
 * so the interpolated height is continuous.
 */
const HEIGHT_RING = 1;
export const GRASS_FIELD_TEXELS_PER_CELL = 48;
/**
 * One texel of per-cell metadata after the height grid: two [0,1) seeds from a
 * proper integer hash of the cell coordinate, computed on the CPU in f64.
 *
 * The blade shader must never hash world positions itself: the classic
 * fract(sin(worldPos)*43758) collapses on f32 GPUs once the argument reaches a
 * few hundred, and each cell then draws its blade heights and leans from a
 * differently *biased* distribution — visible as square tonal seams at every
 * cell border.
 */
const CELL_SEED_SALT = 0x9e3779b9;

export type ShadoGrassBladesConfig = {
  cellSize: number;
  /** Coverage raster resolution per cell edge, normally 32. */
  coverageResolution: number;
  /** Ground-height sample resolution per cell edge, normally 8. */
  heightResolution: number;
  /** Blades generated per cell. This is the density knob; nothing is stored. */
  bladesPerCell: number;
  minHeight: number;
  maxHeight: number;
  bladeWidth: number;
  /**
   * How far a blade tip arcs from vertical, as a fraction of its height.
   * Straight blades read as spikes; the arc is what makes them grass.
   */
  lean?: number;
  /**
   * Extra width applied only when a blade turns edge-on to the eye, as a
   * fraction of its own width. Opaque ribbons are one-dimensional side-on and
   * would otherwise wink out; this keeps the field's density stable as the
   * camera turns.
   */
  edgeOnWidth?: number;
};

/**
 * Derives grass blades in the vertex shader from a coverage field.
 *
 * One instance is one cell. The patch mesh supplies only blade topology, so
 * blade *count* is geometry and blade *placement* is a low-discrepancy sequence
 * offset by the cell — stable across streaming, and free to store.
 *
 * The blade is a cubic bezier from root to tip. That is what makes the normal
 * correct: the surface normal comes from the curve tangent per vertex, so a
 * bent blade lights as a bent blade. Reading a constant normal off the patch
 * mesh would light every blade in the world identically.
 *
 * This plugin sets `shadoFoliageWorld` absolutely rather than displacing it, so
 * it must run before any plugin that displaces. It also overwrites the
 * per-instance character locals so wind and tint vary per blade, not per cell.
 */
export function shadoGrassBlades(config: ShadoGrassBladesConfig): ShadoFoliagePlugin {
  const {
    cellSize,
    coverageResolution,
    heightResolution,
    bladesPerCell,
    minHeight,
    maxHeight,
    bladeWidth,
  } = config;
  if (!(cellSize > 0)) throw new Error('grassBlades.cellSize must be positive');
  if (!(bladesPerCell > 0)) throw new Error('grassBlades.bladesPerCell must be positive');
  if (!(minHeight > 0) || minHeight > maxHeight) {
    throw new Error('grassBlades.minHeight must be positive and not exceed maxHeight');
  }

  const shape = new BABYLON.Vector4(cellSize, coverageResolution, heightResolution, bladesPerCell);
  const size = new BABYLON.Vector4(
    minHeight,
    maxHeight - minHeight,
    bladeWidth,
    config.lean ?? 0.55
  );
  const detail = new BABYLON.Vector4(config.edgeOnWidth ?? 0.9, 0, 0, 0);

  const GLSL_HELPERS = `
uniform vec4 uShadoGrassShape;
uniform vec4 uShadoGrassSize;
uniform vec4 uShadoGrassDetail;
uniform highp sampler2D uShadoGrassField;
attribute vec2 aGrassBlade;

/*
 * Hoskins-style float hash. Inputs must stay small — callers feed exact
 * values in [0, ~300), never world coordinates.
 */
float Shado_grassHash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float Shado_grassFieldValue(int texel, int lane, int row) {
  vec4 packed = texelFetch(uShadoGrassField, ivec2(texel, row), 0);
  return lane == 0 ? packed.x : lane == 1 ? packed.y : lane == 2 ? packed.z : packed.w;
}

/** Absolute ground height at a sample of the ring-extended grid. */
float Shado_grassGround(int x, int z, int extended, int row) {
  int sampleIndex = z * extended + x;
  return Shado_grassFieldValue(
    ${COVERAGE_TEXELS} + sampleIndex / 4,
    sampleIndex - (sampleIndex / 4) * 4,
    row
  );
}

vec3 Shado_grassBezier(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t) {
  float u = 1.0 - t;
  return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3;
}

vec3 Shado_grassBezierTangent(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t) {
  float u = 1.0 - t;
  return 3.0 * u * u * (p1 - p0) + 6.0 * u * t * (p2 - p1) + 3.0 * t * t * (p3 - p2);
}`;

  const WGSL_HELPERS = `
uniform uShadoGrassShape: vec4f;
uniform uShadoGrassSize: vec4f;
uniform uShadoGrassDetail: vec4f;
var uShadoGrassFieldSampler: sampler;
var uShadoGrassField: texture_2d<f32>;
attribute aGrassBlade: vec2f;

fn Shado_grassHash(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, vec3f(p3.y, p3.z, p3.x) + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn Shado_grassFieldValue(texel: i32, lane: i32, row: i32) -> f32 {
  let packed = textureLoad(uShadoGrassField, vec2i(texel, row), 0);
  return packed[lane];
}

fn Shado_grassGround(x: i32, z: i32, extended: i32, row: i32) -> f32 {
  let sampleIndex = z * extended + x;
  return Shado_grassFieldValue(
    ${COVERAGE_TEXELS} + sampleIndex / 4,
    sampleIndex % 4,
    row
  );
}

fn Shado_grassBezier(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let u = 1.0 - t;
  return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3;
}

fn Shado_grassBezierTangent(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  return 3.0 * (1.0 - t) * (1.0 - t) * (p1 - p0)
    + 6.0 * (1.0 - t) * t * (p2 - p1)
    + 3.0 * t * t * (p3 - p2);
}`;

  return {
    name: 'grassBlades',
    uniforms: ['uShadoGrassShape', 'uShadoGrassSize', 'uShadoGrassDetail'],
    glsl: {
      vertexDeclarations: GLSL_HELPERS,
      displace: `
  {
    float cellSize = uShadoGrassShape.x;
    float coverageRes = uShadoGrassShape.y;
    float heightRes = uShadoGrassShape.z;
    int extended = int(heightRes) + ${2 * HEIGHT_RING};
    int fieldRow = int(inst.foliageParams.x + 0.5);
    float blade = aGrassBlade.x;
    float side = aGrassBlade.y;

    // Per-blade character from exact small numbers: the blade index split
    // into two bytes, mixed with the cell's CPU-computed seeds. World
    // coordinates never enter a hash.
    int metaTexel = ${COVERAGE_TEXELS} + (extended * extended + 3) / 4;
    vec2 cellRand = vec2(
      Shado_grassFieldValue(metaTexel, 0, fieldRow),
      Shado_grassFieldValue(metaTexel, 1, fieldRow)
    );
    float bladeHighPart = floor(blade / 256.0);
    vec2 bladeUV = vec2(blade - bladeHighPart * 256.0, bladeHighPart) * 0.00390625;
    float randomYaw = Shado_grassHash(bladeUV + cellRand * 7.31);
    float randomSize = Shado_grassHash(bladeUV.yx + cellRand.yx * 3.97);
    float randomLean = Shado_grassHash(bladeUV + cellRand.yx * 11.73);

    // R2 low-discrepancy roots. Every prefix of this sequence is evenly spread,
    // so raising density adds blades into the gaps instead of reshuffling the
    // field, and no prefix clumps the way a plain hash does. R2 is a lattice
    // though, and a bare lattice moires badly when the field is seen from
    // above, so each root is jittered by up to half a stratum: enough to break
    // the lattice, too little to reintroduce clumping.
    float stratum = 0.62 / sqrt(max(uShadoGrassShape.w, 1.0));
    float randomU = fract(
      blade * 0.7548776662466927 + cellRand.x
        + (Shado_grassHash(bladeUV + cellRand * 5.23) - 0.5) * stratum
    );
    float randomV = fract(
      blade * 0.5698402909980532 + cellRand.y
        + (Shado_grassHash(bladeUV.yx + cellRand * 9.17) - 0.5) * stratum
    );

    // Coverage decides existence before anything else is computed.
    int texelX = int(min(randomU * coverageRes, coverageRes - 1.0));
    int texelZ = int(min(randomV * coverageRes, coverageRes - 1.0));
    int coverageBit = texelZ * int(coverageRes) + texelX;
    int coverageFloat = coverageBit / ${COVERAGE_BITS_PER_FLOAT};
    float bitInFloat = float(coverageBit - coverageFloat * ${COVERAGE_BITS_PER_FLOAT});
    float packedBits = Shado_grassFieldValue(
      coverageFloat / 4,
      coverageFloat - (coverageFloat / 4) * 4,
      fieldRow
    );
    float covered = mod(floor(packedBits / exp2(bitInFloat)), 2.0);

    // Ground height, bilinear over the ring-extended grid. The +0.5 shift puts
    // the blade in the ring's coordinates, where samples are uniformly spaced
    // across cell borders and the interpolation is therefore seamless.
    float sampleU = randomU * heightRes + 0.5;
    float sampleV = randomV * heightRes + 0.5;
    int x0 = int(floor(sampleU));
    int z0 = int(floor(sampleV));
    float fx = sampleU - float(x0);
    float fz = sampleV - float(z0);
    float h00 = Shado_grassGround(x0, z0, extended, fieldRow);
    float h10 = Shado_grassGround(x0 + 1, z0, extended, fieldRow);
    float h01 = Shado_grassGround(x0, z0 + 1, extended, fieldRow);
    float h11 = Shado_grassGround(x0 + 1, z0 + 1, extended, fieldRow);
    float groundY = mix(mix(h00, h10, fx), mix(h01, h11, fx), fz);

    vec3 root = vec3(
      shadoFoliageRoot.x + randomU * cellSize,
      groundY,
      shadoFoliageRoot.z + randomV * cellSize
    );

    float bladeHeight = uShadoGrassSize.x + randomSize * uShadoGrassSize.y;
    float yaw = randomYaw * 6.2831853;
    vec2 facing = vec2(cos(yaw), sin(yaw));
    vec2 across = vec2(-facing.y, facing.x);

    // Cubic bezier from root to tip. The control points trail the bend so the
    // blade leaves the ground near-vertical and curls over toward the tip.
    vec2 bend = facing * (randomLean * 2.0 - 1.0) * uShadoGrassSize.w * bladeHeight;
    vec3 p0 = vec3(0.0);
    vec3 p1 = vec3(bend.x * 0.10, bladeHeight * 0.34, bend.y * 0.10);
    vec3 p2 = vec3(bend.x * 0.42, bladeHeight * 0.72, bend.y * 0.42);
    vec3 p3 = vec3(bend.x, bladeHeight, bend.y);
    float t = shadoFoliageUp;
    vec3 centre = Shado_grassBezier(p0, p1, p2, p3, t);
    vec3 tangent = normalize(Shado_grassBezierTangent(p0, p1, p2, p3, t));

    float taper = 1.0 - smoothstep(0.55, 1.0, t);
    float halfWidth = uShadoGrassSize.z * 0.5 * taper;
    vec3 bladeAcross = vec3(across.x, 0.0, across.y);
    vec3 bladePosition = root + centre + bladeAcross * side * halfWidth;

    vec3 viewDirection = normalize(uShadoFoliageCamera - root);
    vec3 bladeNormal = normalize(cross(bladeAcross, tangent));
    if (dot(bladeNormal, viewDirection) < 0.0) bladeNormal = -bladeNormal;

    // Edge-on thickening: widen along the screen-horizontal direction for this
    // blade as its ribbon turns away, so it never collapses to nothing.
    float facingEye = abs(dot(bladeNormal, viewDirection));
    vec3 screenAcross = cross(viewDirection, tangent);
    float screenLength = length(screenAcross);
    if (screenLength > 0.0001) {
      bladePosition += (screenAcross / screenLength)
        * side * halfWidth * uShadoGrassDetail.x * pow(1.0 - facingEye, 4.0);
    }

    shadoFoliageWorld = bladePosition;
    // Later plugins measure distance from the blade, not the cell corner.
    shadoFoliageAnchor = root;
    shadoFoliageFade *= covered;
    shadoFoliagePhase = randomU;
    shadoFoliageStiffness = randomSize;
    shadoFoliageVariation = randomV;

    // Foliage lighting, from the curve normal. Wrapped diffuse keeps the
    // shadowed side of a blade from going black, and backscatter brightens
    // blades seen against the sun the way a real sward does.
    if (inst.padding1 > 0.5) {
      float wrapped = clamp((dot(bladeNormal, uShadoLightDirection) + 0.45) / 1.45, 0.0, 1.0);
      float backscatter =
        pow(max(dot(viewDirection, -uShadoLightDirection), 0.0), 3.0) * 0.35;
      vShadoLighting = uShadoAmbientColor
        + uShadoLightColor * (wrapped + backscatter * smoothstep(0.25, 1.0, t));
    }
  }`,
    },
    wgsl: {
      vertexDeclarations: WGSL_HELPERS,
      displace: `
  {
    let cellSize = uniforms.uShadoGrassShape.x;
    let coverageRes = uniforms.uShadoGrassShape.y;
    let heightRes = uniforms.uShadoGrassShape.z;
    let extended = i32(heightRes) + ${2 * HEIGHT_RING};
    let fieldRow = i32(inst.foliageParams.x + 0.5);
    let blade = vertexInputs.aGrassBlade.x;
    let side = vertexInputs.aGrassBlade.y;

    let metaTexel = ${COVERAGE_TEXELS} + (extended * extended + 3) / 4;
    let cellRand = vec2f(
      Shado_grassFieldValue(metaTexel, 0, fieldRow),
      Shado_grassFieldValue(metaTexel, 1, fieldRow)
    );
    let bladeHighPart = floor(blade / 256.0);
    let bladeUV = vec2f(blade - bladeHighPart * 256.0, bladeHighPart) * 0.00390625;
    let randomYaw = Shado_grassHash(bladeUV + cellRand * 7.31);
    let randomSize = Shado_grassHash(bladeUV.yx + cellRand.yx * 3.97);
    let randomLean = Shado_grassHash(bladeUV + cellRand.yx * 11.73);

    let stratum = 0.62 / sqrt(max(uniforms.uShadoGrassShape.w, 1.0));
    let randomU = fract(
      blade * 0.7548776662466927 + cellRand.x
        + (Shado_grassHash(bladeUV + cellRand * 5.23) - 0.5) * stratum
    );
    let randomV = fract(
      blade * 0.5698402909980532 + cellRand.y
        + (Shado_grassHash(bladeUV.yx + cellRand * 9.17) - 0.5) * stratum
    );

    let texelX = i32(min(randomU * coverageRes, coverageRes - 1.0));
    let texelZ = i32(min(randomV * coverageRes, coverageRes - 1.0));
    let coverageBit = texelZ * i32(coverageRes) + texelX;
    let coverageFloat = coverageBit / ${COVERAGE_BITS_PER_FLOAT};
    let bitInFloat = f32(coverageBit - coverageFloat * ${COVERAGE_BITS_PER_FLOAT});
    let packedBits = Shado_grassFieldValue(coverageFloat / 4, coverageFloat % 4, fieldRow);
    let covered = floor(packedBits / exp2(bitInFloat)) % 2.0;

    let sampleU = randomU * heightRes + 0.5;
    let sampleV = randomV * heightRes + 0.5;
    let x0 = i32(floor(sampleU));
    let z0 = i32(floor(sampleV));
    let fx = sampleU - f32(x0);
    let fz = sampleV - f32(z0);
    let h00 = Shado_grassGround(x0, z0, extended, fieldRow);
    let h10 = Shado_grassGround(x0 + 1, z0, extended, fieldRow);
    let h01 = Shado_grassGround(x0, z0 + 1, extended, fieldRow);
    let h11 = Shado_grassGround(x0 + 1, z0 + 1, extended, fieldRow);
    let groundY = mix(mix(h00, h10, fx), mix(h01, h11, fx), fz);

    let root = vec3f(
      shadoFoliageRoot.x + randomU * cellSize,
      groundY,
      shadoFoliageRoot.z + randomV * cellSize
    );

    let bladeHeight = uniforms.uShadoGrassSize.x + randomSize * uniforms.uShadoGrassSize.y;
    let yaw = randomYaw * 6.2831853;
    let facing = vec2f(cos(yaw), sin(yaw));
    let across = vec2f(-facing.y, facing.x);

    let bend = facing * (randomLean * 2.0 - 1.0) * uniforms.uShadoGrassSize.w * bladeHeight;
    let p0 = vec3f(0.0);
    let p1 = vec3f(bend.x * 0.10, bladeHeight * 0.34, bend.y * 0.10);
    let p2 = vec3f(bend.x * 0.42, bladeHeight * 0.72, bend.y * 0.42);
    let p3 = vec3f(bend.x, bladeHeight, bend.y);
    let t = shadoFoliageUp;
    let centre = Shado_grassBezier(p0, p1, p2, p3, t);
    let tangent = normalize(Shado_grassBezierTangent(p0, p1, p2, p3, t));

    let taper = 1.0 - smoothstep(0.55, 1.0, t);
    let halfWidth = uniforms.uShadoGrassSize.z * 0.5 * taper;
    let bladeAcross = vec3f(across.x, 0.0, across.y);
    var bladePosition = root + centre + bladeAcross * side * halfWidth;

    let viewDirection = normalize(uniforms.uShadoFoliageCamera - root);
    var bladeNormal = normalize(cross(bladeAcross, tangent));
    bladeNormal = select(-bladeNormal, bladeNormal, dot(bladeNormal, viewDirection) >= 0.0);

    let facingEye = abs(dot(bladeNormal, viewDirection));
    let screenAcross = cross(viewDirection, tangent);
    let screenLength = length(screenAcross);
    if (screenLength > 0.0001) {
      bladePosition = bladePosition + (screenAcross / screenLength)
        * side * halfWidth * uniforms.uShadoGrassDetail.x * pow(1.0 - facingEye, 4.0);
    }

    shadoFoliageWorld = bladePosition;
    // Later plugins measure distance from the blade, not the cell corner.
    shadoFoliageAnchor = root;
    shadoFoliageFade = shadoFoliageFade * covered;
    shadoFoliagePhase = randomU;
    shadoFoliageStiffness = randomSize;
    shadoFoliageVariation = randomV;

    if (inst.padding1 > 0.5) {
      let wrapped = clamp(
        (dot(bladeNormal, uniforms.uShadoLightDirection) + 0.45) / 1.45,
        0.0,
        1.0
      );
      let backscatter =
        pow(max(dot(viewDirection, -uniforms.uShadoLightDirection), 0.0), 3.0) * 0.35;
      vertexOutputs.vShadoLighting = uniforms.uShadoAmbientColor
        + uniforms.uShadoLightColor * (wrapped + backscatter * smoothstep(0.25, 1.0, t));
    }
  }`,
    },
    bind(material) {
      material.setVector4('uShadoGrassShape', shape);
      material.setVector4('uShadoGrassSize', size);
      material.setVector4('uShadoGrassDetail', detail);
    },
  };
}

/**
 * Builds the patch mesh: blade topology only, with no placement baked in.
 *
 * `position.y` carries height along the blade, which the container reads as
 * `shadoFoliageUp` when `sourceHeight` is 1. `aGrassBlade` carries the blade
 * index and which side of the ribbon a vertex is on.
 */
export function createShadoGrassPatch(
  scene: Scene,
  name: string,
  bladeCount: number,
  segments = 4,
  /**
   * Index of the first blade. A far-LOD patch bakes a *later* slice of the R2
   * sequence (e.g. blades 12288..14335), so where rings overlap its blades sit
   * between the near ring's rather than on top of them — extra density instead
   * of z-fighting duplicates — and every ring stays deterministic per cell.
   */
  firstBlade = 0
): Mesh {
  const mesh = new BABYLON.Mesh(name, scene);
  // The topology is fully determined by (bladeCount, segments), so every
  // stream is preallocated and written in place. Growing plain JS arrays a
  // push at a time was the container's real density ceiling: mesh construction
  // stalled long before the GPU cared.
  const rowsPerBlade = segments + 1;
  const verticesPerBlade = rowsPerBlade * 2;
  const vertexCount = bladeCount * verticesPerBlade;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const bladeData = new Float32Array(vertexCount * 2);
  const indices = new Uint32Array(bladeCount * segments * 6);

  let vertex = 0;
  let index = 0;
  for (let blade = 0; blade < bladeCount; blade++) {
    const first = vertex;
    for (let row = 0; row < rowsPerBlade; row++) {
      const t = row / segments;
      for (let lane = 0; lane < 2; lane++) {
        // positions carry only height along the blade; the shader rebuilds the
        // rest. The normal is a placeholder for the same reason: one attribute
        // cannot describe a blade whose bend is chosen per instance.
        positions[vertex * 3 + 1] = t;
        normals[vertex * 3 + 2] = 1;
        uvs[vertex * 2] = 0.5;
        uvs[vertex * 2 + 1] = t;
        bladeData[vertex * 2] = firstBlade + blade;
        bladeData[vertex * 2 + 1] = lane === 0 ? -1 : 1;
        vertex++;
      }
    }
    for (let row = 0; row < segments; row++) {
      const a = first + row * 2;
      indices[index++] = a;
      indices[index++] = a + 1;
      indices[index++] = a + 3;
      indices[index++] = a;
      indices[index++] = a + 3;
      indices[index++] = a + 2;
    }
  }

  const data = new BABYLON.VertexData();
  data.positions = positions;
  data.normals = normals;
  data.uvs = uvs;
  data.indices = indices;
  data.applyToMesh(mesh, false);
  mesh.setVerticesData('aGrassBlade', bladeData, false, 2);
  mesh.isPickable = false;
  mesh.checkCollisions = false;
  return mesh;
}

/** lowbias32-style integer hash of a cell coordinate, mapped to [0, 1). */
function cellSeed01(cellX: number, cellZ: number, seed: number): number {
  let hash = (Math.imul(cellX, 0x27d4_eb2d) ^ Math.imul(cellZ, 0x1656_67b1) ^ seed) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb_352d) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 15), 0x846c_a68b) >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  return hash / 0x1_0000_0000;
}

const cellIndexCache = new WeakMap<ShadoWorldGrassFieldPackage, Map<string, number>>();

function cellIndex(field: ShadoWorldGrassFieldPackage): Map<string, number> {
  let index = cellIndexCache.get(field);
  if (index) return index;
  index = new Map<string, number>();
  for (let cell = 0; cell < field.cells.x.length; cell++) {
    index.set(`${field.cells.x[cell]}:${field.cells.z[cell]}`, cell);
  }
  cellIndexCache.set(field, index);
  return index;
}

const cellMeanCache = new WeakMap<ShadoWorldGrassFieldPackage, Float64Array>();

/** Mean valid ground height per cell, used where a sample has no surface. */
function cellMeans(field: ShadoWorldGrassFieldPackage): Float64Array {
  let means = cellMeanCache.get(field);
  if (means) return means;
  const resolution = field.heightField.resolution;
  const perCell = resolution * resolution;
  const wordsPerCell = field.heightField.wordsPerCell;
  means = new Float64Array(field.cells.x.length);
  for (let cell = 0; cell < means.length; cell++) {
    let total = 0;
    let count = 0;
    for (let sample = 0; sample < perCell; sample++) {
      const word = field.heightField.words[cell * wordsPerCell + (sample >>> 5)]!;
      if (!((word >>> (sample & 31)) & 1)) continue;
      total +=
        field.heightField.minimumY[cell]! +
        (field.heightField.samples[cell * perCell + sample]! / 0xffff) *
          field.heightField.heightRange[cell]!;
      count++;
    }
    means[cell] = count ? total / count : field.heightField.minimumY[cell]!;
  }
  cellMeanCache.set(field, means);
  return means;
}

function groundAt(
  field: ShadoWorldGrassFieldPackage,
  means: Float64Array,
  cell: number,
  sample: number
): number {
  const perCell = field.heightField.resolution * field.heightField.resolution;
  const wordsPerCell = field.heightField.wordsPerCell;
  const word = field.heightField.words[cell * wordsPerCell + (sample >>> 5)]!;
  if (!((word >>> (sample & 31)) & 1)) return means[cell]!;
  return (
    field.heightField.minimumY[cell]! +
    (field.heightField.samples[cell * perCell + sample]! / 0xffff) *
      field.heightField.heightRange[cell]!
  );
}

/**
 * Packs a field package's cells into the data texture the plugin samples.
 *
 * Only the cells named in `cellIndices` are packed, so a streaming caller
 * uploads the resident window rather than the whole zone. Heights are written
 * as absolute world Y on an `(resolution + 2)²` grid whose outer ring is taken
 * from the neighbouring cells, which is what makes the ground continuous
 * across a cell border rather than stepping at it.
 */
export function packShadoGrassFieldData(
  field: ShadoWorldGrassFieldPackage,
  cellIndices: readonly number[],
  /**
   * Destination to write into, sized for the texture's full row capacity. A
   * streaming caller reuses one buffer and one texture across residency
   * changes rather than reallocating both every time it crosses a cell.
   */
  out?: Float32Array
): Float32Array {
  const width = GRASS_FIELD_TEXELS_PER_CELL;
  const height = Math.max(1, cellIndices.length);
  if (out && out.length < width * height * 4) {
    throw new Error(
      `Grass field buffer holds ${out.length / (width * 4)} rows, needs ${height}`
    );
  }
  const data = out ?? new Float32Array(width * height * 4);
  const coverageWords = field.coverage.wordsPerCell;
  const resolution = field.heightField.resolution;
  const extended = resolution + 2 * HEIGHT_RING;
  const required = COVERAGE_TEXELS + Math.ceil((extended * extended) / 4) + 1;
  if (required > width) {
    throw new Error(
      `Grass field resolution ${resolution} needs ${required} texels per cell, ` +
        `but the row stride is ${width}`
    );
  }
  const index = cellIndex(field);
  const means = cellMeans(field);

  cellIndices.forEach((cell, row) => {
    const base = row * width * 4;
    // Coverage arrives as 32-bit words but a float only holds 24 bits exactly,
    // so each word is split into two 16-bit halves.
    for (let word = 0; word < coverageWords; word++) {
      const value = field.coverage.words[cell * coverageWords + word]! >>> 0;
      data[base + word * 2] = value & 0xffff;
      data[base + word * 2 + 1] = (value >>> 16) & 0xffff;
    }

    const cellX = field.cells.x[cell]!;
    const cellZ = field.cells.z[cell]!;
    const heightBase = base + COVERAGE_TEXELS * 4;
    for (let ez = 0; ez < extended; ez++) {
      for (let ex = 0; ex < extended; ex++) {
        let localX = ex - HEIGHT_RING;
        let localZ = ez - HEIGHT_RING;
        let neighbourX = 0;
        let neighbourZ = 0;
        if (localX < 0) {
          neighbourX = -1;
          localX = resolution - 1;
        } else if (localX >= resolution) {
          neighbourX = 1;
          localX = 0;
        }
        if (localZ < 0) {
          neighbourZ = -1;
          localZ = resolution - 1;
        } else if (localZ >= resolution) {
          neighbourZ = 1;
          localZ = 0;
        }
        // Resolve each axis independently, dropping only the one whose
        // neighbour is absent. Clamping both axes into this cell instead would
        // make the two sides of a border replicate *different* cells at the
        // field's outer edge, reintroducing the step the ring exists to remove.
        let source = cell;
        let sampleX = localX;
        let sampleZ = localZ;
        for (const [offsetX, offsetZ] of [
          [neighbourX, neighbourZ],
          [neighbourX, 0],
          [0, neighbourZ],
          [0, 0],
        ] as const) {
          const candidate = index.get(`${cellX + offsetX}:${cellZ + offsetZ}`);
          if (candidate === undefined) continue;
          source = candidate;
          sampleX =
            offsetX === neighbourX
              ? localX
              : Math.min(Math.max(ex - HEIGHT_RING, 0), resolution - 1);
          sampleZ =
            offsetZ === neighbourZ
              ? localZ
              : Math.min(Math.max(ez - HEIGHT_RING, 0), resolution - 1);
          break;
        }
        data[heightBase + ez * extended + ex] = groundAt(
          field,
          means,
          source,
          sampleZ * resolution + sampleX
        );
      }
    }

    // Per-cell randomness the shader can trust. Position must never be hashed
    // on the GPU (f32 sin-hash bias shows up as square tonal seams), so the
    // seeds come from an integer hash here in f64 land.
    const metaBase = heightBase + extended * extended;
    data[metaBase] = cellSeed01(cellX, cellZ, field.seed);
    data[metaBase + 1] = cellSeed01(cellZ, cellX, field.seed ^ CELL_SEED_SALT);
  });
  return data;
}

export function packShadoGrassField(
  scene: Scene,
  field: ShadoWorldGrassFieldPackage,
  cellIndices: readonly number[]
): Texture {
  const width = GRASS_FIELD_TEXELS_PER_CELL;
  const height = Math.max(1, cellIndices.length);
  const data = packShadoGrassFieldData(field, cellIndices);

  const texture = BABYLON.RawTexture.CreateRGBATexture(
    data,
    width,
    height,
    scene,
    false,
    false,
    BABYLON.Texture.NEAREST_SAMPLINGMODE,
    BABYLON.Engine.TEXTURETYPE_FLOAT
  );
  texture.name = 'uShadoGrassField';
  texture.wrapU = BABYLON.Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = BABYLON.Texture.CLAMP_ADDRESSMODE;
  return texture;
}
