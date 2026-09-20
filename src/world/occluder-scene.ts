/**
 * The opaque visual world, assembled for a visibility bake.
 *
 * ## Why this exists
 *
 * The first occlusion bakes used the zone's collision mesh, because it was
 * decoded and to hand. Collision is the wrong scene for occlusion in both
 * directions. It carries invisible barriers and box proxies that seal
 * doorways, which would hide things a player can see through them; and it
 * omits everything a placed object contributes visually, which is most of what
 * actually blocks a view in a built-up zone. A wall you cannot walk through
 * and a wall you cannot see through are different walls.
 *
 * So this assembles the other scene: every placed static object's eligible
 * opaque submeshes, in world space, with the transform the runtime uses.
 *
 * ## The safe direction
 *
 * Every judgement here errs towards NOT blocking. An excluded occluder costs
 * draw calls; an occluder that is not really opaque, or not really there,
 * costs a hole in the world. A material whose opacity cannot be established is
 * excluded, a primitive that is not a triangle list is excluded, a prototype
 * that will not load is excluded and recorded, and when the triangle budget
 * runs out the remaining objects are excluded rather than approximated.
 *
 * ## What it is not
 *
 * It is not a renderer. Geometry is read straight out of the GLB: no Babylon,
 * no GPU, no textures, no materials beyond the one field that says whether
 * light passes through. That keeps it usable from a CLI, a worker and a shader
 * backend without three definitions of the scene drifting apart.
 */
import { shadoWorldStampQuaternion } from './runtime';
import type { ShadoWorldPrimitive, ShadoWorldSpatialPackage } from './types';

const GLB_MAGIC = 0x46546c67;
const GLB_JSON_CHUNK = 0x4e4f534a;
const GLB_BIN_CHUNK = 0x004e4942;
const TRIANGLES = 4;

/** Why a candidate is not in the occluder set. Every one of these is a cost, not a bug. */
export type OccluderExclusion =
  | 'alpha-blended-material'
  | 'alpha-tested-material'
  | 'non-triangle-primitive'
  | 'unsupported-accessor'
  | 'missing-prototype-asset'
  | 'unreadable-prototype-asset'
  | 'stamp-disabled'
  | 'stamp-out-of-phase'
  | 'no-eligible-submesh'
  | 'triangle-budget';

export type GlbPrimitive = {
  /** Node path, for debugging a specific submesh out of a prototype. */
  readonly node: string;
  readonly material: string;
  readonly alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  readonly doubleSided: boolean;
  /** Prototype-local positions with the node hierarchy already applied. */
  readonly positions: Float64Array;
  readonly indices: Uint32Array;
  readonly extras: Record<string, unknown> | undefined;
};

export type OccluderSceneManifest = {
  prototypes: {
    total: number;
    resolved: number;
    /** Prototype ids whose asset could not be read, with the reason. */
    unresolved: { id: string; source: string; reason: OccluderExclusion }[];
  };
  stamps: {
    total: number;
    included: number;
    excluded: Record<OccluderExclusion, number>;
  };
  triangles: {
    /** Triangles in the prototype meshes before eligibility. */
    sourced: number;
    /** After eligibility, per prototype, before instancing. */
    eligible: number;
    /** After instancing: what the bake will actually test against. */
    placed: number;
    budget: number;
  };
  /** Per-submesh eligibility, so an exclusion can be argued with. */
  submeshes: {
    id: string;
    node: string;
    material: string;
    alphaMode: string;
    triangles: number;
    eligible: boolean;
    reason: OccluderExclusion | null;
  }[];
};

export type OccluderSceneInput = {
  world: ShadoWorldSpatialPackage;
  /**
   * Reads a prototype's GLB bytes, already gunzipped, or null when the asset
   * is not available to this caller. Null is a recorded exclusion, never a
   * throw: a bake with one missing prop should report incomplete coverage
   * rather than fail, and the manifest is what stops that being silent.
   */
  loadPrototype: (source: string, id: string) => Uint8Array | null;
  /** Stamps outside this mask are not drawn, so they cannot block. */
  activePhaseMask?: number;
  /**
   * Ceiling on placed occluder triangles. Reaching it excludes the remaining
   * objects, smallest first, and records them as `triangle-budget`. Omitting
   * an occluder is always safe; running a machine out of memory is not.
   */
  maxTriangles?: number;
};

/**
 * Reads triangle geometry and material opacity out of a GLB.
 *
 * Only what a visibility bake needs: node transforms, positions, indices and
 * the material's alpha mode. Normals, UVs, textures, animation and skinning
 * are ignored. Compressed geometry extensions are not supported and their
 * primitives are reported as `unsupported-accessor` rather than guessed at.
 */
export function readGlbPrimitives(bytes: Uint8Array): GlbPrimitive[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== GLB_MAGIC) throw new Error('Not a GLB');
  const jsonLength = view.getUint32(12, true);
  if (view.getUint32(16, true) !== GLB_JSON_CHUNK) throw new Error('GLB has no JSON chunk');
  const gltf = JSON.parse(
    new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)).trimEnd()
  ) as GltfDocument;
  let binary: Uint8Array | null = null;
  let offset = 20 + jsonLength + ((4 - (jsonLength % 4)) % 4);
  while (offset + 8 <= bytes.byteLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    if (chunkType === GLB_BIN_CHUNK) {
      binary = bytes.subarray(offset + 8, offset + 8 + chunkLength);
      break;
    }
    offset += 8 + chunkLength + ((4 - (chunkLength % 4)) % 4);
  }
  if (!binary) return [];

  const out: GlbPrimitive[] = [];
  const scene = gltf.scenes?.[gltf.scene ?? 0];
  const roots = scene?.nodes ?? gltf.nodes?.map((_, index) => index) ?? [];
  const walk = (nodeIndex: number, parent: number[], path: string): void => {
    const node = gltf.nodes?.[nodeIndex];
    if (!node) return;
    const local = nodeMatrix(node);
    const matrix = multiply(parent, local);
    const name = node.name ? `${path}/${node.name}` : `${path}/node${nodeIndex}`;
    if (node.mesh != null) {
      const mesh = gltf.meshes?.[node.mesh];
      mesh?.primitives?.forEach((primitive, index) => {
        const material =
          primitive.material == null ? undefined : gltf.materials?.[primitive.material];
        const geometry = readPrimitive(gltf, binary!, primitive, matrix);
        out.push({
          node: mesh!.primitives!.length === 1 ? name : `${name}#${index}`,
          material: material?.name ?? (primitive.material == null ? '__default' : `material-${primitive.material}`),
          alphaMode: (material?.alphaMode as GlbPrimitive['alphaMode']) ?? 'OPAQUE',
          doubleSided: material?.doubleSided === true,
          positions: geometry.positions,
          indices: geometry.indices,
          extras: { ...(material?.extras ?? {}), ...(mesh?.extras ?? {}), ...(primitive.extras ?? {}) },
        });
      });
    }
    for (const child of node.children ?? []) walk(child, matrix, name);
  };
  for (const root of roots) walk(root, IDENTITY, '');
  return out;
}

/**
 * Is this submesh allowed to hide things?
 *
 * Opacity is read from the material's alpha mode, which is the only statement
 * the asset makes about whether light passes through it. `MASK` is excluded
 * for now even though a masked surface is opaque where it is opaque: a leaf
 * card is alpha-tested, and its conservative opaque coverage is not its
 * triangle. Recovering alpha-tested walls needs certified coverage, which is
 * separate work.
 */
export function occluderEligibility(primitive: GlbPrimitive): OccluderExclusion | null {
  if (primitive.alphaMode === 'BLEND') return 'alpha-blended-material';
  if (primitive.alphaMode === 'MASK') return 'alpha-tested-material';
  if (!primitive.indices.length || !primitive.positions.length) return 'unsupported-accessor';
  return null;
}

/**
 * Assembles every eligible opaque placed object into world-space primitives.
 *
 * The transform is the runtime's: {@link shadoWorldStampQuaternion} resolves
 * the Euler convention, and scale is applied before rotation exactly as the
 * thin-instance matrices do, so an occluder stands where its object is drawn.
 * Negative and nonuniform scale therefore work by construction rather than by
 * a second implementation that could disagree.
 */
export function assembleOccluderScene(
  input: OccluderSceneInput
): { primitives: ShadoWorldPrimitive[]; manifest: OccluderSceneManifest } {
  const stamps = input.world.objects?.stamps;
  const prototypes = input.world.objects?.prototypes;
  const phaseMask = input.activePhaseMask ?? 0xffffffff;
  const maxTriangles = input.maxTriangles ?? 8_000_000;
  const excluded: Record<OccluderExclusion, number> = {
    'alpha-blended-material': 0,
    'alpha-tested-material': 0,
    'non-triangle-primitive': 0,
    'unsupported-accessor': 0,
    'missing-prototype-asset': 0,
    'unreadable-prototype-asset': 0,
    'stamp-disabled': 0,
    'stamp-out-of-phase': 0,
    'no-eligible-submesh': 0,
    'triangle-budget': 0,
  };
  const manifest: OccluderSceneManifest = {
    prototypes: { total: prototypes?.id.length ?? 0, resolved: 0, unresolved: [] },
    stamps: { total: stamps?.id.length ?? 0, included: 0, excluded },
    triangles: { sourced: 0, eligible: 0, placed: 0, budget: maxTriangles },
    submeshes: [],
  };
  if (!stamps || !prototypes) return { primitives: [], manifest };

  /** Eligible submeshes per prototype, read once and instanced many times. */
  const cache = new Map<number, GlbPrimitive[] | null>();
  const eligibleFor = (prototype: number): GlbPrimitive[] | null => {
    const cached = cache.get(prototype);
    if (cached !== undefined) return cached;
    const id = prototypes.id[prototype] ?? `prototype-${prototype}`;
    const source = prototypes.source[prototype] ?? '';
    let bytes: Uint8Array | null = null;
    try {
      bytes = input.loadPrototype(source, id);
    } catch {
      manifest.prototypes.unresolved.push({ id, source, reason: 'unreadable-prototype-asset' });
      cache.set(prototype, null);
      return null;
    }
    if (!bytes) {
      manifest.prototypes.unresolved.push({ id, source, reason: 'missing-prototype-asset' });
      cache.set(prototype, null);
      return null;
    }
    let parts: GlbPrimitive[];
    try {
      parts = readGlbPrimitives(bytes);
    } catch {
      manifest.prototypes.unresolved.push({ id, source, reason: 'unreadable-prototype-asset' });
      cache.set(prototype, null);
      return null;
    }
    manifest.prototypes.resolved += 1;
    const keep: GlbPrimitive[] = [];
    for (const part of parts) {
      const triangles = part.indices.length / 3;
      manifest.triangles.sourced += triangles;
      const reason = occluderEligibility(part);
      manifest.submeshes.push({
        id,
        node: part.node,
        material: part.material,
        alphaMode: part.alphaMode,
        triangles,
        eligible: reason === null,
        reason,
      });
      if (reason === null) {
        manifest.triangles.eligible += triangles;
        keep.push(part);
      }
    }
    cache.set(prototype, keep.length ? keep : null);
    return keep.length ? keep : null;
  };

  /*
   * Largest first, so a triangle budget spends itself on the things that
   * actually hide a view. A keep excluded for budget is a lost rejection; a
   * flowerpot excluded for budget is nothing.
   */
  const order: number[] = [];
  for (let stamp = 0; stamp < stamps.id.length; stamp += 1) order.push(stamp);
  order.sort((left, right) => (stamps.radius[right] ?? 0) - (stamps.radius[left] ?? 0));

  const primitives: ShadoWorldPrimitive[] = [];
  const quaternion = new Float32Array(4);
  for (const stamp of order) {
    if (!stamps.enabled[stamp]) { excluded['stamp-disabled'] += 1; continue; }
    if ((stamps.phaseMask[stamp]! & phaseMask) === 0) { excluded['stamp-out-of-phase'] += 1; continue; }
    const parts = eligibleFor(stamps.prototype[stamp]!);
    if (!parts) { excluded['no-eligible-submesh'] += 1; continue; }
    let triangles = 0;
    for (const part of parts) triangles += part.indices.length / 3;
    if (manifest.triangles.placed + triangles > maxTriangles) {
      excluded['triangle-budget'] += 1;
      continue;
    }
    const matrix = stampMatrix(stamps, stamp, quaternion);
    for (const part of parts) {
      const positions = new Float32Array(part.positions.length);
      for (let offset = 0; offset < part.positions.length; offset += 3) {
        const x = part.positions[offset]!;
        const y = part.positions[offset + 1]!;
        const z = part.positions[offset + 2]!;
        positions[offset] = matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!;
        positions[offset + 1] = matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!;
        positions[offset + 2] = matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!;
      }
      primitives.push({
        name: `${stamps.id[stamp] ?? `stamp-${stamp}`}:${part.node}`,
        material: part.material,
        positions,
        indices: part.indices,
      });
    }
    manifest.triangles.placed += triangles;
    manifest.stamps.included += 1;
  }
  return { primitives, manifest };
}

/** The runtime's stamp transform, column-major, as thin instances receive it. */
function stampMatrix(
  stamps: NonNullable<ShadoWorldSpatialPackage['objects']>['stamps'],
  stamp: number,
  scratch: Float32Array
): number[] {
  const [x, y, z, w] = shadoWorldStampQuaternion(stamps, stamp, scratch) as unknown as number[];
  const x2 = x! + x!, y2 = y! + y!, z2 = z! + z!;
  const xx = x! * x2, xy = x! * y2, xz = x! * z2;
  const yy = y! * y2, yz = y! * z2, zz = z! * z2;
  const wx = w! * x2, wy = w! * y2, wz = w! * z2;
  const sx = stamps.scaleX[stamp]!, sy = stamps.scaleY[stamp]!, sz = stamps.scaleZ[stamp]!;
  return [
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    stamps.positionX[stamp]!, stamps.positionY[stamp]!, stamps.positionZ[stamp]!, 1,
  ];
}

type GltfDocument = {
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: {
    name?: string;
    mesh?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
  }[];
  meshes?: {
    extras?: Record<string, unknown>;
    primitives?: {
      mode?: number;
      material?: number;
      indices?: number;
      attributes?: Record<string, number>;
      extras?: Record<string, unknown>;
    }[];
  }[];
  materials?: { name?: string; alphaMode?: string; doubleSided?: boolean; extras?: Record<string, unknown> }[];
  accessors?: {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
  }[];
  bufferViews?: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
};

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function nodeMatrix(node: NonNullable<GltfDocument['nodes']>[number]): number[] {
  if (node.matrix?.length === 16) return node.matrix.slice();
  const [tx, ty, tz] = node.translation ?? [0, 0, 0];
  const [qx, qy, qz, qw] = node.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale ?? [1, 1, 1];
  const x2 = qx! + qx!, y2 = qy! + qy!, z2 = qz! + qz!;
  const xx = qx! * x2, xy = qx! * y2, xz = qx! * z2;
  const yy = qy! * y2, yz = qy! * z2, zz = qz! * z2;
  const wx = qw! * x2, wy = qw! * y2, wz = qw! * z2;
  return [
    (1 - (yy + zz)) * sx!, (xy + wz) * sx!, (xz - wy) * sx!, 0,
    (xy - wz) * sy!, (1 - (xx + zz)) * sy!, (yz + wx) * sy!, 0,
    (xz + wy) * sz!, (yz - wx) * sz!, (1 - (xx + yy)) * sz!, 0,
    tx!, ty!, tz!, 1,
  ];
}

/** Column-major `parent * child`, matching glTF's own convention. */
function multiply(parent: readonly number[], child: readonly number[]): number[] {
  const out = new Array<number>(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        parent[row]! * child[column * 4]! +
        parent[4 + row]! * child[column * 4 + 1]! +
        parent[8 + row]! * child[column * 4 + 2]! +
        parent[12 + row]! * child[column * 4 + 3]!;
    }
  }
  return out;
}

const COMPONENT_READERS: Record<number, (view: DataView, offset: number) => number> = {
  5120: (view, offset) => view.getInt8(offset),
  5121: (view, offset) => view.getUint8(offset),
  5122: (view, offset) => view.getInt16(offset, true),
  5123: (view, offset) => view.getUint16(offset, true),
  5125: (view, offset) => view.getUint32(offset, true),
  5126: (view, offset) => view.getFloat32(offset, true),
};

const COMPONENT_BYTES: Record<number, number> = {
  5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4,
};

function readPrimitive(
  gltf: GltfDocument,
  binary: Uint8Array,
  primitive: NonNullable<NonNullable<GltfDocument['meshes']>[number]['primitives']>[number],
  matrix: readonly number[]
): { positions: Float64Array; indices: Uint32Array } {
  const empty = { positions: new Float64Array(0), indices: new Uint32Array(0) };
  if ((primitive.mode ?? TRIANGLES) !== TRIANGLES) return empty;
  const positionIndex = primitive.attributes?.POSITION;
  if (positionIndex == null) return empty;
  const source = readAccessor(gltf, binary, positionIndex, 3);
  if (!source) return empty;
  const positions = new Float64Array(source.length);
  for (let offset = 0; offset < source.length; offset += 3) {
    const x = source[offset]!, y = source[offset + 1]!, z = source[offset + 2]!;
    positions[offset] = matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!;
    positions[offset + 1] = matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!;
    positions[offset + 2] = matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!;
  }
  let indices: Uint32Array;
  if (primitive.indices == null) {
    indices = new Uint32Array(positions.length / 3);
    for (let i = 0; i < indices.length; i += 1) indices[i] = i;
  } else {
    const read = readAccessor(gltf, binary, primitive.indices, 1);
    if (!read) return empty;
    indices = Uint32Array.from(read);
  }
  return { positions, indices };
}

/** Flat component values for an accessor, or null when its layout is unsupported. */
function readAccessor(
  gltf: GltfDocument,
  binary: Uint8Array,
  index: number,
  components: number
): Float64Array | null {
  const accessor = gltf.accessors?.[index];
  if (!accessor || accessor.bufferView == null) return null;
  const reader = COMPONENT_READERS[accessor.componentType];
  const size = COMPONENT_BYTES[accessor.componentType];
  if (!reader || !size) return null;
  const bufferView = gltf.bufferViews?.[accessor.bufferView];
  if (!bufferView || bufferView.buffer !== 0) return null;
  const view = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
  const start = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const stride = bufferView.byteStride ?? size * components;
  const out = new Float64Array(accessor.count * components);
  for (let element = 0; element < accessor.count; element += 1) {
    const base = start + element * stride;
    for (let component = 0; component < components; component += 1) {
      const offset = base + component * size;
      if (offset + size > binary.byteLength) return null;
      out[element * components + component] = reader(view, offset);
    }
  }
  return out;
}
