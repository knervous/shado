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
  | 'phase-variant-blocker'
  | 'skinned-geometry'
  | 'morph-targets'
  | 'animated-node'
  | 'no-eligible-submesh'
  | 'triangle-budget';

export type GlbPrimitive = {
  /** Node path, for debugging a specific submesh out of a prototype. */
  readonly node: string;
  readonly material: string;
  readonly alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  readonly doubleSided: boolean;
  /**
   * Why this submesh cannot be treated as geometry that is always exactly
   * where it was read, or null when nothing disqualifies it. Skinning, morph
   * targets and animated ancestors all mean the drawn surface is not the
   * surface in the buffer.
   */
  readonly dynamic: 'skinned-geometry' | 'morph-targets' | 'animated-node' | null;
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
  /** The phase contract the assembly ran under, and the phases it saw. */
  phases: { policy: 'invariant' | 'active-phase'; activeMask: number; worldMask: number };
  /**
   * Placed triangles by how many faces their material draws. Single-sided
   * triangles block only from the front, so this is the share of the occluder
   * set whose usefulness depends on which side a viewer stands.
   */
  sidedness: { doubleSidedTriangles: number; singleSidedTriangles: number };
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
   * How a blocker has to relate to the phases that share one visibility row.
   *
   * `invariant` (the default) admits only stamps present in EVERY phase the
   * world uses, because one static row serves all of them: a wall that exists
   * in phase A and not in phase B would otherwise hide, from phase B, content
   * a player in phase B can see. `active-phase` admits anything drawn in
   * `activePhaseMask`, which is only sound for rows that are themselves bound
   * to that phase and selected as such at runtime -- which the package cannot
   * currently express, so it is opt-in and never the default.
   */
  phasePolicy?: 'invariant' | 'active-phase';
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
  /*
   * Nodes an animation drives, and every descendant of one: a child inherits
   * its ancestor's motion, so a door leaf under an animated hinge is animated
   * even though no channel names it.
   */
  const animatedNodes = new Set<number>();
  for (const animation of gltf.animations ?? []) {
    for (const channel of animation.channels ?? []) {
      if (channel.target?.node != null) animatedNodes.add(channel.target.node);
    }
  }
  const scene = gltf.scenes?.[gltf.scene ?? 0];
  const roots = scene?.nodes ?? gltf.nodes?.map((_, index) => index) ?? [];
  const walk = (
    nodeIndex: number,
    parent: number[],
    path: string,
    animatedAncestor: boolean
  ): void => {
    const node = gltf.nodes?.[nodeIndex];
    if (!node) return;
    const local = nodeMatrix(node);
    const matrix = multiply(parent, local);
    const name = node.name ? `${path}/${node.name}` : `${path}/node${nodeIndex}`;
    const animated = animatedAncestor || animatedNodes.has(nodeIndex);
    if (node.mesh != null) {
      const mesh = gltf.meshes?.[node.mesh];
      mesh?.primitives?.forEach((primitive, index) => {
        const material =
          primitive.material == null ? undefined : gltf.materials?.[primitive.material];
        const geometry = readPrimitive(gltf, binary!, primitive, matrix);
        const dynamic: GlbPrimitive['dynamic'] =
          node.skin != null || primitive.attributes?.JOINTS_0 != null
            ? 'skinned-geometry'
            : primitive.targets?.length
              ? 'morph-targets'
              : animated
                ? 'animated-node'
                : null;
        out.push({
          node: mesh!.primitives!.length === 1 ? name : `${name}#${index}`,
          material: material?.name ?? (primitive.material == null ? '__default' : `material-${primitive.material}`),
          alphaMode: (material?.alphaMode as GlbPrimitive['alphaMode']) ?? 'OPAQUE',
          doubleSided: material?.doubleSided === true,
          dynamic,
          positions: geometry.positions,
          indices: geometry.indices,
          extras: { ...(material?.extras ?? {}), ...(mesh?.extras ?? {}), ...(primitive.extras ?? {}) },
        });
      });
    }
    for (const child of node.children ?? []) walk(child, matrix, name, animated);
  };
  for (const root of roots) walk(root, IDENTITY, '', false);
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
  /*
   * `OPAQUE` is a statement about light, not about permanence. A skinned or
   * morphed mesh, or one under an animated node, is drawn somewhere other
   * than where its buffer says -- a door that swings open is the case that
   * matters, because baking it shut hides a corridor the player walks down.
   */
  if (primitive.dynamic) return primitive.dynamic;
  if (!primitive.indices.length || !primitive.positions.length) return 'unsupported-accessor';
  return null;
}

/** What the base world contributed, and what it could not. */
export type BaseWorldSceneManifest = {
  submeshes: { total: number; eligible: number };
  triangles: { sourced: number; eligible: number };
  excluded: Record<OccluderExclusion, number>;
  sidedness: { doubleSidedTriangles: number; singleSidedTriangles: number };
  /** The transform applied on the way in, mirroring the runtime's import. */
  sourceTransform: 'identity' | 'mirror-x';
};

/**
 * The zone's own render geometry as occluders, rather than its collision mesh.
 *
 * Collision was always a stand-in here, and a poor one in both directions: it
 * carries invisible barriers and box proxies that seal openings a player sees
 * straight through, and it omits the visual surfaces that actually block a
 * view. This reads the promoted world GLB -- the same bytes the runtime draws
 * -- through the same eligibility contract placed objects go through, so one
 * rule decides what may hide anything.
 *
 * The transform defaults to `identity`, and that is not an oversight. The
 * shipped world GLB is the PROMOTED output the runtime draws, so it is
 * already in the package's frame: `sourceTransform: 'mirror-x'` records what
 * import did to the authoring source, not something to do again. Applying it
 * a second time produced a scene that agreed with a mirrored world 36.7% of
 * the time against 3.0% as placed, which is what the bake's frame-agreement
 * check is for. The parameter remains for a caller holding pre-import bytes,
 * and mirroring reverses winding, so the indices go with it.
 */
export function assembleBaseWorldScene(input: {
  glb: Uint8Array;
  sourceTransform?: 'identity' | 'mirror-x';
  maxTriangles?: number;
}): { primitives: ShadoWorldPrimitive[]; manifest: BaseWorldSceneManifest } {
  const sourceTransform = input.sourceTransform ?? 'identity';
  const mirrored = sourceTransform === 'mirror-x';
  const maxTriangles = input.maxTriangles ?? 12_000_000;
  const excluded: Record<OccluderExclusion, number> = {
    'alpha-blended-material': 0,
    'alpha-tested-material': 0,
    'non-triangle-primitive': 0,
    'unsupported-accessor': 0,
    'missing-prototype-asset': 0,
    'unreadable-prototype-asset': 0,
    'stamp-disabled': 0,
    'stamp-out-of-phase': 0,
    'phase-variant-blocker': 0,
    'skinned-geometry': 0,
    'morph-targets': 0,
    'animated-node': 0,
    'no-eligible-submesh': 0,
    'triangle-budget': 0,
  };
  const manifest: BaseWorldSceneManifest = {
    submeshes: { total: 0, eligible: 0 },
    triangles: { sourced: 0, eligible: 0 },
    excluded,
    sidedness: { doubleSidedTriangles: 0, singleSidedTriangles: 0 },
    sourceTransform,
  };
  const primitives: ShadoWorldPrimitive[] = [];
  for (const part of readGlbPrimitives(input.glb)) {
    manifest.submeshes.total += 1;
    const triangles = part.indices.length / 3;
    manifest.triangles.sourced += triangles;
    const reason = occluderEligibility(part);
    if (reason) {
      excluded[reason] += 1;
      continue;
    }
    if (manifest.triangles.eligible + triangles > maxTriangles) {
      excluded['triangle-budget'] += 1;
      continue;
    }
    const positions = new Float32Array(part.positions.length);
    for (let offset = 0; offset < part.positions.length; offset += 3) {
      positions[offset] = mirrored ? -part.positions[offset]! : part.positions[offset]!;
      positions[offset + 1] = part.positions[offset + 1]!;
      positions[offset + 2] = part.positions[offset + 2]!;
    }
    manifest.submeshes.eligible += 1;
    manifest.triangles.eligible += triangles;
    if (part.doubleSided) manifest.sidedness.doubleSidedTriangles += triangles;
    else manifest.sidedness.singleSidedTriangles += triangles;
    primitives.push({
      name: `base:${part.node}`,
      material: part.material,
      doubleSided: part.doubleSided,
      positions,
      indices: mirrored ? reverseWinding(part.indices) : part.indices,
    });
  }
  return { primitives, manifest };
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
  const phasePolicy = input.phasePolicy ?? 'invariant';
  /*
   * The phases this world supports, taken from everything it draws -- and
   * deliberately NOT narrowed by the mask this bake selected.
   *
   * Intersecting the two first is how the protection defeated itself: asking
   * for `activePhaseMask = 1` made the supported set `1`, and a stamp present
   * only in phase 1 then looked invariant across it. The row that comes out
   * is still an ordinary row with no phase restriction the runtime can
   * enforce, so it would hide, from phase 2, a target that phase 2 can see.
   * Invariance is a property of the world, not of the request.
   */
  let worldPhases = 0;
  for (const mask of input.world.cells?.phaseMask ?? []) worldPhases |= mask;
  for (const mask of stamps?.phaseMask ?? []) worldPhases |= mask;
  worldPhases = worldPhases >>> 0;
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
    'phase-variant-blocker': 0,
    'skinned-geometry': 0,
    'morph-targets': 0,
    'animated-node': 0,
    'no-eligible-submesh': 0,
    'triangle-budget': 0,
  };
  const manifest: OccluderSceneManifest = {
    prototypes: { total: prototypes?.id.length ?? 0, resolved: 0, unresolved: [] },
    stamps: { total: stamps?.id.length ?? 0, included: 0, excluded },
    phases: { policy: input.phasePolicy ?? 'invariant', activeMask: input.activePhaseMask ?? 0xffffffff, worldMask: 0 },
    sidedness: { doubleSidedTriangles: 0, singleSidedTriangles: 0 },
    triangles: { sourced: 0, eligible: 0, placed: 0, budget: maxTriangles },
    submeshes: [],
  };
  manifest.phases.worldMask = worldPhases;
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
    const stampPhases = stamps.phaseMask[stamp]! >>> 0;
    // Drawn at all in the requested state? That is what the mask decides.
    if ((stampPhases & phaseMask) === 0) { excluded['stamp-out-of-phase'] += 1; continue; }
    // Present in EVERY phase the world supports? That is what invariance
    // decides, and no mask may relax it.
    if (
      phasePolicy === 'invariant' &&
      worldPhases !== 0 &&
      // `&` yields a signed int32, so an all-bits mask compares as -1 unless
      // it is coerced back to unsigned before the comparison.
      ((stampPhases & worldPhases) >>> 0) !== worldPhases
    ) {
      excluded['phase-variant-blocker'] += 1;
      continue;
    }
    const parts = eligibleFor(stamps.prototype[stamp]!);
    if (!parts) { excluded['no-eligible-submesh'] += 1; continue; }
    let triangles = 0;
    for (const part of parts) triangles += part.indices.length / 3;
    if (manifest.triangles.placed + triangles > maxTriangles) {
      excluded['triangle-budget'] += 1;
      continue;
    }
    const matrix = stampMatrix(stamps, stamp, quaternion);
    /*
     * A negative determinant means this stamp is mirrored, which reverses the
     * winding of every triangle it places. Front-face occlusion reads winding,
     * so the indices have to be reversed to keep the front face in front --
     * otherwise a mirrored wall would block from the side you can see through
     * and pass light on the side you cannot.
     */
    const mirrored = determinant3(matrix) < 0;
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
      const partTriangles = part.indices.length / 3;
      if (part.doubleSided) manifest.sidedness.doubleSidedTriangles += partTriangles;
      else manifest.sidedness.singleSidedTriangles += partTriangles;
      primitives.push({
        name: `${stamps.id[stamp] ?? `stamp-${stamp}`}:${part.node}`,
        material: part.material,
        doubleSided: part.doubleSided,
        positions,
        indices: mirrored ? reverseWinding(part.indices) : part.indices,
      });
    }
    manifest.triangles.placed += triangles;
    manifest.stamps.included += 1;
  }
  return { primitives, manifest };
}

/**
 * Determinant of a column-major matrix's rotation/scale part.
 *
 * Negative means the transform mirrors, which reverses triangle winding and
 * so swaps the front face for the back. Used at every level that can mirror:
 * the glTF node hierarchy and the stamp placement.
 */
function determinant3(m: readonly number[]): number {
  return (
    m[0]! * (m[5]! * m[10]! - m[6]! * m[9]!) -
    m[4]! * (m[1]! * m[10]! - m[2]! * m[9]!) +
    m[8]! * (m[1]! * m[6]! - m[2]! * m[5]!)
  );
}

/** Swaps two corners of every triangle, flipping which face is the front. */
function reverseWinding(indices: Uint32Array): Uint32Array {
  const out = new Uint32Array(indices.length);
  for (let index = 0; index + 2 < indices.length; index += 3) {
    out[index] = indices[index]!;
    out[index + 1] = indices[index + 2]!;
    out[index + 2] = indices[index + 1]!;
  }
  return out;
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
  animations?: { channels?: { target?: { node?: number } }[] }[];
  nodes?: {
    name?: string;
    mesh?: number;
    skin?: number;
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
      targets?: Record<string, number>[];
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
  /*
   * A node or ancestor with a negative determinant has already mirrored these
   * positions, which reverses their winding and therefore swaps which face is
   * the front. Correcting it here means the primitive that leaves this
   * function is always front-face-correct in prototype space, and the stamp
   * transform later corrects only its own mirroring. Each level of the
   * transform is accounted for exactly once, so a mirror expressed on the
   * node and the same mirror expressed on the stamp end up agreeing.
   */
  return {
    positions,
    indices: determinant3(matrix) < 0 ? reverseWinding(indices) : indices,
  };
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
