/**
 * The environment-neutral half of the Shado world packer.
 *
 * Everything here runs unchanged in Node and in a browser worker. It was split
 * out of `worlds.ts` when Libra's authoring pipeline stopped going through a
 * local HTTP service: the editor now compiles a zone package in a worker and
 * writes the artifacts straight to the checkout through the File System Access
 * API, while the CLI keeps packing from disk. Both callers must produce
 * byte-identical output, so the collision policy, primitive import, and
 * compile sequence live in exactly one place — only file I/O differs, and that
 * is injected.
 *
 * `worlds.ts` remains the Node entry point and owns paths, gzip, and the
 * legacy-metadata authoring migration.
 */

import {
  buildShadoWorldLightingManifest,
  compileShadoWorld,
  encodeShadoWorldCollision,
  type ShadoWorldCompileOptions,
  type ShadoWorldPrimitive,
  ShadoCollisionFlags,
} from '../world';

/**
 * Resolves a runtime object URL path (`/eqrequiem/objects/.../final.glb.gz`)
 * to decompressed GLB bytes. The implementation owns both decompression and
 * containment: the CLI resolves against an asset root on disk, the browser
 * against the granted repository directory handle.
 */
export type WorldObjectAssetLoader = (pathname: string) => Promise<Uint8Array>;

let environmentPreparation: (() => Promise<void>) | undefined;

/**
 * Installs whatever Babylon's glTF loader is missing in the host environment.
 * Node supplies an XMLHttpRequest stand-in and a locally hosted Draco decoder;
 * a browser supplies nothing, because it already has both.
 */
export function configureWorldImporter(options: { prepare?: () => Promise<void> }): void {
  environmentPreparation = options.prepare;
}

export type CompileWorldPackageOptions = Omit<
  ShadoWorldCompileOptions,
  'authoring' | 'collisionPrimitives' | 'grassBlockerPrimitives'
> & {
  /** Decompressed GLB bytes for the scene being packed. */
  glb: Uint8Array;
  /** Handedness used by the headless GLB importer. Defaults to Babylon's left. */
  inputHandedness?: 'left' | 'right';
  /**
   * Transform applied while importing geometry. Defaults to `sourceTransform`.
   * Set this to identity when repacking an already-canonical runtime scene but
   * retaining its migration-origin sourceTransform metadata.
   */
  inputTransform?: ShadoWorldCompileOptions['sourceTransform'];
  authoring?: ShadoWorldCompileOptions['authoring'];
  loadObjectAsset?: WorldObjectAssetLoader;
  /** Overrides the authored physics chunk size when the caller supplies one. */
  physicsChunkSize?: number;
};

export type CompiledWorldPackage = {
  world: ReturnType<typeof compileShadoWorld>;
  collision: ReturnType<typeof encodeShadoWorldCollision>;
  lighting: ReturnType<typeof buildShadoWorldLightingManifest>;
  primitiveCount: number;
};

/**
 * Imports a decompressed world GLB and compiles the full spatial, collision,
 * and lighting-plan set in memory. No file is read or written here; the caller
 * decides where the bytes land.
 */
export async function compileWorldPackage(
  options: CompileWorldPackageOptions
): Promise<CompiledWorldPackage> {
  validateGlb(options.glb, options.name);
  // Spatial preprocessing never samples textures, so the importer skips
  // materials. That keeps legacy exports with incomplete image records — an
  // image with neither URI nor bufferView — from failing a valid migration.
  const sourceTransform = options.sourceTransform ?? 'identity';
  const inputTransform = options.inputTransform ?? sourceTransform;
  const imported = await importWorldPrimitives(
    options.glb,
    worldGlbPrimitivePolicies(options.glb),
    inputTransform,
    options.inputHandedness === 'right',
    options.authoring
  );
  const stampedObjectGeometry = await importStampedObjectGeometry(
    options.authoring,
    options.loadObjectAsset
  );
  const collisionPrimitives = [...imported.collision, ...stampedObjectGeometry.collision];
  const primitives = imported.render;
  const physicsChunkSize = options.physicsChunkSize ?? options.authoring?.bake.physicsChunkSize;
  const collision = encodeShadoWorldCollision(collisionPrimitives, { chunkSize: physicsChunkSize });
  const world = compileShadoWorld(primitives, {
    ...options,
    sourceTransform,
    collisionPrimitives,
    grassBlockerPrimitives: stampedObjectGeometry.grassBlockers,
    physicsChunkSize,
  });
  return {
    world,
    collision,
    lighting: buildShadoWorldLightingManifest(world, primitives),
    primitiveCount: primitives.length,
  };
}

const GLB_JSON_CHUNK = 0x4e4f534a;


/**
 * Structural Builder objects render through the stamped-object layer, so they
 * are not present in the base scene GLB. Enabled prototypes are player-solid
 * unless they explicitly opt out with `metadata.clientPhysics=false`; their
 * inspected collision (or render geometry when no dedicated selection exists)
 * is transformed per stamp and merged into the streamed world artifact.
 * `metadata.grassBlocker=true`
 * separately contributes transformed render geometry to the grass coverage
 * mask, so roads, floors, and roofs suppress the terrain blades below them.
 */
export async function importStampedObjectGeometry(
  authoring: ShadoWorldCompileOptions['authoring'],
  loadObjectAsset?: WorldObjectAssetLoader
): Promise<{
  collision: ShadoWorldPrimitive[];
  grassBlockers: ShadoWorldPrimitive[];
}> {
  if (!authoring || !loadObjectAsset) {
    return { collision: [], grassBlockers: [] };
  }
  const prototypes = new Map(
    authoring.objects.prototypes.map(prototype => [prototype.id, prototype])
  );
  const cache = new Map<string, ImportedWorldPrimitives>();
  const collision: ShadoWorldPrimitive[] = [];
  const grassBlockers: ShadoWorldPrimitive[] = [];
  const BABYLON = await import('@babylonjs/core');

  for (const stamp of authoring.objects.stamps) {
    if (!stamp.enabled) continue;
    const prototype = prototypes.get(stamp.prototype);
    if (!prototype) continue;
    // A placed object is solid by default. Authors may opt an explicitly
    // decorative/pass-through prototype or stamp out with clientPhysics=false,
    // but an omitted flag must never produce a visible non-collidable wall.
    const includeCollision =
      prototype.metadata.clientPhysics !== false &&
      stamp.metadata.clientPhysics !== false;
    const includeGrassBlocker =
      prototype.metadata.grassBlocker === true &&
      stamp.metadata.grassBlocker !== false;
    if (!includeCollision && !includeGrassBlocker) continue;
    let sourceGeometry = cache.get(prototype.id);
    if (!sourceGeometry) {
      const sourceUrl = new URL(prototype.source, 'https://shado.invalid');
      if (!sourceUrl.pathname.startsWith('/eqrequiem/')) {
        throw new Error(
          `Player-solid object prototype '${prototype.id}' has unsupported source '${prototype.source}'`
        );
      }
      // Containment of this path inside the caller's asset root is the
      // reader's job: the Node CLI resolves it against a directory, the
      // browser against the granted repository handle.
      const sourceName = decodeURIComponent(sourceUrl.pathname);
      const glb = await loadObjectAsset(sourceName);
      validateGlb(glb, sourceName);
      sourceGeometry = await importWorldPrimitives(
        glb,
        worldGlbPrimitivePolicies(glb),
        'identity',
        true,
        undefined,
        false
      );
      cache.set(prototype.id, sourceGeometry);
    }

    const rotation = BABYLON.Quaternion.RotationYawPitchRoll(
      BABYLON.Tools.ToRadians(stamp.rotationDegrees[1]),
      BABYLON.Tools.ToRadians(stamp.rotationDegrees[0]),
      BABYLON.Tools.ToRadians(stamp.rotationDegrees[2])
    );
    const matrix = BABYLON.Matrix.Compose(
      BABYLON.Vector3.FromArray(stamp.scale),
      rotation,
      BABYLON.Vector3.FromArray(stamp.position)
    ).m;
    const transformPrimitive = (
      primitive: ShadoWorldPrimitive,
      purpose: 'collision' | 'grass-blocker'
    ): ShadoWorldPrimitive => {
      const positions = new Float32Array(primitive.positions.length);
      for (let offset = 0; offset < primitive.positions.length; offset += 3) {
        const x = primitive.positions[offset]!;
        const y = primitive.positions[offset + 1]!;
        const z = primitive.positions[offset + 2]!;
        positions[offset] = x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12];
        positions[offset + 1] = x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13];
        positions[offset + 2] = x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14];
      }
      return {
        ...primitive,
        name: `object:${stamp.id}:${primitive.name}`,
        positions,
        ...(purpose === 'collision'
          ? {
              collisionFlags:
                (primitive.collisionFlags ?? 0) |
                ShadoCollisionFlags.StaticObject |
                ShadoCollisionFlags.PlayerSolid,
            }
          : { extraShader: undefined }),
      };
    };
    if (includeCollision) {
      const collisionSource = sourceGeometry.collision.length
        ? sourceGeometry.collision
        : sourceGeometry.render;
      collision.push(
        ...collisionSource.map(primitive =>
          transformPrimitive(primitive, 'collision')
        )
      );
    }
    if (includeGrassBlocker) {
      grassBlockers.push(
        ...sourceGeometry.render.map(primitive =>
          transformPrimitive(primitive, 'grass-blocker')
        )
      );
    }
  }
  return { collision, grassBlockers };
}

type WorldPrimitivePolicy = {
  material: string;
  collision: boolean;
  collisionFlags: number;
  extraShader?: string;
  visibilityProfile?: string;
  pvsPriority?: string;
};

type ImportedWorldPrimitives = {
  render: ShadoWorldPrimitive[];
  collision: ShadoWorldPrimitive[];
};

export async function importWorldPrimitives(
  glb: Uint8Array,
  sourcePolicies: ReadonlyMap<string, readonly WorldPrimitivePolicy[]>,
  sourceTransform: ShadoWorldCompileOptions['sourceTransform'],
  inputRightHanded = false,
  authoring?: ShadoWorldCompileOptions['authoring'],
  requireCollision = true
): Promise<ImportedWorldPrimitives> {
  // Node needs an XMLHttpRequest stand-in and a locally hosted Draco decoder
  // before Babylon's glTF loader will run; a browser needs neither. The host
  // installs whatever its environment lacks through configureWorldImporter,
  // and it has to land before Babylon is imported so the loader observes it.
  await environmentPreparation?.();
  const BABYLON = await import('@babylonjs/core');
  await import('@babylonjs/loaders');
  const engine = new BABYLON.NullEngine({
    renderWidth: 64,
    renderHeight: 64,
    textureSize: 64,
    deterministicLockstep: false,
    lockstepMaxSteps: 1,
  });
  const scene = new BABYLON.Scene(engine);
  scene.useRightHandedSystem = inputRightHanded;
  try {
    BABYLON.SceneLoader.OnPluginActivatedObservable.addOnce((plugin: any) => {
      if (plugin.name === 'gltf') plugin.skipMaterials = true;
    });
    // Handing the loader the bytes directly avoids materialising a base64
    // data URL twice the size of the GLB — for a 28 MB zone that mattered.
    const container = await BABYLON.LoadAssetContainerAsync(glb, scene, {
      pluginExtension: '.glb',
    });
    container.addAllToScene();
    const geometryOverrides = new Map(
      (authoring?.geometry.meshes ?? []).map(override => [override.mesh, override])
    );
    for (const mesh of scene.meshes) {
      const override = geometryOverrides.get(mesh.name);
      if (!override) continue;
      mesh.position.set(...override.position);
      mesh.scaling.set(...override.scale);
      mesh.rotationQuaternion = BABYLON.Quaternion.RotationYawPitchRoll(
        BABYLON.Tools.ToRadians(override.rotationDegrees[1]),
        BABYLON.Tools.ToRadians(override.rotationDegrees[0]),
        BABYLON.Tools.ToRadians(override.rotationDegrees[2])
      );
    }
    scene.rootNodes.forEach(node => node.computeWorldMatrix(true));
    const primitives: ShadoWorldPrimitive[] = [];
    const collisionPrimitives: ShadoWorldPrimitive[] = [];
    for (const mesh of scene.meshes) {
      const geometryOverride = geometryOverrides.get(mesh.name);
      if (geometryOverride?.enabled === false) continue;
      const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
      const lightmapUvs = mesh.getVerticesData(BABYLON.VertexBuffer.UV2Kind);
      const indices = mesh.getIndices();
      if (!positions?.length || !indices?.length) continue;
      const matrix = mesh.computeWorldMatrix(true).m;
      const worldPositions = new Float32Array(positions.length);
      for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i],
          y = positions[i + 1],
          z = positions[i + 2];
        const worldX = x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12];
        worldPositions[i] = sourceTransform === 'mirror-x' ? -worldX : worldX;
        worldPositions[i + 1] = x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13];
        worldPositions[i + 2] = x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14];
      }
      const subMeshes = mesh.subMeshes?.length
        ? mesh.subMeshes
        : [{ indexStart: 0, indexCount: indices.length, materialIndex: 0 }];
      for (let subIndex = 0; subIndex < subMeshes.length; subIndex++) {
        const subMesh = subMeshes[subIndex];
        const count = subMesh.indexCount - (subMesh.indexCount % 3);
        if (!count) continue;
        const policy = sourcePolicies.get(mesh.name)?.[subIndex];
        const material = geometryOverride?.material ?? policy?.material ?? materialName(mesh.material, subMesh.materialIndex);
        const primitive = {
          name: `${mesh.name || mesh.id}#${subIndex}`,
          material,
          extraShader: policy?.extraShader,
          visibilityProfile: policy?.visibilityProfile,
          pvsPriority: policy?.pvsPriority,
          collisionFlags: policy?.collisionFlags,
          positions: worldPositions,
          indices: Uint32Array.from(indices.slice(subMesh.indexStart, subMesh.indexStart + count)),
          lightmapUvs: lightmapUvs ? Float32Array.from(lightmapUvs) : undefined,
        };
        primitives.push(primitive);
        const collisionEnabled = geometryOverride?.collision === 'enabled'
          ? true
          : geometryOverride?.collision === 'disabled'
            ? false
            : (policy?.collision ?? mesh.name !== 'CLOUD_MDF');
        if (collisionEnabled) {
          collisionPrimitives.push(primitive);
        }
      }
    }
    if (!primitives.length) throw new Error('World GLB has no indexed triangle primitives');
    if (requireCollision && !collisionPrimitives.length) {
      throw new Error('World GLB has no collision-selected triangle primitives');
    }
    return { render: primitives, collision: collisionPrimitives };
  } finally {
    scene.dispose();
    engine.dispose();
  }
}

export function worldGlbPrimitivePolicies(
  bytes: Uint8Array
): Map<string, readonly WorldPrimitivePolicy[]> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const jsonLength = view.getUint32(12, true);
  if (view.getUint32(16, true) !== GLB_JSON_CHUNK || 20 + jsonLength > bytes.byteLength) {
    throw new Error('World GLB is missing its leading JSON chunk');
  }
  const gltf = JSON.parse(
    new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)).trimEnd()
  ) as {
    nodes?: Array<{ name?: string; mesh?: number; extras?: Record<string, unknown> }>;
    meshes?: Array<{
      extras?: Record<string, unknown>;
      primitives?: Array<{ material?: number; extras?: Record<string, unknown> }>;
    }>;
    materials?: Array<{ name?: string; extras?: Record<string, unknown> }>;
  };
  const result = new Map<string, readonly WorldPrimitivePolicy[]>();
  gltf.nodes?.forEach((node, nodeIndex) => {
    if (node.mesh == null) return;
    const sourceMesh = gltf.meshes?.[node.mesh];
    const primitives = sourceMesh?.primitives ?? [];
    const nodeName = node.name || `node${nodeIndex}`;
    const policies = primitives.map((primitive, primitiveIndex) => {
      const meshName =
        primitives.length === 1 ? nodeName : `${nodeName}_primitive${primitiveIndex}`;
      const sourceMaterial =
        primitive.material == null ? undefined : gltf.materials?.[primitive.material];
      const material =
        primitive.material == null
          ? '__default'
          : sourceMaterial?.name || `material-${primitive.material}`;
      const extras = [node.extras, sourceMesh?.extras, primitive.extras, sourceMaterial?.extras];
      const extraShader = extras
        .map(value => extraShaderFromExtras(value))
        .find((value): value is string => Boolean(value));
      const visibilityProfile = extras
        .map(value => stringFromExtras(value, 'requiem_visibility_profile', 'visibilityProfile'))
        .find((value): value is string => Boolean(value));
      const pvsPriority = extras
        .map(value => stringFromExtras(value, 'requiem_pvs_priority', 'pvsPriority'))
        .find((value): value is string => Boolean(value));
      const semanticRoles = extras
        .flatMap(value => [
          stringFromExtras(value, 'requiem_semantic_role', 'semanticRole'),
          stringFromExtras(value, 'requiem_material_role', 'materialRole'),
        ])
        .filter((value): value is string => Boolean(value));
      const visuallyPassthrough =
        extraShader?.toLowerCase() === 'water' ||
        semanticRoles.some(role =>
          /(?:^|[-_])(water|window|glass|banner|textile|ivy|vine|foliage|decal|ornament)(?:$|[-_])/i.test(
            role
          )
        );
      const explicitlyPassthrough =
        visuallyPassthrough ||
        extras.some(value => value?.passThrough === true) ||
        extras.some(value => value?.collision === false) ||
        extras.some(value => value?.clientPhysics === false) ||
        extras.some(value => value?.requiem_client_physics === false) ||
        extras.some(value => {
          const collision = stringFromExtras(value, 'requiem_collision')?.toLowerCase();
          return collision === 'visual-only' || collision === 'none' || collision === 'pass-through';
        });
      const explicitlySolid = extras.some(
        value =>
          value?.blocksPlayer === true ||
          value?.collision === true ||
          value?.clientPhysics === true ||
          value?.requiem_client_physics === true ||
          stringFromExtras(value, 'requiem_collision')?.toLowerCase() === 'solid' ||
          value?.physicsMode === 'static'
      );
      const collision =
        meshName !== 'CLOUD_MDF' &&
        (explicitlySolid || !explicitlyPassthrough);
      const collisionFlags = collisionFlagsFromExtras(extras, extraShader, semanticRoles);
      return {
        material,
        collision,
        collisionFlags,
        extraShader,
        visibilityProfile,
        pvsPriority,
      };
    });
    if (primitives.length === 1) {
      result.set(nodeName, policies);
    } else {
      policies.forEach((policy, primitiveIndex) => {
        result.set(`${nodeName}_primitive${primitiveIndex}`, [policy]);
      });
    }
  });
  return result;
}

function collisionFlagsFromExtras(
  extras: readonly (Record<string, unknown> | undefined)[],
  extraShader: string | undefined,
  semanticRoles: readonly string[]
): number {
  const kind = extras
    .map(value => stringFromExtras(value, 'requiem_collision_kind', 'collisionKind'))
    .find((value): value is string => Boolean(value))
    ?.toLowerCase();
  let flags = ShadoCollisionFlags.PlayerSolid;
  if (
    kind === 'terrain' ||
    kind === 'ground' ||
    extraShader?.toLowerCase() === 'grass' ||
    semanticRoles.some(role => /terrain|ground|path|road|walkable-field/i.test(role))
  ) flags |= ShadoCollisionFlags.Terrain;
  else if (kind === 'object' || kind === 'prop') flags |= ShadoCollisionFlags.StaticObject;
  else flags |= ShadoCollisionFlags.Architecture;
  if (
    extras.some(
      value => value?.alwaysResident === true || value?.requiem_physics_always_resident === true
    )
  ) {
    flags |= ShadoCollisionFlags.AlwaysResident;
  }
  return flags;
}

function extraShaderFromExtras(
  extras: Record<string, unknown> | undefined
): string | undefined {
  const eltania = extras?.eltania;
  if (!eltania || typeof eltania !== 'object' || Array.isArray(eltania)) return undefined;
  const value = (eltania as Record<string, unknown>).extraShader;
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function stringFromExtras(
  extras: Record<string, unknown> | undefined,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    const value = extras?.[key];
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return undefined;
}

function materialName(material: any, materialIndex: number): string {
  const selected = material?.subMaterials?.[materialIndex] ?? material;
  return selected?.name || selected?.id || '__default';
}

export function validateGlb(bytes: Uint8Array, file: string) {
  if (bytes.byteLength < 20 || String.fromCharCode(...bytes.subarray(0, 4)) !== 'glTF') {
    throw new Error(`World input '${file}' is not a binary GLB`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== bytes.byteLength) {
    throw new Error(`World input '${file}' has an invalid GLB header`);
  }
}
