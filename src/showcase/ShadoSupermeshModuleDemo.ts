import { BABYLON, type Scene } from '../babylon';
import {
  ShadoModuleDrawSet,
  splitMeshesIntoModules,
  type ShadoModuleDrawStats,
  type ShadoModuleMesh,
} from '../extensions/ShadoModuleDraw';

/**
 * End-to-end reference for migrating a variant supermesh to module draws.
 *
 * The layout this targets is the Ryzom-style `equip -> submesh` wardrobe: one
 * mesh per (body piece, outfit variant), all merged into a single supermesh,
 * with the unworn variants hidden per instance by a sentinel the vertex shader
 * turns into a degenerate position. That hiding happens after skinning, so
 * every actor pays for the whole wardrobe and displays a fraction of it.
 *
 * This file is deliberately small and complete rather than clever: read it
 * beside SUPERMESH_MODULE_MIGRATION.md. It owns no materials, because what a
 * module's material *is* stays the application's business - only the fact that
 * each module needs its **own** material is universal, and §6 of the guide
 * explains why.
 */

/** One source part of a wardrobe, before any merging. */
export type ShadoWardrobePart = {
  /** Body piece this part belongs to: chest, legs, hands... */
  piece: string;
  /** Which outfit variant of that piece this is. */
  variant: string;
  mesh: ShadoModuleMesh;
};

export type ShadoSupermeshMigrationOptions = {
  /**
   * Custom per-vertex streams to carry across the merge. MergeMeshes only
   * knows Babylon's own vertex kinds and silently drops everything else.
   */
  preserveAttributes?: readonly { kind: string; stride: number }[];
  /**
   * Builds the material for one module. Called once per module: a shared
   * material cannot work, because the module's compact index list is a
   * per-draw binding that lives in the material's WebGPU draw context.
   */
  createModuleMaterial?: (moduleKey: string, mesh: ShadoModuleMesh, moduleIndex: number) => any;
  /** Prefix for the merged mesh names. */
  name?: string;
};

/**
 * A migrated wardrobe: module geometry, its draw ownership, and the mapping
 * back to the parts each module came from.
 */
export class ShadoSupermeshModuleDemo {
  private readonly draws: ShadoModuleDrawSet;
  /** Module index -> the piece/variant it draws. */
  public readonly moduleParts: { piece: string; variant: string }[] = [];
  /** Per (actor, module) membership, mirrored from whatever the app writes. */
  private membership = new Uint8Array(0);
  private moduleCount = 0;

  private constructor(draws: ShadoModuleDrawSet) {
    this.draws = draws;
  }

  /**
   * Step 1-3 of the migration: split, own, materialize.
   *
   * Pass `collapse: true` to group everything under one key. That produces
   * exactly the single merged supermesh you started with and is the honest way
   * to land the change in two steps - ship the collapsed build, confirm nothing
   * moved, then turn the real grouping on.
   */
  public static migrate(
    scene: Scene,
    parts: readonly ShadoWardrobePart[],
    options: ShadoSupermeshMigrationOptions & { collapse?: boolean } = {}
  ): ShadoSupermeshModuleDemo {
    const meshes = parts.map((part) => part.mesh);
    const geometry = splitMeshesIntoModules(meshes, {
      // The grouping key is the whole design decision: everything sharing a key
      // is merged, and a module is shown or hidden as a unit. For a wardrobe
      // that is exactly (piece, variant), because an actor wears one variant
      // per piece.
      groupKey: (_mesh, index) =>
        options.collapse ? 'all' : `${parts[index].piece}:${parts[index].variant}`,
      preserveAttributes: options.preserveAttributes,
      name: (key) => `${options.name ?? 'wardrobe'}#${key}`,
    });

    const demo = new ShadoSupermeshModuleDemo(
      new ShadoModuleDrawSet(scene.getEngine(), geometry)
    );
    demo.moduleCount = geometry.length;
    for (const module of geometry) {
      const first = parts[module.sourceIndices[0]];
      demo.moduleParts.push({ piece: first.piece, variant: first.variant });
    }

    demo.draws.registerThinInstanceAttribute('matrix', 16);
    for (const [index, module] of demo.draws.modules.entries()) {
      // Actor transforms come from the arena, so the Babylon source mesh has no
      // authoritative world bounds; let the visibility pass own culling.
      module.mesh.alwaysSelectAsActiveMesh = true;
      const material = options.createModuleMaterial?.(module.key, module.mesh, index);
      if (material) {
        module.mesh.material = material;
        // WebGPU builds a bind group before the first draw: seed the index
        // resource now, then rebind per draw against the effect.
        module.selection?.bind(material);
      }
    }
    return demo;
  }

  public get modules() {
    return this.draws.modules;
  }

  public get stats(): ShadoModuleDrawStats {
    return this.draws.lastStats;
  }

  /** Adds one actor's instance to every module. Index matches the arena slot. */
  public addActor(matrix: any): number {
    return this.draws.addThinInstance(matrix);
  }

  /**
   * Step 4: records which variant an actor wears for one piece.
   *
   * A real integration mirrors this off the appearance write it already makes
   * rather than keeping a parallel table - see §4 of the guide. The array here
   * stands in for that mirror so the demo is self-contained.
   */
  public setWorn(actorIndex: number, piece: string, variant: string): void {
    const needed = (actorIndex + 1) * this.moduleCount;
    if (needed > this.membership.length) {
      const grown = new Uint8Array(Math.max(needed, this.membership.length * 2));
      grown.set(this.membership);
      this.membership = grown;
    }
    for (let module = 0; module < this.moduleCount; module++) {
      const part = this.moduleParts[module];
      if (part.piece !== piece) continue;
      this.membership[actorIndex * this.moduleCount + module] =
        part.variant === variant ? 1 : 0;
    }
  }

  /** Clears an actor's row, so a recycled slot cannot inherit an outfit. */
  public releaseActor(actorIndex: number): void {
    if (!this.moduleCount) return;
    this.membership.fill(
      0,
      actorIndex * this.moduleCount,
      (actorIndex + 1) * this.moduleCount
    );
  }

  /**
   * Step 5: rebuild the buckets. Call once per frame after visibility, and
   * again after anything rewrites what an actor is wearing.
   */
  public refresh(visibleActorIndices: Uint32Array): ShadoModuleDrawStats {
    return this.draws.refresh(
      visibleActorIndices,
      (actorIndex, moduleIndex) =>
        this.membership[actorIndex * this.moduleCount + moduleIndex] === 1
    );
  }

  /**
   * Call from a module material's per-draw bind, after binding the arena. The
   * module's compact list replaces the arena's global visible list, and the
   * returned count is what the draw should submit.
   */
  public bindModule(moduleIndex: number, target: any): number {
    return this.draws.bindSelection(moduleIndex, target);
  }

  public dispose(): void {
    for (const module of this.draws.modules) module.mesh.dispose();
    this.draws.dispose();
    this.membership = new Uint8Array(0);
  }
}

/**
 * Builds a throwaway wardrobe so the demo runs with no assets: `pieces` body
 * pieces, `variants` outfits each, one box per part.
 */
export function createDemoWardrobe(
  scene: Scene,
  pieces: readonly string[],
  variants: number
): ShadoWardrobePart[] {
  const parts: ShadoWardrobePart[] = [];
  for (const [pieceIndex, piece] of pieces.entries()) {
    for (let variant = 0; variant < variants; variant++) {
      const mesh = BABYLON.MeshBuilder.CreateBox(
        `${piece}_${variant}`,
        { size: 0.4, height: 0.4 + variant * 0.05 },
        scene as any
      );
      mesh.position.y = pieceIndex * 0.5;
      parts.push({ piece, variant: String(variant), mesh: mesh as unknown as ShadoModuleMesh });
    }
  }
  return parts;
}

/**
 * The whole slice in one call, for a smoke run: build a wardrobe, migrate it,
 * dress `actorCount` actors, refresh, and report the work reduction.
 *
 * Returns the same numbers the guide asks you to report, so a migration can be
 * proven rather than asserted.
 */
export function runSupermeshModuleDemo(
  scene: Scene,
  options: { pieces?: readonly string[]; variants?: number; actorCount?: number } = {}
): { stats: ShadoModuleDrawStats; demo: ShadoSupermeshModuleDemo } {
  const pieces = options.pieces ?? ['chest', 'legs', 'feet', 'hands', 'head'];
  const variants = options.variants ?? 4;
  const actorCount = options.actorCount ?? 64;

  const demo = ShadoSupermeshModuleDemo.migrate(
    scene,
    createDemoWardrobe(scene, pieces, variants),
    { name: 'demo' }
  );

  const visible = new Uint32Array(actorCount);
  for (let actor = 0; actor < actorCount; actor++) {
    demo.addActor(BABYLON.Matrix.Translation(actor * 1.2, 0, 0));
    for (const piece of pieces) {
      demo.setWorn(actor, piece, String(actor % variants));
    }
    visible[actor] = actor;
  }

  return { stats: demo.refresh(visible), demo };
}
