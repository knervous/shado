import { ShadoInstanceDrawSelection } from '../ShadoInstanceContainer/ShadoInstanceDrawSelection';
import type { ShadoModuleGeometry, ShadoModuleMesh } from './ShadoModuleGeometry';

/**
 * Whether an actor shows a module this frame.
 *
 * Called once per visible actor per module, so keep it an array read. The
 * natural source is whatever the application already writes to decide which
 * variant an instance wears; deriving a second source of truth here is what
 * makes buckets and pixels disagree.
 */
export type ShadoModuleMembership = (actorIndex: number, moduleIndex: number) => boolean;

/** One module and the compact actor list its draw submits. */
export type ShadoModuleDraw = {
  key: string;
  mesh: ShadoModuleMesh;
  sourceIndices: number[];
  /** Vertices in this module, cached for the work-reduction stats. */
  vertexCount: number;
  /**
   * Compact actor indices for this draw, or null when the set holds a single
   * module that owns every submesh and has nothing to select on.
   */
  selection: ShadoInstanceDrawSelection | null;
  /** Actors submitted by the last refresh. */
  drawnCount: number;
};

export type ShadoModuleDrawStats = {
  visibleActors: number;
  moduleCount: number;
  populatedModules: number;
  submittedVertices: number;
  /** What one merged supermesh would have submitted for the same actors. */
  baselineVertices: number;
  /** baselineVertices / submittedVertices; 1 when nothing is avoided. */
  vertexWorkReduction: number;
};

/**
 * Per-module draw ownership over one shared actor arena.
 *
 * The arena, its visibility pass and its actor records stay exactly as they
 * were: this only changes which actors each piece of geometry draws. A module
 * whose bucket is empty is switched off entirely, which also keeps it out of
 * GPU picking.
 *
 * Materials are deliberately not created here. A module needs its own material
 * because its compact index list is a per-draw binding, and on WebGPU that
 * binding lives in the material's draw context - one material shared across
 * module meshes would let the last bind win. What that material *is* remains
 * the application's business.
 */
export class ShadoModuleDrawSet {
  private readonly _modules: ShadoModuleDraw[] = [];
  private scratch: Uint32Array[] = [];
  private committed: Uint32Array[] = [];
  private committedCounts: number[] = [];
  private stats: ShadoModuleDrawStats;

  /**
   * @param engine  Babylon engine, for the selections' GPU mirrors.
   * @param modules Geometry from `splitMeshesIntoModules`, or any equivalent.
   */
  constructor(engine: any, modules: readonly ShadoModuleGeometry[]) {
    // A single module owns the whole model, so every visible actor draws it and
    // there is no subset to publish. Skipping the selection keeps that case
    // byte-for-byte the supermesh path it replaces.
    const split = modules.length > 1;
    for (const module of modules) {
      this._modules.push({
        key: module.key,
        mesh: module.mesh,
        sourceIndices: module.sourceIndices,
        vertexCount: module.mesh.getTotalVertices() || 0,
        selection: split ? new ShadoInstanceDrawSelection(engine) : null,
        drawnCount: 0,
      });
      this.scratch.push(new Uint32Array(0));
      this.committed.push(new Uint32Array(0));
      this.committedCounts.push(-1);
    }
    this.stats = {
      visibleActors: 0,
      moduleCount: this._modules.length,
      populatedModules: 0,
      submittedVertices: 0,
      baselineVertices: 0,
      vertexWorkReduction: 1,
    };
    // WebGPU builds a material's bind group before its first draw, so give the
    // index resource something real to point at up front. Callers bind the
    // seeded selection to their material right after construction.
    for (const module of this._modules) module.selection?.commit();
  }

  public get modules(): readonly ShadoModuleDraw[] {
    return this._modules;
  }

  /** False when one module owns everything and no selection lists exist. */
  public get isSplit(): boolean {
    return this._modules.length > 1;
  }

  public get lastStats(): ShadoModuleDrawStats {
    return this.stats;
  }

  /** Registers a thin-instance attribute on every module mesh. */
  public registerThinInstanceAttribute(kind: string, stride: number): void {
    for (const module of this._modules) {
      module.mesh.thinInstanceRegisterAttribute(kind, stride);
    }
  }

  /**
   * Adds one thin instance to every module and returns its index.
   *
   * Every module carries every instance so its thin-instance count stays
   * aligned with the actor arena - the count is the draw-count adapter, and
   * GPU picking assigns its ids over that same span. Only the first module
   * refreshes bounding info: a module draw set is expected to run with
   * `alwaysSelectAsActiveMesh`, which makes those bounds inert, and refreshing
   * per module would make spawning O(modules x instances).
   */
  public addThinInstance(matrix: any): number {
    let index = -1;
    for (let i = 0; i < this._modules.length; i++) {
      const added = this._modules[i].mesh.thinInstanceAdd(matrix, i === 0);
      if (i === 0) index = added;
    }
    return index;
  }

  /** Writes one instance's matrix on every module. */
  public setThinInstanceMatrixAt(index: number, matrix: any): void {
    for (let i = 0; i < this._modules.length; i++) {
      this._modules[i].mesh.thinInstanceSetMatrixAt(index, matrix, i === 0);
    }
  }

  public get thinInstanceCount(): number {
    return this._modules[0]?.mesh.thinInstanceCount ?? 0;
  }

  /**
   * Rebuilds every module's compact actor list from the arena's visible list.
   *
   * Pass `container.visibleActorIndices` straight through: preserving that
   * order keeps each draw in whatever order the culler produced. Cheap enough
   * to run every frame - one pass over the visible actors per module - and the
   * GPU upload is skipped for any bucket that came out element-wise unchanged,
   * which is the common case between two frames with the same crowd.
   */
  public refresh(
    visibleActorIndices: Uint32Array,
    isMember: ShadoModuleMembership
  ): ShadoModuleDrawStats {
    const visibleCount = visibleActorIndices.length;
    let submittedVertices = 0;
    let populatedModules = 0;
    let sourceVertices = 0;

    for (let moduleIndex = 0; moduleIndex < this._modules.length; moduleIndex++) {
      const module = this._modules[moduleIndex];
      sourceVertices += module.vertexCount;

      if (!module.selection) {
        // Unsplit: the whole model draws for every visible actor.
        module.drawnCount = visibleCount;
        submittedVertices += module.vertexCount * visibleCount;
        if (visibleCount) populatedModules++;
        continue;
      }

      if (this.scratch[moduleIndex].length < visibleCount) {
        this.scratch[moduleIndex] = new Uint32Array(visibleCount);
        this.committed[moduleIndex] = new Uint32Array(visibleCount);
        this.committedCounts[moduleIndex] = -1;
      }
      const scratch = this.scratch[moduleIndex];
      let count = 0;
      for (let i = 0; i < visibleCount; i++) {
        const actorIndex = visibleActorIndices[i];
        if (isMember(actorIndex, moduleIndex)) scratch[count++] = actorIndex;
      }

      let changed = count !== this.committedCounts[moduleIndex];
      if (!changed) {
        const committed = this.committed[moduleIndex];
        for (let i = 0; i < count; i++) {
          if (committed[i] !== scratch[i]) {
            changed = true;
            break;
          }
        }
      }
      if (changed) {
        this.committed[moduleIndex].set(scratch.subarray(0, count));
        this.committedCounts[moduleIndex] = count;
        module.selection.setActorIndices(scratch.subarray(0, count));
      }

      module.drawnCount = count;
      // An empty bucket has nothing to submit, and switching it off here also
      // keeps it out of the GPU picking pass.
      module.mesh.isVisible = count > 0;
      if (count) populatedModules++;
      submittedVertices += module.vertexCount * count;
    }

    const baselineVertices = sourceVertices * visibleCount;
    this.stats = {
      visibleActors: visibleCount,
      moduleCount: this._modules.length,
      populatedModules,
      submittedVertices,
      baselineVertices,
      vertexWorkReduction:
        submittedVertices > 0 ? baselineVertices / submittedVertices : 1,
    };
    return this.stats;
  }

  /**
   * Uploads and binds one module's index list.
   *
   * Call from the module material's per-draw bind, after binding the arena:
   * the selection replaces the arena's global visible list for this draw.
   * Returns the instance count the draw should submit.
   */
  public bindSelection(moduleIndex: number, target: any): number {
    const module = this._modules[moduleIndex];
    if (!module?.selection) return module?.drawnCount ?? 0;
    module.selection.commit();
    module.selection.bind(target);
    return module.selection.visibleCount;
  }

  /** Disposes the selections. Module meshes belong to the caller. */
  public dispose(): void {
    for (const module of this._modules) module.selection?.dispose();
    this._modules.length = 0;
    this.scratch = [];
    this.committed = [];
    this.committedCounts = [];
  }
}
