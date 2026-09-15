import {
  BABYLON,
  type AbstractEngine,
  type Buffer,
  type Camera,
  type Mesh,
  type Observer,
  type Scene,
  type ShaderMaterial,
} from '../babylon';
import type { ShadoTextureAtlas } from './ShadoTextureAtlas';
import type { ShadoSpriteAlphaMode } from './ShadoDynamicEntityContainer';
import { ShadoSprite2DGpuMotion } from './ShadoSprite2DGpuMotion';
import { ShadoSprite2DGpuVisibility } from './ShadoSprite2DGpuVisibility';
import type { ShadoSprite2DMotionConfig } from './ShadoSprite2DMotionKernel';

const FLOATS_PER_SPRITE = 12;
const BYTES_PER_SPRITE = FLOATS_PER_SPRITE * 4;
const DEFAULT_TILE_SIZE = 8;

export interface ShadoSprite2DInput {
  id: string;
  textureKey?: string;
  position: readonly [number, number];
  size: readonly [number, number];
  rotationRad?: number;
  rotationDeg?: number;
  opacity?: number;
  visible?: boolean;
  selected?: boolean;
  highlighted?: boolean;
  layer?: number;
  order?: number;
  minPixelSize?: number;
}

export interface ShadoSprite2DRendererOptions {
  alphaMode?: ShadoSpriteAlphaMode;
  alphaCutoff?: number;
  tileSize?: number;
  minPixelSize?: number;
  /** WebGPU cutout batches compact visible moving sprites without readback. */
  gpuCulling?: boolean;
}

export interface ShadoSprite2DView {
  center: readonly [number, number];
  halfExtent: readonly [number, number];
  viewportPixels: readonly [number, number];
}

export interface ShadoSprite2DPickResult {
  id: string;
  sprite: ShadoSprite2DInput;
  world: readonly [number, number];
  uv: readonly [number, number];
}

export interface ShadoSprite2DPositionUpdate {
  id: string;
  position: readonly [number, number];
}

export interface ShadoSprite2DStats {
  total: number;
  visible: number;
  tileCount: number;
  recordBytes: number;
  gpuCapacityBytes: number;
  drawListRebuilds: number;
  gpuMotionDispatches: number;
  gpuMotionError: string;
  gpuCullingActive: boolean;
  gpuCullingDispatches: number;
  indirectDrawActive: boolean;
  gpuCullingError: string;
}

export type ShadoSprite2DCpuAccess =
  | { tier: 'all' }
  | { tier: 'visible' }
  | { tier: 'selected' }
  | { tier: 'range'; start: number; count: number }
  | { tier: 'ids'; ids: readonly string[] };

export interface ShadoSprite2DCpuPosition {
  id: string;
  index: number;
  position: readonly [number, number];
  velocity?: readonly [number, number];
}

export interface ShadoSprite2DCpuSnapshot {
  source: 'cpu' | 'gpu';
  generation: number;
  dispatchesAtRequest: number;
  dispatchesAtCompletion: number;
  /** False means more GPU simulation was submitted while the read was pending. */
  inBand: boolean;
  /** True means population/authority changed, so consumers must discard it. */
  stale: boolean;
  entries: ShadoSprite2DCpuPosition[];
}

type SpriteRecord = {
  /** Normalized and owned by the renderer; `position` is updated in place. */
  input: ShadoSprite2DInput & { position: [number, number] };
  insertionOrder: number;
  tileKey: string;
  tileX: number;
  tileY: number;
  /** Index in its tile's record list, for O(1) removal. */
  tileIndex: number;
  /** Index in the draw list, or -1 when not drawn. */
  slot: number;
  /** The sort key the record was placed in the draw list with. */
  slotLayer: number;
  slotOrder: number;
  removed: boolean;
};

/**
 * Optimized locked-camera 2D renderer.
 *
 * It deliberately lives beside ShadoDynamicEntityRenderer instead of replacing
 * it. Consumers can retain the dynamic renderer as a complete compatibility
 * path while adopting the compact 48-byte, tiled 2D path incrementally.
 */
export class ShadoSprite2DRenderer {
  public readonly mesh: Mesh;
  public readonly material: ShaderMaterial;

  private readonly engine: AbstractEngine;
  private readonly records = new Map<string, SpriteRecord>();
  private readonly tileCache = new Map<string, SpriteRecord[]>();
  /**
   * The packed instance order: drawable records in the intersecting tiles,
   * sorted by layer, order and insertion. Kept across frames. A position change
   * repacks one slot; an arrival or departure is merged in place; only a change
   * of tile bounds or LOD bucket (or a mutation touching much of the list)
   * rebuilds it.
   */
  private drawList: SpriteRecord[] = [];
  private drawTileBounds: readonly number[] = [0, 0, -1, -1];
  private drawBoundsSignature = '';
  private membershipDirty = true;
  private readonly pendingMembership = new Set<SpriteRecord>();
  private dirtySlotStart = Number.POSITIVE_INFINITY;
  private dirtySlotEnd = -1;
  private readonly tileSize: number;
  private defaultMinPixelSize: number;
  private readonly alphaMode: ShadoSpriteAlphaMode;
  private readonly alphaCutoff: number;
  private readonly beforeRenderObserver: Observer<Scene> | null;
  private instanceBuffer?: Buffer;
  private capacity = 0;
  private packed = new Float32Array(0);
  private view: ShadoSprite2DView = {
    center: [0, 0],
    halfExtent: [1, 1],
    viewportPixels: [1, 1],
  };
  private revision = 0;
  private insertionCounter = 0;
  private maxHalfExtent = 0;
  private drawSignature = '';
  private drawListRebuilds = 0;
  private gpuMotion?: ShadoSprite2DGpuMotion;
  private gpuVisibility?: ShadoSprite2DGpuVisibility;
  private gpuMotionActive = false;
  private gpuCullingActive = false;
  private readonly gpuCullingRequested: boolean;
  private gpuIndexIds: string[] = [];
  private gpuGeneration = 0;
  private indirectDrawContext?: any;
  private originalIndirectBuffer?: unknown;

  public constructor(
    private readonly scene: Scene,
    private readonly atlas: ShadoTextureAtlas,
    options: ShadoSprite2DRendererOptions = {}
  ) {
    this.engine = scene.getEngine();
    this.tileSize = Math.max(0.25, options.tileSize ?? DEFAULT_TILE_SIZE);
    this.defaultMinPixelSize = Math.max(0, options.minPixelSize ?? 0.75);
    this.alphaMode = options.alphaMode ?? 'cutout';
    this.alphaCutoff = Math.max(0, Math.min(1, options.alphaCutoff ?? 0.35));
    // Atomic compaction has a measurable fixed cost and is not universally a
    // win for four-vertex quads. Keep it explicitly opt-in until the caller's
    // population/visibility ratio proves it beneficial.
    this.gpuCullingRequested = options.gpuCulling === true;
    this.mesh = this.createQuad();
    this.material = this.createMaterial();
    this.mesh.material = this.material;
    this.mesh.alwaysSelectAsActiveMesh = true;
    this.ensureCapacity(1);
    if (this.engine.isWebGPU) {
      this.gpuMotion = new ShadoSprite2DGpuMotion(this.engine as any);
      this.material.setStorageBuffer('uMotionState', this.gpuMotion.state);
      this.gpuVisibility = new ShadoSprite2DGpuVisibility(this.engine as any, this.gpuMotion.state);
      this.bindGpuVisibility();
    }
    this.beforeRenderObserver = scene.onBeforeRenderObservable.add(() => {
      this.rebuildVisibleDrawList();
      this.material.setVector2(
        'uCameraCenter',
        new BABYLON.Vector2(this.view.center[0], this.view.center[1])
      );
      this.material.setVector2(
        'uCameraHalfExtent',
        new BABYLON.Vector2(this.view.halfExtent[0], this.view.halfExtent[1])
      );
      this.material.setFloat('uInstanceCount', Math.max(1, this.drawList.length));
      this.material.setFloat('uUseGpuMotion', this.gpuMotionActive ? 1 : 0);
      this.material.setFloat('uUseGpuCulling', this.gpuCullingActive ? 1 : 0);
      this.material.setFloat('uViewportHeight', this.view.viewportPixels[1]);
      this.material.setFloat('uMinimumPixelSize', this.defaultMinPixelSize);
      if (this.gpuCullingActive && this.gpuVisibility) {
        this.gpuVisibility.dispatch(
          this.view.center,
          this.view.halfExtent,
          this.view.viewportPixels[1],
          this.defaultMinPixelSize
        );
        this.tryAttachIndirectDraw();
      }
    });
  }

  public upsert(input: ShadoSprite2DInput): void {
    this.upsertMany([input]);
  }

  public upsertMany(inputs: readonly ShadoSprite2DInput[]): void {
    if (!inputs.length) return;
    this.restoreCpuAuthority();
    for (const input of inputs) {
      const next = normalizeSprite(input) as SpriteRecord['input'];
      const current = this.records.get(input.id);
      this.maxHalfExtent = Math.max(
        this.maxHalfExtent,
        Math.hypot(next.size[0] * 0.5, next.size[1] * 0.5)
      );
      if (!current) {
        const record: SpriteRecord = {
          input: next,
          insertionOrder: this.insertionCounter++,
          tileKey: '',
          tileX: 0,
          tileY: 0,
          tileIndex: -1,
          slot: -1,
          slotLayer: 0,
          slotOrder: 0,
          removed: false,
        };
        this.records.set(input.id, record);
        this.placeInTile(record);
        this.pendingMembership.add(record);
        continue;
      }
      const previous = current.input;
      current.input = next;
      if (
        previous.visible !== next.visible ||
        previous.layer !== next.layer ||
        previous.order !== next.order ||
        previous.minPixelSize !== next.minPixelSize ||
        previous.size[0] !== next.size[0] ||
        previous.size[1] !== next.size[1]
      ) {
        this.pendingMembership.add(current);
      }
      this.moveTileIfNeeded(current);
      this.repackSlot(current);
    }
    this.revision++;
  }

  public remove(id: string): boolean {
    const record = this.records.get(id);
    if (!record) return false;
    this.restoreCpuAuthority();
    this.records.delete(id);
    this.removeFromTile(record);
    record.removed = true;
    if (record.slot >= 0) this.pendingMembership.add(record);
    else this.pendingMembership.delete(record);
    this.revision++;
    return true;
  }

  public clear(): void {
    if (!this.records.size) return;
    this.records.clear();
    this.rebuildTileCache();
  }

  public setMinPixelSize(value: number): void {
    const next = Math.max(0, value);
    if (Math.abs(next - this.defaultMinPixelSize) < 0.0001) return;
    this.defaultMinPixelSize = next;
    this.membershipDirty = true;
    this.drawSignature = '';
  }

  public setVisible(id: string, visible: boolean): boolean {
    const record = this.records.get(id);
    if (!record || record.input.visible === visible) return false;
    record.input.visible = visible;
    this.restoreCpuAuthority();
    this.pendingMembership.add(record);
    this.revision++;
    return true;
  }

  public setPosition(id: string, position: readonly [number, number]): boolean {
    return this.setPositions([{ id, position }]) > 0;
  }

  /** Apply a simulation tick with one tile-cache revision and one GPU repack. */
  public setPositions(updates: readonly ShadoSprite2DPositionUpdate[]): number {
    let changed = 0;
    for (const update of updates) {
      const record = this.records.get(update.id);
      if (!record) continue;
      const position = record.input.position;
      const x = update.position[0];
      const y = update.position[1];
      if (position[0] === x && position[1] === y) continue;
      if (!changed) this.restoreCpuAuthority();
      position[0] = x;
      position[1] = y;
      this.moveTileIfNeeded(record);
      if (record.slot >= 0) {
        const offset = record.slot * FLOATS_PER_SPRITE;
        this.packed[offset] = x;
        this.packed[offset + 1] = y;
        this.markSlotDirty(record.slot);
      }
      changed++;
    }
    if (changed) this.revision++;
    return changed;
  }

  public setView(view: ShadoSprite2DView): void {
    this.view = {
      center: [view.center[0], view.center[1]],
      halfExtent: [Math.max(0.0001, view.halfExtent[0]), Math.max(0.0001, view.halfExtent[1])],
      viewportPixels: [Math.max(1, view.viewportPixels[0]), Math.max(1, view.viewportPixels[1])],
    };
  }

  public setViewFromOrthographicCamera(camera: Camera): void {
    const position = camera.globalPosition ?? camera.position;
    this.setView({
      center: [position.x, position.z],
      halfExtent: [
        Math.abs(((camera as any).orthoRight ?? 1) - ((camera as any).orthoLeft ?? -1)) * 0.5,
        Math.abs(((camera as any).orthoTop ?? 1) - ((camera as any).orthoBottom ?? -1)) * 0.5,
      ],
      viewportPixels: [this.engine.getRenderWidth(), this.engine.getRenderHeight()],
    });
  }

  /** Keep position and velocity entirely in WebGPU storage after one upload. */
  public enableGpuMotion(
    config: ShadoSprite2DMotionConfig,
    globalStart = 0,
    nowMs = performance.now()
  ): void {
    if (!this.gpuMotion) throw new Error('GPU sprite motion requires WebGPU.');
    const records = this.sortedDrawableRecords();
    const positions = new Float32Array(records.length * 2);
    for (let index = 0; index < records.length; index++) {
      positions[index * 2] = records[index].input.position[0];
      positions[index * 2 + 1] = records[index].input.position[1];
    }
    this.gpuMotion.setPopulation(positions, config, globalStart);
    this.gpuIndexIds = records.map(record => record.input.id);
    this.gpuGeneration++;
    this.material.setStorageBuffer('uMotionState', this.gpuMotion.state);
    this.gpuVisibility?.setMotionState(this.gpuMotion.state);
    if (this.gpuVisibility && this.gpuCullingRequested && this.alphaMode === 'cutout') {
      const packed = new Float32Array(Math.max(1, records.length) * FLOATS_PER_SPRITE);
      for (let index = 0; index < records.length; index++) {
        this.packSpriteInto(packed, index, records[index].input);
      }
      this.detachIndirectDraw();
      this.gpuVisibility.setPopulation(packed, records.length);
      this.bindGpuVisibility();
      this.gpuCullingActive = records.length > 0;
    } else {
      this.gpuCullingActive = false;
    }
    this.gpuMotionActive = true;
    this.drawSignature = '';
    this.gpuMotion.dispatch(nowMs, 0);
  }

  public configureGpuMotion(config: ShadoSprite2DMotionConfig): void {
    if (this.gpuMotionActive) this.gpuMotion?.configure(config);
  }

  public stepGpuMotion(nowMs: number, dtSeconds: number): boolean {
    return this.gpuMotionActive ? (this.gpuMotion?.dispatch(nowMs, dtSeconds) ?? false) : false;
  }

  public get isGpuMotionEnabled(): boolean {
    return this.gpuMotionActive;
  }

  /**
   * Request only the CPU data a caller needs. GPU reads are asynchronous and
   * versioned; ordinary rendering never invokes this path.
   */
  public async readCpuPositions(
    access: ShadoSprite2DCpuAccess = { tier: 'all' }
  ): Promise<ShadoSprite2DCpuSnapshot> {
    if (!this.gpuMotionActive || !this.gpuMotion) {
      const records = this.sortedDrawableRecords();
      const allIds = records.map(record => record.input.id);
      const ids = selectCpuAccessIds(
        access,
        allIds,
        this.drawList.map(record => record.input.id),
        id => this.records.get(id)?.input.selected === true
      );
      const indexById = new Map(allIds.map((id, index) => [id, index]));
      return {
        source: 'cpu',
        generation: this.revision,
        dispatchesAtRequest: 0,
        dispatchesAtCompletion: 0,
        inBand: true,
        stale: false,
        entries: ids.flatMap(id => {
          const record = this.records.get(id);
          return record
            ? [{ id, index: indexById.get(id) ?? -1, position: record.input.position }]
            : [];
        }),
      };
    }

    const generation = this.gpuGeneration;
    const dispatchesAtRequest = this.gpuMotion.dispatchCount;
    const ids = selectCpuAccessIds(
      access,
      this.gpuIndexIds,
      this.drawList.map(record => record.input.id),
      id => this.records.get(id)?.input.selected === true
    );
    const indexById = new Map(this.gpuIndexIds.map((id, index) => [id, index]));
    const indexed = ids
      .map(id => ({ id, index: indexById.get(id) }))
      .filter((entry): entry is { id: string; index: number } => entry.index !== undefined)
      .sort((a, b) => a.index - b.index);
    const runs: Array<{ start: number; entries: typeof indexed }> = [];
    for (const entry of indexed) {
      const run = runs.at(-1);
      if (!run || entry.index !== run.start + run.entries.length) {
        runs.push({ start: entry.index, entries: [entry] });
      } else {
        run.entries.push(entry);
      }
    }
    const chunks = await Promise.all(
      runs.map(async run => ({
        run,
        state: await this.gpuMotion!.readStateRange(run.start, run.entries.length),
      }))
    );
    const entries: ShadoSprite2DCpuPosition[] = [];
    for (const { run, state } of chunks) {
      for (let local = 0; local < run.entries.length; local++) {
        const entry = run.entries[local];
        const offset = local * 4;
        entries.push({
          id: entry.id,
          index: entry.index,
          position: [state[offset], state[offset + 1]],
          velocity: [state[offset + 2], state[offset + 3]],
        });
      }
    }
    const dispatchesAtCompletion = this.gpuMotion.dispatchCount;
    return {
      source: 'gpu',
      generation,
      dispatchesAtRequest,
      dispatchesAtCompletion,
      inBand: dispatchesAtRequest === dispatchesAtCompletion,
      stale: generation !== this.gpuGeneration || !this.gpuMotionActive,
      entries,
    };
  }

  public pickScreen(
    screenX: number,
    screenY: number,
    viewportWidth = this.view.viewportPixels[0],
    viewportHeight = this.view.viewportPixels[1]
  ): ShadoSprite2DPickResult | null {
    // GPU motion deliberately has no per-frame readback, so CPU coordinates
    // are stale. Do not report a false hit.
    if (this.gpuMotionActive) return null;
    const worldX =
      this.view.center[0] + ((screenX / viewportWidth) * 2 - 1) * this.view.halfExtent[0];
    const worldY =
      this.view.center[1] + (1 - (screenY / viewportHeight) * 2) * this.view.halfExtent[1];
    const candidates = this.candidatesForBounds(
      worldX - this.maxHalfExtent,
      worldY - this.maxHalfExtent,
      worldX + this.maxHalfExtent,
      worldY + this.maxHalfExtent
    ).filter(record => !record.removed);
    candidates.sort(compareSpriteRecords).reverse();

    for (const record of candidates) {
      const sprite = record.input;
      if (sprite.visible === false) continue;
      const dx = worldX - sprite.position[0];
      const dy = worldY - sprite.position[1];
      const rotation = sprite.rotationRad ?? 0;
      const c = Math.cos(-rotation);
      const s = Math.sin(-rotation);
      const localX = dx * c - dy * s;
      const localY = dx * s + dy * c;
      const halfWidth = sprite.size[0] * 0.5;
      const halfHeight = sprite.size[1] * 0.5;
      if (Math.abs(localX) > halfWidth || Math.abs(localY) > halfHeight) continue;
      return {
        id: sprite.id,
        sprite,
        world: [worldX, worldY],
        uv: [localX / sprite.size[0] + 0.5, 0.5 - localY / sprite.size[1]],
      };
    }
    return null;
  }

  public getStats(): ShadoSprite2DStats {
    return {
      total: this.records.size,
      visible: this.drawList.length,
      tileCount: this.tileCache.size,
      recordBytes: BYTES_PER_SPRITE,
      gpuCapacityBytes: this.capacity * BYTES_PER_SPRITE,
      drawListRebuilds: this.drawListRebuilds,
      gpuMotionDispatches: this.gpuMotion?.dispatchCount ?? 0,
      gpuMotionError: this.gpuMotion?.lastError ?? '',
      gpuCullingActive: this.gpuCullingActive,
      gpuCullingDispatches: this.gpuVisibility?.dispatchCount ?? 0,
      indirectDrawActive: !!this.indirectDrawContext,
      gpuCullingError: this.gpuVisibility?.lastError ?? '',
    };
  }

  public dispose(): void {
    if (this.beforeRenderObserver) {
      this.scene.onBeforeRenderObservable.remove(this.beforeRenderObserver);
    }
    this.mesh.forcedInstanceCount = 0;
    for (const kind of ['iTransform', 'iUvRect', 'iState']) {
      if (this.mesh.isVerticesDataPresent(kind)) this.mesh.removeVerticesData(kind);
    }
    this.instanceBuffer?.dispose();
    this.detachIndirectDraw();
    this.gpuVisibility?.dispose();
    this.gpuMotion?.dispose();
    this.material.dispose();
    this.mesh.dispose();
  }

  private rebuildTileCache(): void {
    // A population mutation invalidates the stable instance-to-state mapping.
    // Call enableGpuMotion again after the batch mutation is complete.
    this.restoreCpuAuthority();
    this.tileCache.clear();
    this.maxHalfExtent = 0;
    for (const record of this.drawList) record.slot = -1;
    this.drawList = [];
    this.pendingMembership.clear();
    for (const record of this.records.values()) {
      const sprite = record.input;
      this.maxHalfExtent = Math.max(
        this.maxHalfExtent,
        Math.hypot(sprite.size[0] * 0.5, sprite.size[1] * 0.5)
      );
      record.slot = -1;
      this.placeInTile(record);
    }
    this.membershipDirty = true;
    this.revision++;
    this.drawSignature = '';
  }

  private placeInTile(record: SpriteRecord): void {
    const tileX = Math.floor(record.input.position[0] / this.tileSize);
    const tileY = Math.floor(record.input.position[1] / this.tileSize);
    const key = `${tileX}:${tileY}`;
    let tile = this.tileCache.get(key);
    if (!tile) this.tileCache.set(key, (tile = []));
    record.tileKey = key;
    record.tileX = tileX;
    record.tileY = tileY;
    record.tileIndex = tile.length;
    tile.push(record);
  }

  private removeFromTile(record: SpriteRecord): void {
    const tile = this.tileCache.get(record.tileKey);
    if (!tile || tile[record.tileIndex] !== record) return;
    const last = tile.pop()!;
    if (last !== record) {
      tile[record.tileIndex] = last;
      last.tileIndex = record.tileIndex;
    }
    if (!tile.length) this.tileCache.delete(record.tileKey);
    record.tileIndex = -1;
  }

  /** Re-files a record whose position left its tile; a change of drawn-ness is queued. */
  private moveTileIfNeeded(record: SpriteRecord): void {
    const tileX = Math.floor(record.input.position[0] / this.tileSize);
    const tileY = Math.floor(record.input.position[1] / this.tileSize);
    if (tileX === record.tileX && tileY === record.tileY) return;
    const wasInBounds = this.tileInDrawBounds(record.tileX, record.tileY);
    this.removeFromTile(record);
    this.placeInTile(record);
    if (wasInBounds !== this.tileInDrawBounds(tileX, tileY)) {
      this.pendingMembership.add(record);
    }
  }

  private tileInDrawBounds(tileX: number, tileY: number): boolean {
    const bounds = this.drawTileBounds;
    return tileX >= bounds[0] && tileX <= bounds[2] && tileY >= bounds[1] && tileY <= bounds[3];
  }

  private repackSlot(record: SpriteRecord): void {
    if (record.slot < 0) return;
    this.packSprite(record.slot, record.input);
    this.markSlotDirty(record.slot);
  }

  private markSlotDirty(slot: number): void {
    if (slot < this.dirtySlotStart) this.dirtySlotStart = slot;
    if (slot + 1 > this.dirtySlotEnd) this.dirtySlotEnd = slot + 1;
  }

  private shouldDraw(record: SpriteRecord, pixelsPerUnit: number): boolean {
    const sprite = record.input;
    return (
      !record.removed &&
      sprite.visible !== false &&
      this.tileInDrawBounds(record.tileX, record.tileY) &&
      Math.max(sprite.size[0], sprite.size[1]) * pixelsPerUnit >=
        (sprite.minPixelSize ?? this.defaultMinPixelSize)
    );
  }

  private rebuildVisibleDrawList(): void {
    if (this.gpuMotionActive) {
      const signature = `gpu:${this.revision}`;
      if (signature === this.drawSignature) return;
      this.drawSignature = signature;
      // The GPU owns motion: every drawable sprite is in the list, in stable order.
      this.replaceDrawList(this.sortedDrawableRecords());
      this.membershipDirty = true;
      return;
    }
    const minX = this.view.center[0] - this.view.halfExtent[0] - this.maxHalfExtent;
    const maxX = this.view.center[0] + this.view.halfExtent[0] + this.maxHalfExtent;
    const minY = this.view.center[1] - this.view.halfExtent[1] - this.maxHalfExtent;
    const maxY = this.view.center[1] + this.view.halfExtent[1] + this.maxHalfExtent;
    const pixelsPerUnit = this.view.viewportPixels[1] / (this.view.halfExtent[1] * 2);
    const lodBucket = Math.round(Math.log2(Math.max(0.0001, pixelsPerUnit)) * 8);
    const tileBounds = this.tileBounds(minX, minY, maxX, maxY);
    // Every sprite in the intersecting cached tiles is kept: this deliberate
    // one-tile overdraw lets sub-cell camera pans reuse the exact GPU list, and
    // the list is rebuilt only when a tile boundary or LOD bucket changes.
    const boundsSignature = `${tileBounds.join(':')}:${lodBucket}`;
    if (boundsSignature !== this.drawBoundsSignature || this.membershipDirty) {
      this.drawBoundsSignature = boundsSignature;
      this.drawTileBounds = tileBounds;
      this.drawSignature = '';
      this.replaceDrawList(
        this.candidatesForTileBounds(tileBounds).filter(record =>
          this.shouldDraw(record, pixelsPerUnit)
        ).sort(compareSpriteRecords)
      );
      this.membershipDirty = false;
      return;
    }
    if (this.pendingMembership.size) {
      this.mergeMembershipChanges(pixelsPerUnit);
    }
    this.uploadDirtySlots();
  }

  /** Replaces the whole draw list: every slot repacked and uploaded. */
  private replaceDrawList(records: SpriteRecord[]): void {
    for (const record of this.drawList) record.slot = -1;
    this.drawList = records;
    this.pendingMembership.clear();
    this.ensureCapacity(Math.max(1, records.length));
    for (let index = 0; index < records.length; index++) {
      const record = records[index];
      record.slot = index;
      record.slotLayer = record.input.layer ?? 0;
      record.slotOrder = record.input.order ?? 0;
      this.packSprite(index, record.input);
    }
    this.instanceBuffer!.update(this.packed);
    this.dirtySlotStart = Number.POSITIVE_INFINITY;
    this.dirtySlotEnd = -1;
    this.finishDrawList();
    this.drawListRebuilds++;
  }

  /**
   * Applies queued arrivals, departures and re-sorts to the existing list: one
   * compacting pass, a merge of the sorted arrivals, and a repack from the first
   * slot that moved. A change touching much of the list rebuilds it instead.
   */
  private mergeMembershipChanges(pixelsPerUnit: number): void {
    const pending = this.pendingMembership;
    if (pending.size > 64 + this.drawList.length / 4) {
      this.membershipDirty = true;
      pending.clear();
      this.drawSignature = '';
      this.rebuildVisibleDrawList();
      return;
    }
    const arrivals: SpriteRecord[] = [];
    let firstChanged = this.drawList.length;
    for (const record of pending) {
      const draw = this.shouldDraw(record, pixelsPerUnit);
      const resorted =
        record.slot >= 0 &&
        (record.slotLayer !== (record.input.layer ?? 0) ||
          record.slotOrder !== (record.input.order ?? 0));
      if (record.slot >= 0 && (!draw || resorted)) {
        firstChanged = Math.min(firstChanged, record.slot);
        this.drawList[record.slot] = undefined as unknown as SpriteRecord;
        record.slot = -1;
      }
      if (draw && record.slot < 0) arrivals.push(record);
    }
    pending.clear();
    if (firstChanged === this.drawList.length && !arrivals.length) return;

    let write = firstChanged;
    for (let read = firstChanged; read < this.drawList.length; read++) {
      const record = this.drawList[read];
      if (record) this.drawList[write++] = record;
    }
    this.drawList.length = write;
    if (arrivals.length) {
      arrivals.sort(compareSpriteRecords);
      const firstArrivalSlot = lowerBound(this.drawList, arrivals[0]);
      firstChanged = Math.min(firstChanged, firstArrivalSlot);
      const tail = this.drawList.splice(firstArrivalSlot);
      let a = 0;
      let t = 0;
      while (a < arrivals.length || t < tail.length) {
        if (t >= tail.length || (a < arrivals.length && compareSpriteRecords(arrivals[a], tail[t]) < 0)) {
          this.drawList.push(arrivals[a++]);
        } else {
          this.drawList.push(tail[t++]);
        }
      }
    }
    const grew = this.drawList.length > this.capacity;
    this.ensureCapacity(Math.max(1, this.drawList.length));
    for (let index = firstChanged; index < this.drawList.length; index++) {
      const record = this.drawList[index];
      record.slot = index;
      record.slotLayer = record.input.layer ?? 0;
      record.slotOrder = record.input.order ?? 0;
      this.packSprite(index, record.input);
    }
    if (grew) {
      this.instanceBuffer!.update(this.packed);
      this.dirtySlotStart = Number.POSITIVE_INFINITY;
      this.dirtySlotEnd = -1;
    } else if (this.drawList.length > firstChanged) {
      this.markSlotDirty(firstChanged);
      this.markSlotDirty(this.drawList.length - 1);
    }
    this.finishDrawList();
  }

  private uploadDirtySlots(): void {
    if (this.dirtySlotEnd <= this.dirtySlotStart) return;
    const start = this.dirtySlotStart;
    const end = Math.min(this.dirtySlotEnd, this.drawList.length);
    this.dirtySlotStart = Number.POSITIVE_INFINITY;
    this.dirtySlotEnd = -1;
    if (end <= start) return;
    this.instanceBuffer!.updateDirectly(
      this.packed.subarray(start * FLOATS_PER_SPRITE, end * FLOATS_PER_SPRITE),
      start * FLOATS_PER_SPRITE,
      end - start
    );
  }

  private finishDrawList(): void {
    this.mesh.forcedInstanceCount = this.gpuCullingActive
      ? Math.max(1, this.gpuIndexIds.length)
      : this.drawList.length;
    this.mesh.isVisible = this.drawList.length > 0;
  }

  private sortedDrawableRecords(): SpriteRecord[] {
    return Array.from(this.records.values())
      .filter(record => record.input.visible !== false)
      .sort(compareSpriteRecords);
  }

  private packSprite(index: number, sprite: ShadoSprite2DInput): void {
    this.packSpriteInto(this.packed, index, sprite);
  }

  private packSpriteInto(target: Float32Array, index: number, sprite: ShadoSprite2DInput): void {
    const entry = this.atlas.get(sprite.textureKey ?? 'default');
    const offset = index * FLOATS_PER_SPRITE;
    target[offset] = sprite.position[0];
    target[offset + 1] = sprite.position[1];
    target[offset + 2] = sprite.size[0];
    target[offset + 3] = sprite.size[1];
    target[offset + 4] = entry.rect.u0;
    target[offset + 5] = entry.rect.v0;
    target[offset + 6] = entry.rect.u1;
    target[offset + 7] = entry.rect.v1;
    target[offset + 8] = sprite.rotationRad ?? 0;
    target[offset + 9] = entry.layer;
    // Pack logical layer and opacity into one float lane. Layer remains exact
    // for practical 2D ranges and opacity uses the fractional half-unit.
    target[offset + 10] = Math.max(0, sprite.layer ?? 0) + (sprite.opacity ?? 1) * 0.5;
    const flags = (sprite.selected ? 2 : 0) | (sprite.highlighted ? 4 : 0);
    // Preserve the compact 48-byte record. Zero means "use the renderer
    // default"; explicit thresholds are quantized to 1/16 pixel, leaving the
    // low three integer bits for the existing render flags.
    const lodCode =
      sprite.minPixelSize === undefined
        ? 0
        : Math.min(0x1fffff, Math.round(Math.max(0, sprite.minPixelSize) * 16) + 1);
    target[offset + 11] = lodCode * 8 + flags;
  }

  private bindGpuVisibility(): void {
    if (!this.gpuVisibility) return;
    this.material.setStorageBuffer('uSpriteRecords', this.gpuVisibility.records);
    this.material.setStorageBuffer('uVisibleIndices', this.gpuVisibility.visibleIndices);
    this.material.setStorageBuffer('uDrawArgs', this.gpuVisibility.drawArgs);
  }

  private restoreCpuAuthority(): void {
    if (this.gpuMotionActive) this.membershipDirty = true;
    this.gpuMotionActive = false;
    this.gpuCullingActive = false;
    this.detachIndirectDraw();
    this.gpuGeneration++;
  }

  /** Babylon 9 has a WebGPU indirect draw context, but not a public mesh-level setter yet. */
  private tryAttachIndirectDraw(): void {
    if (!this.gpuVisibility || this.indirectDrawContext) return;
    try {
      if (!this.material.isReady(this.mesh, true)) return;
      const context = (this.material as any)._drawWrapper?.drawContext;
      if (!context || !('enableIndirectDraw' in context)) return;
      context.enableIndirectDraw = true;
      this.originalIndirectBuffer = context.indirectDrawBuffer;
      context.indirectDrawBuffer = this.gpuVisibility.drawArgsResource;
      // Prevent Babylon's normal instanced draw setup from overwriting the
      // compute-authored instance count before drawIndexedIndirect executes.
      context._currentInstanceCount = Math.max(1, this.gpuIndexIds.length);
      this.indirectDrawContext = context;
    } catch {
      // The vertex shader still rejects instances beyond the compact count.
      // This preserves the optimization's fragment/overdraw win and is the
      // complete fallback for Babylon versions without the draw-context hook.
      this.detachIndirectDraw();
    }
  }

  private detachIndirectDraw(): void {
    const context = this.indirectDrawContext;
    if (!context) return;
    context.indirectDrawBuffer = this.originalIndirectBuffer;
    context.enableIndirectDraw = false;
    this.indirectDrawContext = undefined;
    this.originalIndirectBuffer = undefined;
  }

  private ensureCapacity(required: number): void {
    if (required <= this.capacity) return;
    let capacity = Math.max(16, this.capacity);
    while (capacity < required) capacity *= 2;
    for (const kind of ['iTransform', 'iUvRect', 'iState']) {
      if (this.mesh.isVerticesDataPresent(kind)) this.mesh.removeVerticesData(kind);
    }
    this.instanceBuffer?.dispose();
    this.capacity = capacity;
    const previous = this.packed;
    this.packed = new Float32Array(capacity * FLOATS_PER_SPRITE);
    this.packed.set(previous.subarray(0, Math.min(previous.length, this.packed.length)));
    this.instanceBuffer = new BABYLON.Buffer(
      this.engine,
      this.packed,
      true,
      FLOATS_PER_SPRITE,
      false,
      true,
      false,
      1,
      'ShadoSprite2D compact records'
    );
    this.mesh.setVerticesBuffer(
      this.instanceBuffer.createVertexBuffer('iTransform', 0, 4, FLOATS_PER_SPRITE, true)
    );
    this.mesh.setVerticesBuffer(
      this.instanceBuffer.createVertexBuffer('iUvRect', 4, 4, FLOATS_PER_SPRITE, true)
    );
    this.mesh.setVerticesBuffer(
      this.instanceBuffer.createVertexBuffer('iState', 8, 4, FLOATS_PER_SPRITE, true)
    );
  }

  private candidatesForBounds(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): SpriteRecord[] {
    return this.candidatesForTileBounds(this.tileBounds(minX, minY, maxX, maxY));
  }

  private candidatesForTileBounds(bounds: readonly number[]): SpriteRecord[] {
    // Each record lives in exactly one tile, so no de-duplication is needed.
    const records: SpriteRecord[] = [];
    const columns = bounds[2] - bounds[0] + 1;
    const rows = bounds[3] - bounds[1] + 1;
    // At extreme zoom-out, walking every coordinate in a mostly empty tile
    // rectangle can dwarf the actual scene. Flip the lookup around and scan
    // the populated cache instead, keeping zoom range independent of density.
    if (!Number.isFinite(columns * rows) || columns * rows > this.tileCache.size * 4) {
      for (const tile of this.tileCache.values()) {
        const first = tile[0];
        if (
          !first ||
          first.tileX < bounds[0] ||
          first.tileX > bounds[2] ||
          first.tileY < bounds[1] ||
          first.tileY > bounds[3]
        )
          continue;
        for (const record of tile) records.push(record);
      }
      return records;
    }
    for (let y = bounds[1]; y <= bounds[3]; y++) {
      for (let x = bounds[0]; x <= bounds[2]; x++) {
        const tile = this.tileCache.get(`${x}:${y}`);
        if (tile) for (const record of tile) records.push(record);
      }
    }
    return records;
  }

  private tileBounds(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): [number, number, number, number] {
    return [
      Math.floor(minX / this.tileSize),
      Math.floor(minY / this.tileSize),
      Math.floor(maxX / this.tileSize),
      Math.floor(maxY / this.tileSize),
    ];
  }

  private tileKey(x: number, y: number): string {
    return `${Math.floor(x / this.tileSize)}:${Math.floor(y / this.tileSize)}`;
  }

  private createQuad(): Mesh {
    const mesh = new BABYLON.Mesh('shado-sprite-2d-optimized', this.scene);
    const data = new BABYLON.VertexData();
    data.positions = [-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0];
    data.uvs = [0, 1, 1, 1, 1, 0, 0, 0];
    data.indices = [0, 1, 2, 0, 2, 3];
    data.applyToMesh(mesh);
    return mesh;
  }

  private createMaterial(): ShaderMaterial {
    const webgpu = this.engine.isWebGPU;
    const shaderName = `${webgpu ? 'shadoSprite2DOptimizedWGSL' : 'shadoSprite2DOptimizedGLSL'}_${this.alphaMode}`;
    installSpriteShaders(shaderName, webgpu, this.alphaMode);
    const material = new BABYLON.ShaderMaterial(
      'shadoSprite2DOptimizedMaterial',
      this.scene,
      shaderName,
      {
        attributes: ['position', 'uv', 'iTransform', 'iUvRect', 'iState'],
        uniforms: [
          'uCameraCenter',
          'uCameraHalfExtent',
          'uInstanceCount',
          'uAlphaCutoff',
          'uUseGpuMotion',
          'uUseGpuCulling',
          'uViewportHeight',
          'uMinimumPixelSize',
        ],
        samplers: ['uAtlas'],
        storageBuffers: webgpu
          ? ['uMotionState', 'uSpriteRecords', 'uVisibleIndices', 'uDrawArgs']
          : [],
        uniformBuffers: ['Scene'],
        needAlphaBlending: this.alphaMode === 'premultiplied',
        shaderLanguage: webgpu ? BABYLON.ShaderLanguage.WGSL : BABYLON.ShaderLanguage.GLSL,
      }
    );
    material.backFaceCulling = false;
    material.forceDepthWrite = this.alphaMode === 'cutout';
    material.alphaMode =
      this.alphaMode === 'premultiplied'
        ? BABYLON.Engine.ALPHA_PREMULTIPLIED_PORTERDUFF
        : BABYLON.Engine.ALPHA_DISABLE;
    material.setTexture('uAtlas', this.atlas.texture);
    material.setFloat('uAlphaCutoff', this.alphaCutoff);
    material.setFloat('uUseGpuMotion', 0);
    material.setFloat('uUseGpuCulling', 0);
    material.setFloat('uViewportHeight', 1);
    material.setFloat('uMinimumPixelSize', this.defaultMinPixelSize);
    return material;
  }
}

function normalizeSprite(input: ShadoSprite2DInput): ShadoSprite2DInput {
  return {
    ...input,
    position: [input.position[0], input.position[1]],
    size: [Math.max(0.0001, input.size[0]), Math.max(0.0001, input.size[1])],
    rotationRad: input.rotationRad ?? ((input.rotationDeg ?? 0) * Math.PI) / 180,
    opacity: Math.max(0, Math.min(1, input.opacity ?? 1)),
    layer: Math.round(input.layer ?? 0),
    order: Math.round(input.order ?? 0),
  };
}

/** First index whose record sorts at or after `record`. */
function lowerBound(records: readonly SpriteRecord[], record: SpriteRecord): number {
  let low = 0;
  let high = records.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (compareSpriteRecords(records[middle], record) < 0) low = middle + 1;
    else high = middle;
  }
  return low;
}

function compareSpriteRecords(a: SpriteRecord, b: SpriteRecord): number {
  return (
    (a.input.layer ?? 0) - (b.input.layer ?? 0) ||
    (a.input.order ?? 0) - (b.input.order ?? 0) ||
    a.insertionOrder - b.insertionOrder
  );
}

function selectCpuAccessIds(
  access: ShadoSprite2DCpuAccess,
  allIds: readonly string[],
  visibleIds: readonly string[],
  isSelected: (id: string) => boolean
): string[] {
  switch (access.tier) {
    case 'all':
      return [...allIds];
    case 'visible':
      return [...visibleIds];
    case 'selected':
      return allIds.filter(isSelected);
    case 'range': {
      const start = Math.max(0, Math.min(allIds.length, Math.trunc(access.start)));
      const count = Math.max(0, Math.trunc(access.count));
      return allIds.slice(start, start + count);
    }
    case 'ids': {
      const valid = new Set(allIds);
      return Array.from(new Set(access.ids)).filter(id => valid.has(id));
    }
  }
}

function installSpriteShaders(
  name: string,
  webgpu: boolean,
  alphaMode: ShadoSpriteAlphaMode
): void {
  if (webgpu) {
    const store = BABYLON.ShaderStore.ShadersStoreWGSL;
    store[`${name}VertexShader`] ??= `
attribute position: vec3f;
attribute uv: vec2f;
attribute iTransform: vec4f;
attribute iUvRect: vec4f;
attribute iState: vec4f;
var<storage, read> uMotionState: array<vec4f>;
var<storage, read> uSpriteRecords: array<vec4f>;
var<storage, read> uVisibleIndices: array<u32>;
var<storage, read> uDrawArgs: array<u32>;
uniform uCameraCenter: vec2f;
uniform uCameraHalfExtent: vec2f;
uniform uInstanceCount: f32;
uniform uUseGpuMotion: f32;
uniform uUseGpuCulling: f32;
uniform uViewportHeight: f32;
uniform uMinimumPixelSize: f32;
varying vUV: vec2f;
varying vLayer: f32;
varying vOpacity: f32;
varying vFlags: f32;
@vertex
fn main(input: VertexInputs) -> FragmentInputs {
  let useGpuCulling = uniforms.uUseGpuCulling > 0.5;
  let visibleCount = select(u32(max(1.0, uniforms.uInstanceCount)), uDrawArgs[1], useGpuCulling);
  var sourceIndex = vertexInputs.instanceIndex;
  var transform = vertexInputs.iTransform;
  var uvRect = vertexInputs.iUvRect;
  var spriteState = vertexInputs.iState;
  if (useGpuCulling) {
    let compactIndex = min(vertexInputs.instanceIndex, max(1u, visibleCount) - 1u);
    sourceIndex = uVisibleIndices[compactIndex];
    transform = uSpriteRecords[sourceIndex * 3u];
    uvRect = uSpriteRecords[sourceIndex * 3u + 1u];
    spriteState = uSpriteRecords[sourceIndex * 3u + 2u];
  }
  let packedState = u32(spriteState.w + 0.5);
  let flags = packedState & 7u;
  let lodCode = packedState >> 3u;
  let minimumPixelSize = select(
    uniforms.uMinimumPixelSize,
    (f32(lodCode) - 1.0) / 16.0,
    lodCode > 0u
  );
  let c = cos(spriteState.x);
  let s = sin(spriteState.x);
  let local = vertexInputs.position.xy * transform.zw;
  let rotated = vec2f(local.x * c - local.y * s, local.x * s + local.y * c);
  let center = select(
    transform.xy,
    uMotionState[sourceIndex].xy,
    uniforms.uUseGpuMotion > 0.5
  );
  let world = center + rotated;
  let insideCompactDraw = !useGpuCulling || vertexInputs.instanceIndex < visibleCount;
  let pixelsPerUnit = uniforms.uViewportHeight / max(0.0002, uniforms.uCameraHalfExtent.y * 2.0);
  let passesLod = max(transform.z, transform.w) * pixelsPerUnit >= minimumPixelSize;
  let clip = select(vec2f(4.0, 4.0), (world - uniforms.uCameraCenter) / uniforms.uCameraHalfExtent, insideCompactDraw && passesLod);
  let logicalLayer = floor(spriteState.z);
  let depth = 0.0009 - min(logicalLayer, 255.0) * 0.000001 -
    (f32(vertexInputs.instanceIndex) / max(1.0, uniforms.uInstanceCount)) * 0.0000005;
  vertexOutputs.position = vec4f(clip, depth, 1.0);
  vertexOutputs.vUV = vec2f(
    mix(uvRect.x, uvRect.z, vertexInputs.uv.x),
    mix(uvRect.y, uvRect.w, vertexInputs.uv.y)
  );
  vertexOutputs.vLayer = spriteState.y;
  vertexOutputs.vOpacity = fract(spriteState.z) * 2.0;
  vertexOutputs.vFlags = f32(flags);
}`;
    store[`${name}FragmentShader`] ??= `
varying vUV: vec2f;
varying vLayer: f32;
varying vOpacity: f32;
varying vFlags: f32;
uniform uAlphaCutoff: f32;
var uAtlasSampler: sampler;
var uAtlas: texture_2d_array<f32>;
@fragment
fn main(input: FragmentInputs) -> FragmentOutputs {
  var color = textureSampleLevel(uAtlas, uAtlasSampler, fragmentInputs.vUV, i32(fragmentInputs.vLayer + 0.5), 0.0);
  color.a *= fragmentInputs.vOpacity;
  if (color.a < uniforms.uAlphaCutoff) { discard; }
  let flags = i32(fragmentInputs.vFlags + 0.5);
  color = vec4f(mix(mix(color.rgb, vec3f(0.78, 0.96, 1.0), f32((flags >> 1) & 1) * 0.55), vec3f(1.0, 0.72, 0.12), f32((flags >> 2) & 1) * 0.48), color.a);
  ${alphaMode === 'premultiplied' ? 'color = vec4f(color.rgb * color.a, color.a);' : ''}
  fragmentOutputs.color = color;
}`;
    return;
  }

  const store = BABYLON.Effect.ShadersStore;
  store[`${name}VertexShader`] ??= `
precision highp float;
attribute vec3 position;
attribute vec2 uv;
attribute vec4 iTransform;
attribute vec4 iUvRect;
attribute vec4 iState;
uniform vec2 uCameraCenter;
uniform vec2 uCameraHalfExtent;
uniform float uInstanceCount;
varying vec2 vUV;
varying float vLayer;
varying float vOpacity;
varying float vFlags;
void main(void) {
  float packedState = floor(iState.w + 0.5);
  float c = cos(iState.x);
  float s = sin(iState.x);
  vec2 local = position.xy * iTransform.zw;
  vec2 rotated = vec2(local.x * c - local.y * s, local.x * s + local.y * c);
  vec2 clip = (iTransform.xy + rotated - uCameraCenter) / uCameraHalfExtent;
  float logicalLayer = floor(iState.z);
  float depth = -0.999 - min(logicalLayer, 255.0) * 0.000001 -
    (float(gl_InstanceID) / max(1.0, uInstanceCount)) * 0.0000005;
  gl_Position = vec4(clip, depth, 1.0);
  vUV = vec2(mix(iUvRect.x, iUvRect.z, uv.x), mix(iUvRect.y, iUvRect.w, uv.y));
  vLayer = iState.y;
  vOpacity = fract(iState.z) * 2.0;
  vFlags = mod(packedState, 8.0);
}`;
  store[`${name}FragmentShader`] ??= `
precision highp float;
varying vec2 vUV;
varying float vLayer;
varying float vOpacity;
varying float vFlags;
uniform float uAlphaCutoff;
uniform highp sampler2DArray uAtlas;
void main(void) {
  vec4 color = textureLod(uAtlas, vec3(vUV, floor(vLayer + 0.5)), 0.0);
  color.a *= vOpacity;
  if (color.a < uAlphaCutoff) discard;
  float selected = mod(floor(vFlags / 2.0), 2.0);
  float highlighted = mod(floor(vFlags / 4.0), 2.0);
  color.rgb = mix(mix(color.rgb, vec3(0.78, 0.96, 1.0), selected * 0.55), vec3(1.0, 0.72, 0.12), highlighted * 0.48);
  ${alphaMode === 'premultiplied' ? 'color.rgb *= color.a;' : ''}
  gl_FragColor = color;
}`;
}
