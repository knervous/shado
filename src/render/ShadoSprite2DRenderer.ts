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
  input: ShadoSprite2DInput;
  insertionOrder: number;
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
  private readonly tileCache = new Map<string, string[]>();
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
  private visibleIds: string[] = [];
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
      this.material.setFloat('uInstanceCount', Math.max(1, this.visibleIds.length));
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
    const current = this.records.get(input.id);
    this.records.set(input.id, {
      input: normalizeSprite(input),
      insertionOrder: current?.insertionOrder ?? this.insertionCounter++,
    });
    this.rebuildTileCache();
  }

  public upsertMany(inputs: readonly ShadoSprite2DInput[]): void {
    for (const input of inputs) {
      const current = this.records.get(input.id);
      this.records.set(input.id, {
        input: normalizeSprite(input),
        insertionOrder: current?.insertionOrder ?? this.insertionCounter++,
      });
    }
    this.rebuildTileCache();
  }

  public remove(id: string): boolean {
    if (!this.records.delete(id)) return false;
    this.rebuildTileCache();
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
    this.drawSignature = '';
  }

  public setVisible(id: string, visible: boolean): boolean {
    const record = this.records.get(id);
    if (!record || record.input.visible === visible) return false;
    record.input = { ...record.input, visible };
    this.restoreCpuAuthority();
    this.revision++;
    this.drawSignature = '';
    return true;
  }

  public setPosition(id: string, position: readonly [number, number]): boolean {
    return this.setPositions([{ id, position }]) > 0;
  }

  /** Apply a simulation tick with one tile-cache revision and one GPU repack. */
  public setPositions(updates: readonly ShadoSprite2DPositionUpdate[]): number {
    const crossed: Array<{ id: string; previousKey: string; nextKey: string }> = [];
    let changed = 0;
    for (const update of updates) {
      const record = this.records.get(update.id);
      if (!record) continue;
      const previous = record.input.position;
      if (previous[0] === update.position[0] && previous[1] === update.position[1]) continue;
      const previousKey = this.tileKey(previous[0], previous[1]);
      const nextKey = this.tileKey(update.position[0], update.position[1]);
      record.input = { ...record.input, position: [update.position[0], update.position[1]] };
      if (previousKey !== nextKey) crossed.push({ id: update.id, previousKey, nextKey });
      changed++;
    }
    if (!changed) return 0;

    // Sparse crossings are cheaper to patch. A broad simulation step is
    // cheaper to rebuild linearly than repeatedly splice large dense tiles.
    if (crossed.length > Math.min(1_024, Math.max(16, this.records.size / 50))) {
      this.rebuildTileCache();
      return changed;
    }
    this.restoreCpuAuthority();
    const affected = new Set<string>();
    for (const move of crossed) {
      const previous = this.tileCache.get(move.previousKey);
      const index = previous?.indexOf(move.id) ?? -1;
      if (previous && index >= 0) {
        previous[index] = previous[previous.length - 1];
        previous.pop();
        if (!previous.length) this.tileCache.delete(move.previousKey);
      }
      let next = this.tileCache.get(move.nextKey);
      if (!next) this.tileCache.set(move.nextKey, (next = []));
      next.push(move.id);
      affected.add(move.previousKey);
      affected.add(move.nextKey);
    }
    for (const key of affected) {
      this.tileCache
        .get(key)
        ?.sort((a, b) => compareSpriteRecords(this.records.get(a)!, this.records.get(b)!));
    }
    this.revision++;
    this.drawSignature = '';
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
        this.visibleIds,
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
      this.visibleIds,
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
    )
      .map(id => this.records.get(id))
      .filter((value): value is SpriteRecord => !!value);
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
      visible: this.visibleIds.length,
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
    for (const [id, record] of this.records) {
      const sprite = record.input;
      this.maxHalfExtent = Math.max(
        this.maxHalfExtent,
        Math.hypot(sprite.size[0] * 0.5, sprite.size[1] * 0.5)
      );
      const key = this.tileKey(sprite.position[0], sprite.position[1]);
      let tile = this.tileCache.get(key);
      if (!tile) this.tileCache.set(key, (tile = []));
      tile.push(id);
    }
    for (const ids of this.tileCache.values()) {
      ids.sort((a, b) => compareSpriteRecords(this.records.get(a)!, this.records.get(b)!));
    }
    this.revision++;
    this.drawSignature = '';
  }

  private rebuildVisibleDrawList(): void {
    const minX = this.view.center[0] - this.view.halfExtent[0] - this.maxHalfExtent;
    const maxX = this.view.center[0] + this.view.halfExtent[0] + this.maxHalfExtent;
    const minY = this.view.center[1] - this.view.halfExtent[1] - this.maxHalfExtent;
    const maxY = this.view.center[1] + this.view.halfExtent[1] + this.maxHalfExtent;
    const pixelsPerUnit = this.view.viewportPixels[1] / (this.view.halfExtent[1] * 2);
    const lodBucket = Math.round(Math.log2(Math.max(0.0001, pixelsPerUnit)) * 8);
    const tileBounds = this.tileBounds(minX, minY, maxX, maxY);
    const signature = this.gpuMotionActive
      ? `gpu:${this.revision}`
      : `${tileBounds.join(':')}:${lodBucket}:${this.revision}`;
    if (signature === this.drawSignature) return;
    this.drawSignature = signature;

    const visible = (
      this.gpuMotionActive
        ? this.sortedDrawableRecords()
        : this.candidatesForTileBounds(tileBounds).map(id => this.records.get(id))
    ).filter((record): record is SpriteRecord => {
      if (!record || record.input.visible === false) return false;
      const sprite = record.input;
      if (
        !this.gpuMotionActive &&
        Math.max(sprite.size[0], sprite.size[1]) * pixelsPerUnit <
          (sprite.minPixelSize ?? this.defaultMinPixelSize)
      )
        return false;
      // Keep every sprite in the intersecting cached tiles. This deliberate
      // one-tile overdraw lets sub-cell camera pans reuse the exact GPU list;
      // visibility is rebuilt only when a tile boundary or LOD bucket changes.
      return true;
    });
    visible.sort(compareSpriteRecords);
    this.visibleIds = visible.map(record => record.input.id);
    this.ensureCapacity(Math.max(1, visible.length));
    this.packed.fill(0);
    for (let index = 0; index < visible.length; index++) {
      this.packSprite(index, visible[index].input);
    }
    this.instanceBuffer!.update(this.packed);
    this.mesh.forcedInstanceCount = this.gpuCullingActive
      ? Math.max(1, this.gpuIndexIds.length)
      : visible.length;
    this.mesh.isVisible = visible.length > 0;
    this.drawListRebuilds++;
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
    this.packed = new Float32Array(capacity * FLOATS_PER_SPRITE);
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

  private candidatesForBounds(minX: number, minY: number, maxX: number, maxY: number): string[] {
    return this.candidatesForTileBounds(this.tileBounds(minX, minY, maxX, maxY));
  }

  private candidatesForTileBounds(bounds: readonly number[]): string[] {
    const ids: string[] = [];
    const seen = new Set<string>();
    const columns = bounds[2] - bounds[0] + 1;
    const rows = bounds[3] - bounds[1] + 1;
    // At extreme zoom-out, walking every coordinate in a mostly empty tile
    // rectangle can dwarf the actual scene. Flip the lookup around and scan
    // the populated cache instead, keeping zoom range independent of density.
    if (!Number.isFinite(columns * rows) || columns * rows > this.tileCache.size * 4) {
      for (const [key, tile] of this.tileCache) {
        const separator = key.indexOf(':');
        const x = Number(key.slice(0, separator));
        const y = Number(key.slice(separator + 1));
        if (x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3]) continue;
        for (const id of tile) {
          if (!seen.has(id)) {
            seen.add(id);
            ids.push(id);
          }
        }
      }
      return ids;
    }
    for (let y = bounds[1]; y <= bounds[3]; y++) {
      for (let x = bounds[0]; x <= bounds[2]; x++) {
        for (const id of this.tileCache.get(`${x}:${y}`) ?? []) {
          if (!seen.has(id)) {
            seen.add(id);
            ids.push(id);
          }
        }
      }
    }
    return ids;
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
