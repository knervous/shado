import { BABYLON, type AbstractEngine, type Mesh, type Scene } from '../babylon';
import { Shado } from '../core/Shado';
import { type PendingField, type ShadoConfig } from '../decorators';
import { NameplateData, createMSDFNameplateLayer, type MSDFNameplateFontAsset } from '../msdf';
import type { InitializeConfig } from '../types';

export type ShadoDynamicEntityNameplateInput = {
  id: string;
  text: string;
  x: number;
  y: number;
  z?: number;
  visible?: boolean;
  fontSize?: number;
  color?: string;
  backgroundColor?: string;
  billboard?: boolean;
};

export type ShadoDynamicEntityNameplateLayerOptions = {
  enabled?: boolean;
  fontAsset?: MSDFNameplateFontAsset;
  fontJsonUrl?: string;
  fontTextureUrl?: string;
  fontSize?: number;
  color?: string;
  backgroundColor?: string;
  padding?: number;
  worldScale?: number;
  zOffset?: number;
  nameLiftWorld?: number;
  renderingGroupId?: number;
  depthTest?: boolean;
  thickness?: number;
  debug?: boolean;
};

type DynamicNameplateRecord = {
  id: string;
  text: string;
  actor: ShadoDynamicNameplateActor;
  /** The `updateActors` pass that last found an input for this record. */
  seen?: number;
};

const DEFAULT_FONT_JSON_URL = 'https://assets.babylonjs.com/fonts/roboto-regular.json';
const DEFAULT_FONT_TEXTURE_URL = 'https://assets.babylonjs.com/fonts/roboto-regular.png';
const DEFAULT_FONT_SIZE = 13;
const DEFAULT_COLOR = '#eef6ff';
const DEFAULT_WORLD_SCALE = 1 / 36;
const DEFAULT_Z_OFFSET = 0.35;
const DEFAULT_NAME_LIFT_WORLD = -0.65;

class ShadoDynamicNameplateActor extends Shado {
  static readonly shadoConfig: ShadoConfig = { name: 'ShadoDynamicNameplateActor', useWasm: false };
  static readonly shadoFields: readonly PendingField[] = [
    { name: 'translation', type: 'vec4' },
    { name: 'nameIndex', type: 'u32' },
    { name: 'nameWorldPerEM', type: 'f32' },
    { name: 'nameLiftWorld', type: 'f32' },
    { name: 'nameplateColor', type: 'vec4' },
    { name: 'visibleFlag', type: 'i32' },
    { name: 'billboardFlag', type: 'f32' },
    { name: 'padding1', type: 'f32' },
    { name: 'padding2', type: 'f32' },
  ];
  translation!: Float32Array;
  nameIndex!: number;
  nameWorldPerEM!: number;
  nameLiftWorld!: number;
  nameplateColor!: Float32Array;
  visibleFlag!: number;
  billboardFlag!: number;
  padding1!: number;
  padding2!: number;

  public constructor(engine: AbstractEngine) {
    super(engine, true);
  }

  public initialize(): void {
    this.translation = new Float32Array([0, 0, 0, 1]);
    this.nameIndex = 0;
    this.nameWorldPerEM = DEFAULT_FONT_SIZE * DEFAULT_WORLD_SCALE;
    this.nameLiftWorld = DEFAULT_NAME_LIFT_WORLD;
    this.nameplateColor = new Float32Array([1, 1, 1, 1]);
    this.visibleFlag = 1;
    this.billboardFlag = 1;
    this.padding1 = 0;
    this.padding2 = 0;
  }
}

class ShadoDynamicNameplateContainer extends Shado {
  static readonly shadoConfig: ShadoConfig = { name: 'ShadoDynamicNameplateContainer', useWasm: false };
  static readonly shadoFields: readonly PendingField[] = [
    { name: 'visibleCount', type: 'u32' },
    { name: 'instancesCount', type: 'u32' },
  ];
  visibleCount!: number;
  instancesCount!: number;

  private readonly records: DynamicNameplateRecord[] = [];

  public static override async initialize(
    engine: unknown,
    config: InitializeConfig = {}
  ): Promise<boolean> {
    const additionalFields: PendingField[] = [
      ...(config.additionalFields ?? []),
      { name: 'instances', type: { arrayOf: { structOf: ShadoDynamicNameplateActor } } },
    ];
    delete (this as any).__cachedSchema;
    return super.initialize(engine, {
      backend: 'datatex',
      wasm: false,
      ...config,
      additionalFields,
    });
  }

  public constructor(engine: AbstractEngine) {
    super(engine);
    this.visibleCount = 0;
    this.instancesCount = 0;
  }

  public get children(): ShadoDynamicNameplateActor[] {
    return this.records.map(record => record.actor);
  }

  public get instanceCount(): number {
    return this.records.length;
  }

  public addNameplate(
    id: string,
    text: string,
    nameplates: NameplateData
  ): ShadoDynamicNameplateActor {
    const actor = this.addStructToArray<ShadoDynamicNameplateActor>('instances');
    actor.initialize();
    actor.nameIndex = nameplates.addName(text);
    actor.emitHeaderDirty();
    this.records.push({ id, text, actor });
    this.instancesCount = this.records.length;
    this.visibleCount = this.records.length;
    return actor;
  }
}

const initByEngine = new WeakMap<AbstractEngine, Promise<void>>();

const ensureDynamicNameplateShado = (engine: AbstractEngine): Promise<void> => {
  let pending = initByEngine.get(engine);
  if (!pending) {
    pending = (async () => {
      const backend = engine.isWebGPU ? 'storage' : 'datatex';
      await ShadoDynamicNameplateActor.initialize(engine, { backend, wasm: false });
      await ShadoDynamicNameplateContainer.initialize(engine, { backend, wasm: false });
      await NameplateData.initialize(engine, { backend, wasm: false });
    })();
    initByEngine.set(engine, pending);
  }
  return pending;
};

const loadDefaultFontAsset = async (
  scene: Scene,
  fontJsonUrl: string,
  fontTextureUrl: string
): Promise<MSDFNameplateFontAsset> => {
  const response = await fetch(fontJsonUrl);
  if (!response.ok) {
    throw new Error(`Failed to load MSDF font json: ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  const font = JSON.parse(text);
  const texture = new BABYLON.Texture(
    fontTextureUrl,
    scene,
    true,
    false,
    BABYLON.Texture.TRILINEAR_SAMPLINGMODE
  );
  const chars = new Map<number, { xadvance?: number }>();
  for (const char of Array.isArray(font.chars) ? font.chars : []) {
    if (typeof char?.id === 'number') {
      chars.set(char.id, char);
    }
  }
  const kerning = new Map<string, number>();
  for (const item of Array.isArray(font.kernings) ? font.kernings : []) {
    if (
      typeof item?.first === 'number' &&
      typeof item?.second === 'number' &&
      typeof item?.amount === 'number'
    ) {
      kerning.set(`${item.first}:${item.second}`, item.amount);
    }
  }
  return {
    textures: [texture],
    _font: font,
    _getChar: (code: number) => chars.get(code),
    _getKerning: (left: number, right: number) => kerning.get(`${left}:${right}`) ?? 0,
  } as MSDFNameplateFontAsset;
};

const rgbaFromColor = (value: string | undefined): [number, number, number, number] => {
  const fallback: [number, number, number, number] = [0.933, 0.965, 1, 1];
  if (!value) {
    return fallback;
  }
  const hex = value.trim();
  const match = /^#?([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(hex);
  if (!match) {
    return fallback;
  }
  const rgb = match[1];
  const alpha = match[2] ?? 'ff';
  return [
    Number.parseInt(rgb.slice(0, 2), 16) / 255,
    Number.parseInt(rgb.slice(2, 4), 16) / 255,
    Number.parseInt(rgb.slice(4, 6), 16) / 255,
    Number.parseInt(alpha, 16) / 255,
  ];
};

/**
 * Parsed colours, by string. A caller that fades plates by distance sends a
 * few hundred distinct strings, each every frame; parsing each one with a regex
 * every frame was garbage the size of the crowd. Bounded: a fade has at most
 * 256 alphas per hue, and the cache is dropped wholesale if it ever grows past
 * that many hues' worth.
 */
const COLOR_CACHE_LIMIT = 8192;
const colorCache = new Map<string, readonly [number, number, number, number]>();
const cachedRgba = (value: string): readonly [number, number, number, number] => {
  let rgba = colorCache.get(value);
  if (!rgba) {
    if (colorCache.size >= COLOR_CACHE_LIMIT) colorCache.clear();
    rgba = rgbaFromColor(value);
    colorCache.set(value, rgba);
  }
  return rgba;
};

export class ShadoDynamicEntityNameplateLayer {
  private readonly scene: Scene;
  private readonly engine: AbstractEngine;
  private enabled: boolean;
  private readyPromise: Promise<void> | null = null;
  /**
   * Set once the font and the Shado streams exist, so a later `sync` can write
   * straight through instead of waiting a microtask. See `sync`.
   */
  private ready = false;
  private fontAsset: MSDFNameplateFontAsset | null = null;
  private ownsFontAsset = false;
  private container: ShadoDynamicNameplateContainer | null = null;
  private nameplates: NameplateData | null = null;
  private mesh: Mesh | null = null;
  private records = new Map<string, DynamicNameplateRecord>();
  /** Counts `updateActors` passes; see `DynamicNameplateRecord.seen`. */
  private updatePass = 0;
  /** Whether a layout exists at all; an empty one is still one. */
  private built = false;
  private latestInputs: readonly ShadoDynamicEntityNameplateInput[] = [];
  private disposed = false;

  public constructor(
    scene: Scene,
    private options: ShadoDynamicEntityNameplateLayerOptions = {}
  ) {
    this.scene = scene;
    this.engine = scene.getEngine();
    this.enabled = options.enabled !== false;
  }

  public setOptions(options: ShadoDynamicEntityNameplateLayerOptions): void {
    const previousFontAsset = this.options.fontAsset;
    this.options = { ...this.options, ...options };
    this.enabled = this.options.enabled !== false;
    if (options.fontAsset && options.fontAsset !== previousFontAsset) {
      this.fontAsset = options.fontAsset;
      this.ownsFontAsset = false;
      this.rebuild(this.latestInputs);
    }
    this.mesh?.setEnabled(this.enabled);
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.mesh?.setEnabled(enabled);
  }

  /**
   * Places the labels for this frame.
   *
   * Synchronous once the layer is ready, which matters more than it looks.
   * Deferring the write through `.then()` -- even on an already-resolved
   * promise -- lands it in a microtask, and the microtask queue is not drained
   * until the task that called `render()` finishes. Every plate was therefore
   * written *after* the frame that was supposed to show it and drawn one frame
   * behind the body it names. Along the view axis, while running forward, that
   * is invisible; strafing makes it entirely lateral and the name slides off
   * its owner until they stop.
   *
   * The first call still has to wait for the font, which is the one case where
   * there is nothing to draw yet anyway.
   */
  public sync(inputs: readonly ShadoDynamicEntityNameplateInput[]): void {
    this.latestInputs = inputs;
    if (this.disposed) {
      return;
    }
    if (this.ready) {
      this.applySync(inputs);
      return;
    }
    void this.ensureReady()
      .then(() => {
        if (!this.disposed) {
          this.applySync(this.latestInputs);
        }
      })
      .catch(error => {
        if (this.options.debug) {
          // eslint-disable-next-line no-console
          console.warn('[shado/render] MSDF nameplates unavailable', error);
        }
      });
  }

  public dispose(): void {
    this.disposed = true;
    this.ready = false;
    this.mesh?.dispose(false, false);
    this.container?.dispose();
    this.nameplates?.dispose();
    if (this.ownsFontAsset) {
      for (const texture of this.fontAsset?.textures ?? []) {
        texture.dispose();
      }
    }
    this.mesh = null;
    this.container = null;
    this.nameplates = null;
    this.fontAsset = null;
    this.records.clear();
  }

  private async ensureReady(): Promise<void> {
    this.readyPromise ??= (async () => {
      await ensureDynamicNameplateShado(this.engine);
      if (this.options.fontAsset) {
        this.fontAsset = this.options.fontAsset;
        this.ownsFontAsset = false;
      } else if (!this.fontAsset) {
        this.fontAsset = await loadDefaultFontAsset(
          this.scene,
          this.options.fontJsonUrl ?? DEFAULT_FONT_JSON_URL,
          this.options.fontTextureUrl ?? DEFAULT_FONT_TEXTURE_URL
        );
        this.ownsFontAsset = true;
      }
      this.ready = true;
    })();
    await this.readyPromise;
  }

  private applySync(inputs: readonly ShadoDynamicEntityNameplateInput[]): void {
    // Rebuild only when something new has to be drawn: an id this layer has
    // never laid out, or an id whose text changed. An entry that has gone is
    // simply hidden by `updateActors` until the next rebuild sweeps it up, so
    // a combat number expiring costs nothing, and the check itself is a map
    // lookup per input rather than a sorted, joined string of every plate.
    if (this.needsRebuild(inputs)) {
      this.rebuild(inputs);
      this.built = true;
    }
    this.updateActors(inputs);
    this.mesh?.setEnabled(this.enabled && this.records.size > 0);
  }

  private rebuild(inputs: readonly ShadoDynamicEntityNameplateInput[]): void {
    if (!this.fontAsset || this.disposed) {
      return;
    }
    this.mesh?.dispose(false, false);
    this.container?.dispose();
    this.nameplates?.dispose();

    const filtered = inputs
      .filter(input => input.text.trim())
      .sort((a, b) => a.id.localeCompare(b.id));

    const nameplates = new NameplateData(this.engine, this.fontAsset);
    const container = new ShadoDynamicNameplateContainer(this.engine);
    const records = new Map<string, DynamicNameplateRecord>();
    for (const input of filtered) {
      const text = input.text.trim();
      const actor = container.addNameplate(input.id, text, nameplates);
      records.set(input.id, { id: input.id, text, actor });
    }
    nameplates.rebuildStreams(container.children);
    const mesh = createMSDFNameplateLayer(
      this.scene,
      container as any,
      nameplates as any,
      this.fontAsset,
      {
        renderingGroupId: this.options.renderingGroupId ?? 1,
        depthTest: this.options.depthTest ?? true,
        thickness: this.options.thickness,
        debug: this.options.debug,
        visibilitySource: 'actor',
      }
    );
    mesh.setEnabled(this.enabled && records.size > 0);

    this.container = container;
    this.nameplates = nameplates;
    this.mesh = mesh;
    this.records = records;
  }

  private needsRebuild(inputs: readonly ShadoDynamicEntityNameplateInput[]): boolean {
    if (!this.built) return true;
    for (let index = 0; index < inputs.length; index++) {
      const input = inputs[index]!;
      const text = input.text;
      if (!text) continue;
      const record = this.records.get(input.id);
      if (record) {
        if (record.text === text || record.text === text.trim()) continue;
        return true;
      }
      if (text.trim()) return true;
    }
    return false;
  }

  private updateActors(inputs: readonly ShadoDynamicEntityNameplateInput[]): void {
    // A stamp per pass instead of an id map rebuilt every frame: each input
    // finds its record directly, and whatever no input reached is hidden.
    const pass = ++this.updatePass;
    const fallbackSize = Number(this.options.fontSize ?? DEFAULT_FONT_SIZE);
    const worldScale = Math.max(0.001, Number(this.options.worldScale ?? DEFAULT_WORLD_SCALE));
    const lift = Number(this.options.nameLiftWorld ?? DEFAULT_NAME_LIFT_WORLD);
    for (let index = 0; index < inputs.length; index++) {
      const input = inputs[index]!;
      const record = this.records.get(input.id);
      if (!record) continue;
      record.seen = pass;
      const actor = record.actor;
      actor.visibleFlag = input.visible !== false && input.text.trim() ? 1 : 0;
      const fontSize = Math.max(8, Number(input.fontSize ?? fallbackSize));
      // Element writes: a literal per plate per frame was garbage the size
      // of the crowd.
      const translation = actor.translation;
      translation[0] = input.x;
      translation[1] = input.z ?? this.options.zOffset ?? DEFAULT_Z_OFFSET;
      translation[2] = input.y;
      translation[3] = 1;
      actor.nameWorldPerEM = fontSize * worldScale;
      actor.nameLiftWorld = lift;
      const rgba = cachedRgba(input.color ?? this.options.color ?? DEFAULT_COLOR);
      const color = actor.nameplateColor;
      color[0] = rgba[0];
      color[1] = rgba[1];
      color[2] = rgba[2];
      color[3] = rgba[3];
      actor.billboardFlag = input.billboard === false ? 0 : 1;
      actor.emitHeaderDirty();
    }
    // `forEach`, not `for...of`: the iterator hands back a fresh [key, value]
    // pair per entry.
    this.records.forEach(record => {
      if (record.seen === pass) return;
      record.actor.visibleFlag = 0;
      record.actor.emitHeaderDirty();
    });
    this.container?.arena.markDirty?.();
  }
}
