import { BABYLON, type RawTexture } from '../babylon';
import { ShadoWorldLightState } from '../world/point-lights';
import type {
  ShadoWorldLightReductionOptions,
  ShadoWorldLightReductionResult,
  ShadoWorldVisibilityCoordinator,
  ShadoWorldVisibilityFrame,
} from '../world/ShadoWorldVisibilityCoordinator';
import type {
  ShadoWorldCompiledPointLight,
  ShadoWorldSpatialPackage,
  WorldVec3,
} from '../world/types';

type RuntimeLightBehavior = {
  mode: 'always' | 'night' | 'schedule';
  onHour: number;
  offHour: number;
  transitionMinutes: number;
  flicker: 'steady' | 'flame' | 'wisp';
  amplitude: number;
  speed: number;
  seed: number;
};

export const SHADO_WORLD_LIGHT_FIELD_BUFFER_NAME = 'uShadoLightField';
export const SHADO_WORLD_LIGHT_FIELD_TEXTURE_NAME = 'uShadoLightFieldTexture';
export const SHADO_WORLD_LIGHT_FIELD_PARAMS_0 = 'uShadoLightFieldParams0';
export const SHADO_WORLD_LIGHT_FIELD_PARAMS_1 = 'uShadoLightFieldParams1';
export const SHADO_WORLD_LIGHT_FIELD_PARAMS_2 = 'uShadoLightFieldParams2';

// Source-compatible aliases for imports written against the first prototype.
export const SHADO_WORLD_LIGHT_BUFFER_NAME = SHADO_WORLD_LIGHT_FIELD_BUFFER_NAME;
export const SHADO_WORLD_LIGHT_INDEX_BUFFER_NAME = SHADO_WORLD_LIGHT_FIELD_BUFFER_NAME;
export const SHADO_WORLD_LIGHT_COUNT_UNIFORM = SHADO_WORLD_LIGHT_FIELD_PARAMS_2;

export type ShadoWorldLightFieldBackend = 'storage' | 'datatex';

export type ShadoWorldLightMaterialTarget = {
  setTexture?(name: string, texture: any): unknown;
  setVector4?(name: string, value: any): unknown;
  updateVector4?(name: string, value: any): unknown;
  /** Force stock Babylon materials to run their plugin bind after field install. */
  markDirty?(forceMaterialDirty?: boolean): unknown;
};
export type ShadoWorldLightEffectTarget = ShadoWorldLightMaterialTarget;

export type ShadoWorldLightFieldDiagnostics = {
  lightCount: number;
  cellCount: number;
  lightReferences: number;
  occupiedCells: number;
  maxLightsPerCell: number;
  averageLightsPerOccupiedCell: number;
  arenaBytes: number;
};

export type ShadoWorldLightBinding = {
  readonly activeCount: number;
  readonly backend: ShadoWorldLightFieldBackend;
  bindMaterial(material: ShadoWorldLightMaterialTarget): void;
  bindEffect(effect: ShadoWorldLightEffectTarget): void;
};

export const SHADO_WORLD_LIGHT_FIELD_UNIFORMS = [
  SHADO_WORLD_LIGHT_FIELD_PARAMS_0,
  SHADO_WORLD_LIGHT_FIELD_PARAMS_1,
  SHADO_WORLD_LIGHT_FIELD_PARAMS_2,
] as const;

export function shadoWorldLightFieldSamplers(backend: ShadoWorldLightFieldBackend): string[] {
  return backend === 'datatex' ? [SHADO_WORLD_LIGHT_FIELD_TEXTURE_NAME] : [];
}

/** Shared WebGPU light-field reader. Work is proportional to lights overlapping one XZ cell. */
export const SHADO_WORLD_LIGHT_FIELD_WGSL = /* wgsl */ `
var uShadoLightFieldTextureSampler: sampler;
var uShadoLightFieldTexture: texture_2d<f32>;
uniform uShadoLightFieldParams0: vec4f;
uniform uShadoLightFieldParams1: vec4f;
uniform uShadoLightFieldParams2: vec4f;

fn shadoLightFieldRead(address: i32) -> f32 {
  let texel = address / 4;
  let width = i32(uniforms.uShadoLightFieldParams2.z);
  let packed = textureLoad(uShadoLightFieldTexture, vec2i(texel % width, texel / width), 0);
  let lane = address - texel * 4;
  if (lane == 0) { return packed.x; }
  if (lane == 1) { return packed.y; }
  if (lane == 2) { return packed.z; }
  return packed.w;
}

fn shadoWorldLightLambert(worldPosition: vec3f, worldNormal: vec3f) -> vec3f {
  let grid = uniforms.uShadoLightFieldParams0;
  let fieldIndex = uniforms.uShadoLightFieldParams1;
  let state = uniforms.uShadoLightFieldParams2;
  if (state.y < 0.5 || grid.z <= 0.0) { return vec3f(0.0); }
  let cellX = i32(floor((worldPosition.x - grid.x) * grid.z));
  let cellZ = i32(floor((worldPosition.z - grid.y) * grid.z));
  let width = i32(grid.w);
  let height = i32(fieldIndex.x);
  if (cellX < 0 || cellZ < 0 || cellX >= width || cellZ >= height) {
    return vec3f(0.0);
  }
  let header = i32(fieldIndex.z) + (cellZ * width + cellX) * 2;
  let first = i32(shadoLightFieldRead(header));
  let end = first + i32(shadoLightFieldRead(header + 1));
  var lighting = vec3f(0.0);
  for (var reference = first; reference < end; reference = reference + 1) {
    let light = i32(shadoLightFieldRead(i32(fieldIndex.w) + reference));
    if (shadoLightFieldRead(i32(state.x) + light) < 0.5) { continue; }
    let row = i32(fieldIndex.y) + light * 8;
    let position = vec3f(
      shadoLightFieldRead(row), shadoLightFieldRead(row + 1), shadoLightFieldRead(row + 2)
    );
    let range = shadoLightFieldRead(row + 3);
    let delta = position - worldPosition;
    let distance = length(delta);
    let attenuation = max(0.0, 1.0 - distance / max(range, 0.0001));
    if (attenuation <= 0.0) { continue; }
    // A 1.5-power shoulder carries useful warm light much farther than the
    // old squared hotspot while still reaching zero cleanly at the boundary.
    let softAttenuation = attenuation * sqrt(attenuation);
    let direction = delta / max(distance, 0.0001);
    // A small in-range floor avoids pitch-black back-facing actor/world
    // surfaces while retaining the cheap single-dot Lambert response.
    let lambert = max(dot(worldNormal, direction), state.w);
    let radiance = vec3f(
      shadoLightFieldRead(row + 4), shadoLightFieldRead(row + 5), shadoLightFieldRead(row + 6)
    );
    lighting += radiance * lambert * softAttenuation;
  }
  // Preserve low and medium contributions exactly, then compress only stacked
  // or malformed authored peaks. This avoids white plaza blowouts without a
  // hard clamp that would flatten flicker and light colour.
  let peak = max(lighting.r, max(lighting.g, lighting.b));
  if (peak > 0.9) {
    let excess = peak - 0.9;
    let compressedPeak = 0.9 + excess / (1.0 + excess / 0.6);
    lighting *= compressedPeak / peak;
  }
  return lighting;
}
`;

/** Shared WebGL2/data-texture light-field reader. */
export const SHADO_WORLD_LIGHT_FIELD_GLSL = /* glsl */ `
uniform highp sampler2D uShadoLightFieldTexture;
uniform vec4 uShadoLightFieldParams0;
uniform vec4 uShadoLightFieldParams1;
uniform vec4 uShadoLightFieldParams2;

float shadoLightFieldRead(int address) {
  int texel = address / 4;
  vec4 packed = texelFetch(
    uShadoLightFieldTexture,
    ivec2(texel % int(uShadoLightFieldParams2.z), texel / int(uShadoLightFieldParams2.z)), 0
  );
  int lane = address - texel * 4;
  return lane == 0 ? packed.x : lane == 1 ? packed.y : lane == 2 ? packed.z : packed.w;
}

vec3 shadoWorldLightLambert(vec3 worldPosition, vec3 worldNormal) {
  vec4 grid = uShadoLightFieldParams0;
  vec4 fieldIndex = uShadoLightFieldParams1;
  vec4 state = uShadoLightFieldParams2;
  if (state.y < 0.5 || grid.z <= 0.0) return vec3(0.0);
  int cellX = int(floor((worldPosition.x - grid.x) * grid.z));
  int cellZ = int(floor((worldPosition.z - grid.y) * grid.z));
  int width = int(grid.w);
  int height = int(fieldIndex.x);
  if (cellX < 0 || cellZ < 0 || cellX >= width || cellZ >= height) return vec3(0.0);
  int header = int(fieldIndex.z) + (cellZ * width + cellX) * 2;
  int first = int(shadoLightFieldRead(header) + 0.5);
  int end = first + int(shadoLightFieldRead(header + 1) + 0.5);
  vec3 lighting = vec3(0.0);
  for (int reference = first; reference < end; ++reference) {
    int light = int(shadoLightFieldRead(int(fieldIndex.w) + reference) + 0.5);
    if (shadoLightFieldRead(int(state.x) + light) < 0.5) continue;
    int row = int(fieldIndex.y) + light * 8;
    vec3 position = vec3(
      shadoLightFieldRead(row), shadoLightFieldRead(row + 1), shadoLightFieldRead(row + 2)
    );
    float range = shadoLightFieldRead(row + 3);
    vec3 delta = position - worldPosition;
    float distanceToLight = length(delta);
    float attenuation = max(0.0, 1.0 - distanceToLight / max(range, 0.0001));
    if (attenuation <= 0.0) continue;
    float softAttenuation = attenuation * sqrt(attenuation);
    vec3 direction = delta / max(distanceToLight, 0.0001);
    float lambert = max(dot(worldNormal, direction), state.w);
    vec3 radiance = vec3(
      shadoLightFieldRead(row + 4), shadoLightFieldRead(row + 5), shadoLightFieldRead(row + 6)
    );
    lighting += radiance * lambert * softAttenuation;
  }
  float peak = max(lighting.r, max(lighting.g, lighting.b));
  if (peak > 0.9) {
    float excess = peak - 0.9;
    float compressedPeak = 0.9 + excess / (1.0 + excess / 0.6);
    lighting *= compressedPeak / peak;
  }
  return lighting;
}
`;

type FieldLayout = {
  originX: number;
  originZ: number;
  cellSize: number;
  width: number;
  height: number;
  lightsBase: number;
  headersBase: number;
  indicesBase: number;
  activeBase: number;
};

/**
 * Cell-addressed light memory shared by WebGPU storage and WebGL2 float textures.
 * There is deliberately no global or per-draw light ceiling.
 */
export class ShadoWorldLightBuffer implements ShadoWorldLightBinding {
  public readonly state: ShadoWorldLightState;
  public readonly backend: ShadoWorldLightFieldBackend;
  public activeCount = 0;
  public lastReduction: ShadoWorldLightReductionResult | null = null;
  public diagnostics!: ShadoWorldLightFieldDiagnostics;

  private texture: RawTexture | null = null;
  private arena = new Float32Array(4);
  private textureUpload = new Float32Array(4);
  private layout!: FieldLayout;
  private textureWidth = 1;
  private textureHeight = 1;
  private readonly radianceScale: number;
  private readonly maxRadiance: number;
  private readonly rangeScale: number;
  private readonly minimumLambert: number;
  private readonly requestedCellSize?: number;
  private readonly engine: any;
  private readonly params0 = new BABYLON.Vector4();
  private readonly params1 = new BABYLON.Vector4();
  private readonly params2 = new BABYLON.Vector4();
  private activePhaseMask = 0xffffffff;
  private readonly runtimeBehaviors: RuntimeLightBehavior[];
  private readonly runtimeScales: Float32Array;
  private runtimeElapsedSeconds = 0;
  private runtimeUpdateElapsedMs = 80;

  public constructor(
    scene: { getEngine(): any },
    public readonly world: ShadoWorldSpatialPackage,
    _coordinator: ShadoWorldVisibilityCoordinator,
    options: {
      cellSize?: number;
      radianceScale?: number;
      maxRadiance?: number;
      rangeScale?: number;
      minimumLambert?: number;
    } = {}
  ) {
    this.engine = scene.getEngine();
    // A float texture is intentionally the common ABI. It works inside
    // Babylon material plugins on both renderers, unlike arbitrary storage.
    this.backend = 'datatex';
    this.state = new ShadoWorldLightState(world.pointLights ?? []);
    const runtimeLights = (world.pointLights ?? []).filter(light => light.runtime);
    this.runtimeBehaviors = runtimeLights.map(resolveRuntimeLightBehavior);
    this.runtimeScales = new Float32Array(this.state.count);
    this.runtimeScales.fill(1);
    // Authored intensity remains linear in the package ABI, but feeding values
    // around 13 directly into an additive Lambert term creates a tiny white
    // hotspot. Map it into a bounded render-space response and expand the
    // effective radius instead: a typical torch peaks near 1.3 and carries a
    // soft contribution for roughly thirty metres. The cap also contains bad
    // legacy data such as Crownward's otherwise isolated intensity-143 light.
    this.radianceScale = Math.max(0, options.radianceScale ?? 0.1);
    this.maxRadiance = Math.max(0, options.maxRadiance ?? 1.75);
    this.rangeScale = Math.max(0.01, options.rangeScale ?? 2.25);
    this.minimumLambert = Math.min(1, Math.max(0, options.minimumLambert ?? 0.12));
    this.requestedCellSize = options.cellSize;
    this.rebuildField();
  }

  /**
   * Applies authored operating hours and cheap deterministic flame motion.
   * The whole compact arena is uploaded at most 12.5 Hz; no Babylon lights,
   * scene nodes, buffer repacks, or per-frame random allocations are created.
   */
  public tickRuntime(deltaMs: number, timeOfDay: number): boolean {
    const elapsedMs = Math.max(0, deltaMs);
    this.runtimeElapsedSeconds += elapsedMs / 1000;
    this.runtimeUpdateElapsedMs += elapsedMs;
    if (this.runtimeUpdateElapsedMs < 80) return false;
    this.runtimeUpdateElapsedMs %= 80;
    const hour = wrapHour(timeOfDay);
    let changed = false;
    let activeChanged = false;
    for (let row = 0; row < this.state.count; row++) {
      const behavior = this.runtimeBehaviors[row]!;
      const schedule = scheduleLevel(behavior, hour);
      const flicker = flickerLevel(behavior, this.runtimeElapsedSeconds);
      const scale = schedule * flicker;
      if (Math.abs(this.runtimeScales[row]! - scale) < 0.0005) continue;
      this.runtimeScales[row] = scale;
      this.writeLightRow(row);
      const activeAddress = this.layout.activeBase + row;
      const nextActive = this.rowIsActive(row) ? 1 : 0;
      if (this.arena[activeAddress] !== nextActive) {
        this.arena[activeAddress] = nextActive;
        activeChanged = true;
      }
      changed = true;
    }
    if (changed) {
      this.uploadArena();
    }
    if (activeChanged) {
      this.activeCount = this.countActiveRows();
      this.lastReduction = null;
    }
    return changed;
  }

  public reduce(
    _planes: ArrayLike<number>,
    _frame: ShadoWorldVisibilityFrame,
    _camera: WorldVec3,
    options: Omit<ShadoWorldLightReductionOptions, 'camera' | 'maxActiveLights'> = {}
  ): ShadoWorldLightReductionResult {
    // Camera/frustum repacking is intentionally absent. The cell adjacency is
    // already the GPU culling structure, so moving the camera causes zero light
    // uploads. This plane changes only when gameplay phase selection changes.
    const activePhaseMask = (options.activePhaseMask ?? 0xffffffff) >>> 0;
    if (this.lastReduction && activePhaseMask === this.activePhaseMask) {
      return this.lastReduction;
    }
    this.activePhaseMask = activePhaseMask;
    const active: number[] = [];
    let changed = false;
    for (let row = 0; row < this.state.count; row++) {
      const enabled = this.rowIsActive(row);
      const next = enabled ? 1 : 0;
      if (this.arena[this.layout.activeBase + row] !== next) {
        this.arena[this.layout.activeBase + row] = next;
        changed = true;
      }
      if (enabled) active.push(row);
    }
    if (changed) this.uploadArena();
    const activeIndices = Uint32Array.from(active);
    const result: ShadoWorldLightReductionResult = {
      visibleIndices: activeIndices,
      activeIndices,
      flags: new Uint8Array(this.state.count),
      totalRuntimeLights: this.state.count,
      spatialCandidateCount: active.length,
      capped: false,
    };
    this.activeCount = active.length;
    this.lastReduction = result;
    return result;
  }

  public updateLight(
    light: string | number,
    patch: Parameters<ShadoWorldLightState['update']>[1]
  ): number {
    const structural = patch.position !== undefined || patch.range !== undefined;
    const row = this.state.update(light, patch);
    if (structural) this.rebuildField();
    else {
      this.writeLightRow(row);
      if (patch.enabled !== undefined || patch.phaseMask !== undefined) {
        this.arena[this.layout.activeBase + row] = this.rowIsActive(row) ? 1 : 0;
      }
      this.uploadArena();
    }
    this.activeCount = this.countActiveRows();
    this.lastReduction = null;
    return row;
  }

  public bindMaterial(material: ShadoWorldLightMaterialTarget): void {
    this.bind(material);
  }
  public bindEffect(effect: ShadoWorldLightEffectTarget): void {
    this.bind(effect);
  }

  public dispose(): void {
    this.texture?.dispose();
    this.texture = null;
    this.activeCount = 0;
    this.lastReduction = null;
  }

  private rebuildField(): void {
    const visibility = this.world.visibility;
    const chosenSize =
      this.requestedCellSize ?? Math.min(64, visibility?.size ?? this.world.tiles.size ?? 64);
    const cellSize = Math.max(4, chosenSize);
    const originX = visibility?.originX ?? this.world.bounds.min[0];
    const originZ = visibility?.originZ ?? this.world.bounds.min[2];
    const useVisibilityGrid = visibility && Math.abs(visibility.size - cellSize) < 0.001;
    const width = useVisibilityGrid
      ? visibility.width
      : Math.max(1, Math.ceil((this.world.bounds.max[0] - originX) / cellSize));
    const height = useVisibilityGrid
      ? visibility.height
      : Math.max(1, Math.ceil((this.world.bounds.max[2] - originZ) / cellSize));
    const cells: number[][] = Array.from({ length: width * height }, () => []);
    for (let light = 0; light < this.state.count; light++) {
      const range = this.state.range[light] * this.rangeScale;
      const minX = clampCell(
        Math.floor((this.state.positionX[light] - range - originX) / cellSize),
        width
      );
      const maxX = clampCell(
        Math.floor((this.state.positionX[light] + range - originX) / cellSize),
        width
      );
      const minZ = clampCell(
        Math.floor((this.state.positionZ[light] - range - originZ) / cellSize),
        height
      );
      const maxZ = clampCell(
        Math.floor((this.state.positionZ[light] + range - originZ) / cellSize),
        height
      );
      for (let z = minZ; z <= maxZ; z++) {
        for (let x = minX; x <= maxX; x++) cells[z * width + x]!.push(light);
      }
    }
    const lightReferences = cells.reduce((sum, cell) => sum + cell.length, 0);
    const lightsBase = 0;
    const headersBase = Math.max(1, this.state.count) * 8;
    const indicesBase = headersBase + cells.length * 2;
    const activeBase = indicesBase + lightReferences;
    this.layout = {
      originX,
      originZ,
      cellSize,
      width,
      height,
      lightsBase,
      headersBase,
      indicesBase,
      activeBase,
    };
    this.arena = new Float32Array(Math.max(4, activeBase + Math.max(1, this.state.count)));
    for (let row = 0; row < this.state.count; row++) {
      this.writeLightRow(row);
      this.arena[activeBase + row] =
        this.rowIsActive(row) ? 1 : 0;
    }
    let reference = 0;
    let occupiedCells = 0;
    let maxLightsPerCell = 0;
    cells.forEach((cell, index) => {
      this.arena[headersBase + index * 2] = reference;
      this.arena[headersBase + index * 2 + 1] = cell.length;
      if (cell.length) occupiedCells++;
      maxLightsPerCell = Math.max(maxLightsPerCell, cell.length);
      for (const row of cell) this.arena[indicesBase + reference++] = row;
    });
    this.diagnostics = {
      lightCount: this.state.count,
      cellCount: cells.length,
      lightReferences,
      occupiedCells,
      maxLightsPerCell,
      averageLightsPerOccupiedCell: occupiedCells ? lightReferences / occupiedCells : 0,
      arenaBytes: this.arena.byteLength,
    };
    this.params0.set(originX, originZ, 1 / cellSize, width);
    this.params1.set(height, lightsBase, headersBase, indicesBase);
    this.recreateResource();
    this.activeCount = this.countActiveRows();
  }

  private writeLightRow(row: number): void {
    const source = row * 8;
    const target = this.layout.lightsBase + source;
    const runtimeScale = this.runtimeScales[row] ?? 1;
    const rawPeak = Math.max(
      this.state.packed[source + 4]!,
      this.state.packed[source + 5]!,
      this.state.packed[source + 6]!,
    );
    const radianceGain = rawPeak > 0
      ? Math.min(
          this.radianceScale * runtimeScale,
          this.maxRadiance / rawPeak,
        )
      : 0;
    for (let lane = 0; lane < 8; lane++) {
      const scale = lane === 3
        ? this.rangeScale
        : lane >= 4 && lane <= 6
          ? radianceGain
          : 1;
      this.arena[target + lane] = this.state.packed[source + lane]! * scale;
    }
  }

  private rowIsActive(row: number): boolean {
    return this.state.enabled[row] !== 0 &&
      (this.state.phaseMask[row] & this.activePhaseMask) !== 0 &&
      (this.runtimeScales[row] ?? 1) > 0.001;
  }

  private recreateResource(): void {
    this.texture?.dispose();
    this.texture = null;
    const texels = Math.max(1, Math.ceil(this.arena.length / 4));
    this.textureWidth = Math.min(2048, texels);
    this.textureHeight = Math.max(1, Math.ceil(texels / this.textureWidth));
    this.textureUpload = new Float32Array(this.textureWidth * this.textureHeight * 4);
    this.textureUpload.set(this.arena);
    this.texture = new BABYLON.RawTexture(
      this.textureUpload,
      this.textureWidth,
      this.textureHeight,
      BABYLON.Engine.TEXTUREFORMAT_RGBA,
      this.engine,
      false,
      false,
      BABYLON.Texture.NEAREST_SAMPLINGMODE,
      BABYLON.Engine.TEXTURETYPE_FLOAT
    );
    this.texture.wrapU = this.texture.wrapV = BABYLON.Texture.CLAMP_ADDRESSMODE;
    this.params2.set(
      this.layout.activeBase,
      this.state.count,
      this.textureWidth,
      this.minimumLambert
    );
  }

  private uploadArena(): void {
    if (this.texture) {
      this.textureUpload.fill(0);
      this.textureUpload.set(this.arena);
      this.texture.update(this.textureUpload);
    }
  }

  private bind(target: ShadoWorldLightMaterialTarget): void {
    target.setTexture?.(SHADO_WORLD_LIGHT_FIELD_TEXTURE_NAME, this.texture);
    const setVector = target.setVector4?.bind(target) ?? target.updateVector4?.bind(target);
    setVector?.(SHADO_WORLD_LIGHT_FIELD_PARAMS_0, this.params0);
    setVector?.(SHADO_WORLD_LIGHT_FIELD_PARAMS_1, this.params1);
    setVector?.(SHADO_WORLD_LIGHT_FIELD_PARAMS_2, this.params2);
  }

  private countActiveRows(): number {
    let count = 0;
    for (let row = 0; row < this.state.count; row++) {
      if (this.arena[this.layout.activeBase + row] >= 0.5) count++;
    }
    return count;
  }
}

function clampCell(value: number, count: number): number {
  return Math.max(0, Math.min(count - 1, value));
}

function resolveRuntimeLightBehavior(light: ShadoWorldCompiledPointLight): RuntimeLightBehavior {
  const metadata = light.metadata ?? {};
  const legacyKind = typeof metadata.kind === 'string' ? metadata.kind.toLowerCase() : '';
  const legacyProfile = typeof metadata.flickerProfile === 'string'
    ? metadata.flickerProfile.toLowerCase()
    : '';
  const flameLike = /fire|flame|torch|lantern|brazier/.test(`${legacyKind} ${legacyProfile}`);
  const legacyInterior = metadata.interior === true;
  const legacyMode = metadata.activationMode;
  const mode = light.activation?.mode ?? (
    legacyMode === 'always' || legacyMode === 'night' || legacyMode === 'schedule'
      ? legacyMode
      : legacyInterior ? 'always' : flameLike ? 'night' : 'always'
  );
  const profile = light.flicker?.profile ?? (
    legacyProfile.includes('wisp') ? 'wisp' : flameLike ? 'flame' : 'steady'
  );
  const legacyAmplitude = finiteMetadataNumber(metadata.flickerAmplitude);
  const legacySpeed = finiteMetadataNumber(metadata.flickerSpeed);
  return {
    mode,
    onHour: light.activation?.onHour ?? finiteMetadataNumber(metadata.onHour) ?? 18,
    offHour: light.activation?.offHour ?? finiteMetadataNumber(metadata.offHour) ?? 6,
    transitionMinutes:
      light.activation?.transitionMinutes ??
      finiteMetadataNumber(metadata.transitionMinutes) ??
      25,
    flicker: profile,
    amplitude: Math.min(0.5, Math.max(0,
      light.flicker?.amplitude ?? legacyAmplitude ?? (profile === 'steady' ? 0 : 0.065)
    )),
    speed: Math.min(30, Math.max(0,
      light.flicker?.speed ?? legacySpeed ?? (profile === 'wisp' ? 2.4 : 6.5)
    )),
    seed: hashLightId(light.id),
  };
}

function scheduleLevel(behavior: RuntimeLightBehavior, hour: number): number {
  if (behavior.mode === 'always') return 1;
  const onHour = behavior.mode === 'night' ? 18 : wrapHour(behavior.onHour);
  const offHour = behavior.mode === 'night' ? 6 : wrapHour(behavior.offHour);
  const duration = positiveModulo(offHour - onHour, 24) || 24;
  const elapsed = positiveModulo(hour - onHour, 24);
  if (elapsed > duration) return 0;
  const fade = Math.min(duration * 0.5, Math.max(0, behavior.transitionMinutes / 60));
  if (fade <= 0) return 1;
  return Math.min(smoothstep(0, fade, elapsed), smoothstep(0, fade, duration - elapsed));
}

function flickerLevel(behavior: RuntimeLightBehavior, seconds: number): number {
  if (behavior.flicker === 'steady' || behavior.amplitude <= 0 || behavior.speed <= 0) return 1;
  const phase = seconds * behavior.speed * Math.PI * 2 + behavior.seed * Math.PI * 2;
  const broad = Math.sin(phase * 0.47) * 0.52;
  const lick = Math.sin(phase * 1.13 + behavior.seed * 11.7) * 0.31;
  const shimmer = Math.sin(phase * 2.41 + behavior.seed * 23.1) * 0.17;
  const noise = behavior.flicker === 'wisp'
    ? broad * 0.8 + lick * 0.2
    : broad + lick + shimmer;
  return Math.max(0, 1 + behavior.amplitude * noise);
}

function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.max(0, Math.min(1, (value - edge0) / Math.max(0.0001, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function wrapHour(hour: number): number {
  return positiveModulo(Number.isFinite(hour) ? hour : 12, 24);
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function finiteMetadataNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function hashLightId(id: string): number {
  let hash = 2166136261;
  for (let index = 0; index < id.length; index++) {
    hash ^= id.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 0xffffffff;
}
