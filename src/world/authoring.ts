import {
  SHADO_WORLD_AUTHORING_EXTRAS_KEY,
  type ShadoWorldAuthoringDocument,
  type ShadoWorldObjectStamp,
  type ShadoWorldAuthoringRegion,
  type ShadoWorldRegionKind,
  type ShadoWorldEnvironmentAuthoring,
} from './types';

const REGION_KINDS = new Set([
  'visibility-cell', 'streaming', 'water', 'lava', 'safe',
  'pvp', 'zone-line', 'audio', 'trigger', 'fx', 'semantic',
]);

const TERRAIN_PROJECTIONS = new Set(['world-xz', 'triplanar', 'hybrid']);

export const DEFAULT_SHADO_WORLD_BAKE_SETTINGS = {
  tileSize: 32,
  maxClusterTriangles: 128,
  minRenderChunkTriangles: 64,
  maxRenderChunkExtent: 256,
  visibilityRegionSize: 64,
  visibilityMaxDistance: 1280,
  physicsChunkSize: 256,
} as const;

export const DEFAULT_SHADO_WORLD_ENVIRONMENT: ShadoWorldEnvironmentAuthoring = {
  sky: { mode: 'solid', color: [0.018, 0.025, 0.04], intensity: 1 },
  fog: { enabled: false, mode: 'linear', color: [0.08, 0.1, 0.14], density: 0.002, start: 100, end: 1800 },
  ambient: { color: [0.65, 0.72, 0.88], intensity: 1.15 },
  weather: { preset: 'clear', intensity: 0, wind: [0, 0, 0] },
  timeOfDay: { hour: 12, cycleSeconds: 0, running: false },
  water: { enabled: false, level: 0, reflections: true },
  audioEmitters: [],
  reflectionProbes: [],
};

export const DEFAULT_SHADO_WORLD_PERFORMANCE_BUDGETS = {
  maxVisibleTriangles: 2_000_000,
  maxDrawCalls: 2_000,
  maxMaterials: 512,
  maxTextures: 1_024,
  // The authored world may contain any number of lights. This bounds only the
  // compact PVS-selected set traversed by one shader draw.
  maxRuntimePointLights: 256,
  maxCollisionTriangles: 1_000_000,
  maxRenderChunks: 20_000,
  maxPvsVisibleCells: 2_048,
} as const;

export function createShadoWorldAuthoring(world: string): ShadoWorldAuthoringDocument {
  if (!world.trim()) throw new Error('World authoring requires a world name');
  return {
    kind: 'shado.world.authoring',
    version: 1,
    world,
    coordinateSystem: 'babylon-y-up',
    revision: 0,
    regions: [],
    objects: {
      prototypes: [],
      stamps: [],
    },
    geometry: {
      meshes: [],
      materials: [],
      patches: [],
    },
    lighting: {
      pointLights: [],
    },
    environment: structuredClone(DEFAULT_SHADO_WORLD_ENVIRONMENT),
    terrain: {
      enabled: false,
      controlMaps: [],
      layers: [],
    },
    bake: { ...DEFAULT_SHADO_WORLD_BAKE_SETTINGS },
    performanceBudgets: { ...DEFAULT_SHADO_WORLD_PERFORMANCE_BUDGETS },
    playability: { fallRecoveryY: -10_000, recoveryPosition: [0, 0, 0], entrances: [], criticalRegions: [], probes: [] },
  };
}

export function validateShadoWorldAuthoring(
  value: unknown,
  expectedWorld?: string
): ShadoWorldAuthoringDocument {
  const document = value as ShadoWorldAuthoringDocument;
  if (
    !document || document.kind !== 'shado.world.authoring' || document.version !== 1 ||
    document.coordinateSystem !== 'babylon-y-up' || !Array.isArray(document.regions)
  ) {
    throw new Error('Unsupported Shado world authoring document');
  }
  if (!document.world || (expectedWorld && document.world !== expectedWorld)) {
    throw new Error(`World authoring target mismatch: expected '${expectedWorld}', got '${document.world}'`);
  }
  // Version 1 region-only documents remain loadable. The normalized object
  // planes are added in memory and included on the next editor save.
  document.objects ??= { prototypes: [], stamps: [] };
  if (!Array.isArray(document.objects.prototypes) || !Array.isArray(document.objects.stamps)) {
    throw new Error('World authoring objects require prototype and stamp arrays');
  }
  document.geometry ??= { meshes: [], materials: [], patches: [] };
  document.geometry.patches ??= [];
  if (!Array.isArray(document.geometry.meshes) || !Array.isArray(document.geometry.materials) || !Array.isArray(document.geometry.patches)) {
    throw new Error('World geometry authoring requires mesh, material, and patch arrays');
  }
  document.lighting ??= { pointLights: [] };
  if (!Array.isArray(document.lighting.pointLights)) {
    throw new Error('World lighting authoring requires a pointLights array');
  }
  document.terrain ??= { enabled: false, controlMaps: [], layers: [] };
  document.bake ??= { ...DEFAULT_SHADO_WORLD_BAKE_SETTINGS };
  document.environment ??= structuredClone(DEFAULT_SHADO_WORLD_ENVIRONMENT);
  document.performanceBudgets ??= { ...DEFAULT_SHADO_WORLD_PERFORMANCE_BUDGETS };
  document.playability ??= { fallRecoveryY: -10_000, recoveryPosition: [0, 0, 0], entrances: [], criticalRegions: [], probes: [] };
  const ids = new Set<string>();
  document.regions.forEach((region, index) => validateRegion(region, index, ids));
  validateObjects(document);
  validateGeometry(document);
  validateLighting(document);
  validateTerrain(document);
  validateBake(document);
  validateEnvironment(document);
  validatePerformanceBudgets(document);
  validatePlayability(document);
  if (!Number.isInteger(document.revision) || document.revision < 0) {
    throw new Error('World authoring revision must be a non-negative integer');
  }
  if (
    document.legacyObjectExclusions !== undefined &&
    (!Array.isArray(document.legacyObjectExclusions) ||
      document.legacyObjectExclusions.some(id => typeof id !== 'string' || !id.trim()) ||
      new Set(document.legacyObjectExclusions).size !== document.legacyObjectExclusions.length)
  ) {
    throw new Error('World authoring legacy object exclusions must be unique non-empty IDs');
  }
  return document;
}

function validateEnvironment(document: ShadoWorldAuthoringDocument): void {
  const environment = document.environment;
  if (!environment || !['solid', 'procedural', 'texture'].includes(environment.sky?.mode) || !['linear', 'exponential', 'exponential-squared'].includes(environment.fog?.mode)) throw new Error('World environment requires valid sky and fog settings');
  validateVec3(environment.sky.color, 'World sky color', false);
  validateVec3(environment.fog.color, 'World fog color', false);
  validateVec3(environment.ambient.color, 'World ambient color', false);
  validateVec3(environment.weather.wind, 'World weather wind', false);
  if (environment.sky.texture !== undefined && !environment.sky.texture.trim()) throw new Error('World sky texture must be a non-empty URL');
  for (const [name, value] of Object.entries({ skyIntensity: environment.sky.intensity, fogDensity: environment.fog.density, fogStart: environment.fog.start, fogEnd: environment.fog.end, ambientIntensity: environment.ambient.intensity, weatherIntensity: environment.weather.intensity, hour: environment.timeOfDay.hour, cycleSeconds: environment.timeOfDay.cycleSeconds, waterLevel: environment.water.level })) if (!Number.isFinite(value)) throw new Error(`World environment ${name} must be finite`);
  const ids = new Set<string>();
  for (const emitter of environment.audioEmitters) { if (!emitter.id?.trim() || ids.has(emitter.id) || !emitter.source?.trim()) throw new Error('World audio emitters require unique IDs and sources'); ids.add(emitter.id); validateVec3(emitter.position, `Audio emitter '${emitter.id}' position`, false); positive(emitter.range, `Audio emitter '${emitter.id}' range`); if (!Number.isFinite(emitter.volume) || emitter.volume < 0) throw new Error(`Audio emitter '${emitter.id}' volume must be non-negative`); validateMetadata(emitter.metadata, `Audio emitter '${emitter.id}'`); }
  ids.clear();
  for (const probe of environment.reflectionProbes) { if (!probe.id?.trim() || ids.has(probe.id)) throw new Error('World reflection probes require unique IDs'); ids.add(probe.id); validateVec3(probe.position, `Reflection probe '${probe.id}' position`, false); validateVec3(probe.size, `Reflection probe '${probe.id}' size`, true); positive(probe.resolution, `Reflection probe '${probe.id}' resolution`); validateMetadata(probe.metadata, `Reflection probe '${probe.id}'`); }
}

function validatePerformanceBudgets(document: ShadoWorldAuthoringDocument): void {
  document.performanceBudgets.maxRuntimePointLights ??=
    DEFAULT_SHADO_WORLD_PERFORMANCE_BUDGETS.maxRuntimePointLights;
  for (const [name, value] of Object.entries(document.performanceBudgets)) {
    if (name === 'maxRuntimePointLights' && value === 0) {
      // Upgrade bake-only version-1 documents to the new active-list budget.
      document.performanceBudgets.maxRuntimePointLights =
        DEFAULT_SHADO_WORLD_PERFORMANCE_BUDGETS.maxRuntimePointLights;
      continue;
    }
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`Performance budget ${name} must be a positive integer`);
    }
  }
}

function validatePlayability(document: ShadoWorldAuthoringDocument): void {
  const playability = document.playability;
  if (!Number.isFinite(playability.fallRecoveryY)) throw new Error('Playability fallRecoveryY must be finite');
  validateVec3(playability.recoveryPosition, 'Playability recovery position', false);
  const ids = new Set<string>();
  for (const entrance of playability.entrances) { if (!entrance.id?.trim() || ids.has(entrance.id)) throw new Error('Playability entrances require unique IDs'); ids.add(entrance.id); validateVec3(entrance.position, `Entrance '${entrance.id}' position`, false); positive(entrance.radius, `Entrance '${entrance.id}' radius`); }
  ids.clear();
  for (const probe of playability.probes) { if (!probe.id?.trim() || ids.has(probe.id)) throw new Error('Playability probes require unique IDs'); ids.add(probe.id); validateVec3(probe.from, `Probe '${probe.id}' from`, false); validateVec3(probe.to, `Probe '${probe.id}' to`, false); }
}

function validateTerrain(document: ShadoWorldAuthoringDocument): void {
  const terrain = document.terrain;
  if (
    !terrain || typeof terrain.enabled !== 'boolean' ||
    !Array.isArray(terrain.controlMaps) ||
    terrain.controlMaps.some(value => typeof value !== 'string' || !value.trim()) ||
    !Array.isArray(terrain.layers)
  ) {
    throw new Error('World terrain authoring requires enabled, controlMaps, and layers');
  }
  const ids = new Set<string>();
  terrain.layers.forEach((layer, index) => {
    if (!layer?.id?.trim() || ids.has(layer.id)) {
      throw new Error(`Terrain layer ${index} has a missing or duplicate stable ID`);
    }
    ids.add(layer.id);
    if (!layer.name?.trim() || !layer.material?.trim()) {
      throw new Error(`Terrain layer '${layer.id}' requires a name and material`);
    }
    if (typeof layer.enabled !== 'boolean' || !TERRAIN_PROJECTIONS.has(layer.projection)) {
      throw new Error(`Terrain layer '${layer.id}' has invalid enabled/projection state`);
    }
    positive(layer.textureScale, `Terrain layer '${layer.id}' texture scale`);
    if (!Number.isFinite(layer.weight) || layer.weight < 0) {
      throw new Error(`Terrain layer '${layer.id}' weight must be non-negative`);
    }
    validateRange(layer.slope, `Terrain layer '${layer.id}' slope`, 0, 1);
    validateRange(layer.altitude, `Terrain layer '${layer.id}' altitude`);
    if (!Number.isFinite(layer.noiseScale) || layer.noiseScale < 0) {
      throw new Error(`Terrain layer '${layer.id}' noise scale must be non-negative`);
    }
    if (layer.control !== undefined && (typeof layer.control !== 'string' || !layer.control.trim())) {
      throw new Error(`Terrain layer '${layer.id}' control must be a non-empty string`);
    }
    validateMetadata(layer.metadata, `Terrain layer '${layer.id}'`);
  });
}

function validateBake(document: ShadoWorldAuthoringDocument): void {
  for (const [name, value] of Object.entries(document.bake)) {
    positive(value, `World bake ${name}`);
    if (!Number.isInteger(value)) throw new Error(`World bake ${name} must be an integer`);
  }
  if (document.bake.maxClusterTriangles > 65_535) {
    throw new Error('World bake maxClusterTriangles exceeds 65535');
  }
}

function validateRange(
  value: unknown,
  label: string,
  minimum = -Number.MAX_VALUE,
  maximum = Number.MAX_VALUE
): void {
  if (
    !Array.isArray(value) || value.length !== 2 ||
    value.some(item => !Number.isFinite(item)) ||
    value[0] > value[1] || value[0] < minimum || value[1] > maximum
  ) {
    throw new Error(`${label} must be an ordered two-number range`);
  }
}

export type LegacyZoneMetadataImportOptions = {
  objectSourcePrefix?: string;
  objectSourceExtension?: string;
  defaultObjectBoundsRadius?: number;
  /**
   * Requiem sidecars are already Y-up. The value records provenance while
   * both accepted inputs use the current source-space placement contract.
   */
  sourceCoordinateSystem?: 'requiem-y-up' | 'babylon-y-up';
};

export type LegacyZoneObjectTransform = {
  x?: number;
  y?: number;
  z?: number;
  rotateX?: number;
  rotateY?: number;
  rotateZ?: number;
  scale?: number;
  scaleX?: number;
  scaleY?: number;
  scaleZ?: number;
};

/**
 * Converts one imported placement into Shado's durable Y-up source-space contract.
 * Runtime consumers must use the returned values verbatim and must not repeat
 * any loader or ObjectCache axis correction.
 */
export function legacyZoneObjectTransformToBabylon(
  transform: LegacyZoneObjectTransform
): Pick<ShadoWorldObjectStamp, 'position' | 'rotationDegrees' | 'scale'> {
  const uniformScale = finite(transform.scale, 1);
  return {
    // Requiem sidecars already carry canonical gameplay placements. Zone
    // geometry receives its separate canonical X reflection during scene and
    // spatial preprocessing; reflecting placements here would be incorrect.
    position: [
      finite(transform.x),
      finite(transform.y),
      finite(transform.z),
    ],
    rotationDegrees: [
      finite(transform.rotateX),
      finite(transform.rotateY),
      finite(transform.rotateZ),
    ],
    scale: [
      finite(transform.scaleX, uniformScale),
      finite(transform.scaleY, uniformScale),
      finite(transform.scaleZ, uniformScale),
    ],
  };
}

/**
 * Promotes the original Requiem zone JSON into durable authoring data. Render
 * models stay deduplicated as prototypes; each placement becomes a stable,
 * fully Babylon-space stamp. No coordinate conversion remains for the client.
 */
export function importLegacyZoneMetadata(
  value: unknown,
  world: string,
  options: LegacyZoneMetadataImportOptions = {}
): ShadoWorldAuthoringDocument {
  const legacy = value as {
    version?: number;
    objects?: Record<string, LegacyZoneObjectTransform[]>;
    regions?: Array<{
      minVertex?: number[];
      maxVertex?: number[];
      center?: number[];
      regionType?: number;
      zoneLineInfo?: unknown;
      [key: string]: unknown;
    }>;
    lights?: unknown[];
    sounds?: unknown[];
  };
  if (!legacy || typeof legacy !== 'object' || Array.isArray(legacy)) {
    throw new Error('Legacy zone metadata must be a JSON object');
  }
  const document = createShadoWorldAuthoring(world);
  const prefix = (options.objectSourcePrefix ?? '/eqrequiem/objects').replace(/\/$/, '');
  const extension = options.objectSourceExtension ?? '/final.glb.gz';
  const boundsRadius = positive(
    options.defaultObjectBoundsRadius ?? 32,
    'default object bounds radius'
  );
  const sourceCoordinateSystem = options.sourceCoordinateSystem ?? 'requiem-y-up';
  const objectEntries = Object.entries(legacy.objects ?? {})
    .filter(([, transforms]) => Array.isArray(transforms))
    .sort(([a], [b]) => a.localeCompare(b));
  for (const [model, transforms] of objectEntries) {
    const prototypeId = stableId(model, 'object');
    document.objects.prototypes.push({
      id: prototypeId,
      source: `${prefix}/${model}${extension}`,
      boundsRadius,
      metadata: {
        legacyModel: model,
        sourceCoordinateSystem,
        generatedAsset: 'final.glb.gz',
      },
    });
    transforms.forEach((transform, index) => {
      const normalized = legacyZoneObjectTransformToBabylon(transform);
      document.objects.stamps.push({
        id: `${prototypeId}-${index}`,
        prototype: prototypeId,
        enabled: true,
        ...normalized,
        phaseMask: 0xffffffff,
        tags: [],
        metadata: {
          legacyIndex: index,
          sourceCoordinateSystem,
          transformNormalizedAtPreprocess: true,
          transformContract: 'requiem-y-up-v2',
        },
      });
    });
  }
  for (const [index, region] of (legacy.regions ?? []).entries()) {
    const min = vec3(region.minVertex);
    const max = vec3(region.maxVertex);
    const sourceCenter: [number, number, number] = region.center?.length === 3
      ? vec3(region.center)
      : [
          (min[0] + max[0]) * 0.5,
          (min[1] + max[1]) * 0.5,
          (min[2] + max[2]) * 0.5,
        ];
    const center: [number, number, number] = [
      sourceCenter[0],
      sourceCenter[1],
      sourceCenter[2],
    ];
    const size: [number, number, number] = [
      Math.max(0.01, Math.abs(max[0] - min[0])),
      Math.max(0.01, Math.abs(max[1] - min[1])),
      Math.max(0.01, Math.abs(max[2] - min[2])),
    ];
    document.regions.push({
      id: `legacy-region-${index}`,
      name: `Legacy region ${index}`,
      kind: legacyRegionKind(Number(region.regionType)),
      enabled: true,
      center,
      size,
      phaseMask: 0xffffffff,
      tags: ['legacy'],
      metadata: {
        legacyRegionType: Number(region.regionType) || 0,
        zoneLineInfo: region.zoneLineInfo ?? null,
        sourceCoordinateSystem,
        transformContract: 'requiem-y-up-v2',
      },
    });
  }
  return validateShadoWorldAuthoring(document, world);
}

/**
 * Normalizes generated authoring rows to the current source-space placement
 * contract. Rows carrying the superseded mirror marker are transformed once;
 * current rows have no runtime compatibility branch.
 */
export function upgradeShadoWorldAuthoring(
  value: unknown,
  expectedWorld?: string
): ShadoWorldAuthoringDocument {
  const document = cloneShadoWorldAuthoring(
    validateShadoWorldAuthoring(value, expectedWorld)
  );
  for (const prototype of document.objects.prototypes) {
    const legacyModel =
      typeof prototype.metadata.legacyModel === 'string'
        ? prototype.metadata.legacyModel
        : undefined;
    if (!legacyModel) continue;
    prototype.source = catalogSourceForLegacyPrototype(prototype.source, legacyModel);
    prototype.metadata.generatedAsset = 'final.glb.gz';
    prototype.metadata.sourceCoordinateSystem = 'requiem-y-up';
  }
  for (const stamp of document.objects.stamps) {
    if (
      Number.isInteger(stamp.metadata.legacyIndex) &&
      stamp.metadata.positionMirroredAtPreprocess === true
    ) {
      stamp.position[0] = -stamp.position[0];
      stamp.rotationDegrees[1] = -stamp.rotationDegrees[1];
      delete stamp.metadata.positionMirroredAtPreprocess;
      stamp.metadata.transformNormalizedAtPreprocess = true;
      stamp.metadata.sourceCoordinateSystem = 'requiem-y-up';
      stamp.metadata.transformContract = 'requiem-y-up-v2';
    }
  }
  for (const region of document.regions) {
    if (
      region.tags.includes('legacy') &&
      region.metadata.positionMirroredAtPreprocess === true
    ) {
      region.center[0] = -region.center[0];
      delete region.metadata.positionMirroredAtPreprocess;
      region.metadata.sourceCoordinateSystem = 'requiem-y-up';
      region.metadata.transformContract = 'requiem-y-up-v2';
    }
  }
  return validateShadoWorldAuthoring(document, expectedWorld);
}

/**
 * Reconciles newly discovered legacy sidecar rows into an editable authoring
 * document. Existing stamps and regions win so editor changes are never
 * overwritten; newly added metadata rows are picked up on every conversion.
 */
export function mergeLegacyZoneMetadata(
  authoringValue: unknown,
  legacyValue: unknown,
  world: string,
  options: LegacyZoneMetadataImportOptions = {}
): ShadoWorldAuthoringDocument {
  const document = upgradeShadoWorldAuthoring(authoringValue, world);
  const promoted = importLegacyZoneMetadata(legacyValue, world, options);
  const exclusions = new Set(document.legacyObjectExclusions ?? []);
  const prototypeIds = new Set(document.objects.prototypes.map(item => item.id));
  for (const prototype of promoted.objects.prototypes) {
    if (exclusions.has(prototype.id)) continue;
    const existing = document.objects.prototypes.find(item => item.id === prototype.id);
    if (!existing) {
      document.objects.prototypes.push(prototype);
      prototypeIds.add(prototype.id);
      continue;
    }
    // Asset routing is generated state and should follow the latest catalog.
    existing.source = prototype.source;
    existing.metadata = { ...existing.metadata, ...prototype.metadata };
  }
  const stampIds = new Set(document.objects.stamps.map(item => item.id));
  for (const stamp of promoted.objects.stamps) {
    if (
      exclusions.has(stamp.prototype) ||
      stampIds.has(stamp.id) ||
      !prototypeIds.has(stamp.prototype)
    ) continue;
    document.objects.stamps.push(stamp);
    stampIds.add(stamp.id);
  }
  const regionIds = new Set(document.regions.map(item => item.id));
  for (const region of promoted.regions) {
    if (regionIds.has(region.id)) continue;
    document.regions.push(region);
    regionIds.add(region.id);
  }
  return validateShadoWorldAuthoring(document, world);
}

function catalogSourceForLegacyPrototype(source: string, model: string): string {
  const match = source.match(/^(.*\/objects)(?:\/|$)/i);
  const prefix = match?.[1] ?? '/eqrequiem/objects';
  return `${prefix}/${model}/final.glb.gz`;
}

export function cloneShadoWorldAuthoring(
  document: ShadoWorldAuthoringDocument
): ShadoWorldAuthoringDocument {
  return JSON.parse(JSON.stringify(document)) as ShadoWorldAuthoringDocument;
}

export function shadoWorldAuthoringExtras(
  document: ShadoWorldAuthoringDocument
): Record<string, ShadoWorldAuthoringDocument> {
  validateShadoWorldAuthoring(document);
  return { [SHADO_WORLD_AUTHORING_EXTRAS_KEY]: cloneShadoWorldAuthoring(document) };
}

export function authoringFromGltfExtras(
  extras: Record<string, unknown> | undefined,
  expectedWorld?: string
): ShadoWorldAuthoringDocument | undefined {
  const value = extras?.[SHADO_WORLD_AUTHORING_EXTRAS_KEY];
  return value === undefined ? undefined : validateShadoWorldAuthoring(value, expectedWorld);
}

function validateRegion(region: ShadoWorldAuthoringRegion, index: number, ids: Set<string>): void {
  if (!region?.id?.trim() || ids.has(region.id)) {
    throw new Error(`Region ${index} has a missing or duplicate stable ID '${region?.id ?? ''}'`);
  }
  ids.add(region.id);
  if (!REGION_KINDS.has(region.kind)) throw new Error(`Region '${region.id}' has invalid kind '${region.kind}'`);
  if (typeof region.name !== 'string') throw new Error(`Region '${region.id}' requires a name`);
  if (typeof region.enabled !== 'boolean') throw new Error(`Region '${region.id}' requires enabled state`);
  validateVec3(region.center, `Region '${region.id}' center`, false);
  validateVec3(region.size, `Region '${region.id}' size`, true);
  if (!Number.isInteger(region.phaseMask) || region.phaseMask < 0 || region.phaseMask > 0xffffffff) {
    throw new Error(`Region '${region.id}' has an invalid phase mask`);
  }
  if (!Array.isArray(region.tags) || region.tags.some(tag => typeof tag !== 'string')) {
    throw new Error(`Region '${region.id}' tags must be strings`);
  }
  if (!region.metadata || Array.isArray(region.metadata) || typeof region.metadata !== 'object') {
    throw new Error(`Region '${region.id}' metadata must be an object`);
  }
}

function validateObjects(document: ShadoWorldAuthoringDocument): void {
  const prototypeIds = new Set<string>();
  document.objects.prototypes.forEach((prototype, index) => {
    if (!prototype?.id?.trim() || prototypeIds.has(prototype.id)) {
      throw new Error(`Object prototype ${index} has a missing or duplicate stable ID`);
    }
    prototypeIds.add(prototype.id);
    if (!prototype.source?.trim()) {
      throw new Error(`Object prototype '${prototype.id}' requires a source`);
    }
    if (!Number.isFinite(prototype.boundsRadius) || prototype.boundsRadius <= 0) {
      throw new Error(`Object prototype '${prototype.id}' requires a positive bounds radius`);
    }
    if (prototype.light) validatePointLightEmitter(prototype.light, `Object prototype '${prototype.id}' light`);
    if (prototype.audio) validateObjectAudioEmitter(prototype.audio, `Object prototype '${prototype.id}' audio`);
    validateMetadata(prototype.metadata, `Object prototype '${prototype.id}'`);
  });
  const stampIds = new Set<string>();
  document.objects.stamps.forEach((stamp: ShadoWorldObjectStamp, index) => {
    if (!stamp?.id?.trim() || stampIds.has(stamp.id)) {
      throw new Error(`Object stamp ${index} has a missing or duplicate stable ID`);
    }
    stampIds.add(stamp.id);
    if (!prototypeIds.has(stamp.prototype)) {
      throw new Error(`Object stamp '${stamp.id}' references unknown prototype '${stamp.prototype}'`);
    }
    if (typeof stamp.enabled !== 'boolean') {
      throw new Error(`Object stamp '${stamp.id}' requires enabled state`);
    }
    validateVec3(stamp.position, `Object stamp '${stamp.id}' position`, false);
    validateVec3(stamp.rotationDegrees, `Object stamp '${stamp.id}' rotation`, false);
    validateVec3(stamp.scale, `Object stamp '${stamp.id}' scale`, true);
    if (!Number.isInteger(stamp.phaseMask) || stamp.phaseMask < 0 || stamp.phaseMask > 0xffffffff) {
      throw new Error(`Object stamp '${stamp.id}' has an invalid phase mask`);
    }
    if (!Array.isArray(stamp.tags) || stamp.tags.some(tag => typeof tag !== 'string')) {
      throw new Error(`Object stamp '${stamp.id}' tags must be strings`);
    }
    if (stamp.light) validatePointLightEmitter(stamp.light, `Object stamp '${stamp.id}' light`);
    if (stamp.audio) validateObjectAudioEmitter(stamp.audio, `Object stamp '${stamp.id}' audio`);
    validateMetadata(stamp.metadata, `Object stamp '${stamp.id}'`);
  });
}

function validateLighting(document: ShadoWorldAuthoringDocument): void {
  const ids = new Set<string>();
  document.lighting.pointLights.forEach((light, index) => {
    if (!light?.id?.trim() || ids.has(light.id)) {
      throw new Error(`Point light ${index} has a missing or duplicate stable ID`);
    }
    ids.add(light.id);
    if (!light.name?.trim()) throw new Error(`Point light '${light.id}' requires a name`);
    validatePointLightEmitter(light, `Point light '${light.id}'`);
    if (!Number.isInteger(light.phaseMask) || light.phaseMask < 0 || light.phaseMask > 0xffffffff) {
      throw new Error(`Point light '${light.id}' has an invalid phase mask`);
    }
    if (!Array.isArray(light.tags) || light.tags.some(tag => typeof tag !== 'string')) {
      throw new Error(`Point light '${light.id}' tags must be strings`);
    }
  });
}

function validatePointLightEmitter(
  light: NonNullable<ShadoWorldAuthoringDocument['objects']['prototypes'][number]['light']>,
  label: string
): void {
  if (typeof light.enabled !== 'boolean' || typeof light.castsShadows !== 'boolean' || typeof light.bake !== 'boolean' || typeof light.runtime !== 'boolean') {
    throw new Error(`${label} requires enabled, shadow, bake, and runtime states`);
  }
  validateVec3(light.offset, `${label} offset`, false);
  validateFiniteTuple(light.color, 3, `${label} color`, 0, 1);
  if (!Number.isFinite(light.intensity) || light.intensity < 0) throw new Error(`${label} intensity must be non-negative`);
  positive(light.range, `${label} range`);
  if (!Number.isFinite(light.radius) || light.radius < 0) throw new Error(`${label} radius must be non-negative`);
  if (light.activation) {
    if (!['always', 'night', 'schedule'].includes(light.activation.mode)) {
      throw new Error(`${label} activation mode is invalid`);
    }
    for (const [name, value] of Object.entries({ onHour: light.activation.onHour, offHour: light.activation.offHour })) {
      if (!Number.isFinite(value) || value < 0 || value > 24) {
        throw new Error(`${label} activation ${name} must be between 0 and 24`);
      }
    }
    if (!Number.isFinite(light.activation.transitionMinutes) || light.activation.transitionMinutes < 0 || light.activation.transitionMinutes > 180) {
      throw new Error(`${label} activation transitionMinutes must be between 0 and 180`);
    }
  }
  if (light.flicker) {
    if (!['steady', 'flame', 'wisp'].includes(light.flicker.profile)) {
      throw new Error(`${label} flicker profile is invalid`);
    }
    if (!Number.isFinite(light.flicker.amplitude) || light.flicker.amplitude < 0 || light.flicker.amplitude > 0.5) {
      throw new Error(`${label} flicker amplitude must be between 0 and 0.5`);
    }
    if (!Number.isFinite(light.flicker.speed) || light.flicker.speed < 0 || light.flicker.speed > 30) {
      throw new Error(`${label} flicker speed must be between 0 and 30`);
    }
  }
  validateMetadata(light.metadata, label);
}

function validateObjectAudioEmitter(
  emitter: NonNullable<ShadoWorldAuthoringDocument['objects']['prototypes'][number]['audio']>,
  label: string
): void {
  if (typeof emitter.enabled !== 'boolean' || typeof emitter.loop !== 'boolean') {
    throw new Error(`${label} requires enabled and loop states`);
  }
  if (!emitter.source?.trim()) throw new Error(`${label} requires a source`);
  validateVec3(emitter.offset, `${label} offset`, false);
  positive(emitter.range, `${label} range`);
  if (!Number.isFinite(emitter.volume) || emitter.volume < 0) {
    throw new Error(`${label} volume must be non-negative`);
  }
  validateMetadata(emitter.metadata, label);
}

function validateGeometry(document: ShadoWorldAuthoringDocument): void {
  const meshNames = new Set<string>();
  document.geometry.meshes.forEach((override, index) => {
    if (!override?.mesh?.trim() || meshNames.has(override.mesh)) {
      throw new Error(`Geometry override ${index} has a missing or duplicate mesh name`);
    }
    meshNames.add(override.mesh);
    if (typeof override.enabled !== 'boolean' || !['inherit', 'enabled', 'disabled'].includes(override.collision)) {
      throw new Error(`Geometry override '${override.mesh}' has invalid enabled/collision state`);
    }
    validateVec3(override.position, `Geometry override '${override.mesh}' position`, false);
    validateVec3(override.rotationDegrees, `Geometry override '${override.mesh}' rotation`, false);
    validateVec3(override.scale, `Geometry override '${override.mesh}' scale`, true);
    if (override.material !== undefined && !override.material.trim()) {
      throw new Error(`Geometry override '${override.mesh}' material must be non-empty`);
    }
    validateMetadata(override.metadata, `Geometry override '${override.mesh}'`);
  });
  const materialIds = new Set<string>();
  document.geometry.materials.forEach((material, index) => {
    if (!material?.id?.trim() || materialIds.has(material.id)) {
      throw new Error(`Authored material ${index} has a missing or duplicate stable ID`);
    }
    materialIds.add(material.id);
    if (!material.name?.trim() || typeof material.enabled !== 'boolean') {
      throw new Error(`Authored material '${material.id}' requires a name and enabled state`);
    }
    validateFiniteTuple(material.baseColor, 4, `Authored material '${material.id}' baseColor`, 0, 1);
    validateFiniteTuple(material.emissive, 3, `Authored material '${material.id}' emissive`, 0);
    for (const [name, value] of [['metallic', material.metallic], ['roughness', material.roughness], ['alphaCutoff', material.alphaCutoff]] as const) {
      if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`Authored material '${material.id}' ${name} must be between 0 and 1`);
    }
    if (!['opaque', 'mask', 'blend'].includes(material.alphaMode) || typeof material.doubleSided !== 'boolean') {
      throw new Error(`Authored material '${material.id}' has invalid alpha/double-sided state`);
    }
    if (!material.textures || Array.isArray(material.textures) || typeof material.textures !== 'object' || Object.values(material.textures).some(value => typeof value !== 'string' || !value.trim())) {
      throw new Error(`Authored material '${material.id}' textures must be non-empty URL strings`);
    }
    validateMetadata(material.metadata, `Authored material '${material.id}'`);
  });
  for (const override of document.geometry.meshes) {
    if (override.material && materialIds.has(override.material)) continue;
    // Non-authored IDs deliberately remain valid because source-GLB material
    // names are discovered only when the scene asset is loaded.
  }
  const patchIds = new Set<string>();
  document.geometry.patches.forEach((patch, index) => {
    if (!patch?.id?.trim() || patchIds.has(patch.id)) throw new Error(`Geometry patch ${index} has a missing or duplicate stable ID`);
    patchIds.add(patch.id);
    if (!patch.mesh?.trim() || !Number.isInteger(patch.primitive) || patch.primitive < 0 || typeof patch.enabled !== 'boolean') {
      throw new Error(`Geometry patch '${patch.id}' requires a mesh, non-negative primitive, and enabled state`);
    }
    if (patch.sourceHash !== undefined && !/^[a-f0-9]{64}$/i.test(patch.sourceHash)) throw new Error(`Geometry patch '${patch.id}' sourceHash must be SHA-256`);
    if (!Array.isArray(patch.operations) || patch.operations.length === 0) throw new Error(`Geometry patch '${patch.id}' requires at least one operation`);
    patch.operations.forEach((operation, operationIndex) => validateGeometryOperation(operation, `Geometry patch '${patch.id}' operation ${operationIndex}`));
    validateMetadata(patch.metadata, `Geometry patch '${patch.id}'`);
  });
}

function validateGeometrySelection(value: unknown, label: string): void {
  if (value === 'all') return;
  if (!Array.isArray(value) || value.some(index => !Number.isInteger(index) || index < 0) || new Set(value).size !== value.length) {
    throw new Error(`${label} must be 'all' or unique non-negative indices`);
  }
}

function validateGeometryOperation(operation: ShadoWorldAuthoringDocument['geometry']['patches'][number]['operations'][number], label: string): void {
  if (!operation || typeof operation !== 'object') throw new Error(`${label} must be an object`);
  if (operation.kind === 'transform-vertices') {
    validateGeometrySelection(operation.vertices, `${label} vertices`);
    validateVec3(operation.translation, `${label} translation`, false);
    validateVec3(operation.rotationDegrees, `${label} rotation`, false);
    validateVec3(operation.scale, `${label} scale`, true);
    return;
  }
  if (operation.kind === 'delete-triangles' || operation.kind === 'flip-triangles') {
    validateGeometrySelection(operation.triangles, `${label} triangles`);
    return;
  }
  if (operation.kind === 'extrude-triangles') {
    validateGeometrySelection(operation.triangles, `${label} triangles`);
    if (!Number.isFinite(operation.distance) || operation.distance === 0) throw new Error(`${label} distance must be finite and non-zero`);
    if (operation.direction !== 'normal') validateVec3(operation.direction, `${label} direction`, false);
    if (typeof operation.cap !== 'boolean') throw new Error(`${label} cap must be boolean`);
    return;
  }
  if (operation.kind === 'recalculate-normals') return;
  if (operation.kind === 'project-uv') {
    if ((operation.uvSet !== 0 && operation.uvSet !== 1) || !['xy', 'xz', 'yz'].includes(operation.plane)) throw new Error(`${label} has an invalid UV target`);
    validateFiniteTuple(operation.scale, 2, `${label} scale`);
    validateFiniteTuple(operation.offset, 2, `${label} offset`);
    if (operation.scale.some(value => value === 0)) throw new Error(`${label} scale components must be non-zero`);
    return;
  }
  if (operation.kind === 'assign-material') {
    validateGeometrySelection(operation.triangles, `${label} triangles`);
    if (!operation.material?.trim()) throw new Error(`${label} material must be non-empty`);
    return;
  }
  throw new Error(`${label} has an unsupported kind`);
}

function validateFiniteTuple(value: unknown, length: number, label: string, minimum = -Number.MAX_VALUE, maximum = Number.MAX_VALUE): void {
  if (!Array.isArray(value) || value.length !== length || value.some(component => !Number.isFinite(component) || component < minimum || component > maximum)) {
    throw new Error(`${label} must be a ${length}-number tuple`);
  }
}

function validateMetadata(value: unknown, label: string): void {
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    throw new Error(`${label} metadata must be an object`);
  }
}

function stableId(value: string, fallback: string): string {
  return value.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-|-$/g, '') || fallback;
}

function finite(value: unknown, fallback = 0): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function positive(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be positive`);
  return value;
}

function vec3(value: ArrayLike<unknown> | undefined): [number, number, number] {
  return [finite(value?.[0]), finite(value?.[1]), finite(value?.[2])];
}

function legacyRegionKind(type: number): ShadoWorldRegionKind {
  if (type === 1 || type === 5 || type === 6) return 'water';
  if (type === 2) return 'lava';
  if (type === 4) return 'zone-line';
  return 'semantic';
}

function validateVec3(value: unknown, label: string, positive: boolean): void {
  if (
    !Array.isArray(value) || value.length !== 3 ||
    value.some(component => !Number.isFinite(component) || (positive && component <= 0))
  ) {
    throw new Error(`${label} must be a finite${positive ? ' positive' : ''} vec3`);
  }
}
