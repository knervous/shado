import {
  SHADO_WORLD_AUTHORING_EXTRAS_KEY,
  type ShadoWorldAuthoringDocument,
  type ShadoWorldObjectStamp,
  type ShadoWorldAuthoringRegion,
  type ShadoWorldRegionKind,
  type ShadoWorldEnvironmentAuthoring,
  type ShadoWorldZoneAmbience,
  type ShadoWorldRegionAmbience,
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
  waterBodies: [],
  audioEmitters: [],
  reflectionProbes: [],
  mediaVolumes: [],
  particleEmitters: [],
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
  // The old zone-wide water switch drove nothing at runtime; water is authored
  // as bodies now. Dropped on load so documents shed it on their next save.
  delete (document.environment as { water?: unknown }).water;
  document.environment.waterBodies ??= [];
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
  validateVisibility(document);
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
  validateZoneAmbience(environment.ambience);
  if (environment.sky.texture !== undefined && !environment.sky.texture.trim()) throw new Error('World sky texture must be a non-empty URL');
  for (const [name, value] of Object.entries({ skyIntensity: environment.sky.intensity, fogDensity: environment.fog.density, fogStart: environment.fog.start, fogEnd: environment.fog.end, ambientIntensity: environment.ambient.intensity, weatherIntensity: environment.weather.intensity, hour: environment.timeOfDay.hour, cycleSeconds: environment.timeOfDay.cycleSeconds })) if (!Number.isFinite(value)) throw new Error(`World environment ${name} must be finite`);
  const ids = new Set<string>();
  for (const emitter of environment.audioEmitters) { if (!emitter.id?.trim() || ids.has(emitter.id) || !emitter.source?.trim()) throw new Error('World audio emitters require unique IDs and sources'); validateComponentSource(emitter.source, `Audio emitter '${emitter.id}'`); ids.add(emitter.id); validateVec3(emitter.position, `Audio emitter '${emitter.id}' position`, false); positive(emitter.range, `Audio emitter '${emitter.id}' range`); if (!Number.isFinite(emitter.volume) || emitter.volume < 0) throw new Error(`Audio emitter '${emitter.id}' volume must be non-negative`); validateMetadata(emitter.metadata, `Audio emitter '${emitter.id}'`); }
  ids.clear();
  validateMediaVolumes(environment);
  validateWaterBodies(document);
  validateVolumetricMedium(environment);
  validateParticleEmitters(environment);
  ids.clear();
  for (const probe of environment.reflectionProbes) { if (!probe.id?.trim() || ids.has(probe.id)) throw new Error('World reflection probes require unique IDs'); ids.add(probe.id); validateVec3(probe.position, `Reflection probe '${probe.id}' position`, false); validateVec3(probe.size, `Reflection probe '${probe.id}' size`, true); positive(probe.resolution, `Reflection probe '${probe.id}' resolution`); validateMetadata(probe.metadata, `Reflection probe '${probe.id}'`); }
}

/** Lakes and rivers: well-formed shapes, and each owning a water region. */
function validateWaterBodies(document: ShadoWorldAuthoringDocument): void {
  const ids = new Set<string>();
  const owners = new Map<string, string>();
  for (const body of document.environment.waterBodies ?? []) {
    const label = `Water body '${body?.id}'`;
    if (!body?.id?.trim() || ids.has(body.id)) throw new Error('World water bodies require unique IDs');
    ids.add(body.id);
    if (body.kind !== 'lake' && body.kind !== 'river') throw new Error(`${label} kind must be lake or river`);
    for (const [name, value] of Object.entries({ level: body.level, flowSpeed: body.flowSpeed, foam: body.foam, depth: body.depth })) {
      if (!Number.isFinite(value)) throw new Error(`${label} ${name} must be finite`);
    }
    if (body.depth <= 0) throw new Error(`${label} depth must be positive`);
    if (body.kind === 'lake') {
      if (!Array.isArray(body.outline) || body.outline.length < 3 || body.outline.some(point => !Array.isArray(point) || !Number.isFinite(point[0]) || !Number.isFinite(point[1]))) {
        throw new Error(`${label} needs an outline of at least three finite points`);
      }
    } else if (!Array.isArray(body.path) || body.path.length < 2 || body.path.some(point => ![point?.x, point?.z, point?.level].every(Number.isFinite) || !(point.width > 0))) {
      throw new Error(`${label} needs a path of at least two points with positive widths`);
    }
    const region = document.regions.find(entry => entry.id === body.regionId);
    if (!region || region.kind !== 'water') throw new Error(`${label} must own a water region; '${body.regionId}' is ${region ? `a ${region.kind} region` : 'missing'}`);
    if (owners.has(body.regionId)) throw new Error(`${label} and '${owners.get(body.regionId)}' cannot share region '${body.regionId}'`);
    owners.set(body.regionId, body.id);
    validateMetadata(body.metadata, label);
  }
}

/**
 * Authored fog banks.
 *
 * Validated here rather than trusted at the runtime, because a volume with a
 * non-finite density does not fail loudly on the GPU -- it makes the froxel
 * grid NaN and the world renders black with no error anywhere.
 */
function validateMediaVolumes(environment: ShadoWorldEnvironmentAuthoring): void {
  const volumes = environment.mediaVolumes;
  if (volumes === undefined) return;
  if (!Array.isArray(volumes)) throw new Error('World media volumes must be an array');
  const seen = new Set<string>();
  for (const volume of volumes) {
    if (!volume?.id?.trim() || seen.has(volume.id)) throw new Error('World media volumes require unique IDs');
    seen.add(volume.id);
    if (volume.shape !== 'box' && volume.shape !== 'sphere') throw new Error(`Media volume '${volume.id}' shape must be box or sphere`);
    validateVec3(volume.position, `Media volume '${volume.id}' position`, false);
    validateVec3(volume.size, `Media volume '${volume.id}' size`, true);
    if (volume.color !== undefined) validateVec3(volume.color, `Media volume '${volume.id}' color`, false);
    if (!Number.isFinite(volume.density) || volume.density < 0) throw new Error(`Media volume '${volume.id}' density must be non-negative`);
    for (const [name, value] of Object.entries({
      yaw: volume.yaw, albedo: volume.albedo, anisotropy: volume.anisotropy,
      heightFalloff: volume.heightFalloff, heightReference: volume.heightReference,
      feather: volume.feather, priority: volume.priority, hoursFeather: volume.hoursFeather,
    })) {
      if (value !== undefined && !Number.isFinite(value)) throw new Error(`Media volume '${volume.id}' ${name} must be finite`);
    }
    if (volume.albedo !== undefined && (volume.albedo < 0 || volume.albedo > 1)) throw new Error(`Media volume '${volume.id}' albedo must be within 0..1`);
    if (volume.anisotropy !== undefined && Math.abs(volume.anisotropy) >= 1) throw new Error(`Media volume '${volume.id}' anisotropy must be within -1..1`);
    if (volume.feather !== undefined && volume.feather < 0) throw new Error(`Media volume '${volume.id}' feather must be non-negative`);
    if (volume.hours !== undefined) {
      if (!Array.isArray(volume.hours) || volume.hours.length !== 2 || volume.hours.some(hour => !Number.isFinite(hour) || hour < 0 || hour > 24)) {
        throw new Error(`Media volume '${volume.id}' hours must be two hours within 0..24`);
      }
    }
    if (volume.metadata !== undefined) validateMetadata(volume.metadata, `Media volume '${volume.id}'`);
  }
}

/**
 * The zone's global medium.
 *
 * Bounded rather than merely finite, because these are the values a tuning
 * UI writes: a slider that can reach an albedo of 400 or a negative density
 * is a slider that can black out a zone, and the failure arrives as a NaN in
 * a compute pass with no name on it.
 */
function validateVolumetricMedium(environment: ShadoWorldEnvironmentAuthoring): void {
  const medium = environment.volumetric;
  if (medium === undefined) return;
  if (!medium || typeof medium !== 'object' || Array.isArray(medium)) throw new Error('World volumetric medium must be an object');
  const range = (name: keyof typeof medium, low: number, high: number): void => {
    const value = medium[name];
    if (value === undefined) return;
    if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) {
      throw new Error(`World volumetric ${String(name)} must be a number within ${low}..${high}`);
    }
  };
  if (medium.enabled !== undefined && typeof medium.enabled !== 'boolean') throw new Error('World volumetric enabled must be a boolean');
  range('density', 0, 1);
  range('heightFalloff', 0, 1);
  range('heightReference', -100_000, 100_000);
  range('albedo', 0, 1);
  range('anisotropy', -0.95, 0.95);
  range('noiseAmount', 0, 1);
  range('noiseScale', 0, 1);
  range('windSpeed', 0, 1_000);
  range('ambientMultiplier', 0, 16);
  range('near', 0.01, 1_000);
  range('far', 1, 20_000);
  range('depthPower', 1, 8);
  range('temporalBlend', 0.01, 1);
  range('strength', 0, 1);
  if (medium.near !== undefined && medium.far !== undefined && medium.far <= medium.near) {
    throw new Error('World volumetric far must be beyond near');
  }
}

/**
 * Ambient particle emitters.
 *
 * The caps are the point. A particle system is the one authored thing that
 * costs frame time in proportion to a number someone typed, and the numbers
 * that hurt -- capacity and emit rate -- are exactly the ones that look
 * harmless in a field. Bounded here so a zone cannot be published with a
 * hundred thousand live particles in it, and bounded again at the runtime,
 * which is the half that protects a player from a zone already shipped.
 */
function validateParticleEmitters(environment: ShadoWorldEnvironmentAuthoring): void {
  const emitters = environment.particleEmitters;
  if (emitters === undefined) return;
  if (!Array.isArray(emitters)) throw new Error('World particle emitters must be an array');
  const seen = new Set<string>();
  for (const emitter of emitters) {
    if (!emitter?.id?.trim() || seen.has(emitter.id)) throw new Error('World particle emitters require unique IDs');
    seen.add(emitter.id);
    const label = `Particle emitter '${emitter.id}'`;
    validateVec3(emitter.position, `${label} position`, false);
    validateVec3(emitter.size, `${label} size`, false);
    validateVec3(emitter.direction1, `${label} direction1`, false);
    validateVec3(emitter.direction2, `${label} direction2`, false);
    if (emitter.gravity !== undefined) validateVec3(emitter.gravity, `${label} gravity`, false);
    for (const channel of ['color1', 'color2', 'colorDead'] as const) {
      const value = emitter[channel];
      if (!Array.isArray(value) || value.length !== 4 || value.some(component => !Number.isFinite(component) || component < 0 || component > 1)) {
        throw new Error(`${label} ${channel} must be four channels within 0..1`);
      }
    }
    const bounded = (name: string, value: unknown, low: number, high: number): void => {
      if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) {
        throw new Error(`${label} ${name} must be a number within ${low}..${high}`);
      }
    };
    bounded('capacity', emitter.capacity, 1, 20_000);
    bounded('emitRate', emitter.emitRate, 0, 10_000);
    bounded('minSize', emitter.minSize, 0, 1_000);
    bounded('maxSize', emitter.maxSize, 0, 1_000);
    bounded('minLifeTime', emitter.minLifeTime, 0.01, 600);
    bounded('maxLifeTime', emitter.maxLifeTime, 0.01, 600);
    bounded('minEmitPower', emitter.minEmitPower, -1_000, 1_000);
    bounded('maxEmitPower', emitter.maxEmitPower, -1_000, 1_000);
    bounded('range', emitter.range, 1, 20_000);
    if (emitter.updateSpeed !== undefined) bounded('updateSpeed', emitter.updateSpeed, 0.0001, 1);
    if (emitter.maxSize < emitter.minSize) throw new Error(`${label} maxSize must be at least minSize`);
    if (emitter.maxLifeTime < emitter.minLifeTime) throw new Error(`${label} maxLifeTime must be at least minLifeTime`);
    if (emitter.blendMode !== undefined && emitter.blendMode !== 'add' && emitter.blendMode !== 'standard') {
      throw new Error(`${label} blendMode must be add or standard`);
    }
    if (emitter.texture !== undefined && !emitter.texture.trim()) throw new Error(`${label} texture must be a non-empty URL`);
    for (const name of ['angularSpeed', 'initialRotation'] as const) {
      const pair = emitter[name];
      if (pair === undefined) continue;
      if (!Array.isArray(pair) || pair.length !== 2 || pair.some(v => !Number.isFinite(v) || Math.abs(v) > 100) || pair[1] < pair[0]) {
        throw new Error(`${label} ${name} must be [min, max] radians within +/-100, min <= max`);
      }
    }
    if (emitter.hours !== undefined) {
      if (!Array.isArray(emitter.hours) || emitter.hours.length !== 2 || emitter.hours.some(hour => !Number.isFinite(hour) || hour < 0 || hour > 24)) {
        throw new Error(`${label} hours must be two hours within 0..24`);
      }
    }
    if (emitter.metadata !== undefined) validateMetadata(emitter.metadata, label);
  }
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

function validateVisibility(document: ShadoWorldAuthoringDocument): void {
  const disocclusion = document.visibility?.disocclusion;
  if (!disocclusion) return;
  if (!Array.isArray(disocclusion.sources) || !disocclusion.sources.length) {
    throw new Error('Disocclusion visibility requires at least one source volume');
  }
  const ids = new Set<string>();
  for (const source of disocclusion.sources) {
    if (!source.id?.trim() || ids.has(source.id)) throw new Error('Disocclusion sources require unique IDs');
    ids.add(source.id);
    validateVec3(source.min, `Disocclusion source '${source.id}' min`, false);
    validateVec3(source.max, `Disocclusion source '${source.id}' max`, false);
    for (let axis = 0; axis < 3; axis++) {
      if (!(source.max[axis]! > source.min[axis]!)) throw new Error(`Disocclusion source '${source.id}' box is empty`);
    }
    positive(source.near, `Disocclusion source '${source.id}' near`);
    if (!(source.far > source.near)) throw new Error(`Disocclusion source '${source.id}' far must exceed near`);
    if (source.sideUp !== undefined) positive(source.sideUp, `Disocclusion source '${source.id}' sideUp`);
  }
  for (const [name, value] of Object.entries(disocclusion.settings ?? {})) {
    positive(value, `Disocclusion setting '${name}'`);
  }
  for (const pose of disocclusion.poses ?? []) {
    if (!pose.name?.trim()) throw new Error('Disocclusion review poses require names');
    validateVec3(pose.at, `Disocclusion pose '${pose.name}' at`, false);
    validateVec3(pose.look, `Disocclusion pose '${pose.name}' look`, false);
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
    // A protrusion moves the ground the player walks on, so it is checked here
    // as well as at package validation: an author saving an impossible lift
    // should hear about it before a bake spends twenty minutes on it.
    if (layer.protrusionMetres !== undefined && (!Number.isFinite(layer.protrusionMetres) || layer.protrusionMetres < 0 || layer.protrusionMetres > 8)) {
      throw new Error(`Terrain layer '${layer.id}' protrusion must be 0 to 8 metres`);
    }
    if (layer.protrusionFalloffMetres !== undefined && (!Number.isFinite(layer.protrusionFalloffMetres) || layer.protrusionFalloffMetres <= 0)) {
      throw new Error(`Terrain layer '${layer.id}' protrusion falloff must be a positive width in metres`);
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
  validateRegionAmbience(region);
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

/**
 * An audio source names a component, `family/element`, not a file.
 *
 * Checked at authoring time because the failure mode otherwise is silence: a
 * source the bank cannot resolve returns no buffer, the layer never starts, and
 * nothing anywhere reports it -- the zone is simply quiet, which looks exactly
 * like an ambience system that does not work.
 */
function validateComponentSource(source: string, label: string): void {
  if (/^[a-z][a-z0-9-]*\/[a-z][a-z0-9_=,-]*$/.test(source)) return;
  // Emitters authored before the library was addressable by component name point
  // at a promoted file. Those are accepted rather than rejected, because the
  // alternative is that every zone holding one cannot be *opened* for authoring
  // -- the validator runs on load -- and the runtime recovers the component from
  // the filename anyway. New authoring should not produce this form.
  if (/(?:^|\/)[a-z][a-z0-9-]*-[a-z_]+-\d+\.[a-z0-9]+$/.test(source)) return;
  throw new Error(
    `${label} source must be a component reference like 'env-water/river', not '${source}'`
  );
}

/**
 * A region's ambience override, which lives in its metadata.
 *
 * Validated here even though metadata is otherwise opaque, for the same reason
 * the zone's is: an unresolvable source makes no sound and reports nothing, so
 * the only place a typo can be caught is where it is written. A region that is
 * quieter than it should be is a bug someone might notice; a region that is
 * silent is a region that looks like it was never authored.
 */
function validateRegionAmbience(region: ShadoWorldAuthoringRegion): void {
  const ambience = region.metadata.ambience as ShadoWorldRegionAmbience | undefined;
  if (ambience === undefined) return;
  if (!ambience || typeof ambience !== 'object' || Array.isArray(ambience)) {
    throw new Error(`Region '${region.id}' ambience must be an object`);
  }
  const label = `Region '${region.id}' ambience`;
  if (ambience.bed) validateComponentSource(ambience.bed, `${label} bed`);
  if (ambience.bedNight) validateComponentSource(ambience.bedNight, `${label} night bed`);
  for (const loop of ambience.loops ?? []) validateComponentSource(loop, `${label} loop`);
  for (const oneshot of ambience.oneshots ?? []) {
    validateComponentSource(oneshot.source, `${label} one-shot`);
    if (!Number.isFinite(oneshot.perMinute) || oneshot.perMinute <= 0) {
      throw new Error(`${label} one-shot '${oneshot.source}' needs a positive perMinute`);
    }
  }
  for (const [name, value] of Object.entries({
    fadeSeconds: ambience.fadeSeconds,
    gain: ambience.gain,
    priority: ambience.priority,
  })) {
    if (value !== undefined && (!Number.isFinite(value) || (value as number) < 0)) {
      throw new Error(`${label} ${name} must be a non-negative number`);
    }
  }
  if (ambience.interior !== undefined && typeof ambience.interior !== 'boolean') {
    throw new Error(`${label} interior must be a boolean`);
  }
  // A region that declares nothing is almost certainly a half-finished edit, and
  // it would silently behave as though it had no ambience at all.
  const declares =
    ambience.bed ||
    ambience.bedNight ||
    ambience.loops?.length ||
    ambience.oneshots?.length ||
    ambience.interior !== undefined ||
    ambience.space;
  if (!declares) {
    throw new Error(`${label} declares nothing; remove it or give it a layer`);
  }
}

/** Doc §32's layers. Validated together because they share the source format. */
function validateZoneAmbience(ambience: ShadoWorldZoneAmbience | undefined): void {
  if (!ambience) return;
  if (ambience.bed) validateComponentSource(ambience.bed, 'Zone ambience bed');
  if (ambience.bedNight) validateComponentSource(ambience.bedNight, 'Zone ambience night bed');
  for (const loop of ambience.loops ?? []) validateComponentSource(loop, 'Zone ambience loop');
  for (const oneshot of ambience.oneshots ?? []) {
    validateComponentSource(oneshot.source, 'Zone ambience one-shot');
    if (!Number.isFinite(oneshot.perMinute) || oneshot.perMinute <= 0) {
      throw new Error(`Zone ambience one-shot '${oneshot.source}' needs a positive perMinute`);
    }
    if (oneshot.hours) {
      for (const hour of oneshot.hours) {
        if (!Number.isFinite(hour) || hour < 0 || hour >= 24) {
          throw new Error(`Zone ambience one-shot '${oneshot.source}' hours must be 0-24`);
        }
      }
    }
  }
  if (ambience.gain !== undefined && (!Number.isFinite(ambience.gain) || ambience.gain < 0)) {
    throw new Error('Zone ambience gain must be non-negative');
  }
}

function validateVec3(value: unknown, label: string, positive: boolean): void {
  if (
    !Array.isArray(value) || value.length !== 3 ||
    value.some(component => !Number.isFinite(component) || (positive && component <= 0))
  ) {
    throw new Error(`${label} must be a finite${positive ? ' positive' : ''} vec3`);
  }
}
