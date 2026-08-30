import type { EltaniaTerrainSurfaceSpec } from './terrain-compile';

export type WorldVec3 = [number, number, number];

export const SHADO_WORLD_AUTHORING_EXTRAS_KEY = 'EXT_shado_world_authoring';

export type ShadoWorldRegionKind =
  | 'visibility-cell'
  | 'streaming'
  | 'water'
  | 'lava'
  | 'safe'
  | 'pvp'
  | 'zone-line'
  | 'audio'
  | 'trigger'
  | 'fx'
  | 'semantic';

export type ShadoWorldTerrainProjection = 'world-xz' | 'triplanar' | 'hybrid';

/** Editable terrain-layer intent shared by Blender, Libra, and the runtime material builder. */
export type ShadoWorldTerrainLayer = {
  id: string;
  name: string;
  enabled: boolean;
  /** Published material/texture-array key. */
  material: string;
  projection: ShadoWorldTerrainProjection;
  /** World units covered by one material tile. */
  textureScale: number;
  /** Base contribution before slope, altitude, noise, and control-map rules. */
  weight: number;
  /** Normalized steepness transition, where 0 is flat and 1 is vertical. */
  slope: [number, number];
  /** Inclusive world-space altitude range. */
  altitude: [number, number];
  /** World-space macro noise scale; 0 disables noise modulation. */
  noiseScale: number;
  /** Optional author-control channel such as `control0.r` or `control1.b`. */
  control?: string;
  metadata: Record<string, unknown>;
};

/**
 * Zone-wide terrain tuning: the dials that belong to the place rather than to
 * any one layer. `biomeTint` in particular is where a zone's mood lives, which
 * is what lets the shared ground palette stay near-neutral and still be used by
 * a cold marsh and a dry highland without either looking borrowed.
 */
export type ShadoWorldTerrainSettings = {
  /** Slope range, in degrees, over which hybrid layers cross to triplanar. */
  triplanarDegrees?: [number, number];
  /** How decisively the locally taller layer wins where two overlap. */
  heightBlendSharpness?: number;
  /** World size of the shared macro colour variation, in metres. */
  macroMetres?: number;
  macroStrength?: number;
  biomeTint?: WorldVec3;
};

export type ShadoWorldTerrainMaterialAuthoring = {
  enabled: boolean;
  controlMaps: string[];
  layers: ShadoWorldTerrainLayer[];
  settings?: ShadoWorldTerrainSettings;
};

/** Settings consumed by the headless world compiler unless a caller explicitly overrides them. */
export type ShadoWorldBakeSettings = {
  tileSize: number;
  maxClusterTriangles: number;
  minRenderChunkTriangles: number;
  maxRenderChunkExtent: number;
  visibilityRegionSize: number;
  visibilityMaxDistance: number;
  physicsChunkSize: number;
};

export type ShadoWorldFxCullProfile = 'near-detail' | 'mid-atmosphere' | 'far-landmark' | 'always';

export type ShadoWorldFxPattern = {
  version: 1;
  /** Runtime factory key such as `grass`, `light-rays`, or `wind-volume`. */
  effect: string;
  placement: 'point' | 'volume' | 'surface';
  culling: {
    profile: ShadoWorldFxCullProfile;
    /** Optional profile override in final Babylon world units. */
    maxDistance?: number;
    /** Width of the shader/LOD transition before hard culling. */
    fadeDistance?: number;
    /** Optional reducer cadence override. */
    updateHz?: number;
    outsideWorldVisible?: boolean;
  };
  budget?: {
    qualityTier?: 'low' | 'medium' | 'high' | 'ultra';
    maximumInstances?: number;
    maximumDraws?: number;
  };
  /** Effect-owned payload kept off the reducer's hot SoA planes. */
  parameters?: Record<string, unknown>;
};

export type ShadoWorldAuthoringRegion = {
  /** Durable identity used by scripts, diffs, and replacement operations. */
  id: string;
  name: string;
  kind: ShadoWorldRegionKind;
  enabled: boolean;
  center: WorldVec3;
  size: WorldVec3;
  phaseMask: number;
  tags: string[];
  /** Tool/game-specific payload deliberately kept outside hot reducer planes. */
  metadata: Record<string, unknown>;
};

export type ShadoWorldObjectPrototype = {
  /** Stable model key used to batch stamped instances into one draw source. */
  id: string;
  /** Runtime asset URL. The client loads this once and creates visible instances from it. */
  source: string;
  /** Conservative unscaled sphere radius used before the model asset is resident. */
  boundsRadius: number;
  /** Optional local-space emitter inherited by every stamp of this prototype. */
  light?: ShadoWorldPointLightEmitter;
  /** Optional local-space spatial audio inherited by every stamp of this prototype. */
  audio?: ShadoWorldAudioEmitter;
  metadata: Record<string, unknown>;
};

export type ShadoWorldObjectStamp = {
  /** Durable identity retained across editor operations and migration reruns. */
  id: string;
  prototype: string;
  enabled: boolean;
  position: WorldVec3;
  /**
   * Babylon Y-X-Z Euler degrees. Legacy coordinate conversion happens during
   * preprocessing; renderers and clients must not invert or swap these axes.
   */
  rotationDegrees: WorldVec3;
  scale: WorldVec3;
  phaseMask: number;
  tags: string[];
  /** Optional per-stamp emitter override. `enabled: false` disables an inherited prototype light. */
  light?: ShadoWorldPointLightEmitter;
  /** Optional per-stamp audio override. `enabled: false` disables inherited prototype audio. */
  audio?: ShadoWorldAudioEmitter;
  metadata: Record<string, unknown>;
};

export type ShadoWorldAuthoringObjects = {
  prototypes: ShadoWorldObjectPrototype[];
  stamps: ShadoWorldObjectStamp[];
};

/** Bake-light intent shared by the authoring preview and offline baker. */
export type ShadoWorldLightActivation = {
  /** `night` is the dusk-to-dawn outdoor default; `always` suits interiors. */
  mode: 'always' | 'night' | 'schedule';
  /** Start of a custom schedule in world-clock hours. */
  onHour: number;
  /** End of a custom schedule in world-clock hours. May wrap across midnight. */
  offHour: number;
  /** Soft switch duration at both ends of the schedule. */
  transitionMinutes: number;
};

export type ShadoWorldLightFlicker = {
  profile: 'steady' | 'flame' | 'wisp';
  /** Fractional intensity variation, normally 0.04-0.15. */
  amplitude: number;
  /** Approximate modulation frequency in cycles per second. */
  speed: number;
};

export type ShadoWorldPointLightEmitter = {
  enabled: boolean;
  /** Local offset for object emitters; standalone lights use this as their authored position. */
  offset: WorldVec3;
  /** Linear RGB color. */
  color: WorldVec3;
  intensity: number;
  range: number;
  radius: number;
  castsShadows: boolean;
  /** Include this source in the deterministic offline lighting plan. */
  bake: boolean;
  /**
   * Keep this source in the mutable Shado runtime-light plane. Runtime lights
   * are reduced by PVS/frustum/range before shaders see their compact indices;
   * this is independent of whether the same source contributes to a bake.
   */
  runtime: boolean;
  /** Runtime operating hours. Omitted legacy flame lights default to night. */
  activation?: ShadoWorldLightActivation;
  /** Smooth deterministic runtime variation; no per-light Babylon object. */
  flicker?: ShadoWorldLightFlicker;
  metadata: Record<string, unknown>;
};

export type ShadoWorldPointLightAuthoring = ShadoWorldPointLightEmitter & {
  id: string;
  name: string;
  phaseMask: number;
  tags: string[];
};

export type ShadoWorldLightingAuthoring = {
  pointLights: ShadoWorldPointLightAuthoring[];
};

/** Spatial-audio intent attached to an object type or overridden by one stamp. */
export type ShadoWorldAudioEmitter = {
  enabled: boolean;
  source: string;
  /** Object-local position transformed by the owning stamp. */
  offset: WorldVec3;
  range: number;
  volume: number;
  loop: boolean;
  metadata: Record<string, unknown>;
};

/** Fully resolved spatial audio used by editor preview and runtime publication. */
export type ShadoWorldCompiledAudioEmitter = {
  id: string;
  sourceKind: 'standalone' | 'object';
  ownerStamp?: string;
  enabled: boolean;
  source: string;
  position: WorldVec3;
  range: number;
  volume: number;
  loop: boolean;
  metadata: Record<string, unknown>;
};

/** Fully resolved world-space point light emitted by compilation. */
export type ShadoWorldCompiledPointLight = {
  id: string;
  name: string;
  source: 'standalone' | 'object';
  ownerStamp?: string;
  enabled: boolean;
  position: WorldVec3;
  color: WorldVec3;
  intensity: number;
  range: number;
  radius: number;
  castsShadows: boolean;
  bake: boolean;
  runtime: boolean;
  activation?: ShadoWorldLightActivation;
  flicker?: ShadoWorldLightFlicker;
  /** Stable render-cell ownership, or -1 outside packaged geometry. */
  cellId: number;
  /** Dense visibility-region ownership, or -1 outside the visibility grid. */
  visibilityRegion: number;
  phaseMask: number;
  tags: string[];
  metadata: Record<string, unknown>;
};

export type ShadoWorldGeometryMeshOverride = {
  /** Stable Babylon/glTF mesh name in the published base scene. */
  mesh: string;
  enabled: boolean;
  /** Absolute local transform in the base scene's Babylon Y-up hierarchy. */
  position: WorldVec3;
  rotationDegrees: WorldVec3;
  scale: WorldVec3;
  /** Authored material ID, or an existing material name from the source GLB. */
  material?: string;
  collision: 'inherit' | 'enabled' | 'disabled';
  metadata: Record<string, unknown>;
};

/** A compact, reproducible selection in one glTF triangle primitive. */
export type ShadoWorldGeometryElementSelection = 'all' | number[];

/**
 * Ordered mesh operations compiled by Libra against the immutable source GLB.
 * Indices refer to the primitive state produced by the preceding operation.
 */
export type ShadoWorldGeometryEditOperation =
  | {
      kind: 'transform-vertices';
      vertices: ShadoWorldGeometryElementSelection;
      translation: WorldVec3;
      rotationDegrees: WorldVec3;
      scale: WorldVec3;
    }
  | { kind: 'delete-triangles'; triangles: ShadoWorldGeometryElementSelection }
  | { kind: 'flip-triangles'; triangles: ShadoWorldGeometryElementSelection }
  | {
      kind: 'extrude-triangles';
      triangles: ShadoWorldGeometryElementSelection;
      distance: number;
      direction: 'normal' | WorldVec3;
      cap: boolean;
    }
  | { kind: 'recalculate-normals' }
  | {
      kind: 'project-uv';
      uvSet: 0 | 1;
      plane: 'xy' | 'xz' | 'yz';
      scale: [number, number];
      offset: [number, number];
    }
  | {
      kind: 'assign-material';
      triangles: ShadoWorldGeometryElementSelection;
      material: string;
    };

/** Non-destructive topology/material journal targeting one stable source primitive. */
export type ShadoWorldGeometryPatch = {
  id: string;
  mesh: string;
  primitive: number;
  enabled: boolean;
  /** Optional guard produced by mesh inspection; compilation fails if the source changed. */
  sourceHash?: string;
  operations: ShadoWorldGeometryEditOperation[];
  metadata: Record<string, unknown>;
};

export type ShadoWorldMaterialTextureSet = {
  albedo?: string;
  normal?: string;
  metallicRoughness?: string;
  emissive?: string;
  occlusion?: string;
};

/** PBR material authored independently of Blender and usable by mesh overrides. */
export type ShadoWorldMaterialAuthoring = {
  id: string;
  name: string;
  enabled: boolean;
  baseColor: [number, number, number, number];
  metallic: number;
  roughness: number;
  emissive: WorldVec3;
  alphaMode: 'opaque' | 'mask' | 'blend';
  alphaCutoff: number;
  doubleSided: boolean;
  textures: ShadoWorldMaterialTextureSet;
  metadata: Record<string, unknown>;
};

export type ShadoWorldGeometryAuthoring = {
  meshes: ShadoWorldGeometryMeshOverride[];
  materials: ShadoWorldMaterialAuthoring[];
  patches: ShadoWorldGeometryPatch[];
};

export type ShadoWorldEnvironmentAuthoring = {
  sky: {
    mode: 'solid' | 'procedural' | 'texture';
    color: WorldVec3;
    texture?: string;
    intensity: number;
  };
  fog: {
    enabled: boolean;
    mode: 'linear' | 'exponential' | 'exponential-squared';
    color: WorldVec3;
    density: number;
    start: number;
    end: number;
  };
  ambient: { color: WorldVec3; intensity: number };
  weather: { preset: string; intensity: number; wind: WorldVec3 };
  timeOfDay: { hour: number; cycleSeconds: number; running: boolean };
  water: { enabled: boolean; level: number; material?: string; reflections: boolean };
  audioEmitters: Array<{ id: string; source: string; position: WorldVec3; range: number; volume: number; loop: boolean; metadata: Record<string, unknown> }>;
  reflectionProbes: Array<{ id: string; position: WorldVec3; size: WorldVec3; resolution: number; refresh: 'bake' | 'once' | 'runtime'; metadata: Record<string, unknown> }>;
};

export type ShadoWorldPerformanceBudgets = {
  maxVisibleTriangles: number;
  maxDrawCalls: number;
  maxMaterials: number;
  maxTextures: number;
  /** Maximum PVS-reduced lights evaluated by one draw, not a world-authoring limit. */
  maxRuntimePointLights: number;
  maxCollisionTriangles: number;
  maxRenderChunks: number;
  maxPvsVisibleCells: number;
};

export type ShadoWorldPlayabilityAuthoring = {
  fallRecoveryY: number;
  recoveryPosition: WorldVec3;
  entrances: Array<{ id: string; position: WorldVec3; radius: number; required: boolean }>;
  criticalRegions: string[];
  probes: Array<{ id: string; from: WorldVec3; to: WorldVec3; kind: 'walk' | 'line-of-sight' | 'fall-recovery'; required: boolean }>;
};

export type ShadoWorldAuthoringDocument = {
  kind: 'shado.world.authoring';
  version: 1;
  world: string;
  coordinateSystem: 'babylon-y-up';
  revision: number;
  /** Durable tombstones preventing legacy metadata merges from restoring removed props. */
  legacyObjectExclusions?: string[];
  regions: ShadoWorldAuthoringRegion[];
  objects: ShadoWorldAuthoringObjects;
  /** Durable base-scene mesh transforms, visibility, collision, and PBR assignments. */
  geometry: ShadoWorldGeometryAuthoring;
  lighting: ShadoWorldLightingAuthoring;
  /** Runtime environment preview and bake intent. */
  environment: ShadoWorldEnvironmentAuthoring;
  terrain: ShadoWorldTerrainMaterialAuthoring;
  bake: ShadoWorldBakeSettings;
  /** Authoring-time performance gates checked before promotion. */
  performanceBudgets: ShadoWorldPerformanceBudgets;
  /** Entrances and deterministic traversal/recovery probes. */
  playability: ShadoWorldPlayabilityAuthoring;
};

export type ShadoWorldPrimitive = {
  name: string;
  material: string;
  /** Optional material-authored runtime role retained by headless preprocessing. */
  extraShader?: string;
  /** Optional authored streaming profile retained from glTF extras. */
  visibilityProfile?: string;
  /** Optional authored PVS priority retained from glTF extras. */
  pvsPriority?: string;
  /** Bitwise ShadoCollisionFlags retained from glTF collision metadata. */
  collisionFlags?: number;
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
  /** Optional glTF TEXCOORD_1 stream used by the offline lightmap baker. */
  lightmapUvs?: ArrayLike<number>;
};

export type ShadoWorldGrassCompileOptions = {
  cellSize?: number;
  density?: number;
  maxPlacements?: number;
  maxPlacementsPerPrimitive?: number;
  minimumUpNormal?: number;
  minHeight?: number;
  maxHeight?: number;
  bladeWidth?: number;
  seed?: number;
};

export type ShadoWorldGrassPackage = {
  version: 1;
  cellSize: number;
  cells: {
    x: number[];
    z: number[];
    firstPlacement: number[];
    placementCount: number[];
  };
  placements: {
    positionX: number[];
    positionY: number[];
    positionZ: number[];
    yaw: number[];
    width: number[];
    height: number[];
    phase: number[];
    stiffness: number[];
    colorVariation: number[];
  };
  /** Fixed-resolution authored-surface mask, including topmost non-grass blockers. */
  coverage?: {
    resolution: number;
    wordsPerCell: number;
    words: number[];
    /** Compact quantized terrain relief used to conform generated blade roots. */
    heightField?: {
      resolution: number;
      wordsPerCell: number;
      words: number[];
      minimumY: number[];
      heightRange: number[];
      samples: number[];
    };
  };
};

/**
 * Density-independent grass authoring. Unlike {@link ShadoWorldGrassCompileOptions}
 * this has no placement caps, because no per-blade records are produced.
 */
export type ShadoWorldGrassFieldCompileOptions = {
  cellSize?: number;
  /** Target blades per square metre. A runtime quality knob, not a package cost. */
  density?: number;
  minimumUpNormal?: number;
  minHeight?: number;
  maxHeight?: number;
  bladeWidth?: number;
  seed?: number;
};

/**
 * Where grass may grow, without saying where each blade is.
 *
 * Size is a function of the grass area alone: one 32-bit word per coverage row
 * and 64 quantized heights per cell, whatever the density.
 */
export type ShadoWorldGrassFieldPackage = {
  version: 2;
  cellSize: number;
  density: number;
  minHeight: number;
  maxHeight: number;
  bladeWidth: number;
  /** Shared by every consumer that derives blades, so placement is reproducible. */
  seed: number;
  cells: {
    x: number[];
    z: number[];
  };
  coverage: {
    resolution: number;
    wordsPerCell: number;
    /** One bit per coverage texel: may a blade root here. */
    words: number[];
  };
  heightField: {
    resolution: number;
    wordsPerCell: number;
    /** One bit per height sample: is this sample valid. */
    words: number[];
    minimumY: number[];
    heightRange: number[];
    /** Ground height, normalized into 0..0xffff across the cell's own range. */
    samples: number[];
  };
};

export type ShadoWorldCompileOptions = {
  name: string;
  source?: string;
  /**
   * Transform applied to the source scene and its extracted geometry before
   * either becomes runtime Babylon Y-up world space.
   */
  sourceTransform?: ShadoWorldSourceTransform;
  tileSize?: number;
  maxClusterTriangles?: number;
  /**
   * Smallest triangle count a render chunk should reach before the compiler
   * stops merging neighbouring cells into it. Render chunks are draw units, so
   * a per-cell fragment of two triangles costs a full mesh and draw call for
   * almost no geometry. Merging trades culling granularity for draw calls.
   */
  minRenderChunkTriangles?: number;
  /**
   * Ceiling on how far a merged render chunk may span, in world units. Chunk
   * visibility is the union of its clusters, so an unbounded merge would keep
   * distant geometry resident whenever any part of it is on screen.
   */
  maxRenderChunkExtent?: number;
  /** Width/depth of continuous camera/entity visibility regions. */
  visibilityRegionSize?: number;
  /** Ordinary-region first-pass envelope. Persistent vista cells bypass it. */
  visibilityMaxDistance?: number;
  /** Offline proximity-grass conversion. Set false to omit tagged grass. */
  grass?: ShadoWorldGrassCompileOptions | false;
  /** Density-independent grass field. Compiled alongside `grass`, never instead of it. */
  grassField?: ShadoWorldGrassFieldCompileOptions | false;
  /** Explicit runtime lighting authority. Vertex-color presence is never used to infer this. */
  runtimeLighting?: ShadoWorldRuntimeLighting;
  authoring?: ShadoWorldAuthoringDocument;
  /** Collision-selected primitives in final runtime coordinates. */
  collisionPrimitives?: readonly ShadoWorldPrimitive[];
  /** Stamped structural geometry that suppresses grass without joining the base render scene. */
  grassBlockerPrimitives?: readonly ShadoWorldPrimitive[];
  /** Width/depth of independently resident Havok collision chunks. */
  physicsChunkSize?: number;
  /** Runtime URL, normally a sibling of the spatial package. */
  collisionSource?: string;
};

export type ShadoWorldSourceTransform = 'identity' | 'mirror-x';

export type ShadoWorldRuntimeLighting = {
  mode: 'dynamic' | 'hybrid' | 'baked';
  /** Declares the semantic role of COLOR_0 instead of guessing from its presence. */
  vertexColors: 'material-tint' | 'baked-irradiance';
};

export type ShadoWorldNavigationModifier = {
  /** Stable authored region row supplying this build operation. */
  region: number;
  /** Recast area ID (0..63). */
  area: number;
  /** Detour polygon flags compiled for this area. */
  flags: number;
  /** Excluded spans are removed rather than assigned a traversable area. */
  excluded: number;
  /** Recast-space AABB center after the runtime-to-Recast boundary transform. */
  centerX: number;
  centerY: number;
  centerZ: number;
  sizeX: number;
  sizeY: number;
  sizeZ: number;
};

export type ShadoWorldBounds = {
  min: WorldVec3;
  max: WorldVec3;
};

export type ShadoWorldCollisionDescriptor = {
  source: string;
  format: 'shado-collision-v2';
  chunkSize: number;
  chunkCount: number;
  /** Unique source triangles before conservative boundary duplication. */
  sourceTriangleCount: number;
  /** Stored chunk-local vertices, including cross-chunk duplication. */
  vertexCount: number;
  /** Stored triangles, including references duplicated across intersected chunks. */
  triangleCount: number;
  bounds: ShadoWorldBounds;
  /** FNV-1a hash of the uncompressed artifact bytes. */
  contentHash: string;
};

export type ShadoWorldSpatialPackage = {
  kind: 'shado.world.spatial';
  version: 5;
  name: string;
  /** Runtime geometry, regions, and object transforms are all Babylon Y-up. */
  coordinateSystem: 'babylon-y-up';
  sourceTransform: ShadoWorldSourceTransform;
  source?: string;
  bounds: ShadoWorldBounds;
  collision: ShadoWorldCollisionDescriptor;
  triangleCount: number;
  /** Runtime lighting policy authored by the packer; never inferred from mesh attributes. */
  lighting?: ShadoWorldRuntimeLighting;
  /** Layered terrain-material intent consumed by the Eltania runtime material builder. */
  terrain?: ShadoWorldTerrainMaterialAuthoring;
  /**
   * The authored terrain resolved against the shared palette at publish time.
   *
   * Carried alongside the authoring form rather than replacing it: the editor
   * still needs the intent to show, and the runtime wants only the constants.
   * Resolving here means a layer naming a material that is not in the palette
   * fails the publish rather than rendering as nothing in a player's browser.
   */
  terrainSurface?: EltaniaTerrainSurfaceSpec;
  /** Base-scene mesh/material overrides already included in compiled bounds and collision. */
  geometry?: ShadoWorldGeometryAuthoring;
  environment?: ShadoWorldEnvironmentAuthoring;
  performanceBudgets?: ShadoWorldPerformanceBudgets;
  playability?: ShadoWorldPlayabilityAuthoring;
  /** Standalone and object-attached sources resolved into final Babylon world space. */
  pointLights?: ShadoWorldCompiledPointLight[];
  materials: string[];
  primitives: Array<{
    name: string;
    material: number;
    vertexCount: number;
    /** Authored vistas that bypass both world visibility and mesh-frustum culling. */
    persistent?: boolean;
  }>;
  clusterIndices: number[];
  /** Cluster IDs grouped by stable source-geometry render chunks. */
  renderChunkClusters: number[];
  clusters: {
    centerX: number[];
    centerY: number[];
    centerZ: number[];
    radius: number[];
    coneX: number[];
    coneY: number[];
    coneZ: number[];
    coneCutoff: number[];
    firstIndex: number[];
    indexCount: number[];
    primitive: number[];
    materialPacket: number[];
    renderChunk: number[];
    lodParent: number[];
    cellId: number[];
  };
  packets: {
    cellId: number[];
    material: number[];
    firstCluster: number[];
    clusterCount: number[];
  };
  renderChunks: {
    primitive: number[];
    material: number[];
    firstClusterRef: number[];
    clusterRefCount: number[];
  };
  /** Stable topology records. Kind 0 is an outdoor streaming tile. */
  cells: {
    kind: number[];
    minX: number[];
    minY: number[];
    minZ: number[];
    maxX: number[];
    maxY: number[];
    maxZ: number[];
    firstCluster: number[];
    clusterCount: number[];
    phaseMask: number[];
  };
  /** Authored visual portal edges. Outdoor-only packages intentionally emit none. */
  portals: {
    fromCell: number[];
    toCell: number[];
    dynamicStateId: number[];
    flags: number[];
  };
  /** Compiled region bounds; metadata remains indexed by the same stable row. */
  regions: {
    id: string[];
    name: string[];
    kind: ShadoWorldRegionKind[];
    enabled: number[];
    centerX: number[];
    centerY: number[];
    centerZ: number[];
    sizeX: number[];
    sizeY: number[];
    sizeZ: number[];
    phaseMask: number[];
    tags: string[][];
    metadata: Record<string, unknown>[];
  };
  /** Static-object batches and culling planes. Stamps retain authoring order. */
  objects?: {
    prototypes: {
      id: string[];
      source: string[];
      boundsRadius: number[];
      firstStampRef: number[];
      stampRefCount: number[];
      metadata: Record<string, unknown>[];
    };
    /** Stamp IDs grouped by prototype without duplicating transform records. */
    prototypeStampRefs: number[];
    stamps: {
      id: string[];
      prototype: number[];
      enabled: number[];
      positionX: number[];
      positionY: number[];
      positionZ: number[];
      rotationX: number[];
      rotationY: number[];
      rotationZ: number[];
      scaleX: number[];
      scaleY: number[];
      scaleZ: number[];
      /** Optional offline-baked irradiance, one linear RGBA value per stamp. */
      irradianceR?: number[];
      irradianceG?: number[];
      irradianceB?: number[];
      irradianceA?: number[];
      radius: number[];
      cellId: number[];
      phaseMask: number[];
      tags: string[][];
      metadata: Record<string, unknown>[];
    };
  };
  /** Offline-converted static grass placements grouped into proximity cells. */
  grass?: ShadoWorldGrassPackage;
  /** Density-independent grass field. Preferred over `grass` when both are present. */
  grassField?: ShadoWorldGrassFieldPackage;
  tiles: {
    size: number;
    originX: number;
    originZ: number;
    x: number[];
    z: number[];
    firstCluster: number[];
    clusterCount: number[];
  };
  /**
   * Limits the compiler applied when merging cell-bounded clusters into draw
   * units. A render chunk may span multiple cells, so consumers validating
   * culling granularity check against these rather than assuming one cell.
   */
  renderChunkLimits: {
    minTriangles: number;
    /** Ceiling on merged XZ extent. Persistent chunks are exempt. */
    maxExtent: number;
  };
  /**
   * Continuous, dense first-pass culling topology. Unlike geometry-derived
   * cells, these regions cover every point inside the package bounds.
   */
  visibility?: {
    version: 1;
    /** Conservative authority used to construct region rows. */
    mode: 'distance-flood';
    size: number;
    originX: number;
    originZ: number;
    width: number;
    height: number;
    maxDistance: number;
    /** Reserved for future authored/height-aware occluders; zero in distance-flood mode. */
    occluderCount: number;
    /** Directed set-bit count across all conservative PVS rows. */
    visibleRegionPairs: number;
    /** Exact render-cell to dense visibility-region ownership. */
    cellRegion: number[];
    /** Regions containing authored persistent vistas such as zoneline horizons. */
    persistentRegions: number[];
    /** Render cells that bypass regional occlusion without widening entity PVS. */
    persistentCells: number[];
    /** Conservative region-to-region potentially-visible rows. */
    pvs: {
      wordsPerRow: number;
      words: number[];
    };
  };
  /**
   * Navigation build inputs share authored identity and tile addressing with
   * the spatial package, but remain a separate Recast/Detour product.
   */
  navigation: {
    runtimeToRecast: 'z-y-negative-x';
    modifiers: {
      region: number[];
      area: number[];
      flags: number[];
      excluded: number[];
      centerX: number[];
      centerY: number[];
      centerZ: number[];
      sizeX: number[];
      sizeY: number[];
      sizeZ: number[];
    };
  };
  /** Optional conservative PVS rows. Bit N in row C means cell N may be visible from C. */
  pvs?: {
    wordsPerRow: number;
    words: number[];
  };
  integrity: {
    algorithm: 'fnv1a32-layout';
    layoutHash: string;
  };
  bvh: {
    root: number;
    nodeCount: number;
    quantizationMin: WorldVec3;
    quantizationExtent: WorldVec3;
    childMinX: number[];
    childMinY: number[];
    childMinZ: number[];
    childMaxX: number[];
    childMaxY: number[];
    childMaxZ: number[];
    childRef: number[];
  };
};
