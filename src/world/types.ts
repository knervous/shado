import type { EltaniaTerrainSurfaceSpec } from './terrain-compile';

export type WorldVec3 = [number, number, number];
/** Linear RGBA, for anything that carries an alpha alongside a colour. */
export type WorldVec4 = [number, number, number, number];

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
  /**
   * World metres the ground itself rises where this layer's control mask is
   * fully painted. A causeway, a metalled street, a levee: ground that stands
   * proud because the mask says so.
   *
   * This is deliberately *not* a shader displacement. The zone bake adds it to
   * the baked height field, so the surface the player sees and the surface the
   * player walks on are the same surface by construction, and the raised strip
   * follows every bend of the mask exactly. A vertex displacement in the
   * terrain material would raise only the render and leave collision flat.
   *
   * A layer with no control channel cannot protrude: there would be no mask to
   * say where.
   */
  protrusionMetres?: number;
  /**
   * World metres over which the protrusion ramps from ground to full height at
   * the mask edge. Without it a uniform lift under a hard mask edge is a
   * vertical wall the player catches on.
   */
  protrusionFalloffMetres?: number;
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
  /** How completely painted path wear clears the layers it runs over. */
  pathSuppression?: number;
  biomeTint?: WorldVec3;
  /**
   * Meshes the terrain material applies to, by name.
   *
   * A packed zone marks its ground with the `grass` shader role and needs
   * nothing here. A presentation scene has no packer and therefore no role, so
   * without this the editor cannot tell its terrain from its horizon ring or
   * its sky dome -- and the terrain preview silently has nothing to paint on.
   */
  groundMeshes?: string[];
};

export type ShadoWorldTerrainMaterialAuthoring = {
  enabled: boolean;
  controlMaps: string[];
  layers: ShadoWorldTerrainLayer[];
  settings?: ShadoWorldTerrainSettings;
};

/**
 * How a visibility package's rows were built.
 *
 * `distance-flood` admits every region inside `maxDistance`: it culls by range
 * only, and is the shipped baseline. `sampled-occlusion` additionally drops
 * pairs where no sampled viewpoint in one region reached any sampled point in
 * the other without crossing occluder geometry. Sampling finds witnesses; the
 * absence of a witness is evidence rather than proof, so that mode is
 * experimental and is never chosen for a package implicitly.
 */
export type ShadoWorldVisibilityMode = 'distance-flood' | 'sampled-occlusion';

/**
 * What a bake did, as opposed to what it was asked to do.
 *
 * `mode` is the authority that actually produced the rows, and it differs from
 * `requestedMode` whenever the bake could not run what was asked: a zone with
 * no eligible occluders cannot be occlusion-tested, and returning fully
 * visible rows labelled `sampled-occlusion` would claim a test that never ran.
 */
export type ShadoWorldVisibilityBakeReport = {
  requestedMode: ShadoWorldVisibilityMode;
  mode: ShadoWorldVisibilityMode;
  fallbackReason: 'no-eligible-occluders' | 'budget-exhausted' | null;
  occluderTriangles: number;
  /** Pairs admitted by the local guarantee, never offered to an occlusion test. */
  forcedLocalPairs: number;
  occlusionTested: number;
  occluded: number;
  pairsBeforeRowFlood: number;
  pairsAfterRowFlood: number;
  /**
   * Milliseconds per stage of the bake itself. These do not include loading or
   * decoding the package, weighting the result, validating it or writing it:
   * a caller that wants an end-to-end figure has to time the end and the other
   * end itself.
   */
  stages: {
    occluderGridMs: number;
    regionSamplingMs: number;
    pairLoopMs: number;
    rowFloodMs: number;
    totalMs: number;
  };
  /** Query work, counted in the hot path rather than inferred from wall time. */
  work: {
    /** Which acceleration structure answered the queries. */
    index: 'bvh' | 'grid' | null;
    /** Prototypes indexed once, and how many times they were placed. */
    instancedPrototypes: number;
    instancedPlacements: number;
    /** Triangles the instanced structure holds, against what expanding cost. */
    instancedUniqueTriangles: number;
    instancedPlacedTriangles: number;
    /**
     * Bytes held by the index structures themselves. This is tracked
     * allocation, not process RSS: neither number substitutes for the other,
     * and the report carries both.
     */
    indexTrackedBytes: number;
    segmentQueries: number;
    blockedQueries: number;
    columnQueries: number;
    /**
     * Index elements whose bounds were tested: grid cells entered by a DDA
     * walk, or hierarchy nodes opened. Comparable across indexes only as
     * "work done to reach the triangles".
     */
    nodeVisits: number;
    triangleTests: number;
    /**
     * What the index stores. For the grid this is triangle references across
     * buckets, which exceeds `occluderTriangles` by its duplication factor;
     * for the hierarchy it is the node count, which is fewer. The two are not
     * the same quantity and are not comparable to each other.
     */
    indexEntries: number;
    /** Regions with no surface to stand on, which are sampled as unknown and admitted. */
    regionsWithoutFloor: number;
  };
  /**
   * How the bake stopped. Anything but `none` means the remaining pairs were
   * ADMITTED without being tested -- a bounded bake gives up selectivity, never
   * correctness, so a truncated run is publishable and merely worse.
   */
  limit: {
    stop: 'none' | 'seconds' | 'segment-queries' | 'cancelled' | 'memory';
    /** Where the bake was when it gave up. */
    stoppedDuring: 'none' | 'index-build' | 'region-sampling' | 'pair-loop';
    pairsAdmittedAfterStop: number;
    /** Regions left unsampled, each of which is therefore admitted unknown. */
    regionsLeftUnsampled: number;
  };
};

/**
 * Bounds on bake work, enforced while it runs.
 *
 * Checking a budget after the fact reports a number; it does not protect a
 * machine. Reaching any of these stops occlusion testing and admits every
 * remaining pair, which is the conservative direction.
 */
export type ShadoWorldVisibilityBudget = {
  /**
   * Wall-clock ceiling for the WHOLE bake, measured from the first thing it
   * does. It covers index construction and region sampling, not only the pair
   * sweep: a deadline that starts after the expensive preprocessing is not a
   * deadline.
   */
  maxSeconds?: number;
  maxSegmentQueries?: number;
  /**
   * Resident bytes, sampled through a host-supplied reader so this stays free
   * of any runtime's process API. Exceeding it stops the bake the same way a
   * deadline does.
   */
  maxResidentBytes?: number;
  residentBytes?: () => number;
  /** Polled at every stage boundary; an `AbortSignal` satisfies this shape. */
  signal?: { readonly aborted: boolean };
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

/**
 * An authored ambient particle emitter: motes, embers, drifting ash.
 *
 * Deliberately close to what a Babylon `ParticleSystem` wants, because the
 * runtime's job here is to place and gate one, not to invent a second
 * animation language on top of it. What is NOT a passthrough is the gating:
 * `range`, `hours` and the viewer's own particle setting decide whether a
 * system exists at all, and an author who cannot see those in the schema will
 * write a zone that runs forty emitters in a town nobody is standing in.
 *
 * Sizes and distances are world units, times are seconds.
 */
export type ShadoWorldParticleEmitter = {
  id: string;
  label?: string;
  enabled?: boolean;
  /** Centre of the emission box, in final Babylon world space. */
  position: WorldVec3;
  /** Half extents of the box particles are born in. */
  size: WorldVec3;
  /** Texture URL. Omitted uses the shared flare the rest of the game uses. */
  texture?: string;
  /** Live particles at once. The runtime clamps this; see the fx module. */
  capacity: number;
  /** Particles born per second. */
  emitRate: number;
  /** rgba at birth, at birth for the second colour, and at death. */
  color1: WorldVec4;
  color2: WorldVec4;
  colorDead: WorldVec4;
  minSize: number;
  maxSize: number;
  minLifeTime: number;
  maxLifeTime: number;
  /** The two corners of the initial velocity cone. */
  direction1: WorldVec3;
  direction2: WorldVec3;
  minEmitPower: number;
  maxEmitPower: number;
  /** Constant acceleration; the dial between motes and falling ash. */
  gravity?: WorldVec3;
  /** Simulation rate. Lower is slower and smoother. */
  updateSpeed?: number;
  /** Additive reads as light, standard as matter. */
  blendMode?: 'add' | 'standard';
  /**
   * How far away the emitter still runs.
   *
   * Ambient particles are a near-field effect and a zone may author many, so
   * each one sleeps until the camera is inside this radius of its box.
   */
  range: number;
  /** Active hours, `[from, to]`, wrapping through midnight. Omitted is always. */
  hours?: [number, number];
  metadata?: Record<string, unknown>;
};

/**
 * The zone's global participating medium — the air everywhere a
 * {@link ShadoWorldMediaVolume} does not override.
 *
 * Every field is optional and falls back to the renderer's own default, so a
 * world that authors none behaves exactly as it did before this existed. Grid
 * dimensions are deliberately absent: they are a performance budget the
 * runtime owns, not a look.
 */
export type ShadoWorldVolumetricMedium = {
  /** False disables the froxel volume for this zone entirely. */
  enabled?: boolean;
  /** Extinction per world unit at `heightReference`. */
  density?: number;
  /** e-folding height of the medium above `heightReference`. */
  heightFalloff?: number;
  /** World Y the density is quoted at. */
  heightReference?: number;
  /** Fraction of extinction that scatters rather than absorbs. */
  albedo?: number;
  /** Henyey-Greenstein anisotropy; positive is forward scattering. */
  anisotropy?: number;
  /** Depth of the animated density variation; 0 is a homogeneous medium. */
  noiseAmount?: number;
  /**
   * World-space frequency of that variation.
   *
   * This is the dial between fog that *drifts* and fog that *pulses*. At a
   * low frequency one noise cell is wider than the view, so the whole medium
   * brightens and dims together and reads as breathing; features small
   * enough to pass through the frame read as movement.
   */
  noiseScale?: number;
  /** How fast the variation drifts, world units per second. */
  windSpeed?: number;
  /** Sky/multiscatter in-scatter, in units of the medium's own colour. */
  ambientMultiplier?: number;
  /** Nearest view depth the volume covers. */
  near?: number;
  /** Farthest view depth the volume covers. */
  far?: number;
  /** Depth distribution exponent; 1 is uniform. */
  depthPower?: number;
  /** Weight of the new frame in the temporal blend; 1 disables reprojection. */
  temporalBlend?: number;
  /** Overall dial on the composite. 0 disables the whole system. */
  strength?: number;
};

/**
 * An authored region of participating media: a fog bank with bounds.
 *
 * The runtime medium was one global set of parameters per zone, which is the
 * right default for an outdoor world under one sky and useless for "thick fog
 * on the church grounds and nowhere else". A volume raises (or lowers) the
 * medium inside its shape plus its feather, and leaves the rest of the zone at
 * the zone default -- which stays the volume of last resort, so a world with
 * no `mediaVolumes` renders exactly as it did before this existed.
 *
 * Composition is by `priority` ascending, then by `id`, so overlapping volumes
 * produce the same result whatever order they were authored in. Each volume in
 * turn blends the medium toward its own parameters by its coverage weight;
 * within its solid interior the weight is 1 and it simply wins.
 *
 * Two runtime limits bound what a volume can promise, and neither is a
 * property of the volume: the froxel grid reaches about 400 world units from
 * the camera, and generated `ShadoMaterial` surfaces -- foliage, grass blades,
 * actors -- do not sample it. See M02/M03 in `docs/astra-audit.md`.
 */
export type ShadoWorldMediaVolume = {
  id: string;
  /** Author-facing label; never read by the runtime. */
  label?: string;
  enabled?: boolean;
  shape: 'box' | 'sphere';
  /** Centre in final Babylon world space, the same frame as a point light. */
  position: WorldVec3;
  /** Box half-extents. A sphere uses `size[0]` as its radius. */
  size: WorldVec3;
  /** Rotation about Y, degrees. Boxes only. */
  yaw?: number;
  /** Extinction per world unit at `heightReference`, inside the volume. */
  density: number;
  /** Fraction of extinction that scatters rather than absorbs. */
  albedo?: number;
  /** Henyey-Greenstein anisotropy; positive is forward scattering. */
  anisotropy?: number;
  /**
   * Scattering colour. Omitted means inherit the zone's fog colour, which is
   * what keeps an unauthored tint moving with the hour instead of freezing a
   * midday blue into a dusk scene.
   */
  color?: WorldVec3;
  /** e-folding height above `heightReference` inside the volume. */
  heightFalloff?: number;
  /** World Y the density is quoted at. Defaults to the volume's floor. */
  heightReference?: number;
  /** Distance outside the shape over which coverage falls 1 -> 0. */
  feather?: number;
  /** Higher composes later, and therefore wins where volumes overlap. */
  priority?: number;
  /**
   * Hours the volume is active, `[from, to]` on a 24-hour clock; `from > to`
   * wraps through midnight. Omitted is always. Edges cross-fade over
   * `hoursFeather`.
   */
  hours?: [number, number];
  hoursFeather?: number;
  metadata?: Record<string, unknown>;
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
  /**
   * The zone has no sky over it, so nothing may light it as though it had.
   *
   * The sky rig gives every zone a hemispheric fill and a directional sun with
   * a 6% floor, which is right outdoors at night and wrong four storeys
   * underground: a fill from above lights vertical walls and leaves
   * downward-facing vaults dark, so a sealed crypt rendered with pale cold
   * walls under black vaults and read as a film set rather than a tomb. The
   * authored ambient could not counteract it — taking that to 0.02 changed
   * nothing, which is how the sky rig was found to be the source.
   *
   * Absent means outdoor, so every existing zone keeps the sky it has.
   */
  interior?: boolean;
  weather: { preset: string; intensity: number; wind: WorldVec3 };
  timeOfDay: { hour: number; cycleSeconds: number; running: boolean };
  /**
   * Authored lakes and rivers. Each owns a `kind: 'water'` region, which the
   * validator requires: the navmesh, the lake reseat and the runtime's region
   * tree all find water by its region, so a body without one is water the
   * rest of the world cannot see.
   */
  waterBodies?: ShadoWorldWaterBody[];
  audioEmitters: Array<{ id: string; source: string; position: WorldVec3; range: number; volume: number; loop: boolean; metadata: Record<string, unknown> }>;
  reflectionProbes: Array<{ id: string; position: WorldVec3; size: WorldVec3; resolution: number; refresh: 'bake' | 'once' | 'runtime'; metadata: Record<string, unknown> }>;
  /** Authored participating-media volumes. Absent or empty is the zone default alone. */
  mediaVolumes?: ShadoWorldMediaVolume[];
  /** The zone's global medium. Absent leaves every renderer default in place. */
  volumetric?: ShadoWorldVolumetricMedium;
  /** Authored ambient particle emitters. */
  particleEmitters?: ShadoWorldParticleEmitter[];
  /** How the zone sounds when nothing is happening in it. */
  ambience?: ShadoWorldZoneAmbience;
};

/**
 * The zone's ambience, in the layers doc §32 asks for: a continuous bed, medium
 * loops under it, and sparse one-shots over the top. The positional layer is
 * `audioEmitters`, which already existed.
 *
 * Every field names a **component**, `family/element`, never a file. A zone that
 * names `river_02.ogg` breaks the next time the library is promoted, and the
 * library is promoted every time the corpus changes.
 */
export type ShadoWorldZoneAmbience = {
  /** The continuous bed, e.g. `env-bed/field`, `env-bed/cave`. */
  bed?: string;
  /** Swapped in between dusk and dawn. Omitted keeps `bed` all night. */
  bedNight?: string;
  /** Medium loops under the bed everywhere in the zone, e.g. `env-wind/leaves`. */
  loops?: string[];
  /** Sparse one-shots: a bird, a branch, an insect. */
  oneshots?: Array<{
    source: string;
    /** Expected occurrences per minute. */
    perMinute: number;
    /** `[from, to]` on a 24-hour clock; `from > to` wraps through midnight. */
    hours?: [number, number];
  }>;
  /** Scales every ambience layer in this zone. Omitted is 1. */
  gain?: number;
};

/**
 * Ambience for one region, which overrides the zone's while the listener is
 * inside it. Lives in `region.metadata.ambience`, because region metadata is
 * where tool and game payload belongs and the hot columnar planes are not.
 *
 * This is the layer that makes movement through a zone mean something (doc §32).
 * A zone-wide bed sounds identical in a market, a cloister and a burial yard; a
 * district that names its own is the difference between a place and a backdrop.
 *
 * Overrides are per field. A region that declares only `loops` keeps the zone's
 * bed under them, and a region that declares only `interior` changes nothing but
 * the acoustics -- which is usually all a doorway needs to do.
 */
export type ShadoWorldRegionAmbience = {
  /** Replaces the zone bed while inside. */
  bed?: string;
  bedNight?: string;
  /** Replaces the zone's medium loops while inside. */
  loops?: string[];
  /** Replaces the zone's stochastic layer while inside. */
  oneshots?: Array<{ source: string; perMinute: number; hours?: [number, number] }>;
  /** Scales the zone gain rather than replacing it. */
  gain?: number;
  /**
   * Under cover: rain is heard from beneath a roof rather than on the ground,
   * and the acoustic space becomes a room.
   */
  interior?: boolean;
  /**
   * The acoustic space to enter, when `interior` is too blunt a word for it --
   * a cathedral and a cellar are both indoors and share no tail.
   */
  space?: string;
  /**
   * Larger wins where regions overlap. Omitted means the smallest region wins,
   * which is what makes a shrine inside a market behave the way it reads.
   */
  priority?: number;
  /** Crossfade on crossing the boundary, seconds. Omitted is a doorway's length. */
  fadeSeconds?: number;
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

/**
 * A lake or a river, authored as a shape rather than a mesh.
 *
 * The surface is generated from this at bake time against the zone's final
 * ground (sculpting included), so its depth-to-bed and flow attributes can
 * never drift from the ground they describe. Everything is in world units.
 */
export type ShadoWorldWaterBody = {
  id: string;
  name: string;
  enabled: boolean;
  kind: 'lake' | 'river';
  /** Lake: the surface height. River: the height of any path point without its own. */
  level: number;
  /** Lake: the waterline, a closed polygon in world XZ. */
  outline?: Array<[number, number]>;
  /** River: the centreline, upstream first; each point carries its width and surface height. */
  path?: Array<{ x: number; z: number; width: number; level: number }>;
  /** Current in units per second: along the path for a river, along `flowDirection` for a lake. */
  flowSpeed: number;
  /** Lake drift direction in XZ; ignored by rivers. */
  flowDirection?: [number, number];
  /** Extra foam, 0..1, on top of what depth and slope produce. */
  foam: number;
  /** How far below the surface the owned region reaches, to take in the bed. */
  depth: number;
  /** The `kind: 'water'` region this body owns. */
  regionId: string;
  metadata: Record<string, unknown>;
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
  /** Offline visibility bakes the promotion runs for this zone. */
  visibility?: ShadoWorldVisibilityAuthoring;
};

/**
 * Visibility bakes a zone asks its promotion to run. Declaring disocclusion
 * sources makes the disocclusion PVS a mandatory promotion stage: the sidecar
 * is baked from the just-published package and published beside it.
 */
export type ShadoWorldVisibilityAuthoring = {
  disocclusion?: {
    /** Why these sources and settings (how they were resolved). */
    note?: string;
    /** Source volumes (a camera anywhere inside one uses its rows). */
    sources: Array<{
      id: string;
      min: [number, number, number];
      max: [number, number, number];
      /** Capture near and far distance, world units. */
      near: number;
      far: number;
      /** Tan of the vertical half-angle the side faces capture (default the baker's). */
      sideUp?: number;
    }>;
    /** Numeric overrides of the baker's buffer settings. */
    settings?: Record<string, number>;
    /**
     * Review poses for the proving ground: the first is the source pose, the
     * rest are checks. Not read at runtime.
     */
    poses?: Array<{ name: string; at: [number, number, number]; look: [number, number, number]; fov?: number; hour?: number }>;
  };
};

export type ShadoWorldPrimitive = {
  name: string;
  /** Placed-object stamp this primitive was expanded from (occluder assembly). */
  stamp?: number;
  material: string;
  /** Optional material-authored runtime role retained by headless preprocessing. */
  extraShader?: string;
  /** Optional authored streaming profile retained from glTF extras. */
  visibilityProfile?: string;
  /** Optional authored PVS priority retained from glTF extras. */
  pvsPriority?: string;
  /** Bitwise ShadoCollisionFlags retained from glTF collision metadata. */
  collisionFlags?: number;
  /**
   * Whether the source material draws both faces. Consumed by occlusion: a
   * single-sided surface is invisible from behind, so it must not hide
   * anything from a viewpoint on that side. Undefined means unknown, and an
   * unknown surface is treated as two-sided, which is the historical
   * behaviour and the one that blocks more -- callers that know better
   * (anything reading a glTF material) should say so.
   */
  doubleSided?: boolean;
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
  /**
   * Which visibility authority to bake. Defaults to `distance-flood`.
   * `sampled-occlusion` is experimental and never selected implicitly.
   */
  visibilityMode?: ShadoWorldVisibilityMode;
  /**
   * Split source columns into per-floor volumes (visibility v2). Only takes
   * effect with `sampled-occlusion`; see `compileShadoWorldVisibility`.
   */
  visibilityVerticalVolumes?: boolean;
  /** The supported camera heights for v2 volumes; part of package identity. */
  visibilityCameraExtent?: { minY: number; maxY: number };
  /**
   * Bake diagnostics for the visibility pass: which authority actually ran,
   * what it rejected and why it fell back. Reported rather than stored, so
   * package bytes stay deterministic.
   */
  visibilityReport?: (report: ShadoWorldVisibilityBakeReport) => void;
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
  /**
   * Where the painted terrain is not grass, over the terrain's own world
   * rectangle. Built by `terrainGrassSuppression` from the zone's control maps.
   *
   * Without it the bake grows a lawn through every dirt track and paved square
   * in the zone, because a painted floor is the same triangle as the meadow
   * beside it and geometry alone cannot tell them apart.
   */
  grassTerrainSuppression?: {
    width: number;
    height: number;
    values: Uint8Array;
    worldMin: readonly [number, number];
    worldMax: readonly [number, number];
    /** 0..1; a texel at or above this is not grass. Defaults to 0.35. */
    threshold?: number;
  };
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
    /**
     * 1: rows are indexed by region, and nothing else is on the wire.
     * 2: rows are indexed by VOLUME on the source side, then one union row per
     *    region; `sourceDomain` and `volumes` are required.
     *
     * A reader that does not know a version must refuse it or fall back to
     * the flood -- never guess a layout. v1 packages read exactly as before.
     */
    version: 1 | 2;
    /** How the rows were built. See {@link ShadoWorldVisibilityMode}. */
    mode: ShadoWorldVisibilityMode;
    size: number;
    originX: number;
    originZ: number;
    width: number;
    height: number;
    maxDistance: number;
    /** Occluder triangles the bake tested against; zero in distance-flood mode. */
    occluderCount: number;
    /** Directed set-bit count across all conservative PVS rows. */
    visibleRegionPairs: number;
    /** Exact render-cell to dense visibility-region ownership. */
    cellRegion: number[];
    /** Regions containing authored persistent vistas such as zoneline horizons. */
    persistentRegions: number[];
    /** Render cells that bypass regional occlusion without widening entity PVS. */
    persistentCells: number[];
    /**
     * Vertically separated source volumes, when the bake produced them.
     *
     * A 2D region is a column spanning every height, so it holds a room AND
     * the roof above it, and one row has to serve both: the row is therefore
     * a rooftop row, and no amount of better sampling can cull the room. That
     * is measured, not assumed -- sampling every floor in Crypts instead of
     * only the roof changed nothing at all.
     *
     * These split a region into vertical bands, each with its own row. Rows
     * are indexed by VOLUME on the source side and by region on the target
     * side, so the bitset is `volumes.count + width * height` rows of
     * `width * height` bits: one row per volume, then one conservative union
     * row per region for a camera whose height no volume covers.
     *
     * Absent on a legacy package, where rows are indexed by region and a
     * reader must keep working exactly as before.
     */
    volumes?: {
      count: number;
      /** Which region each volume stands in. Sorted by region, then by minY. */
      region: number[];
      /**
       * The band each volume covers, in world Y. FINITE: every band lies
       * inside `sourceDomain`. JSON has no infinity, and an Infinity written
       * here came back as null -- which then failed validation, or worse,
       * coerced to zero.
       */
      minY: number[];
      maxY: number[];
      /**
       * Where each region's volumes start: region r owns volumes
       * `[regionOffset[r], regionOffset[r + 1])`. Length is regionCount + 1.
       * Lets a reader binary-search one region's bands instead of scanning
       * every volume in the zone per camera update.
       */
      regionOffset: number[];
    };
    /**
     * v2: the camera heights the volumes describe, in world Y.
     *
     * A camera outside it answers from its region's union row. It is part of
     * the package's identity, and raising it is an explicit bake input, not
     * something a reader infers.
     */
    sourceDomain?: { minY: number; maxY: number };
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
