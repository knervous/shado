import type { ShadoWorldObjectPrototype, WorldVec3 } from './types';

export type ShadoBuilderModuleRole =
  | 'foundation'
  | 'floor'
  | 'wall-straight'
  | 'wall-corner-inner'
  | 'wall-corner-outer'
  | 'wall-cap'
  | 'opening-door'
  | 'opening-window'
  | 'column'
  | 'roof'
  | 'stair'
  | 'ramp'
  | 'road'
  | 'bridge'
  | 'terrain-transition'
  | 'trim'
  | 'prop-anchor'
  | 'foliage'
  | 'light-anchor'
  | 'region-anchor'
  | 'audio-anchor'
  | 'fx-anchor';

export type ShadoBuilderPieceQuality =
  | 'certified-reusable'
  | 'legacy-reusable'
  | 'procedural-elementary';

export type ShadoBuilderProvenanceStatus =
  | 'approved'
  | 'unreviewed'
  | 'restricted-reference';

export interface ShadoBuilderSocket {
  id: string;
  type: string;
  position: WorldVec3;
  rotationDegrees: WorldVec3;
  accepts: string[];
  tolerance: number;
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface ShadoBuilderModule {
  id: string;
  name: string;
  role: ShadoBuilderModuleRole;
  enabled: boolean;
  /** Complete published prototype installed when an assembly is committed. */
  prototype: ShadoWorldObjectPrototype;
  /** Local X/Y/Z dimensions in canonical Babylon Y-up units. */
  dimensions: WorldVec3;
  /** Repeatable length along spanAxis. */
  span: number;
  spanAxis: 'x' | 'z';
  /** Additional Y offset applied after surface grounding. */
  groundOffset: number;
  sockets: ShadoBuilderSocket[];
  materialRoles: string[];
  quality: ShadoBuilderPieceQuality;
  provenance: {
    status: ShadoBuilderProvenanceStatus;
    source: string;
    licenseExpression?: string;
    contentHash?: string;
  };
  inspection: {
    contentHash: string;
    meshCount: number;
    triangleCount: number;
    materialCount: number;
    collisionAvailable: boolean;
    aabb: { min: WorldVec3; max: WorldVec3 };
    pivot: WorldVec3;
  };
  review: {
    artDirectionApproved: boolean;
    modularFitApproved: boolean;
    reviewedBy: string;
    notes: string;
  };
  /** Required whenever quality is procedural-elementary. */
  fallbackReason?: string;
  weight: number;
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface ShadoBuilderMaterialPalette {
  id: string;
  revision: number;
  name: string;
  state: 'draft' | 'certified' | 'deprecated';
  roles: Record<string, string[]>;
  metadata: Record<string, unknown>;
}

export interface ShadoBuilderKit {
  id: string;
  revision: number;
  name: string;
  description: string;
  state: 'draft' | 'certified' | 'deprecated';
  coordinateSystem: 'babylon-y-up';
  gridSize: number;
  preferReusablePieces: true;
  allowProceduralFallback: boolean;
  modules: ShadoBuilderModule[];
  supportedPalettes: string[];
  metadata: Record<string, unknown>;
}

export interface ShadoBuilderDiagnostic {
  code: string;
  severity: 'info' | 'warning' | 'error';
  message: string;
  moduleId?: string;
  instanceId?: string;
  segment?: number;
}

export interface ShadoBuilderCertification {
  ok: boolean;
  kitId: string;
  kitRevision: number;
  diagnostics: ShadoBuilderDiagnostic[];
  summary: {
    modules: number;
    certifiedReusable: number;
    legacyReusable: number;
    proceduralElementary: number;
    errors: number;
    warnings: number;
  };
}

export interface ShadoBuilderWallRunRequest {
  assemblyId: string;
  kitId: string;
  kitRevision: number;
  anchors: WorldVec3[];
  seed: number;
  maxSlopeDegrees: number;
  allowProceduralFallback: boolean;
  proceduralFallbackReason?: string;
  metadata?: Record<string, unknown>;
}

export type ShadoBuilderBlueprintKind =
  | 'wall-run'
  | 'rectangular-building'
  | 'road-spline'
  | 'room-graph'
  | 'stair-ramp'
  | 'terrain-transition'
  | 'dressing-field'
  | 'nested-assembly';

export interface ShadoBuilderSolveRequest extends ShadoBuilderWallRunRequest {
  blueprint?: ShadoBuilderBlueprintKind;
  paletteId?: string;
  styleProfileId?: string;
  parameters?: Record<string, unknown>;
  pinnedInstances?: Record<string, Partial<Pick<ShadoBuilderAssemblyInstance, 'moduleId' | 'position' | 'rotationDegrees' | 'scale'>>>;
  nestedAssemblies?: ShadoBuilderAssembly[];
}

export interface ShadoBuilderStyleProfile {
  id: string;
  revision: number;
  name: string;
  region: string;
  summary: string;
  requiredTags: string[];
  encouragedTags: string[];
  forbiddenTags: string[];
  discoveryTags: string[];
  materialRoles: string[];
  /** A healthy range prevents a profile from turning into a rigid theme template. */
  discoveryRange: { minimum: number; maximum: number };
  metadata: Record<string, unknown>;
}

export interface ShadoBuilderAssemblyInstance {
  id: string;
  moduleId: string;
  prototypeId: string;
  quality: ShadoBuilderPieceQuality;
  position: WorldVec3;
  rotationDegrees: WorldVec3;
  scale: WorldVec3;
  segment: number;
  ownership?: 'parametric' | 'pinned' | 'detached';
  /** Immutable module evidence allows nested assemblies to span certified kits. */
  moduleSnapshot?: ShadoBuilderModule;
  metadata: Record<string, unknown>;
}

export interface ShadoBuilderAssembly {
  kind: 'shado.builder.assembly';
  version: 1;
  id: string;
  kitId: string;
  kitRevision: number;
  blueprint: ShadoBuilderBlueprintKind;
  state: 'proposal' | 'committed';
  anchors: WorldVec3[];
  seed: number;
  inputHash: string;
  outputHash: string;
  instances: ShadoBuilderAssemblyInstance[];
  socketEdges?: Array<{ fromInstance: string; fromSocket: string; toInstance: string; toSocket: string }>;
  nestedAssemblyIds?: string[];
  paletteId?: string;
  styleProfileId?: string;
  parameters?: Record<string, unknown>;
  diagnostics: ShadoBuilderDiagnostic[];
  qualitySummary: {
    certifiedReusable: number;
    legacyReusable: number;
    proceduralElementary: number;
  };
  metadata: Record<string, unknown>;
}

export function certifyShadoBuilderKit(kit: ShadoBuilderKit): ShadoBuilderCertification {
  const diagnostics: ShadoBuilderDiagnostic[] = [];
  const ids = new Set<string>();
  const prototypeIds = new Set<string>();
  if (!validId(kit.id)) diagnostics.push(error('kit-id', `Kit id '${kit.id}' must use lowercase letters, numbers, hyphens, or underscores.`));
  if (kit.coordinateSystem !== 'babylon-y-up') diagnostics.push(error('coordinate-system', 'Builder kits must use canonical Babylon Y-up space.'));
  if (!(kit.gridSize > 0) || !Number.isFinite(kit.gridSize)) diagnostics.push(error('grid-size', 'Kit gridSize must be positive and finite.'));
  if (kit.preferReusablePieces !== true) diagnostics.push(error('reusable-policy', 'Kits must prefer reusable pieces.'));
  for (const module of kit.modules) {
    if (ids.has(module.id)) diagnostics.push(error('duplicate-module', `Module id '${module.id}' is duplicated.`, module.id));
    ids.add(module.id);
    if (!validId(module.id)) diagnostics.push(error('module-id', `Module id '${module.id}' is invalid.`, module.id));
    if (!validId(module.prototype.id)) diagnostics.push(error('prototype-id', `Prototype id '${module.prototype.id}' is invalid.`, module.id));
    if (prototypeIds.has(module.prototype.id)) diagnostics.push(warning('shared-prototype', `Prototype '${module.prototype.id}' is shared by multiple modules; confirm this is intentional.`, module.id));
    prototypeIds.add(module.prototype.id);
    if (!module.prototype.source.trim()) diagnostics.push(error('prototype-source', 'Module prototype source is required.', module.id));
    if (!module.dimensions.every((value) => Number.isFinite(value) && value > 0)) diagnostics.push(error('module-dimensions', 'Module dimensions must be positive finite X/Y/Z values.', module.id));
    if (!(module.span > 0) || !Number.isFinite(module.span)) diagnostics.push(error('module-span', 'Module span must be positive and finite.', module.id));
    if (!Number.isFinite(module.groundOffset)) diagnostics.push(error('ground-offset', 'Module groundOffset must be finite.', module.id));
    if (!(module.weight > 0) || !Number.isFinite(module.weight)) diagnostics.push(error('module-weight', 'Module weight must be positive and finite.', module.id));
    if (module.provenance.status === 'restricted-reference') diagnostics.push(error('restricted-reference', `Module '${module.id}' uses restricted-reference content and cannot be certified.`, module.id));
    if (module.quality === 'certified-reusable') {
      if (module.provenance.status !== 'approved') diagnostics.push(error('provenance-unapproved', `Certified reusable module '${module.id}' requires approved provenance.`, module.id));
      if (!/^[a-f0-9]{64}$/i.test(module.provenance.contentHash ?? '')) diagnostics.push(error('asset-hash', `Certified reusable module '${module.id}' requires a SHA-256 content hash.`, module.id));
      if (!module.inspection || module.inspection.contentHash !== module.provenance.contentHash) diagnostics.push(error('inspection-hash', `Certified reusable module '${module.id}' requires inspection evidence matching its source hash.`, module.id));
      else {
        if (!(module.inspection.meshCount > 0) || !(module.inspection.triangleCount > 0)) diagnostics.push(error('inspection-geometry', `Certified reusable module '${module.id}' has no inspected render geometry.`, module.id));
        if (!(module.inspection.materialCount > 0)) diagnostics.push(error('inspection-material', `Certified reusable module '${module.id}' has no inspected material.`, module.id));
        if (!module.inspection.collisionAvailable && structuralRole(module.role)) diagnostics.push(error('inspection-collision', `Structural module '${module.id}' requires inspected collision geometry.`, module.id));
        if (!module.inspection.aabb?.min.every(Number.isFinite) || !module.inspection.aabb?.max.every(Number.isFinite) || !module.inspection.pivot?.every(Number.isFinite)) diagnostics.push(error('inspection-bounds', `Certified reusable module '${module.id}' has invalid inspected bounds or pivot.`, module.id));
      }
      if (!module.review?.artDirectionApproved) diagnostics.push(error('art-direction-review', `Certified reusable module '${module.id}' requires Eltania art-direction approval.`, module.id));
      if (!module.review?.modularFitApproved) diagnostics.push(error('modular-fit-review', `Certified reusable module '${module.id}' requires golden-scene modular-fit approval.`, module.id));
      if (!module.review?.reviewedBy.trim()) diagnostics.push(error('reviewer', `Certified reusable module '${module.id}' requires a named reviewer.`, module.id));
      if (module.metadata.lodPolicy === undefined) diagnostics.push(warning('lod-policy', `Module '${module.id}' has no recorded LOD policy.`, module.id));
      if (module.metadata.occluderPolicy === undefined) diagnostics.push(warning('occluder-policy', `Module '${module.id}' has no recorded occluder policy.`, module.id));
    }
    if (module.quality === 'procedural-elementary') {
      if (!module.fallbackReason?.trim()) diagnostics.push(error('procedural-reason', `Procedural elementary module '${module.id}' requires a fallback reason.`, module.id));
      diagnostics.push(warning('procedural-elementary', `Module '${module.id}' is an elementary procedural fallback and will not be selected while a reusable candidate exists.`, module.id));
    }
    const socketIds = new Set<string>();
    for (const socket of module.sockets) {
      if (socketIds.has(socket.id)) diagnostics.push(error('duplicate-socket', `Socket '${socket.id}' is duplicated in module '${module.id}'.`, module.id));
      socketIds.add(socket.id);
      if (!socket.type.trim()) diagnostics.push(error('socket-type', `Socket '${socket.id}' requires a type.`, module.id));
      if (!socket.position.every(Number.isFinite) || !socket.rotationDegrees.every(Number.isFinite)) diagnostics.push(error('socket-transform', `Socket '${socket.id}' has an invalid transform.`, module.id));
      if (!Number.isFinite(socket.tolerance) || socket.tolerance < 0) diagnostics.push(error('socket-tolerance', `Socket '${socket.id}' has an invalid tolerance.`, module.id));
    }
    if (module.role === 'wall-straight' && module.sockets.filter((socket) => socket.type === 'wall.end').length < 2) diagnostics.push(error('wall-sockets', `Wall module '${module.id}' requires two wall.end sockets.`, module.id));
  }
  const targetsWall = kit.modules.some((module) => module.role.startsWith('wall-')) || (Array.isArray(kit.metadata.blueprints) && kit.metadata.blueprints.includes('wall-run'));
  if (targetsWall && !kit.modules.some((module) => module.enabled && module.role === 'wall-straight' && module.quality === 'certified-reusable')) diagnostics.push(error('missing-certified-wall', 'A production wall kit requires at least one enabled certified reusable wall-straight module.'));
  if (!kit.modules.some((module) => module.enabled && module.quality === 'certified-reusable')) diagnostics.push(error('missing-certified-module', 'A production kit requires at least one enabled certified reusable module.'));
  const summary = {
    modules: kit.modules.length,
    certifiedReusable: kit.modules.filter((module) => module.quality === 'certified-reusable').length,
    legacyReusable: kit.modules.filter((module) => module.quality === 'legacy-reusable').length,
    proceduralElementary: kit.modules.filter((module) => module.quality === 'procedural-elementary').length,
    errors: diagnostics.filter((item) => item.severity === 'error').length,
    warnings: diagnostics.filter((item) => item.severity === 'warning').length,
  };
  return { ok: summary.errors === 0, kitId: kit.id, kitRevision: kit.revision, diagnostics, summary };
}

/** Deterministic modular wall solver. Surface projection is applied by the Libra host. */
export function solveShadoBuilderWallRun(kit: ShadoBuilderKit, request: ShadoBuilderWallRunRequest): ShadoBuilderAssembly {
  if (request.kitId !== kit.id || request.kitRevision !== kit.revision) throw new Error(`kit revision mismatch: requested ${request.kitId}@${request.kitRevision}, found ${kit.id}@${kit.revision}`);
  if (!validId(request.assemblyId)) throw new Error('assemblyId must use lowercase letters, numbers, hyphens, or underscores');
  if (request.anchors.length < 2 || request.anchors.some((anchor) => anchor.length !== 3 || !anchor.every(Number.isFinite))) throw new Error('wall-run requires at least two finite X/Y/Z anchors');
  if (!Number.isInteger(request.seed)) throw new Error('seed must be an integer');
  const diagnostics: ShadoBuilderDiagnostic[] = [];
  const candidates = kit.modules.filter((module) => module.enabled && module.role === 'wall-straight');
  const selectedPool = preferredPool(candidates, kit, request, diagnostics);
  const instances: ShadoBuilderAssemblyInstance[] = [];
  for (let segment = 0; segment < request.anchors.length - 1; segment += 1) {
    const start = request.anchors[segment]!;
    const end = request.anchors[segment + 1]!;
    const dx = end[0] - start[0];
    const dz = end[2] - start[2];
    const distance = Math.hypot(dx, dz);
    if (distance < 0.001) {
      diagnostics.push({ ...error('zero-segment', `Wall segment ${segment} has no horizontal length.`), segment });
      continue;
    }
    const module = selectModule(selectedPool, distance, request.seed + segment);
    const count = Math.max(1, Math.round(distance / module.span));
    const occupied = count * module.span;
    const residual = distance - occupied;
    const tolerance = Math.max(kit.gridSize * 0.05, module.sockets.reduce((value, socket) => Math.max(value, socket.tolerance), 0));
    if (Math.abs(residual) > tolerance) diagnostics.push({ ...warning('span-residual', `Segment ${segment} leaves ${round(Math.abs(residual))} units of ${residual > 0 ? 'gap' : 'overlap'} using ${count} × ${module.id}; add a transition module or move an anchor.`), moduleId: module.id, segment });
    const ux = dx / distance;
    const uz = dz / distance;
    const startOffset = residual / 2;
    const yaw = module.spanAxis === 'x'
      ? -Math.atan2(dz, dx) * 180 / Math.PI
      : Math.atan2(dx, dz) * 180 / Math.PI;
    for (let index = 0; index < count; index += 1) {
      const along = startOffset + module.span * (index + 0.5);
      const t = Math.min(1, Math.max(0, along / distance));
      const instanceId = `s${segment}-p${index}`;
      instances.push({
        id: instanceId,
        moduleId: module.id,
        prototypeId: module.prototype.id,
        quality: module.quality,
        position: [round(start[0] + ux * along), round(start[1] + (end[1] - start[1]) * t + module.groundOffset), round(start[2] + uz * along)],
        rotationDegrees: [0, round(yaw), 0],
        scale: [1, 1, 1],
        segment,
        moduleSnapshot: structuredClone(module),
        metadata: { builderKit: kit.id, moduleRole: module.role, reusableQuality: module.quality },
      });
    }
  }
  if (request.anchors.length > 2 && !kit.modules.some((module) => module.enabled && (module.role === 'wall-corner-inner' || module.role === 'wall-corner-outer'))) diagnostics.push(warning('corner-module-missing', 'The run turns but the kit has no certified corner module; review segment joins before commit.'));
  const qualitySummary = {
    certifiedReusable: instances.filter((instance) => instance.quality === 'certified-reusable').length,
    legacyReusable: instances.filter((instance) => instance.quality === 'legacy-reusable').length,
    proceduralElementary: instances.filter((instance) => instance.quality === 'procedural-elementary').length,
  };
  const inputHash = hashShadoBuilderValue({ kit: `${kit.id}@${kit.revision}`, request });
  const base = {
    kind: 'shado.builder.assembly' as const,
    version: 1 as const,
    id: request.assemblyId,
    kitId: kit.id,
    kitRevision: kit.revision,
    blueprint: 'wall-run' as const,
    state: 'proposal' as const,
    anchors: structuredClone(request.anchors),
    seed: request.seed,
    inputHash,
    instances,
    diagnostics,
    qualitySummary,
    metadata: { ...(request.metadata ?? {}), preferReusablePieces: true },
  };
  return { ...base, outputHash: hashShadoBuilderAssembly(base) };
}

/** Pure blueprint dispatcher shared by Libra's HTTP host and OPFS worker. */
export function solveShadoBuilderAssembly(kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest): ShadoBuilderAssembly {
  const blueprint = request.blueprint ?? 'wall-run';
  if (blueprint === 'wall-run') {
    const wall = solveShadoBuilderWallRun(kit, request);
    const diagnostics = [...wall.diagnostics];
    const instances = applyPinnedInstances(wall.instances, request.pinnedInstances ?? {}, diagnostics, kit);
    return decorateAssembly({ ...wall, instances, diagnostics }, request, kit);
  }
  validateSolveRequest(kit, request);
  const diagnostics: ShadoBuilderDiagnostic[] = [];
  let instances: ShadoBuilderAssemblyInstance[] = [];
  if (blueprint === 'rectangular-building') instances = solveRectangularBuilding(kit, request, diagnostics);
  else if (blueprint === 'road-spline') instances = solveLinearRole(kit, request, diagnostics, ['road', 'bridge'], false);
  else if (blueprint === 'stair-ramp') instances = solveLinearRole(kit, request, diagnostics, ['stair', 'ramp'], false);
  else if (blueprint === 'terrain-transition') instances = solveLinearRole(kit, request, diagnostics, ['terrain-transition'], false);
  else if (blueprint === 'room-graph') instances = solveRoomGraph(kit, request, diagnostics);
  else if (blueprint === 'dressing-field') instances = solveDressingField(kit, request, diagnostics);
  else if (blueprint === 'nested-assembly') instances = solveNestedAssembly(request, diagnostics);
  instances = applyPinnedInstances(instances, request.pinnedInstances ?? {}, diagnostics, kit);
  const base = assemblyBase(kit, request, blueprint, instances, diagnostics);
  return { ...base, outputHash: hashShadoBuilderAssembly(base) };
}

export function validateShadoBuilderStyle(assembly: ShadoBuilderAssembly, kit: ShadoBuilderKit, profile: ShadoBuilderStyleProfile): ShadoBuilderDiagnostic[] {
  const tags = new Set<string>();
  for (const instance of assembly.instances) {
    const module = kit.modules.find((entry) => entry.id === instance.moduleId);
    for (const tag of module?.tags ?? []) tags.add(tag);
  }
  const diagnostics: ShadoBuilderDiagnostic[] = [];
  for (const tag of profile.requiredTags) if (!tags.has(tag)) diagnostics.push(warning('style-required-tag', `${profile.name} expects an authored '${tag}' layer; add a compatible reusable module or record a deliberate exception.`));
  for (const tag of profile.forbiddenTags) if (tags.has(tag)) diagnostics.push(error('style-forbidden-tag', `${profile.name} forbids '${tag}' in production assemblies.`));
  const discoveryCount = profile.discoveryTags.filter((tag) => tags.has(tag)).length;
  if (discoveryCount < profile.discoveryRange.minimum) diagnostics.push(warning('discovery-too-thin', `${profile.name} has ${discoveryCount} discovery layer(s); its World Bible profile expects at least ${profile.discoveryRange.minimum}.`));
  if (discoveryCount > profile.discoveryRange.maximum) diagnostics.push(warning('discovery-too-loud', `${profile.name} has ${discoveryCount} discovery layer(s); restrain clues so the location still reads as an inhabited place rather than exposition.`));
  return diagnostics;
}

export const ELTANIA_BUILDER_STYLE_PROFILES: ShadoBuilderStyleProfile[] = [
  style('talios-crownward', 'Talios Crownward', 'talios_crownward', 'Old civic stone rebuilt over visibly older foundations; formal, inhabited, coastal, and only selectively magical.', ['inhabited-present', 'weathered-coastal-stone'], ['dark-timber', 'bronze-copper', 'salt-stain', 'civic-order'], ['legacy-eq-identity', 'theme-park-facade'], ['older-foundation', 'recovered-archive', 'restrained-returning-magic', 'class-discovery'], ['masonry.primary', 'timber.structural', 'metal.hardware'], 1, 3),
  style('talios-lowharbor', 'Talios Lowharbor', 'talios_lowharbor', 'Working docks, fish markets, shipyards, warehouses, inns, foreign trade, and lower-city shrines.', ['inhabited-present', 'working-waterfront'], ['weathered-coastal-stone', 'dark-timber', 'salt-stain', 'cosmopolitan'], ['legacy-eq-identity', 'pristine-monumentality'], ['smuggler-trace', 'older-foundation', 'foreign-rumor', 'class-discovery'], ['timber.structural', 'masonry.primary', 'metal.hardware'], 1, 4),
  style('greymarch', 'The Greymarch', 'greymarch', 'Rolling lived-in hinterland where readable safety gives way to old roads, burial earthworks, and asymmetric danger.', ['inhabited-present', 'cool-grass'], ['wind-shaped-tree', 'wet-rock', 'old-field-wall', 'ruined-road'], ['legacy-eq-identity', 'uniform-danger'], ['burial-earthwork', 'broken-foundation', 'hidden-path', 'class-discovery'], ['terrain.primary', 'stone.field', 'vegetation.primary'], 1, 4),
  style('greenveil', 'The Greenveil & Alderhollow', 'greenveil', 'Dense vertical woodland and architecture that grows around trees, streams, roots, and gathering clearings.', ['inhabited-present', 'dense-vertical-woodland'], ['elevated-wood-path', 'stone-circle', 'root-reclaimed-stone', 'subtle-primal-magic'], ['legacy-eq-identity', 'stone-city-copy'], ['ecological-anomaly', 'hidden-path', 'old-magical-wound', 'class-discovery'], ['timber.structural', 'vegetation.primary', 'stone.ancient'], 1, 4),
  style('blackroot', 'Blackroot Warrens', 'blackroot_warrens', 'Natural cave progressively occupied and fortified by competent people, with deeper ancient excavation layers.', ['inhabited-present', 'natural-cave'], ['timber-brace', 'stolen-supply', 'cookfire', 'rough-smithing'], ['legacy-eq-identity', 'supernatural-villain-everywhere'], ['excavated-ancient-stone', 'blackened-glass', 'relic-shipment', 'class-discovery'], ['rock.primary', 'timber.structural', 'metal.salvaged'], 1, 4),
  style('mournvault', 'Mournvault', 'mournvault_1', 'Funerary layers descend through different histories toward restrained geometry that does not belong to the tomb culture.', ['funerary-layer', 'restrained-blue-grey-stone'], ['oxidized-metal', 'old-religious-carving', 'water-damage'], ['legacy-eq-identity', 'hollow-star-wallpaper'], ['older-buried-structure', 'black-glass', 'wrong-geometry', 'class-discovery'], ['stone.funeral', 'metal.oxidized', 'stone.ancient'], 1, 3),
];

function solveRectangularBuilding(kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest, diagnostics: ShadoBuilderDiagnostic[]): ShadoBuilderAssemblyInstance[] {
  if (request.anchors.length < 2) throw new Error('rectangular-building requires opposite-corner anchors');
  const a = request.anchors[0]!; const b = request.anchors[1]!;
  const y = a[1];
  const corners: WorldVec3[] = [[a[0], y, a[2]], [b[0], y, a[2]], [b[0], y, b[2]], [a[0], y, b[2]], [a[0], y, a[2]]];
  const wallRequest = { ...request, blueprint: 'wall-run' as const, anchors: corners };
  const wall = solveShadoBuilderWallRun(kit, wallRequest);
  const instances = wall.instances.map((entry) => ({ ...entry, id: `wall-${entry.id}` }));
  diagnostics.push(...wall.diagnostics);
  const floor = bestPool(kit.modules.filter((entry) => entry.enabled && entry.role === 'floor'), kit, request, diagnostics)[0];
  const roof = bestPool(kit.modules.filter((entry) => entry.enabled && entry.role === 'roof'), kit, request, diagnostics)[0];
  const center: WorldVec3 = [round((a[0] + b[0]) / 2), y, round((a[2] + b[2]) / 2)];
  if (floor) instances.push(instanceFor(floor, 'floor-0', center, 4)); else diagnostics.push(warning('building-floor-missing', 'No reusable floor module is available; walls remain previewable without synthesized floor geometry.'));
  if (roof) instances.push(instanceFor(roof, 'roof-0', [center[0], y + Number(request.parameters?.storyHeight ?? 4), center[2]], 5)); else diagnostics.push(warning('building-roof-missing', 'No reusable roof module is available; the building remains intentionally open rather than generating a primitive roof.'));
  return instances;
}

function solveLinearRole(kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest, diagnostics: ShadoBuilderDiagnostic[], roles: ShadoBuilderModuleRole[], closed: boolean): ShadoBuilderAssemblyInstance[] {
  const anchors = closed ? [...request.anchors, request.anchors[0]!] : request.anchors;
  if (anchors.length < 2) throw new Error(`${request.blueprint} requires at least two anchors`);
  const pool = bestPool(kit.modules.filter((entry) => entry.enabled && roles.includes(entry.role)), kit, request, diagnostics);
  if (!pool.length) throw new Error(`kit has no eligible reusable ${roles.join('/')} module`);
  const instances: ShadoBuilderAssemblyInstance[] = [];
  for (let segment = 0; segment < anchors.length - 1; segment += 1) {
    const a = anchors[segment]!; const b = anchors[segment + 1]!; const dx = b[0] - a[0]; const dy = b[1] - a[1]; const dz = b[2] - a[2]; const distance = Math.hypot(dx, dz);
    if (distance < 0.001) continue;
    const module = selectModule(pool, distance, request.seed + segment); const count = Math.max(1, Math.round(distance / module.span));
    for (let index = 0; index < count; index += 1) { const t = (index + 0.5) / count; instances.push(instanceFor(module, `s${segment}-p${index}`, [round(a[0] + dx * t), round(a[1] + dy * t + module.groundOffset), round(a[2] + dz * t)], segment, [0, round(module.spanAxis === 'x' ? -Math.atan2(dz, dx) * 180 / Math.PI : Math.atan2(dx, dz) * 180 / Math.PI), 0])); }
  }
  return instances;
}

function solveRoomGraph(kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest, diagnostics: ShadoBuilderDiagnostic[]): ShadoBuilderAssemblyInstance[] {
  const rooms = Array.isArray(request.parameters?.rooms) ? request.parameters.rooms as Array<{ id?: string; min: WorldVec3; max: WorldVec3 }> : [];
  if (!rooms.length) return solveRectangularBuilding(kit, request, diagnostics);
  const instances: ShadoBuilderAssemblyInstance[] = [];
  rooms.forEach((room, index) => { const solved = solveRectangularBuilding(kit, { ...request, assemblyId: `${request.assemblyId}-room-${index}`, anchors: [room.min, room.max] }, diagnostics); instances.push(...solved.map((entry) => ({ ...entry, id: `room-${room.id ?? index}-${entry.id}`, metadata: { ...entry.metadata, roomId: room.id ?? String(index) } }))); });
  return instances;
}

function solveDressingField(kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest, diagnostics: ShadoBuilderDiagnostic[]): ShadoBuilderAssemblyInstance[] {
  if (request.anchors.length < 2) throw new Error('dressing-field requires minimum and maximum bounds anchors');
  const pool = bestPool(kit.modules.filter((entry) => entry.enabled && ['prop-anchor', 'light-anchor', 'foliage', 'trim'].includes(entry.role)), kit, request, diagnostics);
  if (!pool.length) throw new Error('kit has no eligible reusable dressing module');
  const count = Math.max(1, Math.min(5000, Math.floor(Number(request.parameters?.count ?? 24)))); const a = request.anchors[0]!; const b = request.anchors[1]!;
  return Array.from({ length: count }, (_, index) => { const module = pool[index % pool.length]!; const x = seeded(request.seed, index * 2); const z = seeded(request.seed, index * 2 + 1); return instanceFor(module, `dress-${index}`, [round(a[0] + (b[0] - a[0]) * x), round(a[1]), round(a[2] + (b[2] - a[2]) * z)], index, [0, round(seeded(request.seed, index + 7000) * 360), 0]); });
}

function solveNestedAssembly(request: ShadoBuilderSolveRequest, diagnostics: ShadoBuilderDiagnostic[]): ShadoBuilderAssemblyInstance[] {
  const children = request.nestedAssemblies ?? [];
  if (!children.length) throw new Error('nested-assembly requires nestedAssemblies');
  const instances: ShadoBuilderAssemblyInstance[] = [];
  for (const child of children) { if (child.state !== 'committed') diagnostics.push(warning('nested-proposal', `Nested assembly '${child.id}' is not committed.`)); for (const entry of child.instances) { const moduleId = `${child.id}-${entry.moduleId}`; const prototypeId = `${child.id}-${entry.prototypeId}`; const moduleSnapshot = entry.moduleSnapshot ? { ...structuredClone(entry.moduleSnapshot), id: moduleId, prototype: { ...structuredClone(entry.moduleSnapshot.prototype), id: prototypeId } } : undefined; instances.push({ ...entry, id: `${child.id}-${entry.id}`, moduleId, prototypeId, ...(moduleSnapshot ? { moduleSnapshot } : {}), metadata: { ...entry.metadata, nestedAssembly: child.id, nestedKit: child.kitId, nestedKitRevision: child.kitRevision } }); } }
  return instances;
}

function assemblyBase(kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest, blueprint: ShadoBuilderBlueprintKind, instances: ShadoBuilderAssemblyInstance[], diagnostics: ShadoBuilderDiagnostic[]): Omit<ShadoBuilderAssembly, 'outputHash'> {
  const qualitySummary = { certifiedReusable: instances.filter((entry) => entry.quality === 'certified-reusable').length, legacyReusable: instances.filter((entry) => entry.quality === 'legacy-reusable').length, proceduralElementary: instances.filter((entry) => entry.quality === 'procedural-elementary').length };
  return { kind: 'shado.builder.assembly', version: 1, id: request.assemblyId, kitId: kit.id, kitRevision: kit.revision, blueprint, state: 'proposal', anchors: structuredClone(request.anchors), seed: request.seed, inputHash: hashShadoBuilderValue({ kit: `${kit.id}@${kit.revision}`, request }), instances, diagnostics, qualitySummary, socketEdges: buildSocketEdges(instances, kit), ...(request.paletteId ? { paletteId: request.paletteId } : {}), ...(request.styleProfileId ? { styleProfileId: request.styleProfileId } : {}), parameters: structuredClone(request.parameters ?? {}), nestedAssemblyIds: request.nestedAssemblies?.map((entry) => entry.id) ?? [], metadata: { ...(request.metadata ?? {}), preferReusablePieces: true } };
}

function decorateAssembly(assembly: ShadoBuilderAssembly, request: ShadoBuilderSolveRequest, kit: ShadoBuilderKit): ShadoBuilderAssembly { const base = { ...assembly, socketEdges: buildSocketEdges(assembly.instances, kit), ...(request.paletteId ? { paletteId: request.paletteId } : {}), ...(request.styleProfileId ? { styleProfileId: request.styleProfileId } : {}), parameters: structuredClone(request.parameters ?? {}), nestedAssemblyIds: request.nestedAssemblies?.map((entry) => entry.id) ?? [] }; return { ...base, outputHash: hashShadoBuilderAssembly(base) }; }
function validateSolveRequest(kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest): void { if (request.kitId !== kit.id || request.kitRevision !== kit.revision) throw new Error('kit revision mismatch'); if (!validId(request.assemblyId)) throw new Error('invalid assemblyId'); if (!Number.isInteger(request.seed)) throw new Error('seed must be an integer'); if (request.anchors.some((anchor) => !anchor.every(Number.isFinite))) throw new Error('anchors must be finite X/Y/Z values'); }
function bestPool(candidates: ShadoBuilderModule[], kit: ShadoBuilderKit, request: ShadoBuilderSolveRequest, diagnostics: ShadoBuilderDiagnostic[]): ShadoBuilderModule[] { if (!candidates.length) return []; return preferredPool(candidates, kit, request, diagnostics); }
function instanceFor(module: ShadoBuilderModule, id: string, position: WorldVec3, segment: number, rotationDegrees: WorldVec3 = [0, 0, 0]): ShadoBuilderAssemblyInstance { return { id, moduleId: module.id, prototypeId: module.prototype.id, quality: module.quality, position, rotationDegrees, scale: [1, 1, 1], segment, ownership: 'parametric', moduleSnapshot: structuredClone(module), metadata: { moduleRole: module.role, reusableQuality: module.quality } }; }
function applyPinnedInstances(instances: ShadoBuilderAssemblyInstance[], pins: ShadoBuilderSolveRequest['pinnedInstances'], diagnostics: ShadoBuilderDiagnostic[], kit: ShadoBuilderKit): ShadoBuilderAssemblyInstance[] { return instances.map((entry) => { const pin = pins?.[entry.id]; if (!pin) return entry; let base = entry; if (pin.moduleId && pin.moduleId !== entry.moduleId) { const module = kit.modules.find((candidate) => candidate.id === pin.moduleId && candidate.enabled); if (!module) diagnostics.push({ code: 'pinned-module-unknown', severity: 'error', message: `Pinned instance '${entry.id}' references unknown or disabled module '${pin.moduleId}'.`, instanceId: entry.id, moduleId: pin.moduleId }); else { diagnostics.push({ code: 'pinned-module', severity: 'info', message: `Pinned instance '${entry.id}' keeps module '${pin.moduleId}'.`, instanceId: entry.id, moduleId: pin.moduleId }); base = { ...entry, moduleId: module.id, prototypeId: module.prototype.id, quality: module.quality, moduleSnapshot: structuredClone(module), metadata: { ...entry.metadata, moduleRole: module.role, reusableQuality: module.quality } }; } } const { moduleId: _moduleId, ...transformPin } = pin; return { ...base, ...transformPin, ownership: 'pinned', metadata: { ...base.metadata, pinned: true } }; }); }
function buildSocketEdges(instances: ShadoBuilderAssemblyInstance[], kit: ShadoBuilderKit): NonNullable<ShadoBuilderAssembly['socketEdges']> { const edges: NonNullable<ShadoBuilderAssembly['socketEdges']> = []; const groups = new Map<number, ShadoBuilderAssemblyInstance[]>(); for (const entry of instances) groups.set(entry.segment, [...(groups.get(entry.segment) ?? []), entry]); for (const entries of groups.values()) for (let index = 0; index < entries.length - 1; index += 1) { const from = entries[index]!; const to = entries[index + 1]!; const fromSockets = kit.modules.find((entry) => entry.id === from.moduleId)?.sockets ?? []; const toSockets = kit.modules.find((entry) => entry.id === to.moduleId)?.sockets ?? []; const fromSocket = fromSockets.find((entry) => entry.id === 'end') ?? fromSockets.at(-1); const toSocket = toSockets.find((entry) => entry.id === 'start') ?? toSockets[0]; if (fromSocket && toSocket && (fromSocket.accepts.includes(toSocket.type) || toSocket.accepts.includes(fromSocket.type))) edges.push({ fromInstance: from.id, fromSocket: fromSocket.id, toInstance: to.id, toSocket: toSocket.id }); } return edges; }
function seeded(seed: number, index: number): number { let value = Math.imul(seed ^ index, 0x45d9f3b); value = Math.imul(value ^ value >>> 16, 0x45d9f3b); return ((value ^ value >>> 16) >>> 0) / 0xffffffff; }
function style(id: string, name: string, region: string, summary: string, requiredTags: string[], encouragedTags: string[], forbiddenTags: string[], discoveryTags: string[], materialRoles: string[], minimum: number, maximum: number): ShadoBuilderStyleProfile { return { id, revision: 1, name, region, summary, requiredTags, encouragedTags, forbiddenTags, discoveryTags, materialRoles, discoveryRange: { minimum, maximum }, metadata: { source: 'docs/shadows-of-eltania/01-world-bible.md', philosophy: 'coherence-with-discovery' } }; }

export function hashShadoBuilderValue(value: unknown): string {
  const source = stableStringify(value);
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < source.length; index += 1) {
    const code = source.charCodeAt(index);
    first ^= code;
    first = Math.imul(first, 0x01000193) >>> 0;
    second ^= code + index;
    second = Math.imul(second, 0x85ebca6b) >>> 0;
  }
  return `${first.toString(16).padStart(8, '0')}${second.toString(16).padStart(8, '0')}`;
}

/**
 * Hashes the authored assembly graph while deliberately excluding lifecycle state.
 * Preview approval therefore remains valid when the exact proposal is marked committed.
 */
export function hashShadoBuilderAssembly(value: Omit<ShadoBuilderAssembly, 'outputHash'> | ShadoBuilderAssembly): string {
  const { outputHash: _outputHash, state: _state, ...graph } = value as ShadoBuilderAssembly;
  return hashShadoBuilderValue(graph);
}

function preferredPool(candidates: ShadoBuilderModule[], kit: ShadoBuilderKit, request: ShadoBuilderWallRunRequest, diagnostics: ShadoBuilderDiagnostic[]): ShadoBuilderModule[] {
  const certified = candidates.filter((module) => module.quality === 'certified-reusable' && module.provenance.status === 'approved');
  if (certified.length) return certified;
  const legacy = candidates.filter((module) => module.quality === 'legacy-reusable' && module.provenance.status !== 'restricted-reference');
  if (legacy.length) {
    diagnostics.push(warning('legacy-reusable-selected', 'No certified reusable wall piece is available; the solver selected a legacy reusable module.'));
    return legacy;
  }
  const procedural = candidates.filter((module) => module.quality === 'procedural-elementary' && module.provenance.status !== 'restricted-reference');
  const reason = request.proceduralFallbackReason?.trim();
  if (!procedural.length) throw new Error('kit has no eligible wall-straight module');
  if (!kit.allowProceduralFallback || !request.allowProceduralFallback || !reason) throw new Error('only procedural elementary geometry is available; an enabled policy and explicit fallback reason are required');
  diagnostics.push(warning('procedural-fallback-selected', `Procedural elementary geometry was selected because no reusable piece exists. Reason: ${reason}`));
  return procedural;
}

function selectModule(candidates: ShadoBuilderModule[], distance: number, seed: number): ShadoBuilderModule {
  return [...candidates].sort((left, right) => {
    const leftResidual = Math.abs(distance - Math.max(1, Math.round(distance / left.span)) * left.span);
    const rightResidual = Math.abs(distance - Math.max(1, Math.round(distance / right.span)) * right.span);
    if (leftResidual !== rightResidual) return leftResidual - rightResidual;
    if (left.weight !== right.weight) return right.weight - left.weight;
    const leftRank = hashShadoBuilderValue(`${seed}:${left.id}`);
    const rightRank = hashShadoBuilderValue(`${seed}:${right.id}`);
    return leftRank.localeCompare(rightRank) || left.id.localeCompare(right.id);
  })[0]!;
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`).join(',')}}`;
}

function validId(value: string): boolean { return /^[a-z0-9][a-z0-9_-]{0,127}$/.test(value); }
function structuralRole(role: ShadoBuilderModuleRole): boolean { return ['foundation', 'floor', 'wall-straight', 'wall-corner-inner', 'wall-corner-outer', 'wall-cap', 'opening-door', 'opening-window', 'column', 'roof', 'stair', 'ramp', 'road', 'bridge', 'terrain-transition'].includes(role); }
function round(value: number): number { return Math.round(value * 10_000) / 10_000; }
function error(code: string, message: string, moduleId?: string): ShadoBuilderDiagnostic { return { code, severity: 'error', message, ...(moduleId ? { moduleId } : {}) }; }
function warning(code: string, message: string, moduleId?: string): ShadoBuilderDiagnostic { return { code, severity: 'warning', message, ...(moduleId ? { moduleId } : {}) }; }
