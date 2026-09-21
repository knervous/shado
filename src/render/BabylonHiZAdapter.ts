import {
  BABYLON,
  type AbstractMesh,
  type Camera,
  type Mesh,
  type Observer,
  type Scene,
  type ShaderMaterial,
  type StorageBuffer,
  type SubMesh,
  type WebGPUEngine,
} from '../babylon';
// Deep import: the client's runtime Babylon facade leaves DepthRenderer out of
// the game's core chunk; only a page that loads the Hi-Z entry pays for it.
import { DepthRenderer } from '@babylonjs/core/Rendering/depthRenderer.js';
// The instanced material's `#include<sceneUboDeclaration>`; a slim runtime
// facade may not have registered it, and an unregistered include is fetched
// over HTTP.
import '@babylonjs/core/ShadersWGSL/ShadersInclude/sceneUboDeclaration.js';
import { ShadoWorldHiZ, type ShadoWorldHiZRunResult } from '../world/hiz/ShadoWorldHiZ';
import { probeHiZCapability, type HiZCapabilityReport } from './hiz-capability-probe';
import { SHADO_HIZ_DRAW_ARGS_WORDS } from '../world/hiz/wgsl';
import type { ShadoHiZBatch, ShadoHiZCandidate, ShadoHiZViewInput } from '../world/hiz/types';

/**
 * The ONE place Hi-Z touches Babylon internals (docs/pvs-hiz-prototype.md H3).
 *
 * Sequence per frame, all on Babylon's single frame encoder:
 *   occluder depth RTT (DepthRenderer, this camera only, full resolution)
 *   -> onBeforeDrawPhase: ShadoWorldHiZ seed/reduce/reset/cull/finalize
 *   -> copy each batch's 20-byte argument block into the indirect buffer of
 *      that submesh's draw context FOR THIS CAMERA'S RENDER PASS
 *   -> Babylon's main pass issues drawIndexedIndirect.
 *
 * Draw contexts are per (submesh, render pass id) and every camera owns a
 * pass id, so shadow maps, reflections, the depth prepass and other cameras
 * keep drawing normally. Nothing here calls setEnabled or reads results back.
 */

/** Babylon internals the bridge relies on; checked once at construction. */
interface WebGPUDrawContextLike {
  enableIndirectDraw: boolean;
  indirectDrawBuffer?: GPUBuffer;
  _currentInstanceCount: number;
}

export interface BabylonHiZMeshTarget {
  readonly mesh: Mesh;
  /** Stable id reported back in flags/debug (defaults to mesh.uniqueId). */
  readonly id?: number;
}

/**
 * An instanced prototype drawn by one mesh with `material` from
 * `createBabylonHiZInstancedMaterial`. Instance transforms live in a storage
 * buffer; the vertex shader fetches them through the compacted member list.
 */
export interface BabylonHiZInstancedTarget {
  readonly mesh: Mesh;
  readonly material: ShaderMaterial;
  /** Column-major world matrices, 16 floats per member. */
  readonly matrices: Float32Array;
  /** World AABB per member: minX,minY,minZ,maxX,maxY,maxZ. */
  readonly bounds: Float32Array;
  readonly idBase?: number;
}

export type BabylonHiZMode = 'off' | 'hiz';

export interface BabylonHiZStatus {
  readonly backend: 'webgpu' | 'webgl2' | 'other';
  readonly requested: BabylonHiZMode;
  readonly actual: BabylonHiZMode;
  readonly reason: string;
  readonly babylonVersion: string;
  readonly frameId: number;
  readonly candidates: number;
  readonly batches: number;
  readonly attachedDrawContexts: number;
  readonly lastRun?: ShadoWorldHiZRunResult;
  readonly depthSize: readonly [number, number];
  /** Depth prepass renders so far; stays flat while off. */
  readonly depthRenders: number;
  /** H0.5 capability probe on this device ('pending' until it finishes). */
  readonly capability: HiZCapabilityReport | 'pending';
  readonly deviceLosses: number;
  /** GPU bytes held by the pyramid, tables and outputs (not the depth target). */
  readonly gpuBytes: number;
  /** Hi-Z depth targets registered on the camera: 1 while on, 0 while off. */
  readonly depthTargets: number;
  /**
   * Per-frame cost, exponential averages over recent frames. CPU is encode
   * time on the main thread (depth prepass render call; cull dispatches +
   * argument copies). GPU compute is timestamp-measured when available;
   * Babylon exposes no per-render-target timestamps, so the depth prepass's
   * GPU time is not separable (null here, always).
   */
  readonly cost: { cpuDepthMs: number; cpuCullMs: number; gpuComputeMs: number | null; gpuDepthMs: null };
  readonly convention: 'normal' | 'reversed';
  readonly topLeftOrigin: boolean;
  readonly errors: readonly string[];
}

interface BatchBinding {
  batch: number;
  /** Index of this batch's (single) candidate; -1 for instanced targets. */
  candidate: number;
  subMesh: SubMesh;
  mesh: Mesh;
  /**
   * `mesh`: a plain or thin-instanced mesh culled whole by its bound, drawn
   * all-or-nothing with its current instance count. `instanced`: a compacted
   * batch drawn through createBabylonHiZInstancedMaterial.
   */
  kind: 'mesh' | 'instanced';
  /** Instance count Babylon itself will ask for; attach pins it. */
  babylonInstances: number;
  /** Last world bound published for `candidate` (float64: compared to Babylon's doubles). */
  bound: Float64Array;
  context?: WebGPUDrawContextLike;
}

/** Babylon releases below which the private draw-context bridge is unverified. */
const MIN_BABYLON = [9, 27, 1];

export class BabylonHiZAdapter {
  private hizCore?: ShadoWorldHiZ;
  /** The GPU core (undefined when the backend is unsupported). */
  public get hiz(): ShadoWorldHiZ | undefined {
    return this.hizCore;
  }
  /** H0.5: nothing is rejected until the device proved compute -> indirect draw. */
  private capability: HiZCapabilityReport | 'pending' = 'pending';
  private contextObservers: Array<() => void> = [];
  private lastTargets: { meshes: readonly BabylonHiZMeshTarget[]; instanced: readonly BabylonHiZInstancedTarget[] } | null = null;
  private deviceLosses = 0;
  private depth?: DepthRenderer;
  private readonly engine: WebGPUEngine;
  private bindings: BatchBinding[] = [];
  private instanced: Array<{ target: BabylonHiZInstancedTarget; batch: number; matrices: StorageBuffer }> = [];
  private retired: Array<{ target: BabylonHiZInstancedTarget; batch: number; matrices: StorageBuffer }> = [];
  private candidateIds: number[] = [];
  private occluders: AbstractMesh[] = [];
  private mode: BabylonHiZMode = 'off';
  private reason = 'not started';
  private lastRun?: ShadoWorldHiZRunResult;
  private beforeDraw?: Observer<Scene>;
  private resizeObserver?: Observer<unknown>;
  private readonly depthMatrix = new Float32Array(16);
  private depthFrame = -1;
  /** The current depth target has rendered with every occluder ready at least once. */
  private depthReady = false;
  /** Depth-prepass renders since construction (V2: zero while off). */
  private depthRenders = 0;
  private depthStartedAt = 0;
  /** Bumped by setTargets; debug readbacks spanning a rebuild are stale. */
  private targetGeneration = 0;
  private costDepth = 0;
  private costCull = 0;
  /** Occluder submeshes the depth pass draws, for the readiness sweep. */
  private occluderSubMeshes: Array<{ subMesh: SubMesh; instanced: boolean }> = [];
  private readyCursor = 0;
  private readonly errors: string[] = [];
  /**
   * Pixel row 0 of the depth texture is the top of the view. Read from the
   * depth target every frame: Babylon's WebGPU engine renders into render
   * targets y-flipped unless the wrapper sets `_disableEngineYFlip`.
   */
  public topLeftOrigin = false;
  /**
   * Profiling only: stop the per-frame sequence after a stage to attribute
   * cost. 'depth' renders the prepass and nothing else; 'compute' also runs
   * the passes but attaches nothing (draws stay ordinary); 'full' is normal.
   * Anything but 'full' culls nothing.
   */
  public debugStage: 'full' | 'depth' | 'compute' | 'no-copy' = 'full';
  /**
   * Mesh targets drawing fewer triangles than this (index count / 3 x
   * instances, this frame) stay on the ordinary draw. Every indirect draw has
   * a fixed GPU cost -- measured on Crownward at ~7 ms for ~1,700 of them --
   * so culling a small mesh can cost more than drawing it.
   */
  public minTargetTriangles = 0;
  /** Depth convention; follows engine.useReverseDepthBuffer. */
  public convention: 'normal' | 'reversed' = 'normal';

  public constructor(
    private readonly scene: Scene,
    private readonly camera: Camera
  ) {
    this.engine = scene.getEngine() as WebGPUEngine;
    const unsupported = BabylonHiZAdapter.unsupportedReason(scene);
    if (unsupported) {
      this.reason = unsupported;
      return;
    }
    if ((this.engine as any).useReverseDepthBuffer) this.convention = 'reversed';
    this.hizCore = new ShadoWorldHiZ(this.engine);
    this.runProbe();
    // Device loss (Babylon rebuilds its own device when it handles loss):
    // every draw context and GPU object of ours is gone. Admit, then rebuild
    // on restore and prove the new device before culling again.
    const lost = this.engine.onContextLostObservable.add(() => {
      this.deviceLosses++;
      this.capability = 'pending';
      for (const binding of this.bindings) binding.context = undefined;
      this.reason = 'device lost';
    });
    const restored = this.engine.onContextRestoredObservable.add(() => this.rebuildAfterDeviceLoss());
    this.contextObservers.push(
      () => this.engine.onContextLostObservable.remove(lost),
      () => this.engine.onContextRestoredObservable.remove(restored)
    );
    // No depth target until Hi-Z is switched on: off costs no rendering work.
    this.resizeObserver = this.engine.onResizeObservable.add(() => {
      if (this.mode === 'hiz') this.createDepth();
    }) as Observer<unknown>;
    this.beforeDraw = scene.onBeforeDrawPhaseObservable.add(() => this.onBeforeDraw());
    this.reason = 'idle';
  }

  /** Why this scene cannot run WebGPU Hi-Z, or '' when it can. */
  public static unsupportedReason(scene: Scene): string {
    const engine = scene.getEngine() as any;
    if (!engine.isWebGPU) return 'backend is not WebGPU';
    if (!engine._device || !engine._renderEncoder) return 'WebGPU device/encoder missing';
    const version = String(BABYLON.Engine.Version ?? '').split('-')[0]!.split('.').map(Number);
    for (let i = 0; i < 3; i++) {
      if ((version[i] ?? 0) > MIN_BABYLON[i]!) break;
      if ((version[i] ?? 0) < MIN_BABYLON[i]!) return `Babylon ${BABYLON.Engine.Version} < ${MIN_BABYLON.join('.')}`;
    }
    if (typeof engine._endCurrentRenderPass !== 'function') return 'Babylon WebGPU pass API changed';
    if (engine.snapshotRendering) return 'snapshot rendering is on (would replay stale arguments)';
    return '';
  }

  /** Static opaque occluders rendered into the depth prepass. */
  public setOccluders(meshes: readonly AbstractMesh[]): void {
    this.occluders = [...meshes];
    if (this.depth) this.depth.getDepthMap().renderList = this.occluders;
    this.occluderSubMeshes = [];
    for (const mesh of this.occluders) {
      const instanced = (mesh as any).hasThinInstances === true || ((mesh as any).instances?.length ?? 0) > 0;
      for (const subMesh of mesh.subMeshes ?? []) {
        const material = subMesh.getMaterial();
        // The same skips DepthRenderer itself applies.
        if (!material || material.disableDepthWrite || subMesh.verticesCount === 0) continue;
        this.occluderSubMeshes.push({ subMesh, instanced });
      }
    }
    // New occluders may still be compiling their depth effect.
    this.depthReady = false;
    this.readyCursor = 0;
  }

  /**
   * Babylon skips an occluder whose depth effect is still compiling, and a
   * fresh target's first passes are empty. A partial depth only admits more,
   * so this gates honesty rather than safety: nothing is reported as culling
   * until every occluder submesh's depth effect is ready. Bounded per frame,
   * latched once a sweep completes.
   */
  private sweepDepthReadiness(): boolean {
    if (this.depthReady) return true;
    const depth = this.depth;
    if (!depth) return false;
    for (let budget = 256; budget > 0 && this.readyCursor < this.occluderSubMeshes.length; budget--) {
      const { subMesh, instanced } = this.occluderSubMeshes[this.readyCursor]!;
      if (!subMesh.getMesh().isDisposed() && !depth.isReady(subMesh, instanced)) return false;
      this.readyCursor++;
    }
    this.depthReady = this.readyCursor >= this.occluderSubMeshes.length;
    return this.depthReady;
  }

  /**
   * Publishes what Hi-Z may cull. Each indexed submesh of a plain or
   * thin-instanced mesh is one all-or-nothing batch tested by the mesh's
   * world bound (for thin instances, Babylon's instance-inclusive bound; the
   * owner must call thinInstanceRefreshBoundingInfo when it rewrites them).
   * Instance counts and bounds are re-read every frame, so streaming thin
   * instances needs no rebuild; adding/removing meshes does.
   *
   * Never culled (left on the ordinary draw): non-indexed submeshes, meshes
   * driving `forcedInstanceCount` themselves (their instances live on the
   * GPU, outside Babylon's bound), and materials whose draw wrapper is shared
   * across meshes (`_storeEffectOnSubMeshes` false).
   */
  public setTargets(meshes: readonly BabylonHiZMeshTarget[], instanced: readonly BabylonHiZInstancedTarget[] = []): void {
    if (!this.hiz) return;
    this.lastTargets = { meshes: [...meshes], instanced: [...instanced] };
    this.targetGeneration++;
    this.detachAll();
    // An instanced target dropped from the set still draws every member
    // through its material, so it keeps its matrices and stays bound (to the
    // new tables, uncompacted) until it is published again or disposed.
    const kept = new Set(instanced.map((target) => target.mesh));
    for (const entry of [...this.instanced, ...this.retired]) {
      if (kept.has(entry.target.mesh)) entry.matrices.dispose();
      else if (!this.retired.includes(entry)) this.retired.push(entry);
    }
    this.retired = this.retired.filter((entry) => !kept.has(entry.target.mesh) && !entry.target.mesh.isDisposed());
    this.instanced = [];
    const candidates: ShadoHiZCandidate[] = [];
    const batches: ShadoHiZBatch[] = [];
    const bindings: BatchBinding[] = [];
    this.candidateIds = [];
    for (const target of meshes) {
      const mesh = target.mesh;
      if (mesh.isDisposed() || !mesh.getIndices()?.length || mesh.forcedInstanceCount > 0) continue;
      // Not refreshBoundingInfo(): on a thin-instanced mesh that would drop
      // the instance extents back to the prototype's.
      mesh.computeWorldMatrix(true);
      const box = mesh.getBoundingInfo().boundingBox;
      const instances = thinInstanceCount(mesh);
      for (const subMesh of mesh.subMeshes ?? []) {
        const material = subMesh.getMaterial();
        if (material && !(material as any)._storeEffectOnSubMeshes) continue;
        const batch = batches.length;
        batches.push({ indexCount: subMesh.indexCount, firstIndex: subMesh.indexStart, capacity: 1, wholeInstances: instances });
        const id = target.id ?? mesh.uniqueId;
        const candidate = candidates.length;
        candidates.push({
          id,
          min: [box.minimumWorld.x, box.minimumWorld.y, box.minimumWorld.z],
          max: [box.maximumWorld.x, box.maximumWorld.y, box.maximumWorld.z],
          batch,
          member: 0,
        });
        this.candidateIds.push(id);
        bindings.push({
          batch,
          candidate,
          subMesh,
          mesh,
          kind: 'mesh',
          babylonInstances: instances,
          bound: Float64Array.of(
            box.minimumWorld.x, box.minimumWorld.y, box.minimumWorld.z,
            box.maximumWorld.x, box.maximumWorld.y, box.maximumWorld.z
          ),
        });
      }
    }
    for (const target of instanced) {
      const count = Math.floor(target.bounds.length / 6);
      const subMesh = target.mesh.subMeshes?.[0];
      if (!count || !subMesh || !target.mesh.getIndices()?.length) continue;
      const batch = batches.length;
      batches.push({ indexCount: subMesh.indexCount, firstIndex: subMesh.indexStart, capacity: count });
      for (let i = 0; i < count; i++) {
        const b = target.bounds;
        const id = (target.idBase ?? 0) + i;
        candidates.push({
          id,
          min: [b[i * 6]!, b[i * 6 + 1]!, b[i * 6 + 2]!],
          max: [b[i * 6 + 3]!, b[i * 6 + 4]!, b[i * 6 + 5]!],
          batch,
          member: i,
        });
        this.candidateIds.push(id);
      }
      const matrices = new BABYLON.StorageBuffer(this.engine, Math.max(64, target.matrices.byteLength), BABYLON.Constants.BUFFER_CREATIONFLAG_READWRITE, 'Shado Hi-Z instance matrices');
      matrices.update(target.matrices);
      target.mesh.forcedInstanceCount = count;
      // The prototype's own box says nothing about where members are; the
      // members' bounds are the cull input.
      target.mesh.alwaysSelectAsActiveMesh = true;
      bindings.push({ batch, candidate: -1, subMesh, mesh: target.mesh, kind: 'instanced', babylonInstances: count, bound: new Float64Array(6) });
      this.instanced.push({ target, batch, matrices });
    }
    this.hiz.setCandidates(candidates, batches);
    this.bindings = bindings;
    this.bindInstancedMaterials(false);
  }

  /** Candidate index -> caller id, for decoding debug flags. */
  public get ids(): readonly number[] {
    return this.candidateIds;
  }

  /**
   * 'off' removes the depth prepass target and detaches every indirect draw:
   * no depth render, compute or copy runs while off. 'hiz' creates exactly one
   * fresh target, and nothing is rejected until it has rendered this frame.
   */
  public setMode(mode: BabylonHiZMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    if (mode === 'off') {
      this.detachAll();
      this.disposeDepth();
      this.lastRun = undefined;
      this.reason = 'mode off';
    } else if (this.hiz) {
      this.createDepth();
    }
  }

  public status(): BabylonHiZStatus {
    const size = this.depth?.getDepthMap().getSize();
    return {
      backend: (this.engine as any).isWebGPU ? 'webgpu' : (this.engine as any).webGLVersion === 2 ? 'webgl2' : 'other',
      requested: this.mode,
      actual:
        this.hiz &&
        this.mode === 'hiz' &&
        this.capability !== 'pending' &&
        this.capability.ok &&
        this.bindings.some((binding) => binding.context) &&
        this.lastRun?.complete &&
        !this.lastRun.admitAll
          ? 'hiz'
          : 'off',
      reason: this.reason,
      babylonVersion: BABYLON.Engine.Version,
      frameId: this.engine.frameId,
      candidates: this.hiz?.candidates ?? 0,
      batches: this.hiz?.batches ?? 0,
      attachedDrawContexts: this.bindings.filter(b => b.context).length,
      lastRun: this.lastRun,
      depthSize: [size?.width ?? 0, size?.height ?? 0],
      depthRenders: this.depthRenders,
      capability: this.capability,
      deviceLosses: this.deviceLosses,
      gpuBytes: this.hiz?.gpuBytes() ?? 0,
      cost: {
        cpuDepthMs: this.costDepth,
        cpuCullMs: this.costCull,
        gpuComputeMs: this.hiz?.gpuComputeMs() ?? null,
        gpuDepthMs: null,
      },
      depthTargets: this.camera.customRenderTargets.filter((target) => (target as any).__shadoHiZ).length,
      convention: this.convention,
      topLeftOrigin: this.topLeftOrigin,
      errors: [...this.errors, ...(this.hiz?.lastErrors ?? [])],
    };
  }

  /** The occluder depth render target (debug view). */
  public get depthTexture() {
    return this.depth?.getDepthMap();
  }

  private runProbe(): void {
    const device = (this.engine as any)._device as GPUDevice | undefined;
    if (!device) {
      this.capability = { ok: false, reason: 'no WebGPU device', limits: {}, optional: { timestampQuery: false, indirectFirstInstance: false }, ms: 0 };
      return;
    }
    this.capability = 'pending';
    void probeHiZCapability(device).then((report) => {
      if ((this.engine as any)._device !== device) return; // superseded by a device loss
      this.capability = report;
      if (!report.ok) {
        this.detachAll();
        this.disposeDepth();
        this.reason = `capability probe failed: ${report.reason}`;
      }
    });
  }

  private rebuildAfterDeviceLoss(): void {
    this.disposeDepth();
    for (const entry of [...this.instanced, ...this.retired]) entry.matrices.dispose();
    this.instanced = [];
    this.retired = [];
    this.bindings = [];
    this.hizCore = new ShadoWorldHiZ(this.engine);
    this.runProbe();
    if (this.lastTargets) this.setTargets(this.lastTargets.meshes, this.lastTargets.instanced);
    this.setOccluders(this.occluders);
    if (this.mode === 'hiz') this.createDepth();
    this.reason = 'rebuilt after device loss';
  }

  public dispose(): void {
    for (const remove of this.contextObservers.splice(0)) remove();
    this.detachAll();
    if (this.beforeDraw) this.scene.onBeforeDrawPhaseObservable.remove(this.beforeDraw);
    if (this.resizeObserver) this.engine.onResizeObservable.remove(this.resizeObserver as any);
    this.disposeDepth();
    for (const entry of [...this.instanced, ...this.retired]) {
      entry.target.mesh.forcedInstanceCount = 0;
      entry.matrices.dispose();
    }
    this.instanced = [];
    this.retired = [];
    this.hiz?.dispose();
  }

  private createDepth(): void {
    this.disposeDepth();
    // Full render resolution, single sample, float, gl_FragCoord.z; clear = far.
    const depth = new DepthRenderer(this.scene, BABYLON.Constants.TEXTURETYPE_FLOAT, this.camera, true, BABYLON.Constants.TEXTURE_NEAREST_SAMPLINGMODE, false, 'Shado Hi-Z occluder depth');
    depth.clearColor = new BABYLON.Color4(this.convention === 'normal' ? 1 : 0, 0, 0, 1);
    const map = depth.getDepthMap();
    map.renderList = this.occluders;
    (map as any).__shadoHiZ = true;
    map.onBeforeRenderObservable.add(() => {
      this.depthMatrix.set(this.scene.getTransformMatrix().m);
      this.depthFrame = this.engine.frameId;
      this.depthRenders++;
      this.depthStartedAt = performance.now();
    });
    map.onAfterRenderObservable.add(() => {
      this.costDepth = ema(this.costDepth, performance.now() - this.depthStartedAt);
    });
    // Not registered in scene._depthRenderer (that slot belongs to whatever
    // enableDepthRenderer() the game uses): a custom target of THIS camera
    // renders once per frame for it, before its main pass.
    this.camera.customRenderTargets.push(map);
    this.depth = depth;
    // A new target owns nothing yet: no rejection until it has rendered.
    this.depthFrame = -1;
    this.depthReady = false;
    this.readyCursor = 0;
    this.depthMatrix.fill(NaN);
    this.detachAll();
    this.hiz?.resize(map.getSize().width, map.getSize().height);
  }

  private disposeDepth(): void {
    if (!this.depth) return;
    const map = this.depth.getDepthMap();
    const at = this.camera.customRenderTargets.indexOf(map);
    if (at >= 0) this.camera.customRenderTargets.splice(at, 1);
    this.depth.dispose();
    this.depth = undefined;
  }

  private onBeforeDraw(): void {
    if (!this.hiz || this.scene.activeCamera !== this.camera) return;
    if (this.mode !== 'hiz') {
      this.reason = 'mode off';
      return;
    }
    const began = performance.now();
    try {
      this.cullForDraw();
    } finally {
      this.costCull = ema(this.costCull, performance.now() - began);
    }
  }

  /**
   * Debug only (a readback a frame or two late; nothing on the render path
   * waits for it): the mesh targets the last cull rejected, and the triangles
   * Babylon submitted for enabled mesh targets vs. what the cull let through.
   */
  public async debugCull(): Promise<{
    rejected: Mesh[];
    targetTriangles: number;
    submittedTriangles: number;
  } | 'stale' | null> {
    const hiz = this.hiz;
    if (!hiz || this.status().actual !== 'hiz') return null;
    const generation = this.targetGeneration;
    const [flags, args] = await Promise.all([hiz.readFlags(), hiz.readDrawArgs()]);
    // Targets rebuilt while reading (objects streaming): these flags describe
    // a candidate set that no longer exists.
    if (generation !== this.targetGeneration) return 'stale';
    const rejected = new Set<Mesh>();
    let targetTriangles = 0;
    let submittedTriangles = 0;
    for (const binding of this.bindings) {
      if (binding.kind !== 'mesh' || binding.mesh.isDisposed() || !binding.mesh.isEnabled()) continue;
      const triangles = (binding.subMesh.indexCount / 3) * binding.babylonInstances;
      targetTriangles += triangles;
      // Only an attached draw context actually draws from the arguments;
      // a released one (below minTargetTriangles, shared wrapper...) draws
      // everything whatever the cull said.
      if (!binding.context) {
        submittedTriangles += triangles;
        continue;
      }
      const drawn = args[binding.batch * SHADO_HIZ_DRAW_ARGS_WORDS + 1] ?? binding.babylonInstances;
      submittedTriangles += (binding.subMesh.indexCount / 3) * Math.min(drawn, binding.babylonInstances);
      if (binding.candidate >= 0 && flags[binding.candidate] === 0) rejected.add(binding.mesh);
    }
    return { rejected: [...rejected], targetTriangles, submittedTriangles };
  }

  private cullForDraw(): void {
    if (this.capability === 'pending' || !this.capability.ok) {
      this.detachAll();
      this.reason = this.capability === 'pending' ? 'probing device capability' : `capability probe failed: ${this.capability.reason}`;
      return;
    }
    if (this.debugStage === 'depth') {
      this.detachAll();
      this.reason = 'profiling: depth prepass only';
      return;
    }
    if (this.mode !== 'hiz') {
      this.reason = 'mode off';
      return;
    }
    if (!this.bindings.length) {
      this.reason = 'no targets';
      return;
    }
    const map = this.depth?.getDepthMap();
    const matrix = this.scene.getTransformMatrix();
    let admit = '';
    if (!map || this.depthFrame !== this.engine.frameId) admit = 'depth not rendered this frame';
    else if (!this.sweepDepthReadiness()) admit = 'depth pass not ready';
    else if (!sameMatrix(matrix.m, this.depthMatrix)) admit = 'view changed after the depth pass';
    else if (!this.occluders.length) admit = 'no occluders';
    this.topLeftOrigin = !!(map?.renderTarget as any)?._disableEngineYFlip;
    const view: ShadoHiZViewInput = {
      frameId: this.engine.frameId,
      viewProjection: matrix.m,
      viewportWidth: this.engine.getRenderWidth(),
      viewportHeight: this.engine.getRenderHeight(),
      convention: this.convention,
      ndcHalfZRange: (this.engine as any).isNDCHalfZRange === true,
      topLeftOrigin: this.topLeftOrigin,
      worldEpoch: 0,
      opaqueEpoch: 0,
    };
    const run = this.hiz!.run(map ?? null, view, admit);
    this.lastRun = run;
    if (!run.complete) {
      // A pass is still compiling: last frame's arguments would be stale.
      this.detachAll();
      this.reason = 'compute passes not ready';
      return;
    }
    if (this.debugStage === 'compute') {
      this.detachAll();
      this.reason = 'profiling: depth + compute, nothing attached';
      return;
    }
    this.reason = run.admitAll ? `admit-all: ${run.reason}` : 'culling';
    this.attachAndCopy();
  }

  private attachAndCopy(): void {
    const engine = this.engine as any;
    const hiz = this.hiz!;
    const args = hiz.drawArgs.getBuffer().underlyingResource as GPUBuffer;
    const passId = this.camera.renderPassId;
    let encoderReady = false;
    for (const binding of this.bindings) {
      const mesh = binding.mesh;
      // A mesh Babylon will not draw this frame needs no arguments. One that
      // it will draw always gets THIS frame's block below, never a stale one.
      if (mesh.isDisposed() || !mesh.isEnabled()) continue;
      const material = binding.subMesh.getMaterial();
      const perSubMesh = !!material && (material as any)._storeEffectOnSubMeshes;
      if (binding.kind === 'mesh') {
        // A shared draw wrapper (material swapped), or a mesh that started
        // driving its own instance count on the GPU: its bound and count are
        // not ours to vouch for, so it goes back to the ordinary draw.
        if (!perSubMesh || mesh.forcedInstanceCount > 0) {
          this.release(binding);
          continue;
        }
        this.syncMesh(binding);
        if ((binding.subMesh.indexCount / 3) * binding.babylonInstances < this.minTargetTriangles) {
          this.release(binding);
          continue;
        }
      }
      const wrapper = perSubMesh
        ? (binding.subMesh as any)._getDrawWrapper(passId)
        : (material as any)?._getDrawWrapper?.();
      const context = wrapper?.drawContext as WebGPUDrawContextLike | undefined;
      if (!context || !('enableIndirectDraw' in context)) {
        // Not drawn yet in this pass: Babylon will draw it normally.
        binding.context = undefined;
        continue;
      }
      if (binding.context !== context) {
        context.enableIndirectDraw = true;
        binding.context = context;
      }
      // Babylon rewrites the block only when the instance count it wants
      // changes; pin it so the compute-authored count is never overwritten.
      context._currentInstanceCount = binding.babylonInstances;
      if (!context.indirectDrawBuffer) continue;
      // Profiling only: indirect draws with whatever arguments they hold.
      if (this.debugStage === 'no-copy') continue;
      if (!encoderReady) {
        engine._endCurrentRenderPass();
        encoderReady = true;
      }
      engine._renderEncoder.copyBufferToBuffer(
        args,
        binding.batch * SHADO_HIZ_DRAW_ARGS_WORDS * 4,
        context.indirectDrawBuffer,
        0,
        SHADO_HIZ_DRAW_ARGS_WORDS * 4
      );
    }
    this.bindInstancedMaterials(true);
  }

  /**
   * Re-reads a mesh target's instance count and world bound. Both reach the
   * GPU as queue writes, which land before this frame's command buffer, so
   * the cull that already ran in this frame's encoder sees them.
   */
  private syncMesh(binding: BatchBinding): void {
    const hiz = this.hiz!;
    const instances = thinInstanceCount(binding.mesh);
    if (instances !== binding.babylonInstances) {
      binding.babylonInstances = instances;
      hiz.setWholeInstances(binding.batch, instances);
    }
    const box = binding.mesh.getBoundingInfo().boundingBox;
    const b = binding.bound;
    const lo = box.minimumWorld;
    const hi = box.maximumWorld;
    if (b[0] !== lo.x || b[1] !== lo.y || b[2] !== lo.z || b[3] !== hi.x || b[4] !== hi.y || b[5] !== hi.z) {
      b[0] = lo.x; b[1] = lo.y; b[2] = lo.z; b[3] = hi.x; b[4] = hi.y; b[5] = hi.z;
      hiz.updateBounds(binding.candidate, [lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]);
    }
  }

  private release(binding: BatchBinding): void {
    const context = binding.context;
    if (!context) return;
    context.enableIndirectDraw = false;
    // Force Babylon to rewrite its own arguments on the next draw.
    context._currentInstanceCount = -1;
    binding.context = undefined;
  }

  private detachAll(): void {
    for (const binding of this.bindings) this.release(binding);
    this.bindInstancedMaterials(false);
  }

  private bindInstancedMaterials(compacted: boolean): void {
    if (!this.hiz) return;
    const bind = (entry: { target: BabylonHiZInstancedTarget; batch: number; matrices: StorageBuffer }, culled: boolean) => {
      const material = entry.target.material;
      material.setStorageBuffer('hizMatrices', entry.matrices);
      // Always valid bindings: the tables are recreated by setCandidates.
      material.setStorageBuffer('hizVisible', this.hiz!.visibleMembers);
      material.setStorageBuffer('hizOverflow', this.hiz!.overflow);
      material.setFloat('hizSegment', culled ? this.hiz!.segmentOffset(entry.batch) : 0);
      material.setFloat('hizBatch', culled ? entry.batch : 0);
      material.setFloat('hizCompacted', culled ? 1 : 0);
    };
    for (const entry of this.instanced) bind(entry, compacted);
    for (const entry of this.retired) bind(entry, false);
  }
}

/** Instances Babylon draws for a mesh target: its thin instances, or 1. */
function thinInstanceCount(mesh: Mesh): number {
  const count = (mesh as any).thinInstanceCount as number | undefined;
  return count && count > 0 ? count : 1;
}

function ema(previous: number, sample: number): number {
  return previous === 0 ? sample : previous * 0.9 + sample * 0.1;
}

function sameMatrix(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  for (let i = 0; i < 16; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * A minimal lit WGSL material for an instanced Hi-Z batch. `instanceIndex`
 * walks the compacted segment when `hizCompacted` is 1, or every member in
 * order when the batch is drawn normally or overflowed.
 */
export function createBabylonHiZInstancedMaterial(scene: Scene, name: string, color: readonly [number, number, number]): ShaderMaterial {
  const shaderName = 'shadoHiZInstanced';
  const store = BABYLON.ShaderStore.ShadersStoreWGSL;
  store[`${shaderName}VertexShader`] ??= /* wgsl */ `
#include<sceneUboDeclaration>
attribute position: vec3f;
attribute normal: vec3f;
var<storage, read> hizMatrices: array<mat4x4f>;
var<storage, read> hizVisible: array<u32>;
var<storage, read> hizOverflow: array<u32>;
uniform hizSegment: f32;
uniform hizBatch: f32;
uniform hizCompacted: f32;
varying vNormal: vec3f;
@vertex
fn main(input: VertexInputs) -> FragmentInputs {
  var member = vertexInputs.instanceIndex;
  let batch = u32(uniforms.hizBatch + 0.5);
  if (uniforms.hizCompacted > 0.5 && hizOverflow[batch] == 0u) {
    member = hizVisible[u32(uniforms.hizSegment + 0.5) + vertexInputs.instanceIndex];
  }
  let world = hizMatrices[member];
  vertexOutputs.position = scene.viewProjection * world * vec4f(vertexInputs.position, 1.0);
  vertexOutputs.vNormal = normalize((world * vec4f(vertexInputs.normal, 0.0)).xyz);
}`;
  store[`${shaderName}FragmentShader`] ??= /* wgsl */ `
uniform hizColor: vec3f;
varying vNormal: vec3f;
@fragment
fn main(input: FragmentInputs) -> FragmentOutputs {
  let light = 0.35 + 0.65 * max(dot(normalize(fragmentInputs.vNormal), normalize(vec3f(0.4, 0.9, 0.3))), 0.0);
  fragmentOutputs.color = vec4f(uniforms.hizColor * light, 1.0);
}`;
  const material = new BABYLON.ShaderMaterial(name, scene, shaderName, {
    attributes: ['position', 'normal'],
    uniforms: ['hizSegment', 'hizBatch', 'hizCompacted', 'hizColor'],
    uniformBuffers: ['Scene'],
    storageBuffers: ['hizMatrices', 'hizVisible', 'hizOverflow'],
    shaderLanguage: BABYLON.ShaderLanguage.WGSL,
  });
  material.setColor3('hizColor', new BABYLON.Color3(color[0], color[1], color[2]));
  material.setFloat('hizCompacted', 0);
  return material;
}
