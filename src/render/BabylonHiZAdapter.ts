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
  public readonly hiz?: ShadoWorldHiZ;
  private depth?: DepthRenderer;
  private readonly engine: WebGPUEngine;
  private bindings: BatchBinding[] = [];
  private instanced: Array<{ target: BabylonHiZInstancedTarget; batch: number; matrices: StorageBuffer }> = [];
  private candidateIds: number[] = [];
  private occluders: AbstractMesh[] = [];
  private mode: BabylonHiZMode = 'off';
  private reason = 'not started';
  private lastRun?: ShadoWorldHiZRunResult;
  private beforeDraw?: Observer<Scene>;
  private resizeObserver?: Observer<unknown>;
  private readonly depthMatrix = new Float32Array(16);
  private depthFrame = -1;
  private readonly errors: string[] = [];
  /**
   * Pixel row 0 of the depth texture is the top of the view. Read from the
   * depth target every frame: Babylon's WebGPU engine renders into render
   * targets y-flipped unless the wrapper sets `_disableEngineYFlip`.
   */
  public topLeftOrigin = false;
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
    this.hiz = new ShadoWorldHiZ(this.engine);
    this.createDepth();
    this.resizeObserver = this.engine.onResizeObservable.add(() => this.createDepth()) as Observer<unknown>;
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
    this.detachAll();
    for (const entry of this.instanced) entry.matrices.dispose();
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

  public setMode(mode: BabylonHiZMode): void {
    this.mode = mode;
    if (mode === 'off') this.detachAll();
  }

  public status(): BabylonHiZStatus {
    const size = this.depth?.getDepthMap().getSize();
    return {
      backend: (this.engine as any).isWebGPU ? 'webgpu' : (this.engine as any).webGLVersion === 2 ? 'webgl2' : 'other',
      requested: this.mode,
      actual: this.hiz && this.mode === 'hiz' && this.lastRun?.complete && !this.lastRun.admitAll ? 'hiz' : 'off',
      reason: this.reason,
      babylonVersion: BABYLON.Engine.Version,
      frameId: this.engine.frameId,
      candidates: this.hiz?.candidates ?? 0,
      batches: this.hiz?.batches ?? 0,
      attachedDrawContexts: this.bindings.filter(b => b.context).length,
      lastRun: this.lastRun,
      depthSize: [size?.width ?? 0, size?.height ?? 0],
      convention: this.convention,
      topLeftOrigin: this.topLeftOrigin,
      errors: [...this.errors, ...(this.hiz?.lastErrors ?? [])],
    };
  }

  /** The occluder depth render target (debug view). */
  public get depthTexture() {
    return this.depth?.getDepthMap();
  }

  public dispose(): void {
    this.detachAll();
    if (this.beforeDraw) this.scene.onBeforeDrawPhaseObservable.remove(this.beforeDraw);
    if (this.resizeObserver) this.engine.onResizeObservable.remove(this.resizeObserver as any);
    this.disposeDepth();
    for (const entry of this.instanced) {
      entry.target.mesh.forcedInstanceCount = 0;
      entry.matrices.dispose();
    }
    this.instanced = [];
    this.hiz?.dispose();
  }

  private createDepth(): void {
    this.disposeDepth();
    // Full render resolution, single sample, float, gl_FragCoord.z; clear = far.
    const depth = new DepthRenderer(this.scene, BABYLON.Constants.TEXTURETYPE_FLOAT, this.camera, true, BABYLON.Constants.TEXTURE_NEAREST_SAMPLINGMODE, false, 'Shado Hi-Z occluder depth');
    depth.clearColor = new BABYLON.Color4(this.convention === 'normal' ? 1 : 0, 0, 0, 1);
    const map = depth.getDepthMap();
    map.renderList = this.occluders;
    map.onBeforeRenderObservable.add(() => {
      this.depthMatrix.set(this.scene.getTransformMatrix().m);
      this.depthFrame = this.engine.frameId;
    });
    // Not registered in scene._depthRenderer (that slot belongs to whatever
    // enableDepthRenderer() the game uses): a custom target of THIS camera
    // renders once per frame for it, before its main pass.
    this.camera.customRenderTargets.push(map);
    this.depth = depth;
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
    if (!this.bindings.length) {
      this.reason = 'no targets';
      return;
    }
    const map = this.depth?.getDepthMap();
    const matrix = this.scene.getTransformMatrix();
    let admit = '';
    if (!map || this.depthFrame !== this.engine.frameId) admit = 'depth not rendered this frame';
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
    const run = this.hiz.run(map ?? null, view, admit);
    this.lastRun = run;
    if (!run.complete) {
      // A pass is still compiling: last frame's arguments would be stale.
      this.detachAll();
      this.reason = 'compute passes not ready';
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
        if (!perSubMesh) {
          // The material changed to one with a shared draw wrapper.
          this.release(binding);
          continue;
        }
        this.syncMesh(binding);
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
    for (const { target, batch, matrices } of this.instanced) {
      const material = target.material;
      material.setStorageBuffer('hizMatrices', matrices);
      material.setStorageBuffer('hizVisible', this.hiz.visibleMembers);
      material.setStorageBuffer('hizOverflow', this.hiz.overflow);
      material.setFloat('hizSegment', this.hiz.segmentOffset(batch));
      material.setFloat('hizBatch', batch);
      material.setFloat('hizCompacted', compacted ? 1 : 0);
    }
  }
}

/** Instances Babylon draws for a mesh target: its thin instances, or 1. */
function thinInstanceCount(mesh: Mesh): number {
  const count = (mesh as any).thinInstanceCount as number | undefined;
  return count && count > 0 ? count : 1;
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
