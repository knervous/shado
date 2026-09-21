import { BABYLON, type AbstractMesh, type Camera, type Mesh, type Observer, type Scene, type StandardMaterial } from '../babylon';

/**
 * Bounded hardware occlusion queries (docs/pvs-hiz-prototype.md G1) -- the
 * WebGL2 fallback's optional runtime layer, labelled "queries", not Hi-Z.
 *
 * Each group (a spatial set of meshes) has a proxy: its world AABB, expanded
 * a little, drawn with colour and depth writes off in a late rendering group,
 * after every opaque surface this frame. The query brackets exactly that
 * draw. A group is hidden from THIS camera's main pass (the active-mesh
 * candidates; shadows and other passes keep it) only while:
 *   - its last query returned no samples (engine result > 0 is "visible"),
 *   - camera, projection, viewport, the group's bounds and the opaque epoch
 *     are all exactly what they were when that query ran,
 *   - and the result is at most `maxAgeFrames` rendered frames old.
 * Anything else -- unknown, pending, failed, stale, over budget, context
 * lost -- admits. At most `maxNewPerFrame` queries start per frame and
 * `maxPending` are outstanding; query objects are pooled.
 *
 * The deliberate cost of this policy: while the camera moves nothing stays
 * hidden, because every movement invalidates every result.
 */
export interface OcclusionQueryGroup {
  readonly id: string;
  readonly meshes: readonly AbstractMesh[];
}

export interface OcclusionQueryOptions {
  maxNewPerFrame?: number;
  maxPending?: number;
  maxAgeFrames?: number;
  /** Proxy expansion: relative, plus an absolute margin in world units. */
  marginScale?: number;
  marginAbsolute?: number;
  /** Rendering group the proxies draw in (depth kept from earlier groups). */
  renderingGroupId?: number;
}

export interface OcclusionQueryStatus {
  readonly enabled: boolean;
  readonly backend: string;
  readonly reason: string;
  readonly groups: number;
  readonly hiddenGroups: number;
  readonly hiddenMeshes: number;
  readonly hiddenSample: readonly string[];
  readonly pending: number;
  readonly issuedLastFrame: number;
  readonly pooled: number;
  /** Mean age in frames of the results that currently hide something. */
  readonly meanHiddenResultAge: number;
  /** Main-thread time spent scheduling, issuing and polling, per frame (EMA). */
  readonly cpuMs: number;
  readonly queriesIssued: number;
  readonly contextLosses: number;
}

interface GroupState {
  group: OcclusionQueryGroup;
  proxy: Mesh;
  triangles: number;
  bounds: Float64Array;
  /** Last completed result. */
  result: { visible: boolean; frame: number; epoch: string; bounds: Float64Array } | null;
  pending: { query: WebGLQuery | unknown; frame: number; epoch: string; bounds: Float64Array } | null;
  /** Scheduled to issue a query this frame. */
  scheduled: boolean;
  begun: boolean;
}

const ALGORITHM = 1; // AbstractMesh.OCCLUSION_ALGORITHM_TYPE_CONSERVATIVE

export class BabylonOcclusionQueries {
  private states: GroupState[] = [];
  private pool: unknown[] = [];
  private enabled = false;
  private frame = 0;
  private opaqueEpoch = 0;
  private issuedLastFrame = 0;
  private queriesIssued = 0;
  private contextLosses = 0;
  private cpuMs = 0;
  private reason = 'idle';
  /** Set when this backend's query results cannot be trusted: never enables. */
  public readonly unsupported: string = '';
  private hidden = new Set<AbstractMesh>();
  private readonly material: StandardMaterial;
  private readonly observers: Array<() => void> = [];
  private readonly originalCandidates: Scene['getActiveMeshCandidates'];
  private readonly options: Required<OcclusionQueryOptions>;

  public constructor(
    private readonly scene: Scene,
    private readonly camera: Camera,
    options: OcclusionQueryOptions = {}
  ) {
    this.options = {
      maxNewPerFrame: options.maxNewPerFrame ?? 32,
      maxPending: options.maxPending ?? 64,
      maxAgeFrames: options.maxAgeFrames ?? 2,
      marginScale: options.marginScale ?? 0.02,
      marginAbsolute: options.marginAbsolute ?? 0.05,
      renderingGroupId: options.renderingGroupId ?? 3,
    };
    const engine = scene.getEngine() as any;
    if (typeof engine.createQuery !== 'function' || typeof engine.beginOcclusionQuery !== 'function') {
      throw new Error('this engine has no occlusion queries (import the engine.query extension)');
    }
    if (engine.isWebGPU) {
      // Babylon 9.27.1's WebGPU isQueryResultAvailable(i) is true whenever ANY
      // read-back of the whole query set covers slot i, so a stale value (or
      // an unresolved 0) from an earlier use of the slot reads as this
      // query's answer -- measured hiding visible walls. There is no per-query
      // completion to wait on; refuse rather than guess. WebGPU has Hi-Z.
      this.unsupported =
        'WebGPU occlusion-query availability is per query set, not per query, in Babylon 9.27.1 (stale slots read as occluded); use Hi-Z on WebGPU';
    }
    this.material = new BABYLON.StandardMaterial('shado-occlusion-proxy', scene);
    this.material.disableColorWrite = true;
    this.material.disableDepthWrite = true;
    this.material.backFaceCulling = false;
    this.material.disableLighting = true;
    this.material.freeze();
    // Keep the depth of earlier groups: the proxies test against it.
    scene.setRenderingAutoClearDepthStencil(this.options.renderingGroupId, false, false, false);

    // Per-pass decision: the main camera's active-mesh candidates only.
    this.originalCandidates = scene.getActiveMeshCandidates.bind(scene);
    scene.getActiveMeshCandidates = () => {
      const all = this.originalCandidates();
      if (!this.enabled || !this.hidden.size || scene.activeCamera !== this.camera) return all;
      const kept: AbstractMesh[] = [];
      for (let i = 0; i < all.length; i++) {
        const mesh = all.data[i]!;
        if (!this.hidden.has(mesh)) kept.push(mesh);
      }
      return { data: kept, length: kept.length } as any;
    };

    const before = scene.onBeforeRenderObservable.add(() => this.beginFrame());
    const lost = engine.onContextLostObservable.add(() => this.onContextLost());
    this.observers.push(
      () => scene.onBeforeRenderObservable.remove(before),
      () => engine.onContextLostObservable.remove(lost)
    );
  }

  public setEnabled(on: boolean): void {
    if (this.unsupported) {
      this.reason = `unsupported: ${this.unsupported}`;
      return;
    }
    if (on === this.enabled) return;
    this.enabled = on;
    this.hidden.clear();
    for (const state of this.states) {
      state.proxy.setEnabled(on);
      state.result = null;
    }
    this.reason = on ? 'waiting for first results' : 'off';
  }

  /** Occluders moved, streamed or were removed: every hidden result is void. */
  public bumpOpaqueEpoch(): void {
    this.opaqueEpoch++;
  }

  public setGroups(groups: readonly OcclusionQueryGroup[]): void {
    for (const state of this.states) {
      this.releasePending(state);
      state.proxy.dispose();
    }
    this.hidden.clear();
    this.states = groups
      .filter((group) => group.meshes.length)
      .map((group) => {
        const proxy = BABYLON.MeshBuilder.CreateBox(`shado-occlusion-proxy:${group.id}`, { size: 1 }, this.scene);
        proxy.material = this.material;
        proxy.isPickable = false;
        proxy.doNotSyncBoundingInfo = true;
        proxy.alwaysSelectAsActiveMesh = true;
        proxy.renderingGroupId = this.options.renderingGroupId;
        proxy.setEnabled(this.enabled);
        const state: GroupState = {
          group,
          proxy,
          triangles: group.meshes.reduce((sum, mesh) => sum + (mesh.getTotalIndices?.() ?? 0) / 3, 0),
          bounds: new Float64Array(6),
          result: null,
          pending: null,
          scheduled: false,
          begun: false,
        };
        // The query brackets exactly the proxy's own draw, in the main pass.
        // onBeforeDraw, not onBeforeRender: a mesh whose effect is not ready
        // returns before drawing, and a begun query with no end leaves the
        // target "already active" for every later begin.
        proxy.onBeforeDrawObservable.add(() => this.beginQuery(state));
        proxy.onAfterRenderObservable.add(() => this.endQuery(state));
        return state;
      });
  }

  public status(): OcclusionQueryStatus {
    let pending = 0;
    let hiddenGroups = 0;
    let ageSum = 0;
    for (const state of this.states) {
      if (state.pending) pending++;
      if (this.isHidden(state)) {
        hiddenGroups++;
        ageSum += this.frame - state.result!.frame;
      }
    }
    const engine = this.scene.getEngine() as any;
    return {
      enabled: this.enabled,
      backend: engine.isWebGPU ? 'webgpu' : `webgl${engine.webGLVersion ?? ''}`,
      reason: this.reason,
      groups: this.states.length,
      hiddenGroups,
      hiddenMeshes: this.hidden.size,
      hiddenSample: [...this.hidden].slice(0, 12).map((mesh) => mesh.name),
      pending,
      issuedLastFrame: this.issuedLastFrame,
      pooled: this.pool.length,
      meanHiddenResultAge: hiddenGroups ? ageSum / hiddenGroups : 0,
      cpuMs: this.cpuMs,
      queriesIssued: this.queriesIssued,
      contextLosses: this.contextLosses,
    };
  }

  public dispose(): void {
    for (const remove of this.observers.splice(0)) remove();
    this.scene.getActiveMeshCandidates = this.originalCandidates;
    const engine = this.scene.getEngine() as any;
    for (const state of this.states) {
      if (state.pending) engine.deleteQuery?.(state.pending.query);
      state.proxy.dispose();
    }
    for (const query of this.pool) engine.deleteQuery?.(query);
    this.pool = [];
    this.states = [];
    this.material.dispose();
  }

  // --------------------------------------------------------------- frame ---

  private viewEpoch(): string {
    const m = this.scene.getTransformMatrix().m;
    const engine = this.scene.getEngine();
    let key = `${engine.getRenderWidth()}x${engine.getRenderHeight()}|${this.opaqueEpoch}|`;
    for (let i = 0; i < 16; i++) key += `${m[i]},`;
    return key;
  }

  private isHidden(state: GroupState): boolean {
    const result = state.result;
    return (
      !!result &&
      !result.visible &&
      result.epoch === this.currentEpoch &&
      this.frame - result.frame <= this.options.maxAgeFrames &&
      sameBounds(result.bounds, state.bounds)
    );
  }

  private currentEpoch = '';

  private beginFrame(): void {
    if (!this.enabled || this.scene.activeCamera !== this.camera) return;
    const started = performance.now();
    this.frame++;
    // The transform the proxies will be drawn with (the camera's, this frame).
    this.camera.getViewMatrix();
    this.scene.updateTransformMatrix();
    this.currentEpoch = this.viewEpoch();
    const engine = this.scene.getEngine() as any;

    // 1. Collect finished queries; never block on one that is not.
    for (const state of this.states) {
      const pending = state.pending;
      if (!pending) continue;
      let available = false;
      try {
        available = !!engine.isQueryResultAvailable(pending.query);
      } catch {
        available = false;
      }
      if (!available) continue;
      let visible = true;
      try {
        visible =
          typeof engine.isOcclusionQueryVisible === 'function'
            ? engine.isOcclusionQueryVisible(pending.query)
            : Number(engine.getQueryResult(pending.query)) > 0;
      } catch {
        visible = true; // a failed read admits
      }
      state.result = { visible, frame: pending.frame, epoch: pending.epoch, bounds: pending.bounds };
      this.pool.push(pending.query);
      state.pending = null;
    }

    // 2. Proxies follow their groups' bounds; decide what is hidden now.
    this.hidden.clear();
    for (const state of this.states) {
      updateProxy(state, this.options);
      if (this.isHidden(state)) for (const mesh of state.group.meshes) this.hidden.add(mesh);
    }

    // 3. Schedule new queries within budget: groups with no result for this
    //    view (or one about to expire), most expensive first.
    let pendingCount = this.states.reduce((n, s) => n + (s.pending ? 1 : 0), 0);
    const due = this.states
      .filter((state) => {
        state.scheduled = false;
        if (state.pending) return false;
        const r = state.result;
        return !r || r.epoch !== this.currentEpoch || !sameBounds(r.bounds, state.bounds) || this.frame - r.frame >= this.options.maxAgeFrames - 1;
      })
      .sort((a, b) => b.triangles - a.triangles);
    let issued = 0;
    for (const state of due) {
      if (issued >= this.options.maxNewPerFrame || pendingCount >= this.options.maxPending) break;
      state.scheduled = true;
      issued++;
      pendingCount++;
    }
    for (const state of this.states) state.proxy.isVisible = state.scheduled;
    this.issuedLastFrame = issued;
    this.reason = this.states.length ? `querying (${issued} new, ${pendingCount} pending)` : 'no groups';
    this.cpuMs = ema(this.cpuMs, performance.now() - started);
  }

  private beginQuery(state: GroupState): void {
    state.begun = false;
    if (!state.scheduled || state.pending || this.scene.activeCamera !== this.camera) return;
    const engine = this.scene.getEngine() as any;
    const query = this.pool.pop() ?? engine.createQuery();
    let begun = false;
    try {
      begun = engine.beginOcclusionQuery(ALGORITHM, query) !== false;
    } catch {
      begun = false;
    }
    if (!begun) {
      this.pool.push(query);
      return;
    }
    state.begun = true;
    state.pending = { query, frame: this.frame, epoch: this.currentEpoch, bounds: Float64Array.from(state.bounds) };
    this.queriesIssued++;
  }

  private endQuery(state: GroupState): void {
    // End only what began.
    if (!state.begun) return;
    state.begun = false;
    state.scheduled = false;
    (this.scene.getEngine() as any).endOcclusionQuery(ALGORITHM);
  }

  private releasePending(state: GroupState): void {
    if (!state.pending) return;
    this.pool.push(state.pending.query);
    state.pending = null;
  }

  private onContextLost(): void {
    // Every query object belongs to the lost context: drop them all, admit.
    this.contextLosses++;
    this.pool = [];
    for (const state of this.states) {
      state.pending = null;
      state.result = null;
      state.begun = false;
    }
    this.hidden.clear();
    this.reason = 'context lost: admitting';
  }
}

function updateProxy(state: GroupState, options: Required<OcclusionQueryOptions>): void {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (const mesh of state.group.meshes) {
    if (mesh.isDisposed()) continue;
    const box = mesh.getBoundingInfo().boundingBox;
    x0 = Math.min(x0, box.minimumWorld.x);
    y0 = Math.min(y0, box.minimumWorld.y);
    z0 = Math.min(z0, box.minimumWorld.z);
    x1 = Math.max(x1, box.maximumWorld.x);
    y1 = Math.max(y1, box.maximumWorld.y);
    z1 = Math.max(z1, box.maximumWorld.z);
  }
  const b = state.bounds;
  if (!(x1 >= x0)) return;
  b[0] = x0; b[1] = y0; b[2] = z0; b[3] = x1; b[4] = y1; b[5] = z1;
  // Expanded so a group that IS drawn cannot hide its own proxy (the query
  // runs after it has written depth).
  const pad = (a: number, c: number) => (c - a) * options.marginScale + options.marginAbsolute;
  const px = pad(x0, x1), py = pad(y0, y1), pz = pad(z0, z1);
  const proxy = state.proxy;
  proxy.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  proxy.scaling.set(x1 - x0 + 2 * px, y1 - y0 + 2 * py, z1 - z0 + 2 * pz);
}

function sameBounds(a: Float64Array, b: Float64Array): boolean {
  for (let i = 0; i < 6; i++) if (a[i] !== b[i]) return false;
  return true;
}

function ema(previous: number, sample: number): number {
  return previous === 0 ? sample : previous * 0.9 + sample * 0.1;
}
