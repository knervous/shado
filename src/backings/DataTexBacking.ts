import { EMPTY_UPLOAD_STATS, type BackendKind, type GPUBacking, type GPUUploadStats, type Segment } from '../types';
import type { Shado } from '../core/Shado';
import { encodeGpuFloatUploadRange } from './encodeGpuFloatUpload';
import {
  getShadoRendererAdapter,
  type ShadoRendererAdapter,
  type ShadoRendererTexture,
} from '../renderer/ShadoRendererAdapter';

export class DataTexBacking implements GPUBacking {
  public kind: BackendKind = 'datatex';
  private renderer?: ShadoRendererAdapter;
  private bufTex?: ShadoRendererTexture;
  private texW = 2048;
  private texH = 1;
  private capTexels = this.texW * this.texH;
  private capFloats = this.capTexels * 4;

  private staging?: Float32Array;
  private lastUsedFloats = 0;
  private lastStats: GPUUploadStats = EMPTY_UPLOAD_STATS;
  /**
   * Materials seeded through bindMaterial, and the include name they were
   * seeded with.
   *
   * A material holds whatever texture it was handed until it is handed another
   * one, but growing the arena has to replace the buffer texture outright -
   * a RawTexture cannot be resized. Without this set, every material seeded
   * before the first growth keeps a disposed texture, and Babylon's
   * Material.isReady() rejects a material whose textures are not ready. The
   * result is a material that never becomes ready and therefore never draws,
   * with no shader error to point at it. The effect path re-binds per draw and
   * so repairs itself; the material path only gets seeded once, which is
   * exactly why it needs to be replayed here.
   */
  private readonly boundMaterials = new Set<any>();
  private boundIncludeName?: string;

  constructor(
    private engine: any,
    private schema: any,
    private owner: Shado
  ) {
    // Adapter lookup remains lazy for renderer-neutral schema/memory use.
    // Production initialization installs an adapter before the first commit.
    try {
      this.renderer = getShadoRendererAdapter(engine);
    } catch {
      this.renderer = undefined;
    }
  }

  private getRenderer(): ShadoRendererAdapter {
    const renderer = (this.renderer ??= getShadoRendererAdapter(this.engine));
    if (!renderer.createDataTexture) {
      throw new Error(
        `Renderer ${renderer.id} does not support Shado's data-texture backend. Use storage instead.`
      );
    }
    return renderer;
  }

  reserveFloats(minFloats: number) {
    if (minFloats <= this.capFloats && this.bufTex) return;
    const needTexels = Math.max(1, Math.ceil(minFloats / 4));
    const needH = Math.max(1, Math.ceil(needTexels / this.texW));
    this.texH = Math.max(this.texH, needH);
    this.capTexels = this.texW * this.texH;
    this.capFloats = this.capTexels * 4;

    const nextStaging = new Float32Array(this.capFloats);
    if (this.staging) {
      nextStaging.set(this.staging.subarray(0, Math.min(this.staging.length, this.capFloats)));
    }

    this.bufTex?.dispose();
    this.bufTex = this.getRenderer().createDataTexture!(
      nextStaging,
      this.texW,
      this.texH,
      `${this.schema.name} data texture`
    );
    this.staging = nextStaging;
    this.lastUsedFloats = Math.min(this.lastUsedFloats, this.capFloats);
    (this.owner.arena as any)?.markDirty?.();
    this.rebindMaterials();
  }

  /** Hand every seeded material the replacement texture and fresh offsets. */
  private rebindMaterials() {
    if (!this.bufTex || this.boundIncludeName === undefined) return;
    for (const material of this.boundMaterials) {
      this.applyBindings(material, this.boundIncludeName);
    }
  }

  commit(): GPUUploadStats {
    const arena = this.owner.arena;
    if (!arena?.isDirty?.()) return (this.lastStats = EMPTY_UPLOAD_STATS);
    const payload: Float32Array = this.owner.prepareUnifiedForUpload();
    this.reserveFloats(payload.length);

    if (!this.staging || this.staging.length !== this.capFloats) {
      this.staging = new Float32Array(this.capFloats);
      (arena as any).markDirty?.();
    }

    // The staging buffer doubles as the persistent GPU-encoded mirror: only
    // dirty subranges are re-copied and integer-encoded, then the texture is
    // updated in one call. Partial raw-texture upload is not portable through
    // Babylon's high-level API, so the storage backing remains the preferred
    // path for sparse updates on WebGPU.
    const ranges =
      (arena as any).consumeDirtyRanges?.() ?? [{ start: 0, end: payload.byteLength }];
    let encodedBytes = 0;
    for (const range of ranges) {
      const startF = range.start >>> 2;
      const endF = Math.min(payload.length, (range.end + 3) >>> 2);
      if (endF <= startF) continue;
      encodeGpuFloatUploadRange(this.schema, this.owner, payload, this.staging, startF, endF);
      encodedBytes += (endF - startF) * 4;
    }
    if (payload.length < this.lastUsedFloats) {
      this.staging.fill(0, payload.length, this.lastUsedFloats);
    }
    this.lastUsedFloats = payload.length;
    if (!encodedBytes) return (this.lastStats = EMPTY_UPLOAD_STATS);
    this.bufTex!.update(this.staging);
    return (this.lastStats = {
      uploadCalls: 1,
      uploadedBytes: this.staging.byteLength,
      encodedBytes,
    });
  }

  public getLastUploadStats(): GPUUploadStats {
    return this.lastStats;
  }

  bind(effect: any, includeName: string) {
    if (!this.bufTex) return;
    this.applyBindings(effect, includeName);
  }

  bindMaterial(material: any, includeName: string) {
    this.boundIncludeName = includeName;
    if (!this.boundMaterials.has(material)) {
      this.boundMaterials.add(material);
      // Drop the reference with the material rather than pinning a disposed
      // one here for the lifetime of the arena.
      material?.onDisposeObservable?.addOnce?.(() => {
        this.boundMaterials.delete(material);
      });
    }
    if (!this.bufTex) return;
    this.applyBindings(material, includeName);
  }

  private applyBindings(target: any, includeName: string) {
    const texture = this.bufTex;
    if (!texture) return;
    const renderer = this.getRenderer();
    const self: any = this.owner;
    const schema = this.schema;

    // Bind buffer texture
    renderer.bindDataTexture!(target, `u${includeName}BufTex`, texture);
    renderer.setInt!(target, `u${includeName}BufTexWidth`, this.texW);
    renderer.setInt!(target, `u${includeName}HeaderBase`, self._headerSeg.offF | 0);

    // Bind var array uniforms
    for (const f of Object.keys(schema.varArrays)) {
      const seg: Segment = self._varSeg[f];
      const stride = schema.varArrays[f].floatStride;
      const count = Math.floor((seg?.lenF ?? 0) / stride);
      renderer.setInt!(target, `u${includeName}_${f}Base`, seg?.offF | 0);
      renderer.setInt!(target, `u${includeName}_${f}Stride`, stride | 0);
      renderer.setInt!(target, `u${includeName}_${f}Count`, count | 0);
    }

    // Bind struct array uniforms
    for (const f of Object.keys(schema.structArrays)) {
      const seg: Segment = self._structSeg[f];
      const stride = schema.structArrays[f].schema.headerFloatCount;
      const count = (self._structArrayCount?.[f] as number) | 0;
      renderer.setInt!(target, `u${includeName}_${f}Base`, seg?.offF | 0);
      renderer.setInt!(target, `u${includeName}_${f}Stride`, stride | 0);
      renderer.setInt!(target, `u${includeName}_${f}Count`, count | 0);
    }
  }

  dispose() {
    this.boundMaterials.clear();
    this.bufTex?.dispose();
  }
}
