import {
  BABYLON,
  type AbstractEngine,
  type Effect,
  type Mesh,
  type Ray,
  type Scene,
  type ShaderMaterial,
  type Texture,
} from '../babylon';
import type { ShadoConcreteCtor } from '../types';
import {
  resolveShadoDynamicEntityRenderMode,
  ShadoDynamicEntityContainer,
  type ShadoDynamicEntityGeometryMode,
  type ShadoSpriteAlphaMode,
  type ShadoSpritePresentation,
  type ShadoSpriteSortMode,
} from './ShadoDynamicEntityContainer';
import type { ShadoTextureAtlas } from './ShadoTextureAtlas';
import {
  installShadoDynamicEntityClickPicking,
  normalizePickingOptions,
  pickShadoDynamicEntityAtPointer,
  pickShadoDynamicEntityWithRay,
  type ShadoDynamicEntityAsyncPickingOptions,
  type ShadoDynamicEntityPickResult,
  type ShadoPickingHandle,
} from './ShadoAsyncPicking';

export interface ShadoDynamicEntityRendererOptions {
  mesh?: Mesh;
  geometry?: ShadoDynamicEntityGeometryMode;
  billboard?: boolean;
  presentation?: ShadoSpritePresentation;
  alphaMode?: ShadoSpriteAlphaMode;
  alphaCutoff?: number;
  pivot?: readonly [number, number];
  sortMode?: ShadoSpriteSortMode;
  log?: boolean;
  sortDrawList?: boolean;
  picking?: boolean | ShadoDynamicEntityAsyncPickingOptions;
  meshIndex?: number;
  meshTypeId?: number;
  meshTexture?: Texture | null;
}

export interface ShadoDynamicEntityMeshVariant {
  meshIndex: number;
  mesh: Mesh;
  meshTexture?: Texture | null;
  picking?: boolean | ShadoDynamicEntityAsyncPickingOptions;
}

export interface ShadoDynamicEntityMeshVariantRendererOptions extends Omit<
  ShadoDynamicEntityRendererOptions,
  'geometry' | 'mesh' | 'meshIndex' | 'meshTypeId' | 'meshTexture' | 'picking'
> {
  variants: readonly ShadoDynamicEntityMeshVariant[];
}

export class ShadoDynamicEntityRenderer {
  public readonly mesh: Mesh;
  public readonly material: ShaderMaterial;
  private readonly scene: Scene;
  private readonly engine: AbstractEngine;
  private readonly beforeRenderObserver: any;
  private readonly fallbackMeshTexture?: Texture;
  private readonly pickingDefaults: ShadoDynamicEntityAsyncPickingOptions;
  private pickingHandle?: ShadoPickingHandle;
  private loggedFirstDraw = false;
  /** Instances submitted by this variant's last draw, for instrumentation. */
  public lastSubmittedInstances = 0;

  public constructor(
    scene: Scene,
    public readonly container: ShadoDynamicEntityContainer,
    public readonly atlas: ShadoTextureAtlas,
    options: ShadoDynamicEntityRendererOptions = {}
  ) {
    this.scene = scene;
    this.engine = scene.getEngine();
    this.container.setAtlas(atlas);
    const mode = resolveShadoDynamicEntityRenderMode(options);
    const { geometry, billboard, presentation, alphaMode } = mode;
    const alphaCutoff = Math.max(0, Math.min(1, options.alphaCutoff ?? 0.35));
    const pivot =
      options.pivot ??
      (presentation === 'billboard-y' || presentation === 'billboard-screen'
        ? ([0.5, 0] as const)
        : ([0.5, 0.5] as const));
    const sortMode =
      options.sortMode ??
      (options.sortDrawList || alphaMode === 'premultiplied' ? 'camera-back-to-front' : 'none');
    this.pickingDefaults = { geometry, presentation, pivot };
    this.container.configureRenderMode(mode);
    const meshIndexInput = options.meshIndex ?? options.meshTypeId;
    const meshIndex = Number.isFinite(meshIndexInput) ? Number(meshIndexInput) : 0;
    this.fallbackMeshTexture = options.meshTexture
      ? undefined
      : BABYLON.RawTexture.CreateRGBATexture(
          new Uint8Array([255, 255, 255, 255]),
          1,
          1,
          scene,
          false,
          false,
          BABYLON.Texture.NEAREST_SAMPLINGMODE
        );
    // WGSL statically declares both atlas and mesh texture views. Keep a real
    // 2D fallback bound even when the atlas path is selected; a 2D-array view
    // cannot satisfy a texture_2d binding on WebGPU.
    const meshTexture = (options.meshTexture ?? this.fallbackMeshTexture)!;
    this.mesh =
      options.mesh ??
      (geometry === 'plane'
        ? BABYLON.MeshBuilder.CreatePlane('shado-dynamic-entity-planes', { size: 1 }, scene)
        : BABYLON.MeshBuilder.CreateBox('shado-dynamic-entities', { size: 1 }, scene));
    this.mesh.alwaysSelectAsActiveMesh = true;

    const useStorageWGSL =
      this.engine.isWebGPU && (container.constructor as any).backingPreference === 'storage';
    const shaderIo = (container.constructor as ShadoConcreteCtor).shaderIO(this.engine);
    const shaderNames = container.getShaderNamesForRenderMode(mode);
    const uniforms = [
      'worldViewProjection',
      'uShadoEntityMeshIndex',
      'uShadoDrawOffset',
      'uUseShadoEntityMeshTexture',
      'uShadoAlphaCutoff',
      ...(presentation ? ['uShadoSpritePivot'] : []),
      ...(useStorageWGSL ? [] : shaderIo.uniforms),
    ];
    if (billboard) uniforms.push('view');

    this.material = new BABYLON.ShaderMaterial('shadoDynamicEntityMaterial', scene, shaderNames, {
      attributes: ['position', 'uv'],
      uniforms,
      samplers: ['uShadoEntityAtlas', 'uShadoEntityMeshTexture', ...shaderIo.samplers],
      uniformBuffers: ['Scene'],
      needAlphaBlending: alphaMode === 'premultiplied',
      shaderLanguage: useStorageWGSL ? BABYLON.ShaderLanguage.WGSL : BABYLON.ShaderLanguage.GLSL,
    });
    this.material.backFaceCulling = geometry === 'box' || geometry === 'spriteSlab';
    this.material.forceDepthWrite = alphaMode === 'cutout';
    this.material.alphaMode =
      alphaMode === 'premultiplied'
        ? BABYLON.Engine.ALPHA_PREMULTIPLIED_PORTERDUFF
        : BABYLON.Engine.ALPHA_DISABLE;
    this.material.setTexture('uShadoEntityAtlas', atlas.texture);
    this.material.setTexture('uShadoEntityMeshTexture', meshTexture);
    this.material.setFloat('uShadoEntityMeshIndex', meshIndex);
    this.material.setFloat('uUseShadoEntityMeshTexture', options.meshTexture ? 1 : 0);
    this.material.setFloat('uShadoAlphaCutoff', alphaCutoff);
    if (presentation) this.material.setVector2('uShadoSpritePivot', new BABYLON.Vector2(...pivot));
    this.mesh.material = this.material;

    this.container.syncDrawList({ sort: sortMode === 'camera-back-to-front' });

    if (options.log) {
      this.material.onCompiled = (effect: Effect) => {
        console.debug('[shado/render] material compiled', {
          mesh: this.mesh.name,
          uniforms: effect.getUniformNames?.(),
          samplers: effect.getSamplers?.(),
        });
      };
      this.material.onError = (_effect: Effect, errors: string) => {
        console.error('[shado/render] material error', errors);
      };
    }

    const syncMaterial = () => {
      // The first variant in a frame performs the dirty-guarded upload; peers
      // become no-ops. Babylon owns the actual draw through its public material
      // and forced-instance-count path.
      if (sortMode === 'camera-back-to-front') {
        const camera = this.scene.activeCamera;
        if (camera) {
          const position = camera.globalPosition ?? camera.position;
          const forward = camera.getForwardRay().direction;
          this.container.sortDrawListByCamera(
            [position.x, position.y, position.z],
            [forward.x, forward.y, forward.z]
          );
        }
      }
      this.container.syncGpu((this.engine as any).frameId ?? 0);
      this.container.bindMaterial(this.material);
      this.material.setTexture('uShadoEntityAtlas', this.atlas.texture);
      this.material.setTexture('uShadoEntityMeshTexture', meshTexture);
      this.material.setFloat('uShadoEntityMeshIndex', meshIndex);
      this.material.setFloat('uUseShadoEntityMeshTexture', options.meshTexture ? 1 : 0);
      this.material.setFloat('uShadoAlphaCutoff', alphaCutoff);
      const { offset: drawOffset, count: drawCount } = this.container.getMeshDrawRange(meshIndex);
      this.material.setFloat('uShadoDrawOffset', drawOffset);
      this.mesh.forcedInstanceCount = Math.max(0, drawCount | 0);
      this.mesh.isVisible = this.mesh.forcedInstanceCount > 0;
      this.lastSubmittedInstances = this.mesh.forcedInstanceCount;

      if (options.log && drawCount > 0 && !this.loggedFirstDraw) {
        this.loggedFirstDraw = true;
        console.debug('[shado/render] first draw scheduled', {
          mesh: this.mesh.name,
          drawCount,
          entityCount: this.container.entityCount,
          indices: this.mesh.getTotalIndices(),
          vertices: this.mesh.getTotalVertices(),
        });
      }
    };
    syncMaterial();
    this.beforeRenderObserver = scene.onBeforeRenderObservable.add(syncMaterial);

    const picking = normalizePickingOptions(options.picking);
    if (picking) {
      this.setAsyncPicking(picking);
    }
  }

  public static createMeshVariantRenderers(
    scene: Scene,
    container: ShadoDynamicEntityContainer,
    atlas: ShadoTextureAtlas,
    options: ShadoDynamicEntityMeshVariantRendererOptions
  ): ShadoDynamicEntityRenderer[] {
    return options.variants.map(
      variant =>
        new ShadoDynamicEntityRenderer(scene, container, atlas, {
          billboard: false,
          geometry: 'mesh',
          log: options.log,
          mesh: variant.mesh,
          meshIndex: variant.meshIndex,
          meshTexture: variant.meshTexture,
          picking: variant.picking,
          alphaMode: options.alphaMode,
          alphaCutoff: options.alphaCutoff,
          sortMode: options.sortMode,
          sortDrawList: options.sortDrawList,
        })
    );
  }

  public setAsyncPicking(options: boolean | ShadoDynamicEntityAsyncPickingOptions): void {
    this.pickingHandle?.dispose();
    const normalized = normalizePickingOptions(options);
    if (!normalized) {
      this.pickingHandle = undefined;
      return;
    }
    this.pickingHandle = installShadoDynamicEntityClickPicking(
      this.scene,
      this.mesh,
      this.container,
      { ...this.pickingDefaults, ...normalized }
    );
  }

  public pickAsync(
    pointerX = this.scene.pointerX,
    pointerY = this.scene.pointerY,
    options: ShadoDynamicEntityAsyncPickingOptions = {}
  ): Promise<ShadoDynamicEntityPickResult | null> {
    return pickShadoDynamicEntityAtPointer(
      this.scene,
      this.mesh,
      this.container,
      pointerX,
      pointerY,
      { ...this.pickingDefaults, ...options }
    );
  }

  public pickWithRay(
    ray: Ray,
    options: ShadoDynamicEntityAsyncPickingOptions = {}
  ): ShadoDynamicEntityPickResult | null {
    return pickShadoDynamicEntityWithRay(this.mesh, this.container, ray, {
      ...this.pickingDefaults,
      ...options,
    });
  }

  public dispose(): void {
    this.scene.onBeforeRenderObservable.remove(this.beforeRenderObserver);
    this.mesh.forcedInstanceCount = 0;
    this.pickingHandle?.dispose();
    this.material.dispose();
    this.fallbackMeshTexture?.dispose();
  }
}
