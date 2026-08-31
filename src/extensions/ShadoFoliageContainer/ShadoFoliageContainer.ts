import type { Mesh, Scene } from '../../babylon';
import { BABYLON } from '../../babylon';
import { field, gpuStruct } from '../../decorators';
import type { ShadoMaterial } from '../../materials/ShadoMaterial';
import { ShadoActor } from '../ShadoActor';
import {
  ShadoInstanceContainer,
  type ShadoInstanceContainerOptions,
  type ShadoInstanceGLSLHooks,
  type ShadoInstanceWGSLHooks,
} from '../ShadoInstanceContainer/ShadoInstanceContainer';
import {
  resolveShadoFoliagePlugins,
  type ShadoFoliageFrame,
  type ShadoFoliagePlugin,
  type ShadoFoliagePluginSpec,
  type ShadoFoliageShaderFragments,
} from './plugins';

/**
 * A rigid, unanimated instance of a foliage prototype.
 *
 * `foliageParams` is the whole per-instance authoring surface, and it is the
 * reason foliage does not need the actor machinery a character needs:
 *
 * - `x` — wind phase, 0..1. Decorrelates neighbouring plants.
 * - `y` — stiffness, 0..1. How much this instance resists the wind.
 * - `z` — colour variation, 0..1. Drives the `tint` plugin.
 * - `w` — free, reserved for prototype-specific plugins.
 */
@gpuStruct({ name: 'ShadoFoliageActor' })
export class ShadoFoliageActor extends ShadoActor {
  @field('vec4') foliageParams!: Float32Array;

  public override initialize() {
    super.initialize();
    this.foliageParams = new Float32Array([0, 0, 0, 0]);
  }
}

export type ShadoFoliageContainerOptions = Omit<
  ShadoInstanceContainerOptions,
  | 'vat'
  | 'vatQuality'
  | 'vatOptions'
  | 'vatPosePalette'
  | 'vatPosePaletteCapacity'
  | 'prebakedVat'
  | 'packedVat'
  | 'animationRanges'
  | 'materialUniforms'
  | 'materialBind'
> & {
  /** Behavior applied to every instance, resolved in the order given. */
  plugins: readonly (ShadoFoliagePluginSpec | ShadoFoliagePlugin)[];
  /**
   * Height of the source mesh in its own local units, used to normalize bend
   * along the plant. Read from the mesh bounding box when omitted.
   */
  sourceHeight?: number;
  /**
   * The point foliage reacts to, normally the player. Sampled once per frame.
   * Defaults to the origin, which leaves distance-driven plugins inert.
   */
  focus?: () => readonly [number, number, number];
  /** Animation clock in seconds. Defaults to the scene's own elapsed time. */
  timeSource?: () => number;
};

/**
 * Static instanced foliage: one draw per prototype, no skeleton, no VAT, no
 * physics, no picking, and no per-instance CPU work after upload.
 *
 * This is deliberately the same arena, SoA visibility, frustum culling, and
 * upload path a character container uses. Foliage differs only in that it is
 * rigid and that its motion is entirely a function of world position and time,
 * which is what the plugins express.
 */
export class ShadoFoliageContainer extends ShadoInstanceContainer<ShadoFoliageActor> {
  private _plugins: ShadoFoliagePlugin[] = [];
  private _inverseHeight = 1;
  private _focus: () => readonly [number, number, number] = () => ORIGIN;
  private _scene?: Scene;
  private readonly _cameraVector = new BABYLON.Vector3(0, 0, 0);
  private _timeSource?: () => number;
  private _elapsedSeconds = 0;
  private _lastFrameMs = 0;
  private readonly _focusVector = new BABYLON.Vector3(0, 0, 0);
  private readonly _frame: { timeSeconds: number; focus: [number, number, number] } = {
    timeSeconds: 0,
    focus: [0, 0, 0],
  };

  /** Behavior currently compiled into this container's material. */
  public get plugins(): readonly ShadoFoliagePlugin[] {
    return this._plugins;
  }

  /**
   * Compiles a behavior set into this container without attaching meshes.
   *
   * `attachFoliage` is the normal entry point; this is separated so the
   * generated shader can be inspected, and so a caller that already drives
   * `attachMeshes` itself can still opt into foliage behavior.
   *
   * Returns the material options the plugin set requires.
   */
  public configureFoliage(
    options: ShadoFoliageContainerOptions,
    meshes: readonly Mesh[] = []
  ): { materialUniforms: string[]; materialBind: (material: ShadoMaterial<any>) => void } {
    if (this._plugins.length) {
      throw new Error('configureFoliage: this container already has compiled foliage plugins');
    }
    // Specs and already-resolved plugins may be mixed, so resolve only the
    // declarative entries and keep caller-built plugins as they are.
    this._plugins = resolveMixed(options.plugins);
    if (options.sourceHeight !== undefined || meshes.length) {
      this._inverseHeight = 1 / resolveSourceHeight(meshes, options.sourceHeight);
    }
    if (options.focus) this._focus = options.focus;
    this._timeSource = options.timeSource;

    const materialUniforms = [
      'uShadoFoliageTime',
      'uShadoFoliageFocus',
      'uShadoFoliageCamera',
      'uShadoFoliageInverseHeight',
      ...this._plugins.flatMap(plugin => plugin.uniforms),
    ];
    const duplicate = materialUniforms.find(
      (name, index) => materialUniforms.indexOf(name) !== index
    );
    if (duplicate) {
      throw new Error(`Foliage plugins declare conflicting uniform '${duplicate}'`);
    }
    return {
      materialUniforms,
      materialBind: material => this.bindFoliageUniforms(material),
    };
  }

  /**
   * Attaches foliage prototype meshes. The plugin list is fixed from this point
   * because it is compiled into the shader; a different behavior set is a
   * different container.
   */
  public async attachFoliage(
    scene: Scene,
    meshes: Mesh[],
    options: ShadoFoliageContainerOptions
  ): Promise<ShadoMaterial<any>> {
    const { materialUniforms, materialBind } = this.configureFoliage(options, meshes);
    // Plugins that reason about facing — edge-on thickening, view-dependent
    // normals — need the eye, which is not the focus. The focus is the player.
    this._scene = scene;
    return this.attachMeshes(scene, meshes, null, {
      ...options,
      // Foliage is rigid by definition. Requesting VAT here would demand a
      // skeleton and silently switch the generated shader to the skinned path,
      // where this container's hook locals do not exist.
      vat: 'none',
      vatQuality: 'rigid',
      // The atlas already carries every source texture's pixels; binding the
      // originals as well added three texture slots and pushed the pipeline
      // past WebGPU's per-stage uniform-buffer limit — the whole draw then
      // silently failed validation and the foliage vanished.
      sourceTextures: false,
      picking: options.picking ?? false,
      materialUniforms,
      materialBind,
    });
  }

  private bindFoliageUniforms(material: ShadoMaterial<any>): void {
    const now = Date.now();
    if (this._timeSource) {
      this._elapsedSeconds = this._timeSource();
    } else {
      const delta = this._lastFrameMs ? (now - this._lastFrameMs) * 0.001 : 0;
      // A tab restored after minutes in the background would otherwise advance
      // the wind by that entire gap in one frame and snap every plant.
      this._elapsedSeconds += Math.min(delta, 0.25);
    }
    this._lastFrameMs = now;

    const focus = this._focus();
    this._focusVector.set(focus[0], focus[1], focus[2]);
    material.setFloat('uShadoFoliageTime', this._elapsedSeconds);
    material.setFloat('uShadoFoliageInverseHeight', this._inverseHeight);
    material.setVector3('uShadoFoliageFocus', this._focusVector);
    const camera = this._scene?.activeCamera;
    if (camera) this._cameraVector.copyFrom(camera.globalPosition);
    material.setVector3('uShadoFoliageCamera', this._cameraVector);

    this._frame.timeSeconds = this._elapsedSeconds;
    this._frame.focus[0] = focus[0];
    this._frame.focus[1] = focus[1];
    this._frame.focus[2] = focus[2];
    for (const plugin of this._plugins) {
      plugin.bind(material, this._frame as ShadoFoliageFrame);
    }
  }

  protected override getGLSLHooks(): ShadoInstanceGLSLHooks {
    const collect = (key: keyof ShadoFoliageShaderFragments) =>
      this._plugins.map(plugin => plugin.glsl[key] ?? '').join('');
    return {
      vertexDeclarations: `
uniform float uShadoFoliageTime;
uniform float uShadoFoliageInverseHeight;
uniform vec3 uShadoFoliageFocus;
uniform vec3 uShadoFoliageCamera;
varying float vShadoFoliageFade;
${collect('vertexDeclarations')}`,
      fragmentDeclarations: `
varying float vShadoFoliageFade;
${collect('fragmentDeclarations')}`,
      vertexInstance: collect('vertexInstance'),
      vertexAfterPosition: `
  vec3 shadoFoliageRoot = T.xyz;
  // The point distance-based plugins measure from. Defaults to the instance
  // origin — right for a tree, whose instance IS the plant — but a plugin that
  // derives many plants from one instance must overwrite it per plant, or wind
  // gusts, fades, and bends all step at instance (cell) granularity.
  vec3 shadoFoliageAnchor = T.xyz;
  vec3 shadoFoliageWorld = p;
  float shadoFoliageScale = T.w;
  float shadoFoliageUp = clamp(position.y * uShadoFoliageInverseHeight, 0.0, 1.0);
  float shadoFoliageFade = 1.0;
  // Per-instance character, as mutable locals rather than direct reads of the
  // actor header. A plugin that derives many plants from one instance, such as
  // grass blades from a cell, overwrites these per blade, and every later
  // plugin then varies per blade instead of per cell.
  float shadoFoliagePhase = inst.foliageParams.x;
  float shadoFoliageStiffness = inst.foliageParams.y;
  float shadoFoliageVariation = inst.foliageParams.z;
${collect('displace')}
  gl_Position = worldViewProjection * vec4(shadoFoliageWorld, 1.0);
  // Reject the whole instance outside clip space rather than paying for its
  // fragments. Vertex work is still spent, which is why cell-level residency
  // remains the caller's job.
  if (shadoFoliageFade <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  vShadoFoliageFade = shadoFoliageFade;`,
      // Leaf textures are alpha cutouts, and this material draws opaque: a
      // low-alpha texel must DISCARD, and everything kept must write alpha 1.
      // Writing the sampled alpha through an opaque pipeline onto a
      // premultiplied canvas literally punches transparent holes in the page.
      fragmentSurface: `
  if (surface.a < 0.4) discard;
  surface.a = 1.0;
${collect('fragmentSurface')}`,
    };
  }

  protected override getWGSLHooks(): ShadoInstanceWGSLHooks {
    const collect = (key: keyof ShadoFoliageShaderFragments) =>
      this._plugins.map(plugin => plugin.wgsl[key] ?? '').join('');
    return {
      // WGSL has no loose uniforms: Babylon builds its leftover uniform struct
      // from these declarations, not from the names passed to the material. An
      // undeclared uniform fails to compile at first use, on the device only.
      vertexDeclarations: `
uniform uShadoFoliageTime: f32;
uniform uShadoFoliageInverseHeight: f32;
uniform uShadoFoliageFocus: vec3f;
uniform uShadoFoliageCamera: vec3f;
varying vShadoFoliageFade: f32;
${collect('vertexDeclarations')}`,
      fragmentDeclarations: `
varying vShadoFoliageFade: f32;
${collect('fragmentDeclarations')}`,
      vertexInstance: collect('vertexInstance'),
      vertexAfterPosition: `
  let shadoFoliageRoot = translation.xyz;
  var shadoFoliageAnchor = translation.xyz;
  var shadoFoliageWorld = worldPosition;
  let shadoFoliageScale = translation.w;
  let shadoFoliageUp = clamp(
    vertexInputs.position.y * uniforms.uShadoFoliageInverseHeight,
    0.0,
    1.0
  );
  var shadoFoliageFade = 1.0;
  var shadoFoliagePhase = inst.foliageParams.x;
  var shadoFoliageStiffness = inst.foliageParams.y;
  var shadoFoliageVariation = inst.foliageParams.z;
${collect('displace')}
  vertexOutputs.position = uniforms.worldViewProjection * vec4f(shadoFoliageWorld, 1.0);
  if (shadoFoliageFade <= 0.0) {
    vertexOutputs.position = vec4f(2.0, 2.0, 2.0, 1.0);
  }
  vertexOutputs.vShadoFoliageFade = shadoFoliageFade;`,
      fragmentSurface: `
  if (surface.a < 0.4) { discard; }
  surface = vec4f(surface.rgb, 1.0);
${collect('fragmentSurface')}`,
    };
  }
}

const ORIGIN: readonly [number, number, number] = [0, 0, 0];

/**
 * Derives stable per-instance foliage parameters from a world position.
 *
 * Placement must survive streaming: a plant that unloads and reloads has to
 * come back with the same phase and colour, so this is a hash of where it is,
 * never a random draw at spawn time.
 */
export function seedFoliageParams(
  x: number,
  z: number,
  out: Float32Array = new Float32Array(4)
): Float32Array {
  let hash = Math.imul(Math.round(x * 64) | 0, 0x27d4_eb2d);
  hash = (hash ^ Math.imul(Math.round(z * 64) | 0, 0x9e37_79b1)) >>> 0;
  const next = () => {
    hash = (Math.imul(hash, 1_664_525) + 1_013_904_223) >>> 0;
    return hash / 0x1_0000_0000;
  };
  out[0] = next();
  out[1] = next();
  out[2] = next();
  out[3] = next();
  return out;
}

function resolveMixed(
  plugins: readonly (ShadoFoliagePluginSpec | ShadoFoliagePlugin)[]
): ShadoFoliagePlugin[] {
  const specs = plugins.filter((plugin): plugin is ShadoFoliagePluginSpec =>
    !isResolvedPlugin(plugin)
  );
  const resolved = resolveShadoFoliagePlugins(specs);
  let cursor = 0;
  return plugins.map(plugin =>
    isResolvedPlugin(plugin) ? plugin : resolved[cursor++]!
  );
}

function isResolvedPlugin(
  plugin: ShadoFoliagePluginSpec | ShadoFoliagePlugin
): plugin is ShadoFoliagePlugin {
  return typeof (plugin as ShadoFoliagePlugin).bind === 'function';
}

function resolveSourceHeight(meshes: readonly Mesh[], declared?: number): number {
  if (declared !== undefined) {
    if (!Number.isFinite(declared) || declared <= 0) {
      throw new Error('attachFoliage: sourceHeight must be positive');
    }
    return declared;
  }
  let maximum = 0;
  for (const mesh of meshes) {
    const bounds = mesh.getBoundingInfo?.()?.boundingBox;
    if (bounds) maximum = Math.max(maximum, bounds.maximum.y);
  }
  if (!(maximum > 0)) {
    throw new Error(
      'attachFoliage: could not derive sourceHeight from the meshes; pass it explicitly'
    );
  }
  return maximum;
}
